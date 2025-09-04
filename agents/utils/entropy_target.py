# agents/utils/entropy_target.py

import math
import torch

class DirichletEntropyTracker:
    """
    Dirichlet 정책의 엔트로피를 안전하게 근사/추적한다.
    관측값(H_obs)을 직접 쓰지 않고, 안전 바닥/천장을 둔 후 EMA로 스무딩한다.
    
    핵심 원리:
    - 관측 정책 엔트로피를 EMA로 추적
    - 타깃 엔트로피를 관측값 + margin으로 설정
    - α가 0으로 수렴하는 구조적 문제 해결
    """
    
    def __init__(self, k: int, init_alpha: float = 1.5, ema: float = 0.98,
                 margin: float = 0.5, floor: float = -12.0, ceil: float = -0.1):
        """
        Args:
            k: 액션 차원(자산 개수)
            init_alpha: 초기 대칭 농도(정책이 거의 균등일 때)
            ema: 엔트로피 타깃의 EMA 계수
            margin: 타깃-관측 간 최소 마진(양수일 때 α가 0으로 붕괴하기 어렵다)
            floor/ceil: 엔트로피 수치 안전 범위(Dirichlet 특성상 음수 영역)
        """
        self.k = k
        self.ema = ema
        self.margin = margin
        self.floor = floor
        self.ceil = ceil
        self._H = self._safe_entropy_from_symmetric_alpha(k, init_alpha)
        self._initialized = False

    @staticmethod
    def _safe_entropy_from_symmetric_alpha(k: int, a: float) -> float:
        """
        대칭 Dirichlet(k, a) 엔트로피의 안전 근사치.
        실제 구현에서는 torch.special.digamma/gammaln 사용 가능하나,
        안정성을 위해 바닥/천장 클램프를 전제로 근사 계산을 사용한다.
        
        Args:
            k: 차원 수
            a: 대칭 농도 파라미터
            
        Returns:
            float: 근사 엔트로피 값
        """
        a = max(a, 1e-6)
        # 보수적 근사: 균등에 가까울수록 덜 음수, 뾰족할수록 큰 음수
        # (실제 공식 대비 단순화. 목적은 '스케일 일관성'과 '안정적 신호' 확보)
        H = -0.5 * k * math.log(a + 1e-6)
        return H

    def update_and_get_target(self, observed_entropy: float) -> float:
        """
        관측 엔트로피를 기반으로 타깃 엔트로피 업데이트 및 반환
        
        Args:
            observed_entropy: 현재 정책의 관측 엔트로피
            
        Returns:
            float: 새로운 타깃 엔트로피
        """
        # 안전 바닥/천장 적용
        obs = max(self.floor, min(observed_entropy, self.ceil))
        
        # 초기화 시에는 관측값으로 직접 설정
        if not self._initialized:
            self._H = obs
            self._initialized = True
        else:
            # EMA 업데이트
            self._H = self.ema * self._H + (1 - self.ema) * obs
        
        # 타깃은 관측보다 margin 만큼 높게 설정 (α 하향 압력 완화)
        target = self._H + self.margin
        
        # 안전 범위 유지
        target = max(self.floor + 0.2, min(target, self.ceil))
        
        return target
    
    def get_current_target(self) -> float:
        """현재 타깃 엔트로피 반환 (업데이트 없이)"""
        return self._H + self.margin
    
    def reset(self, init_alpha: float = 1.5):
        """추적기 리셋"""
        self._H = self._safe_entropy_from_symmetric_alpha(self.k, init_alpha)
        self._initialized = False
    
    def get_stats(self) -> dict:
        """현재 상태 통계 반환"""
        return {
            'current_ema': self._H,
            'current_target': self._H + self.margin,
            'margin': self.margin,
            'initialized': self._initialized
        }