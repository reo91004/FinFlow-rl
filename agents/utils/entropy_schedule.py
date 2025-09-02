# agents/utils/entropy_schedule.py

import numpy as np
import torch
from typing import Dict, Tuple, Optional
from collections import deque
from utils.logger import BIPDLogger
from agents.utils.dirichlet_entropy import target_entropy_from_symmetric_alpha


class RegimeAdaptiveEntropyScheduler:
    """
    T-Cell 신호 기반 레짐 적응형 엔트로피 스케줄러
    
    평시/위기/회복 레짐에 따라 동적으로 Dirichlet α* 파라미터를 조정하여
    포트폴리오 다양성을 시장 상황에 맞게 최적화
    """
    
    def __init__(self, 
                 action_dim: int,
                 alpha_peace: float = 2.0,      # 평시: 적당한 다양성
                 alpha_crisis: float = 0.8,     # 위기: 높은 다양성 (리스크 분산)
                 alpha_recovery: float = 1.2,   # 회복: 균형잡힌 다양성
                 stability_window: int = 20,    # 시장 안정도 추정 윈도우
                 crisis_threshold: float = 0.6, # 위기 판단 임계값
                 transition_smoothing: float = 0.9, # 전환 완화 계수
                 logger_name: str = "EntropyScheduler"):
        
        self.action_dim = action_dim
        self.logger = BIPDLogger(logger_name)
        
        # 레짐별 α* 설정
        self.alpha_peace = alpha_peace
        self.alpha_crisis = alpha_crisis  
        self.alpha_recovery = alpha_recovery
        
        # 스케줄링 파라미터
        self.stability_window = stability_window
        self.crisis_threshold = crisis_threshold
        self.transition_smoothing = transition_smoothing
        
        # 상태 추적
        self.crisis_history = deque(maxlen=stability_window)
        self.current_alpha_star = alpha_peace  # 초기값
        self.target_alpha_star = alpha_peace
        self.target_entropy = target_entropy_from_symmetric_alpha(action_dim, alpha_peace)
        
        # 통계 추적
        self.regime_stats = {
            'peace_count': 0,
            'crisis_count': 0, 
            'recovery_count': 0,
            'transitions': 0,
            'entropy_gap_history': deque(maxlen=100)
        }
        
        self.logger.info(f"레짐 적응형 엔트로피 스케줄러 초기화")
        self.logger.info(f"  α* 설정: 평시={alpha_peace}, 위기={alpha_crisis}, 회복={alpha_recovery}")
        self.logger.info(f"  초기 목표 엔트로피: {self.target_entropy:.3f}")
    
    def update_regime_and_get_target_entropy(self, 
                                           crisis_level: float,
                                           market_stability: Optional[float] = None) -> Tuple[float, Dict]:
        """
        T-Cell 신호 기반 레짐 업데이트 및 목표 엔트로피 반환
        
        Args:
            crisis_level: T-Cell에서 감지한 전체 위기 레벨 (0-1)
            market_stability: 시장 안정도 추정값 (선택적)
            
        Returns:
            tuple: (목표 엔트로피, 레짐 정보)
        """
        # 위기 이력 업데이트
        self.crisis_history.append(crisis_level)
        
        # 시장 안정도 추정 (제공되지 않은 경우)
        if market_stability is None:
            if len(self.crisis_history) >= 5:
                recent_crises = list(self.crisis_history)[-5:]
                market_stability = 1.0 - np.mean(recent_crises)
            else:
                market_stability = 1.0 - crisis_level
        
        # 레짐 분류
        regime_info = self._classify_regime(crisis_level, market_stability)
        regime_type = regime_info['regime']
        
        # 목표 α* 계산
        self.target_alpha_star = self._calculate_target_alpha(regime_info)
        
        # 부드러운 전환 적용
        if abs(self.current_alpha_star - self.target_alpha_star) > 1e-3:
            self.current_alpha_star = (
                self.transition_smoothing * self.current_alpha_star + 
                (1 - self.transition_smoothing) * self.target_alpha_star
            )
            self.regime_stats['transitions'] += 1
        
        # 목표 엔트로피 계산
        self.target_entropy = target_entropy_from_symmetric_alpha(
            self.action_dim, self.current_alpha_star
        )
        
        # 통계 업데이트
        self.regime_stats[f'{regime_type}_count'] += 1
        
        # 상세 정보 구성
        detailed_info = {
            **regime_info,
            'current_alpha_star': self.current_alpha_star,
            'target_alpha_star': self.target_alpha_star,
            'target_entropy': self.target_entropy,
            'market_stability': market_stability,
            'transition_rate': abs(self.current_alpha_star - self.target_alpha_star)
        }
        
        return self.target_entropy, detailed_info
    
    def _classify_regime(self, crisis_level: float, market_stability: float) -> Dict:
        """레짐 분류 로직"""
        
        # 위기 확률 계산
        crisis_prob = crisis_level
        
        # 회복 확률 계산 (안정도 기반)
        recovery_prob = max(0.0, 1.0 - market_stability) * (1.0 - crisis_prob)
        
        # 평시 확률 계산
        peace_prob = max(0.0, 1.0 - crisis_prob - recovery_prob)
        
        # 주요 레짐 결정
        if crisis_prob >= self.crisis_threshold:
            regime = 'crisis'
            confidence = crisis_prob
        elif recovery_prob > peace_prob and recovery_prob > 0.3:
            regime = 'recovery'
            confidence = recovery_prob
        else:
            regime = 'peace'
            confidence = peace_prob
        
        return {
            'regime': regime,
            'confidence': confidence,
            'crisis_prob': crisis_prob,
            'recovery_prob': recovery_prob,
            'peace_prob': peace_prob,
            'probabilities': {
                'crisis': crisis_prob,
                'recovery': recovery_prob,
                'peace': peace_prob
            }
        }
    
    def _calculate_target_alpha(self, regime_info: Dict) -> float:
        """가중 평균으로 목표 α* 계산"""
        probs = regime_info['probabilities']
        
        weighted_alpha = (
            probs['peace'] * self.alpha_peace +
            probs['crisis'] * self.alpha_crisis +
            probs['recovery'] * self.alpha_recovery
        )
        
        # 안전한 범위로 클리핑
        return np.clip(weighted_alpha, 0.3, 5.0)
    
    def track_entropy_gap(self, actual_entropy: float):
        """실제 엔트로피와 목표 엔트로피 간 갭 추적"""
        if actual_entropy is not None and not np.isnan(actual_entropy):
            gap = abs(actual_entropy - self.target_entropy)
            self.regime_stats['entropy_gap_history'].append(gap)
    
    def get_regime_statistics(self) -> Dict:
        """레짐 통계 반환"""
        total_updates = sum([
            self.regime_stats['peace_count'],
            self.regime_stats['crisis_count'],
            self.regime_stats['recovery_count']
        ])
        
        stats = {
            'total_updates': total_updates,
            'regime_distribution': {},
            'transitions': self.regime_stats['transitions'],
            'current_alpha_star': self.current_alpha_star,
            'target_entropy': self.target_entropy
        }
        
        if total_updates > 0:
            stats['regime_distribution'] = {
                'peace': self.regime_stats['peace_count'] / total_updates,
                'crisis': self.regime_stats['crisis_count'] / total_updates,
                'recovery': self.regime_stats['recovery_count'] / total_updates
            }
        
        # 엔트로피 갭 통계
        if self.regime_stats['entropy_gap_history']:
            gaps = list(self.regime_stats['entropy_gap_history'])
            stats['entropy_gap_stats'] = {
                'mean': np.mean(gaps),
                'std': np.std(gaps),
                'max': np.max(gaps)
            }
        
        return stats
    
    def reset_statistics(self):
        """통계 초기화"""
        self.regime_stats = {
            'peace_count': 0,
            'crisis_count': 0,
            'recovery_count': 0, 
            'transitions': 0,
            'entropy_gap_history': deque(maxlen=100)
        }
        self.logger.info("레짐 통계가 초기화되었습니다")


def create_entropy_scheduler_for_symbols(symbols_count: int) -> RegimeAdaptiveEntropyScheduler:
    """심볼 개수에 맞는 엔트로피 스케줄러 생성"""
    
    # 심볼 수에 따른 α* 조정
    if symbols_count <= 5:
        # 소수 자산: 상대적으로 높은 다양성 필요
        alpha_peace = 1.5
        alpha_crisis = 0.6 
        alpha_recovery = 1.0
    elif symbols_count <= 15:
        # 중간 자산: 균형잡힌 설정
        alpha_peace = 2.0
        alpha_crisis = 0.8
        alpha_recovery = 1.2
    else:
        # 다수 자산 (30개 등): 적당한 집중도 허용
        alpha_peace = 2.5
        alpha_crisis = 1.0
        alpha_recovery = 1.5
    
    return RegimeAdaptiveEntropyScheduler(
        action_dim=symbols_count,
        alpha_peace=alpha_peace,
        alpha_crisis=alpha_crisis,
        alpha_recovery=alpha_recovery
    )


# 테스트 함수
def test_entropy_scheduler():
    """엔트로피 스케줄러 테스트"""
    logger = BIPDLogger("EntropySchedulerTest")
    
    # 30개 자산용 스케줄러 생성
    scheduler = create_entropy_scheduler_for_symbols(30)
    
    logger.info("=== 엔트로피 스케줄러 테스트 시작 ===")
    
    # 다양한 시나리오 테스트
    scenarios = [
        (0.1, 0.9, "평시 시장"),
        (0.8, 0.3, "위기 시장"),
        (0.4, 0.6, "회복 시장"),
        (0.9, 0.2, "극심한 위기"),
        (0.05, 0.95, "매우 안정한 시장")
    ]
    
    for crisis_level, stability, description in scenarios:
        entropy, info = scheduler.update_regime_and_get_target_entropy(
            crisis_level, stability
        )
        
        logger.info(f"\n시나리오: {description}")
        logger.info(f"  위기 레벨: {crisis_level:.2f}, 안정도: {stability:.2f}")
        logger.info(f"  감지된 레짐: {info['regime']} (신뢰도: {info['confidence']:.2f})")
        logger.info(f"  α*: {info['current_alpha_star']:.3f} -> {info['target_alpha_star']:.3f}")
        logger.info(f"  목표 엔트로피: {entropy:.3f}")
    
    # 통계 출력
    stats = scheduler.get_regime_statistics()
    logger.info(f"\n=== 통계 요약 ===")
    logger.info(f"총 업데이트: {stats['total_updates']}")
    logger.info(f"레짐 분포: {stats['regime_distribution']}")
    logger.info(f"전환 횟수: {stats['transitions']}")
    
    logger.info("엔트로피 스케줄러 테스트 완료")


if __name__ == "__main__":
    test_entropy_scheduler()