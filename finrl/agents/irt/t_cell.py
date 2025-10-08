# finrl/agents/irt/t_cell.py

"""
T-Cell: 경량 위기 감지 시스템

이전 버전과의 차이:
- Isolation Forest 제거 (복잡도 감소)
- 단일 신경망으로 z, d, c 동시 출력
- 온라인 정규화로 안정성 확보

출력:
- z: 위기 타입 점수 [B, K] (다차원)
- d: 공자극 임베딩 [B, D] (IRT 비용 함수용)
- c: 스칼라 위기 레벨 [B, 1] (복제자 가열용)

의존성: torch
사용처: BCellIRTActor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

class TCellMinimal(nn.Module):
    """경량 T-Cell: 위기 감지 + 공자극 임베딩"""

    def __init__(self,
                 in_dim: int,
                 emb_dim: int = 128,
                 n_types: int = 4,
                 momentum: float = 0.99):
        """
        Args:
            in_dim: 입력 차원 (시장 특성, 예: 12)
            emb_dim: 공자극 임베딩 차원
            n_types: 위기 타입 수 (변동성, 유동성, 상관관계, 시스템)
            momentum: 온라인 정규화 모멘텀
        """
        super().__init__()

        self.n_types = n_types
        self.emb_dim = emb_dim
        self.momentum = momentum

        # ===== Phase 1.9 Tier 3: 위기 로깅 =====
        self._debug_counter = 0
        self._debug_interval = 5000  # 5000 step마다 출력 (선택사항: 빈도 감소)

        # 단일 인코더 (효율성)
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, n_types + emb_dim)
        )

        # 온라인 정규화 통계 (학습 중 업데이트)
        self.register_buffer('mu', torch.zeros(n_types))
        self.register_buffer('sigma', torch.ones(n_types))
        self.register_buffer('count', torch.zeros(1))

        # 위기 타입별 가중치 (학습 가능)
        self.alpha = nn.Parameter(torch.ones(n_types) / n_types)

    def forward(self,
                features: torch.Tensor,
                update_stats: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            features: 시장 특성 [B, F]
            update_stats: 통계 업데이트 여부 (학습=True, 평가=False)

        Returns:
            z: 위기 타입 점수 [B, K]
            d: 공자극 임베딩 [B, D]
            c: 스칼라 위기 레벨 [B, 1]
        """
        h = self.encoder(features)  # [B, K+D]

        # 분리
        z = h[:, :self.n_types]      # [B, K]
        d = h[:, self.n_types:]      # [B, D]

        # 온라인 정규화 (학습 시, batch > 1)
        if update_stats and self.training and z.size(0) > 1:
            with torch.no_grad():
                batch_mu = z.mean(dim=0)
                batch_sigma = z.std(dim=0) + 1e-6

                # EMA 업데이트
                self.mu = self.momentum * self.mu + (1 - self.momentum) * batch_mu
                self.sigma = self.momentum * self.sigma + (1 - self.momentum) * batch_sigma
                self.count += 1

        # 표준화
        z_std = (z - self.mu) / (self.sigma + 1e-6)  # [B, K]

        # 가중 합산 → 시그모이드 (0-1 범위)
        alpha_norm = F.softmax(self.alpha, dim=0)  # [K]
        c = torch.sigmoid(
            (z_std * alpha_norm).sum(dim=-1, keepdim=True)
        )  # [B, 1]

        # ===== Phase 1.9 Tier 3: Crisis Logging =====
        if update_stats and self.training:
            self._debug_counter += 1
            if self._debug_counter % self._debug_interval == 0:
                print(f"\n[T-Cell Debug - Step {self._debug_counter}]")
                print(f"  Crisis level: mean={c.mean().item():.4f}, "
                      f"min={c.min().item():.4f}, max={c.max().item():.4f}")
                print(f"  Crisis type scores (raw z): {z[0, :].tolist()}")
                print(f"  Crisis type scores (std z): {z_std[0, :].tolist()}")
                print(f"  Alpha weights: {alpha_norm.tolist()}")

        return z, d, c

    def get_crisis_interpretation(self, z: torch.Tensor) -> Dict[str, float]:
        """
        위기 타입 해석 (시각화용)

        Args:
            z: 위기 타입 점수 [B, K]

        Returns:
            해석 딕셔너리
        """
        crisis_types = ['Volatility', 'Liquidity', 'Correlation', 'Systemic']

        z_std = (z - self.mu) / (self.sigma + 1e-6)
        z_prob = torch.sigmoid(z_std)  # [B, K]

        interpretation = {}
        for i, ctype in enumerate(crisis_types[:self.n_types]):
            interpretation[ctype] = z_prob[:, i].mean().item()

        return interpretation
