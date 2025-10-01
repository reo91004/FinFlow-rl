# src/algorithms/critics/redq.py

"""
REDQ (Randomized Ensemble Double Q-learning) Critic

핵심 아이디어:
- N개 Q-network 앙상블 (예: N=10)
- 매 업데이트마다 M개 서브셋 샘플 (예: M=2)
- Min Q 사용으로 overestimation bias 완화

근거: Chen et al. (2021) "Randomized Ensembled Double Q-learning"

의존성: torch
사용처: TrainerIRT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class QNetwork(nn.Module):
    """단일 Q-network"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256]):
        super().__init__()

        layers = []
        in_dim = state_dim + action_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [B, S]
            action: [B, A]

        Returns:
            Q: [B, 1]
        """
        x = torch.cat([state, action], dim=-1)
        return self.network(x)

class REDQCritic(nn.Module):
    """REDQ Critic 앙상블"""

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 n_critics: int = 10,
                 m_sample: int = 2,
                 hidden_dims: List[int] = [256, 256]):
        """
        Args:
            state_dim: 상태 차원
            action_dim: 행동 차원
            n_critics: 앙상블 크기
            m_sample: 서브셋 크기 (target 계산용)
            hidden_dims: 은닉층 차원
        """
        super().__init__()

        self.n_critics = n_critics
        self.m_sample = m_sample

        # N개 독립적인 Q-network
        self.critics = nn.ModuleList([
            QNetwork(state_dim, action_dim, hidden_dims)
            for _ in range(n_critics)
        ])

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> List[torch.Tensor]:
        """
        모든 critics 출력

        Returns:
            List of [B, 1] tensors (길이 N)
        """
        return [critic(state, action) for critic in self.critics]

    def get_target_q(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Target Q 계산: M개 서브셋의 최솟값

        Args:
            state: [B, S]
            action: [B, A]

        Returns:
            target_q: [B, 1]
        """
        with torch.no_grad():
            # M개 critics 랜덤 선택
            indices = torch.randperm(self.n_critics)[:self.m_sample]

            q_values = []
            for idx in indices:
                q = self.critics[idx](state, action)
                q_values.append(q)

            # Min Q (overestimation bias 완화)
            target_q = torch.min(torch.stack(q_values), dim=0)[0]

        return target_q

    def get_all_critics(self) -> List[nn.Module]:
        """모든 critics 반환 (fitness 계산용)"""
        return list(self.critics)