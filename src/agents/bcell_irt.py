# src/agents/bcell_irt.py

"""
B-Cell Actor with IRT (Immune Replicator Transport)

핵심 기능:
1. 에피토프 인코딩: 상태 → 다중 토큰
2. IRT 연산: OT + Replicator 혼합
3. Dirichlet 디코딩: 혼합 → 포트폴리오 가중치
4. EMA 메모리: w_prev 관리

의존성: IRT, TCellMinimal, QNetwork
사용처: TrainerIRT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List

from src.immune.irt import IRT
from src.immune.t_cell import TCellMinimal

class BCellIRTActor(nn.Module):
    """IRT 기반 B-Cell Actor"""

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 emb_dim: int = 128,
                 m_tokens: int = 6,
                 M_proto: int = 8,
                 alpha: float = 0.3,
                 ema_beta: float = 0.9,
                 market_feature_dim: int = 12):
        """
        Args:
            state_dim: 상태 차원 (예: 43)
            action_dim: 행동 차원 (예: 30)
            emb_dim: 임베딩 차원
            m_tokens: 에피토프 토큰 수
            M_proto: 프로토타입 수
            alpha: OT-Replicator 결합 비율
            ema_beta: EMA 메모리 계수
            market_feature_dim: 시장 특성 차원 (FeatureExtractor 출력)
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.emb_dim = emb_dim
        self.m = m_tokens
        self.M = M_proto
        self.ema_beta = ema_beta

        # ===== 에피토프 인코더 =====
        self.epitope_encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, m_tokens * emb_dim)
        )

        # ===== 프로토타입 키 (학습 가능) =====
        # Xavier 초기화
        self.proto_keys = nn.Parameter(
            torch.randn(M_proto, emb_dim) / (emb_dim ** 0.5)
        )

        # ===== 프로토타입별 Dirichlet 디코더 =====
        # 각 프로토타입은 독립적인 정책 (전문가)
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(emb_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, action_dim),
                nn.Softplus()  # 양수 concentration 보장
            )
            for _ in range(M_proto)
        ])

        # ===== IRT 연산자 =====
        self.irt = IRT(
            emb_dim=emb_dim,
            m_tokens=m_tokens,
            M_proto=M_proto,
            alpha=alpha
        )

        # ===== T-Cell 통합 =====
        self.t_cell = TCellMinimal(
            in_dim=market_feature_dim,
            emb_dim=emb_dim
        )

        # ===== 이전 가중치 (EMA) =====
        self.register_buffer('w_prev', torch.full((1, M_proto), 1.0/M_proto))

    def _compute_fitness(self,
                        state: torch.Tensor,
                        critics: List[nn.Module]) -> torch.Tensor:
        """
        각 프로토타입의 적합도 (fitness) 계산

        방법: 각 프로토타입 정책으로 행동 샘플 → Critics로 Q값 평가

        Args:
            state: [B, S]
            critics: QNetwork 리스트 (REDQ)

        Returns:
            fitness: [B, M]
        """
        B = state.size(0)
        fitness = torch.zeros(B, self.M, device=state.device)

        with torch.no_grad():
            K_batch = self.proto_keys.unsqueeze(0).expand(B, -1, -1)  # [B, M, D]

            for j in range(self.M):
                # 프로토타입 j의 concentration
                conc_j = self.decoders[j](K_batch[:, j, :])  # [B, A]

                # Dirichlet 분포에서 샘플 (exploration 증가: min 1.0→0.5, max 100→50)
                conc_j_clamped = torch.clamp(conc_j, min=0.5, max=50.0)
                dist_j = torch.distributions.Dirichlet(conc_j_clamped)
                action_j = dist_j.sample()  # [B, A]

                # Critics로 Q값 평가 (앙상블 평균)
                q_values = []
                for critic in critics:
                    q = critic(state, action_j)
                    q_values.append(q.squeeze(-1))  # [B]

                fitness[:, j] = torch.stack(q_values).mean(dim=0)

        return fitness

    def forward(self,
                state: torch.Tensor,
                critics: Optional[List[nn.Module]] = None,
                deterministic: bool = False) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            state: [B, S]
            critics: QNetwork 리스트 (fitness 계산용)
            deterministic: 결정적 행동 (평가 시)

        Returns:
            action: [B, A] - 포트폴리오 가중치
            info: 해석 정보 (w, P, crisis 등)
        """
        B = state.size(0)

        # ===== Step 1: T-Cell 위기 감지 =====
        market_features = state[:, :12]  # 시장 특성 추출
        z, danger_embed, crisis_level = self.t_cell(
            market_features,
            update_stats=self.training
        )

        # ===== Step 2: 에피토프 인코딩 =====
        E = self.epitope_encoder(state).view(B, self.m, self.emb_dim)  # [B, m, D]

        # ===== Step 3: 프로토타입 확장 =====
        K = self.proto_keys.unsqueeze(0).expand(B, -1, -1)  # [B, M, D]

        # ===== Step 4: Fitness 계산 =====
        if critics is not None and not deterministic:
            fitness = self._compute_fitness(state, critics)
        else:
            # 평가 모드 또는 critics 없음: 균등 fitness
            fitness = torch.ones(B, self.M, device=state.device)

        # ===== Step 5: IRT 연산 =====
        w_prev_batch = self.w_prev.expand(B, -1)  # [B, M]

        w, P = self.irt(
            E=E,
            K=K,
            danger=danger_embed,
            w_prev=w_prev_batch,
            fitness=fitness,
            crisis_level=crisis_level,
            proto_conf=None  # 필요 시 추가
        )

        # ===== Step 6: Dirichlet 혼합 정책 =====
        # 각 프로토타입의 concentration 계산
        concentrations = torch.stack([
            self.decoders[j](K[:, j, :]) for j in range(self.M)
        ], dim=1)  # [B, M, A]

        # IRT 가중치로 혼합
        mixed_conc = torch.einsum('bm,bma->ba', w, concentrations) + 1.0  # [B, A]

        if deterministic:
            # 결정적: Dirichlet 평균 (mode)
            action = (mixed_conc - 1) / (mixed_conc.sum(dim=-1, keepdim=True) - self.action_dim)
            action = torch.clamp(action, min=0.0)
            action = action / (action.sum(dim=-1, keepdim=True) + 1e-8)
        else:
            # 확률적: Dirichlet 샘플 (exploration 증가: min 1.0→0.5, max 100→50)
            mixed_conc_clamped = torch.clamp(mixed_conc, min=0.5, max=50.0)
            dist = torch.distributions.Dirichlet(mixed_conc_clamped)
            action = dist.sample()

        # ===== Step 7: EMA 업데이트 (w_prev) =====
        if self.training:
            with torch.no_grad():
                self.w_prev = (
                    self.ema_beta * self.w_prev
                    + (1 - self.ema_beta) * w.detach().mean(dim=0, keepdim=True)
                )

        # ===== Step 8: 해석 정보 수집 =====
        info = {
            'w': w.detach(),  # [B, M] - 프로토타입 가중치
            'P': P.detach(),  # [B, m, M] - 수송 계획
            'crisis_level': crisis_level.detach(),  # [B, 1]
            'crisis_types': z.detach(),  # [B, K]
            'fitness': fitness.detach()  # [B, M]
        }

        return action, info