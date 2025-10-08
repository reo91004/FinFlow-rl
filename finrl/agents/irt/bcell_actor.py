# finrl/agents/irt/bcell_actor.py

"""
B-Cell Actor with IRT (Immune Replicator Transport)

핵심 기능:
1. 에피토프 인코딩: 상태 → 다중 토큰
2. IRT 연산: OT + Replicator 혼합
3. Gaussian + Softmax 디코딩: 혼합 → 포트폴리오 가중치
4. EMA 메모리: w_prev 관리

의존성: IRT, TCellMinimal
사용처: IRTPolicy
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List

from finrl.agents.irt.irt_operator import IRT
from finrl.agents.irt.t_cell import TCellMinimal

class BCellIRTActor(nn.Module):
    """IRT 기반 B-Cell Actor (간소화 버전)"""

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 emb_dim: int = 128,
                 m_tokens: int = 6,
                 M_proto: int = 8,
                 alpha: float = 0.3,
                 ema_beta: float = 0.9,
                 market_feature_dim: int = 12,
                 log_std_min: float = -20,
                 log_std_max: float = 2,
                 use_shared_decoder: bool = False,
                 **irt_kwargs):
        """
        Args:
            state_dim: 상태 차원 (예: Dow 30의 경우 181)
            action_dim: 행동 차원 (예: 30)
            emb_dim: 임베딩 차원
            m_tokens: 에피토프 토큰 수
            M_proto: 프로토타입 수
            alpha: OT-Replicator 결합 비율
            ema_beta: EMA 메모리 계수
            market_feature_dim: 시장 특성 차원 (T-Cell 입력, 기본 12)
            log_std_min: 로그 표준편차 최솟값 (Gaussian 정책)
            log_std_max: 로그 표준편차 최댓값 (Gaussian 정책)
            **irt_kwargs: IRT 파라미터 (eps, eta_0, eta_1 등)
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.emb_dim = emb_dim
        self.m = m_tokens
        self.M = M_proto
        self.ema_beta = ema_beta
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.market_feature_dim = market_feature_dim
        self.use_shared_decoder = use_shared_decoder

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

        # ===== 프로토타입별 Gaussian 디코더 =====
        # 각 프로토타입은 독립적인 정책 (전문가)
        if use_shared_decoder:
            # 프로토타입 조건부 공유 디코더
            # 입력: [proto_representation, proto_one_hot]
            self.shared_decoder = nn.Sequential(
                nn.Linear(emb_dim + M_proto, 128),  # +M_proto는 원-핫 인코딩용
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, action_dim * 2)  # mu와 log_std 모두 출력
            )
            # 호환성을 위한 플레이스홀더
            self.mu_decoders = None
            self.log_std_decoders = None
        else:
            # 분리된 디코더 (기본값)
            # μ 디코더 (평균)
            self.mu_decoders = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(emb_dim, 128),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(128, action_dim)
                )
                for _ in range(M_proto)
            ])

            # log_std 디코더 (표준편차)
            self.log_std_decoders = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(emb_dim, 128),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(128, action_dim)
                )
                for _ in range(M_proto)
            ])
            # 호환성을 위한 플레이스홀더
            self.shared_decoder = None

        # ===== IRT 연산자 =====
        self.irt = IRT(
            emb_dim=emb_dim,
            m_tokens=m_tokens,
            M_proto=M_proto,
            alpha=alpha,
            **irt_kwargs
        )

        # ===== T-Cell 통합 =====
        self.t_cell = TCellMinimal(
            in_dim=market_feature_dim,
            emb_dim=emb_dim
        )

        # ===== 이전 가중치 (EMA) =====
        self.register_buffer('w_prev', torch.full((1, M_proto), 1.0/M_proto))

        # ===== Phase 1.9 권장: 로깅 속성 명시적 초기화 =====
        self._market_logged = False

    def forward(self,
                state: torch.Tensor,
                fitness: Optional[torch.Tensor] = None,
                deterministic: bool = False) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            state: [B, S]
            fitness: 프로토타입 적합도 [B, M] (선택사항, 없으면 균등)
            deterministic: 결정적 행동 (평가 시)

        Returns:
            action: [B, A] - 포트폴리오 가중치
            info: 해석 정보 (w, P, crisis 등)
        """
        B = state.size(0)

        # ===== 단계 1: T-Cell 위기 감지 =====
        # FinRL 상태 구조: [balance(1), prices(30), shares(30), tech_indicators(240)]
        # 기술적 지표: [macd*30, boll_ub*30, boll_lb*30, rsi_30*30,
        #               cci_30*30, dx_30*30, close_30_sma*30, close_60_sma*30]

        # 시장 특성 추출 (12차원):
        # 1. 시장 전체 특성 (4개): balance, price_mean, price_std, cash_ratio
        # 2. 기술적 지표 대표값 (8개): 각 지표의 첫 번째 주식 값

        balance = state[:, 0:1]  # [B, 1]
        prices = state[:, 1:31]  # [B, 30]
        shares = state[:, 31:61]  # [B, 30]

        # 시장 통계량
        price_mean = prices.mean(dim=1, keepdim=True)  # [B, 1]
        price_std = prices.std(dim=1, keepdim=True) + 1e-8  # [B, 1]
        total_value = balance + (prices * shares).sum(dim=1, keepdim=True)  # [B, 1]
        cash_ratio = balance / (total_value + 1e-8)  # [B, 1]

        # Technical indicators (8개 필수)
        # macd, boll_ub, boll_lb, rsi_30, cci_30, dx_30, close_30_sma, close_60_sma
        # FinRL Dow30 표준: state_dim = 1 + 30*2 + 30*8 = 301
        expected_state_dim = 61 + 8 * 30  # 301
        assert state.size(1) >= expected_state_dim, \
            f"State dimension too small: {state.size(1)}, expected >= {expected_state_dim} " \
            f"(FinRL Dow30 standard: balance(1) + prices(30) + shares(30) + tech_indicators(8*30))"

        tech_indices = [61 + i * 30 for i in range(8)]
        tech_features = state[:, tech_indices]  # [B, 8]

        market_features = torch.cat([
            balance,           # [B, 1] - 현금 보유량
            price_mean,        # [B, 1] - 평균 주가
            price_std,         # [B, 1] - 주가 변동성
            cash_ratio,        # [B, 1] - 현금 비율
            tech_features      # [B, 8] - 기술적 지표들
        ], dim=1)  # [B, 12]

        # ===== Phase 1.9 Tier 3: Market Features Logging =====
        if self.training and not hasattr(self, '_market_logged'):
            self._market_logged = True
            features_np = market_features[0].detach().cpu().numpy()
            print(f"\n[Market Features Range - First Batch]")
            print(f"  Balance: {features_np[0]:.2f}")
            print(f"  Price mean: {features_np[1]:.2f}")
            print(f"  Price std: {features_np[2]:.2f}")
            print(f"  Cash ratio: {features_np[3]:.4f}")
            print(f"  Tech features (8): {features_np[4:].tolist()}")

        z, danger_embed, crisis_level = self.t_cell(
            market_features,
            update_stats=self.training
        )

        # ===== Step 2: 에피토프 인코딩 =====
        E = self.epitope_encoder(state).view(B, self.m, self.emb_dim)  # [B, m, D]

        # ===== Step 3: 프로토타입 확장 =====
        K = self.proto_keys.unsqueeze(0).expand(B, -1, -1)  # [B, M, D]

        # ===== Step 4: Fitness 설정 =====
        if fitness is None:
            # Critic 없음: 균등 fitness
            fitness = torch.ones(B, self.M, device=state.device)

        # ===== Step 5: IRT 연산 =====
        w_prev_batch = self.w_prev.expand(B, -1)  # [B, M]

        w, P, irt_debug = self.irt(
            E=E,
            K=K,
            danger=danger_embed,
            w_prev=w_prev_batch,
            fitness=fitness,
            crisis_level=crisis_level,
            proto_conf=None
        )

        # ===== NEW: Prototype Diversity Regularization =====
        if self.training:
            diversity_loss = self._compute_diversity_loss(w)
            self._diversity_loss = diversity_loss
        else:
            self._diversity_loss = 0.0

        # ===== Step 6: Projected Gaussian Policy =====
        # 각 프로토타입의 Gaussian 파라미터 계산
        if self.use_shared_decoder:
            # Shared decoder with prototype conditioning
            mus = []
            log_stds = []

            for j in range(self.M):
                # Create one-hot encoding for prototype j
                proto_onehot = torch.zeros(B, self.M, device=state.device)
                proto_onehot[:, j] = 1.0

                # Concatenate prototype representation with one-hot
                decoder_input = torch.cat([K[:, j, :], proto_onehot], dim=-1)

                # Get output from shared decoder
                output = self.shared_decoder(decoder_input)  # [B, action_dim * 2]

                # Split into mu and log_std
                mu_j = output[:, :self.action_dim]
                log_std_j = output[:, self.action_dim:]

                mus.append(mu_j)
                log_stds.append(log_std_j)

            mus = torch.stack(mus, dim=1)  # [B, M, A]
            log_stds = torch.stack(log_stds, dim=1)  # [B, M, A]
        else:
            # Separate decoders (default)
            mus = torch.stack([
                self.mu_decoders[j](K[:, j, :]) for j in range(self.M)
            ], dim=1)  # [B, M, A]

            log_stds = torch.stack([
                self.log_std_decoders[j](K[:, j, :]) for j in range(self.M)
            ], dim=1)  # [B, M, A]

        log_stds = torch.clamp(log_stds, self.log_std_min, self.log_std_max)
        stds = log_stds.exp()

        # ===== Fix: Gaussian Mixture Variance (Law of Total Variance) =====
        # 올바른 분산 계산: Var[X] = E[Var[X|Y]] + Var[E[X|Y]]
        # = Σw_i·σ_i² + Σw_i·μ_i² - (Σw_i·μ_i)²
        mixed_mu = torch.einsum('bm,bma->ba', w, mus)      # [B, A]

        # E[X²] = Σw_i·(σ_i² + μ_i²)
        ex2 = torch.einsum('bm,bma->ba', w, stds**2 + mus**2)  # [B, A]

        # Var[X] = E[X²] - E[X]²
        mixed_var = ex2 - mixed_mu**2
        mixed_std = torch.sqrt(torch.clamp(mixed_var, min=1e-6))  # [B, A]

        # Gaussian 샘플링
        if deterministic:
            z = mixed_mu
        else:
            eps = torch.randn_like(mixed_mu)
            z = mixed_mu + eps * mixed_std

        # Tier 2: Euclidean projection onto constrained simplex (min_weight = 2%)
        action = self._project_to_simplex(z, min_weight=0.02)  # [B, A]

        # ===== Phase 1.9 Tier 1: Max Concentration Constraint (30%) =====
        # 금융 규제: UCITS 20%, Mutual Fund 25%, Hedge Fund 15-30%
        # Sweet spot: 30% (Sharpe 0.93, MDD -22%)
        action = torch.clamp(action, min=0.02, max=0.30)
        action = action / action.sum(dim=-1, keepdim=True)  # Renormalize

        # Log probability 계산 (unconstrained Gaussian)
        # Projection gradient는 SAC policy gradient에서 암묵적으로 처리됨
        if not deterministic:
            log_prob_gaussian = -0.5 * (
                ((z - mixed_mu) / mixed_std) ** 2
                + 2 * torch.log(mixed_std)
                + np.log(2 * np.pi)
            )
            log_prob = log_prob_gaussian.sum(dim=-1, keepdim=True)  # [B, 1]
        else:
            log_prob = None

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
            'fitness': fitness.detach(),  # [B, M]
            # IRT 분해 정보 (시각화용)
            'w_rep': irt_debug['w_rep'].detach(),  # [B, M] - Replicator 출력
            'w_ot': irt_debug['w_ot'].detach(),    # [B, M] - OT 출력
            'cost_matrix': irt_debug['cost_matrix'].detach(),  # [B, m, M]
            'eta': irt_debug['eta'].detach(),       # [B, 1] - Crisis learning rate
            'adaptive_alpha': irt_debug.get('adaptive_alpha', torch.tensor(self.irt.alpha)).detach(),  # [B, 1] - Hybrid mixing ratio
            # Gaussian policy 정보
            'mu': mixed_mu.detach(),  # [B, A] - 혼합된 mean
            'std': mixed_std.detach(),  # [B, A] - 혼합된 std
            'z': z.detach(),  # [B, A] - Gaussian 샘플
            # Diversity regularization
            'diversity_loss': getattr(self, '_diversity_loss', 0.0)
        }

        # Log prob 정보 (deterministic=False일 때만)
        if log_prob is not None:
            info['log_prob'] = log_prob.detach()
            info['log_prob_gaussian'] = log_prob_gaussian.detach()

        # Store for XAI integration in IRTPolicy
        self.last_irt_info = {
            'w': w.detach(),
            'w_rep': irt_debug['w_rep'].detach(),
            'w_ot': irt_debug['w_ot'].detach(),
            'crisis_level': crisis_level.detach(),
            'adaptive_alpha': irt_debug.get('adaptive_alpha', torch.tensor(self.irt.alpha)).detach(),
            'fitness': fitness.detach() if fitness is not None else None,
            'diversity_loss': getattr(self, '_diversity_loss', 0.0)
        }

        return action, log_prob, info

    def _project_to_simplex(self, z: torch.Tensor, min_weight: float = 0.02) -> torch.Tensor:
        """
        Euclidean projection onto constrained probability simplex.

        Constraints:
        - sum(w) = 1
        - w >= min_weight (기본: 2% → 최대 집중도 50%)

        Algorithm:
        1. 변수 변환: w = w' + min_weight (w' >= 0)
        2. sum(w) = sum(w') + n*min_weight = 1
        3. sum(w') = 1 - n*min_weight
        4. Project z onto simplex with sum = 1 - n*min_weight
        5. w = w' + min_weight

        Reference:
        - Duchi et al. (2008) "Efficient Projections onto the l1-Ball"

        Args:
            z: unconstrained vector [B, A]
            min_weight: minimum weight per asset (default: 0.02 = 2%)

        Returns:
            action: projected onto constrained simplex [B, A]
        """
        n = z.shape[-1]

        # Compute target sum for excess variables
        # w = w' + min_weight, sum(w) = 1
        # => sum(w') = 1 - n*min_weight
        target_sum = 1.0 - n * min_weight

        # Check feasibility
        if target_sum < 0:
            # min_weight too large, fallback to uniform
            return torch.full_like(z, 1.0 / n)

        # Project z onto simplex with sum = target_sum
        z_sorted, _ = torch.sort(z, dim=-1, descending=True)
        cumsum = torch.cumsum(z_sorted, dim=-1)

        k = torch.arange(1, n + 1, device=z.device, dtype=z.dtype)
        condition = z_sorted + (target_sum - cumsum) / k > 0
        rho = condition.sum(dim=-1, keepdim=True) - 1  # [B, 1]

        # Threshold
        theta = (cumsum.gather(-1, rho) - target_sum) / (rho.float() + 1)

        # Project
        w_excess = torch.clamp(z - theta, min=0)  # [B, A], sum = target_sum

        # Add minimum weight
        w_final = w_excess + min_weight  # [B, A], sum = 1

        # Numerical stability: ensure exact sum=1
        w_final = w_final / w_final.sum(dim=-1, keepdim=True)

        return w_final

    def _compute_diversity_loss(self, w: torch.Tensor) -> torch.Tensor:
        """
        Prototype diversity regularization

        목표:
        1. Batch 평균에서 모든 프로토타입 균등 사용
        2. 개별 state에서는 상황에 맞는 프로토타입 선택

        Args:
            w: [B, M] Prototype weights

        Returns:
            loss: scalar
        """
        eps = 1e-8
        M = w.shape[1]
        target_entropy = np.log(M)  # Uniform distribution entropy

        # 1. Batch average entropy (균등 분포 장려)
        w_mean = w.mean(dim=0)  # [M]
        entropy_mean = -(w_mean * torch.log(w_mean + eps)).sum()
        entropy_loss = (target_entropy - entropy_mean) ** 2

        # 2. Individual variance (다양성 장려)
        # Across prototypes, then batch
        w_var = w.var(dim=1).mean()
        var_penalty = -0.01 * w_var  # Negative to maximize variance

        return entropy_loss + var_penalty
