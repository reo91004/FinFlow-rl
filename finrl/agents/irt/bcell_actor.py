# finrl/agents/irt/bcell_actor.py

"""
B-Cell Actor with IRT (Immune Replicator Transport)

핵심 기능:
1. 에피토프 인코딩: 상태 → 다중 토큰
2. IRT 연산: OT + Replicator 혼합
3. Dirichlet 디코딩: 혼합 → 포트폴리오 가중치
4. EMA 메모리: w_prev 관리

의존성: IRT, TCellMinimal
사용처: IRTPolicy
"""

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
                 alpha_min: float = 0.06,
                 alpha_max: Optional[float] = None,
                 ema_beta: float = 0.85,
                 market_feature_dim: int = 12,
                 dirichlet_min: float = 0.5,
                 dirichlet_max: float = 50.0,
                 # Phase 3.5 Step 2: 다중 신호 위기 감지
                 w_r: float = 0.6,
                 w_s: float = 0.25,
                 w_c: float = 0.15,
                 # Phase B: 바이어스 EMA 보정
                 eta_b: float = 2e-3,
                 **irt_kwargs):
        """
        Args:
            state_dim: 상태 차원 (예: 181 for Dow 30)
            action_dim: 행동 차원 (예: 30)
            emb_dim: 임베딩 차원
            m_tokens: 에피토프 토큰 수
            M_proto: 프로토타입 수
            alpha: OT-Replicator 결합 비율 (후진 호환, alpha_max로 사용됨)
            alpha_min: 위기 시 최소 α (default: 0.06, Replicator 가중)
            alpha_max: 평시 최대 α (default: None → alpha 사용, OT 가중)
            ema_beta: EMA 메모리 계수
            market_feature_dim: 시장 특성 차원 (T-Cell 입력, 기본 12)
            dirichlet_min: Dirichlet concentration minimum (핸드오버: 0.5)
            dirichlet_max: Dirichlet concentration maximum (핸드오버: 50.0)
            w_r: 시장 위기 신호 가중치 (T-Cell 출력)
            w_s: Sharpe 신호 가중치 (DSR bonus)
            w_c: CVaR 신호 가중치
            eta_b: 바이어스 학습률 (crisis_regime_pct 중립화용)
            **irt_kwargs: IRT 파라미터 (eps, eta_0, eta_1 등)
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.emb_dim = emb_dim
        self.m = m_tokens
        self.M = M_proto
        self.ema_beta = ema_beta
        self.dirichlet_min = dirichlet_min
        self.dirichlet_max = dirichlet_max
        self.market_feature_dim = market_feature_dim

        # Phase 3.5 Step 2: 위기 신호 가중치
        self.w_r = w_r
        self.w_s = w_s
        self.w_c = w_c

        # Phase B: 바이어스 EMA 보정
        self.eta_b = eta_b
        self.register_buffer('crisis_bias', torch.zeros(1))

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
            alpha=alpha,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            **irt_kwargs
        )

        # ===== T-Cell 통합 =====
        self.t_cell = TCellMinimal(
            in_dim=market_feature_dim,
            emb_dim=emb_dim
        )

        # ===== 이전 가중치 (EMA) =====
        self.register_buffer('w_prev', torch.full((1, M_proto), 1.0/M_proto))

    def forward(self,
                state: torch.Tensor,
                fitness: Optional[torch.Tensor] = None,
                deterministic: bool = False) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            state: [B, S]
            fitness: 프로토타입 적합도 [B, M] (optional, 없으면 균등)
            deterministic: 결정적 행동 (평가 시)

        Returns:
            action: [B, A] - 포트폴리오 가중치
            info: 해석 정보 (w, P, crisis 등)
        """
        B = state.size(0)

        # ===== Step 1: T-Cell 위기 감지 =====
        # FinRL state 구조: [balance(1), prices(30), shares(30), tech_indicators(240)]
        # Tech indicators: [macd*30, boll_ub*30, boll_lb*30, rsi_30*30,
        #                   cci_30*30, dx_30*30, close_30_sma*30, close_60_sma*30]

        # Market features 추출 (12차원):
        # 1. 시장 전체 특성 (4개): balance, price_mean, price_std, cash_ratio
        # 2. Technical indicators 대표값 (8개): 각 indicator의 첫 번째 주식 값

        balance = state[:, 0:1]  # [B, 1]
        prices = state[:, 1:31]  # [B, 30]
        shares = state[:, 31:61]  # [B, 30]

        # 시장 통계량
        price_mean = prices.mean(dim=1, keepdim=True)  # [B, 1]
        price_std = prices.std(dim=1, keepdim=True) + 1e-8  # [B, 1]
        total_value = balance + (prices * shares).sum(dim=1, keepdim=True)  # [B, 1]
        cash_ratio = balance / (total_value + 1e-8)  # [B, 1]

        # Technical indicators (각 지표의 첫 번째 주식 값)
        # macd, boll_ub, boll_lb, rsi_30, cci_30, dx_30, close_30_sma, close_60_sma
        tech_indices = [61, 91, 121, 151, 181, 211, 241, 271]
        tech_features = state[:, tech_indices]  # [B, 8]

        market_features = torch.cat([
            balance,           # [B, 1] - 현금 보유량
            price_mean,        # [B, 1] - 평균 주가
            price_std,         # [B, 1] - 주가 변동성
            cash_ratio,        # [B, 1] - 현금 비율
            tech_features      # [B, 8] - 기술적 지표들
        ], dim=1)  # [B, 12]

        z, danger_embed, crisis_base = self.t_cell(
            market_features,
            update_stats=self.training
        )

        # Phase 3.5 Step 2: DSR/CVaR 신호 추출 및 위기 레벨 결합
        # state 마지막 2개 차원: [dsr_bonus, cvar_value]
        # state_dim은 reward_type='dsr_cvar'일 때 +2 되어 있음
        # 따라서 항상 마지막 2개 차원을 시도하되, 존재하지 않으면 0으로 처리
        if state.size(1) >= self.state_dim - 2:
            # DSR/CVaR가 state에 포함된 경우
            delta_sharpe = state[:, -2:-1]  # [B, 1] - DSR bonus (원본 스케일 ~0.1)
            cvar = state[:, -1:]  # [B, 1] - CVaR value (원본 스케일 ~0.01)

            # 각 신호를 [0, 1] 범위로 스케일링
            # crisis_base는 이미 sigmoid 통과 → [0, 1]
            # DSR: 10배 증폭 후 sigmoid (0.1 → ~0.73)
            delta_sharpe_scaled = torch.sigmoid(delta_sharpe * 10.0)
            # CVaR: 절댓값 + 50배 증폭 후 sigmoid (0.01 → ~0.62)
            cvar_scaled = torch.sigmoid(torch.abs(cvar) * 50.0)

            # 3개 신호 가중 평균 (w_r + w_s + w_c = 1.0 → 출력도 [0, 1])
            crisis_level_raw = (
                self.w_r * crisis_base
                + self.w_s * delta_sharpe_scaled
                + self.w_c * cvar_scaled
            )  # [B, 1]

            # Phase B: 바이어스 보정 (crisis_regime_pct → 0.5)
            crisis_level = crisis_level_raw + self.crisis_bias.to(state.device)

            # Phase E: 버그 픽스 - B=1(DummyVecEnv)에서도 바이어스 업데이트
            # 학습 중 바이어스 EMA 업데이트 (배치 크기 무관)
            if self.training:
                with torch.no_grad():
                    p = (crisis_level > 0.5).float().mean()
                    self.crisis_bias = self.crisis_bias - self.eta_b * (p - 0.5)
        else:
            # 후진 호환: DSR/CVaR 없으면 T-Cell 출력만 사용
            delta_sharpe = torch.zeros(B, 1, device=state.device)
            cvar = torch.zeros(B, 1, device=state.device)
            crisis_level = crisis_base + self.crisis_bias.to(state.device)

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

        # Phase A: delta_sharpe 전달 보증 (이미 line 186-207에서 추출됨)
        delta_sharpe_tensor = delta_sharpe

        w, P, irt_debug = self.irt(
            E=E,
            K=K,
            danger=danger_embed,
            w_prev=w_prev_batch,
            fitness=fitness,
            crisis_level=crisis_level,
            delta_sharpe=delta_sharpe_tensor,
            proto_conf=None
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
            # mode = (α - 1) / (Σα - K) for α > 1
            # 안전한 계산: softmax
            action = F.softmax(mixed_conc, dim=-1)
        else:
            # 확률적: Dirichlet 샘플
            mixed_conc_clamped = torch.clamp(mixed_conc, min=self.dirichlet_min, max=self.dirichlet_max)
            dist = torch.distributions.Dirichlet(mixed_conc_clamped)
            action = dist.sample()

        # Simplex 보장 (수치 안정성)
        action = torch.clamp(action, min=0.0, max=1.0)
        action = action / (action.sum(dim=-1, keepdim=True) + 1e-8)

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
            'alpha_c': irt_debug['alpha_c'].detach(),  # [B, 1] - Dynamic OT-Replicator mixing ratio
            # Dirichlet 정보 (log_prob 계산용)
            'concentrations': concentrations.detach(),  # [B, M, A] - 프로토타입별 concentration
            'mixed_conc': mixed_conc.detach(),  # [B, A] - 혼합된 concentration
            'mixed_conc_clamped': mixed_conc_clamped if not deterministic else mixed_conc.detach(),  # [B, A]
            # Phase 3.5 Step 2: 다중 신호 위기 감지 정보
            'crisis_base': crisis_base.detach(),  # [B, 1] - T-Cell 기본 위기 신호
            'delta_sharpe': delta_sharpe.detach() if state.size(1) >= self.state_dim - 2 else torch.zeros(B, 1, device=state.device),
            'cvar': cvar.detach() if state.size(1) >= self.state_dim - 2 else torch.zeros(B, 1, device=state.device),
        }

        return action, info
