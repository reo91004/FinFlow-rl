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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List, Any

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
                 ema_beta: float = 0.5,  # Phase 2.2a: 0.65 → 0.5 (faster adaptation to fitness signals)
                 market_feature_dim: int = 12,
                 dirichlet_min: float = 0.1,  # Phase 2: 0.8 → 0.1 (allow sparsity for exploration)
                 dirichlet_max: float = 5.0,  # Phase 2: keep 5.0 (Dirichlet stochastic path)
                 action_temp: float = 0.3,  # Phase 2: 0.7 → 0.3 (sharper for concentration)
                 # Phase 3.5 Step 2: 다중 신호 위기 감지
                 w_r: float = 0.55,
                 w_s: float = -0.25,
                 w_c: float = 0.20,
                 # Phase B: 바이어스 EMA 보정
                 eta_b: float = 0.02,
                 eta_b_min: float = 0.002,
                 eta_b_decay_steps: int = 30000,
                 eta_T: float = 0.01,
                 eta_b_warmup_steps: int = 10000,
                 eta_b_warmup_value: float = 0.05,
                 # T-Cell 가드
                 crisis_target: float = 0.5,  # 목표 crisis_regime_pct
                 crisis_guard_rate_init: float = 0.30,
                 crisis_guard_rate_final: float = 0.05,
                 crisis_guard_warmup_steps: int = 10000,
                 hysteresis_up: float = 0.55,
                 hysteresis_down: float = 0.45,
                 adaptive_hysteresis: bool = True,
                 hysteresis_quantile: float = 0.85,
                 hysteresis_min_gap: float = 0.1,
                 crisis_history_len: int = 512,
                 crisis_guard_rate: Optional[float] = None,
                 k_s: float = 6.0,
                 k_c: float = 6.0,
                 k_b: float = 4.0,
                 p_star: float = 0.35,
                 temperature_min: float = 0.7,
                 temperature_max: float = 1.1,
                 stat_momentum: float = 0.95,
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
            dirichlet_max: Dirichlet concentration maximum (핸드오버: 5.0)
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
        self.action_temp = action_temp  # Phase-2
        self.market_feature_dim = market_feature_dim

        # Phase 3.5 Step 2: 위기 신호 가중치
        self.w_r = w_r
        self.w_s = w_s
        self.w_c = w_c

        # Phase 1: 신호 스케일 파라미터
        self.k_s = k_s
        self.k_c = k_c
        self.k_b = k_b

        # Phase B: 바이어스/온도 보정
        self.eta_b_max = eta_b
        self.eta_b_min = eta_b_min
        self.eta_b_decay_steps = eta_b_decay_steps
        self.eta_T = eta_T
        self.p_star = p_star
        self.eta_b_warmup_steps = max(int(eta_b_warmup_steps), 0)
        self.eta_b_warmup_value = float(eta_b_warmup_value)
        self.temperature_min = temperature_min
        self.temperature_max = temperature_max
        self.stat_momentum = stat_momentum
        self.stats_eps = 1e-6
        self._orientation_checked = False

        self.register_buffer('crisis_bias', torch.zeros(1))
        self.register_buffer('crisis_temperature', torch.ones(1))
        self.register_buffer('crisis_prev_regime', torch.zeros(1))
        self.register_buffer('crisis_step', torch.zeros(1))
        self.register_buffer('sharpe_mean', torch.zeros(1))
        self.register_buffer('sharpe_var', torch.ones(1))
        self.register_buffer('cvar_mean', torch.zeros(1))
        self.register_buffer('cvar_var', torch.ones(1))
        self.register_buffer('crisis_base_mean', torch.zeros(1))
        self.register_buffer('crisis_base_var', torch.ones(1))
        self.register_buffer('bias_warm_started', torch.zeros(1))
        self._crisis_initialized = False
        self._verify_signal_orientation()

        # T-Cell 가드
        self.crisis_target = crisis_target
        self.crisis_guard_rate_init = crisis_guard_rate_init
        self.crisis_guard_rate_final = crisis_guard_rate_final
        self.crisis_guard_warmup_steps = crisis_guard_warmup_steps
        self.hysteresis_up = hysteresis_up
        self.hysteresis_down = hysteresis_down
        self.adaptive_hysteresis = adaptive_hysteresis
        self.hysteresis_quantile = float(max(min(hysteresis_quantile, 0.99), 0.5))
        self.hysteresis_min_gap = float(max(hysteresis_min_gap, 0.01))
        history_len = max(int(crisis_history_len), 64)
        self.register_buffer("crisis_history", torch.zeros(history_len))
        self.register_buffer("crisis_hist_ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("crisis_hist_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("hysteresis_up_state", torch.tensor(float(hysteresis_up)))
        self.register_buffer("hysteresis_down_state", torch.tensor(float(hysteresis_down)))
        if crisis_guard_rate is not None:
            self.crisis_guard_rate_init = float(crisis_guard_rate)
            self.crisis_guard_rate_final = float(crisis_guard_rate)
            self.crisis_guard_warmup_steps = 0

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
        # Phase 3.1: Tanh transformation to allow prototype suppression
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(emb_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, action_dim),
                nn.Tanh(),  # [-1, 1] allows both positive and negative logits
            )
            for _ in range(M_proto)
        ])
        
        # Phase 3.1: Initialize with diversity (keep from 2.2a)
        for decoder_idx, decoder in enumerate(self.decoders):
            # decoder[-2] is the final Linear(128, action_dim) layer before Tanh
            final_linear = decoder[-2]
            # Initialize with variance to break symmetry
            torch.nn.init.normal_(final_linear.bias, mean=0.0, std=0.5)
            # Offset each prototype differently
            final_linear.bias.data += (decoder_idx - self.M / 2) * 0.2

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

    def _load_from_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str]
    ) -> None:
        """
        Backward compatibility for checkpoints saved before Phase 1 buffers were added.
        Missing buffers are initialized to sensible defaults instead of raising.
        """
        compatibility_buffers = [
            'crisis_bias',
            'crisis_temperature',
            'crisis_prev_regime',
            'crisis_step',
            'sharpe_mean',
            'sharpe_var',
            'cvar_mean',
            'cvar_var',
            'crisis_base_mean',
            'crisis_base_var',
            'bias_warm_started',
        ]

        for name in compatibility_buffers:
            full_key = prefix + name
            if full_key not in state_dict:
                state_dict[full_key] = getattr(self, name).detach().clone()

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs
        )

    def _current_eta_b(self) -> float:
        step = float(self.crisis_step.item())
        if step < self.eta_b_warmup_steps:
            return self.eta_b_warmup_value
        adjusted_step = max(step - self.eta_b_warmup_steps, 0.0)
        if self.eta_b_decay_steps <= 0:
            return self.eta_b_max
        ratio = min(max(adjusted_step / float(self.eta_b_decay_steps), 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * ratio))
        return self.eta_b_min + (self.eta_b_max - self.eta_b_min) * cosine

    def _current_guard_rate(self) -> float:
        if not self.training:
            return self.crisis_guard_rate_final
        if self.crisis_guard_warmup_steps <= 0:
            return self.crisis_guard_rate_final
        step = float(self.crisis_step.item())
        progress = min(max(step / float(self.crisis_guard_warmup_steps), 0.0), 1.0)
        return self.crisis_guard_rate_init + (self.crisis_guard_rate_final - self.crisis_guard_rate_init) * progress

    def _update_hysteresis_thresholds(self, crisis_samples: torch.Tensor) -> None:
        if not self.adaptive_hysteresis or not self.training:
            return
        value = float(crisis_samples.detach().mean().clamp(0.0, 1.0).item())
        ptr = int(self.crisis_hist_ptr.item())
        self.crisis_history[ptr] = value
        ptr = (ptr + 1) % self.crisis_history.numel()
        self.crisis_hist_ptr.fill_(ptr)
        count = int(self.crisis_hist_count.item())
        count = min(count + 1, self.crisis_history.numel())
        self.crisis_hist_count.fill_(count)
        if count < 32:
            return
        history = self.crisis_history[:count].to(crisis_samples.device)
        quantile = torch.quantile(history, self.hysteresis_quantile).item()
        up = max(min(quantile, 1.0), 0.0)
        down = max(up - self.hysteresis_min_gap, 0.0)
        self.hysteresis_up_state.fill_(up)
        self.hysteresis_down_state.fill_(down)

    def _current_hysteresis_thresholds(self) -> Tuple[float, float]:
        if self.adaptive_hysteresis:
            up = float(self.hysteresis_up_state.item())
            down = float(self.hysteresis_down_state.item())
        else:
            up = self.hysteresis_up
            down = self.hysteresis_down
        return up, down

    def _verify_signal_orientation(self) -> None:
        if self._orientation_checked:
            return
        def sigmoid(x: float) -> float:
            return 1.0 / (1.0 + math.exp(-x))

        sharpe_good = sigmoid(-self.k_s * 1.0)
        sharpe_bad = sigmoid(-self.k_s * -1.0)
        cvar_good = sigmoid(self.k_c * 1.0)
        cvar_bad = sigmoid(self.k_c * -1.0)
        base_good = sigmoid(self.k_b * 1.0)
        base_bad = sigmoid(self.k_b * -1.0)

        if not (sharpe_good < sharpe_bad):
            raise ValueError("Sharpe signal transform orientation invalid (expected Sharpe↑ → crisis↓).")
        if not (cvar_good > cvar_bad):
            raise ValueError("CVaR signal transform orientation invalid (expected CVaR↑ → crisis↑).")
        if not (base_good > base_bad):
            raise ValueError("Base crisis transform orientation invalid (expected base↑ → crisis↑).")
        self._orientation_checked = True

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

        if not self._crisis_initialized:
            self._crisis_initialized = True

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

        z, danger_embed, crisis_affine_raw, crisis_base_sigmoid = self.t_cell(
            market_features,
            update_stats=self.training
        )

        def _update_running_stats(sample: torch.Tensor,
                                  mean_buf: torch.Tensor,
                                  var_buf: torch.Tensor) -> None:
            if not self.training:
                return
            if sample.ndim == 1:
                sample_view = sample.view(-1, 1)
            else:
                sample_view = sample
            with torch.no_grad():
                batch_mean = sample_view.detach().mean(dim=0)
                if sample_view.size(0) > 1:
                    batch_var = sample_view.detach().var(dim=0, unbiased=False)
                else:
                    batch_var = torch.zeros_like(batch_mean)
                mean_buf.mul_(self.stat_momentum).add_(batch_mean * (1 - self.stat_momentum))
                var_buf.mul_(self.stat_momentum).add_(batch_var * (1 - self.stat_momentum))
                var_buf.clamp_(min=self.stats_eps)

        # Base crisis component (T-Cell)
        _update_running_stats(crisis_affine_raw, self.crisis_base_mean, self.crisis_base_var)
        base_mean = self.crisis_base_mean.to(state.device)
        base_std = torch.sqrt(self.crisis_base_var.to(state.device) + self.stats_eps)
        base_z = (crisis_affine_raw - base_mean) / base_std
        crisis_base_component = torch.sigmoid(self.k_b * base_z)

        # Phase 3.5 Step 2: DSR/CVaR 신호 추출 및 위기 레벨 결합
        # state 마지막 2개 차원: [dsr_bonus, cvar_value]
        # state_dim은 reward_type='dsr_cvar'일 때 +2 되어 있음
        # 따라서 항상 마지막 2개 차원을 시도하되, 존재하지 않으면 0으로 처리
        delta_sharpe_raw = torch.zeros(B, 1, device=state.device)
        cvar_raw = torch.zeros(B, 1, device=state.device)
        delta_component = torch.full_like(crisis_base_component, 0.5)
        cvar_component = torch.full_like(crisis_base_component, 0.5)

        if state.size(1) >= self.state_dim - 2:
            # DSR/CVaR가 state에 포함된 경우
            delta_sharpe_raw = state[:, -2:-1]  # [B, 1] - DSR bonus (원본 스케일 ~0.1)
            cvar_raw = state[:, -1:]  # [B, 1] - CVaR value (원본 스케일 ~0.01)

            _update_running_stats(delta_sharpe_raw, self.sharpe_mean, self.sharpe_var)
            _update_running_stats(cvar_raw, self.cvar_mean, self.cvar_var)

            sharpe_mean = self.sharpe_mean.to(state.device)
            sharpe_std = torch.sqrt(self.sharpe_var.to(state.device) + self.stats_eps)
            cvar_mean = self.cvar_mean.to(state.device)
            cvar_std = torch.sqrt(self.cvar_var.to(state.device) + self.stats_eps)

            sharpe_z = (delta_sharpe_raw - sharpe_mean) / sharpe_std
            cvar_z = (cvar_raw - cvar_mean) / cvar_std

            delta_component = torch.sigmoid(-self.k_s * sharpe_z)
            cvar_component = torch.sigmoid(self.k_c * cvar_z)

        crisis_raw = (
            self.w_r * crisis_base_component
            + self.w_s * delta_component
            + self.w_c * cvar_component
        )

        if self.training and self.bias_warm_started.item() < 0.5:
            with torch.no_grad():
                mean_raw = crisis_raw.detach().mean()
                temp = self.crisis_temperature.to(state.device)
                temp_clamped = torch.clamp(temp, min=self.stats_eps)
                target = mean_raw.new_tensor(self.p_star)
                target = torch.clamp(target, self.stats_eps, 1 - self.stats_eps)
                target_logit = torch.log(target / (1 - target))
                new_bias = mean_raw - temp_clamped * target_logit
                self.crisis_bias.copy_(new_bias)
                self.bias_warm_started.fill_(1.0)

        bias = self.crisis_bias.to(state.device)
        temperature = self.crisis_temperature.to(state.device)
        crisis_affine = (crisis_raw - bias) / (temperature + self.stats_eps)
        crisis_level_pre_guard = torch.sigmoid(crisis_affine)

        p_hat = crisis_level_pre_guard.detach().mean()

        if self.training:
            with torch.no_grad():
                self.crisis_step.add_(1.0)
                eta_b_now = self._current_eta_b()
                self.crisis_bias.add_(eta_b_now * (p_hat - self.p_star))
                temperature_update = 1.0 + self.eta_T * (p_hat - self.p_star)
                new_temperature = torch.clamp(
                    self.crisis_temperature * temperature_update,
                    min=self.temperature_min,
                    max=self.temperature_max
                )
                self.crisis_temperature.copy_(new_temperature)

        guard_rate = self._current_guard_rate()
        guard_target = crisis_level_pre_guard.new_full(crisis_level_pre_guard.shape, self.crisis_target)
        crisis_level = torch.clamp(
            crisis_level_pre_guard + guard_rate * (guard_target - crisis_level_pre_guard),
            min=0.0,
            max=1.0
        )

        self._update_hysteresis_thresholds(crisis_level_pre_guard)
        hysteresis_up_val, hysteresis_down_val = self._current_hysteresis_thresholds()
        hysteresis_up_tensor = crisis_level_pre_guard.new_tensor(hysteresis_up_val)
        hysteresis_down_tensor = crisis_level_pre_guard.new_tensor(hysteresis_down_val)

        prev_regime_flag = (self.crisis_prev_regime > 0.5).float()
        if prev_regime_flag.item() >= 0.5:
            crisis_regime = torch.where(
                crisis_level < hysteresis_down_tensor,
                torch.zeros_like(crisis_level),
                torch.ones_like(crisis_level)
            )
        else:
            crisis_regime = torch.where(
                crisis_level > hysteresis_up_tensor,
                torch.ones_like(crisis_level),
                torch.zeros_like(crisis_level)
            )

        with torch.no_grad():
            new_prev = (crisis_regime.mean() > 0.5).float()
            self.crisis_prev_regime.copy_(new_prev.view(1))

        delta_sharpe = delta_sharpe_raw
        cvar = cvar_raw

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
        # Phase 3.1: Tanh output [-1, 1] → scale to [-5, 10] → clamp [0.01, 10]
        concentrations_raw = torch.stack([
            self.decoders[j](K[:, j, :]) for j in range(self.M)
        ], dim=1)  # [B, M, A] with Tanh output ∈ [-1, 1]
        
        # Transform: [-1, 1] → [-5, 10] allows true suppression
        # x * 7.5 + 2.5: -1 → -5, 0 → 2.5, 1 → 10
        concentrations = concentrations_raw * 7.5 + 2.5  # [B, M, A]
        concentrations = torch.clamp(concentrations, min=0.01)  # Ensure positive for Dirichlet

        # IRT 가중치로 혼합
        mixed_conc = torch.einsum('bm,bma->ba', w, concentrations)  # [B, A]

        if deterministic:
            # 결정적: softmax with temperature (logit-based, NOT Dirichlet)
            # Phase 3.1: mixed_conc ∈ [0.01, 10] can be used as logits
            # Lower temp → sharper (amplifies differences)
            # Higher temp → flatter (smooths differences)
            action = F.softmax(mixed_conc / self.action_temp, dim=-1)
        else:
            # 확률적: Dirichlet 샘플 (probability-based, different from deterministic!)
            # Phase 3.1: mixed_conc already positive after clamp, use directly as α
            # α < 1: Sparse (corners), α = 1: Uniform, α > 1: Peaked near uniform
            # Clamp to [dirichlet_min, dirichlet_max] for numerical stability
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
            'crisis_level_pre_guard': crisis_level_pre_guard.detach(),  # [B, 1]
            'crisis_raw': crisis_raw.detach(),  # [B, 1]
            'crisis_bias': self.crisis_bias.detach().clone(),  # [1]
            'crisis_temperature': self.crisis_temperature.detach().clone(),  # [1]
            'crisis_prev_regime': self.crisis_prev_regime.detach().clone(),  # [1]
            'crisis_base_component': crisis_base_component.detach(),  # [B, 1]
            'delta_component': delta_component.detach(),  # [B, 1]
            'cvar_component': cvar_component.detach(),  # [B, 1]
            'crisis_regime': crisis_regime.detach(),  # [B, 1]
            'crisis_guard_rate': torch.tensor(guard_rate, device=state.device),
            'bias_warm_started': self.bias_warm_started.detach().clone(),
            'crisis_types': z.detach(),  # [B, K]
            'fitness': fitness.detach(),  # [B, M]
            'hysteresis_up': torch.tensor(hysteresis_up_val, device=state.device),
            'hysteresis_down': torch.tensor(hysteresis_down_val, device=state.device),
            # IRT 분해 정보 (시각화용)
            'w_rep': irt_debug['w_rep'].detach(),  # [B, M] - Replicator 출력
            'w_ot': irt_debug['w_ot'].detach(),    # [B, M] - OT 출력
            'cost_matrix': irt_debug['cost_matrix'].detach(),  # [B, m, M]
            'eta': irt_debug['eta'].detach(),       # [B, 1] - Crisis learning rate
            'alpha_c': irt_debug['alpha_c'].detach(),  # [B, 1] - Dynamic OT-Replicator mixing ratio
            # Dirichlet 정보 (log_prob 계산용)
            'concentrations': concentrations.detach(),  # [B, M, A] - 프로토타입별 concentration
            'mixed_conc': mixed_conc.detach(),  # [B, A] - 혼합된 concentration (raw)
            'mixed_conc_clamped': mixed_conc_clamped.detach() if not deterministic else mixed_conc.detach(),  # [B, A]
            # Phase 3.5 Step 2: 다중 신호 위기 감지 정보
            'crisis_base': crisis_base_sigmoid.detach(),  # [B, 1] - T-Cell 기본 위기 신호 (sigmoid)
            'crisis_base_raw': crisis_affine_raw.detach(),  # [B, 1] - T-Cell affine
            'delta_sharpe': delta_sharpe.detach() if state.size(1) >= self.state_dim - 2 else torch.zeros(B, 1, device=state.device),
            'cvar': cvar.detach() if state.size(1) >= self.state_dim - 2 else torch.zeros(B, 1, device=state.device),
        }

        return action, info
