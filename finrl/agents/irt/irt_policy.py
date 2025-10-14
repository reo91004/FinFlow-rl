# finrl/agents/irt/irt_policy.py

"""
Stable Baselines3용 IRT Custom Policy

SB3의 SACPolicy를 상속하여 IRT Actor를 통합한다.

핵심 설계:
- SAC의 기본 Critic (2 Q-networks) 재사용
- Actor만 IRT로 교체 (BCellIRTActor)
- Value network는 SB3 기본 사용

사용법:
    from stable_baselines3 import SAC
    from finrl.agents.irt.irt_policy import IRTPolicy

    model = SAC(
        policy=IRTPolicy,
        env=env,
        policy_kwargs={
            "emb_dim": 128,
            "m_tokens": 6,
            "M_proto": 8,
            "alpha": 0.3
        }
    )
"""

import torch
import torch.nn as nn
import weakref
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from gymnasium import spaces

from stable_baselines3.sac.policies import SACPolicy, Actor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.preprocessing import get_action_dim

from finrl.agents.irt.bcell_actor import BCellIRTActor


class IRTActorWrapper(Actor):
    """
    BCellIRTActor를 SB3의 Actor 인터페이스로 wrapping

    SAC가 기대하는 메서드들을 제공한다:
    - forward(): mean actions 반환
    - action_log_prob(): action과 log_prob 반환
    """

    def __init__(
        self,
        irt_actor: BCellIRTActor,
        features_dim: int,
        action_space: spaces.Box,
        policy: Optional['IRTPolicy'] = None
    ):
        """
        Args:
            irt_actor: BCellIRTActor 인스턴스
            features_dim: feature extractor 출력 차원
            action_space: action space
            policy: parent IRTPolicy (Critic 참조용)
        """
        # Actor 부모 클래스 초기화를 건너뛰고 nn.Module만 초기화
        nn.Module.__init__(self)

        self.irt_actor = irt_actor
        self.features_dim = features_dim
        self.action_space = action_space
        self.action_dim = int(action_space.shape[0])

        # Policy를 weakref로 저장 (순환 참조 방지)
        self._policy_ref = weakref.ref(policy) if policy is not None else None

        # Dirichlet 파라미터 저장
        self.dirichlet_min = irt_actor.dirichlet_min
        self.dirichlet_max = irt_actor.dirichlet_max

        # 마지막 forward의 IRT info 저장 (평가/시각화용)
        self._last_irt_info = None

    def _compute_fitness(self, obs: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Critic 기반 fitness 계산 (공통 helper method)

        Args:
            obs: [B, features_dim]

        Returns:
            fitness: [B, M] - 프로토타입별 Q-value, None if critic unavailable
        """
        B = obs.size(0)
        M = self.irt_actor.M

        fitness = None

        # Weakref로부터 policy 가져오기
        policy = self._policy_ref() if self._policy_ref is not None else None

        if policy is not None and hasattr(policy, 'critic'):
            with torch.no_grad():
                # 각 프로토타입의 샘플 행동 생성
                K = self.irt_actor.proto_keys  # [M, D]
                proto_actions = []

                for j in range(M):
                    # 프로토타입 j의 concentration
                    conc_j = self.irt_actor.decoders[j](K[j:j+1].expand(B, -1))  # [B, action_dim]
                    conc_j_clamped = torch.clamp(conc_j, min=self.dirichlet_min, max=self.dirichlet_max)

                    # 샘플 행동 (mode 사용: 더 안정적)
                    # Dirichlet mode = (α - 1) / (Σα - K) for α > 1
                    # 안전하게 softmax 사용
                    a_j = torch.softmax(conc_j_clamped, dim=-1)  # [B, action_dim]
                    proto_actions.append(a_j)

                proto_actions = torch.stack(proto_actions, dim=1)  # [B, M, action_dim]

                # Critic Q-value 계산 (Twin Q 중 최소값 사용)
                q_values = []
                for j in range(M):
                    # SB3 Critic: forward(obs, actions) → [q1, q2]
                    q_vals = policy.critic(obs, proto_actions[:, j])  # Tuple[Tensor, Tensor]

                    # Twin Q의 최소값 (conservative)
                    if isinstance(q_vals, tuple):
                        q_min = torch.min(q_vals[0], q_vals[1]).squeeze(-1)  # [B]
                    else:
                        q_min = q_vals.squeeze(-1) if q_vals.ndim > 1 else q_vals  # [B]

                    q_values.append(q_min)

                fitness = torch.stack(q_values, dim=1)  # [B, M]

        return fitness

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Forward pass - mean actions 반환

        Args:
            obs: [B, features_dim]
            deterministic: 결정적 행동 여부

        Returns:
            mean_actions: [B, action_dim]
        """
        # Ensure float32 dtype (환경에서 float64로 들어올 수 있음)
        obs = obs.float()

        # Critic 기반 fitness 계산
        fitness = self._compute_fitness(obs)

        # IRT Actor forward
        action, info = self.irt_actor(
            state=obs,
            fitness=fitness,
            deterministic=deterministic
        )

        # 마지막 IRT info 저장 (평가/시각화용)
        self._last_irt_info = info

        return action

    def action_log_prob(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action and compute log probability

        Args:
            obs: [B, features_dim]

        Returns:
            action: [B, action_dim]
            log_prob: [B, 1]
        """
        # Ensure float32 dtype (환경에서 float64로 들어올 수 있음)
        obs = obs.float()

        # ===== Step 1: Fitness 계산 (Critic Q-network 사용) =====
        fitness = self._compute_fitness(obs)

        # ===== Step 2: IRT forward with fitness =====
        action, info = self.irt_actor(
            state=obs,
            fitness=fitness,
            deterministic=False
        )

        # 마지막 IRT info 저장 (평가/시각화용)
        self._last_irt_info = info

        # ===== Step 3: Log probability 계산 =====
        # info에서 Dirichlet concentration 가져오기
        mixed_conc_clamped = info['mixed_conc_clamped']

        # Dirichlet distribution으로 log_prob 계산
        dist = torch.distributions.Dirichlet(mixed_conc_clamped)
        log_prob = dist.log_prob(action).unsqueeze(-1)  # [B, 1]

        return action, log_prob

    def get_std(self) -> torch.Tensor:
        """
        Return standard deviation (Dirichlet의 경우 approximation)

        Dirichlet의 variance는 α_i(α_0 - α_i) / (α_0^2 * (α_0 + 1))
        여기서는 간단히 0.1 반환 (SAC의 log_std 학습을 우회)
        """
        return torch.ones(self.action_dim) * 0.1


class IRTPolicy(SACPolicy):
    """
    IRT Policy for SAC

    SB3의 SACPolicy를 상속하여 IRT Actor를 통합한다.
    SAC의 기본 Critic을 재사용하고, Actor만 IRT로 교체한다.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        # IRT-specific parameters
        emb_dim: int = 128,
        m_tokens: int = 6,
        M_proto: int = 8,
        alpha: float = 0.45,  # Phase F: alpha_max 기본값
        alpha_min: float = 0.08,  # Phase F: Rep 경로 확보
        alpha_max: Optional[float] = None,
        ema_beta: float = 0.55,  # Phase 1.5: 0.70 → 0.55 (faster responsiveness)
        market_feature_dim: Optional[int] = None,
        stock_dim: Optional[int] = None,
        tech_indicator_count: Optional[int] = None,
        has_dsr_cvar: bool = False,
        dirichlet_min: float = 0.05,
        dirichlet_max: float = 6.0,
        action_temp: float = 0.50,
        eps: float = 0.03,  # Phase-F2': 0.05 → 0.03 (OT 평탄화 완화)
        max_iters: int = 30,
        replicator_temp: float = 0.4,
        eta_0: float = 0.05,
        eta_1: float = 0.12,  # Phase E: 민감도 완화
        alpha_update_rate: float = 1.0,
        alpha_feedback_gain: float = 0.25,
        alpha_feedback_bias: float = 0.0,
        directional_decay_min: float = 0.05,
        alpha_noise_std: float = 0.0,
        gamma: float = 0.90,  # Phase 1.5: cost responsiveness ↑ (0.85 → 0.90)
        # Phase 1: Crisis calibration defaults
        w_r: float = 0.55,
        w_s: float = -0.25,
        w_c: float = 0.20,
        eta_b: float = 0.02,
        eta_b_min: float = 0.002,
        eta_b_decay_steps: int = 30000,
        eta_T: float = 0.01,
        p_star: float = 0.35,
        temperature_min: float = 0.7,
        temperature_max: float = 1.1,
        stat_momentum: float = 0.92,
        eta_b_warmup_steps: int = 10000,
        eta_b_warmup_value: float = 0.05,
        alpha_crisis_source: str = "pre_guard",
        # T-Cell guard + hysteresis
        crisis_target: float = 0.5,  # 목표 crisis_regime_pct
        crisis_guard_rate_init: float = 0.07,
        crisis_guard_rate_final: float = 0.02,
        crisis_guard_warmup_steps: int = 7500,
        # Phase 1.5: 히스테리시스 임계치 재조정 (0.55/0.45 → 0.52/0.42)
        hysteresis_up: float = 0.52,
        hysteresis_down: float = 0.42,
        adaptive_hysteresis: bool = True,
        # Phase 1.5: 분위수 상향 (0.65 → 0.72)
        hysteresis_quantile: float = 0.72,
        hysteresis_min_gap: float = 0.03,
        crisis_history_len: int = 512,
        k_s: float = 6.0,
        k_c: float = 6.0,
        k_b: float = 4.0,
        crisis_guard_rate: Optional[float] = None,
    ):
        """
        Args:
            observation_space: 관측 공간
            action_space: 행동 공간
            lr_schedule: 학습률 스케줄
            (SACPolicy 기본 파라미터들...)
            emb_dim: IRT 임베딩 차원
            m_tokens: 에피토프 토큰 수
            M_proto: 프로토타입 수
            alpha: OT-Replicator 혼합 비율 (후진 호환, alpha_max 기본값)
            alpha_min: 위기 시 최소 α (Phase F: 0.08)
            alpha_max: 평시 최대 α (Phase F: 0.45)
            ema_beta: EMA 메모리 계수
            market_feature_dim: 시장 특성 차원
            dirichlet_min: Dirichlet concentration minimum
            dirichlet_max: Dirichlet concentration maximum
            eps: Sinkhorn 엔트로피
            replicator_temp: Replicator softmax 온도 (Phase F: 1.4, >1이면 평탄화)
            eta_0: 기본 학습률 (Replicator)
            eta_1: 위기 증가량 (Replicator, Phase E: 0.12)
            gamma: 공자극 가중치 (OT 비용 함수, Phase E: 0.85)
            w_r: 시장 위기 신호 가중치 (Phase 1 재조정 기본값 0.55)
            w_s: Sharpe 신호 가중치 (Phase 1: 음수로 위기 완화 반영)
            w_c: CVaR 신호 가중치 (Phase 1: 0.20)
            eta_b: 바이어스 초기 학습률 (Phase 1: 0.02)
            eta_b_min: 코사인 감쇠 후 최소 학습률
            eta_b_decay_steps: 바이어스 학습률 감쇠 스텝
            eta_T: 온도 적응 학습률
            p_star: 목표 위기 점유율
            temperature_min/temperature_max: 온도 클램프 범위
            stat_momentum: 위기 신호 통계용 EMA 모멘텀
            eta_b_warmup_steps/value: 바이어스 초기 워밍업 설정
            crisis_guard_rate_init/final: 가드 강도 스케줄
            crisis_guard_warmup_steps: 가드 스케줄 워밍업 스텝
            hysteresis_up/down: 위기 전환 히스테리시스 임계값
            k_s/k_c/k_b: Sharpe·CVaR·기본 위기 시그모이드 기울기
            crisis_guard_rate: (선택) 기존 고정 가드 비율 호환성 오버라이드
        """
        # IRT 파라미터 저장
        self.emb_dim = emb_dim
        self.m_tokens = m_tokens
        self.M_proto = M_proto
        self.alpha_irt = alpha
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.ema_beta = ema_beta
        self.stock_dim_meta = stock_dim
        self.tech_indicator_count = (
            int(tech_indicator_count) if tech_indicator_count is not None else None
        )
        self.has_dsr_cvar = bool(has_dsr_cvar)
        if market_feature_dim is not None:
            self.market_feature_dim = int(market_feature_dim)
        elif self.tech_indicator_count is not None:
            self.market_feature_dim = 4 + self.tech_indicator_count
        else:
            self.market_feature_dim = 12
        self.dirichlet_min = dirichlet_min
        self.dirichlet_max = dirichlet_max
        self.action_temp = action_temp  # Phase-2
        self.eps = eps
        self.max_iters = max_iters
        self.replicator_temp = replicator_temp
        self.eta_0 = eta_0
        self.eta_1 = eta_1
        self.alpha_update_rate = alpha_update_rate
        self.alpha_feedback_gain = alpha_feedback_gain
        self.alpha_feedback_bias = alpha_feedback_bias
        self.directional_decay_min = directional_decay_min
        self.alpha_noise_std = alpha_noise_std
        self.gamma = gamma
        self.w_r = w_r
        self.w_s = w_s
        self.w_c = w_c
        self.eta_b = eta_b
        self.eta_b_min = eta_b_min
        self.eta_b_decay_steps = eta_b_decay_steps
        self.eta_T = eta_T
        self.p_star = p_star
        self.temperature_min = temperature_min
        self.temperature_max = temperature_max
        self.stat_momentum = stat_momentum
        self.eta_b_warmup_steps = eta_b_warmup_steps
        self.eta_b_warmup_value = eta_b_warmup_value
        if alpha_crisis_source not in {"pre_guard", "post_guard"}:
            raise ValueError(
                f"alpha_crisis_source must be 'pre_guard' or 'post_guard', got {alpha_crisis_source}"
            )
        self.alpha_crisis_source = alpha_crisis_source
        self.crisis_target = crisis_target  # T-Cell 가드
        self.crisis_guard_rate_init = crisis_guard_rate_init
        self.crisis_guard_rate_final = crisis_guard_rate_final
        self.crisis_guard_warmup_steps = crisis_guard_warmup_steps
        self.hysteresis_up = hysteresis_up
        self.hysteresis_down = hysteresis_down
        self.adaptive_hysteresis = adaptive_hysteresis
        self.hysteresis_quantile = hysteresis_quantile
        self.hysteresis_min_gap = hysteresis_min_gap
        self.crisis_history_len = crisis_history_len
        self.k_s = k_s
        self.k_c = k_c
        self.k_b = k_b
        self.crisis_guard_rate = crisis_guard_rate  # Optional legacy override

        # SACPolicy 초기화
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            use_sde=use_sde,
            log_std_init=log_std_init,
            use_expln=use_expln,
            clip_mean=clip_mean,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            n_critics=n_critics,
            share_features_extractor=share_features_extractor,
        )

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> IRTActorWrapper:
        """
        Create IRT Actor

        SACPolicy의 make_actor를 override하여 IRT Actor를 생성한다.
        """
        # Features extractor가 없으면 기본 생성
        if features_extractor is None:
            features_extractor = self.make_features_extractor()

        features_dim = features_extractor.features_dim
        action_dim = get_action_dim(self.action_space)

        # BCellIRTActor 생성
        bcell_actor = BCellIRTActor(
            state_dim=features_dim,
            action_dim=action_dim,
            emb_dim=self.emb_dim,
            m_tokens=self.m_tokens,
            M_proto=self.M_proto,
            alpha=self.alpha_irt,
            alpha_min=self.alpha_min,
            alpha_max=self.alpha_max,
            ema_beta=self.ema_beta,
            market_feature_dim=self.market_feature_dim,
            stock_dim=self.stock_dim_meta or action_dim,
            tech_indicator_count=self.tech_indicator_count,
            has_dsr_cvar=self.has_dsr_cvar,
            dirichlet_min=self.dirichlet_min,
            dirichlet_max=self.dirichlet_max,
            action_temp=self.action_temp,  # Phase-2
            eps=self.eps,
            max_iters=self.max_iters,
            replicator_temp=self.replicator_temp,
            eta_0=self.eta_0,
            eta_1=self.eta_1,
            alpha_update_rate=self.alpha_update_rate,
            alpha_feedback_gain=self.alpha_feedback_gain,
            alpha_feedback_bias=self.alpha_feedback_bias,
            directional_decay_min=self.directional_decay_min,
            alpha_noise_std=self.alpha_noise_std,
            gamma=self.gamma,
            w_r=self.w_r,
            w_s=self.w_s,
            w_c=self.w_c,
            eta_b=self.eta_b,
            eta_b_min=self.eta_b_min,
            eta_b_decay_steps=self.eta_b_decay_steps,
            eta_T=self.eta_T,
            p_star=self.p_star,
            temperature_min=self.temperature_min,
            temperature_max=self.temperature_max,
            stat_momentum=self.stat_momentum,
            eta_b_warmup_steps=self.eta_b_warmup_steps,
            eta_b_warmup_value=self.eta_b_warmup_value,
            alpha_crisis_source=self.alpha_crisis_source,
            crisis_target=self.crisis_target,  # T-Cell 가드
            crisis_guard_rate_init=self.crisis_guard_rate_init,
            crisis_guard_rate_final=self.crisis_guard_rate_final,
            crisis_guard_warmup_steps=self.crisis_guard_warmup_steps,
            hysteresis_up=self.hysteresis_up,
            hysteresis_down=self.hysteresis_down,
            adaptive_hysteresis=self.adaptive_hysteresis,
            hysteresis_quantile=self.hysteresis_quantile,
            hysteresis_min_gap=self.hysteresis_min_gap,
            crisis_history_len=self.crisis_history_len,
            k_s=self.k_s,
            k_c=self.k_c,
            k_b=self.k_b,
            crisis_guard_rate=self.crisis_guard_rate  # Legacy override if provided
        )

        # Wrapper로 감싸기 (self 전달: Critic 참조용)
        actor = IRTActorWrapper(
            irt_actor=bcell_actor,
            features_dim=features_dim,
            action_space=self.action_space,
            policy=self
        )

        return actor

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        """SB3 체크포인트 저장용 파라미터"""
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                emb_dim=self.emb_dim,
                m_tokens=self.m_tokens,
                M_proto=self.M_proto,
                alpha=self.alpha_irt,
                alpha_min=self.alpha_min,
                alpha_max=self.alpha_max,
                ema_beta=self.ema_beta,
                market_feature_dim=self.market_feature_dim,
                stock_dim=self.stock_dim_meta,
                tech_indicator_count=self.tech_indicator_count,
                has_dsr_cvar=self.has_dsr_cvar,
                dirichlet_min=self.dirichlet_min,
                dirichlet_max=self.dirichlet_max,
                action_temp=self.action_temp,
                eps=self.eps,
                max_iters=self.max_iters,
                replicator_temp=self.replicator_temp,
                eta_0=self.eta_0,
                eta_1=self.eta_1,
                alpha_update_rate=self.alpha_update_rate,
                alpha_feedback_gain=self.alpha_feedback_gain,
                alpha_feedback_bias=self.alpha_feedback_bias,
                directional_decay_min=self.directional_decay_min,
                gamma=self.gamma,
                w_r=self.w_r,
                w_s=self.w_s,
                w_c=self.w_c,
                eta_b=self.eta_b,
                eta_b_min=self.eta_b_min,
                eta_b_decay_steps=self.eta_b_decay_steps,
                eta_T=self.eta_T,
                p_star=self.p_star,
                temperature_min=self.temperature_min,
                temperature_max=self.temperature_max,
                stat_momentum=self.stat_momentum,
                eta_b_warmup_steps=self.eta_b_warmup_steps,
                eta_b_warmup_value=self.eta_b_warmup_value,
                alpha_crisis_source=self.alpha_crisis_source,
                crisis_target=self.crisis_target,
                crisis_guard_rate_init=self.crisis_guard_rate_init,
                crisis_guard_rate_final=self.crisis_guard_rate_final,
                crisis_guard_warmup_steps=self.crisis_guard_warmup_steps,
                hysteresis_up=self.hysteresis_up,
                hysteresis_down=self.hysteresis_down,
                adaptive_hysteresis=self.adaptive_hysteresis,
                hysteresis_quantile=self.hysteresis_quantile,
                hysteresis_min_gap=self.hysteresis_min_gap,
                crisis_history_len=self.crisis_history_len,
                k_s=self.k_s,
                k_c=self.k_c,
                k_b=self.k_b,
                crisis_guard_rate=self.crisis_guard_rate,
            )
        )
        return data

    def get_irt_info(self) -> Optional[Dict]:
        """
        마지막 forward의 IRT 정보 반환 (평가/시각화용)

        Returns:
            dict: IRT info containing:
                - w: [B, M] - 프로토타입 혼합 가중치
                - w_rep: [B, M] - Replicator 출력
                - w_ot: [B, M] - OT 출력
                - crisis_level: [B, 1] - 위기 레벨 (guard 후)
                - crisis_level_pre_guard: [B, 1] - guard 적용 전 확률
                - crisis_raw: [B, 1] - 가중 합산 결과
                - crisis_bias / crisis_temperature: [1] - EMA 보정 상태
                - crisis_prev_regime: [1] - 이전 레짐 플래그
                - crisis_base_component / delta_component / cvar_component: [B, 1] - 신호별 기여
                - crisis_regime: [B, 1] - 히스테리시스 기반 레짐
                - crisis_guard_rate: float - 현재 guard 비율
                - crisis_types: [B, K] - 위기 타입
                - cost_matrix: [B, m, M] - Immunological cost
                - P: [B, m, M] - 수송 계획
                - fitness: [B, M] - 프로토타입 적합도
                - eta: [B, 1] - Crisis-adaptive learning rate
                - alpha_c: [B, 1] - Dynamic OT-Replicator mixing ratio
                - crisis_base / crisis_base_raw: [B, 1] - T-Cell baseline (sigmoid/logit)
                - delta_sharpe / cvar: [B, 1] - 입력 Sharpe, CVaR 신호
            None: IRTPolicy가 아니거나 아직 forward 안 함
        """
        if hasattr(self, 'actor') and hasattr(self.actor, '_last_irt_info'):
            return self.actor._last_irt_info
        return None


class IRTActorCriticPolicy(IRTPolicy):
    """
    IRT Actor + SB3 기본 Critic

    SAC는 Actor와 Critic을 별도로 관리하므로, 이 클래스는 주로 호환성을 위한 것이다.
    실제로는 IRTPolicy를 SAC의 policy로 사용하면 된다.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


# Alias for compatibility
IRT_Policy = IRTPolicy
