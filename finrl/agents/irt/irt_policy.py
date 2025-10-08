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
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
)
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
        policy: Optional["IRTPolicy"] = None,
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

        # Gaussian policy 파라미터 저장
        self.log_std_min = irt_actor.log_std_min
        self.log_std_max = irt_actor.log_std_max

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

        if policy is not None and hasattr(policy, "critic"):
            with torch.no_grad():
                # 각 프로토타입의 샘플 행동 생성
                K = self.irt_actor.proto_keys  # [M, D]
                proto_actions = []

                for j in range(M):
                    # 프로토타입 j의 Gaussian mean
                    mu_j = self.irt_actor.mu_decoders[j](
                        K[j : j + 1].expand(B, -1)
                    )  # [B, action_dim]
                    log_std_j = self.irt_actor.log_std_decoders[j](
                        K[j : j + 1].expand(B, -1)
                    )
                    log_std_j = torch.clamp(
                        log_std_j, self.log_std_min, self.log_std_max
                    )

                    # Mean action (deterministic, 안정적)
                    a_j = torch.softmax(mu_j, dim=-1)  # [B, action_dim]
                    proto_actions.append(a_j)

                proto_actions = torch.stack(proto_actions, dim=1)  # [B, M, action_dim]

                # Critic Q-value 계산 (Twin Q 중 최소값 사용)
                q_values = []
                for j in range(M):
                    # SB3 Critic: forward(obs, actions) → [q1, q2]
                    q_vals = policy.critic(
                        obs, proto_actions[:, j]
                    )  # Tuple[Tensor, Tensor]

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
        action, log_prob, info = self.irt_actor(
            state=obs, fitness=fitness, deterministic=deterministic
        )

        # 마지막 IRT info 저장 (평가/시각화용)
        self._last_irt_info = info

        return action

    def action_log_prob(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action and compute log probability

        BCellIRTActor가 log_prob를 직접 반환하므로 단순 전달

        Args:
            obs: [B, features_dim]

        Returns:
            action: [B, action_dim]
            log_prob: [B, 1]
        """
        # Ensure float32 dtype (환경에서 float64로 들어올 수 있음)
        obs = obs.float()

        # Fitness 계산 (Critic Q-network 사용)
        fitness = self._compute_fitness(obs)

        # BCellIRTActor에서 log_prob 직접 계산됨
        action, log_prob, info = self.irt_actor(
            state=obs, fitness=fitness, deterministic=False
        )

        # 마지막 IRT info 저장 (평가/시각화용)
        self._last_irt_info = info

        # XAI regularization 계산 (training 시에만)
        if self.training and hasattr(self.irt_actor, 'last_irt_info'):
            irt_info = self.irt_actor.last_irt_info
            if irt_info is not None and self._policy_ref is not None:
                policy = self._policy_ref()
                if policy is not None and hasattr(policy, '_compute_xai_regularization'):
                    xai_loss = policy._compute_xai_regularization(irt_info)
                    # XAI loss를 policy에 저장
                    policy._xai_loss = xai_loss

                    # Diversity loss도 함께 저장 (BCellIRTActor에서 계산됨)
                    diversity_loss = irt_info.get('diversity_loss', 0.0)
                    policy._diversity_loss = diversity_loss

        # Alpha scheduler 업데이트 (adaptive mode에서 실제 log_prob 사용)
        policy = self._policy_ref() if self._policy_ref is not None else None
        if policy is not None and hasattr(policy, 'alpha_scheduler') and policy.alpha_scheduler is not None:
            # Check if scheduler is in adaptive mode
            if policy.alpha_scheduler.schedule_type == 'adaptive':
                # Increment step count
                policy.step_count += obs.size(0)  # Batch size만큼 증가

                # Update alpha with real log_prob (detached for entropy estimation)
                # We detach because alpha update only needs the entropy value, not gradients
                # log_prob shape: [batch_size, 1] → scalar mean
                log_prob_detached = log_prob.detach().mean()
                new_alpha, alpha_loss = policy.alpha_scheduler.update(
                    policy.step_count,
                    log_prob_detached  # Pass detached tensor (will be converted to scalar internally)
                )

                # Apply updated alpha to IRT operator
                self.irt_actor.irt.alpha = new_alpha.item()

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
        alpha: float = 0.3,
        ema_beta: float = 0.9,
        market_feature_dim: int = 12,
        log_std_min: float = -20,
        log_std_max: float = 2,
        eps: float = 0.10,
        eta_0: float = 0.05,
        eta_1: float = 0.15,
        alpha_scheduler: Optional[Any] = None,  # AlphaScheduler instance (adaptive mode)
        xai_reg_weight: float = 0.01,  # XAI regularization weight
        use_shared_decoder: bool = False,  # Use shared decoder (ablation study)
    ):
        """
        Args:
            observation_space: 관측 공간
            action_space: 행동 공간
            lr_schedule: 학습률 스케줄
            (SAC반본 파라미터들...)
            emb_dim: IRT 임베딩 차원
            m_tokens: 에피토프 토큰 수
            M_proto: 프로토타입 수
            alpha: OT-Replicator 혼합 비율
            ema_beta: EMA 메모리 계수
            market_feature_dim: 시장 특성 차원
            log_std_min: Log standard deviation minimum (Gaussian policy)
            log_std_max: Log standard deviation maximum (Gaussian policy)
            eps: Sinkhorn 엔트로피
            eta_0: 기본 학습률 (Replicator)
            eta_1: 위기 증가량 (Replicator)
            alpha_scheduler: Optional AlphaScheduler for adaptive alpha tuning
        """
        # IRT 파라미터 저장
        # IRT 파라미터를 먼저 저장 (make_actor에서 사용됨)
        self.emb_dim = emb_dim
        self.m_tokens = m_tokens
        self.M_proto = M_proto
        self.alpha_irt = alpha
        self.ema_beta = ema_beta
        self.market_feature_dim = market_feature_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.eps = eps
        self.eta_0 = eta_0
        self.eta_1 = eta_1
        self.alpha_scheduler = alpha_scheduler
        self.xai_reg_weight = xai_reg_weight
        self.use_shared_decoder = use_shared_decoder
        self.step_count = 0  # Track steps for alpha scheduler
        self._xai_loss = 0.0  # Initialize XAI loss

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

    def make_actor(
        self, features_extractor: Optional[BaseFeaturesExtractor] = None
    ) -> IRTActorWrapper:
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
            ema_beta=self.ema_beta,
            market_feature_dim=self.market_feature_dim,
            log_std_min=self.log_std_min,
            log_std_max=self.log_std_max,
            use_shared_decoder=self.use_shared_decoder,
            eps=self.eps,
            eta_0=self.eta_0,
            eta_1=self.eta_1,
        )

        # Wrapper로 감싸기 (self 전달: Critic 참조용)
        actor = IRTActorWrapper(
            irt_actor=bcell_actor,
            features_dim=features_dim,
            action_space=self.action_space,
            policy=self,
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
                ema_beta=self.ema_beta,
                market_feature_dim=self.market_feature_dim,
                log_std_min=self.log_std_min,
                log_std_max=self.log_std_max,
                eps=self.eps,
                eta_0=self.eta_0,
                eta_1=self.eta_1,
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
                - crisis_level: [B, 1] - 위기 레벨
                - crisis_types: [B, K] - 위기 타입
                - cost_matrix: [B, m, M] - Immunological cost
                - P: [B, m, M] - 수송 계획
                - fitness: [B, M] - 프로토타입 적합도
            None: IRTPolicy가 아니거나 아직 forward 안 함
        """
        if hasattr(self, "actor") and hasattr(self.actor, "_last_irt_info"):
            return self.actor._last_irt_info
        return None

    def _compute_xai_regularization(self, irt_info: dict) -> torch.Tensor:
        """
        XAI regularization loss

        목표:
        1. w_rep, w_ot 모두 다양한 프로토타입 활용 (entropy 최대화)
        2. Crisis 시 Adaptive 메커니즘이 특정 프로토타입에 집중 (HHI 조정)

        Args:
            irt_info: BCellIRTActor.forward() 반환 정보
                - w_rep: [B, M] Adaptive mechanism (Replicator) 출력
                - w_ot: [B, M] Exploratory mechanism (OT) 출력
                - crisis_level: [B, 1]

        Returns:
            loss: scalar
        """
        w_rep = irt_info['w_rep']  # [B, M]
        w_ot = irt_info['w_ot']    # [B, M]
        crisis_level = irt_info['crisis_level']  # [B, 1]

        eps = 1e-8

        # 1. Diversity: Entropy 최대화 (collapse 방지)
        entropy_rep = -(w_rep * torch.log(w_rep + eps)).sum(dim=-1).mean()
        entropy_ot = -(w_ot * torch.log(w_ot + eps)).sum(dim=-1).mean()

        # 2. Crisis-adaptive specialization
        # HHI: 집중도 측정 (균등 분포 = 1/M, 완전 집중 = 1)
        hhi_rep = (w_rep ** 2).sum(dim=-1)  # [B]

        # Target HHI: 평시 0.3 (분산), 위기 0.7 (집중)
        target_hhi = 0.3 + 0.4 * crisis_level.squeeze(-1)  # [B]
        hhi_loss = ((hhi_rep - target_hhi) ** 2).mean()

        # 3. Total loss
        # Entropy는 최대화 → 부호 반전
        # HHI는 target에 근접 → L2 loss
        loss = (
            -0.01 * entropy_rep      # Adaptive diversity
            -0.01 * entropy_ot       # Exploratory diversity
            + 0.05 * hhi_loss        # Crisis-adaptive concentration
        )

        return loss

    def forward_actor(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Forward pass in the actor network with XAI losses computation

        This is called during training to get actions and compute losses.
        We override this to add XAI and diversity losses to the computation graph.
        """
        # Get action from actor
        mean_actions, log_std, kwargs = self.get_distribution(obs)

        # Get action distribution
        distribution = self.action_dist.proba_distribution(mean_actions, log_std)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)

        # Compute XAI and diversity losses if in training mode
        if self.training and hasattr(self, 'actor') and hasattr(self.actor, 'irt_actor'):
            if hasattr(self.actor.irt_actor, 'last_irt_info'):
                irt_info = self.actor.irt_actor.last_irt_info
                if irt_info is not None:
                    # Store losses for later use in actor optimization
                    self._current_xai_loss = self._compute_xai_regularization(irt_info)
                    diversity_loss = irt_info.get('diversity_loss', 0.0)
                    if not isinstance(diversity_loss, torch.Tensor):
                        diversity_loss = torch.tensor(diversity_loss, device=obs.device)
                    self._current_diversity_loss = diversity_loss
                else:
                    self._current_xai_loss = torch.tensor(0.0, device=obs.device)
                    self._current_diversity_loss = torch.tensor(0.0, device=obs.device)
            else:
                self._current_xai_loss = torch.tensor(0.0, device=obs.device)
                self._current_diversity_loss = torch.tensor(0.0, device=obs.device)
        else:
            self._current_xai_loss = torch.tensor(0.0, device=obs.device) if torch.is_tensor(obs) else 0.0
            self._current_diversity_loss = torch.tensor(0.0, device=obs.device) if torch.is_tensor(obs) else 0.0

        return actions, log_prob

    def compute_actor_loss(self, log_prob: torch.Tensor, q_value: torch.Tensor) -> torch.Tensor:
        """
        Compute actor loss with XAI/Diversity regularization

        This method computes the standard SAC actor loss and adds our custom losses.
        It's designed to be called within the SAC training loop.
        """
        # Standard SAC actor loss
        if self.ent_coef_tensor is not None:
            # Entropy regularization
            actor_loss = (self.ent_coef_tensor * log_prob - q_value).mean()
        else:
            # No entropy regularization
            actor_loss = -q_value.mean()

        # Add XAI and diversity losses
        if hasattr(self, '_current_xai_loss') and hasattr(self, '_current_diversity_loss'):
            total_regularization = self._current_xai_loss + self._current_diversity_loss
            actor_loss = actor_loss + self.xai_reg_weight * total_regularization

        return actor_loss


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
