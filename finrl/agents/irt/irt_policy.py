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
        action_space: spaces.Box
    ):
        """
        Args:
            irt_actor: BCellIRTActor 인스턴스
            features_dim: feature extractor 출력 차원
            action_space: action space
        """
        # Actor 부모 클래스 초기화를 건너뛰고 nn.Module만 초기화
        nn.Module.__init__(self)

        self.irt_actor = irt_actor
        self.features_dim = features_dim
        self.action_space = action_space
        self.action_dim = int(action_space.shape[0])

        # Dirichlet 파라미터 저장
        self.dirichlet_min = irt_actor.dirichlet_min
        self.dirichlet_max = irt_actor.dirichlet_max

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Forward pass - mean actions 반환

        Args:
            obs: [B, features_dim]
            deterministic: 결정적 행동 여부

        Returns:
            mean_actions: [B, action_dim]
        """
        # IRT Actor forward
        action, info = self.irt_actor(
            state=obs,
            fitness=None,
            deterministic=deterministic
        )
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
        # ===== IRT forward (한 번만 호출) =====
        action, info = self.irt_actor(
            state=obs,
            fitness=None,
            deterministic=False
        )

        # ===== Log probability 계산 =====
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
        alpha: float = 0.3,
        ema_beta: float = 0.9,
        market_feature_dim: int = 12,
        dirichlet_min: float = 0.5,
        dirichlet_max: float = 50.0,
        eps: float = 0.10,
        eta_0: float = 0.05,
        eta_1: float = 0.15,
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
            alpha: OT-Replicator 혼합 비율
            ema_beta: EMA 메모리 계수
            market_feature_dim: 시장 특성 차원
            dirichlet_min: Dirichlet concentration minimum
            dirichlet_max: Dirichlet concentration maximum
            eps: Sinkhorn 엔트로피
            eta_0: 기본 학습률 (Replicator)
            eta_1: 위기 증가량 (Replicator)
        """
        # IRT 파라미터 저장
        self.emb_dim = emb_dim
        self.m_tokens = m_tokens
        self.M_proto = M_proto
        self.alpha_irt = alpha
        self.ema_beta = ema_beta
        self.market_feature_dim = market_feature_dim
        self.dirichlet_min = dirichlet_min
        self.dirichlet_max = dirichlet_max
        self.eps = eps
        self.eta_0 = eta_0
        self.eta_1 = eta_1

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
            ema_beta=self.ema_beta,
            market_feature_dim=self.market_feature_dim,
            dirichlet_min=self.dirichlet_min,
            dirichlet_max=self.dirichlet_max,
            eps=self.eps,
            eta_0=self.eta_0,
            eta_1=self.eta_1
        )

        # Wrapper로 감싸기
        actor = IRTActorWrapper(
            irt_actor=bcell_actor,
            features_dim=features_dim,
            action_space=self.action_space
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
                dirichlet_min=self.dirichlet_min,
                dirichlet_max=self.dirichlet_max,
                eps=self.eps,
                eta_0=self.eta_0,
                eta_1=self.eta_1,
            )
        )
        return data


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
