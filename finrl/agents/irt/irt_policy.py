# finrl/agents/irt/irt_policy.py

"""
Stable Baselines3용 IRT Custom Policy

SB3의 BasePolicy를 상속하여 IRT Actor를 통합한다.

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

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.common.type_aliases import Schedule

from finrl.agents.irt.bcell_actor import BCellIRTActor


class IRTPolicy(BasePolicy):
    """
    IRT Policy for SAC

    SB3의 BasePolicy를 상속하여 IRT Actor를 통합한다.
    SAC는 별도로 Critic을 관리하므로, Policy는 Actor만 정의하면 된다.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
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
        **kwargs
    ):
        """
        Args:
            observation_space: 관측 공간
            action_space: 행동 공간
            lr_schedule: 학습률 스케줄
            features_extractor_class: Feature extractor 클래스
            features_extractor_kwargs: Feature extractor 인자
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
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
        )

        self.emb_dim = emb_dim
        self.m_tokens = m_tokens
        self.M_proto = M_proto
        self.alpha = alpha

        # Features extractor (SB3 기본)
        self.features_extractor = features_extractor_class(
            self.observation_space,
            **(features_extractor_kwargs or {})
        )
        features_dim = self.features_extractor.features_dim

        # Action space 검증
        assert isinstance(action_space, spaces.Box), "IRT only supports Box action space"

        state_dim = features_dim
        action_dim = int(action_space.shape[0])

        # ===== IRT Actor =====
        self.irt_actor = BCellIRTActor(
            state_dim=state_dim,
            action_dim=action_dim,
            emb_dim=emb_dim,
            m_tokens=m_tokens,
            M_proto=M_proto,
            alpha=alpha,
            ema_beta=ema_beta,
            market_feature_dim=market_feature_dim,
            dirichlet_min=dirichlet_min,
            dirichlet_max=dirichlet_max,
            eps=eps,
            eta_0=eta_0,
            eta_1=eta_1
        )

        # ===== IRT 중간 데이터 저장 (시각화용) =====
        self.last_info = None

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        """SB3 체크포인트 저장용 파라미터"""
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                emb_dim=self.emb_dim,
                m_tokens=self.m_tokens,
                M_proto=self.M_proto,
                alpha=self.alpha,
            )
        )
        return data

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Forward pass (SAC의 predict 호출 시 사용)

        Args:
            obs: 관측 [B, obs_dim]
            deterministic: 결정적 행동

        Returns:
            actions: [B, action_dim]
        """
        # Features extraction
        features = self.features_extractor(obs)

        # IRT forward (fitness는 SAC가 Critic으로 계산)
        action, info = self.irt_actor(
            state=features,
            fitness=None,  # SAC의 Critic이 별도로 관리
            deterministic=deterministic
        )

        # IRT 중간 데이터 저장 (시각화용)
        self.last_info = info

        return action

    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        SB3의 predict 메서드에서 호출
        """
        return self.forward(observation, deterministic=deterministic)

    def get_irt_info(self) -> Optional[Dict]:
        """
        마지막 forward pass의 IRT 중간 데이터 반환

        Returns:
            info: IRT 중간 데이터 (w, w_rep, w_ot, crisis_level, 등)
                  None if no forward pass has been executed yet
        """
        return self.last_info

    def set_training_mode(self, mode: bool) -> None:
        """
        학습/평가 모드 전환
        """
        self.train(mode)
        self.irt_actor.train(mode)


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
