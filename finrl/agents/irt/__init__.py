# finrl/agents/irt/__init__.py

"""
IRT (Immune Replicator Transport) 모듈

핵심 구성 요소:
- IRT: Optimal Transport + Replicator Dynamics 결합
- TCellMinimal: 경량 위기 감지 시스템
- BCellIRTActor: IRT 기반 포트폴리오 관리 Actor
- Sinkhorn: 엔트로피 정규화 최적수송

사용법:
    from finrl.agents.irt import IRT, BCellIRTActor, TCellMinimal

    # IRT Actor 생성
    actor = BCellIRTActor(
        state_dim=181,
        action_dim=30,
        emb_dim=128,
        m_tokens=6,
        M_proto=8,
        alpha=0.3
    )

    # Forward pass
    action, info = actor(state)
"""

from finrl.agents.irt.irt_operator import IRT, Sinkhorn
from finrl.agents.irt.t_cell import TCellMinimal
from finrl.agents.irt.bcell_actor import BCellIRTActor
from finrl.agents.irt.irt_policy import IRTPolicy, IRTActorCriticPolicy

__all__ = [
    'IRT',
    'Sinkhorn',
    'TCellMinimal',
    'BCellIRTActor',
    'IRTPolicy',
    'IRTActorCriticPolicy',
]

__version__ = '1.0.0'
