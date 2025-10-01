# tests/test_irt.py

"""
IRT 단위 테스트

테스트 항목:
- Sinkhorn 수렴
- IRT forward pass
- T-Cell 출력
- B-Cell Actor 동작
"""

import torch
import torch.nn.functional as F
import pytest
import numpy as np
from src.immune.irt import IRT, Sinkhorn
from src.immune.t_cell import TCellMinimal
from src.agents.bcell_irt import BCellIRTActor
from src.algorithms.critics.redq import REDQCritic, QNetwork

def test_sinkhorn_convergence():
    """Sinkhorn 수렴 테스트"""
    sinkhorn = Sinkhorn(max_iters=20, eps=0.05)

    B, m, M = 4, 6, 8
    C = torch.randn(B, m, M)
    u = torch.full((B, m, 1), 1.0/m)
    v = torch.full((B, 1, M), 1.0/M)

    P = sinkhorn(C, u, v)

    # 제약 검증
    assert P.shape == (B, m, M)
    assert torch.allclose(P.sum(dim=2), u.squeeze(-1), atol=1e-2)
    assert torch.allclose(P.sum(dim=1), v.squeeze(1), atol=1e-2)
    assert (P >= 0).all()

def test_irt_forward():
    """IRT forward pass 테스트"""
    irt = IRT(emb_dim=64, m_tokens=4, M_proto=6, alpha=0.3)

    B = 2
    E = torch.randn(B, 4, 64)
    K = torch.randn(B, 6, 64)
    danger = torch.randn(B, 64)
    w_prev = torch.ones(B, 6) / 6
    fitness = torch.randn(B, 6)
    crisis = torch.tensor([[0.3], [0.7]])

    w, P = irt(E, K, danger, w_prev, fitness, crisis)

    # 검증
    assert w.shape == (B, 6)
    assert P.shape == (B, 4, 6)
    assert torch.allclose(w.sum(dim=1), torch.ones(B), atol=1e-3)
    assert (w >= 0).all() and (w <= 1).all()

def test_tcell_output():
    """T-Cell 출력 테스트"""
    t_cell = TCellMinimal(in_dim=12, emb_dim=128, n_types=4)

    B = 8
    features = torch.randn(B, 12)

    z, d, c = t_cell(features, update_stats=False)

    # 차원 검증
    assert z.shape == (B, 4)  # 위기 타입
    assert d.shape == (B, 128)  # 공자극 임베딩
    assert c.shape == (B, 1)  # 스칼라 위기 레벨

    # 범위 검증
    assert (c >= 0).all() and (c <= 1).all()  # 시그모이드 출력

def test_bcell_actor():
    """B-Cell Actor 테스트"""
    actor = BCellIRTActor(
        state_dim=43,
        action_dim=30,
        emb_dim=128,
        m_tokens=6,
        M_proto=8,
        alpha=0.3
    )

    B = 4
    state = torch.randn(B, 43)

    # Critics 생성 (fitness 계산용)
    critics = [QNetwork(43, 30, [256, 256]) for _ in range(3)]

    # Forward pass
    action, info = actor(state, critics=critics, deterministic=False)

    # 검증
    assert action.shape == (B, 30)
    assert torch.allclose(action.sum(dim=1), torch.ones(B), atol=1e-3)
    assert (action >= 0).all() and (action <= 1).all()

    # 정보 딕셔너리 검증
    assert 'w' in info
    assert 'P' in info
    assert 'crisis_level' in info
    assert info['w'].shape == (B, 8)  # 프로토타입 가중치

def test_redq_critic():
    """REDQ Critic 테스트"""
    critic = REDQCritic(
        state_dim=43,
        action_dim=30,
        n_critics=5,
        m_sample=2
    )

    B = 8
    state = torch.randn(B, 43)
    action = torch.randn(B, 30)
    action = F.softmax(action, dim=-1)  # 포트폴리오 가중치

    # Forward pass
    q_values = critic(state, action)

    # 검증
    assert len(q_values) == 5  # n_critics
    for q in q_values:
        assert q.shape == (B, 1)

    # Target Q 계산
    target_q = critic.get_target_q(state, action)
    assert target_q.shape == (B, 1)

def test_irt_extreme_cases():
    """IRT 극한 케이스 테스트"""
    # α=0 (순수 Replicator)
    irt_rep = IRT(emb_dim=64, m_tokens=4, M_proto=6, alpha=0.0)

    # α=1 (순수 OT)
    irt_ot = IRT(emb_dim=64, m_tokens=4, M_proto=6, alpha=1.0)

    B = 2
    E = torch.randn(B, 4, 64)
    K = torch.randn(B, 6, 64)
    danger = torch.randn(B, 64)
    w_prev = torch.ones(B, 6) / 6
    fitness = torch.randn(B, 6)
    crisis = torch.tensor([[0.5], [0.5]])

    w_rep, _ = irt_rep(E, K, danger, w_prev, fitness, crisis)
    w_ot, _ = irt_ot(E, K, danger, w_prev, fitness, crisis)

    # 두 극한이 다른 결과를 내는지 확인
    assert not torch.allclose(w_rep, w_ot, atol=1e-2)

def test_crisis_adaptation():
    """위기 적응 테스트"""
    actor = BCellIRTActor(state_dim=43, action_dim=30)

    B = 2
    # 정상 상태 vs 위기 상태
    state_normal = torch.randn(B, 43)
    state_normal[:, :12] = 0  # 낮은 위기 신호

    state_crisis = torch.randn(B, 43)
    state_crisis[:, :12] = 3  # 높은 위기 신호

    action_normal, info_normal = actor(state_normal, deterministic=True)
    action_crisis, info_crisis = actor(state_crisis, deterministic=True)

    # 위기 레벨이 다른지 확인
    assert info_crisis['crisis_level'].mean() > info_normal['crisis_level'].mean()

if __name__ == '__main__':
    # 모든 테스트 실행
    test_sinkhorn_convergence()
    test_irt_forward()
    test_tcell_output()
    test_bcell_actor()
    test_redq_critic()
    test_irt_extreme_cases()
    test_crisis_adaptation()
    print("모든 테스트 통과!")

    # pytest로 실행
    # pytest tests/test_irt.py -v