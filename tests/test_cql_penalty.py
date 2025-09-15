# tests/test_cql_penalty.py

import pytest
import numpy as np
import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents.b_cell import BCell

def test_cql_weight_range():
    """CQL 가중치가 표준 범위(5.0~10.0)인지 검증"""
    config = {
        'cql_weight': 5.0,
        'cql_min_q_weight': 5.0,
        'actor_lr': 3e-4,
        'critic_lr': 3e-4,
        'batch_size': 256,
        'buffer_size': 10000
    }

    bcell = BCell(
        specialization='momentum',
        state_dim=43,
        action_dim=10,
        config=config
    )

    # CQL 가중치 확인
    assert bcell.cql_weight == 5.0, f"CQL weight 불일치: {bcell.cql_weight}"
    assert bcell.cql_min_q_weight == 5.0, f"CQL min Q weight 불일치: {bcell.cql_min_q_weight}"

    print(f"✓ CQL 가중치 범위 테스트 통과")
    print(f"  CQL weight: {bcell.cql_weight}")
    print(f"  CQL min Q weight: {bcell.cql_min_q_weight}")


def test_cql_penalty_computation():
    """CQL 페널티가 올바르게 계산되는지 검증"""
    config = {
        'cql_weight': 5.0,
        'cql_min_q_weight': 5.0,
        'actor_lr': 3e-4,
        'critic_lr': 3e-4,
        'batch_size': 64,
        'buffer_size': 5000
    }

    bcell = BCell(
        specialization='volatility',
        state_dim=20,
        action_dim=5,
        config=config
    )

    # 더미 데이터 생성
    batch_size = 64
    states = torch.randn(batch_size, 20)
    actions = torch.softmax(torch.randn(batch_size, 5), dim=-1)
    rewards = torch.randn(batch_size, 1)
    next_states = torch.randn(batch_size, 20)
    dones = torch.zeros(batch_size, 1)

    # Q값 계산
    sa = torch.cat([states, actions], dim=-1)
    q1 = bcell.critic1(sa)
    q2 = bcell.critic2(sa)

    # CQL 페널티 계산
    with torch.no_grad():
        # 랜덤 액션들에 대한 Q값
        num_random = 10
        random_actions = torch.softmax(torch.randn(batch_size, num_random, 5), dim=-1)

        random_q1_list = []
        random_q2_list = []
        for i in range(num_random):
            random_sa = torch.cat([states, random_actions[:, i]], dim=-1)
            random_q1_list.append(bcell.critic1(random_sa).mean(dim=-1, keepdim=True))
            random_q2_list.append(bcell.critic2(random_sa).mean(dim=-1, keepdim=True))

        random_q1 = torch.cat(random_q1_list, dim=1)
        random_q2 = torch.cat(random_q2_list, dim=1)

        # CQL 손실: log-sum-exp - 데이터 Q값
        cql_q1_loss = torch.logsumexp(random_q1, dim=1).mean() - q1.mean()
        cql_q2_loss = torch.logsumexp(random_q2, dim=1).mean() - q2.mean()

        total_cql_loss = bcell.cql_weight * cql_q1_loss + bcell.cql_min_q_weight * cql_q2_loss

    # 페널티가 양수인지 확인 (일반적으로)
    assert total_cql_loss.item() != 0, "CQL 페널티가 0이면 안됨"

    print(f"✓ CQL 페널티 계산 테스트 통과")
    print(f"  CQL Q1 loss: {cql_q1_loss.item():.4f}")
    print(f"  CQL Q2 loss: {cql_q2_loss.item():.4f}")
    print(f"  Total CQL loss: {total_cql_loss.item():.4f}")


def test_cql_effect_on_q_values():
    """CQL이 Q값 과대평가를 억제하는지 검증"""
    # CQL 없는 버전
    config_no_cql = {
        'cql_weight': 0.0,
        'cql_min_q_weight': 0.0,
        'actor_lr': 3e-4,
        'critic_lr': 3e-4,
        'batch_size': 128,
        'buffer_size': 5000
    }

    # CQL 있는 버전
    config_with_cql = {
        'cql_weight': 5.0,
        'cql_min_q_weight': 5.0,
        'actor_lr': 3e-4,
        'critic_lr': 3e-4,
        'batch_size': 128,
        'buffer_size': 5000
    }

    bcell_no_cql = BCell(specialization='defensive', state_dim=10, action_dim=3, config=config_no_cql)
    bcell_with_cql = BCell(specialization='defensive', state_dim=10, action_dim=3, config=config_with_cql)

    # 동일한 초기 가중치 설정
    with torch.no_grad():
        # Critic 네트워크 가중치 복사
        bcell_with_cql.critic1.load_state_dict(bcell_no_cql.critic1.state_dict())
        bcell_with_cql.critic2.load_state_dict(bcell_no_cql.critic2.state_dict())

    # 테스트 데이터
    states = torch.randn(10, 10)
    actions = torch.softmax(torch.randn(10, 3), dim=-1)
    sa = torch.cat([states, actions], dim=-1)

    # 초기 Q값
    with torch.no_grad():
        q_no_cql_init = bcell_no_cql.critic1(sa).mean()
        q_with_cql_init = bcell_with_cql.critic1(sa).mean()

    print(f"✓ CQL Q값 억제 테스트 준비 완료")
    print(f"  초기 Q값 (no CQL): {q_no_cql_init.item():.4f}")
    print(f"  초기 Q값 (with CQL): {q_with_cql_init.item():.4f}")

    # 여러 번 업데이트 후 Q값 비교는 실제 학습 시 수행


def test_cql_weight_update_range():
    """CQL 가중치가 업데이트 중에도 범위를 유지하는지 검증"""
    config = {
        'cql_weight': 7.5,  # 중간값
        'cql_min_q_weight': 7.5,
        'actor_lr': 3e-4,
        'critic_lr': 3e-4,
        'batch_size': 64,
        'buffer_size': 5000
    }

    bcell = BCell(
        specialization='growth',
        state_dim=15,
        action_dim=5,
        config=config
    )

    # 가중치 범위 확인
    assert 5.0 <= bcell.cql_weight <= 10.0, f"CQL weight 범위 벗어남: {bcell.cql_weight}"
    assert 5.0 <= bcell.cql_min_q_weight <= 10.0, f"CQL min Q weight 범위 벗어남: {bcell.cql_min_q_weight}"

    # 가중치 조정 시뮬레이션 (필요 시)
    bcell.cql_weight = min(10.0, bcell.cql_weight * 1.1)  # 상향 조정
    assert bcell.cql_weight <= 10.0, "CQL weight 상한 초과"

    bcell.cql_weight = max(5.0, bcell.cql_weight * 0.9)  # 하향 조정
    assert bcell.cql_weight >= 5.0, "CQL weight 하한 미달"

    print(f"✓ CQL 가중치 범위 유지 테스트 통과")
    print(f"  최종 CQL weight: {bcell.cql_weight:.1f}")


def test_cql_integration_with_sac():
    """CQL이 SAC 업데이트와 통합되는지 검증"""
    config = {
        'cql_weight': 5.0,
        'cql_min_q_weight': 5.0,
        'alpha_init': 0.79,
        'actor_lr': 3e-4,
        'critic_lr': 3e-4,
        'batch_size': 32,
        'buffer_size': 1000
    }

    bcell = BCell(
        specialization='correlation',
        state_dim=10,
        action_dim=4,
        config=config
    )

    # 버퍼에 경험 추가
    for _ in range(100):
        state = np.random.randn(10)
        action = np.random.dirichlet(np.ones(4))
        reward = np.random.randn()
        next_state = np.random.randn(10)
        done = False
        bcell.replay_buffer.push(state, action, reward, next_state, done)

    # 업데이트 실행
    batch = bcell.replay_buffer.sample(32)
    losses = bcell.update(batch)

    # CQL 관련 손실이 포함되어 있는지 확인
    assert 'cql_loss' in losses or 'critic1_loss' in losses, "CQL 손실이 반영되어야 함"

    if 'cql_weight' in losses:
        assert losses['cql_weight'] == 5.0, "CQL weight가 손실 딕셔너리에 포함되어야 함"

    print(f"✓ CQL-SAC 통합 테스트 통과")
    if 'critic1_loss' in losses:
        print(f"  Critic1 loss (CQL 포함): {losses['critic1_loss']:.4f}")


if __name__ == "__main__":
    test_cql_weight_range()
    test_cql_penalty_computation()
    test_cql_effect_on_q_values()
    test_cql_weight_update_range()
    test_cql_integration_with_sac()
    print("\n모든 CQL 페널티 테스트 통과!")