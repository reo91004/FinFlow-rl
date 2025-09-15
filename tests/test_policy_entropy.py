# tests/test_policy_entropy.py

import pytest
import numpy as np
import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents.b_cell import BCell

def test_sac_alpha_tuning():
    """SAC 알파 자동 튜닝 검증"""
    config = {
        'alpha_init': 0.79,
        'alpha_min': 5e-4,
        'alpha_max': 2.0,
        'alpha_lr': 3e-4,
        'target_entropy_ratio': 0.5,
        'actor_lr': 3e-4,
        'critic_lr': 3e-4,
        'batch_size': 256,
        'buffer_size': 10000,
        'cql_weight': 5.0,
        'cql_min_q_weight': 5.0
    }

    bcell = BCell(
        specialization='momentum',
        state_dim=43,
        action_dim=10,
        config=config
    )

    # 초기 알파값 검증
    assert bcell.alpha == 0.79, f"초기 알파값 불일치: {bcell.alpha}"
    assert bcell.target_entropy == -0.5 * 10, "목표 엔트로피 계산 오류"

    # 버퍼에 경험 추가
    for _ in range(300):
        state = np.random.randn(43)
        action = np.random.dirichlet(np.ones(10))
        reward = np.random.randn()
        next_state = np.random.randn(43)
        done = False

        bcell.replay_buffer.push(state, action, reward, next_state, done)

    # 여러 번 업데이트하며 알파 변화 추적
    alpha_history = [bcell.alpha]

    for _ in range(10):
        batch = bcell.replay_buffer.sample(256)
        states = torch.FloatTensor(np.array([t.state for t in batch]))

        # 정책 업데이트 (내부에서 알파 튜닝)
        losses = bcell.update(batch)
        alpha_history.append(bcell.alpha)

    # 알파가 변화했는지 검증
    alpha_changes = np.diff(alpha_history)
    assert np.any(np.abs(alpha_changes) > 1e-6), "알파가 전혀 변하지 않음"

    # 알파가 범위 내에 있는지 검증
    assert all(config['alpha_min'] <= a <= config['alpha_max'] for a in alpha_history), \
        f"알파가 범위를 벗어남: {alpha_history}"

    print(f"✓ SAC 알파 튜닝 테스트 통과")
    print(f"  초기 알파: {alpha_history[0]:.4f}")
    print(f"  최종 알파: {alpha_history[-1]:.4f}")
    print(f"  변화량: {alpha_history[-1] - alpha_history[0]:.4f}")


def test_entropy_regularization():
    """엔트로피 정규화 효과 검증"""
    config = {
        'alpha_init': 0.79,
        'actor_lr': 3e-4,
        'critic_lr': 3e-4,
        'batch_size': 128,
        'buffer_size': 5000
    }

    bcell = BCell(
        specialization='volatility',
        state_dim=20,
        action_dim=5,
        config=config
    )

    # 랜덤 상태에서 액션 샘플링
    state = torch.randn(1, 20)

    # 여러 번 샘플링하여 분산 측정
    actions = []
    log_probs = []

    for _ in range(100):
        with torch.no_grad():
            action, log_prob = bcell.actor.sample(state)
            actions.append(action.numpy())
            log_probs.append(log_prob.item())

    actions = np.array(actions).squeeze()

    # 액션 분산 검증 (높은 엔트로피 = 높은 분산)
    action_std = np.std(actions, axis=0)
    assert np.all(action_std > 0.01), "액션 분산이 너무 낮음 (탐색 부족)"

    # 엔트로피 추정
    mean_log_prob = np.mean(log_probs)
    entropy_estimate = -mean_log_prob

    print(f"✓ 엔트로피 정규화 테스트 통과")
    print(f"  평균 액션 표준편차: {np.mean(action_std):.4f}")
    print(f"  추정 엔트로피: {entropy_estimate:.4f}")


def test_alpha_convergence():
    """알파가 목표 엔트로피로 수렴하는지 검증"""
    config = {
        'alpha_init': 1.5,  # 높은 초기값
        'alpha_lr': 1e-3,   # 빠른 학습률
        'target_entropy_ratio': 0.5,
        'actor_lr': 3e-4,
        'critic_lr': 3e-4,
        'batch_size': 64,
        'buffer_size': 5000
    }

    bcell = BCell(
        specialization='defensive',
        state_dim=10,
        action_dim=3,
        config=config
    )

    target_entropy = bcell.target_entropy

    # 목표보다 낮은 엔트로피 시뮬레이션
    low_entropy_log_probs = torch.tensor([target_entropy - 1.0] * 64)

    alpha_history = []
    for _ in range(50):
        # 알파 업데이트
        alpha_loss = -(bcell.log_alpha * (low_entropy_log_probs.detach() + target_entropy)).mean()

        bcell.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        bcell.alpha_optimizer.step()

        bcell.alpha = bcell.log_alpha.exp().clamp(config['alpha_min'], config['alpha_max']).item()
        alpha_history.append(bcell.alpha)

    # 알파가 증가했는지 검증 (낮은 엔트로피 → 알파 증가)
    assert alpha_history[-1] > alpha_history[0], "알파가 증가해야 함"

    print(f"✓ 알파 수렴 테스트 통과")
    print(f"  초기 알파: {alpha_history[0]:.4f}")
    print(f"  최종 알파: {alpha_history[-1]:.4f}")


if __name__ == "__main__":
    test_sac_alpha_tuning()
    test_entropy_regularization()
    test_alpha_convergence()
    print("\n모든 SAC 엔트로피 테스트 통과!")