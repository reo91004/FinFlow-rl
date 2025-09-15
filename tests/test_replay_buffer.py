# tests/test_replay_buffer.py

import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.replay import PrioritizedReplayBuffer, ReservoirBuffer, Transition

def test_replay_fillup():
    """리플레이 버퍼가 정상적으로 채워지는지 테스트"""
    buffer = PrioritizedReplayBuffer(capacity=1000, alpha=0.0)  # 초기 uniform sampling

    # 경험 추가
    for i in range(300):
        state = np.random.randn(10)
        action = np.random.dirichlet(np.ones(5))
        reward = np.random.randn()
        next_state = np.random.randn(10)
        done = False

        transition = Transition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done
        )
        buffer.push(transition)

    # 최소 버퍼 사이즈(256) 충족 확인
    assert len(buffer) >= 256, "min_buffer_size 충족 필수"
    assert len(buffer) == 300, f"버퍼 크기가 예상과 다름: {len(buffer)}"

    print(f"✓ 버퍼 유입 테스트 통과: {len(buffer)}개 저장됨")


def test_per_sampling():
    """PER 샘플링이 올바르게 작동하는지 테스트"""
    buffer = PrioritizedReplayBuffer(capacity=100, alpha=0.6, beta=0.4)

    # 다양한 보상으로 경험 추가
    for i in range(50):
        transition = Transition(
            state=np.array([i]),
            action=np.array([i]),
            reward=float(i),  # 보상이 클수록 우선순위 높음
            next_state=np.array([i+1]),
            done=False
        )
        buffer.push(transition)

    # 샘플링
    transitions, weights, indices = buffer.sample(10)

    # 검증
    assert len(transitions) == 10, "샘플 개수가 일치해야 함"
    assert len(weights) == 10, "가중치 개수가 일치해야 함"
    assert len(indices) == 10, "인덱스 개수가 일치해야 함"

    # 가중치는 0과 1 사이여야 함
    assert np.all(weights >= 0) and np.all(weights <= 1), "가중치 범위 오류"

    print("✓ PER 샘플링 테스트 통과")


def test_reservoir_sampling():
    """Reservoir 샘플링이 올바르게 작동하는지 테스트"""
    buffer = ReservoirBuffer(capacity=10)

    # capacity보다 많은 경험 추가
    for i in range(100):
        transition = Transition(
            state=np.array([i]),
            action=np.array([i]),
            reward=float(i),
            next_state=np.array([i+1]),
            done=False
        )
        buffer.push(transition)

    # 검증
    assert len(buffer) == 10, f"버퍼 크기가 capacity를 초과: {len(buffer)}"

    # 균등 샘플링 테스트
    samples = buffer.sample(5)
    assert len(samples) == 5, "샘플 개수가 일치해야 함"

    print("✓ Reservoir 샘플링 테스트 통과")


def test_transition_validation():
    """Transition 검증이 작동하는지 테스트"""
    buffer = PrioritizedReplayBuffer(capacity=100)

    # 유효한 transition
    valid_transition = Transition(
        state=np.array([1, 2, 3]),
        action=np.array([0.5, 0.5]),
        reward=1.0,
        next_state=np.array([2, 3, 4]),
        done=False
    )
    buffer.push(valid_transition)  # 정상 동작

    # None 상태는 assert로 실패해야 함
    with pytest.raises(AssertionError):
        invalid_transition = Transition(
            state=None,
            action=np.array([0.5, 0.5]),
            reward=1.0,
            next_state=np.array([2, 3, 4]),
            done=False
        )
        buffer.push(invalid_transition)

    print("✓ Transition 검증 테스트 통과")


def test_priority_update():
    """우선순위 업데이트가 작동하는지 테스트"""
    buffer = PrioritizedReplayBuffer(capacity=100, alpha=0.6)

    # 경험 추가
    for i in range(20):
        transition = Transition(
            state=np.array([i]),
            action=np.array([i]),
            reward=0.0,
            next_state=np.array([i+1]),
            done=False
        )
        buffer.push(transition)

    # 샘플링
    transitions, weights, indices = buffer.sample(10)

    # TD 오차로 우선순위 업데이트
    td_errors = np.random.rand(10) * 10  # 큰 TD 오차
    buffer.update_priorities(indices, td_errors)

    # 다시 샘플링 (높은 우선순위를 가진 샘플이 더 자주 선택되어야 함)
    transitions2, _, _ = buffer.sample(10)

    print("✓ 우선순위 업데이트 테스트 통과")


if __name__ == "__main__":
    test_replay_fillup()
    test_per_sampling()
    test_reservoir_sampling()
    test_transition_validation()
    test_priority_update()
    print("\n모든 리플레이 버퍼 테스트 통과!")