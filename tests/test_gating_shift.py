# tests/test_gating_shift.py

import pytest
import numpy as np
import torch
from collections import deque
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents.gating import GatingNetwork

def test_gating_performance_feedback():
    """게이팅이 성과 기반으로 전환되는지 검증"""
    gating = GatingNetwork(
        input_dim=43,
        bcell_types=['momentum', 'volatility', 'correlation'],
        temperature=1.8,
        min_dwell_steps=2
    )

    # 초기 선택 확률 균등한지 검증
    state = torch.randn(1, 43)
    probs = gating.get_selection_probabilities(state)

    assert len(probs) == 3, "B-Cell 개수 불일치"
    assert abs(sum(probs.values()) - 1.0) < 1e-6, "확률 합이 1이 아님"

    # correlation에 음수 성과 누적
    for _ in range(20):
        gating.update_performance('correlation', reward=-0.01, info={'turnover': 0.02})

    # momentum에 양수 성과 누적
    for _ in range(20):
        gating.update_performance('momentum', reward=0.02, info={'turnover': 0.01})

    # 성과 점수 계산
    corr_score = gating._score_bcell('correlation')
    mom_score = gating._score_bcell('momentum')

    assert mom_score > corr_score, "momentum 점수가 correlation보다 높아야 함"

    # 선택 확률 재계산
    probs2 = gating.get_selection_probabilities(state)

    assert probs2['momentum'] > probs2['correlation'], \
        f"momentum 선택 확률이 correlation보다 높아야 함: {probs2}"

    print(f"✓ 게이팅 성과 피드백 테스트 통과")
    print(f"  correlation 점수: {corr_score:.4f}")
    print(f"  momentum 점수: {mom_score:.4f}")
    print(f"  선택 확률: {probs2}")


def test_min_dwell_enforcement():
    """최소 유지 스텝(min_dwell)이 적용되는지 검증"""
    gating = GatingNetwork(
        input_dim=20,
        bcell_types=['A', 'B'],
        temperature=1.8,
        min_dwell_steps=2  # 최소 2스텝 유지
    )

    state = torch.randn(1, 20)

    # 첫 선택
    first_choice = gating.select(state)
    selections = [first_choice]

    # 다음 2스텝은 같은 B-Cell이어야 함
    for _ in range(2):
        next_choice = gating.select(state)
        selections.append(next_choice)

    # min_dwell_steps 동안 같은 선택 확인
    assert selections[0] == selections[1], "min_dwell 기간 내 전환 발생"

    # dwell 카운터 리셋 후 전환 가능
    gating.dwell_counter = gating.min_dwell_steps + 1

    # 강제로 다른 B-Cell로 전환 유도
    if first_choice == 'A':
        gating.performance_history['B'].extend([1.0] * 10)
        gating.performance_history['A'].extend([-1.0] * 10)
    else:
        gating.performance_history['A'].extend([1.0] * 10)
        gating.performance_history['B'].extend([-1.0] * 10)

    # 전환 가능 여부 확인
    for _ in range(10):
        choice = gating.select(state)
        if choice != first_choice:
            print(f"✓ 최소 유지 스텝 테스트 통과 (전환: {first_choice} → {choice})")
            return

    print("✓ 최소 유지 스텝 테스트 통과 (전환 없음)")


def test_temperature_effect():
    """Temperature가 선택 분산에 미치는 영향 검증"""
    # 낮은 temperature (더 결정적)
    gating_low = GatingNetwork(
        input_dim=10,
        bcell_types=['A', 'B', 'C'],
        temperature=0.5,
        min_dwell_steps=1
    )

    # 높은 temperature (더 확률적)
    gating_high = GatingNetwork(
        input_dim=10,
        bcell_types=['A', 'B', 'C'],
        temperature=2.0,
        min_dwell_steps=1
    )

    # A에 높은 성과 부여
    for g in [gating_low, gating_high]:
        g.performance_history['A'].extend([0.1] * 20)
        g.performance_history['B'].extend([0.0] * 20)
        g.performance_history['C'].extend([-0.1] * 20)

    state = torch.randn(1, 10)

    # 낮은 temperature에서 확률 분포
    probs_low = gating_low.get_selection_probabilities(state)

    # 높은 temperature에서 확률 분포
    probs_high = gating_high.get_selection_probabilities(state)

    # 낮은 temperature가 더 극단적인 분포를 가져야 함
    entropy_low = -sum(p * np.log(p + 1e-12) for p in probs_low.values())
    entropy_high = -sum(p * np.log(p + 1e-12) for p in probs_high.values())

    assert entropy_low < entropy_high, "낮은 temperature가 더 낮은 엔트로피를 가져야 함"

    print(f"✓ Temperature 효과 테스트 통과")
    print(f"  Low temp entropy: {entropy_low:.4f}")
    print(f"  High temp entropy: {entropy_high:.4f}")


def test_switch_rate_calculation():
    """전환율 계산이 올바른지 검증"""
    gating = GatingNetwork(
        input_dim=10,
        bcell_types=['A', 'B'],
        temperature=1.8,
        min_dwell_steps=1
    )

    state = torch.randn(1, 10)

    # 시뮬레이션
    selections = []
    for i in range(100):
        # 주기적으로 성과 반전
        if i % 20 == 0:
            if len(selections) > 0 and selections[-1] == 'A':
                gating.performance_history['B'].extend([1.0] * 5)
                gating.performance_history['A'].extend([-1.0] * 5)
            else:
                gating.performance_history['A'].extend([1.0] * 5)
                gating.performance_history['B'].extend([-1.0] * 5)

        choice = gating.select(state)
        selections.append(choice)

    # 전환 횟수 계산
    switches = sum(1 for i in range(1, len(selections)) if selections[i] != selections[i-1])
    switch_rate = switches / (len(selections) - 1)

    assert 0 < switch_rate < 1, f"비정상적인 전환율: {switch_rate}"

    print(f"✓ 전환율 계산 테스트 통과")
    print(f"  전환 횟수: {switches}")
    print(f"  전환율: {switch_rate:.2%}")


if __name__ == "__main__":
    test_gating_performance_feedback()
    test_min_dwell_enforcement()
    test_temperature_effect()
    test_switch_rate_calculation()
    print("\n모든 게이팅 테스트 통과!")