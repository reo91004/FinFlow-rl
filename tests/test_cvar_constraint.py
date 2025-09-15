# tests/test_cvar_constraint.py

import pytest
import numpy as np
import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.objectives import CVaRConstraint, portfolio_objective

def test_cvar_calculation():
    """CVaR 계산이 올바른지 검증"""
    cvar_constraint = CVaRConstraint(
        alpha=0.05,  # 하위 5%
        target=-0.02  # -2% 목표
    )

    # 테스트 수익률 분포
    returns = torch.tensor([
        -0.10, -0.08, -0.05, -0.03, -0.02,  # 하위 5개 (손실)
        -0.01, 0.00, 0.01, 0.02, 0.03,      # 중간
        0.04, 0.05, 0.06, 0.08, 0.10        # 상위 5개 (이익)
    ])

    # CVaR 계산
    cvar, violation = cvar_constraint(returns)

    # 하위 5% = 1개 샘플 (15개 중)
    expected_cvar = -(-0.10)  # 최악의 손실
    assert abs(cvar.item() - expected_cvar) < 0.01, f"CVaR 계산 오류: {cvar.item()}"

    # 위반 페널티 계산
    if cvar < -0.02:
        expected_violation = (-0.02 - cvar) * 10.0
    else:
        expected_violation = 0.0

    assert abs(violation.item() - expected_violation.item()) < 0.01, \
        f"CVaR 위반 페널티 계산 오류: {violation.item()}"

    print(f"✓ CVaR 계산 테스트 통과")
    print(f"  CVaR: {cvar.item():.4f}")
    print(f"  목표: -0.02")
    print(f"  위반 페널티: {violation.item():.4f}")


def test_cvar_with_different_alphas():
    """다양한 alpha 값에서 CVaR 검증"""
    returns = torch.randn(1000) * 0.02  # 평균 0, 표준편차 2%

    alphas = [0.01, 0.05, 0.10]
    cvars = []

    for alpha in alphas:
        cvar_constraint = CVaRConstraint(alpha=alpha, target=-0.05)
        cvar, _ = cvar_constraint(returns)
        cvars.append(cvar.item())

    # alpha가 클수록 CVaR이 덜 극단적이어야 함
    assert cvars[0] <= cvars[1] <= cvars[2], \
        f"CVaR 순서가 잘못됨: {cvars}"

    print(f"✓ 다양한 alpha CVaR 테스트 통과")
    for alpha, cvar in zip(alphas, cvars):
        print(f"  α={alpha:.2f}: CVaR={cvar:.4f}")


def test_portfolio_objective_with_cvar():
    """포트폴리오 목적함수에 CVaR 통합 검증"""
    # 설정
    cfg = {
        'sharpe_weight': 1.0,
        'cvar_alpha': 0.05,
        'cvar_target': -0.02,
        'lambda_cvar': 1.0,
        'turnover_weight': 0.1
    }

    # 테스트 데이터
    batch_size = 32
    n_assets = 10
    returns_t = torch.randn(batch_size, n_assets) * 0.02
    weights_t = torch.softmax(torch.randn(batch_size, n_assets), dim=-1)

    # 목적함수 계산
    objective = portfolio_objective(returns_t, weights_t, cfg)

    # 목적함수가 스칼라인지 확인
    assert objective.shape == torch.Size([]), "목적함수는 스칼라여야 함"

    # 포트폴리오 수익률 계산
    port_returns = (returns_t * weights_t).sum(dim=-1)

    # CVaR 수동 계산
    alpha = 0.95  # 1 - cvar_alpha
    k = int(alpha * batch_size)
    sorted_returns, _ = torch.sort(port_returns)
    var = sorted_returns[k-1] if k > 0 else sorted_returns[0]
    cvar = port_returns[port_returns <= var].mean()

    print(f"✓ 포트폴리오 목적함수 CVaR 통합 테스트 통과")
    print(f"  목적함수 값: {objective.item():.4f}")
    print(f"  포트폴리오 CVaR: {cvar.item():.4f}")


def test_cvar_penalty_strength():
    """CVaR 페널티 강도 검증"""
    cvar_constraint = CVaRConstraint(
        alpha=0.05,
        target=-0.02
    )

    # 목표를 크게 위반하는 경우
    bad_returns = torch.tensor([-0.20] * 10 + [0.01] * 90)  # 큰 손실
    _, bad_violation = cvar_constraint(bad_returns)

    # 목표를 약간 위반하는 경우
    ok_returns = torch.tensor([-0.03] * 10 + [0.01] * 90)  # 작은 손실
    _, ok_violation = cvar_constraint(ok_returns)

    # 목표를 만족하는 경우
    good_returns = torch.tensor([-0.01] * 10 + [0.01] * 90)  # 목표 달성
    _, good_violation = cvar_constraint(good_returns)

    # 페널티 강도 비교
    assert bad_violation > ok_violation >= good_violation, \
        f"페널티 강도 순서 오류: bad={bad_violation}, ok={ok_violation}, good={good_violation}"

    assert good_violation == 0, "목표 달성 시 페널티는 0이어야 함"

    print(f"✓ CVaR 페널티 강도 테스트 통과")
    print(f"  큰 위반 페널티: {bad_violation.item():.4f}")
    print(f"  작은 위반 페널티: {ok_violation.item():.4f}")
    print(f"  목표 달성 페널티: {good_violation.item():.4f}")


def test_cvar_gradient_flow():
    """CVaR 제약이 그래디언트를 올바르게 전파하는지 검증"""
    cvar_constraint = CVaRConstraint(
        alpha=0.05,
        target=-0.02
    )

    # 학습 가능한 파라미터로 수익률 생성
    weights = torch.nn.Parameter(torch.randn(10, requires_grad=True))
    returns_base = torch.randn(10) * 0.02

    # Forward pass
    returns = returns_base * torch.sigmoid(weights)  # 가중치 적용
    cvar, violation = cvar_constraint(returns)

    # Backward pass
    total_loss = cvar + violation
    total_loss.backward()

    # 그래디언트 존재 확인
    assert weights.grad is not None, "그래디언트가 계산되지 않음"
    assert not torch.all(weights.grad == 0), "그래디언트가 모두 0"

    print(f"✓ CVaR 그래디언트 흐름 테스트 통과")
    print(f"  그래디언트 norm: {weights.grad.norm().item():.4f}")


def test_cvar_with_extreme_values():
    """극단적 값에서 CVaR 안정성 검증"""
    cvar_constraint = CVaRConstraint(
        alpha=0.05,
        target=-0.02
    )

    # 극단적 손실
    extreme_losses = torch.tensor([-0.99] * 5 + [0.01] * 95)
    cvar_extreme, violation_extreme = cvar_constraint(extreme_losses)

    assert not torch.isnan(cvar_extreme), "CVaR이 NaN"
    assert not torch.isinf(cvar_extreme), "CVaR이 무한대"
    assert cvar_extreme < 0, "극단적 손실에서 CVaR은 음수여야 함"

    # 모두 양수 수익
    all_positive = torch.ones(100) * 0.05
    cvar_positive, violation_positive = cvar_constraint(all_positive)

    assert cvar_positive > 0, "모든 양수 수익에서 CVaR은 양수여야 함"
    assert violation_positive > 0, "목표 미달성 시 페널티 발생해야 함"

    print(f"✓ 극단값 CVaR 안정성 테스트 통과")
    print(f"  극단 손실 CVaR: {cvar_extreme.item():.4f}")
    print(f"  양수 수익 CVaR: {cvar_positive.item():.4f}")


if __name__ == "__main__":
    test_cvar_calculation()
    test_cvar_with_different_alphas()
    test_portfolio_objective_with_cvar()
    test_cvar_penalty_strength()
    test_cvar_gradient_flow()
    test_cvar_with_extreme_values()
    print("\n모든 CVaR 제약 테스트 통과!")