# tests/debug_environment.py - 환경 디버깅 스크립트

import os
import sys
import warnings

warnings.filterwarnings("ignore")

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from config import *
from data import DataLoader
from core.environment import PortfolioEnvironment
from data.features import FeatureExtractor


def debug_environment():
    """환경의 실제 동작을 디버깅"""

    print("=" * 80)
    print("환경 디버깅 시작")
    print("=" * 80)

    # 1. 데이터 로드 (소규모 테스트용)
    print("[1] 테스트 데이터 생성...")
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    symbols = SYMBOLS[:5]  # 5개 종목만 사용

    # 더 현실적인 가격 데이터 생성
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, (100, 5))  # 일일 평균 0.05%, 변동성 2%
    prices = pd.DataFrame(
        np.cumprod(1 + returns, axis=0) * 100, index=dates, columns=symbols
    )

    print(f"가격 데이터 생성 완료: {prices.shape}")
    print(f"가격 범위: {prices.min().min():.2f} ~ {prices.max().max():.2f}")

    # 2. 환경 초기화
    print("\n[2] 환경 초기화...")
    feature_extractor = FeatureExtractor(lookback_window=LOOKBACK_WINDOW)
    env = PortfolioEnvironment(
        price_data=prices,
        feature_extractor=feature_extractor,
        initial_capital=100000,  # 작은 금액으로 테스트
        transaction_cost=0.001,
    )

    print(f"환경 생성 완료: {env.n_assets}개 자산, 최대 {env.max_steps} 스텝")

    # 3. 환경 reset 테스트
    print("\n[3] 환경 리셋 테스트...")
    initial_state = env.reset()
    print(f"초기 상태 형태: {initial_state.shape}")
    print(f"초기 상태 범위: [{initial_state.min():.3f}, {initial_state.max():.3f}]")
    print(f"상태 구성:")
    print(f"  - 시장 특성: {initial_state[:12]}")
    print(f"  - 위기 수준: {initial_state[12]:.4f}")
    print(f"  - 이전 가중치: {initial_state[13:]}")
    print(f"  - 가중치 합: {initial_state[13:].sum():.6f}")

    # 4. 10 스텝 실행하여 보상 추적
    print("\n[4] 10 스텝 실행 테스트...")
    rewards = []
    portfolio_values = []
    weight_changes = []

    state = initial_state
    for step in range(10):
        # 랜덤 가중치 생성 (테스트용)
        random_weights = np.random.dirichlet(np.ones(env.n_assets))

        # 스텝 실행
        next_state, reward, done, info = env.step(random_weights)

        rewards.append(reward)
        portfolio_values.append(info["portfolio_value"])
        weight_changes.append(info["weight_change"])

        print(
            f"  스텝 {step+1:2d}: 보상={reward:+7.4f}, 가치={info['portfolio_value']:8.0f}, "
            f"수익률={info['portfolio_return']:+6.2%}, 비용={info['transaction_cost']:6.0f}"
        )

        if done:
            print(f"  에피소드 조기 종료 (스텝 {step+1})")
            break

        state = next_state

    # 5. 보상 분포 분석
    print(f"\n[5] 보상 분포 분석...")
    print(f"  평균 보상: {np.mean(rewards):+7.4f}")
    print(f"  보상 표준편차: {np.std(rewards):7.4f}")
    print(f"  보상 범위: [{min(rewards):+6.3f}, {max(rewards):+6.3f}]")
    print(f"  제로 보상 비율: {(np.array(rewards) == 0).mean():.1%}")

    # 6. 포트폴리오 성과 분석
    print(f"\n[6] 포트폴리오 성과...")
    metrics = env.get_portfolio_metrics()
    print(f"  최종 가치: {metrics.get('final_value', 0):,.0f}")
    print(f"  총 수익률: {metrics.get('total_return', 0):+6.2%}")
    print(f"  변동성: {metrics.get('volatility', 0):6.2%}")
    print(f"  샤프 비율: {metrics.get('sharpe_ratio', 0):+6.3f}")
    print(f"  최대 낙폭: {metrics.get('max_drawdown', 0):6.2%}")

    # 7. 가중치 검증 통계
    print(f"\n[7] 가중치 검증...")
    print(env.get_validation_summary())

    # 8. 특수 케이스 테스트
    print(f"\n[8] 특수 케이스 테스트...")

    # 8-1. 극단적 가중치
    extreme_weights = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    _, reward_extreme, _, info_extreme = env.step(extreme_weights)
    print(f"  극단적 가중치 보상: {reward_extreme:+7.4f}")

    # 8-2. 음수 가중치 (검증 테스트)
    negative_weights = np.array([1.5, -0.3, 0.2, 0.3, 0.3])
    _, reward_negative, _, info_negative = env.step(negative_weights)
    print(f"  음수 가중치 보상: {reward_negative:+7.4f}")

    # 8-3. NaN 가중치
    try:
        nan_weights = np.array([np.nan, 0.25, 0.25, 0.25, 0.25])
        _, reward_nan, _, info_nan = env.step(nan_weights)
        print(f"  NaN 가중치 보상: {reward_nan:+7.4f}")
    except Exception as e:
        print(f"  NaN 가중치 에러: {e}")

    print("\n" + "=" * 80)
    print("환경 디버깅 완료")
    print("=" * 80)

    return {
        "rewards": rewards,
        "portfolio_values": portfolio_values,
        "metrics": metrics,
        "initial_state": initial_state,
    }


def analyze_reward_issues(debug_results):
    """보상 함수의 문제점 구체적 분석"""

    print("\n" + "=" * 60)
    print("보상 함수 문제점 분석")
    print("=" * 60)

    rewards = debug_results["rewards"]

    # 1. 보상 희소성 (Sparsity) 검사
    zero_rewards = (np.array(rewards) == 0).sum()
    small_rewards = (np.abs(np.array(rewards)) < 0.001).sum()

    print(f"1. 보상 희소성:")
    print(
        f"   제로 보상: {zero_rewards}/{len(rewards)} ({zero_rewards/len(rewards):.1%})"
    )
    print(
        f"   미세 보상: {small_rewards}/{len(rewards)} ({small_rewards/len(rewards):.1%})"
    )

    if zero_rewards > len(rewards) * 0.8:
        print("   ⚠️  보상이 너무 희소합니다! 학습 신호가 부족할 수 있습니다.")

    # 2. 보상 방향성 검사
    positive_rewards = (np.array(rewards) > 0).sum()
    negative_rewards = (np.array(rewards) < 0).sum()

    print(f"\n2. 보상 방향성:")
    print(
        f"   양수 보상: {positive_rewards}/{len(rewards)} ({positive_rewards/len(rewards):.1%})"
    )
    print(
        f"   음수 보상: {negative_rewards}/{len(rewards)} ({negative_rewards/len(rewards):.1%})"
    )

    if abs(positive_rewards - negative_rewards) < len(rewards) * 0.2:
        print("   ✅ 보상이 균형잡혀 있습니다.")
    else:
        print("   ⚠️  보상이 한쪽으로 편향되어 있습니다!")

    # 3. 보상 크기 검사
    reward_magnitude = np.mean(np.abs(rewards))
    reward_std = np.std(rewards)

    print(f"\n3. 보상 크기:")
    print(f"   평균 절대값: {reward_magnitude:.4f}")
    print(f"   표준편차: {reward_std:.4f}")
    print(f"   신호대잡음비: {reward_magnitude/max(reward_std, 1e-8):.2f}")

    if reward_magnitude < 0.01:
        print("   ⚠️  보상 크기가 너무 작습니다! 학습이 어려울 수 있습니다.")
    elif reward_magnitude > 1.0:
        print("   ⚠️  보상 크기가 너무 큽니다! 학습이 불안정할 수 있습니다.")
    else:
        print("   ✅ 보상 크기가 적절합니다.")


if __name__ == "__main__":
    try:
        debug_results = debug_environment()
        analyze_reward_issues(debug_results)

        print(f"\n🎯 핵심 발견사항:")
        print(f"   1. 환경이 정상 작동하는지 확인되었습니다")
        print(f"   2. 보상 함수의 구체적 문제점이 파악되었습니다")
        print(f"   3. 다음 단계에서 에이전트 파라미터를 점검해야 합니다")

    except Exception as e:
        print(f"\n❌ 디버깅 실행 중 오류: {e}")
        import traceback

        traceback.print_exc()
