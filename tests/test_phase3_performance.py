# tests/test_phase3_performance.py

"""
Phase 3 성능 검증 테스트
레짐 적응형 엔트로피 스케줄링의 실제 성능 개선 효과를 A/B 테스트로 검증한다.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from datetime import datetime
import json

from config import *
from tests.ab_testing_framework import ABTestingFramework
from core.system import ImmunePortfolioSystem
from core.environment import PortfolioEnvironment
from data.features import FeatureExtractor
from agents.utils.entropy_schedule import RegimeAdaptiveEntropyScheduler


def test_phase3_performance():
    """Phase 3 성능 개선 효과를 A/B 테스트로 검증"""

    print("=" * 80)
    print("Phase 3 성능 검증 A/B 테스트")
    print("=" * 80)

    # 테스트 환경 설정
    n_assets = 10  # 빠른 테스트를 위해 10개 자산만 사용
    episodes_per_test = 20  # 각 테스트당 20 에피소드

    print(f"테스트 설정:")
    print(f"  - 자산 수: {n_assets}")
    print(f"  - 에피소드 수: {episodes_per_test}")
    print(f"  - 테스트 시나리오: 2개 (베이스라인 vs Phase 3)")

    # A/B 테스팅 프레임워크 초기화
    ab_framework = ABTestingFramework()

    # 더미 시장 데이터 생성 (다양한 레짐 포함)
    dates = pd.date_range("2020-01-01", periods=200, freq="D")
    symbols = SYMBOLS[:n_assets]

    # 위기 구간 (50-80일), 회복 구간 (80-120일), 평시 구간 (나머지)
    np.random.seed(GLOBAL_SEED)
    returns = np.random.randn(200, n_assets) * 0.01

    # 위기 구간에 높은 변동성과 상관관계 추가
    returns[50:80] += np.random.randn(30, n_assets) * 0.03  # 3배 변동성
    crisis_common = np.random.randn(30) * 0.02
    for i in range(n_assets):
        returns[50:80, i] += crisis_common * 0.7  # 높은 상관관계

    # 회복 구간에 점진적 안정화
    recovery_decay = np.linspace(2.0, 1.0, 40)
    for i in range(40):
        returns[80 + i] *= recovery_decay[i]

    # 가격 데이터로 변환
    prices = pd.DataFrame(np.exp(returns.cumsum()), index=dates, columns=symbols)

    print(
        f"  - 시장 데이터 기간: {dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')}"
    )
    print(
        f"  - 위기 구간: {dates[50].strftime('%Y-%m-%d')} ~ {dates[79].strftime('%Y-%m-%d')}"
    )
    print(
        f"  - 회복 구간: {dates[80].strftime('%Y-%m-%d')} ~ {dates[119].strftime('%Y-%m-%d')}"
    )

    # 실험 구성 정의
    baseline_config = {
        "name": "baseline",
        "description": "Standard BIPD without Phase 3 improvements",
        "use_adaptive_entropy": False,
        "entropy_scheduler": None,
    }

    phase3_config = {
        "name": "phase3_adaptive",
        "description": "BIPD with Phase 3 regime-adaptive entropy scheduling",
        "use_adaptive_entropy": True,
        "entropy_scheduler": RegimeAdaptiveEntropyScheduler(
            peace_alpha=8.0,
            crisis_alpha=2.0,
            recovery_alpha=4.0,
            crisis_threshold=0.4,
            smoothing_factor=0.3,
        ),
    }

    configs = [baseline_config, phase3_config]

    print(f"\n[1/3] 실험 구성:")
    for config in configs:
        print(f"  - {config['name']}: {config['description']}")

    # 각 구성에 대해 성능 테스트 실행
    all_results = {}

    for config in configs:
        print(f"\n[2/3] {config['name']} 구성 테스트 중...")

        config_results = []

        for episode in range(episodes_per_test):
            print(f"  에피소드 {episode + 1}/{episodes_per_test}", end="\r")

            # 환경 초기화
            feature_extractor = FeatureExtractor(lookback_window=LOOKBACK_WINDOW)
            env = PortfolioEnvironment(
                price_data=prices,
                feature_extractor=feature_extractor,
                initial_capital=INITIAL_CAPITAL,
                transaction_cost=TRANSACTION_COST,
            )

            # 시스템 초기화
            state_dim = FEATURE_DIM + 1 + n_assets
            immune_system = ImmunePortfolioSystem(
                n_assets=n_assets, state_dim=state_dim
            )

            # Phase 3 설정 적용
            if config["use_adaptive_entropy"]:
                for bcell_name, bcell in immune_system.bcells.items():
                    bcell.use_adaptive_entropy = True
                    bcell.entropy_scheduler = config["entropy_scheduler"]

            # 에피소드 실행
            state = env.reset()
            episode_return = 0
            episode_volatility = []
            episode_regimes = []
            steps = 0

            while steps < 150:  # 최대 150 스텝
                # 의사결정
                weights, decision_info = immune_system.decide(state, training=True)

                # 환경 스텝
                next_state, reward, done, info = env.step(weights)

                # 시스템 업데이트
                immune_system.update(state, weights, reward, next_state, done)

                # 메트릭 수집
                episode_return += reward
                episode_volatility.append(abs(reward))

                # 레짐 정보 수집 (Phase 3만)
                if config["use_adaptive_entropy"] and "crisis_level" in decision_info:
                    crisis_level = decision_info["crisis_level"]
                    if crisis_level >= 0.4:
                        regime = "crisis"
                    elif crisis_level <= 0.2:
                        regime = "peace"
                    else:
                        regime = "recovery"
                    episode_regimes.append(regime)

                state = next_state
                steps += 1

                if done:
                    break

            # 에피소드 결과 저장
            result = {
                "episode": episode,
                "total_return": episode_return,
                "final_portfolio_value": info["portfolio_value"],
                "volatility": np.mean(episode_volatility) if episode_volatility else 0,
                "sharpe_ratio": episode_return / (np.std(episode_volatility) + 1e-6),
                "steps": steps,
                "regime_distribution": (
                    dict(pd.Series(episode_regimes).value_counts())
                    if episode_regimes
                    else {}
                ),
            }

            config_results.append(result)

        all_results[config["name"]] = config_results
        print(
            f"\n  {config['name']} 완료: 평균 수익률 {np.mean([r['total_return'] for r in config_results]):.4f}"
        )

    print(f"\n[3/3] A/B 테스트 결과 분석...")

    # A/B 테스팅 프레임워크를 사용한 통계 분석
    baseline_returns = [r["total_return"] for r in all_results["baseline"]]
    phase3_returns = [r["total_return"] for r in all_results["phase3_adaptive"]]

    baseline_sharpe = [r["sharpe_ratio"] for r in all_results["baseline"]]
    phase3_sharpe = [r["sharpe_ratio"] for r in all_results["phase3_adaptive"]]

    baseline_volatility = [r["volatility"] for r in all_results["baseline"]]
    phase3_volatility = [r["volatility"] for r in all_results["phase3_adaptive"]]

    # 실험 추가 및 분석
    experiment_id = ab_framework.add_experiment(
        name="Phase3_Adaptive_Entropy",
        description="Regime-adaptive entropy scheduling vs baseline",
        configs=[baseline_config, phase3_config],
    )

    ab_framework.add_results(
        experiment_id,
        "baseline",
        {
            "returns": baseline_returns,
            "sharpe_ratio": baseline_sharpe,
            "volatility": baseline_volatility,
        },
    )

    ab_framework.add_results(
        experiment_id,
        "phase3_adaptive",
        {
            "returns": phase3_returns,
            "sharpe_ratio": phase3_sharpe,
            "volatility": phase3_volatility,
        },
    )

    # 통계적 유의성 검증
    analysis_results = ab_framework.analyze_experiment(experiment_id)

    # 결과 출력
    print("\n" + "=" * 80)
    print("Phase 3 성능 검증 결과")
    print("=" * 80)

    print(f"베이스라인 성능:")
    print(
        f"  평균 수익률: {np.mean(baseline_returns):.6f} ± {np.std(baseline_returns):.6f}"
    )
    print(
        f"  평균 샤프비: {np.mean(baseline_sharpe):.4f} ± {np.std(baseline_sharpe):.4f}"
    )
    print(
        f"  평균 변동성: {np.mean(baseline_volatility):.6f} ± {np.std(baseline_volatility):.6f}"
    )

    print(f"\nPhase 3 성능:")
    print(
        f"  평균 수익률: {np.mean(phase3_returns):.6f} ± {np.std(phase3_returns):.6f}"
    )
    print(f"  평균 샤프비: {np.mean(phase3_sharpe):.4f} ± {np.std(phase3_sharpe):.4f}")
    print(
        f"  평균 변동성: {np.mean(phase3_volatility):.6f} ± {np.std(phase3_volatility):.6f}"
    )

    # 개선 효과
    return_improvement = (
        (np.mean(phase3_returns) - np.mean(baseline_returns))
        / abs(np.mean(baseline_returns))
        * 100
    )
    sharpe_improvement = (
        (np.mean(phase3_sharpe) - np.mean(baseline_sharpe))
        / abs(np.mean(baseline_sharpe))
        * 100
    )
    volatility_reduction = (
        (np.mean(baseline_volatility) - np.mean(phase3_volatility))
        / np.mean(baseline_volatility)
        * 100
    )

    print(f"\n개선 효과:")
    print(f"  수익률 개선: {return_improvement:+.2f}%")
    print(f"  샤프비 개선: {sharpe_improvement:+.2f}%")
    print(f"  변동성 감소: {volatility_reduction:+.2f}%")

    # 통계적 유의성
    print(f"\n통계적 유의성 (p < 0.05):")
    if "returns" in analysis_results["metrics"]:
        returns_pvalue = analysis_results["metrics"]["returns"].get("p_value", 1.0)
        print(
            f"  수익률 차이: p={returns_pvalue:.4f} {'✓' if returns_pvalue < 0.05 else '✗'}"
        )

    if "sharpe_ratio" in analysis_results["metrics"]:
        sharpe_pvalue = analysis_results["metrics"]["sharpe_ratio"].get("p_value", 1.0)
        print(
            f"  샤프비 차이: p={sharpe_pvalue:.4f} {'✓' if sharpe_pvalue < 0.05 else '✗'}"
        )

    # 레짐별 분석 (Phase 3만)
    if all_results["phase3_adaptive"][0]["regime_distribution"]:
        print(f"\nPhase 3 레짐 분포 (평균):")
        all_regimes = {}
        for result in all_results["phase3_adaptive"]:
            for regime, count in result["regime_distribution"].items():
                all_regimes[regime] = all_regimes.get(regime, 0) + count

        total_regime_counts = sum(all_regimes.values())
        for regime, count in all_regimes.items():
            percentage = count / total_regime_counts * 100
            print(f"  {regime}: {percentage:.1f}%")

    # 권장사항
    print(f"\n권장사항:")
    if return_improvement > 5:
        print("  ✅ Phase 3는 수익률을 유의미하게 개선합니다")
    elif return_improvement > 0:
        print("  ⚠️  Phase 3는 수익률을 소폭 개선하지만 추가 최적화가 필요합니다")
    else:
        print("  ❌ Phase 3는 현재 설정에서 수익률을 개선하지 못합니다")

    if volatility_reduction > 5:
        print("  ✅ Phase 3는 리스크를 효과적으로 관리합니다")
    elif volatility_reduction > 0:
        print("  ⚠️  Phase 3는 리스크를 소폭 감소시킵니다")
    else:
        print("  ❌ Phase 3는 리스크 관리에 개선이 필요합니다")

    # 결과를 파일에 저장
    results_summary = {
        "timestamp": datetime.now().isoformat(),
        "test_config": {
            "n_assets": n_assets,
            "episodes_per_test": episodes_per_test,
            "test_period": f"{dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')}",
        },
        "performance": {
            "baseline": {
                "mean_return": float(np.mean(baseline_returns)),
                "std_return": float(np.std(baseline_returns)),
                "mean_sharpe": float(np.mean(baseline_sharpe)),
                "mean_volatility": float(np.mean(baseline_volatility)),
            },
            "phase3": {
                "mean_return": float(np.mean(phase3_returns)),
                "std_return": float(np.std(phase3_returns)),
                "mean_sharpe": float(np.mean(phase3_sharpe)),
                "mean_volatility": float(np.mean(phase3_volatility)),
            },
        },
        "improvements": {
            "return_improvement_pct": float(return_improvement),
            "sharpe_improvement_pct": float(sharpe_improvement),
            "volatility_reduction_pct": float(volatility_reduction),
        },
        "statistical_significance": analysis_results,
    }

    # 결과 파일 저장
    os.makedirs("logs", exist_ok=True)
    results_file = f"logs/phase3_performance_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    print(f"\n결과가 저장되었습니다: {results_file}")
    print("=" * 80)

    return results_summary


if __name__ == "__main__":
    try:
        results = test_phase3_performance()

        # 핵심 발견사항 요약
        return_improvement = results["improvements"]["return_improvement_pct"]
        sharpe_improvement = results["improvements"]["sharpe_improvement_pct"]
        volatility_reduction = results["improvements"]["volatility_reduction_pct"]

        print(f"\n핵심 발견사항:")
        print(f"   1. Phase 3 레짐 적응형 엔트로피 스케줄링 성능 검증 완료")
        print(
            f"   2. 수익률 개선: {return_improvement:+.1f}%, 샤프비 개선: {sharpe_improvement:+.1f}%"
        )
        print(f"   3. 변동성 감소: {volatility_reduction:+.1f}%")

        if return_improvement > 0 and volatility_reduction > 0:
            print(f"   4. ✅ Phase 3 구현이 성공적으로 성능을 개선했습니다")
        else:
            print(f"   4. ⚠️  추가 하이퍼파라미터 조정이 필요할 수 있습니다")

    except Exception as e:
        print(f"\n❌ Phase 3 성능 검증 중 오류: {e}")
        import traceback

        traceback.print_exc()
