#!/usr/bin/env python3
# tests/test_new_structure.py

"""
새로운 디렉토리 구조 테스트
"""

import sys
import traceback


def test_imports():
    """각 모듈의 import 테스트"""

    results = []

    # 1. Models
    print("Testing models...")
    try:
        from src.models.actor import DirichletActor

        results.append(("models.actor", "✓"))
    except Exception as e:
        results.append(("models.actor", f"✗ {str(e)[:50]}"))

    try:
        from src.models.critic import QuantileNetwork

        results.append(("models.critic", "✓"))
    except Exception as e:
        results.append(("models.critic", f"✗ {str(e)[:50]}"))

    try:
        from src.models.iql import IQLAgent

        results.append(("models.iql", "✓"))
    except Exception as e:
        results.append(("models.iql", f"✗ {str(e)[:50]}"))

    # 2. Trading
    print("Testing trading...")
    try:
        from src.trading.env import PortfolioEnv

        results.append(("trading.env", "✓"))
    except Exception as e:
        results.append(("trading.env", f"✗ {str(e)[:50]}"))

    try:
        from src.trading.objectives import DifferentialSharpe

        results.append(("trading.objectives", "✓"))
    except Exception as e:
        results.append(("trading.objectives", f"✗ {str(e)[:50]}"))

    try:
        from src.trading.replay import PrioritizedReplayBuffer

        results.append(("trading.replay", "✓"))
    except Exception as e:
        results.append(("trading.replay", f"✗ {str(e)[:50]}"))

    # 3. Analysis
    print("Testing analysis...")
    try:
        from src.analysis.backtest import BacktestEngine

        results.append(("analysis.backtest", "✓"))
    except Exception as e:
        results.append(("analysis.backtest", f"✗ {str(e)[:50]}"))

    # 4. Utils
    print("Testing utils...")
    try:
        from src.analysis.metrics import calculate_sharpe_ratio

        results.append(("utils.metrics", "✓"))
    except Exception as e:
        results.append(("utils.metrics", f"✗ {str(e)[:50]}"))

    try:
        from src.analysis.visualization import plot_portfolio_weights

        results.append(("utils.visualization", "✓"))
    except Exception as e:
        results.append(("utils.visualization", f"✗ {str(e)[:50]}"))

    try:
        from src.utils.optimizer_utils import polyak_update

        results.append(("utils.optimizer_utils", "✓"))
    except Exception as e:
        results.append(("utils.optimizer_utils", f"✗ {str(e)[:50]}"))

    # 5. Existing agents
    print("Testing existing agents...")
    try:
        from src.agents.t_cell import TCell

        results.append(("agents.t_cell", "✓"))
    except Exception as e:
        results.append(("agents.t_cell", f"✗ {str(e)[:50]}"))

    try:
        from src.agents.b_cell import BCell

        results.append(("agents.b_cell", "✓"))
    except Exception as e:
        results.append(("agents.b_cell", f"✗ {str(e)[:50]}"))

    return results


def print_results(results):
    """결과 출력"""
    print("\n" + "=" * 60)
    print("IMPORT TEST RESULTS")
    print("=" * 60)

    passed = 0
    failed = 0

    for module, status in results:
        print(f"{module:30} {status}")
        if status == "✓":
            passed += 1
        else:
            failed += 1

    print("-" * 60)
    print(f"Total: {len(results)} | Passed: {passed} | Failed: {failed}")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    print("Testing new directory structure...")
    results = test_imports()
    success = print_results(results)

    if success:
        print("\n✅ All imports successful! New structure is working.")
        sys.exit(0)
    else:
        print("\n❌ Some imports failed. Need to fix import paths.")
        sys.exit(1)
