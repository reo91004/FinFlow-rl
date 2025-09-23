#!/usr/bin/env python3
# tests/test_integration.py

"""
FinFlow-RL 통합 테스트
시스템이 실제로 작동하는지 검증
"""

import sys
import traceback
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))


def test_imports():
    """모든 핵심 모듈 import 테스트"""
    print("1. Import 테스트...")
    try:
        # Core modules
        from src.environments.portfolio_env import PortfolioEnv
        from src.environments.reward_functions import DifferentialSharpe, CVaRConstraint
        from src.data.replay_buffer import PrioritizedReplayBuffer
        from src.data.offline_dataset import OfflineDataset

        # Agents
        from src.algorithms.online.t_cell import TCell
        from src.algorithms.online.b_cell import BCell
        from src.algorithms.online.memory import MemoryCell
        from src.algorithms.offline.iql import IQLAgent

        # Data
        from src.data.market_loader import DataLoader
        from src.data.feature_extractor import FeatureExtractor

        # Utils
        from src.utils.logger import FinFlowLogger

        print("✓ 모든 모듈 import 성공")
        return True
    except ImportError as e:
        print(f"✗ Import 실패: {e}")
        traceback.print_exc()
        return False


def test_environment():
    """환경 생성 및 기본 동작 테스트"""
    print("\n2. 환경 테스트...")
    try:
        import numpy as np
        import pandas as pd
        from src.environments.portfolio_env import PortfolioEnv

        # Create dummy price data
        prices = pd.DataFrame(
            np.random.randn(100, 5).cumsum(axis=0) + 100,
            columns=["Asset1", "Asset2", "Asset3", "Asset4", "Asset5"],
        )

        # Create environment
        env = PortfolioEnv(price_data=prices, initial_capital=1000000, transaction_cost=0.001)

        # Reset
        state, info = env.reset()
        print(f"  - 상태 차원: {state.shape}")

        # Step
        action = np.ones(5) / 5  # Equal weights
        next_state, reward, terminated, truncated, info = env.step(action)
        print(f"  - 보상: {reward:.4f}")
        print(f"  - 포트폴리오 가치: ${info['portfolio_value']:,.2f}")

        print("✓ 환경 테스트 성공")
        return True
    except Exception as e:
        print(f"✗ 환경 테스트 실패: {e}")
        traceback.print_exc()
        return False


def test_agents():
    """에이전트 생성 테스트"""
    print("\n3. 에이전트 테스트...")
    try:
        import torch
        import numpy as np
        from src.algorithms.online.t_cell import TCell
        from src.algorithms.online.b_cell import BCell
        from src.algorithms.online.memory import MemoryCell
        from src.algorithms.offline.iql import IQLAgent

        device = torch.device("cpu")
        state_dim = 43
        action_dim = 5

        # T-Cell
        t_cell = TCell(feature_dim=12)
        print("  - T-Cell 생성 완료")

        # B-Cell
        b_cell = BCell(
            state_dim=state_dim,
            action_dim=action_dim,
            config={"gamma": 0.99},
            device=device,
        )
        print("  - B-Cell 생성 완료")

        # Memory Cell
        memory_cell = MemoryCell(capacity=100)
        print("  - Memory Cell 생성 완료")

        # Gating Network - 아직 구현되지 않음
        # gating = GatingNetwork(state_dim=state_dim, num_experts=5)
        # print("  - Gating Network 생성 완료")

        # IQL Agent
        iql = IQLAgent(state_dim=state_dim, action_dim=action_dim, device=device)
        print("  - IQL Agent 생성 완료")

        # Test forward pass
        dummy_state = torch.randn(1, state_dim)
        dummy_crisis = 0.3
        dummy_guidance = {"has_guidance": False}

        # Gating decision - 아직 구현되지 않음
        # decision = gating(dummy_state, dummy_guidance, dummy_crisis)
        # print(f"  - 선택된 B-Cell: {decision.selected_bcell}")

        # B-Cell action
        action = b_cell.select_action(dummy_state, deterministic=True)
        print(f"  - 액션 차원: {action.shape}")

        print("✓ 에이전트 테스트 성공")
        return True
    except Exception as e:
        print(f"✗ 에이전트 테스트 실패: {e}")
        traceback.print_exc()
        return False


def test_mini_training():
    """미니 학습 루프 테스트"""
    print("\n4. 미니 학습 테스트...")
    try:
        import torch
        import numpy as np
        import pandas as pd
        from src.environments.portfolio_env import PortfolioEnv
        from src.algorithms.offline.iql import IQLAgent

        # Setup
        device = torch.device("cpu")
        prices = pd.DataFrame(
            np.random.randn(100, 5).cumsum(axis=0) + 100, columns=[f"Asset{i+1}" for i in range(5)]
        )

        env = PortfolioEnv(price_data=prices)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        agent = IQLAgent(state_dim=state_dim, action_dim=action_dim, device=device)

        # Collect some data
        experiences = []
        state, _ = env.reset()

        for _ in range(10):
            action = np.random.dirichlet(np.ones(action_dim))
            next_state, reward, terminated, truncated, _ = env.step(action)

            experiences.append(
                {
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "next_state": next_state,
                    "done": terminated or truncated,
                }
            )

            state = next_state
            if terminated or truncated:
                state, _ = env.reset()

        # Batch update
        if experiences:
            states = torch.FloatTensor([e["state"] for e in experiences])
            actions = torch.FloatTensor([e["action"] for e in experiences])
            rewards = torch.FloatTensor([e["reward"] for e in experiences]).unsqueeze(1)
            next_states = torch.FloatTensor([e["next_state"] for e in experiences])
            dones = torch.FloatTensor([e["done"] for e in experiences]).unsqueeze(1)

            losses = agent.update(states, actions, rewards, next_states, dones)
            print(f"  - Value Loss: {losses['value_loss']:.4f}")
            print(f"  - Q Loss: {losses['q_loss']:.4f}")
            print(f"  - Actor Loss: {losses['actor_loss']:.4f}")

        print("✓ 미니 학습 테스트 성공")
        return True
    except Exception as e:
        print(f"✗ 미니 학습 테스트 실패: {e}")
        traceback.print_exc()
        return False


def main():
    """메인 테스트 실행"""
    print("=" * 60)
    print("FinFlow-RL 통합 테스트")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Environment", test_environment()))
    results.append(("Agents", test_agents()))
    results.append(("Training", test_mini_training()))

    # Summary
    print("\n" + "=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s}: {status}")

    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)

    print("-" * 60)
    print(f"총 {total_tests}개 중 {total_passed}개 통과")

    if total_passed == total_tests:
        print("\n🎉 모든 테스트 통과! 시스템이 정상 작동합니다.")
        return 0
    else:
        print("\n⚠️ 일부 테스트 실패. 위 오류를 확인하세요.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
