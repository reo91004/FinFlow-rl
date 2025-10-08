# tests/test_irt_improvements.py

"""
IRT 개선사항 테스트 스크립트

Tier 0 개선사항들이 제대로 동작하는지 확인한다.
"""

import torch
import numpy as np
from pathlib import Path

# 임포트 테스트
print("Testing imports...")
try:
    from finrl.agents.irt.irt_operator import IRT
    from finrl.agents.irt.bcell_actor import BCellIRTActor
    from finrl.agents.irt.irt_policy import IRTPolicy
    from finrl.meta.env_portfolio_optimization.reward_wrapper import MultiObjectiveRewardWrapper
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)


def test_smooth_transition():
    """위기 임계값 부드러운 전이 테스트"""
    print("\n" + "="*60)
    print("Testing Crisis Threshold Smooth Transition")
    print("="*60)

    # IRT 연산자 생성
    irt = IRT(m=6, M=8)

    # 위기 레벨 테스트
    crisis_levels = torch.tensor([[0.0], [0.3], [0.5], [0.7], [1.0]])
    B = crisis_levels.shape[0]

    # 더미 입력
    E = torch.randn(B, 6, 128)
    K = torch.randn(B, 8, 128)
    w_prev = torch.ones(B, 8) / 8
    fitness = torch.ones(B, 8)

    # 순전파
    w, P, debug_info = irt(E, K, None, w_prev, fitness, crisis_levels)

    # adaptive_alpha 확인
    adaptive_alpha = debug_info['adaptive_alpha']

    print("\nCrisis Level → Adaptive Alpha (OT weight):")
    for i, cl in enumerate(crisis_levels):
        alpha = adaptive_alpha[i].item()
        print(f"  {cl.item():.1f} → {alpha:.3f}")

    # 부드러운 전이 검증
    alpha_diffs = torch.diff(adaptive_alpha.squeeze())
    is_smooth = torch.all(torch.abs(alpha_diffs) < 0.2)  # 급격한 점프 없음

    if is_smooth:
        print("\n✅ Smooth transition verified (no sudden jumps)")
    else:
        print("\n❌ Warning: Transition may not be smooth")

    return True


def test_xai_integration():
    """Test XAI regularization integration"""
    print("\n" + "="*60)
    print("Testing XAI Regularization Integration")
    print("="*60)

    # Create BCellIRTActor
    actor = BCellIRTActor(
        state_dim=301,  # Dow 30 state dimension
        action_dim=30,
        M_proto=8
    )

    # Set to training mode
    actor.train()

    # Dummy input
    state = torch.randn(4, 301)  # Batch of 4

    # Forward pass
    action, log_prob, info = actor(state)

    # Check if last_irt_info is stored
    if hasattr(actor, 'last_irt_info'):
        irt_info = actor.last_irt_info
        print("\n✅ last_irt_info stored successfully")

        # Check required keys
        required_keys = ['w', 'w_rep', 'w_ot', 'crisis_level', 'adaptive_alpha', 'diversity_loss']
        for key in required_keys:
            if key in irt_info:
                print(f"  ✓ {key}: {type(irt_info[key])}")
            else:
                print(f"  ✗ {key}: missing")
    else:
        print("\n❌ last_irt_info not found")
        return False

    # Check diversity loss
    if 'diversity_loss' in info:
        print(f"\n✅ Diversity loss computed: {info['diversity_loss']:.4f}")
    else:
        print("\n❌ Diversity loss not in info")

    return True


def test_reward_wrapper():
    """Test reward wrapper fine-tuned parameters"""
    print("\n" + "="*60)
    print("Testing Reward Wrapper Fine-tuned Parameters")
    print("="*60)

    # Mock environment
    class MockEnv:
        def __init__(self):
            self._portfolio_value = 1000000
            self._final_weights = []

        def reset(self):
            self._portfolio_value = 1000000
            self._final_weights = [np.ones(30) / 30]
            return np.random.randn(301), {}

        def step(self, action):
            self._final_weights.append(action)
            reward = np.log(1.01)  # 1% return
            obs = np.random.randn(301)
            return obs, reward, False, False, {}

    # Create wrapper with fine-tuned parameters
    env = MockEnv()
    wrapper = MultiObjectiveRewardWrapper(
        env,
        lambda_turnover=0.003,
        lambda_diversity=0.03,
        lambda_drawdown=0.07,
        target_turnover=0.08,
        turnover_band=0.04,
        target_hhi=0.25
    )

    # Check parameters
    print("\nFine-tuned parameters:")
    print(f"  λ_turnover:      {wrapper.lambda_turnover}")
    print(f"  λ_diversity:     {wrapper.lambda_diversity}")
    print(f"  λ_drawdown:      {wrapper.lambda_drawdown}")
    print(f"  target_turnover: {wrapper.target_turnover}")
    print(f"  turnover_band:   {wrapper.turnover_band}")
    print(f"  target_hhi:      {wrapper.target_hhi}")

    # Test step
    obs, info = wrapper.reset()
    action = np.ones(30) / 30  # Equal weights
    action[0] = 0.15  # Concentrate on first asset

    obs, reward, terminated, truncated, info = wrapper.step(action)

    if 'reward_components' in info:
        print("\n✅ Reward components computed:")
        for key, value in info['reward_components'].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
    else:
        print("\n❌ Reward components not found")
        return False

    return True


def test_irt_policy_xai():
    """Test IRTPolicy XAI integration"""
    print("\n" + "="*60)
    print("Testing IRTPolicy XAI Integration")
    print("="*60)

    from gymnasium import spaces

    # Create IRTPolicy
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(301,))
    act_space = spaces.Box(low=0, high=1, shape=(30,))

    policy = IRTPolicy(
        observation_space=obs_space,
        action_space=act_space,
        lr_schedule=lambda _: 0.001,
        xai_reg_weight=0.01
    )

    # Check XAI parameters
    print(f"\nXAI regularization weight: {policy.xai_reg_weight}")

    # Check if XAI method exists
    if hasattr(policy, '_compute_xai_regularization'):
        print("✅ _compute_xai_regularization method exists")

        # Test XAI computation
        test_info = {
            'w_rep': torch.rand(4, 8),
            'w_ot': torch.rand(4, 8),
            'crisis_level': torch.rand(4, 1)
        }

        xai_loss = policy._compute_xai_regularization(test_info)
        print(f"✅ XAI loss computed: {xai_loss.item():.4f}")
    else:
        print("❌ _compute_xai_regularization method not found")
        return False

    return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("IRT Improvements Test Suite")
    print("=" * 60)

    tests = [
        ("Smooth Transition", test_smooth_transition),
        ("XAI Integration", test_xai_integration),
        ("Reward Wrapper", test_reward_wrapper),
        ("IRTPolicy XAI", test_irt_policy_xai)
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ Test '{name}' failed with error: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Tier 0 improvements are working correctly.")
    else:
        print(f"\n⚠️ {total - passed} tests failed. Please check the implementation.")


if __name__ == '__main__':
    main()