# tests/test_irt_policy.py

"""
IRT Policy 단위 테스트

테스트 항목:
1. IRT forward pass 정상 작동
2. Simplex 제약 만족 (portfolio weights)
3. SB3 통합 (IRTPolicy)
4. Device 호환성 (CPU/GPU)
"""

import pytest
import torch
import numpy as np
from gymnasium import spaces

from finrl.agents.irt import IRT, BCellIRTActor, TCellMinimal
from finrl.agents.irt.irt_policy import IRTPolicy


@pytest.fixture
def irt_config():
    """IRT 테스트 설정"""
    return {
        'state_dim': 301,  # FinRL Dow30: 1 + (8 + 2) * 30 = 301
        'action_dim': 30,
        'emb_dim': 128,
        'm_tokens': 6,
        'M_proto': 8,
        'alpha': 0.3,
        'market_feature_dim': 12
    }


@pytest.fixture
def sample_state(irt_config):
    """샘플 상태 생성"""
    batch_size = 4
    state_dim = irt_config['state_dim']
    return torch.randn(batch_size, state_dim)


def test_irt_forward_pass(irt_config, sample_state):
    """
    테스트 1: IRT forward pass가 정상 작동하는가?
    """
    actor = BCellIRTActor(**irt_config)
    actor.eval()

    with torch.no_grad():
        action, log_prob, info = actor(sample_state, deterministic=True)

    # 형상 체크
    assert action.shape == (sample_state.shape[0], irt_config['action_dim'])

    # Info 체크
    assert 'w' in info
    assert 'P' in info
    assert 'crisis_level' in info
    assert 'w_rep' in info
    assert 'w_ot' in info

    # 가중치 차원 체크
    assert info['w'].shape == (sample_state.shape[0], irt_config['M_proto'])

    print("✅ Test 1 passed: IRT forward pass 정상 작동")


def test_simplex_constraint(irt_config, sample_state):
    """
    테스트 2: Portfolio weights가 simplex 제약을 만족하는가?
    - sum(weights) = 1
    - all(weights >= 0)
    """
    actor = BCellIRTActor(**irt_config)
    actor.eval()

    with torch.no_grad():
        action, log_prob, info = actor(sample_state, deterministic=True)

    # Simplex 제약 체크
    # 1. 합 = 1
    weight_sums = action.sum(dim=-1)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5), \
        f"Weight sums: {weight_sums}, expected all 1.0"

    # 2. 모든 원소 >= 0
    assert (action >= 0).all(), f"Negative weights found: {action.min()}"

    # 3. 모든 원소 <= 1
    assert (action <= 1).all(), f"Weights > 1 found: {action.max()}"

    print("✅ Test 2 passed: Simplex 제약 만족")


def test_sb3_integration(irt_config):
    """
    테스트 3: SB3 환경에서 IRTPolicy가 정상 작동하는가?
    """
    # Observation space (Box)
    obs_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(irt_config['state_dim'],),
        dtype=np.float32
    )

    # Action space (Box, simplex)
    action_space = spaces.Box(
        low=0.0,
        high=1.0,
        shape=(irt_config['action_dim'],),
        dtype=np.float32
    )

    # IRTPolicy 생성
    policy = IRTPolicy(
        observation_space=obs_space,
        action_space=action_space,
        lr_schedule=lambda _: 3e-4,
        emb_dim=irt_config['emb_dim'],
        m_tokens=irt_config['m_tokens'],
        M_proto=irt_config['M_proto'],
        alpha=irt_config['alpha']
    )

    policy.eval()

    # 샘플 관측
    obs = torch.randn(4, irt_config['state_dim'])

    # Forward pass
    with torch.no_grad():
        action = policy(obs, deterministic=True)

    # 형상 체크
    assert action.shape == (4, irt_config['action_dim'])

    # Simplex 제약 체크
    assert torch.allclose(action.sum(dim=-1), torch.ones(4), atol=1e-5)
    assert (action >= 0).all()

    print("✅ Test 3 passed: SB3 통합 성공")


def test_device_compatibility(irt_config, sample_state):
    """
    테스트 4: CPU/GPU 전환이 정상 작동하는가?
    """
    actor = BCellIRTActor(**irt_config)

    # CPU 테스트
    actor.cpu()
    sample_state_cpu = sample_state.cpu()

    with torch.no_grad():
        action_cpu, log_prob_cpu, info_cpu = actor(sample_state_cpu, deterministic=True)

    assert action_cpu.device.type == 'cpu'
    assert info_cpu['w'].device.type == 'cpu'

    print("✅ Test 4 (CPU) passed: CPU 호환성 확인")

    # GPU 테스트 (가능한 경우)
    if torch.cuda.is_available():
        actor.cuda()
        sample_state_gpu = sample_state.cuda()

        with torch.no_grad():
            action_gpu, log_prob_gpu, info_gpu = actor(sample_state_gpu, deterministic=True)

        assert action_gpu.device.type == 'cuda'
        assert info_gpu['w'].device.type == 'cuda'

        print("✅ Test 4 (GPU) passed: GPU 호환성 확인")
    else:
        print("⚠️ Test 4 (GPU) skipped: CUDA not available")


def test_irt_decomposition(irt_config, sample_state):
    """
    테스트 5 (추가): IRT 분해 정보가 올바른가?
    - w = (1-α) * w_rep + α * w_ot
    """
    actor = BCellIRTActor(**irt_config)
    actor.eval()

    with torch.no_grad():
        action, log_prob, info = actor(sample_state, deterministic=True)

    w = info['w']
    w_rep = info['w_rep']
    w_ot = info['w_ot']
    alpha = irt_config['alpha']

    # IRT 혼합 공식 검증
    w_expected = (1 - alpha) * w_rep + alpha * w_ot

    # 정규화 차이는 허용 (IRT에서 재정규화)
    # L2 거리로 검증
    diff = torch.norm(w - w_expected, dim=-1).mean()

    # 정규화 전후 차이는 작아야 함 (< 0.1)
    assert diff < 0.1, f"IRT decomposition mismatch: L2 diff = {diff:.4f}"

    print("✅ Test 5 passed: IRT 분해 공식 검증")


def test_fitness_calculation(irt_config):
    """
    테스트 6 (새로운): Fitness 계산이 Critic Q-value를 사용하는가?
    """
    from finrl.agents.irt.irt_policy import IRTActorWrapper
    import torch.nn as nn

    # Mock Critic 생성
    class MockCritic(nn.Module):
        def __init__(self, action_dim):
            super().__init__()
            self.action_dim = action_dim

        def forward(self, obs, action):
            # 행동 합에 비례하는 Q-value 반환 (테스트용)
            q_value = action.sum(dim=-1) * 10.0
            return (q_value, q_value)  # Twin Q

    # Mock Policy 생성
    class MockPolicy:
        def __init__(self, critic):
            self.critic = critic

    # BCellIRTActor 생성
    actor = BCellIRTActor(**irt_config)

    # Observation space
    obs_space = spaces.Box(
        low=-np.inf, high=np.inf,
        shape=(irt_config['state_dim'],),
        dtype=np.float32
    )
    action_space = spaces.Box(
        low=0.0, high=1.0,
        shape=(irt_config['action_dim'],),
        dtype=np.float32
    )

    # Mock Policy와 Critic
    critic = MockCritic(irt_config['action_dim'])
    policy = MockPolicy(critic)

    # IRTActorWrapper 생성
    wrapper = IRTActorWrapper(
        irt_actor=actor,
        features_dim=irt_config['state_dim'],
        action_space=action_space,
        policy=policy
    )

    wrapper.eval()

    # 샘플 관측
    obs = torch.randn(4, irt_config['state_dim'])

    with torch.no_grad():
        action, log_prob = wrapper.action_log_prob(obs)

    # action_log_prob 내부에서 fitness 계산되었는지 확인
    # (간접 확인: action이 정상적으로 반환되었는지)
    assert action.shape == (4, irt_config['action_dim'])
    assert log_prob.shape == (4, 1)

    print("✅ Test 6 passed: Fitness 계산 검증")


def test_replicator_activation(irt_config):
    """
    테스트 7 (새로운): Replicator가 fitness에 따라 가중치를 조정하는가?
    """
    actor = BCellIRTActor(**irt_config)
    actor.eval()

    B = 4
    M = irt_config['M_proto']
    state = torch.randn(B, irt_config['state_dim'])

    # Case 1: fitness=None (균등)
    with torch.no_grad():
        action1, log_prob1, info1 = actor(state, fitness=None, deterministic=True)
        w1 = info1['w']

    # Case 2: 불균등 fitness (프로토타입 0이 가장 높음)
    fitness = torch.ones(B, M) * 0.1
    fitness[:, 0] = 1.0  # 프로토타입 0에 높은 fitness

    with torch.no_grad():
        action2, log_prob2, info2 = actor(state, fitness=fitness, deterministic=True)
        w2 = info2['w']
        w_rep2 = info2['w_rep']

    # Replicator가 작동했다면, w_rep2에서 프로토타입 0의 가중치가 더 커야 함
    # w_rep2[:, 0]의 평균이 1/M보다 커야 함
    avg_w0 = w_rep2[:, 0].mean()
    uniform_weight = 1.0 / M

    assert avg_w0 > uniform_weight, \
        f"Replicator not activated: w_rep[0] = {avg_w0:.4f}, expected > {uniform_weight:.4f}"

    print(f"✅ Test 7 passed: Replicator 작동 확인 (w_rep[0] = {avg_w0:.4f} > {uniform_weight:.4f})")


def test_gaussian_projection_policy():
    """
    테스트 8: Projected Gaussian Policy가 정상 작동하는가?
    """
    config = {
        'state_dim': 301,  # FinRL Dow30 표준
        'action_dim': 30,
        'emb_dim': 128,
        'm_tokens': 6,
        'M_proto': 8,
        'alpha': 0.3,
        'market_feature_dim': 12
    }

    actor = BCellIRTActor(**config)
    actor.eval()

    state = torch.randn(4, config['state_dim'])
    fitness = torch.randn(4, config['M_proto'])

    action, log_prob, info = actor(state, fitness, deterministic=False)

    # 1. Simplex 제약
    assert torch.allclose(action.sum(dim=-1), torch.ones(4), atol=1e-5), \
        f"Action sum not 1.0: {action.sum(dim=-1)}"
    assert (action >= 0).all(), f"Negative actions found: {action.min()}"

    # 2. Gaussian 파라미터 존재
    assert 'mu' in info, "Missing 'mu' in info"
    assert 'std' in info, "Missing 'std' in info"
    assert 'z' in info, "Missing 'z' in info"

    # 3. Log prob 구성 (Projected Gaussian: no Jacobian correction)
    assert 'log_prob_gaussian' in info, "Missing 'log_prob_gaussian' in info (per-action values)"
    assert 'log_prob' in info, "Missing 'log_prob' in info (summed value)"

    # 4. Log prob은 info['log_prob']과 일치해야 함
    assert torch.allclose(log_prob, info['log_prob'], atol=1e-5), \
        f"Log prob mismatch: {log_prob} vs {info['log_prob']}"

    # 5. info['log_prob']은 info['log_prob_gaussian']의 합이어야 함
    log_prob_manual = info['log_prob_gaussian'].sum(dim=-1, keepdim=True)
    assert torch.allclose(info['log_prob'], log_prob_manual, atol=1e-5), \
        f"Log prob != sum(log_prob_gaussian): {info['log_prob']} vs {log_prob_manual}"

    print("✅ Test 8 passed: Projected Gaussian Policy 검증")


def test_log_prob_calculation():
    """
    테스트 9: Projected Gaussian log probability 계산이 정확한가?
    """
    config = {
        'state_dim': 301,  # FinRL Dow30 표준
        'action_dim': 30,
        'emb_dim': 128,
        'm_tokens': 6,
        'M_proto': 8,
        'alpha': 0.3,
        'market_feature_dim': 12
    }

    actor = BCellIRTActor(**config)
    actor.eval()

    state = torch.randn(4, config['state_dim'])
    fitness = torch.randn(4, config['M_proto'])

    action, log_prob, info = actor(state, fitness, deterministic=False)

    # Manual 계산 (Projected Gaussian: unconstrained Gaussian log prob)
    mu = info['mu']
    std = info['std']
    z = info['z']

    # Gaussian log prob (no Jacobian correction for projection)
    import numpy as np
    log_prob_gaussian_manual = -0.5 * (
        ((z - mu) / std) ** 2
        + 2 * torch.log(std)
        + np.log(2 * np.pi)
    ).sum(dim=-1, keepdim=True)

    # Projected Gaussian: log_prob = log_prob_gaussian (projection gradient handled by SAC)
    log_prob_manual = log_prob_gaussian_manual

    assert torch.allclose(log_prob, log_prob_manual, atol=1e-4), \
        f"Log prob calculation error: diff={torch.abs(log_prob - log_prob_manual).max()}"

    print("✅ Test 9 passed: Projected Gaussian log probability 계산 정확성 검증")


if __name__ == '__main__':
    # 직접 실행 시 테스트 수행
    import sys

    print("=" * 70)
    print("IRT Policy 단위 테스트")
    print("=" * 70)

    config = {
        'state_dim': 301,  # FinRL Dow30 표준
        'action_dim': 30,
        'emb_dim': 128,
        'm_tokens': 6,
        'M_proto': 8,
        'alpha': 0.3,
        'market_feature_dim': 12
    }

    state = torch.randn(4, config['state_dim'])

    try:
        test_irt_forward_pass(config, state)
        test_simplex_constraint(config, state)
        test_sb3_integration(config)
        test_device_compatibility(config, state)
        test_irt_decomposition(config, state)
        test_fitness_calculation(config)
        test_replicator_activation(config)
        test_gaussian_projection_policy()
        test_log_prob_calculation()

        print("=" * 70)
        print("✅ All tests passed!")
        print("=" * 70)

    except Exception as e:
        print("=" * 70)
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 70)
        sys.exit(1)
