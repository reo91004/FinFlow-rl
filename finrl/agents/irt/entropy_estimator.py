# finrl/agents/irt/entropy_estimator.py

"""
Policy Entropy Estimator for Projected Distributions

이론적 기반:
- Haarnoja et al. (2018): SAC Appendix B.2 - Constrained action space의 effective entropy
- Ahmed et al. (2019): Adaptive target entropy for automatic tuning

Projected distribution의 entropy 문제:
- Gaussian 정책: z ~ N(μ, σ²)
- Projected 정책: a = proj_simplex(z)
- 실제 엔트로피: H(a) = H(z) - E[log |det J_proj|]
- 현재 구현: log p(a) ≈ log p(z) (Jacobian 무시)

해결책: 경험적 추정
1. 여러 state 샘플링
2. 각 state에서 여러 action 샘플링
3. H ≈ -E[log p(a|s)] 추정

사용법:
    from finrl.agents.irt.entropy_estimator import PolicyEntropyEstimator

    estimator = PolicyEntropyEstimator(n_states=100, n_samples_per_state=20)
    mean_entropy, std_entropy = estimator.estimate(policy, env)

    target_entropy = 0.7 * mean_entropy  # 보수적 타겟
"""

import torch
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class PolicyEntropyEstimator:
    """
    Empirical policy entropy estimator.

    Method:
    1. Sample N states from environment (random walk)
    2. For each state, sample K actions from policy
    3. Estimate H(π) ≈ -E[log π(a|s)]

    참고:
    - State 샘플링: random action으로 environment exploration
    - Action 샘플링: policy.actor.action_log_prob() 사용
    - Entropy: -E[log p(a|s)] (Monte Carlo estimation)
    - Device: Policy의 device 자동 감지 및 동기화
    """

    def __init__(self, n_states: int = 100, n_samples_per_state: int = 20):
        """
        Args:
            n_states: 샘플링할 state 수 (기본: 100)
            n_samples_per_state: 각 state당 action 샘플 수 (기본: 20)
        """
        self.n_states = n_states
        self.n_samples = n_samples_per_state

    def estimate(self, policy, env) -> Tuple[float, float]:
        """
        Policy entropy를 empirically 추정한다.

        Args:
            policy: SACPolicy with actor.action_log_prob() method
            env: Gym/Gymnasium environment

        Returns:
            mean_entropy: Average entropy across states (nats)
            std_entropy: Standard deviation (uncertainty measure)
        """
        logger.info(f"Estimating policy entropy over {self.n_states} states...")

        # ===== Device 자동 감지 =====
        # Policy의 첫 번째 파라미터에서 device 확인
        device = next(policy.parameters()).device
        logger.debug(f"  Policy device: {device}")

        # ===== Step 1: State 샘플링 (Gymnasium API) =====
        # Random walk으로 다양한 state 수집
        states = []

        # Gymnasium API: reset() returns (obs, info)
        obs, _ = env.reset()

        for i in range(self.n_states):
            states.append(obs)

            # Random action으로 state diversity 확보
            action = env.action_space.sample()

            # Gymnasium API: step() returns (obs, reward, terminated, truncated, info)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if done:
                obs, _ = env.reset()

        # NumPy array로 변환 후 device로 이동
        states = torch.FloatTensor(np.array(states)).to(device)

        # ===== Step 2: 각 State에서 Entropy 추정 =====
        entropies = []

        with torch.no_grad():
            for i, state in enumerate(states):
                # 각 state에서 여러 번 독립적으로 샘플링
                log_probs = []

                for _ in range(self.n_samples):
                    # Single state [1, state_dim]
                    state_single = state.unsqueeze(0)

                    # Sample action and get log prob
                    _, log_prob = policy.actor.action_log_prob(state_single)

                    # log_prob shape: [1, 1] or [1]
                    log_probs.append(log_prob.item())  # Scalar

                # Entropy ≈ -E[log p(a|s)]
                entropy = -np.mean(log_probs)
                entropies.append(entropy)

                if (i + 1) % 20 == 0:
                    logger.debug(f"  Progress: {i+1}/{self.n_states} states")

        # ===== Step 3: 통계량 계산 =====
        mean_entropy = np.mean(entropies)
        std_entropy = np.std(entropies)

        logger.info(f"  Estimated entropy: {mean_entropy:.3f} ± {std_entropy:.3f} nats")

        return mean_entropy, std_entropy
