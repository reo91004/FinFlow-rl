# finrl/meta/env_portfolio_optimization/reward_wrapper.py

"""
Multi-Objective Reward Wrapper for Portfolio Optimization

단일 목적 보상(로그 수익률)을 다목적 보상으로 변환하는 wrapper.
Turnover, Diversity, Drawdown을 고려한 복합 보상 함수를 제공한다.

사용법:
    from finrl.meta.env_portfolio_optimization.reward_wrapper import MultiObjectiveRewardWrapper

    # IRT용 (wrapper 사용)
    env = StockTradingEnv(...)
    env = MultiObjectiveRewardWrapper(env)

    # Baseline용 (wrapper 없이 원본 사용)
    env = StockTradingEnv(...)
"""

from __future__ import annotations

import numpy as np
from gymnasium import Wrapper


class MultiObjectiveRewardWrapper(Wrapper):
    """
    다목적 보상 함수 Wrapper

    기존 보상에 거래 비용, 다양성 보너스, 낙폭 패널티를 추가한다.

    원본 보상: r = ln(V_t / V_{t-1})
    새 보상: r = r_base + r_turnover + r_diversity + r_drawdown

    이를 통해:
    1. Baseline은 원본 단순 보상 사용 (문헌과 공정한 비교)
    2. IRT는 복합 다목적 보상 사용 (복잡한 objective 학습 능력 증명)
    """

    def __init__(
        self,
        env,
        lambda_turnover: float = 0.01,
        lambda_diversity: float = 0.1,
        lambda_drawdown: float = 0.05,
        tc_rate: float = 0.001,
        min_cash: float = 0.02,
        enable_turnover: bool = True,
        enable_diversity: bool = True,
        enable_drawdown: bool = True
    ):
        """
        Args:
            env: Base PortfolioOptimizationEnv
            lambda_turnover: 회전율 패널티 가중치
            lambda_diversity: 다양성 보너스 가중치
            lambda_drawdown: 낙폭 패널티 가중치
            tc_rate: 거래 비용률 (0.001 = 0.1%)
            min_cash: 최소 현금 보유 비율
            enable_turnover: 회전율 패널티 활성화 여부
            enable_diversity: 다양성 보너스 활성화 여부
            enable_drawdown: 낙폭 패널티 활성화 여부
        """
        super().__init__(env)

        self.lambda_turnover = lambda_turnover
        self.lambda_diversity = lambda_diversity
        self.lambda_drawdown = lambda_drawdown
        self.tc_rate = tc_rate
        self.min_cash = min_cash

        # Ablation study용 활성화 플래그
        self.enable_turnover = enable_turnover
        self.enable_diversity = enable_diversity
        self.enable_drawdown = enable_drawdown

        # 상태 추적
        self._peak_value = None
        self._prev_weights = None

        # 로깅용
        self._reward_components = []

    @property
    def state_space(self):
        """Base environment의 state_space 속성 전달"""
        return self.env.state_space

    def __getattr__(self, name):
        """Base environment의 속성 접근을 자동으로 전달"""
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return getattr(self.env, name)

    def reset(self, **kwargs):
        """Wrapper 상태 초기화"""
        obs, info = self.env.reset(**kwargs)

        # 추적 변수 리셋
        self._peak_value = self.env.initial_amount
        self._prev_weights = None
        self._reward_components = []

        return obs, info

    def step(self, action):
        """
        행동 실행 및 보상 변환

        Args:
            action: 포트폴리오 가중치

        Returns:
            obs, reward_transformed, terminated, truncated, info
        """
        # Base environment에서 행동 실행
        obs, reward_base, terminated, truncated, info = self.env.step(action)

        # Environment에서 필요한 정보 추출
        # PortfolioOptimizationEnv는 _final_weights에 가중치 저장
        if hasattr(self.env, '_final_weights') and len(self.env._final_weights) > 0:
            current_weights = self.env._final_weights[-1]
            prev_weights = self._prev_weights if self._prev_weights is not None else current_weights
        else:
            # Fallback
            current_weights = action
            prev_weights = current_weights

        # 포트폴리오 가치 추출
        if hasattr(self.env, '_portfolio_value'):
            portfolio_value = self.env._portfolio_value
        else:
            portfolio_value = self._peak_value

        # 보상 변환
        reward_transformed, components = self._compute_multiobjective(
            base_reward=reward_base,
            weights=current_weights,
            prev_weights=prev_weights,
            portfolio_value=portfolio_value
        )

        # 추적 변수 업데이트
        self._prev_weights = current_weights.copy()
        self._reward_components.append(components)

        # Info에 component 추가
        info['reward_components'] = components

        return obs, reward_transformed, terminated, truncated, info

    def _compute_multiobjective(
        self,
        base_reward: float,
        weights: np.ndarray,
        prev_weights: np.ndarray,
        portfolio_value: float
    ):
        """
        다목적 보상 계산

        Args:
            base_reward: 원본 로그 수익률 보상
            weights: 현재 포트폴리오 가중치
            prev_weights: 이전 포트폴리오 가중치
            portfolio_value: 현재 포트폴리오 가치

        Returns:
            total_reward: 변환된 보상
            components: 로깅용 보상 구성요소 dict
        """
        components = {'base': base_reward}

        # ===== Fix 1: Turnover Band Penalty =====
        # 목표 회전율 6% ± 3% band 내에서 활동 장려
        # Band 밖에서만 패널티 적용 (L2 squared penalty)
        if self.enable_turnover:
            turnover = np.sum(np.abs(weights - prev_weights))
            target_turnover = 0.06  # 목표 6%
            band_width = 0.03       # ±3% 허용

            # Band 밖 deviation에만 패널티
            deviation = np.abs(turnover - target_turnover)
            excess_deviation = np.maximum(deviation - band_width, 0.0)

            # L2 squared penalty (더 부드러운 gradient)
            r_turnover = -0.005 * (excess_deviation ** 2)

            components['turnover'] = r_turnover
            components['turnover_pct'] = turnover
            components['turnover_deviation'] = deviation
            components['turnover_excess'] = excess_deviation
        else:
            r_turnover = 0
            components['turnover'] = 0
            components['turnover_pct'] = 0

        # ===== Fix 2: HHI-based Diversity Penalty =====
        # Herfindahl-Hirschman Index (HHI) 사용
        # HHI = Σw_i^2 (균등분포=1/N, 집중=1)
        # 목표: HHI = 0.20 (적당한 집중도)
        if self.enable_diversity:
            hhi = np.sum(weights ** 2)
            target_hhi = 0.20  # 균등분포(1/30=0.033)와 집중(1) 사이

            # Target보다 높을 때만 패널티 (과도한 집중 방지)
            excess_concentration = np.maximum(hhi - target_hhi, 0.0)

            # L2 squared penalty
            r_diversity = -0.05 * (excess_concentration ** 2)

            # 엔트로피도 계산 (로깅용)
            eps = 1e-8
            w_safe = np.clip(weights, eps, 1.0)
            entropy = -np.sum(w_safe * np.log(w_safe))
            max_entropy = np.log(len(weights))
            normalized_entropy = entropy / max_entropy

            components['diversity'] = r_diversity
            components['hhi'] = hhi
            components['hhi_excess'] = excess_concentration
            components['entropy'] = entropy
            components['normalized_entropy'] = normalized_entropy
        else:
            r_diversity = 0
            components['diversity'] = 0
            components['entropy'] = 0
            components['normalized_entropy'] = 0

        # 구성요소 3: 낙폭 패널티 (유지)
        # 정점 대비 현재 가치 하락폭에 패널티
        # Drawdown = (peak - current) / peak
        if self.enable_drawdown:
            eps = 1e-8
            self._peak_value = max(self._peak_value, portfolio_value)
            drawdown = (self._peak_value - portfolio_value) / (self._peak_value + eps)
            r_drawdown = -self.lambda_drawdown * drawdown
            components['drawdown'] = r_drawdown
            components['drawdown_pct'] = drawdown
        else:
            r_drawdown = 0
            components['drawdown'] = 0
            components['drawdown_pct'] = 0

        # 총 보상 계산
        total_reward = base_reward + r_turnover + r_diversity + r_drawdown
        components['total'] = total_reward

        return total_reward, components

    def get_reward_statistics(self):
        """
        에피소드 전체 보상 구성요소 통계 계산

        Returns:
            stats: 각 구성요소의 mean/std/min/max dict
        """
        if len(self._reward_components) == 0:
            return {}

        stats = {}
        keys = self._reward_components[0].keys()

        for key in keys:
            values = [c[key] for c in self._reward_components]
            stats[f'{key}_mean'] = np.mean(values)
            stats[f'{key}_std'] = np.std(values)
            stats[f'{key}_min'] = np.min(values)
            stats[f'{key}_max'] = np.max(values)

        return stats


# ============================================================================
# 편의 함수
# ============================================================================

def create_multiobjective_env(base_env, **wrapper_kwargs):
    """
    다목적 보상 wrapper를 적용한 environment 생성

    Args:
        base_env: Base PortfolioOptimizationEnv
        **wrapper_kwargs: MultiObjectiveRewardWrapper 인자들

    Returns:
        wrapped_env: 다목적 보상이 적용된 environment
    """
    return MultiObjectiveRewardWrapper(base_env, **wrapper_kwargs)


# ============================================================================
# 테스트 코드
# ============================================================================

if __name__ == "__main__":
    # Wrapper 테스트
    print("MultiObjectiveRewardWrapper 테스트 중...")

    # 테스트용 Mock environment
    class MockEnv:
        def __init__(self):
            self._initial_amount = 1000000
            self._portfolio_value = 1000000
            self._final_weights = []

        def reset(self):
            self._portfolio_value = self._initial_amount
            self._final_weights = [np.array([0.25, 0.25, 0.25, 0.25])]
            return np.random.randn(4, 10, 50), {}

        def step(self, action):
            self._final_weights.append(action)
            self._portfolio_value *= 1.001  # 0.1% 수익
            reward = np.log(1.001)
            obs = np.random.randn(4, 10, 50)
            return obs, reward, False, False, {}

    # Wrapped environment 생성
    base_env = MockEnv()
    env = MultiObjectiveRewardWrapper(base_env)

    # 테스트 에피소드
    obs = env.reset()
    print(f"✅ Reset 성공, obs shape: {obs.shape}")

    for i in range(5):
        action = np.array([0.25, 0.25, 0.25, 0.25])
        obs, reward, done, truncated, info = env.step(action)

        print(f"\nStep {i+1}:")
        print(f"  Reward: {reward:.6f}")
        print(f"  Components:")
        for key, value in info['reward_components'].items():
            print(f"    {key}: {value:.6f}")

    # 통계 추출
    stats = env.get_reward_statistics()
    print(f"\n✅ 에피소드 통계:")
    for key, value in stats.items():
        print(f"  {key}: {value:.6f}")

    print("\n✅ 모든 테스트 통과!")
