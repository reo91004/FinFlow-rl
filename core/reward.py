# core/reward.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from collections import deque


class RewardCalculator:
    """고도화된 보상 함수 계산기"""

    def __init__(
        self,
        lookback_window=20,
        transaction_cost_rate=0.001,
        target_volatility=0.15,
        target_max_drawdown=0.1,
    ):
        self.lookback_window = lookback_window
        self.transaction_cost_rate = transaction_cost_rate
        self.target_volatility = target_volatility
        self.target_max_drawdown = target_max_drawdown

        # 성과 추적
        self.return_history = deque(maxlen=252)  # 1년간 수익률
        self.portfolio_values = deque(maxlen=252)
        self.weight_history = deque(maxlen=5)  # 최근 5일 가중치 변화 추적

    def calculate_comprehensive_reward(
        self,
        current_return: float,
        previous_weights: np.ndarray,
        current_weights: np.ndarray,
        market_features: np.ndarray,
        crisis_level: float,
    ) -> Dict[str, float]:
        """종합적인 보상 계산"""

        # 기본 수익률 보상
        return_reward = current_return * 100

        # 위험 조정 성과 보상
        risk_adjusted_reward = self._calculate_risk_adjusted_reward()

        # 거래 비용 패널티
        transaction_cost_penalty = self._calculate_transaction_cost_penalty(
            previous_weights, current_weights
        )

        # 목표 지향적 보상
        target_reward = self._calculate_target_based_reward()

        # 시장 적응성 보상
        adaptation_reward = self._calculate_adaptation_reward(
            market_features, crisis_level
        )

        # 포트폴리오 집중도 패널티
        concentration_penalty = self._calculate_concentration_penalty(current_weights)

        # 종합 보상 계산
        total_reward = (
            return_reward
            + risk_adjusted_reward
            + transaction_cost_penalty
            + target_reward
            + adaptation_reward
            + concentration_penalty
        )

        # 이력 업데이트
        self._update_history(current_return, current_weights)

        return {
            "total_reward": total_reward,
            "return_reward": return_reward,
            "risk_adjusted_reward": risk_adjusted_reward,
            "transaction_cost_penalty": transaction_cost_penalty,
            "target_reward": target_reward,
            "adaptation_reward": adaptation_reward,
            "concentration_penalty": concentration_penalty,
        }

    def _calculate_risk_adjusted_reward(self) -> float:
        """위험 조정 성과 보상 (샤프/소르티노 지수 기반)"""
        if len(self.return_history) < self.lookback_window:
            return 0.0

        returns = np.array(list(self.return_history)[-self.lookback_window :])

        if len(returns) < 2:
            return 0.0

        # 샤프 지수 계산
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe_ratio = mean_return / (std_return + 1e-8) * np.sqrt(252)

        # 소르티노 지수 계산 (하방 변동성만 고려)
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            downside_std = np.std(negative_returns)
            sortino_ratio = mean_return / (downside_std + 1e-8) * np.sqrt(252)
        else:
            sortino_ratio = sharpe_ratio

        # 위험 조정 보상 (샤프와 소르티노의 평균)
        risk_adjusted_score = (sharpe_ratio + sortino_ratio) / 2

        # 정규화 및 스케일링
        reward = np.tanh(risk_adjusted_score) * 10

        return reward

    def _calculate_transaction_cost_penalty(
        self, previous_weights: np.ndarray, current_weights: np.ndarray
    ) -> float:
        """거래 비용 패널티"""
        if previous_weights is None or len(previous_weights) != len(current_weights):
            return 0.0

        # 가중치 변화량 계산
        weight_changes = np.abs(current_weights - previous_weights)
        total_turnover = np.sum(weight_changes) / 2  # 실제 거래량

        # 거래 비용 패널티
        cost_penalty = -total_turnover * self.transaction_cost_rate * 1000

        return cost_penalty

    def _calculate_target_based_reward(self) -> float:
        """목표 지향적 보상"""
        if len(self.return_history) < self.lookback_window:
            return 0.0

        returns = np.array(list(self.return_history)[-self.lookback_window :])

        # 변동성 목표 달성 보상
        current_volatility = np.std(returns) * np.sqrt(252)
        volatility_deviation = abs(current_volatility - self.target_volatility)
        volatility_reward = max(0, 5 - volatility_deviation * 50)

        # 최대 낙폭 목표 달성 보상
        max_drawdown = self._calculate_max_drawdown()
        if max_drawdown < self.target_max_drawdown:
            drawdown_reward = 10
        else:
            drawdown_reward = max(
                -10, -abs(max_drawdown - self.target_max_drawdown) * 100
            )

        return volatility_reward + drawdown_reward

    def _calculate_adaptation_reward(
        self, market_features: np.ndarray, crisis_level: float
    ) -> float:
        """시장 적응성 보상"""
        adaptation_reward = 0.0

        # 위기 상황에서의 적절한 대응 보상
        if crisis_level > 0.5:  # 고위기 상황
            # 최근 수익률이 양수이면 보상
            if len(self.return_history) > 0 and self.return_history[-1] > 0:
                adaptation_reward += crisis_level * 15
            else:
                adaptation_reward -= crisis_level * 10

        # 시장 변동성에 대한 적응 보상
        if len(market_features) > 0:
            market_volatility = market_features[0] if market_features[0] > 0 else 0.1
            if len(self.return_history) > 5:
                portfolio_volatility = np.std(list(self.return_history)[-5:])
                # 시장 변동성이 높을 때 포트폴리오 변동성을 낮게 유지하면 보상
                if market_volatility > 0.3 and portfolio_volatility < market_volatility:
                    adaptation_reward += 5

        return adaptation_reward

    def _calculate_concentration_penalty(self, weights: np.ndarray) -> float:
        """포트폴리오 집중도 패널티"""
        # 허핀달-허쉬만 지수 계산
        hhi = np.sum(weights**2)

        # 과도한 집중에 대한 패널티
        if hhi > 0.4:  # 40% 이상 집중시 패널티
            concentration_penalty = -(hhi - 0.4) * 20
        else:
            concentration_penalty = 0.0

        return concentration_penalty

    def _calculate_max_drawdown(self) -> float:
        """최대 낙폭 계산"""
        if len(self.portfolio_values) < 2:
            return 0.0

        values = np.array(list(self.portfolio_values))
        cumulative_values = np.cumprod(1 + values)
        running_max = np.maximum.accumulate(cumulative_values)
        drawdowns = (cumulative_values - running_max) / running_max

        return abs(np.min(drawdowns))

    def _update_history(self, current_return: float, current_weights: np.ndarray):
        """이력 업데이트"""
        self.return_history.append(current_return)
        self.weight_history.append(current_weights.copy())

        # 포트폴리오 가치 업데이트
        if len(self.portfolio_values) == 0:
            self.portfolio_values.append(current_return)
        else:
            last_value = self.portfolio_values[-1]
            self.portfolio_values.append(current_return)

    def get_performance_metrics(self) -> Dict[str, float]:
        """성과 지표 반환"""
        if len(self.return_history) < 2:
            return {}

        returns = np.array(list(self.return_history))

        return {
            "annualized_return": np.mean(returns) * 252,
            "annualized_volatility": np.std(returns) * np.sqrt(252),
            "sharpe_ratio": np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252),
            "max_drawdown": self._calculate_max_drawdown(),
            "total_return": np.prod(1 + returns) - 1,
            "win_rate": (
                len(returns[returns > 0]) / len(returns) if len(returns) > 0 else 0
            ),
        }
