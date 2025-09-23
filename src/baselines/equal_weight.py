# src/baselines/equal_weight.py

import numpy as np
import pandas as pd
from typing import Dict, Optional
from src.data.market_loader import DataLoader
from src.utils.logger import FinFlowLogger


class EqualWeightStrategy:
    """
    균등 가중치 포트폴리오 전략
    모든 자산에 동일한 비중 할당
    """

    def __init__(self):
        """초기화"""
        self.logger = FinFlowLogger("EqualWeight")
        self.portfolio_values = []

    def backtest(self, config: Dict) -> Dict:
        """
        백테스트 실행

        Args:
            config: 설정 딕셔너리

        Returns:
            백테스트 메트릭
        """
        self.logger.info("균등 가중치 전략 백테스트 시작")

        # 데이터 로드
        loader = DataLoader(config.get('data', {}))
        price_data = loader.load()

        # 테스트 데이터만 사용 (마지막 20%)
        n = len(price_data)
        test_start = int(n * 0.8)
        test_data = price_data.iloc[test_start:]

        # 초기 자본
        initial_capital = config.get('env', {}).get('initial_balance', 1000000)
        transaction_cost = config.get('env', {}).get('transaction_cost', 0.001)

        # 자산 수
        n_assets = len(test_data.columns)

        # 균등 가중치
        weights = np.ones(n_assets) / n_assets

        # 포트폴리오 시뮬레이션
        portfolio_value = initial_capital
        self.portfolio_values = [portfolio_value]
        returns = []

        for i in range(1, len(test_data)):
            # 일일 수익률
            daily_returns = test_data.iloc[i] / test_data.iloc[i-1] - 1

            # 포트폴리오 수익률 (거래비용 고려)
            portfolio_return = np.sum(weights * daily_returns) - transaction_cost

            # 포트폴리오 가치 업데이트
            portfolio_value *= (1 + portfolio_return)
            self.portfolio_values.append(portfolio_value)
            returns.append(portfolio_return)

        # 메트릭 계산
        returns = np.array(returns)
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)

        # Maximum Drawdown
        portfolio_values = np.array(self.portfolio_values)
        running_max = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - running_max) / running_max
        mdd = np.min(drawdown)

        # 전체 수익률
        total_return = (portfolio_values[-1] - initial_capital) / initial_capital

        metrics = {
            'sharpe': sharpe,
            'returns': total_return,
            'annual_return': total_return * (252 / len(test_data)),
            'std': np.std(returns) * np.sqrt(252),
            'mdd': mdd,
            'calmar': (total_return * 252 / len(test_data)) / abs(mdd) if mdd != 0 else 0,
            'final_value': portfolio_values[-1],
            'n_days': len(test_data),
        }

        self.logger.info(f"백테스트 완료: Sharpe={sharpe:.3f}, Return={total_return:.1%}, MDD={mdd:.1%}")

        return metrics