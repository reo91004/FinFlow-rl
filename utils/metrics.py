# bipd/utils/metrics.py

import numpy as np
import pandas as pd

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """샤프 비율 계산"""
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate / 252
    return np.sqrt(252) * excess_returns.mean() / returns.std()

def calculate_max_drawdown(returns):
    """최대 낙폭 계산"""
    if len(returns) == 0:
        return 0.0
    
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    return drawdown.min()

def calculate_volatility(returns):
    """연율화 변동성 계산"""
    if len(returns) == 0:
        return 0.0
    return returns.std() * np.sqrt(252)


def calculate_portfolio_metrics(returns):
    """포트폴리오 종합 메트릭 계산"""
    if len(returns) == 0:
        return {
            'total_return': 0.0,
            'annual_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }
    
    total_return = (1 + returns).prod() - 1
    annual_return = (1 + returns).prod() ** (252 / len(returns)) - 1
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'volatility': calculate_volatility(returns),
        'sharpe_ratio': calculate_sharpe_ratio(returns),
        'max_drawdown': calculate_max_drawdown(returns)
    }

def calculate_concentration_index(weights):
    """포트폴리오 집중도 지수 (Herfindahl Index)"""
    return np.sum(weights ** 2)

