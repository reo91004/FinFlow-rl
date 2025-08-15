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

def calculate_calmar_ratio(returns):
    """칼마 비율 계산"""
    annual_return = (1 + returns).prod() ** (252 / len(returns)) - 1
    max_dd = abs(calculate_max_drawdown(returns))
    
    if max_dd == 0:
        return np.inf if annual_return > 0 else 0
    
    return annual_return / max_dd

def calculate_portfolio_metrics(returns):
    """포트폴리오 종합 메트릭 계산"""
    if len(returns) == 0:
        return {
            'total_return': 0.0,
            'annual_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'calmar_ratio': 0.0
        }
    
    total_return = (1 + returns).prod() - 1
    annual_return = (1 + returns).prod() ** (252 / len(returns)) - 1
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'volatility': calculate_volatility(returns),
        'sharpe_ratio': calculate_sharpe_ratio(returns),
        'max_drawdown': calculate_max_drawdown(returns),
        'calmar_ratio': calculate_calmar_ratio(returns)
    }

def calculate_concentration_index(weights):
    """포트폴리오 집중도 지수 (Herfindahl Index)"""
    return np.sum(weights ** 2)

def calculate_diversification_ratio(weights, cov_matrix):
    """분산투자 비율"""
    portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
    asset_vols = np.sqrt(np.diag(cov_matrix))
    weighted_avg_vol = np.dot(weights, asset_vols)
    
    if portfolio_vol == 0:
        return 1.0
    
    return weighted_avg_vol / portfolio_vol