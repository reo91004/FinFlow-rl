# bipd/utils/metrics.py

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
# jarque_bera는 현재 사용되지 않으므로 제거
from scipy import stats


# 공통 유틸리티 함수
def safe_float(value):
    """안전한 타입 변환 함수 (NumPy 스칼라 처리)"""
    if hasattr(value, 'item'):
        return float(value.item())
    return float(value)

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """샤프 비율 계산"""
    if len(returns) == 0:
        return 0.0
    
    # 입력 타입 안전성 검증
    if hasattr(returns, 'values'):  # pandas Series/DataFrame
        returns = returns.values
    returns = np.asarray(returns)
    
    if returns.std() == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate / 252
    return np.sqrt(252) * excess_returns.mean() / returns.std()

def calculate_max_drawdown(returns):
    """최대 낙폭 계산"""
    if len(returns) == 0:
        return 0.0
    
    # 입력 타입 안전성 검증
    if hasattr(returns, 'values'):  # pandas Series/DataFrame
        returns = returns.values
    returns = np.asarray(returns)
    
    # NumPy 기반 계산으로 통일
    cumulative = np.cumprod(1 + returns)
    rolling_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - rolling_max) / rolling_max
    return np.min(drawdown)

def calculate_volatility(returns):
    """연율화 변동성 계산"""
    if len(returns) == 0:
        return 0.0
    
    # 입력 타입 안전성 검증
    if hasattr(returns, 'values'):  # pandas Series/DataFrame
        returns = returns.values
    returns = np.asarray(returns)
    
    return returns.std() * np.sqrt(252)

def calculate_calmar_ratio(returns):
    """칼마 비율 계산"""
    if len(returns) == 0:
        return 0.0
    
    # 입력 타입 안전성 검증
    if hasattr(returns, 'values'):  # pandas Series/DataFrame
        returns = returns.values
    returns = np.asarray(returns)
    
    annual_return = np.prod(1 + returns) ** (252 / len(returns)) - 1
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
    
    # 입력 타입 안전성 검증
    if hasattr(returns, 'values'):  # pandas Series/DataFrame
        returns = returns.values
    returns = np.asarray(returns)
    
    total_return = np.prod(1 + returns) - 1
    annual_return = np.prod(1 + returns) ** (252 / len(returns)) - 1
    
    # 공통 safe_float 함수 사용
    
    return {
        'total_return': safe_float(total_return),
        'annual_return': safe_float(annual_return),
        'volatility': safe_float(calculate_volatility(returns)),
        'sharpe_ratio': safe_float(calculate_sharpe_ratio(returns)),
        'max_drawdown': safe_float(calculate_max_drawdown(returns)),
        'calmar_ratio': safe_float(calculate_calmar_ratio(returns))
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

def calculate_deflated_sharpe_ratio(returns, num_trials=1, skewness=None, kurtosis=None, risk_free_rate=0.02):
    """
    Deflated Sharpe Ratio (DSR) 계산
    
    백테스트 과최적화를 보정한 통계적으로 유의한 샤프 비율
    
    Args:
        returns: 수익률 시계열
        num_trials: 백테스트에서 시도된 전략 수 (기본값 1)
        skewness: 수익률의 왜도 (None이면 자동 계산)
        kurtosis: 수익률의 첨도 (None이면 자동 계산)
        risk_free_rate: 무위험 수익률
        
    Returns:
        dict: {
            'sharpe_ratio': 전통적 샤프 비율,
            'deflated_sharpe_ratio': 보정된 DSR,
            'threshold': 통계적 유의성 임계값,
            'p_value': p-값,
            'is_significant': 통계적 유의성 (5% 수준)
        }
    """
    if len(returns) == 0:
        return {
            'sharpe_ratio': 0.0,
            'deflated_sharpe_ratio': 0.0,
            'threshold': 0.0,
            'p_value': 1.0,
            'is_significant': False
        }
    
    # 입력 타입 안전성 검증
    if hasattr(returns, 'values'):  # pandas Series/DataFrame
        returns = returns.values
    returns = np.asarray(returns)
    
    # 전통적 샤프 비율 계산
    sharpe_ratio = calculate_sharpe_ratio(returns, risk_free_rate)
    
    n = len(returns)
    if n < 10:  # 샘플이 너무 작으면 신뢰할 수 없음
        return {
            'sharpe_ratio': sharpe_ratio,
            'deflated_sharpe_ratio': 0.0,
            'threshold': 0.0,
            'p_value': 1.0,
            'is_significant': False
        }
    
    # 수익률 분포 특성 계산
    if skewness is None:
        skewness = stats.skew(returns)
    if kurtosis is None:
        kurtosis = stats.kurtosis(returns, fisher=True)  # excess kurtosis
    
    # DSR 임계값 계산 (Bailey & López de Prado, 2012)
    # 다중 테스트 보정을 위한 Bonferroni adjustment
    alpha = 0.05  # 유의수준
    corrected_alpha = alpha / num_trials
    
    # z-score for corrected alpha
    z_alpha = stats.norm.ppf(1 - corrected_alpha/2)
    
    # 표준화된 왜도와 첨도의 분산 추정
    skew_var = 6.0 * (n - 2) / ((n + 1) * (n + 3))
    kurt_var = 24.0 * n * (n - 2) * (n - 3) / ((n + 1)**2 * (n + 3) * (n + 5))
    
    # 샤프 비율의 분산 (고차 모멘트 보정)
    sr_var = (1 + 0.5 * sharpe_ratio**2 - skewness * sharpe_ratio + 
              (kurtosis - 1) / 4.0 * sharpe_ratio**2) / (n - 1)
    
    # 임계 샤프 비율
    threshold_sr = z_alpha * np.sqrt(sr_var)
    
    # DSR 계산
    if sr_var > 0:
        dsr = (sharpe_ratio - threshold_sr) / np.sqrt(sr_var)
    else:
        dsr = 0.0
    
    # p-값 계산
    if sr_var > 0:
        t_stat = sharpe_ratio / np.sqrt(sr_var)
        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
        p_value = min(p_value * num_trials, 1.0)  # Bonferroni correction
    else:
        p_value = 1.0
    
    is_significant = abs(sharpe_ratio) > threshold_sr and p_value < alpha
    
    # 공통 safe_float 함수 사용
    
    return {
        'sharpe_ratio': safe_float(sharpe_ratio),
        'deflated_sharpe_ratio': safe_float(dsr),
        'threshold': safe_float(threshold_sr),
        'p_value': safe_float(p_value),
        'is_significant': bool(is_significant),
        'num_observations': int(n),
        'skewness': safe_float(skewness),
        'kurtosis': safe_float(kurtosis)
    }

def calculate_comprehensive_metrics(returns, num_trials=1, risk_free_rate=0.02):
    """종합적인 성과 지표 계산 (DSR 포함)"""
    base_metrics = calculate_portfolio_metrics(returns)
    dsr_metrics = calculate_deflated_sharpe_ratio(returns, num_trials, risk_free_rate=risk_free_rate)
    
    # 기본 지표와 DSR 지표 결합
    comprehensive_metrics = {**base_metrics, **dsr_metrics}
    
    return comprehensive_metrics