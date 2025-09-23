# src/analysis/__init__.py

"""
FinFlow Analysis Module

분석, 모니터링, 시각화, 백테스팅 관련 모듈
"""

from .metrics import MetricsCalculator
from .visualization import (
    plot_portfolio_weights,
    plot_equity_curve,
    plot_returns_distribution,
    create_performance_dashboard
)
from .monitor import PerformanceMonitor, Alert
from .explainer import XAIExplainer, DecisionReport
# BacktestEngine과 BacktestResult는 아직 구현되지 않음
# from .backtest import BacktestEngine, BacktestResult

__all__ = [
    # Metrics
    'MetricsCalculator',

    # Visualization
    'plot_portfolio_weights',
    'plot_equity_curve',
    'plot_returns_distribution',
    'create_performance_dashboard',

    # Monitoring
    'PerformanceMonitor',
    'Alert',

    # XAI
    'XAIExplainer',
    'DecisionReport',

    # Backtest - 아직 구현되지 않음
    # 'BacktestEngine',
    # 'BacktestResult',
]