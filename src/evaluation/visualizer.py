# src/analysis/visualization.py

"""
시각화: 포트폴리오 성과 및 분석 시각화

목적: 학습/평가 결과의 시각적 표현
의존: matplotlib, seaborn
사용처: 분석 스크립트, 보고서 생성
역할: 직관적인 성과 시각화 제공

구현 내용:
- 포트폴리오 가중치 변화 (bar/pie chart)
- 누적 수익률 곡선
- 낙폭 차트
- 리스크-수익 산점도
- 히트맵 (상관관계, 특성 중요도)
- 학습 곡선 (에피소드별 보상)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Union, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def plot_portfolio_weights(weights: np.ndarray,
                          asset_names: List[str],
                          title: str = "Portfolio Allocation",
                          save_path: Optional[Path] = None,
                          figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Plot portfolio weights as bar chart or pie chart
    
    Args:
        weights: Portfolio weights
        asset_names: Asset names
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        fig: Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Filter out small weights for cleaner visualization
    threshold = 0.01
    mask = weights > threshold
    filtered_weights = weights[mask]
    filtered_names = [asset_names[i] for i in range(len(weights)) if mask[i]]
    
    # Bar chart
    colors = plt.cm.Set3(np.linspace(0, 1, len(filtered_weights)))
    bars = ax1.bar(range(len(filtered_weights)), filtered_weights, color=colors)
    ax1.set_xticks(range(len(filtered_weights)))
    ax1.set_xticklabels(filtered_names, rotation=45, ha='right')
    ax1.set_ylabel('Weight')
    ax1.set_title(f'{title} - Bar Chart')
    ax1.set_ylim([0, max(filtered_weights) * 1.1])
    
    # Add value labels on bars
    for bar, weight in zip(bars, filtered_weights):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{weight:.1%}', ha='center', va='bottom')
    
    # Pie chart
    ax2.pie(filtered_weights, labels=filtered_names, autopct='%1.1f%%',
            colors=colors, startangle=90)
    ax2.set_title(f'{title} - Pie Chart')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    return fig

def plot_equity_curve(values: Union[np.ndarray, pd.Series],
                     benchmark: Optional[Union[np.ndarray, pd.Series]] = None,
                     title: str = "Portfolio Value",
                     save_path: Optional[Path] = None,
                     figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Plot portfolio equity curve
    
    Args:
        values: Portfolio values
        benchmark: Benchmark values
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert to pandas for easier plotting
    if not isinstance(values, pd.Series):
        values = pd.Series(values)
    
    # Normalize to start at 1
    normalized_values = values / values.iloc[0]
    
    # Plot strategy
    ax.plot(normalized_values.index, normalized_values.values, 
            label='Strategy', linewidth=2, color='blue')
    
    # Plot benchmark if provided
    if benchmark is not None:
        if not isinstance(benchmark, pd.Series):
            benchmark = pd.Series(benchmark)
        normalized_benchmark = benchmark / benchmark.iloc[0]
        ax.plot(normalized_benchmark.index, normalized_benchmark.values,
               label='Benchmark', linewidth=2, color='gray', alpha=0.7)
        
        # Fill between
        ax.fill_between(normalized_values.index,
                       normalized_values.values,
                       normalized_benchmark.values,
                       where=(normalized_values.values >= normalized_benchmark.values),
                       color='green', alpha=0.3, label='Outperformance')
        ax.fill_between(normalized_values.index,
                       normalized_values.values,
                       normalized_benchmark.values,
                       where=(normalized_values.values < normalized_benchmark.values),
                       color='red', alpha=0.3, label='Underperformance')
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Normalized Value')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Add statistics box
    from src.evaluation.metrics import calculate_sharpe_ratio, calculate_max_drawdown
    
    returns = values.pct_change().dropna()
    sharpe = calculate_sharpe_ratio(returns)
    max_dd = calculate_max_drawdown(values)
    final_value = values.iloc[-1] / values.iloc[0]
    
    stats_text = f'Final: {final_value:.2f}x\n'
    stats_text += f'Sharpe: {sharpe:.2f}\n'
    stats_text += f'Max DD: {max_dd:.1%}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    return fig

def plot_drawdown(values: Union[np.ndarray, pd.Series],
                 title: str = "Drawdown",
                 save_path: Optional[Path] = None,
                 figsize: Tuple[int, int] = (12, 4)) -> plt.Figure:
    """
    Plot drawdown chart
    
    Args:
        values: Portfolio values
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if not isinstance(values, pd.Series):
        values = pd.Series(values)
    
    # Calculate drawdown
    cumulative = values / values.iloc[0]
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max * 100
    
    # Plot
    ax.fill_between(drawdown.index, drawdown.values, 0,
                   color='red', alpha=0.3)
    ax.plot(drawdown.index, drawdown.values, color='red', linewidth=1)
    
    # Mark maximum drawdown
    max_dd_idx = drawdown.idxmin()
    max_dd_value = drawdown.min()
    ax.scatter(max_dd_idx, max_dd_value, color='darkred', s=100, zorder=5)
    ax.annotate(f'Max DD: {max_dd_value:.1f}%',
               xy=(max_dd_idx, max_dd_value),
               xytext=(10, -10), textcoords='offset points',
               fontsize=10, color='darkred',
               arrowprops=dict(arrowstyle='->', color='darkred'))
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Drawdown (%)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([min(max_dd_value * 1.1, -1), 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    return fig

def plot_returns_distribution(returns: Union[np.ndarray, pd.Series],
                            title: str = "Returns Distribution",
                            save_path: Optional[Path] = None,
                            figsize: Tuple[int, int] = (12, 5)) -> plt.Figure:
    """
    Plot returns distribution with statistics
    
    Args:
        returns: Return series
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        fig: Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    # Histogram
    n, bins, patches = ax1.hist(returns * 100, bins=50, density=True,
                                alpha=0.7, color='blue', edgecolor='black')
    
    # Fit normal distribution
    mu, sigma = np.mean(returns * 100), np.std(returns * 100)
    x = np.linspace(returns.min() * 100, returns.max() * 100, 100)
    normal_dist = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
                  np.exp(-0.5 * ((x - mu) / sigma) ** 2))
    ax1.plot(x, normal_dist, 'r-', linewidth=2, label='Normal fit')
    
    # Add mean and median lines
    ax1.axvline(mu, color='green', linestyle='--', linewidth=2, label=f'Mean: {mu:.2f}%')
    ax1.axvline(np.median(returns * 100), color='orange', linestyle='--',
               linewidth=2, label=f'Median: {np.median(returns * 100):.2f}%')
    
    ax1.set_xlabel('Return (%)')
    ax1.set_ylabel('Density')
    ax1.set_title('Return Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(returns, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics text
    from src.evaluation.metrics import calculate_var, calculate_cvar
    
    skew = stats.skew(returns)
    kurt = stats.kurtosis(returns)
    var_95 = calculate_var(returns, 0.05)
    cvar_95 = calculate_cvar(returns, 0.05)
    
    stats_text = f'Skewness: {skew:.3f}\n'
    stats_text += f'Kurtosis: {kurt:.3f}\n'
    stats_text += f'VaR(95%): {var_95*100:.2f}%\n'
    stats_text += f'CVaR(95%): {cvar_95*100:.2f}%'
    
    fig.text(0.5, -0.05, stats_text, ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    return fig

def plot_feature_importance(importance_dict: Dict[str, float],
                          top_n: int = 15,
                          title: str = "Feature Importance",
                          save_path: Optional[Path] = None,
                          figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot feature importance
    
    Args:
        importance_dict: Feature importance dictionary
        top_n: Number of top features to show
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort and select top features
    sorted_features = sorted(importance_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    top_features = sorted_features[:top_n]
    
    features = [f[0] for f in top_features]
    values = [f[1] for f in top_features]
    
    # Color based on positive/negative
    colors = ['green' if v > 0 else 'red' for v in values]
    
    # Create horizontal bar chart
    bars = ax.barh(range(len(features)), values, color=colors, alpha=0.7)
    
    # Customize
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.set_xlabel('Importance')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        x = bar.get_width()
        ax.text(x, bar.get_y() + bar.get_height()/2,
               f'{val:.3f}', ha='left' if x > 0 else 'right',
               va='center', fontsize=9)
    
    # Add zero line
    ax.axvline(x=0, color='black', linewidth=1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    return fig

def plot_rolling_metrics(returns: pd.Series,
                        window: int = 63,
                        metrics: List[str] = ['sharpe', 'volatility'],
                        title: str = "Rolling Metrics",
                        save_path: Optional[Path] = None,
                        figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Plot rolling metrics
    
    Args:
        returns: Return series
        window: Rolling window size
        metrics: List of metrics to plot
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        fig: Matplotlib figure
    """
    from src.evaluation.metrics import calculate_sharpe_ratio
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=figsize, sharex=True)
    
    if n_metrics == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics):
        if metric == 'sharpe':
            rolling = returns.rolling(window).apply(
                lambda x: calculate_sharpe_ratio(x)
            )
            ax.plot(rolling.index, rolling.values, label='Sharpe Ratio')
            ax.axhline(y=1.5, color='g', linestyle='--', alpha=0.5, label='Target')
            ax.set_ylabel('Sharpe Ratio')
            ax.set_title('Rolling Sharpe Ratio')
            
        elif metric == 'volatility':
            rolling = returns.rolling(window).std() * np.sqrt(252) * 100
            ax.plot(rolling.index, rolling.values, label='Volatility', color='orange')
            ax.set_ylabel('Volatility (%)')
            ax.set_title('Rolling Volatility')
            
        elif metric == 'drawdown':
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.rolling(window, min_periods=1).max()
            dd = (cumulative - running_max) / running_max * 100
            ax.fill_between(dd.index, dd.values, 0, color='red', alpha=0.3)
            ax.plot(dd.index, dd.values, color='red', linewidth=1)
            ax.set_ylabel('Drawdown (%)')
            ax.set_title('Rolling Drawdown')
        
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Date')
    plt.suptitle(f'{title} (Window: {window} days)')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    return fig

def plot_correlation_matrix(returns_df: pd.DataFrame,
                          title: str = "Correlation Matrix",
                          save_path: Optional[Path] = None,
                          figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Plot correlation matrix heatmap
    
    Args:
        returns_df: DataFrame of returns
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate correlation
    corr_matrix = returns_df.corr()
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix), k=1)
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
               cmap='coolwarm', center=0, vmin=-1, vmax=1,
               square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
               ax=ax)
    
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    return fig

def plot_regime_timeline(regimes: pd.Series,
                        values: Optional[pd.Series] = None,
                        title: str = "Market Regimes",
                        save_path: Optional[Path] = None,
                        figsize: Tuple[int, int] = (14, 6)) -> plt.Figure:
    """
    Plot regime timeline with optional overlay
    
    Args:
        regimes: Series of regime labels
        values: Optional value series to overlay
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define regime colors
    regime_colors = {
        'Normal': 'green',
        'Elevated Risk': 'yellow',
        'High Volatility': 'orange',
        'Crisis': 'red'
    }
    
    # Plot regime backgrounds
    for regime, color in regime_colors.items():
        mask = regimes == regime
        if mask.any():
            ax.fill_between(regimes.index, 0, 1, where=mask,
                          color=color, alpha=0.3, label=regime,
                          transform=ax.get_xaxis_transform())
    
    # Overlay values if provided
    if values is not None:
        ax2 = ax.twinx()
        ax2.plot(values.index, values.values, color='blue', linewidth=2)
        ax2.set_ylabel('Portfolio Value', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
    
    ax.set_xlabel('Date')
    ax.set_title(title)
    ax.legend(loc='upper left', ncol=4)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    return fig

def create_performance_dashboard(data: Dict,
                                save_path: Optional[Path] = None,
                                figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
    """
    Create comprehensive performance dashboard
    
    Args:
        data: Dictionary containing various performance data
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        fig: Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Equity curve (top, spanning 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    if 'values' in data:
        values = data['values']
        benchmark = data.get('benchmark')
        plot_equity_curve(values, benchmark, ax=ax1)
    
    # 2. Portfolio weights (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    if 'weights' in data and 'asset_names' in data:
        weights = data['weights'][-1] if len(data['weights'].shape) > 1 else data['weights']
        plot_portfolio_weights(weights, data['asset_names'], ax=ax2)
    
    # 3. Drawdown (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    if 'values' in data:
        plot_drawdown(data['values'], ax=ax3)
    
    # 4. Returns distribution (middle center)
    ax4 = fig.add_subplot(gs[1, 1])
    if 'returns' in data:
        plot_returns_distribution(data['returns'], ax=ax4)
    
    # 5. Rolling Sharpe (middle right)
    ax5 = fig.add_subplot(gs[1, 2])
    if 'returns' in data:
        rolling_sharpe = pd.Series(data['returns']).rolling(63).apply(
            lambda x: calculate_sharpe_ratio(x)
        )
        ax5.plot(rolling_sharpe.index, rolling_sharpe.values)
        ax5.axhline(y=1.5, color='g', linestyle='--', alpha=0.5)
        ax5.set_title('Rolling Sharpe (3M)')
        ax5.grid(True, alpha=0.3)
    
    # 6. Monthly returns heatmap (bottom, spanning all columns)
    ax6 = fig.add_subplot(gs[2, :])
    if 'returns' in data:
        returns_series = pd.Series(data['returns'])
        if hasattr(returns_series.index, 'to_period'):
            monthly_returns = returns_series.groupby(
                returns_series.index.to_period('M')
            ).apply(lambda x: (1 + x).prod() - 1)
            
            # Reshape for heatmap
            years = monthly_returns.index.year.unique()
            months = range(1, 13)
            heatmap_data = pd.DataFrame(index=years, columns=months)
            
            for idx, ret in monthly_returns.items():
                heatmap_data.loc[idx.year, idx.month] = ret
            
            sns.heatmap(heatmap_data.astype(float) * 100, annot=True, fmt='.1f',
                       cmap='RdYlGn', center=0, ax=ax6,
                       xticklabels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            ax6.set_title('Monthly Returns (%)')
            ax6.set_xlabel('Month')
            ax6.set_ylabel('Year')
    
    plt.suptitle('Performance Dashboard', fontsize=16, y=1.02)
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    return fig