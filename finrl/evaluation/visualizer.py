# finrl/evaluation/visualizer.py

"""
포트폴리오 성과 시각화

평가 결과를 시각화하여 분석을 용이하게 한다.

일반 포트폴리오 시각화:
- plot_portfolio_value: 포트폴리오 가치 추이
- plot_returns: 일일 수익률 분포
- plot_drawdown: Drawdown 차트

IRT 특화 시각화:
- plot_irt_decomposition: IRT 분해 (w_rep vs w_ot)
- plot_portfolio_weights: 포트폴리오 가중치 스택 차트
- plot_crisis_levels: T-Cell 위기 레벨
- plot_prototype_weights: 프로토타입 가중치 + 엔트로피
- plot_stock_analysis: Top 10 holdings
- plot_performance_timeline: 성능 타임라인
- plot_benchmark_comparison: 벤치마크 비교
- plot_risk_dashboard: 리스크 대시보드
- plot_tcell_analysis: T-Cell 위기 타입 분석
- plot_attribution_analysis: Top 10 contributors
- plot_cost_matrix: Immunological cost matrix

통합 함수:
- plot_all: 모든 시각화 자동 생성
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, List
from pathlib import Path

# 한글 폰트 설정 (선택사항, 환경에 따라 조정)
plt.rcParams['axes.unicode_minus'] = False

def plot_portfolio_value(values: np.ndarray,
                        dates: Optional[List] = None,
                        title: str = "Portfolio Value",
                        save_path: Optional[str] = None):
    """
    포트폴리오 가치 추이 그래프

    Args:
        values: 포트폴리오 가치 배열 [T]
        dates: 날짜 리스트 (optional)
        title: 그래프 제목
        save_path: 저장 경로 (optional)
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    if dates is not None:
        ax.plot(dates, values, linewidth=2, color='steelblue')
        ax.set_xlabel('Date')
    else:
        ax.plot(values, linewidth=2, color='steelblue')
        ax.set_xlabel('Step')

    ax.set_ylabel('Portfolio Value ($)')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 초기값과 최종값 표시
    initial_value = values[0]
    final_value = values[-1]
    total_return = (final_value - initial_value) / initial_value * 100

    ax.axhline(y=initial_value, color='gray', linestyle='--', alpha=0.5, label=f'Initial: ${initial_value:,.0f}')
    ax.axhline(y=final_value, color='green' if final_value > initial_value else 'red',
               linestyle='--', alpha=0.5, label=f'Final: ${final_value:,.0f} ({total_return:+.1f}%)')

    ax.legend(loc='best')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_returns(returns: np.ndarray,
                title: str = "Daily Returns Distribution",
                save_path: Optional[str] = None):
    """
    일일 수익률 분포 그래프

    Args:
        returns: 수익률 배열 [T]
        title: 그래프 제목
        save_path: 저장 경로 (optional)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 시계열 그래프
    ax1.plot(returns, linewidth=1, color='steelblue', alpha=0.7)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Daily Return (%)')
    ax1.set_title('Daily Returns Over Time', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # 히스토그램
    ax2.hist(returns, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.axvline(x=returns.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {returns.mean():.3f}%')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Daily Return (%)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Returns Distribution', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_drawdown(values: np.ndarray,
                 dates: Optional[List] = None,
                 title: str = "Drawdown",
                 save_path: Optional[str] = None):
    """
    Drawdown 차트

    Args:
        values: 포트폴리오 가치 배열 [T]
        dates: 날짜 리스트 (optional)
        title: 그래프 제목
        save_path: 저장 경로 (optional)
    """
    # Drawdown 계산
    running_max = np.maximum.accumulate(values)
    drawdown = (values - running_max) / running_max * 100

    fig, ax = plt.subplots(figsize=(12, 6))

    if dates is not None:
        ax.fill_between(dates, drawdown, 0, color='red', alpha=0.3)
        ax.plot(dates, drawdown, linewidth=1, color='darkred')
        ax.set_xlabel('Date')
    else:
        ax.fill_between(range(len(drawdown)), drawdown, 0, color='red', alpha=0.3)
        ax.plot(drawdown, linewidth=1, color='darkred')
        ax.set_xlabel('Step')

    ax.set_ylabel('Drawdown (%)')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Max Drawdown 표시
    max_dd = drawdown.min()
    max_dd_idx = drawdown.argmin()

    if dates is not None:
        ax.scatter(dates[max_dd_idx], max_dd, color='darkred', s=100, zorder=5,
                  label=f'Max Drawdown: {max_dd:.2f}%')
    else:
        ax.scatter(max_dd_idx, max_dd, color='darkred', s=100, zorder=5,
                  label=f'Max Drawdown: {max_dd:.2f}%')

    ax.legend(loc='best')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ========== IRT 특화 시각화 ==========

def plot_irt_decomposition(w_rep: np.ndarray,
                           w_ot: np.ndarray,
                           title: str = "IRT Decomposition",
                           save_path: Optional[str] = None):
    """
    IRT 분해: w = (1-α)·w_rep + α·w_ot

    Args:
        w_rep: Replicator component [T, N]
        w_ot: OT component [T, N]
        title: 그래프 제목
        save_path: 저장 경로 (optional)
    """
    # 시간 축 평균
    avg_w_rep = w_rep.mean(axis=1)
    avg_w_ot = w_ot.mean(axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # (1) Decomposition Over Time
    ax1 = axes[0, 0]
    ax1.plot(avg_w_rep, label='Replicator Component', alpha=0.8, linewidth=2)
    ax1.plot(avg_w_ot, label='OT Component', alpha=0.8, linewidth=2)
    ax1.fill_between(range(len(avg_w_rep)), 0, avg_w_rep, alpha=0.2)
    ax1.fill_between(range(len(avg_w_ot)), 0, avg_w_ot, alpha=0.2)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Average Weight')
    ax1.set_title('IRT Decomposition Over Time', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # (2) Correlation Scatter
    ax2 = axes[0, 1]
    ax2.scatter(avg_w_rep, avg_w_ot, alpha=0.5, s=20)
    corr = np.corrcoef(avg_w_rep, avg_w_ot)[0, 1]
    ax2.set_xlabel('Replicator Component')
    ax2.set_ylabel('OT Component')
    ax2.set_title(f'Correlation: {corr:.3f}', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # (3) Distribution
    ax3 = axes[1, 0]
    # bins를 데이터에 맞게 동적 조정 (범위가 작을 때 대응)
    try:
        bins_rep = min(max(len(np.unique(avg_w_rep)), 3), 20)
        bins_ot = min(max(len(np.unique(avg_w_ot)), 3), 20)
        ax3.hist(avg_w_rep, bins=bins_rep, alpha=0.6, label='Replicator', edgecolor='black')
        ax3.hist(avg_w_ot, bins=bins_ot, alpha=0.6, label='OT', edgecolor='black')
    except ValueError:
        # 데이터 범위가 너무 작으면 scatter plot 사용
        ax3.scatter(avg_w_rep, np.zeros_like(avg_w_rep), alpha=0.6, label='Replicator', s=20)
        ax3.scatter(avg_w_ot, np.ones_like(avg_w_ot) * 0.5, alpha=0.6, label='OT', s=20)
        ax3.set_ylabel('Component')
    ax3.set_xlabel('Weight Value')
    if not isinstance(ax3, plt.Axes) or 'Component' not in ax3.get_ylabel():
        ax3.set_ylabel('Frequency')
    ax3.set_title('Component Distribution', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # (4) Cumulative Contribution
    ax4 = axes[1, 1]
    cumsum_rep = np.cumsum(avg_w_rep)
    cumsum_ot = np.cumsum(avg_w_ot)
    ax4.plot(cumsum_rep, label='Cumulative Replicator', linewidth=2)
    ax4.plot(cumsum_ot, label='Cumulative OT', linewidth=2)
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Cumulative Weight')
    ax4.set_title('Cumulative Contribution', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_portfolio_weights(weights: np.ndarray,
                           asset_names: Optional[List[str]] = None,
                           title: str = "Portfolio Weights Over Time",
                           save_path: Optional[str] = None):
    """
    포트폴리오 가중치 스택 차트

    Args:
        weights: 포트폴리오 가중치 [T, N]
        asset_names: 자산 이름 리스트 (optional)
        title: 그래프 제목
        save_path: 저장 경로 (optional)
    """
    n_assets = weights.shape[1]

    if asset_names is None:
        asset_names = [f'Asset {i+1}' for i in range(n_assets)]

    fig, ax = plt.subplots(figsize=(14, 8))

    # 스택 영역 차트
    bottom = np.zeros(len(weights))
    colors = plt.cm.tab20(np.linspace(0, 1, n_assets))

    for i in range(n_assets):
        ax.fill_between(range(len(weights)), bottom, bottom + weights[:, i],
                        alpha=0.8, color=colors[i], label=asset_names[i])
        bottom += weights[:, i]

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Weight')
    ax.set_ylim(0, 1)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_crisis_levels(crisis_levels: np.ndarray,
                       title: str = "Crisis Level Detection",
                       save_path: Optional[str] = None):
    """
    T-Cell 위기 레벨 시각화

    Args:
        crisis_levels: 위기 레벨 배열 [T]
        title: 그래프 제목
        save_path: 저장 경로 (optional)
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(crisis_levels, linewidth=2, color='red', label='Crisis Level')
    ax.fill_between(range(len(crisis_levels)), 0, crisis_levels, alpha=0.3, color='red')

    # 임계값 표시
    ax.axhline(0.7, color='darkred', linestyle='--', alpha=0.7, label='High Crisis Threshold')
    ax.axhline(0.3, color='orange', linestyle='--', alpha=0.7, label='Medium Crisis Threshold')

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Crisis Level')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_prototype_weights(proto_weights: np.ndarray,
                           title: str = "Prototype Weights",
                           save_path: Optional[str] = None):
    """
    프로토타입 가중치 및 엔트로피 시각화

    Args:
        proto_weights: 프로토타입 가중치 [T, M]
        title: 그래프 제목
        save_path: 저장 경로 (optional)
    """
    n_protos = proto_weights.shape[1]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # 개별 프로토타입 가중치
    for i in range(n_protos):
        axes[0].plot(proto_weights[:, i], alpha=0.7, label=f'Proto {i+1}')

    axes[0].set_title('Prototype Weights Over Time', fontsize=14)
    axes[0].set_ylabel('Weight')
    axes[0].legend(ncol=4)
    axes[0].grid(True, alpha=0.3)

    # 프로토타입 엔트로피 (다양성 지표)
    entropy = -np.sum(proto_weights * np.log(proto_weights + 1e-8), axis=1)
    axes[1].plot(entropy, linewidth=2, color='purple', label='Entropy')
    axes[1].fill_between(range(len(entropy)), 0, entropy, alpha=0.3, color='purple')

    axes[1].set_title('Prototype Diversity (Entropy)', fontsize=14)
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Entropy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_stock_analysis(weights: np.ndarray,
                       symbols: List[str],
                       title: str = "Top 10 Holdings",
                       save_path: Optional[str] = None):
    """
    Top 10 holdings 분석

    Args:
        weights: 포트폴리오 가중치 [T, N]
        symbols: 주식 심볼 리스트
        title: 그래프 제목
        save_path: 저장 경로 (optional)
    """
    avg_weights = weights.mean(axis=0)
    top_indices = np.argsort(avg_weights)[::-1][:10]

    fig, ax = plt.subplots(figsize=(10, 6))
    top_symbols = [symbols[i] if i < len(symbols) else f'Asset_{i}' for i in top_indices]
    top_weights = avg_weights[top_indices]

    ax.barh(top_symbols, top_weights, color='steelblue', edgecolor='black')
    ax.set_xlabel('Average Weight')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_performance_timeline(returns: np.ndarray,
                              title: str = "Performance Timeline",
                              save_path: Optional[str] = None):
    """
    성능 타임라인 (누적 수익률)

    Args:
        returns: 수익률 배열 [T]
        title: 그래프 제목
        save_path: 저장 경로 (optional)
    """
    cumulative = np.cumprod(1 + returns) - 1

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(cumulative, linewidth=2, label='Cumulative Return')
    ax.fill_between(range(len(cumulative)), 0, cumulative, alpha=0.3)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Cumulative Return')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_benchmark_comparison(returns: np.ndarray,
                              benchmark_returns: Optional[np.ndarray] = None,
                              title: str = "Benchmark Comparison",
                              save_path: Optional[str] = None):
    """
    벤치마크 비교

    Args:
        returns: 전략 수익률 [T]
        benchmark_returns: 벤치마크 수익률 [T] (optional)
        title: 그래프 제목
        save_path: 저장 경로 (optional)
    """
    cumulative = np.cumprod(1 + returns) - 1

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(cumulative, linewidth=2, label='IRT Strategy', color='blue')

    if benchmark_returns is not None:
        cumulative_benchmark = np.cumprod(1 + benchmark_returns) - 1
        ax.plot(cumulative_benchmark, linewidth=2, label='Benchmark', color='gray', alpha=0.7)
    else:
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5, label='Zero Return')

    ax.set_xlabel('Time Step')
    ax.set_ylabel('Cumulative Return')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_risk_dashboard(returns: np.ndarray,
                        metrics: dict,
                        title: str = "Risk Dashboard",
                        save_path: Optional[str] = None):
    """
    리스크 대시보드 (VaR/CVaR, Drawdown)

    Args:
        returns: 수익률 배열 [T]
        metrics: 메트릭 딕셔너리
        title: 그래프 제목
        save_path: 저장 경로 (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (1) Returns Distribution
    axes[0, 0].hist(returns, bins='auto', alpha=0.7, edgecolor='black', color='steelblue')
    axes[0, 0].axvline(returns.mean(), color='red', linestyle='--', linewidth=2,
                       label=f'Mean = {returns.mean():.4f}')
    axes[0, 0].set_xlabel('Return')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Returns Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # (2) Metrics Summary
    axes[0, 1].axis('off')
    metric_text = f"""
    Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}
    Sortino Ratio: {metrics.get('sortino_ratio', 0):.3f}
    Max Drawdown: {metrics.get('max_drawdown', 0):.3f}
    Volatility: {metrics.get('volatility', 0):.3f}
    """
    axes[0, 1].text(0.1, 0.5, metric_text, fontsize=14, verticalalignment='center',
                    family='monospace')
    axes[0, 1].set_title('Key Metrics', fontsize=12, fontweight='bold')

    # (3) Cumulative Returns
    cumulative = np.cumprod(1 + returns) - 1
    axes[1, 0].plot(cumulative, linewidth=2, color='green')
    axes[1, 0].fill_between(range(len(cumulative)), 0, cumulative, alpha=0.3, color='green')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Cumulative Return')
    axes[1, 0].set_title('Cumulative Returns', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # (4) Drawdown
    cumulative_wealth = 1 + cumulative
    running_max = np.maximum.accumulate(cumulative_wealth)
    drawdown = (cumulative_wealth - running_max) / running_max

    axes[1, 1].fill_between(range(len(drawdown)), 0, drawdown, alpha=0.5, color='red')
    axes[1, 1].plot(drawdown, linewidth=1, color='darkred')
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Drawdown')
    axes[1, 1].set_title('Drawdown', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_tcell_analysis(crisis_types: np.ndarray,
                        crisis_levels: np.ndarray,
                        title: str = "T-Cell Analysis",
                        save_path: Optional[str] = None):
    """
    T-Cell 위기 타입 분석

    Args:
        crisis_types: 위기 타입 배열 [T, K]
        crisis_levels: 위기 레벨 배열 [T]
        title: 그래프 제목
        save_path: 저장 경로 (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (1) Crisis Types Average
    avg_types = crisis_types.mean(axis=0)
    axes[0].bar(range(len(avg_types)), avg_types, color='coral', edgecolor='black')
    axes[0].set_xlabel('Crisis Type Index')
    axes[0].set_ylabel('Average Activation')
    axes[0].set_title('Average Crisis Type Distribution', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')

    # (2) Crisis Level Timeline
    axes[1].plot(crisis_levels, linewidth=2, color='red', label='Crisis Level')
    axes[1].fill_between(range(len(crisis_levels)), 0, crisis_levels, alpha=0.3, color='red')
    axes[1].axhline(0.5, color='darkred', linestyle='--', alpha=0.7, label='Threshold = 0.5')
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Crisis Level')
    axes[1].set_title('Crisis Level Over Time', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_attribution_analysis(weights: np.ndarray,
                              returns: np.ndarray,
                              symbols: List[str],
                              title: str = "Top 10 Contributors",
                              save_path: Optional[str] = None):
    """
    Attribution 분석 (Top 10 contributors)

    Args:
        weights: 포트폴리오 가중치 [T, N]
        returns: 수익률 배열 [T]
        symbols: 주식 심볼 리스트
        title: 그래프 제목
        save_path: 저장 경로 (optional)
    """
    avg_weights = weights.mean(axis=0)
    total_return = np.prod(1 + returns) - 1
    contributions = avg_weights * total_return

    top_indices = np.argsort(np.abs(contributions))[::-1][:10]
    top_symbols = [symbols[i] if i < len(symbols) else f'Asset_{i}' for i in top_indices]
    top_contributions = contributions[top_indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['green' if c > 0 else 'red' for c in top_contributions]
    ax.barh(top_symbols, top_contributions, color=colors, edgecolor='black')
    ax.set_xlabel('Contribution to Total Return')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_cost_matrix(cost_matrices: np.ndarray,
                    title: str = "Average Immunological Cost Matrix",
                    save_path: Optional[str] = None):
    """
    Immunological Cost Matrix 히트맵

    Args:
        cost_matrices: Cost matrix 배열 [T, m, M]
        title: 그래프 제목
        save_path: 저장 경로 (optional)
    """
    if len(cost_matrices) == 0:
        print("  Cost matrix 데이터가 없어 시각화를 건너뜁니다.")
        return

    avg_cost = np.mean(cost_matrices, axis=0)
    m, M = avg_cost.shape

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(avg_cost, cmap='RdYlGn_r', aspect='auto')
    ax.set_xlabel('Prototype Index (M)')
    ax.set_ylabel('Epitope Index (m)')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(range(M))
    ax.set_xticklabels([f'P{i+1}' for i in range(M)])
    ax.set_yticks(range(m))
    ax.set_yticklabels([f'E{i+1}' for i in range(m)])
    plt.colorbar(im, ax=ax, label='Cost')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ========== 통합 함수 ==========

def plot_all(portfolio_values: np.ndarray,
            dates: Optional[List] = None,
            output_dir: str = "evaluation_plots",
            irt_data: Optional[dict] = None):
    """
    모든 시각화를 한 번에 생성

    일반 포트폴리오: 3개 시각화
    IRT 모델: 14개 시각화 (IRT 데이터 포함 시)

    Args:
        portfolio_values: 포트폴리오 가치 배열 [T]
        dates: 날짜 리스트 (optional)
        output_dir: 저장 디렉토리
        irt_data: IRT 특화 데이터 (optional)
            - w_rep: Replicator component [T, N]
            - w_ot: OT component [T, N]
            - weights: 포트폴리오 가중치 [T, N]
            - crisis_levels: 위기 레벨 [T]
            - crisis_types: 위기 타입 [T, K]
            - prototype_weights: 프로토타입 가중치 [T, M]
            - symbols: 주식 심볼 리스트
            - cost_matrices: Cost matrix [T, m, M] (optional)
            - benchmark_returns: 벤치마크 수익률 [T] (optional)
            - metrics: 메트릭 딕셔너리
    """
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 기본 포트폴리오 시각화 (항상 생성)
    print(f"시각화를 생성합니다...")

    # 1. Portfolio Value
    plot_portfolio_value(
        portfolio_values,
        dates=dates,
        save_path=str(output_path / "portfolio_value.png")
    )

    # 2. Returns
    returns = np.diff(portfolio_values) / portfolio_values[:-1] * 100
    plot_returns(
        returns,
        save_path=str(output_path / "returns_distribution.png")
    )

    # 3. Drawdown
    plot_drawdown(
        portfolio_values,
        dates=dates,
        save_path=str(output_path / "drawdown.png")
    )

    # IRT 특화 시각화 (irt_data 있을 때만)
    if irt_data is not None:
        # 4. IRT Decomposition
        if 'w_rep' in irt_data and 'w_ot' in irt_data:
            plot_irt_decomposition(
                irt_data['w_rep'],
                irt_data['w_ot'],
                save_path=str(output_path / "irt_decomposition.png")
            )

        # 5. Portfolio Weights
        if 'weights' in irt_data:
            symbols = irt_data.get('symbols', None)
            plot_portfolio_weights(
                irt_data['weights'],
                asset_names=symbols,
                save_path=str(output_path / "portfolio_weights.png")
            )

        # 6. Crisis Levels
        if 'crisis_levels' in irt_data:
            plot_crisis_levels(
                irt_data['crisis_levels'],
                save_path=str(output_path / "crisis_levels.png")
            )

        # 7. Prototype Weights
        if 'prototype_weights' in irt_data:
            plot_prototype_weights(
                irt_data['prototype_weights'],
                save_path=str(output_path / "prototype_weights.png")
            )

        # 8. Stock Analysis
        if 'weights' in irt_data and 'symbols' in irt_data:
            plot_stock_analysis(
                irt_data['weights'],
                irt_data['symbols'],
                save_path=str(output_path / "stock_analysis.png")
            )

        # 9. Performance Timeline
        plot_performance_timeline(
            returns,
            save_path=str(output_path / "performance_timeline.png")
        )

        # 10. Benchmark Comparison
        benchmark_returns = irt_data.get('benchmark_returns', None)
        plot_benchmark_comparison(
            returns,
            benchmark_returns=benchmark_returns,
            save_path=str(output_path / "benchmark_comparison.png")
        )

        # 11. Risk Dashboard
        if 'metrics' in irt_data:
            plot_risk_dashboard(
                returns,
                irt_data['metrics'],
                save_path=str(output_path / "risk_dashboard.png")
            )

        # 12. T-Cell Analysis
        if 'crisis_types' in irt_data and 'crisis_levels' in irt_data:
            plot_tcell_analysis(
                irt_data['crisis_types'],
                irt_data['crisis_levels'],
                save_path=str(output_path / "tcell_analysis.png")
            )

        # 13. Attribution Analysis
        if 'weights' in irt_data and 'symbols' in irt_data:
            plot_attribution_analysis(
                irt_data['weights'],
                returns,
                irt_data['symbols'],
                save_path=str(output_path / "attribution_analysis.png")
            )

        # 14. Cost Matrix
        if 'cost_matrices' in irt_data:
            plot_cost_matrix(
                irt_data['cost_matrices'],
                save_path=str(output_path / "cost_matrix.png")
            )

        print(f"  시각화 14개 생성 완료: {output_dir}")
        print(f"  일반: portfolio_value, returns_distribution, drawdown")
        print(f"  IRT: irt_decomposition, portfolio_weights, crisis_levels, prototype_weights,")
        print(f"       stock_analysis, performance_timeline, benchmark_comparison, risk_dashboard,")
        print(f"       tcell_analysis, attribution_analysis, cost_matrix")
    else:
        print(f"  시각화 3개 생성 완료: {output_dir}")
        print(f"  - portfolio_value.png")
        print(f"  - returns_distribution.png")
        print(f"  - drawdown.png")


def save_evaluation_results(results: dict, save_dir: Path, config: dict = None):
    """
    평가 결과를 JSON으로 저장

    두 개의 JSON 파일을 생성:
    - evaluation_results.json: 모든 원시 데이터
    - evaluation_insights.json: 구조화된 해석 정보

    Args:
        results: 평가 결과 딕셔너리 (returns, weights, crisis_levels, ...)
        save_dir: 저장 디렉토리
        config: 설정 딕셔너리 (optional, insights 생성용)
    """
    import json
    from datetime import datetime

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. evaluation_results.json 저장
    results_copy = results.copy()

    # numpy 배열을 리스트로 변환
    for key, value in results_copy.items():
        if isinstance(value, np.ndarray):
            results_copy[key] = value.tolist()
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
            results_copy[key] = [v.tolist() for v in value]

    results_path = save_dir / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results_copy, f, indent=2)

    print(f"평가 결과 저장 완료: {results_path}")

    # 2. evaluation_insights.json 생성 및 저장
    insights = _generate_insights(results, config)

    insights_path = save_dir / 'evaluation_insights.json'
    with open(insights_path, 'w') as f:
        json.dump(insights, f, indent=2, ensure_ascii=False)

    print(f"해석 정보 저장 완료: {insights_path}")


def _generate_insights(results: dict, config: dict = None) -> dict:
    """
    구조화된 해석 정보 생성

    그림 없이도 JSON으로 IRT 의사결정을 해석할 수 있도록
    핵심 인사이트를 추출한다.

    Args:
        results: evaluation_results.json과 동일한 구조
        config: 설정 딕셔너리 (optional)

    Returns:
        insights: 구조화된 해석 정보
    """
    returns = np.array(results.get('returns', []))
    weights = np.array(results.get('weights', []))
    crisis_levels = np.array(results.get('crisis_levels', []))
    crisis_types = np.array(results.get('crisis_types', []))
    proto_weights = np.array(results.get('prototype_weights', []))
    w_rep = np.array(results.get('w_rep', []))
    w_ot = np.array(results.get('w_ot', []))
    eta_list = np.array(results.get('eta', []))
    symbols = results.get('symbols', [])
    metrics = results.get('metrics', {})

    n_steps = len(returns)
    n_assets = weights.shape[1] if len(weights.shape) > 1 else 0

    # ===== 1. Summary =====
    summary = {
        'total_return': float(metrics.get('total_return', 0.0)),
        'sharpe_ratio': float(metrics.get('sharpe_ratio', 0.0)),
        'sortino_ratio': float(metrics.get('sortino_ratio', 0.0)),
        'calmar_ratio': float(metrics.get('calmar_ratio', 0.0)),
        'max_drawdown': float(metrics.get('max_drawdown', 0.0)),
        'avg_crisis_level': float(crisis_levels.mean()) if len(crisis_levels) > 0 else 0.0,
        'total_steps': int(n_steps)
    }

    # ===== 2. Top Holdings =====
    top_holdings = []
    if n_assets > 0:
        avg_weights = weights.mean(axis=0)
        top_indices = np.argsort(avg_weights)[::-1][:10]

        # 수익 기여도 계산
        total_return = metrics.get('total_return', 0.0)
        weight_contributions = avg_weights * total_return

        for idx in top_indices:
            top_holdings.append({
                'symbol': symbols[idx] if idx < len(symbols) else f'Asset_{idx}',
                'avg_weight': float(avg_weights[idx]),
                'contribution': float(weight_contributions[idx])
            })

    # ===== 3. Crisis vs Normal Analysis =====
    crisis_threshold = 0.5
    crisis_mask = crisis_levels > crisis_threshold if len(crisis_levels) > 0 else np.array([])
    normal_mask = ~crisis_mask if len(crisis_mask) > 0 else np.array([])

    crisis_returns = returns[crisis_mask] if crisis_mask.sum() > 0 else np.array([0])
    normal_returns = returns[normal_mask] if normal_mask.sum() > 0 else np.array([0])

    def safe_sharpe(rets):
        if len(rets) == 0 or rets.std() == 0:
            return 0.0
        return float(rets.mean() / rets.std() * np.sqrt(252))

    crisis_vs_normal = {
        'crisis': {
            'sharpe': safe_sharpe(crisis_returns),
            'avg_return': float(crisis_returns.mean()),
            'volatility': float(crisis_returns.std()),
            'steps': int(crisis_mask.sum()) if len(crisis_mask) > 0 else 0
        },
        'normal': {
            'sharpe': safe_sharpe(normal_returns),
            'avg_return': float(normal_returns.mean()),
            'volatility': float(normal_returns.std()),
            'steps': int(normal_mask.sum()) if len(normal_mask) > 0 else 0
        }
    }

    # ===== 4. IRT Decomposition =====
    alpha = config.get('irt', {}).get('alpha', 0.3) if config else 0.3
    w_rep_norm = np.linalg.norm(w_rep, axis=1) if len(w_rep.shape) > 1 else np.array([0])
    w_ot_norm = np.linalg.norm(w_ot, axis=1) if len(w_ot.shape) > 1 else np.array([0])

    w_rep_contrib = (1 - alpha) * w_rep_norm.mean() if len(w_rep_norm) > 0 else 0.0
    w_ot_contrib = alpha * w_ot_norm.mean() if len(w_ot_norm) > 0 else 0.0
    total_contrib = w_rep_contrib + w_ot_contrib

    irt_decomposition = {
        'avg_w_rep_contribution': float(w_rep_contrib / total_contrib) if total_contrib > 0 else 0.0,
        'avg_w_ot_contribution': float(w_ot_contrib / total_contrib) if total_contrib > 0 else 0.0,
        'correlation_w_rep_w_ot': float(np.corrcoef(w_rep.flatten(), w_ot.flatten())[0, 1]) if len(w_rep.flatten()) > 1 else 0.0,
        'avg_eta': float(eta_list.mean()) if len(eta_list) > 0 else 0.0,
        'max_eta': float(eta_list.max()) if len(eta_list) > 0 else 0.0,
        'min_eta': float(eta_list.min()) if len(eta_list) > 0 else 0.0
    }

    # ===== 5. Prototype Analysis =====
    avg_proto_weights = proto_weights.mean(axis=0) if len(proto_weights.shape) > 1 else np.array([])
    top_proto_indices = np.argsort(avg_proto_weights)[::-1][:3].tolist() if len(avg_proto_weights) > 0 else []

    # Entropy: -Σ p·log(p)
    entropy = -np.sum(proto_weights * np.log(proto_weights + 1e-8), axis=1) if len(proto_weights.shape) > 1 else np.array([])

    prototype_analysis = {
        'most_used_prototypes': top_proto_indices,
        'prototype_avg_weights': avg_proto_weights.tolist() if len(avg_proto_weights) > 0 else [],
        'avg_entropy': float(entropy.mean()) if len(entropy) > 0 else 0.0,
        'max_entropy': float(entropy.max()) if len(entropy) > 0 else 0.0,
        'min_entropy': float(entropy.min()) if len(entropy) > 0 else 0.0
    }

    # ===== 6. Risk Metrics =====
    risk_metrics = {
        'VaR_5': float(metrics.get('var_5', 0.0)),
        'CVaR_5': float(metrics.get('cvar_5', 0.0)),
        'downside_deviation': float(metrics.get('downside_deviation', 0.0)),
        'avg_turnover': float(metrics.get('avg_turnover', 0.0))
    }

    # ===== 7. T-Cell Insights =====
    crisis_regime_pct = crisis_mask.sum() / len(crisis_mask) if len(crisis_mask) > 0 else 0.0

    # Crisis types: [n_steps, K] → K차원별 평균
    avg_crisis_types = crisis_types.mean(axis=0) if len(crisis_types.shape) > 1 else np.array([])
    top_crisis_type_indices = np.argsort(avg_crisis_types)[::-1][:3].tolist() if len(avg_crisis_types) > 0 else []

    tcell_insights = {
        'crisis_regime_pct': float(crisis_regime_pct),
        'top_crisis_types': top_crisis_type_indices,
        'avg_crisis_type_distribution': avg_crisis_types.tolist() if len(avg_crisis_types) > 0 else [],
        'avg_danger_level': float(crisis_levels.mean()) if len(crisis_levels) > 0 else 0.0
    }

    # ===== 종합 =====
    insights = {
        'summary': summary,
        'top_holdings': top_holdings,
        'crisis_vs_normal': crisis_vs_normal,
        'irt_decomposition': irt_decomposition,
        'prototype_analysis': prototype_analysis,
        'risk_metrics': risk_metrics,
        'tcell_insights': tcell_insights,
        'timestamp': str(pd.Timestamp.now())
    }

    return insights
