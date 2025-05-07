"""
데이터 및 결과 시각화 모듈

학습 과정 및 결과를 다양한 시각적 형태로 표현하는 기능을 제공합니다.
학습 곡선, 포트폴리오 성과, 자산 배분, 특성 중요도 등을 시각화합니다.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime

from src.constants import FEATURE_NAMES, STOCK_TICKERS, RESULTS_BASE_PATH

def plot_portfolio_performance(portfolio_values, test_dates, benchmark_values=None, benchmark_name=None, save_path=None):
    """
    포트폴리오 성과를 시각화합니다.
    
    Args:
        portfolio_values (np.ndarray): 포트폴리오 가치 배열
        test_dates (pd.DatetimeIndex): 테스트 날짜 인덱스
        benchmark_values (np.ndarray, optional): 벤치마크 가치 배열
        benchmark_name (str, optional): 벤치마크 이름
        save_path (str, optional): 그래프 저장 경로
    """
    plt.figure(figsize=(12, 6))
    
    # 날짜 형식 확인 및 변환
    if isinstance(test_dates[0], str):
        dates = pd.to_datetime(test_dates)
    else:
        dates = test_dates
    
    # 포트폴리오 가치 그래프
    plt.plot(dates, portfolio_values, 'b-', linewidth=2, label='Portfolio')
    
    # 벤치마크 추가 (있을 경우)
    if benchmark_values is not None:
        plt.plot(dates, benchmark_values, 'r-', linewidth=1.5, alpha=0.8, 
                 label=benchmark_name if benchmark_name else 'Benchmark')
    
    # 그래프 스타일링
    plt.title('Portfolio Performance', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Value (₩)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best', fontsize=12)
    
    # x축 날짜 포맷 설정
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # 3개월 간격
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # 저장 또는 출력
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_asset_weights_history(weights_history, test_dates, asset_names=None, save_path=None):
    """
    시간에 따른 자산 가중치 변화를 시각화합니다.
    
    Args:
        weights_history (np.ndarray): 자산 가중치 배열 (시간 x 자산)
        test_dates (pd.DatetimeIndex): 테스트 날짜 인덱스
        asset_names (list, optional): 자산 이름 리스트
        save_path (str, optional): 그래프 저장 경로
    """
    plt.figure(figsize=(12, 8))
    
    # 날짜 변환
    if isinstance(test_dates[0], str):
        dates = pd.to_datetime(test_dates)
    else:
        dates = test_dates
    
    # 데이터프레임 생성
    n_assets = weights_history.shape[1]
    
    if asset_names is None:
        # 자산 이름이 없으면 임의 생성
        asset_names = [f"Asset {i+1}" for i in range(n_assets-1)] + ["Cash"]
    elif len(asset_names) < n_assets-1:
        # 리스트 길이가 부족하면 나머지 채우기
        asset_names = list(asset_names) + [f"Asset {i+1}" for i in range(len(asset_names), n_assets-1)]
        asset_names.append("Cash")
    
    df = pd.DataFrame(weights_history, index=dates, columns=asset_names)
    
    # 스택 영역 그래프 그리기
    ax = df.plot.area(stacked=True, alpha=0.7, figsize=(12, 8))
    
    # 그래프 스타일링
    plt.title('Asset Allocation Over Time', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Allocation', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # y축 범위 설정
    plt.ylim(0, 1)
    
    # x축 날짜 설정
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    # 범례 위치 조정
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
              ncol=min(5, len(asset_names)), fontsize=10)
    
    plt.tight_layout()
    
    # 저장 또는 출력
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_daily_returns(daily_returns, test_dates, title="Daily Returns", save_path=None):
    """
    일별 수익률을 시각화합니다.
    
    Args:
        daily_returns (np.ndarray): 일별 수익률 배열
        test_dates (pd.DatetimeIndex): 테스트 날짜 인덱스
        title (str): 그래프 제목
        save_path (str, optional): 그래프 저장 경로
    """
    plt.figure(figsize=(12, 6))
    
    # 날짜 변환
    if isinstance(test_dates[0], str):
        dates = pd.to_datetime(test_dates)
    else:
        dates = test_dates
    
    # 수익률 퍼센트 변환
    returns_pct = daily_returns * 100
    
    # 막대 그래프 색상 설정 (양수: 초록, 음수: 빨강)
    colors = ['green' if r >= 0 else 'red' for r in returns_pct]
    
    # 막대 그래프 그리기
    plt.bar(dates, returns_pct, color=colors, alpha=0.7, width=2)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # 그래프 스타일링
    plt.title(title, fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Return (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    # x축 날짜 설정
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # 저장 또는 출력
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_cumulative_returns(cumulative_returns, test_dates, benchmark_cum_returns=None, benchmark_name=None, save_path=None):
    """
    누적 수익률을 시각화합니다.
    
    Args:
        cumulative_returns (np.ndarray): 누적 수익률 배열
        test_dates (pd.DatetimeIndex): 테스트 날짜 인덱스
        benchmark_cum_returns (np.ndarray, optional): 벤치마크 누적 수익률 배열
        benchmark_name (str, optional): 벤치마크 이름
        save_path (str, optional): 그래프 저장 경로
    """
    plt.figure(figsize=(12, 6))
    
    # 날짜 변환
    if isinstance(test_dates[0], str):
        dates = pd.to_datetime(test_dates)
    else:
        dates = test_dates
    
    # 누적 수익률 퍼센트 변환
    cum_returns_pct = cumulative_returns * 100
    
    # 포트폴리오 누적 수익률 그래프
    plt.plot(dates, cum_returns_pct, 'b-', linewidth=2, label='Portfolio')
    
    # 벤치마크 추가 (있을 경우)
    if benchmark_cum_returns is not None:
        benchmark_returns_pct = benchmark_cum_returns * 100
        plt.plot(dates, benchmark_returns_pct, 'r-', linewidth=1.5, alpha=0.8, 
                 label=benchmark_name if benchmark_name else 'Benchmark')
    
    # 그래프 스타일링
    plt.title('Cumulative Return', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best', fontsize=12)
    
    # x축 날짜 설정
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    # 0% 수평선 추가
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    
    # 저장 또는 출력
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_drawdowns(portfolio_values, test_dates, save_path=None):
    """
    포트폴리오 드로다운을 시각화합니다.
    
    Args:
        portfolio_values (np.ndarray): 포트폴리오 가치 배열
        test_dates (pd.DatetimeIndex): 테스트 날짜 인덱스
        save_path (str, optional): 그래프 저장 경로
    """
    plt.figure(figsize=(12, 6))
    
    # 날짜 변환
    if isinstance(test_dates[0], str):
        dates = pd.to_datetime(test_dates)
    else:
        dates = test_dates
    
    # 최대 누적 가치 및 드로다운 계산
    cum_max = np.maximum.accumulate(portfolio_values)
    drawdowns = (cum_max - portfolio_values) / cum_max
    drawdowns_pct = drawdowns * 100
    
    # 드로다운 그래프
    plt.fill_between(dates, 0, -drawdowns_pct, alpha=0.3, color='red')
    plt.plot(dates, -drawdowns_pct, 'r-', linewidth=1.5)
    
    # 그래프 스타일링
    plt.title('Portfolio Drawdowns', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Drawdown (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # x축 날짜 설정
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    # y축 반전 (하향 = 손실)
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    
    # 저장 또는 출력
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_training_history(reward_history, loss_history=None, val_reward_history=None, save_path=None):
    """
    학습 과정 이력을 시각화합니다.
    
    Args:
        reward_history (list): 에피소드별 보상 이력
        loss_history (list, optional): 손실 이력
        val_reward_history (list, optional): 검증 보상 이력
        save_path (str, optional): 그래프 저장 경로
    """
    if loss_history is None and val_reward_history is None:
        # 보상만 있을 경우 단일 그래프
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, len(reward_history) + 1), reward_history, 'b-', linewidth=1.5)
        plt.title('Training Reward History', fontsize=16)
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Reward', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
    else:
        # 여러 그래프 필요한 경우
        fig, axs = plt.subplots(2 if loss_history else 1, 1, figsize=(12, 10 if loss_history else 6))
        
        # 보상 그래프
        ax1 = axs[0] if loss_history else axs
        ax1.plot(range(1, len(reward_history) + 1), reward_history, 'b-', linewidth=1.5, label='Training')
        
        # 검증 보상 추가 (있을 경우)
        if val_reward_history:
            # [(episode, reward), ...] 형식일 경우
            if isinstance(val_reward_history[0], tuple) or isinstance(val_reward_history[0], list):
                episodes, rewards = zip(*val_reward_history)
                ax1.plot(episodes, rewards, 'r-', linewidth=1.5, label='Validation')
            else:
                val_episodes = [i * (len(reward_history) // len(val_reward_history)) for i in range(1, len(val_reward_history) + 1)]
                ax1.plot(val_episodes, val_reward_history, 'r-', linewidth=1.5, label='Validation')
        
        ax1.set_title('Reward History', fontsize=16)
        ax1.set_xlabel('Episode', fontsize=12)
        ax1.set_ylabel('Reward', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(loc='best')
        
        # 손실 그래프 (있을 경우)
        if loss_history:
            ax2 = axs[1]
            ax2.plot(range(1, len(loss_history) + 1), loss_history, 'g-', linewidth=1.5)
            ax2.set_title('Loss History', fontsize=16)
            ax2.set_xlabel('Update Step', fontsize=12)
            ax2.set_ylabel('Loss', fontsize=12)
            ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
    
    # 저장 또는 출력
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_feature_importance(feature_importance, feature_names=None, save_path=None):
    """
    XAI 분석 결과 특성 중요도를 시각화합니다.
    
    Args:
        feature_importance (np.ndarray): 특성 중요도 배열
        feature_names (list, optional): 특성 이름 리스트
        save_path (str, optional): 그래프 저장 경로
    """
    plt.figure(figsize=(10, 8))
    
    # 특성 이름이 없으면 기본값 사용
    if feature_names is None:
        feature_names = FEATURE_NAMES if len(feature_importance) == len(FEATURE_NAMES) else [f"Feature {i+1}" for i in range(len(feature_importance))]
    
    # 중요도 기준 정렬
    sorted_idx = np.argsort(feature_importance)
    sorted_importance = feature_importance[sorted_idx]
    sorted_names = [feature_names[i] for i in sorted_idx]
    
    # 수평 막대 그래프
    plt.barh(range(len(sorted_importance)), sorted_importance, align='center', alpha=0.8)
    plt.yticks(range(len(sorted_importance)), sorted_names)
    
    # 그래프 스타일링
    plt.title('Feature Importance', fontsize=16)
    plt.xlabel('Importance', fontsize=12)
    plt.tight_layout()
    
    # 저장 또는 출력
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_feature_correlation(data, feature_names=None, save_path=None):
    """
    특성 간 상관관계를 히트맵으로 시각화합니다.
    
    Args:
        data (np.ndarray): 특성 데이터 배열
        feature_names (list, optional): 특성 이름 리스트
        save_path (str, optional): 그래프 저장 경로
    """
    plt.figure(figsize=(12, 10))
    
    # 특성 이름이 없으면 기본값 사용
    if feature_names is None:
        feature_names = FEATURE_NAMES if data.shape[1] == len(FEATURE_NAMES) else [f"Feature {i+1}" for i in range(data.shape[1])]
    
    # 데이터프레임으로 변환
    df = pd.DataFrame(data, columns=feature_names)
    
    # 상관계수 계산
    corr = df.corr()
    
    # 히트맵 그리기
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
    
    # 그래프 스타일링
    plt.title('Feature Correlation', fontsize=16)
    plt.tight_layout()
    
    # 저장 또는 출력
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_monthly_returns_heatmap(daily_returns, test_dates, save_path=None):
    """
    월별 수익률 히트맵을 그립니다.
    
    Args:
        daily_returns (np.ndarray): 일별 수익률 배열
        test_dates (pd.DatetimeIndex): 테스트 날짜 인덱스
        save_path (str, optional): 그래프 저장 경로
    """
    plt.figure(figsize=(12, 8))
    
    # 날짜 및 수익률 데이터프레임 생성
    dates = pd.to_datetime(test_dates)
    returns_df = pd.DataFrame(daily_returns, index=dates, columns=['return'])
    
    # 연도 및 월 추출
    returns_df['year'] = returns_df.index.year
    returns_df['month'] = returns_df.index.month
    
    # 월별 누적 수익률 계산
    monthly_returns = returns_df.groupby(['year', 'month'])['return'].apply(
        lambda x: (1 + x).prod() - 1
    ).reset_index()
    
    # 피벗 테이블 생성
    pivoted = monthly_returns.pivot(index='year', columns='month', values='return')
    
    # 월 이름 설정
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    pivoted.columns = [month_names[i-1] for i in pivoted.columns]
    
    # 히트맵 그리기
    sns.heatmap(pivoted * 100, annot=True, fmt=".1f", cmap='RdYlGn', center=0,
                linewidths=.5, cbar_kws={'label': 'Monthly Return (%)'})
    
    # 그래프 스타일링
    plt.title('Monthly Returns (%)', fontsize=16)
    plt.tight_layout()
    
    # 저장 또는 출력
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def create_performance_dashboard(backtest_result, benchmark_results=None, save_dir=None):
    """
    백테스팅 결과에 대한 종합 성능 대시보드를 생성합니다.
    
    Args:
        backtest_result (dict): 백테스팅 결과 딕셔너리
        benchmark_results (dict, optional): 벤치마크 결과 딕셔너리
        save_dir (str, optional): 대시보드 저장 디렉토리
    """
    # 저장 디렉토리 생성
    if save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(RESULTS_BASE_PATH, f"dashboard_{timestamp}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 포트폴리오 가치 그래프
    portfolio_values = backtest_result["portfolio_values"]
    test_dates = backtest_result["test_dates"]
    
    if benchmark_results:
        # 첫 번째 벤치마크 사용
        benchmark_name, benchmark_data = next(iter(benchmark_results.items()))
        benchmark_values = benchmark_data["values"]
        plot_portfolio_performance(portfolio_values, test_dates, benchmark_values, benchmark_name,
                                save_path=os.path.join(save_dir, "portfolio_performance.png"))
    else:
        plot_portfolio_performance(portfolio_values, test_dates,
                                save_path=os.path.join(save_dir, "portfolio_performance.png"))
    
    # 2. 누적 수익률 그래프
    cum_returns = backtest_result["cumulative_returns"]
    
    if benchmark_results:
        benchmark_name, benchmark_data = next(iter(benchmark_results.items()))
        benchmark_cum_returns = np.cumprod(1 + np.array(benchmark_data["returns"])) - 1
        plot_cumulative_returns(cum_returns, test_dates, benchmark_cum_returns, benchmark_name,
                             save_path=os.path.join(save_dir, "cumulative_returns.png"))
    else:
        plot_cumulative_returns(cum_returns, test_dates,
                             save_path=os.path.join(save_dir, "cumulative_returns.png"))
    
    # 3. 일별 수익률 그래프
    daily_returns = backtest_result["daily_returns"]
    plot_daily_returns(daily_returns, test_dates,
                      save_path=os.path.join(save_dir, "daily_returns.png"))
    
    # 4. 드로다운 그래프
    plot_drawdowns(portfolio_values, test_dates,
                  save_path=os.path.join(save_dir, "drawdowns.png"))
    
    # 5. 자산 배분 그래프
    weights = backtest_result["portfolio_weights"]
    plot_asset_weights_history(weights, test_dates, STOCK_TICKERS,
                             save_path=os.path.join(save_dir, "asset_allocation.png"))
    
    # 6. 월별 수익률 히트맵
    plot_monthly_returns_heatmap(daily_returns, test_dates,
                               save_path=os.path.join(save_dir, "monthly_returns_heatmap.png"))
    
    # 7. 성능 지표 테이블 (텍스트 파일)
    from src.evaluation.evaluation import create_performance_table
    
    perf_table = create_performance_table(backtest_result, benchmark_results,
                                        save_path=os.path.join(save_dir, "performance_metrics.txt"))
    
    return save_dir 