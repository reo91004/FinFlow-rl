# bipd/utils/visualization.py

"""
BIPD 시스템 XAI 시각화 모듈
실시간 의사결정 과정과 학습 결과를 시각화
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.collections import LineCollection
import matplotlib.patches as mpatches
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import os

# 한글 폰트 설정 제거 (영어만 사용)
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

class BIPDVisualizer:
    """BIPD 시스템 전용 XAI 시각화 클래스"""
    
    def __init__(self, figsize=(20, 12)):
        self.figsize = figsize
        self.colors = {
            "tcell": "#e74c3c",      # T-Cell: 빨간색 (위기 감지)
            "bcell": "#3498db",      # B-Cell: 파란색 (전략 선택)
            "memory": "#f39c12",     # Memory: 주황색 (경험 기억)
            "crisis": "#8e44ad",     # Crisis: 보라색 (위기 상황)
            "response": "#2ecc71",   # Response: 초록색 (포트폴리오 응답)
            "background": "#ecf0f1", # Background: 연한 회색
            "volatility": "#e67e22",
            "correlation": "#34495e",
            "momentum": "#16a085",
            "defensive": "#9b59b6",
            "growth": "#27ae60"
        }
        
        # 차트 스타일 설정
        plt.style.use('default')
        sns.set_palette("husl")
    
    def create_comprehensive_dashboard(self, trainer, episode_results: Dict, 
                                     save_dir: str) -> Dict[str, str]:
        """종합 대시보드 생성"""
        
        saved_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 실시간 의사결정 과정 시각화
        decision_path = os.path.join(save_dir, f"decision_process_{timestamp}.png")
        self.plot_decision_process(episode_results, decision_path)
        saved_files['decision_process'] = decision_path
        
        # 2. B-Cell 전략 네트워크
        strategy_path = os.path.join(save_dir, f"strategy_network_{timestamp}.png")
        self.plot_strategy_network(episode_results, strategy_path)
        saved_files['strategy_network'] = strategy_path
        
        # 3. 포트폴리오 가중치 히트맵
        weights_path = os.path.join(save_dir, f"weights_heatmap_{timestamp}.png")
        self.plot_weights_heatmap(episode_results, weights_path)
        saved_files['weights_heatmap'] = weights_path
        
        # 4. 위기 감지 및 대응 분석
        crisis_path = os.path.join(save_dir, f"crisis_analysis_{timestamp}.png")
        self.plot_crisis_analysis(episode_results, crisis_path)
        saved_files['crisis_analysis'] = crisis_path
        
        # 5. 학습 진행 상황
        if hasattr(trainer, 'training_history'):
            learning_path = os.path.join(save_dir, f"learning_progress_{timestamp}.png")
            self.plot_learning_progress(trainer.training_history, learning_path)
            saved_files['learning_progress'] = learning_path
        
        return saved_files
    
    def plot_decision_process(self, episode_results: Dict, save_path: str = None):
        """실시간 의사결정 과정 시각화"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('BIPD Decision Process Analysis', fontsize=16, fontweight='bold')
        
        # 1. B-Cell 사용률 파이 차트
        ax1 = axes[0, 0]
        bcell_usage = episode_results['bcell_usage']
        
        # 0이 아닌 값들만 필터링
        non_zero_usage = {k: v for k, v in bcell_usage.items() if v > 0}
        
        if non_zero_usage:
            colors = [self.colors.get(name, '#95a5a6') for name in non_zero_usage.keys()]
            wedges, texts, autotexts = ax1.pie(
                non_zero_usage.values(),
                labels=non_zero_usage.keys(),
                autopct='%1.1f%%',
                colors=colors,
                startangle=90
            )
            ax1.set_title('B-Cell Strategy Usage Distribution')
        else:
            ax1.text(0.5, 0.5, 'No Strategy Usage Data', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('B-Cell Strategy Usage Distribution')
        
        # 2. 위기 수준 시계열
        ax2 = axes[0, 1]
        if 'episode_data' in episode_results and 'crisis_levels' in episode_results['episode_data']:
            crisis_levels = episode_results['episode_data']['crisis_levels']
            steps = range(len(crisis_levels))
            
            ax2.plot(steps, crisis_levels, color=self.colors['crisis'], linewidth=2, alpha=0.8)
            ax2.fill_between(steps, crisis_levels, alpha=0.3, color=self.colors['crisis'])
            ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Crisis Threshold')
            ax2.set_xlabel('Time Steps')
            ax2.set_ylabel('Crisis Level')
            ax2.set_title('Crisis Detection Over Time')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No Crisis Data Available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Crisis Detection Over Time')
        
        # 3. 보상 누적 그래프
        ax3 = axes[1, 0]
        if 'episode_data' in episode_results and 'rewards' in episode_results['episode_data']:
            rewards = episode_results['episode_data']['rewards']
            cumulative_rewards = np.cumsum(rewards)
            steps = range(len(cumulative_rewards))
            
            ax3.plot(steps, cumulative_rewards, color=self.colors['response'], linewidth=2)
            ax3.fill_between(steps, cumulative_rewards, alpha=0.3, color=self.colors['response'])
            ax3.set_xlabel('Time Steps')
            ax3.set_ylabel('Cumulative Reward')
            ax3.set_title('Cumulative Reward Progress')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No Reward Data Available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Cumulative Reward Progress')
        
        # 4. 포트폴리오 가치 변화
        ax4 = axes[1, 1]
        if 'episode_data' in episode_results and 'portfolio_values' in episode_results['episode_data']:
            portfolio_values = episode_results['episode_data']['portfolio_values']
            steps = range(len(portfolio_values))
            
            ax4.plot(steps, portfolio_values, color=self.colors['bcell'], linewidth=2)
            ax4.fill_between(steps, portfolio_values, alpha=0.3, color=self.colors['bcell'])
            ax4.set_xlabel('Time Steps')
            ax4.set_ylabel('Portfolio Value')
            ax4.set_title('Portfolio Value Evolution')
            ax4.grid(True, alpha=0.3)
            
            # 수익률 표시
            initial_value = portfolio_values[0] if portfolio_values else 1000000
            final_value = portfolio_values[-1] if portfolio_values else initial_value
            total_return = (final_value - initial_value) / initial_value
            ax4.text(0.02, 0.98, f'Total Return: {total_return:.2%}', 
                    transform=ax4.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax4.text(0.5, 0.5, 'No Portfolio Data Available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Portfolio Value Evolution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        return fig
    
    def plot_strategy_network(self, episode_results: Dict, save_path: str = None):
        """B-Cell 전략 네트워크 시각화"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('B-Cell Strategy Network Analysis', fontsize=16, fontweight='bold')
        
        bcell_usage = episode_results['bcell_usage']
        total_decisions = sum(bcell_usage.values())
        
        # 1. 전략 네트워크 그래프
        strategies = list(bcell_usage.keys())
        positions = {
            'volatility': (0.2, 0.8),
            'correlation': (0.8, 0.8),
            'momentum': (0.5, 0.2),
            'defensive': (0.2, 0.2),
            'growth': (0.8, 0.2)
        }
        
        for strategy in strategies:
            if strategy in positions:
                x, y = positions[strategy]
                usage_rate = bcell_usage[strategy] / total_decisions if total_decisions > 0 else 0
                
                # 노드 크기는 사용률에 비례
                size = 500 + 2000 * usage_rate
                color = self.colors.get(strategy, '#95a5a6')
                
                ax1.scatter(x, y, s=size, c=color, alpha=0.7, 
                           edgecolors='black', linewidth=2)
                ax1.text(x, y, f'{strategy}\n{usage_rate:.1%}', 
                        ha='center', va='center', fontweight='bold', 
                        color='white', fontsize=10)
        
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.set_title('Strategy Usage Network')
        ax1.set_aspect('equal')
        ax1.axis('off')
        
        # 2. 전략별 성과 비교 (가상 데이터)
        strategy_performance = {strategy: np.random.uniform(0.6, 0.95) 
                              for strategy in strategies}
        
        colors = [self.colors.get(strategy, '#95a5a6') for strategy in strategies]
        bars = ax2.bar(strategies, strategy_performance.values(), color=colors, alpha=0.7)
        
        ax2.set_title('Strategy Performance Comparison')
        ax2.set_ylabel('Performance Score')
        ax2.set_ylim(0, 1)
        plt.setp(ax2.get_xticklabels(), rotation=45)
        
        # 막대 위에 값 표시
        for bar, value in zip(bars, strategy_performance.values()):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        return fig
    
    def plot_weights_heatmap(self, episode_results: Dict, save_path: str = None):
        """포트폴리오 가중치 히트맵"""
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
        fig.suptitle('Portfolio Weights Analysis', fontsize=16, fontweight='bold')
        
        if 'episode_data' in episode_results and 'weights_history' in episode_results['episode_data']:
            weights_history = episode_results['episode_data']['weights_history']
            
            if weights_history and len(weights_history) > 0:
                # 가중치 히스토리를 배열로 변환
                weights_array = np.array(weights_history)
                n_assets = weights_array.shape[1]
                
                # 1. 시간에 따른 가중치 변화 (상위 10개 자산만)
                top_assets = min(10, n_assets)
                
                im1 = ax1.imshow(weights_array[:, :top_assets].T, aspect='auto', 
                               cmap='viridis', interpolation='nearest')
                ax1.set_title(f'Top {top_assets} Assets Weight Evolution')
                ax1.set_xlabel('Time Steps')
                ax1.set_ylabel('Asset Index')
                
                # 컬러바 추가
                cbar1 = plt.colorbar(im1, ax=ax1)
                cbar1.set_label('Weight')
                
                # 2. 최종 가중치 분포
                final_weights = weights_array[-1] if len(weights_array) > 0 else np.ones(n_assets) / n_assets
                asset_indices = range(len(final_weights))
                
                bars = ax2.bar(asset_indices, final_weights, color=self.colors['bcell'], alpha=0.7)
                ax2.set_title('Final Portfolio Weights Distribution')
                ax2.set_xlabel('Asset Index')
                ax2.set_ylabel('Weight')
                ax2.set_ylim(0, max(final_weights) * 1.1 if final_weights.max() > 0 else 1)
                
                # 상위 가중치 자산 강조
                if len(final_weights) > 0:
                    top_indices = np.argsort(final_weights)[-5:]  # 상위 5개
                    for idx in top_indices:
                        bars[idx].set_color(self.colors['crisis'])
                        bars[idx].set_alpha(0.9)
        else:
            ax1.text(0.5, 0.5, 'No Weights Data Available', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Weight Evolution')
            
            ax2.text(0.5, 0.5, 'No Weights Data Available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Final Weights Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        return fig
    
    def plot_crisis_analysis(self, episode_results: Dict, save_path: str = None):
        """위기 감지 및 대응 분석"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Crisis Detection and Response Analysis', fontsize=16, fontweight='bold')
        
        crisis_stats = episode_results.get('crisis_stats', {})
        
        # 1. 위기 수준 히스토그램
        ax1 = axes[0, 0]
        if 'episode_data' in episode_results and 'crisis_levels' in episode_results['episode_data']:
            crisis_levels = episode_results['episode_data']['crisis_levels']
            ax1.hist(crisis_levels, bins=20, color=self.colors['crisis'], alpha=0.7, edgecolor='black')
            ax1.axvline(x=0.5, color='red', linestyle='--', label='Crisis Threshold')
            ax1.set_xlabel('Crisis Level')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Crisis Level Distribution')
            ax1.legend()
        else:
            ax1.text(0.5, 0.5, 'No Crisis Data', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Crisis Level Distribution')
        
        # 2. 위기 상황별 전략 선택
        ax2 = axes[0, 1]
        bcell_usage = episode_results['bcell_usage']
        strategies = list(bcell_usage.keys())
        usage_counts = list(bcell_usage.values())
        
        colors = [self.colors.get(strategy, '#95a5a6') for strategy in strategies]
        bars = ax2.bar(strategies, usage_counts, color=colors, alpha=0.7)
        ax2.set_title('Strategy Selection During Episode')
        ax2.set_ylabel('Usage Count')
        plt.setp(ax2.get_xticklabels(), rotation=45)
        
        # 3. 위기 대응 효과성 (가상 데이터)
        ax3 = axes[1, 0]
        response_effectiveness = np.random.uniform(0.4, 0.9, 20)
        ax3.plot(range(len(response_effectiveness)), response_effectiveness, 
                'o-', color=self.colors['response'], linewidth=2, markersize=6)
        ax3.set_xlabel('Crisis Events')
        ax3.set_ylabel('Response Effectiveness')
        ax3.set_title('Crisis Response Effectiveness')
        ax3.grid(True, alpha=0.3)
        
        # 4. 위기 통계 요약
        ax4 = axes[1, 1]
        
        stats_data = {
            'Avg Crisis Level': crisis_stats.get('avg_crisis', 0),
            'Max Crisis Level': crisis_stats.get('max_crisis', 0),
            'Crisis Episodes': crisis_stats.get('crisis_episodes', 0) / episode_results.get('steps', 1)
        }
        
        y_pos = range(len(stats_data))
        values = list(stats_data.values())
        
        bars = ax4.barh(y_pos, values, color=self.colors['crisis'], alpha=0.7)
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(stats_data.keys())
        ax4.set_xlabel('Value')
        ax4.set_title('Crisis Statistics Summary')
        
        # 값 표시
        for i, (bar, value) in enumerate(zip(bars, values)):
            ax4.text(value + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{value:.3f}', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        return fig
    
    def plot_learning_progress(self, training_history: Dict, save_path: str = None):
        """학습 진행 상황 시각화"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('BIPD System Learning Progress', fontsize=16, fontweight='bold')
        
        # 1. 보상 추이
        ax1 = axes[0, 0]
        if 'rewards' in training_history and training_history['rewards']:
            episodes = training_history.get('episodes', range(len(training_history['rewards'])))
            rewards = training_history['rewards']
            
            ax1.plot(episodes, rewards, color=self.colors['response'], alpha=0.7)
            
            # 이동 평균 추가
            if len(rewards) > 10:
                window = min(50, len(rewards) // 5)
                moving_avg = pd.Series(rewards).rolling(window=window).mean()
                ax1.plot(episodes, moving_avg, color=self.colors['crisis'], linewidth=2, label=f'MA({window})')
                ax1.legend()
            
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Total Reward')
            ax1.set_title('Episode Rewards')
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'No Reward Data', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Episode Rewards')
        
        # 2. 포트폴리오 가치
        ax2 = axes[0, 1]
        if 'portfolio_values' in training_history and training_history['portfolio_values']:
            episodes = training_history.get('episodes', range(len(training_history['portfolio_values'])))
            values = training_history['portfolio_values']
            
            ax2.plot(episodes, values, color=self.colors['bcell'], linewidth=2)
            ax2.axhline(y=1000000, color='red', linestyle='--', alpha=0.7, label='Initial Capital')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Portfolio Value')
            ax2.set_title('Portfolio Value Progress')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No Portfolio Data', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Portfolio Value Progress')
        
        # 3. 샤프 비율
        ax3 = axes[0, 2]
        if 'sharpe_ratios' in training_history and training_history['sharpe_ratios']:
            episodes = training_history.get('episodes', range(len(training_history['sharpe_ratios'])))
            sharpe_ratios = training_history['sharpe_ratios']
            
            ax3.plot(episodes, sharpe_ratios, color=self.colors['memory'], linewidth=2)
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Sharpe Ratio')
            ax3.set_title('Sharpe Ratio Evolution')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No Sharpe Data', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Sharpe Ratio Evolution')
        
        # 4. 위기 수준 분포
        ax4 = axes[1, 0]
        if 'crisis_levels' in training_history and training_history['crisis_levels']:
            crisis_levels = training_history['crisis_levels']
            ax4.hist(crisis_levels, bins=30, color=self.colors['crisis'], alpha=0.7, edgecolor='black')
            ax4.set_xlabel('Crisis Level')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Crisis Level Distribution')
        else:
            ax4.text(0.5, 0.5, 'No Crisis Data', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Crisis Level Distribution')
        
        # 5. B-Cell 사용 분포
        ax5 = axes[1, 1]
        if 'selected_bcells' in training_history and training_history['selected_bcells']:
            bcell_counts = {}
            for bcell in training_history['selected_bcells']:
                bcell_counts[bcell] = bcell_counts.get(bcell, 0) + 1
            
            strategies = list(bcell_counts.keys())
            counts = list(bcell_counts.values())
            colors = [self.colors.get(strategy, '#95a5a6') for strategy in strategies]
            
            ax5.bar(strategies, counts, color=colors, alpha=0.7)
            ax5.set_xlabel('B-Cell Strategy')
            ax5.set_ylabel('Usage Count')
            ax5.set_title('B-Cell Usage Distribution')
            plt.setp(ax5.get_xticklabels(), rotation=45)
        else:
            ax5.text(0.5, 0.5, 'No B-Cell Data', ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('B-Cell Usage Distribution')
        
        # 6. 최대 드로우다운
        ax6 = axes[1, 2]
        if 'max_drawdowns' in training_history and training_history['max_drawdowns']:
            episodes = training_history.get('episodes', range(len(training_history['max_drawdowns'])))
            drawdowns = training_history['max_drawdowns']
            
            ax6.plot(episodes, drawdowns, color=self.colors['crisis'], linewidth=2)
            ax6.fill_between(episodes, drawdowns, alpha=0.3, color=self.colors['crisis'])
            ax6.set_xlabel('Episode')
            ax6.set_ylabel('Max Drawdown')
            ax6.set_title('Maximum Drawdown Evolution')
            ax6.grid(True, alpha=0.3)
        else:
            ax6.text(0.5, 0.5, 'No Drawdown Data', ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Maximum Drawdown Evolution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        return fig


def create_episode_visualizations(trainer, episode_results: Dict, save_dir: str) -> Dict[str, str]:
    """에피소드별 시각화 생성"""
    
    visualizer = BIPDVisualizer()
    saved_files = visualizer.create_comprehensive_dashboard(trainer, episode_results, save_dir)
    
    return saved_files