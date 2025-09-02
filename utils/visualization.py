# bipd/utils/visualization.py

"""
BIPD 시스템 XAI 시각화 모듈
생물학적 면역 시스템 메타포를 활용한 설명 가능한 AI 시각화
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
from matplotlib.collections import LineCollection
import matplotlib.patches as mpatches
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import os
import json

# 한글 폰트 설정 제거 (영어만 사용)
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

class BIPDVisualizer:
    """BIPD 시스템 전용 XAI 시각화 클래스
    
    생물학적 면역 시스템의 메타포를 활용하여 포트폴리오 의사결정 과정을 
    직관적이고 설명 가능한 방식으로 시각화
    """
    
    def __init__(self, figsize=(20, 12)):
        self.figsize = figsize
        self.colors = {
            "tcell": "#e74c3c",      # T-Cell: 빨간색 (위기 감지)
            "bcell": "#3498db",      # B-Cell: 파란색 (전략 실행)
            "memory": "#f39c12",     # Memory: 주황색 (경험 저장)
            "normal": "#27ae60",     # 정상 상태: 초록색
            "warning": "#f1c40f",    # 경고 상태: 노란색
            "crisis": "#e74c3c",     # 위기 상태: 빨간색
            "background": "#ecf0f1"  # 배경색
        }
        
        # 위기 레벨에 따른 색상 맵
        self.crisis_colors = {
            "low": self.colors["normal"],
            "medium": self.colors["warning"], 
            "high": self.colors["crisis"]
        }
        
    def create_decision_explanation_dashboard(self, decision_data: Dict, save_path: str) -> str:
        """포트폴리오 의사결정 과정 설명 대시보드 생성
        
        Args:
            decision_data: 의사결정 데이터 (T-Cell, B-Cell, Memory 정보 포함)
            save_path: 저장 경로
        """
        
        fig = plt.figure(figsize=self.figsize)
        
        # 전체 레이아웃: 2x3 그리드
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. T-Cell 위기 감지 현황
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_tcell_detection(ax1, decision_data.get('tcell_data', {}))
        
        # 2. B-Cell 전략 선택 과정
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_bcell_selection(ax2, decision_data.get('bcell_data', {}))
        
        # 3. Memory Cell 유사 경험 참조
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_memory_influence(ax3, decision_data.get('memory_data', {}))
        
        # 4. 포트폴리오 가중치 분배
        ax4 = fig.add_subplot(gs[1, 2:])
        self._plot_portfolio_allocation(ax4, decision_data.get('portfolio_data', {}))
        
        # 5. 종합 의사결정 흐름도
        ax5 = fig.add_subplot(gs[2, :])
        self._plot_decision_flow(ax5, decision_data)
        
        # 전체 제목
        fig.suptitle(f"BIPD Decision Explanation Dashboard - Step {decision_data.get('step', 0)}", 
                    fontsize=16, fontweight='bold')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def _plot_tcell_detection(self, ax, tcell_data: Dict):
        """T-Cell 위기 감지 현황 시각화"""
        
        crisis_types = ['Volatility', 'Correlation', 'Volume', 'Overall']
        crisis_levels = [
            tcell_data.get('volatility_crisis', 0),
            tcell_data.get('correlation_crisis', 0), 
            tcell_data.get('volume_crisis', 0),
            tcell_data.get('overall_crisis', 0)
        ]
        
        # 방사형 차트
        angles = np.linspace(0, 2 * np.pi, len(crisis_types), endpoint=False).tolist()
        crisis_levels += crisis_levels[:1]  # 원형 완성
        angles += angles[:1]
        
        ax.plot(angles, crisis_levels, 'o-', linewidth=2, color=self.colors['tcell'])
        ax.fill(angles, crisis_levels, alpha=0.25, color=self.colors['tcell'])
        
        # 임계값 선들
        for threshold, label, color in [(0.4, 'Medium', 'orange'), (0.7, 'High', 'red')]:
            threshold_line = [threshold] * len(angles)
            ax.plot(angles, threshold_line, '--', alpha=0.7, color=color, label=f'{label} threshold')
        
        ax.set_ylim(0, 1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(crisis_types)
        ax.set_title('T-Cell: Crisis Detection Status', fontweight='bold', color=self.colors['tcell'])
        ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
        ax.grid(True, alpha=0.3)
    
    def _plot_bcell_selection(self, ax, bcell_data: Dict):
        """B-Cell 전략 선택 과정 시각화"""
        
        strategies = list(bcell_data.get('strategy_scores', {}).keys())
        scores = list(bcell_data.get('strategy_scores', {}).values())
        selected_strategy = bcell_data.get('selected_strategy', '')
        
        if not strategies:
            strategies = ['Volatility', 'Correlation', 'Momentum', 'Defensive', 'Growth']
            scores = [0.2] * 5
            selected_strategy = 'Defensive'
        
        # 막대 차트
        colors = [self.colors['bcell'] if s == selected_strategy else '#bdc3c7' for s in strategies]
        bars = ax.bar(strategies, scores, color=colors, alpha=0.8)
        
        # 선택된 전략 강조
        for i, (strategy, bar) in enumerate(zip(strategies, bars)):
            if strategy == selected_strategy:
                bar.set_edgecolor('red')
                bar.set_linewidth(3)
                # 선택 이유 표시
                ax.annotate('SELECTED', 
                           xy=(i, scores[i]), 
                           xytext=(i, scores[i] + 0.15),
                           ha='center', fontweight='bold', color='red',
                           arrowprops=dict(arrowstyle='->', color='red'))
        
        ax.set_ylim(0, max(scores) * 1.3 if scores else 1)
        ax.set_ylabel('Strategy Score')
        ax.set_title('B-Cell: Strategy Selection Process', fontweight='bold', color=self.colors['bcell'])
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_memory_influence(self, ax, memory_data: Dict):
        """Memory Cell 유사 경험 참조 시각화"""
        
        similar_episodes = memory_data.get('similar_episodes', [])
        similarity_scores = memory_data.get('similarity_scores', [])
        
        if not similar_episodes:
            # 더미 데이터
            similar_episodes = [f'Episode {i}' for i in [245, 187, 312, 89, 156]]
            similarity_scores = [0.85, 0.78, 0.72, 0.69, 0.64]
        
        # 유사도 점수 막대 차트
        bars = ax.barh(similar_episodes, similarity_scores, color=self.colors['memory'], alpha=0.7)
        
        # 유사도가 높은 순서로 정렬되었다고 가정하고 그라데이션 효과
        for i, bar in enumerate(bars):
            alpha = 0.9 - (i * 0.15)  # 점진적으로 투명도 증가
            bar.set_alpha(max(alpha, 0.3))
        
        ax.set_xlim(0, 1)
        ax.set_xlabel('Similarity Score')
        ax.set_title('Memory Cell: Similar Past Experiences', fontweight='bold', color=self.colors['memory'])
        ax.grid(True, alpha=0.3, axis='x')
        
        # 가장 유사한 경험 강조
        if similar_episodes:
            ax.text(similarity_scores[0] + 0.02, 0, 'Most Similar', 
                   va='center', fontweight='bold', color=self.colors['memory'])
    
    def _plot_portfolio_allocation(self, ax, portfolio_data: Dict):
        """포트폴리오 가중치 분배 시각화"""
        
        weights = portfolio_data.get('weights', {})
        top_holdings = portfolio_data.get('top_holdings', [])
        
        if not weights:
            # 더미 데이터 (상위 10개 종목)
            symbols = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA', 'JPM', 'JNJ', 'V']
            weights = {symbol: np.random.uniform(0.05, 0.15) for symbol in symbols}
            top_holdings = list(weights.keys())[:5]
        
        if weights:
            # 상위 종목들만 표시
            sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:10]
            symbols, weight_values = zip(*sorted_weights)
            
            # 색상 구분 (상위 5개는 진한 색)
            colors = [self.colors['bcell'] if symbol in top_holdings[:5] 
                     else '#95a5a6' for symbol in symbols]
            
            bars = ax.bar(symbols, weight_values, color=colors, alpha=0.8)
            
            # 상위 종목 강조
            for i, (symbol, bar) in enumerate(zip(symbols, bars)):
                if symbol in top_holdings[:3]:  # 상위 3개만 라벨
                    ax.text(i, weight_values[i] + 0.005, f'{weight_values[i]:.1%}', 
                           ha='center', va='bottom', fontweight='bold')
            
            ax.set_ylabel('Portfolio Weight')
            ax.set_title('Portfolio Allocation Decision', fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
            
            # 총합 표시
            total_weight = sum(weight_values)
            ax.text(0.02, 0.98, f'Total: {total_weight:.1%}', 
                   transform=ax.transAxes, va='top', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def _plot_decision_flow(self, ax, decision_data: Dict):
        """종합 의사결정 흐름도 시각화"""
        
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.axis('off')
        
        # 플로우 단계들
        steps = [
            (1, 4, "Market\nData", self.colors['background']),
            (3, 4, "T-Cell\nCrisis Detection", self.colors['tcell']),
            (5, 5, "B-Cell\nStrategy Selection", self.colors['bcell']),
            (5, 3, "Memory Cell\nSimilar Experience", self.colors['memory']),
            (7, 4, "Portfolio\nOptimization", self.colors['normal']),
            (9, 4, "Final\nAllocation", self.colors['bcell'])
        ]
        
        # 박스들 그리기
        for x, y, label, color in steps:
            box = FancyBboxPatch((x-0.4, y-0.3), 0.8, 0.6,
                               boxstyle="round,pad=0.1",
                               facecolor=color, alpha=0.7,
                               edgecolor='black', linewidth=1)
            ax.add_patch(box)
            ax.text(x, y, label, ha='center', va='center', fontweight='bold', fontsize=10)
        
        # 화살표들
        arrows = [
            ((1.4, 4), (2.6, 4)),     # Market Data -> T-Cell
            ((3.4, 4), (4.6, 5)),     # T-Cell -> B-Cell
            ((3.4, 4), (4.6, 3)),     # T-Cell -> Memory
            ((5.4, 5), (6.6, 4.2)),   # B-Cell -> Portfolio
            ((5.4, 3), (6.6, 3.8)),   # Memory -> Portfolio
            ((7.4, 4), (8.6, 4))      # Portfolio -> Final
        ]
        
        for start, end in arrows:
            ax.annotate('', xy=end, xytext=start,
                       arrowprops=dict(arrowstyle='->', lw=2, color='black', alpha=0.7))
        
        # 현재 상태 정보
        current_crisis = decision_data.get('tcell_data', {}).get('overall_crisis', 0)
        selected_strategy = decision_data.get('bcell_data', {}).get('selected_strategy', 'Unknown')
        
        info_text = f"Current Crisis Level: {current_crisis:.2f}\nSelected Strategy: {selected_strategy}"
        ax.text(5, 1.5, info_text, ha='center', va='center', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
        
        ax.set_title('BIPD Decision Flow: Biological Immune System Metaphor', 
                    fontsize=14, fontweight='bold', pad=20)

    def create_training_progress_dashboard(self, training_history: Dict, save_path: str) -> str:
        """훈련 진행 상황 대시보드 생성"""
        
        fig, axes = plt.subplots(2, 3, figsize=self.figsize)
        axes = axes.flatten()
        
        # 1. 에피소드별 보상
        if 'rewards' in training_history:
            rewards = training_history['rewards']
            episodes = range(1, len(rewards) + 1)
            
            axes[0].plot(episodes, rewards, alpha=0.7, linewidth=1, color=self.colors['bcell'])
            if len(rewards) > 20:
                # 이동평균 추가
                window = min(50, len(rewards) // 10)
                moving_avg = pd.Series(rewards).rolling(window=window, center=True).mean()
                axes[0].plot(episodes, moving_avg, linewidth=2, color=self.colors['tcell'], label=f'MA({window})')
                axes[0].legend()
            
            axes[0].set_xlabel('Episode')
            axes[0].set_ylabel('Episode Reward')
            axes[0].set_title('Training Progress: Episode Rewards')
            axes[0].grid(True, alpha=0.3)
        
        # 2. T-Cell 위기 감지 비율
        if 'crisis_rates' in training_history:
            crisis_data = training_history['crisis_rates']
            episodes = list(crisis_data.keys())
            rates = list(crisis_data.values())
            
            axes[1].plot(episodes, rates, color=self.colors['tcell'], linewidth=2)
            axes[1].axhline(y=0.15, color='red', linestyle='--', alpha=0.7, label='Target Rate')
            axes[1].set_xlabel('Episode')
            axes[1].set_ylabel('Crisis Detection Rate')
            axes[1].set_title('T-Cell: Crisis Detection Rates')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        # 3. B-Cell 전략 사용 빈도
        if 'strategy_usage' in training_history:
            strategy_counts = training_history['strategy_usage']
            strategies = list(strategy_counts.keys())
            counts = list(strategy_counts.values())
            
            axes[2].pie(counts, labels=strategies, autopct='%1.1f%%', 
                       colors=[self.colors['bcell'], self.colors['memory'], 
                              self.colors['normal'], self.colors['warning'], 
                              self.colors['crisis']][:len(strategies)])
            axes[2].set_title('B-Cell: Strategy Usage Distribution')
        
        # 4. 포트폴리오 성과 지표
        if 'portfolio_metrics' in training_history:
            metrics = training_history['portfolio_metrics']
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())
            
            bars = axes[3].bar(metric_names, metric_values, color=self.colors['normal'], alpha=0.7)
            axes[3].set_ylabel('Metric Value')
            axes[3].set_title('Portfolio Performance Metrics')
            axes[3].tick_params(axis='x', rotation=45)
            
            # 값 표시
            for bar, value in zip(bars, metric_values):
                axes[3].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # 5. Memory Cell 활용도
        if 'memory_usage' in training_history:
            memory_data = training_history['memory_usage']
            episodes = list(memory_data.keys())
            usage_rates = list(memory_data.values())
            
            axes[4].plot(episodes, usage_rates, color=self.colors['memory'], linewidth=2)
            axes[4].set_xlabel('Episode')
            axes[4].set_ylabel('Memory Utilization Rate')
            axes[4].set_title('Memory Cell: Utilization Over Time')
            axes[4].grid(True, alpha=0.3)
        
        # 6. 시스템 안정성 지표
        if 'stability_metrics' in training_history:
            stability = training_history['stability_metrics']
            
            labels = ['Q-Value\nStability', 'Alpha\nStability', 'Loss\nConvergence', 
                     'Weight\nStability', 'Overall\nStability']
            values = [stability.get(key, 0.5) for key in 
                     ['q_stability', 'alpha_stability', 'loss_convergence', 
                      'weight_stability', 'overall_stability']]
            
            # 레이더 차트
            angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
            values += values[:1]
            angles += angles[:1]
            
            axes[5].plot(angles, values, 'o-', linewidth=2, color=self.colors['bcell'])
            axes[5].fill(angles, values, alpha=0.25, color=self.colors['bcell'])
            axes[5].set_ylim(0, 1)
            axes[5].set_xticks(angles[:-1])
            axes[5].set_xticklabels(labels, fontsize=9)
            axes[5].set_title('System Stability Metrics')
            axes[5].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path

    def create_comprehensive_dashboard(self, trainer, episode_results: Dict, save_dir: str) -> Dict[str, str]:
        """종합적인 XAI 대시보드 생성"""
        
        saved_files = {}
        
        try:
            # 1. 의사결정 과정 설명 대시보드
            if 'decision_data' in episode_results:
                decision_path = os.path.join(save_dir, 'decision_explanation.png')
                saved_files['decision'] = self.create_decision_explanation_dashboard(
                    episode_results['decision_data'], decision_path)
            
            # 2. 훈련 진행 상황 대시보드  
            if hasattr(trainer, 'training_history') and trainer.training_history:
                training_path = os.path.join(save_dir, 'training_progress.png')
                saved_files['training'] = self.create_training_progress_dashboard(
                    trainer.training_history, training_path)
            
            # 3. 기본 성과 차트
            performance_path = os.path.join(save_dir, 'performance_chart.png')
            perf_file = self._create_simple_performance_chart(episode_results, performance_path)
            if perf_file:
                saved_files['performance'] = perf_file
            
            # 4. 결과 데이터 JSON 저장
            results_path = os.path.join(save_dir, 'episode_results.json')
            saved_files['results_json'] = self._save_results_json(episode_results, results_path)
            
        except Exception as e:
            print(f"대시보드 생성 중 오류 발생: {e}")
        
        return saved_files
    
    def _create_simple_performance_chart(self, episode_results: Dict, save_path: str) -> Optional[str]:
        """간단한 성과 차트 생성"""
        
        try:
            rewards = episode_results.get('episode_rewards', episode_results.get('rewards', []))
            if len(rewards) == 0:
                return None
                
            fig, ax = plt.subplots(figsize=(12, 6))
            
            episodes = range(1, len(rewards) + 1)
            ax.plot(episodes, rewards, 'b-', alpha=0.7, linewidth=1)
            
            # 이동평균 추가
            if len(rewards) > 10:
                window = min(50, len(rewards) // 10)
                moving_avg = pd.Series(rewards).rolling(window=window, center=True).mean()
                ax.plot(episodes, moving_avg, 'r-', linewidth=2, label=f'Moving Average (w={window})')
                ax.legend()
            
            ax.set_xlabel('Episode')
            ax.set_ylabel('Reward')
            ax.set_title('BIPD Training Performance')
            ax.grid(True, alpha=0.3)
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return save_path
            
        except Exception as e:
            print(f"성과 차트 생성 실패: {e}")
            return None
    
    def _save_results_json(self, episode_results: Dict, save_path: str) -> str:
        """결과 데이터를 JSON으로 저장"""
        
        # NumPy 배열을 JSON 직렬화 가능하도록 변환
        serializable_results = {}
        for key, value in episode_results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif hasattr(value, 'item'):  # PyTorch/NumPy scalars
                serializable_results[key] = float(value.item())
            else:
                serializable_results[key] = value
        
        with open(save_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        return save_path


def create_episode_visualizations(trainer, episode_results: Dict, save_dir: str) -> Dict[str, str]:
    """에피소드별 시각화 생성"""
    
    visualizer = BIPDVisualizer()
    saved_files = visualizer.create_comprehensive_dashboard(trainer, episode_results, save_dir)
    
    return saved_files