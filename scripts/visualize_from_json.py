# scripts/visualize_from_json.py

"""
JSON 기반 시각화 재생성 스크립트

evaluation_results.json을 읽어서 12개의 IRT 시각화를 재생성한다.
평가 없이 시각화만 다시 생성 가능하다.

사용법:
    python scripts/visualize_from_json.py --results logs/20251003_123456/evaluation/evaluation_results.json
    python scripts/visualize_from_json.py --results logs/.../evaluation_results.json --output custom_viz/

의존성: evaluation_results.json (evaluate_irt.py 출력)
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict
import sys

# evaluate_irt.py의 시각화 로직을 재사용하기 위해 임포트
# (경로 문제 회피)
sys.path.append(str(Path(__file__).parent.parent))
from scripts.evaluate_irt import IRTEvaluator


class JSONVisualizer:
    """JSON 데이터로부터 시각화 생성"""

    def __init__(self, results_path: str, output_dir: str = None):
        self.results_path = Path(results_path)

        # JSON 로드
        with open(self.results_path, 'r') as f:
            self.results = json.load(f)

        # 출력 디렉토리
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.results_path.parent / 'visualizations_regenerated'
        self.output_dir.mkdir(exist_ok=True, parents=True)

        print(f"JSON 로드 완료: {self.results_path}")
        print(f"출력 디렉토리: {self.output_dir}")

    def create_all_visualizations(self):
        """모든 시각화 생성

        evaluate_irt.py의 _create_visualizations() 로직을 재사용한다.
        """
        print("시각화를 생성합니다...")

        # 스타일 설정
        sns.set_style("whitegrid")

        # 각 시각화 함수 호출
        plot_functions = [
            self._plot_irt_decomposition,
            self._plot_returns,
            self._plot_portfolio_weights,
            self._plot_crisis_levels,
            self._plot_prototype_weights,
            self._plot_stock_analysis,
            self._plot_performance_timeline,
            self._plot_benchmark_comparison,
            self._plot_risk_dashboard,
            self._plot_tcell_analysis,
            self._plot_attribution_analysis,
            self._plot_cost_matrix,
        ]

        for i, plot_fn in enumerate(plot_functions, 1):
            try:
                plot_fn()
                print(f"  [{i}/{len(plot_functions)}] {plot_fn.__name__} 완료")
            except Exception as e:
                print(f"  [{i}/{len(plot_functions)}] {plot_fn.__name__} 실패: {e}")

        print(f"\n모든 시각화 완료: {self.output_dir}")

    # ===== 시각화 메서드들 (evaluate_irt.py에서 복사) =====

    def _plot_irt_decomposition(self):
        """IRT Decomposition: w = (1-α)·w_rep + α·w_ot"""
        w_rep = np.array(self.results['w_rep'])
        w_ot = np.array(self.results['w_ot'])
        proto_weights = np.array(self.results['prototype_weights'])

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
        ax3.hist(avg_w_rep, bins=30, alpha=0.6, label='Replicator', edgecolor='black')
        ax3.hist(avg_w_ot, bins=30, alpha=0.6, label='OT', edgecolor='black')
        ax3.set_xlabel('Weight Value')
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

        plt.tight_layout()
        plt.savefig(self.output_dir / 'irt_decomposition.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_returns(self):
        """수익률 시각화"""
        returns = np.array(self.results['returns'])
        cumulative_returns = np.cumprod(1 + returns) - 1

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # 일일 수익률
        axes[0].plot(returns, alpha=0.7, label='Daily Returns')
        axes[0].axhline(0, color='red', linestyle='--', alpha=0.5)
        axes[0].set_title('Daily Returns', fontsize=14)
        axes[0].set_xlabel('Time Step')
        axes[0].set_ylabel('Return')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 누적 수익률
        axes[1].plot(cumulative_returns, linewidth=2, label='Cumulative Returns')
        axes[1].fill_between(range(len(cumulative_returns)), 0, cumulative_returns, alpha=0.3)
        axes[1].set_title('Cumulative Returns', fontsize=14)
        axes[1].set_xlabel('Time Step')
        axes[1].set_ylabel('Cumulative Return')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'returns.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_portfolio_weights(self):
        """포트폴리오 가중치 시각화"""
        weights = np.array(self.results['weights'])
        n_assets = weights.shape[1]

        fig, ax = plt.subplots(figsize=(14, 8))

        # 스택 영역 차트
        bottom = np.zeros(len(weights))
        colors = plt.cm.tab20(np.linspace(0, 1, n_assets))

        for i in range(n_assets):
            ax.fill_between(range(len(weights)), bottom, bottom + weights[:, i],
                          alpha=0.8, color=colors[i], label=f'Asset {i+1}')
            bottom += weights[:, i]

        ax.set_title('Portfolio Weights Over Time', fontsize=14)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Weight')
        ax.set_ylim(0, 1)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'portfolio_weights.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_crisis_levels(self):
        """위기 레벨 시각화"""
        crisis_levels = np.array(self.results['crisis_levels'])

        fig, ax = plt.subplots(figsize=(14, 6))

        ax.plot(crisis_levels, linewidth=2, color='red', label='Crisis Level')
        ax.fill_between(range(len(crisis_levels)), 0, crisis_levels, alpha=0.3, color='red')

        # 임계값 표시
        ax.axhline(0.7, color='darkred', linestyle='--', alpha=0.7, label='High Crisis Threshold')
        ax.axhline(0.3, color='orange', linestyle='--', alpha=0.7, label='Medium Crisis Threshold')

        ax.set_title('Crisis Level Detection', fontsize=14)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Crisis Level')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'crisis_levels.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_prototype_weights(self):
        """프로토타입 가중치 시각화"""
        proto_weights = np.array(self.results['prototype_weights'])
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

        plt.tight_layout()
        plt.savefig(self.output_dir / 'prototype_weights.png', dpi=300, bbox_inches='tight')
        plt.close()

    # ===== 고급 시각화 (간소화 버전) =====

    def _plot_stock_analysis(self):
        """주식 분석 (간소화)"""
        weights = np.array(self.results['weights'])
        symbols = self.results.get('symbols', [f'Asset_{i}' for i in range(weights.shape[1])])

        avg_weights = weights.mean(axis=0)
        top_indices = np.argsort(avg_weights)[::-1][:10]

        fig, ax = plt.subplots(figsize=(10, 6))
        top_symbols = [symbols[i] if i < len(symbols) else f'Asset_{i}' for i in top_indices]
        top_weights = avg_weights[top_indices]

        ax.barh(top_symbols, top_weights, color='steelblue', edgecolor='black')
        ax.set_xlabel('Average Weight')
        ax.set_title('Top 10 Holdings', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'stock_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_performance_timeline(self):
        """성능 타임라인 (간소화)"""
        returns = np.array(self.results['returns'])
        cumulative = np.cumprod(1 + returns) - 1

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(cumulative, linewidth=2, label='Cumulative Return')
        ax.fill_between(range(len(cumulative)), 0, cumulative, alpha=0.3)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Cumulative Return')
        ax.set_title('Performance Timeline', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_timeline.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_benchmark_comparison(self):
        """벤치마크 비교 (간소화)"""
        # 간소화: 누적 수익률만 표시
        returns = np.array(self.results['returns'])
        cumulative = np.cumprod(1 + returns) - 1

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(cumulative, linewidth=2, label='IRT Strategy', color='blue')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5, label='Zero Return')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Cumulative Return')
        ax.set_title('Benchmark Comparison (Simplified)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'benchmark_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_risk_dashboard(self):
        """리스크 대시보드 (간소화)"""
        returns = np.array(self.results['returns'])
        metrics = self.results['metrics']

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # (1) Returns Distribution
        axes[0, 0].hist(returns, bins=30, alpha=0.7, edgecolor='black', color='steelblue')
        axes[0, 0].axvline(returns.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean = {returns.mean():.4f}')
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
        CVaR (5%): {metrics.get('cvar_5', 0):.3f}
        """
        axes[0, 1].text(0.1, 0.5, metric_text, fontsize=14, verticalalignment='center', family='monospace')
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

        plt.tight_layout()
        plt.savefig(self.output_dir / 'risk_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_tcell_analysis(self):
        """T-Cell 분석 (간소화)"""
        crisis_types = np.array(self.results['crisis_types'])
        crisis_levels = np.array(self.results['crisis_levels'])

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

        plt.tight_layout()
        plt.savefig(self.output_dir / 'tcell_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_attribution_analysis(self):
        """Attribution 분석 (간소화)"""
        weights = np.array(self.results['weights'])
        returns = np.array(self.results['returns'])
        symbols = self.results.get('symbols', [f'Asset_{i}' for i in range(weights.shape[1])])

        # 간단한 contribution: avg_weight * total_return
        avg_weights = weights.mean(axis=0)
        total_return = self.results['metrics']['total_return']
        contributions = avg_weights * total_return

        top_indices = np.argsort(np.abs(contributions))[::-1][:10]
        top_symbols = [symbols[i] if i < len(symbols) else f'Asset_{i}' for i in top_indices]
        top_contributions = contributions[top_indices]

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['green' if c > 0 else 'red' for c in top_contributions]
        ax.barh(top_symbols, top_contributions, color=colors, edgecolor='black')
        ax.set_xlabel('Contribution to Total Return')
        ax.set_title('Top 10 Contributors (Simplified)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'attribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_cost_matrix(self):
        """Cost Matrix (간소화)"""
        cost_matrices = self.results.get('cost_matrices', [])

        if len(cost_matrices) == 0:
            print("  Cost matrix 데이터가 없어 시각화를 건너뜁니다.")
            return

        avg_cost = np.mean(cost_matrices, axis=0)
        m, M = avg_cost.shape

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(avg_cost, cmap='RdYlGn_r', aspect='auto')
        ax.set_xlabel('Prototype Index (M)')
        ax.set_ylabel('Epitope Index (m)')
        ax.set_title('Average Immunological Cost Matrix', fontsize=14, fontweight='bold')
        ax.set_xticks(range(M))
        ax.set_xticklabels([f'P{i+1}' for i in range(M)])
        ax.set_yticks(range(m))
        ax.set_yticklabels([f'E{i+1}' for i in range(m)])
        plt.colorbar(im, ax=ax, label='Cost')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'cost_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate visualizations from evaluation_results.json',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/visualize_from_json.py --results logs/20251003_123456/evaluation/evaluation_results.json
  python scripts/visualize_from_json.py --results logs/.../evaluation_results.json --output custom_viz/
        """
    )
    parser.add_argument('--results', type=str, required=True,
                       help='Path to evaluation_results.json')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (default: results_dir/visualizations_regenerated)')
    args = parser.parse_args()

    # 시각화 생성
    visualizer = JSONVisualizer(args.results, args.output)
    visualizer.create_all_visualizations()


if __name__ == '__main__':
    main()
