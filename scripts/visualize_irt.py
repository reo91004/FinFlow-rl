# scripts/visualize_irt.py

"""
IRT 시각화 스크립트 (DEPRECATED)

⚠️ DEPRECATED (v2.0.3): 이 스크립트는 더 이상 사용되지 않는다.
시각화는 이제 evaluate_irt.py에서 자동으로 생성된다.

대신 사용:
    python main.py --mode evaluate --resume logs/*/checkpoints/best_model.pth

또는:
    python scripts/evaluate_irt.py --checkpoint logs/*/checkpoints/best_model.pth

---

Legacy 사용법 (단독 실행):
python scripts/visualize_irt.py --results logs/YYYYMMDD_HHMMSS/evaluation/evaluation_results.json
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class IRTVisualizer:
    """IRT 결과 시각화"""

    def __init__(self, results_path: str):
        self.results_path = Path(results_path)

        # 결과 로드
        with open(self.results_path, 'r') as f:
            self.results = json.load(f)

        # 출력 디렉토리
        self.output_dir = self.results_path.parent / 'visualizations'
        self.output_dir.mkdir(exist_ok=True)

        # 스타일 설정
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)

    def plot_all(self):
        """모든 시각화 실행"""
        self.plot_returns()
        self.plot_portfolio_weights()
        self.plot_crisis_levels()
        self.plot_prototype_weights()
        print(f"모든 시각화 완료: {self.output_dir}")

    def plot_returns(self):
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
        axes[1].fill_between(range(len(cumulative_returns)), 0, cumulative_returns,
                            alpha=0.3)
        axes[1].set_title('Cumulative Returns', fontsize=14)
        axes[1].set_xlabel('Time Step')
        axes[1].set_ylabel('Cumulative Return')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'returns.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_portfolio_weights(self):
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

    def plot_crisis_levels(self):
        """위기 레벨 시각화"""
        crisis_levels = np.array(self.results['crisis_levels'])

        fig, ax = plt.subplots(figsize=(14, 6))

        ax.plot(crisis_levels, linewidth=2, color='red', label='Crisis Level')
        ax.fill_between(range(len(crisis_levels)), 0, crisis_levels,
                       alpha=0.3, color='red')

        # 임계값 표시
        ax.axhline(0.7, color='darkred', linestyle='--', alpha=0.7,
                  label='High Crisis Threshold')
        ax.axhline(0.3, color='orange', linestyle='--', alpha=0.7,
                  label='Medium Crisis Threshold')

        ax.set_title('Crisis Level Detection', fontsize=14)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Crisis Level')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'crisis_levels.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_prototype_weights(self):
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
        axes[1].fill_between(range(len(entropy)), 0, entropy,
                           alpha=0.3, color='purple')

        axes[1].set_title('Prototype Diversity (Entropy)', fontsize=14)
        axes[1].set_xlabel('Time Step')
        axes[1].set_ylabel('Entropy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'prototype_weights.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='IRT Visualization Script')
    parser.add_argument('--results', type=str, required=True,
                       help='Path to evaluation results JSON file')
    args = parser.parse_args()

    visualizer = IRTVisualizer(args.results)
    visualizer.plot_all()

if __name__ == '__main__':
    main()