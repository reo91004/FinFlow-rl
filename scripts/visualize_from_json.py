# scripts/visualize_from_json.py

"""
JSON 기반 시각화 재생성 스크립트

evaluation_results.json을 읽어서 14개의 IRT 시각화를 재생성한다.
평가 없이 시각화만 다시 생성 가능하다.

사용법:
    python scripts/visualize_from_json.py --results logs/irt/20251004_*/evaluation_results.json
    python scripts/visualize_from_json.py --results logs/irt/.../evaluation_results.json --output custom_viz/

의존성: evaluation_results.json (train_irt.py 또는 evaluate.py 출력)
"""

import argparse
import json
import numpy as np
from pathlib import Path
import sys

# 상위 디렉토리 import
sys.path.append(str(Path(__file__).parent.parent))

from finrl.evaluation.visualizer import (
    plot_irt_decomposition,
    plot_portfolio_weights,
    plot_crisis_levels,
    plot_prototype_weights,
    plot_stock_analysis,
    plot_performance_timeline,
    plot_benchmark_comparison,
    plot_risk_dashboard,
    plot_tcell_analysis,
    plot_attribution_analysis,
    plot_cost_matrix,
    plot_portfolio_value,
    plot_returns,
    plot_drawdown
)


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
        """모든 시각화 생성"""
        print("\n시각화를 생성합니다...")

        # numpy 배열로 변환
        results_np = {}
        for key, value in self.results.items():
            if isinstance(value, list):
                try:
                    results_np[key] = np.array(value)
                except:
                    results_np[key] = value
            else:
                results_np[key] = value

        # IRT 데이터 확인
        has_irt = all(k in results_np for k in ['w_rep', 'w_ot', 'prototype_weights'])

        # 기본 시각화 (항상 생성)
        plot_count = 0

        # 1. Portfolio Value
        if 'values' in results_np:
            plot_portfolio_value(
                results_np['values'],
                save_path=str(self.output_dir / "portfolio_value.png")
            )
            plot_count += 1
            print(f"  [{plot_count}] portfolio_value.png 생성")

        # 2. Returns
        if 'returns' in results_np:
            plot_returns(
                results_np['returns'],
                save_path=str(self.output_dir / "returns_distribution.png")
            )
            plot_count += 1
            print(f"  [{plot_count}] returns_distribution.png 생성")

        # 3. Drawdown
        if 'values' in results_np:
            plot_drawdown(
                results_np['values'],
                save_path=str(self.output_dir / "drawdown.png")
            )
            plot_count += 1
            print(f"  [{plot_count}] drawdown.png 생성")

        # IRT 특화 시각화 (IRT 데이터가 있는 경우만)
        if has_irt:
            print("\nIRT 시각화 생성 중...")

            # 4. IRT Decomposition
            if 'w_rep' in results_np and 'w_ot' in results_np:
                plot_irt_decomposition(
                    results_np['w_rep'],
                    results_np['w_ot'],
                    save_path=str(self.output_dir / "irt_decomposition.png")
                )
                plot_count += 1
                print(f"  [{plot_count}] irt_decomposition.png 생성")

            # 5. Portfolio Weights
            if 'weights' in results_np:
                plot_portfolio_weights(
                    results_np['weights'],
                    results_np.get('symbols', None),
                    save_path=str(self.output_dir / "portfolio_weights.png")
                )
                plot_count += 1
                print(f"  [{plot_count}] portfolio_weights.png 생성")

            # 6. Crisis Levels
            if 'crisis_levels' in results_np:
                plot_crisis_levels(
                    results_np['crisis_levels'],
                    save_path=str(self.output_dir / "crisis_levels.png")
                )
                plot_count += 1
                print(f"  [{plot_count}] crisis_levels.png 생성")

            # 7. Prototype Weights
            if 'prototype_weights' in results_np:
                plot_prototype_weights(
                    results_np['prototype_weights'],
                    save_path=str(self.output_dir / "prototype_weights.png")
                )
                plot_count += 1
                print(f"  [{plot_count}] prototype_weights.png 생성")

            # 8. Stock Analysis
            if 'weights' in results_np and 'symbols' in results_np:
                plot_stock_analysis(
                    results_np['weights'],
                    results_np['symbols'],
                    save_path=str(self.output_dir / "stock_analysis.png")
                )
                plot_count += 1
                print(f"  [{plot_count}] stock_analysis.png 생성")

            # 9. Performance Timeline
            if 'returns' in results_np:
                plot_performance_timeline(
                    results_np['returns'],
                    save_path=str(self.output_dir / "performance_timeline.png")
                )
                plot_count += 1
                print(f"  [{plot_count}] performance_timeline.png 생성")

            # 10. Benchmark Comparison
            if 'returns' in results_np:
                benchmark_returns = results_np.get('benchmark_returns', None)
                plot_benchmark_comparison(
                    results_np['returns'],
                    benchmark_returns=benchmark_returns,
                    save_path=str(self.output_dir / "benchmark_comparison.png")
                )
                plot_count += 1
                print(f"  [{plot_count}] benchmark_comparison.png 생성")

            # 11. Risk Dashboard
            if 'returns' in results_np and 'metrics' in results_np:
                plot_risk_dashboard(
                    results_np['returns'],
                    results_np['metrics'],
                    save_path=str(self.output_dir / "risk_dashboard.png")
                )
                plot_count += 1
                print(f"  [{plot_count}] risk_dashboard.png 생성")

            # 12. T-Cell Analysis
            if 'crisis_types' in results_np and 'crisis_levels' in results_np:
                plot_tcell_analysis(
                    results_np['crisis_types'],
                    results_np['crisis_levels'],
                    save_path=str(self.output_dir / "tcell_analysis.png")
                )
                plot_count += 1
                print(f"  [{plot_count}] tcell_analysis.png 생성")

            # 13. Attribution Analysis
            if 'weights' in results_np and 'symbols' in results_np and 'returns' in results_np:
                plot_attribution_analysis(
                    results_np['weights'],
                    results_np['returns'],
                    results_np['symbols'],
                    save_path=str(self.output_dir / "attribution_analysis.png")
                )
                plot_count += 1
                print(f"  [{plot_count}] attribution_analysis.png 생성")

            # 14. Cost Matrix
            if 'cost_matrices' in results_np:
                plot_cost_matrix(
                    results_np['cost_matrices'],
                    save_path=str(self.output_dir / "cost_matrix.png")
                )
                plot_count += 1
                print(f"  [{plot_count}] cost_matrix.png 생성")

        print(f"\n총 {plot_count}개 시각화 완료: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate visualizations from evaluation_results.json',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python scripts/visualize_from_json.py --results logs/irt/20251004_120000/evaluation_results.json
  python scripts/visualize_from_json.py --results logs/irt/.../evaluation_results.json --output custom_viz/
        """
    )

    parser.add_argument('--results', type=str, required=True,
                       help='Path to evaluation_results.json')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for visualizations (default: same dir as JSON + "_regenerated")')

    args = parser.parse_args()

    # 시각화 생성
    visualizer = JSONVisualizer(args.results, args.output)
    visualizer.create_all_visualizations()


if __name__ == '__main__':
    main()
