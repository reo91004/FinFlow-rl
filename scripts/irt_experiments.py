# scripts/irt_experiments.py

"""
IRT Experiments Suite

모든 IRT 관련 실험, 분석, 시각화를 하나의 파일로 통합.

통합된 파일들:
- grid_search_reward.sh -> run_grid_search()
- run_3way_comparison.sh -> run_3way_comparison()
- run_ablation.sh -> run_ablation_study()
- analyze_grid_search.py -> analyze_grid_search()
- analyze_3way.py -> analyze_3way()
- visualize_alpha_transition.py -> visualize_alpha_transition()
"""

import json
import argparse
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# 데이터 분석 임포트
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 시각화용 PyTorch
import torch


# ===============================================================================
# SECTION 1: 실험 실행 함수들 (Shell 스크립트 기능)
# ===============================================================================


def run_grid_search(
    output_dir: str = "logs/grid_search",
    episodes: int = 50,
    lambda_t_values: List[float] = None,
    lambda_d_values: List[float] = None,
    lambda_dd_values: List[float] = None,
) -> None:
    """
    최적 다목표 보상 파라미터를 위한 그리드 서치.

    Original: grid_search_reward.sh
    """
    # 제공되지 않은 경우 기본값 사용
    if lambda_t_values is None:
        lambda_t_values = [0.002, 0.003, 0.004]
    if lambda_d_values is None:
        lambda_d_values = [0.02, 0.03, 0.04]
    if lambda_dd_values is None:
        lambda_dd_values = [0.05, 0.07, 0.09]

    # 출력 디렉토리 생성
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 로그 파일
    log_file = output_path / "grid_search.log"

    print("=" * 60)
    print("Grid Search for Multi-Objective Reward")
    print("=" * 60)

    total_combinations = (
        len(lambda_t_values) * len(lambda_d_values) * len(lambda_dd_values)
    )
    print(f"Total combinations: {total_combinations}")
    print(f"Episodes per test: {episodes}")
    print(f"Output directory: {output_dir}")
    print()

    # 결과 저장
    results_csv = output_path / "results.csv"
    results_data = []

    current = 0
    for lambda_t in lambda_t_values:
        for lambda_d in lambda_d_values:
            for lambda_dd in lambda_dd_values:
                current += 1

                # 하위 디렉토리 생성
                combo_dir = output_path / f"{lambda_t}_{lambda_d}_{lambda_dd}"
                combo_dir.mkdir(parents=True, exist_ok=True)

                print(
                    f"[{current}/{total_combinations}] Testing: λ_t={lambda_t}, λ_d={lambda_d}, λ_dd={lambda_dd}"
                )
                print(f"  Output: {combo_dir}")

                # 학습 실행
                cmd = [
                    "python",
                    "scripts/train_irt.py",
                    "--episodes",
                    str(episodes),
                    "--lambda-turnover",
                    str(lambda_t),
                    "--lambda-diversity",
                    str(lambda_d),
                    "--lambda-drawdown",
                    str(lambda_dd),
                    "--output",
                    str(combo_dir),
                    "--no-plot",
                ]

                train_log = combo_dir / "train.log"
                with open(train_log, "w") as f:
                    result = subprocess.run(cmd, stdout=f, stderr=f)

                if result.returncode == 0:
                    print("  ✅ Success")

                    # 메트릭 추출
                    json_file = combo_dir / "irt" / "evaluation_insights.json"
                    if json_file.exists():
                        with open(json_file) as f:
                            data = json.load(f)

                        sharpe = data.get("summary", {}).get("sharpe_ratio", 0)
                        turnover = (
                            data.get("risk_metrics", {}).get("avg_turnover", 0) * 100
                        )

                        print(f"  Sharpe ratio: {sharpe:.3f}")
                        print(f"  Turnover: {turnover:.1f}%")

                        results_data.append(
                            {
                                "lambda_turnover": lambda_t,
                                "lambda_diversity": lambda_d,
                                "lambda_drawdown": lambda_dd,
                                "sharpe_ratio": sharpe,
                                "turnover_pct": turnover,
                            }
                        )
                else:
                    print(f"  ❌ Failed (check {train_log})")

                print()

    # CSV로 결과 저장
    if results_data:
        df = pd.DataFrame(results_data)
        df.to_csv(results_csv, index=False)
        print(f"Results saved to {results_csv}")

    print("Grid search complete!")


def run_3way_comparison(
    output_dir: str = "logs/3way_comparison",
    episodes: int = 200,
    train_start: str = "2009-01-01",
    train_end: str = "2020-01-01",
    test_start: str = "2020-01-01",
    test_end: str = "2023-01-01",
) -> None:
    """
    3방향 모델 비교: 기준 SAC vs SAC+MO vs IRT+MO.

    Original: run_3way_comparison.sh
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    log_file = output_path / "comparison.log"

    print("=" * 60)
    print("3-Way Model Comparison Experiment")
    print("=" * 60)
    print(f"Episodes: {episodes}")
    print(f"Train: {train_start} to {train_end}")
    print(f"Test: {test_start} to {test_end}")
    print()

    experiments = [
        {
            "name": "Baseline SAC",
            "dir": "baseline_sac",
            "cmd": [
                "python",
                "scripts/train.py",
                "--model",
                "sac",
                "--episodes",
                str(episodes),
                "--train-start",
                train_start,
                "--train-end",
                train_end,
                "--test-start",
                test_start,
                "--test-end",
                test_end,
                "--output",
                str(output_path / "baseline_sac"),
            ],
        },
        {
            "name": "IRT + Multi-Objective",
            "dir": "irt_multiobjective",
            "cmd": [
                "python",
                "scripts/train_irt.py",
                "--episodes",
                str(episodes),
                "--train-start",
                train_start,
                "--train-end",
                train_end,
                "--test-start",
                test_start,
                "--test-end",
                test_end,
                "--lambda-turnover",
                "0.003",
                "--lambda-diversity",
                "0.03",
                "--lambda-drawdown",
                "0.07",
                "--output",
                str(output_path / "irt_multiobjective"),
            ],
        },
    ]

    # Note: SAC+MO는 수정된 학습 코드가 필요하므로 특별 처리 필요
    # 현재는 기준 SAC와 IRT+MO만 실행

    for i, exp in enumerate(experiments, 1):
        print(f"[{i}/{len(experiments)}] Running {exp['name']}...")
        print(f"  Output: {output_path / exp['dir']}")

        exp_log = output_path / f"{exp['dir']}.log"
        with open(exp_log, "w") as f:
            result = subprocess.run(exp["cmd"], stdout=f, stderr=f)

        if result.returncode == 0:
            print("  ✅ Success")

            # 메트릭 추출 시도
            json_paths = [
                output_path / exp["dir"] / "evaluation_insights.json",
                output_path / exp["dir"] / "sac" / "evaluation_insights.json",
                output_path / exp["dir"] / "irt" / "evaluation_insights.json",
            ]

            for json_file in json_paths:
                if json_file.exists():
                    with open(json_file) as f:
                        data = json.load(f)

                    sharpe = data.get("summary", {}).get("sharpe_ratio", 0)
                    returns = data.get("summary", {}).get("cumulative_return", 0)

                    print(f"  Sharpe Ratio: {sharpe:.3f}")
                    print(f"  Cumulative Return: {returns:.1f}%")
                    break
        else:
            print(f"  ❌ Failed (check {exp_log})")

        print()

    print("3-Way comparison complete!")


def run_ablation_study(
    output_dir: str = "logs/ablation_study",
    episodes: int = 100,
    train_start: str = "2009-01-01",
    train_end: str = "2020-01-01",
    test_start: str = "2020-01-01",
    test_end: str = "2023-01-01",
) -> None:
    """
    구성 요소 필요성에 대한 절제 연구.

    Original: run_ablation.sh
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("IRT Ablation Study")
    print("=" * 60)
    print(f"Episodes per test: {episodes}")
    print()

    # 연구 1: 프로토타입 개수 (M 값)
    print("Study 1: Prototype Count Ablation")
    print("-" * 40)

    prototype_results = []
    for M in [4, 6, 8]:
        print(f"Testing with M={M} prototypes...")

        output_subdir = output_path / f"M_{M}"

        cmd = [
            "python",
            "scripts/train_irt.py",
            "--episodes",
            str(episodes),
            "--train-start",
            train_start,
            "--train-end",
            train_end,
            "--test-start",
            test_start,
            "--test-end",
            test_end,
            "--M-proto",
            str(M),
            "--output",
            str(output_subdir),
            "--no-plot",
        ]

        log_file = output_path / f"M_{M}.log"
        with open(log_file, "w") as f:
            result = subprocess.run(cmd, stdout=f, stderr=f)

        if result.returncode == 0:
            print(f"  ✅ M={M}: Success")

            json_file = output_subdir / "irt" / "evaluation_insights.json"
            if json_file.exists():
                with open(json_file) as f:
                    data = json.load(f)

                sharpe = data.get("summary", {}).get("sharpe_ratio", 0)
                returns = data.get("summary", {}).get("cumulative_return", 0)

                print(f"    Sharpe: {sharpe:.3f}, Return: {returns:.1f}%")

                prototype_results.append(
                    {"M": M, "sharpe_ratio": sharpe, "cumulative_return": returns}
                )
        else:
            print(f"  ❌ M={M}: Failed")

    # 프로토타입 절제 결과 저장
    if prototype_results:
        df = pd.DataFrame(prototype_results)
        df.to_csv(output_path / "prototype_ablation.csv", index=False)

    print("\nAblation study complete!")


# ===============================================================================
# SECTION 2: 분석 함수들 (Python 분석 스크립트)
# ===============================================================================


def analyze_grid_search(
    input_dir: str, top_n: int = 5, create_plot: bool = True
) -> None:
    """
    그리드 서치 결과 분석.

    Original: analyze_grid_search.py
    """
    base_dir = Path(input_dir)
    if not base_dir.exists():
        print(f"Error: Directory {base_dir} not found")
        return

    # 결과 로드
    results = []

    # 먼저 CSV 파일 확인
    csv_file = base_dir / "results.csv"
    if csv_file.exists():
        df = pd.read_csv(csv_file)
    else:
        # 디렉토리에서 수집
        for exp_dir in base_dir.glob("*_*_*"):
            if not exp_dir.is_dir():
                continue

            parts = exp_dir.name.split("_")
            if len(parts) != 3:
                continue

            try:
                lambda_t, lambda_d, lambda_dd = map(float, parts)
            except ValueError:
                continue

            json_file = exp_dir / "irt" / "evaluation_insights.json"
            if not json_file.exists():
                continue

            with open(json_file) as f:
                data = json.load(f)

            results.append(
                {
                    "lambda_turnover": lambda_t,
                    "lambda_diversity": lambda_d,
                    "lambda_drawdown": lambda_dd,
                    "sharpe_ratio": data.get("summary", {}).get("sharpe_ratio", 0),
                    "turnover_pct": data.get("risk_metrics", {}).get("avg_turnover", 0)
                    * 100,
                    "max_drawdown": data.get("risk_metrics", {}).get("max_drawdown", 0),
                    "cumulative_return": data.get("summary", {}).get(
                        "cumulative_return", 0
                    ),
                }
            )

        df = pd.DataFrame(results)

    if df.empty:
        print("No results found")
        return

    # 최고 구성 찾기
    best_configs = df.nlargest(top_n, "sharpe_ratio")

    # 분석 출력
    print("\n" + "=" * 60)
    print("Grid Search Analysis Results")
    print("=" * 60)
    print(f"\nTotal configurations tested: {len(df)}")
    print(
        f"Sharpe ratio range: [{df['sharpe_ratio'].min():.3f}, {df['sharpe_ratio'].max():.3f}]"
    )

    print("\n" + "=" * 60)
    print(f"Top {top_n} Configurations")
    print("=" * 60)

    display_cols = [
        "lambda_turnover",
        "lambda_diversity",
        "lambda_drawdown",
        "sharpe_ratio",
        "turnover_pct",
    ]
    print("\n")
    print(best_configs[display_cols].to_string(index=False, float_format="%.3f"))

    # 최고 구성 상세 정보
    if not best_configs.empty:
        best = best_configs.iloc[0]

        print("\n" + "=" * 60)
        print("Best Configuration")
        print("=" * 60)
        print(f"λ_turnover:  {best['lambda_turnover']:.4f}")
        print(f"λ_diversity: {best['lambda_diversity']:.4f}")
        print(f"λ_drawdown:  {best['lambda_drawdown']:.4f}")
        print(f"\nPerformance:")
        print(f"  Sharpe ratio:     {best['sharpe_ratio']:.3f}")
        print(f"  Turnover:         {best['turnover_pct']:.1f}%")

        # 최고 구성 저장
        config_file = base_dir / "best_config.json"
        config = {
            "lambda_turnover": float(best["lambda_turnover"]),
            "lambda_diversity": float(best["lambda_diversity"]),
            "lambda_drawdown": float(best["lambda_drawdown"]),
            "sharpe_ratio": float(best["sharpe_ratio"]),
            "turnover_pct": float(best["turnover_pct"]),
        }

        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

        print(f"\nBest configuration saved to {config_file}")

    # 요청 시 히트맵 생성
    if create_plot and len(df) > 1:
        _create_grid_search_heatmap(df, base_dir)


def _create_grid_search_heatmap(df: pd.DataFrame, output_dir: Path):
    """그리드 서치 히트맵 생성 도우미 함수"""
    unique_dd = sorted(df["lambda_drawdown"].unique())
    n_plots = len(unique_dd)

    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))

    if n_plots == 1:
        axes = [axes]

    for i, dd in enumerate(unique_dd):
        ax = axes[i]

        df_dd = df[df["lambda_drawdown"] == dd]

        pivot = df_dd.pivot(
            index="lambda_diversity", columns="lambda_turnover", values="sharpe_ratio"
        )

        im = ax.imshow(pivot, cmap="RdYlGn", aspect="auto")

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{x:.3f}" for x in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{y:.3f}" for y in pivot.index])

        ax.set_xlabel("λ_turnover", fontsize=10)
        ax.set_ylabel("λ_diversity", fontsize=10)
        ax.set_title(f"λ_drawdown = {dd:.2f}", fontsize=11)

        # 셀에 값 추가
        for j in range(len(pivot.index)):
            for k in range(len(pivot.columns)):
                value = pivot.iloc[j, k]
                if not np.isnan(value):
                    color = "white" if value < pivot.values.mean() else "black"
                    ax.text(
                        k,
                        j,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=9,
                    )

        plt.colorbar(im, ax=ax, label="Sharpe Ratio")

    plt.suptitle(
        "Grid Search Results: Sharpe Ratio Heatmap", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()

    plot_file = output_dir / "grid_search_heatmap.png"
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    print(f"Heatmap saved to {plot_file}")
    plt.close()


def analyze_3way(input_dir: str, create_plot: bool = True) -> None:
    """
    3방향 비교 결과 분석.

    Original: analyze_3way.py
    """
    base_dir = Path(input_dir)
    if not base_dir.exists():
        print(f"Error: Directory {base_dir} not found")
        return

    # 세 실험에서 결과 로드
    configs = {
        "baseline_sac": "Baseline SAC",
        "sac_multiobjective": "SAC + Multi-Obj",
        "irt_multiobjective": "IRT + Multi-Obj",
    }

    results = {}

    for config_dir, config_name in configs.items():
        json_paths = [
            base_dir / config_dir / "evaluation_insights.json",
            base_dir / config_dir / "sac" / "evaluation_insights.json",
            base_dir / config_dir / "irt" / "evaluation_insights.json",
        ]

        json_file = None
        for path in json_paths:
            if path.exists():
                json_file = path
                break

        if json_file:
            with open(json_file) as f:
                data = json.load(f)

            results[config_name] = {
                "sharpe_ratio": data.get("summary", {}).get("sharpe_ratio", 0),
                "cumulative_return": data.get("summary", {}).get(
                    "cumulative_return", 0
                ),
                "max_drawdown": data.get("risk_metrics", {}).get("max_drawdown", 0),
                "avg_turnover": data.get("risk_metrics", {}).get("avg_turnover", 0),
                "win_rate": data.get("performance_metrics", {}).get("win_rate", 0),
                "profit_loss_ratio": data.get("performance_metrics", {}).get(
                    "profit_loss_ratio", 0
                ),
            }
        else:
            print(f"Warning: No results found for {config_name}")
            results[config_name] = {
                k: 0
                for k in [
                    "sharpe_ratio",
                    "cumulative_return",
                    "max_drawdown",
                    "avg_turnover",
                    "win_rate",
                    "profit_loss_ratio",
                ]
            }

    # 기여도 계산
    baseline_sharpe = results.get("Baseline SAC", {}).get("sharpe_ratio", 0)
    sac_mo_sharpe = results.get("SAC + Multi-Obj", {}).get("sharpe_ratio", 0)
    irt_mo_sharpe = results.get("IRT + Multi-Obj", {}).get("sharpe_ratio", 0)

    reward_contribution = sac_mo_sharpe - baseline_sharpe
    architecture_contribution = irt_mo_sharpe - sac_mo_sharpe
    total_contribution = irt_mo_sharpe - baseline_sharpe

    # 백분율 개선 계산
    reward_improvement = (
        ((sac_mo_sharpe / baseline_sharpe - 1) * 100) if baseline_sharpe != 0 else 0
    )
    architecture_improvement = (
        ((irt_mo_sharpe / sac_mo_sharpe - 1) * 100) if sac_mo_sharpe != 0 else 0
    )
    total_improvement = (
        ((irt_mo_sharpe / baseline_sharpe - 1) * 100) if baseline_sharpe != 0 else 0
    )

    # 분석 출력
    print("\n" + "=" * 70)
    print("3-WAY MODEL COMPARISON ANALYSIS")
    print("=" * 70)

    # 성능 테이블
    df = pd.DataFrame(results).T
    df = df[["sharpe_ratio", "cumulative_return", "max_drawdown", "avg_turnover"]]

    print("\n📊 Performance Metrics:")
    print("-" * 70)
    print(df.to_string(float_format="%.3f"))

    # 기여도 분석
    print("\n" + "=" * 70)
    print("CONTRIBUTION ANALYSIS")
    print("=" * 70)

    print(f"\n🎯 Sharpe Ratio Decomposition:")
    print("-" * 40)
    print(f"Baseline SAC:           {baseline_sharpe:.3f}")
    print(
        f"SAC + Multi-Objective:  {sac_mo_sharpe:.3f} (Δ = {reward_contribution:+.3f})"
    )
    print(
        f"IRT + Multi-Objective:  {irt_mo_sharpe:.3f} (Δ = {architecture_contribution:+.3f})"
    )

    print(f"\n📈 Contribution Breakdown:")
    print("-" * 40)
    print(
        f"Reward Design:      {reward_contribution:+.3f} ({reward_improvement:+.1f}%)"
    )
    print(
        f"IRT Architecture:   {architecture_contribution:+.3f} ({architecture_improvement:+.1f}%)"
    )
    print(f"Total Improvement:  {total_contribution:+.3f} ({total_improvement:+.1f}%)")

    # 요약 저장
    summary = {
        "configurations": results,
        "contributions": {
            "reward_contribution": reward_contribution,
            "architecture_contribution": architecture_contribution,
            "total_contribution": total_contribution,
            "reward_improvement": reward_improvement,
            "architecture_improvement": architecture_improvement,
            "total_improvement": total_improvement,
        },
    }

    summary_file = base_dir / "3way_comparison_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to {summary_file}")

    # 요청 시 플롯 생성
    if create_plot:
        _create_3way_comparison_plots(results, base_dir)


def _create_3way_comparison_plots(results: dict, output_dir: Path):
    """3방향 비교 플롯 생성 도우미 함수"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    configs = list(results.keys())

    metrics = [
        ("sharpe_ratio", "Sharpe Ratio"),
        ("cumulative_return", "Cumulative Return (%)"),
        ("max_drawdown", "Max Drawdown (%)"),
        ("avg_turnover", "Avg Turnover (%)"),
        ("win_rate", "Win Rate (%)"),
        ("profit_loss_ratio", "Profit/Loss Ratio"),
    ]

    for idx, (metric_key, metric_name) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]

        values = [results[config].get(metric_key, 0) for config in configs]

        bars = ax.bar(
            range(len(configs)), values, color=["blue", "green", "red"], alpha=0.7
        )

        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{value:.2f}" if abs(value) < 100 else f"{value:.0f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(configs, rotation=45, ha="right")
        ax.set_title(metric_name, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # 최고 성능자 강조
        best_idx = (
            values.index(max(values))
            if metric_key != "max_drawdown"
            else values.index(min(values))
        )
        bars[best_idx].set_edgecolor("black")
        bars[best_idx].set_linewidth(2)

    plt.suptitle(
        "3-Way Model Comparison: Performance Metrics", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()

    plot_file = output_dir / "3way_comparison_metrics.png"
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    print(f"\nComparison plot saved to {plot_file}")
    plt.close()


# ===============================================================================
# SECTION 3: 시각화 함수들
# ===============================================================================


def visualize_alpha_transition(output_dir: str = "logs") -> None:
    """
    위기 적응형 alpha 전이 시각화.

    Original: visualize_alpha_transition.py
    """

    # alpha 계산을 위한 도우미 함수들
    def hard_threshold_alpha(crisis_levels, threshold=0.5, alpha_base=0.30):
        """기존 hard threshold 방식"""
        alpha_crisis = 0.06
        alpha_normal = alpha_base

        alphas = torch.where(
            crisis_levels > threshold,
            torch.tensor(alpha_crisis),
            torch.tensor(alpha_normal),
        )

        return alphas

    def smooth_cosine_alpha(crisis_levels, alpha_normal=0.30, alpha_crisis=0.06):
        """개선된 smooth cosine interpolation 방식"""
        pi = torch.tensor(3.14159265)

        smooth_alpha = (
            alpha_normal
            + (alpha_crisis - alpha_normal) * (1 - torch.cos(pi * crisis_levels)) / 2
        )

        smooth_alpha = torch.clamp(smooth_alpha, min=alpha_crisis, max=alpha_normal)

        return smooth_alpha

    def compute_gradient(alphas, crisis_levels):
        """경사도 계산"""
        dc = crisis_levels[1] - crisis_levels[0]
        gradients = torch.zeros_like(alphas)
        gradients[1:-1] = (alphas[2:] - alphas[:-2]) / (2 * dc)
        gradients[0] = (alphas[1] - alphas[0]) / dc
        gradients[-1] = (alphas[-1] - alphas[-2]) / dc

        return gradients

    # 위기 레벨 생성
    crisis_levels = torch.linspace(0, 1, 1000)

    # alpha 계산
    alphas_hard = hard_threshold_alpha(crisis_levels)
    alphas_smooth = smooth_cosine_alpha(crisis_levels)

    # 경사도 계산
    gradients_hard = compute_gradient(alphas_hard, crisis_levels)
    gradients_smooth = compute_gradient(alphas_smooth, crisis_levels)

    # Replicator 가중치
    replicator_hard = 1 - alphas_hard
    replicator_smooth = 1 - alphas_smooth

    # 시각화 생성
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 플롯 1: Alpha 전이 비교
    ax1 = axes[0, 0]
    ax1.plot(
        crisis_levels,
        alphas_hard,
        "r--",
        linewidth=2,
        label="Hard Threshold (Before)",
        alpha=0.8,
    )
    ax1.plot(
        crisis_levels,
        alphas_smooth,
        "b-",
        linewidth=2,
        label="Smooth Cosine (After)",
        alpha=0.8,
    )
    ax1.axvline(x=0.5, color="gray", linestyle=":", alpha=0.5, label="Old Threshold")
    ax1.set_xlabel("Crisis Level", fontsize=11)
    ax1.set_ylabel("Alpha (OT Weight)", fontsize=11)
    ax1.set_title("OT Weight Transition Comparison", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 0.35])

    # 플롯 2: Replicator 가중치
    ax2 = axes[0, 1]
    ax2.plot(
        crisis_levels,
        replicator_hard * 100,
        "r--",
        linewidth=2,
        label="Hard Threshold (Before)",
        alpha=0.8,
    )
    ax2.plot(
        crisis_levels,
        replicator_smooth * 100,
        "b-",
        linewidth=2,
        label="Smooth Cosine (After)",
        alpha=0.8,
    )
    ax2.axvline(x=0.5, color="gray", linestyle=":", alpha=0.5)
    ax2.set_xlabel("Crisis Level", fontsize=11)
    ax2.set_ylabel("Replicator Weight (%)", fontsize=11)
    ax2.set_title("Adaptive Mechanism Weight", fontsize=13, fontweight="bold")
    ax2.legend(loc="lower right", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([60, 100])

    # 플롯 3: 경사도 비교
    ax3 = axes[1, 0]
    ax3.plot(
        crisis_levels,
        gradients_hard,
        "r--",
        linewidth=2,
        label="Hard Threshold (Discontinuous)",
        alpha=0.8,
    )
    ax3.plot(
        crisis_levels,
        gradients_smooth,
        "b-",
        linewidth=2,
        label="Smooth Cosine (Continuous)",
        alpha=0.8,
    )
    ax3.set_xlabel("Crisis Level", fontsize=11)
    ax3.set_ylabel("d(Alpha)/d(Crisis)", fontsize=11)
    ax3.set_title("Gradient Analysis", fontsize=13, fontweight="bold")
    ax3.legend(loc="upper right", fontsize=10)
    ax3.grid(True, alpha=0.3)

    # 플롯 4: 메커니즘 지배가 있는 위기 구역
    ax4 = axes[1, 1]

    ax4.axvspan(0.0, 0.3, alpha=0.2, color="green", label="Normal Zone")
    ax4.axvspan(0.3, 0.7, alpha=0.2, color="yellow", label="Transition Zone")
    ax4.axvspan(0.7, 1.0, alpha=0.2, color="red", label="Crisis Zone")

    ax4.plot(
        crisis_levels,
        alphas_smooth * 100,
        "g-",
        linewidth=2,
        label="OT (Exploratory)",
        alpha=0.8,
    )
    ax4.plot(
        crisis_levels,
        replicator_smooth * 100,
        "r-",
        linewidth=2,
        label="Replicator (Adaptive)",
        alpha=0.8,
    )

    ax4.set_xlabel("Crisis Level", fontsize=11)
    ax4.set_ylabel("Mechanism Weight (%)", fontsize=11)
    ax4.set_title("Crisis-Adaptive Mechanism Mixing", fontsize=13, fontweight="bold")
    ax4.legend(loc="center right", fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 100])

    plt.suptitle(
        "Crisis-Adaptive Alpha Mixing Analysis", fontsize=15, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    # 플롯 저장
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    output_file = output_path / "alpha_transition_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Visualization saved to {output_file}")
    plt.close()

    # 통계 출력
    print("\n" + "=" * 60)
    print("Alpha Transition Analysis")
    print("=" * 60)

    crisis_points = [0.0, 0.3, 0.5, 0.7, 1.0]

    print("\n| Crisis Level | Hard Alpha | Smooth Alpha | Difference |")
    print("|--------------|------------|--------------|------------|")

    for cp in crisis_points:
        idx = int(cp * (len(crisis_levels) - 1))
        h_val = alphas_hard[idx].item()
        s_val = alphas_smooth[idx].item()
        diff = abs(h_val - s_val)
        print(f"| {cp:12.1f} | {h_val:10.3f} | {s_val:12.3f} | {diff:10.3f} |")

    # 경사도 통계
    max_grad_hard = torch.abs(gradients_hard).max().item()
    max_grad_smooth = torch.abs(gradients_smooth).max().item()

    print(f"\nMax |gradient| (Hard):   {max_grad_hard:.2f}")
    print(f"Max |gradient| (Smooth): {max_grad_smooth:.2f}")
    print(f"Gradient ratio:          {max_grad_hard/max_grad_smooth:.1f}x")


# ===============================================================================
# SECTION 4: 메인 CLI 인터페이스
# ===============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="IRT Experiments Suite - 통합 실험/분석/시각화 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 실험 실행
  python irt_experiments.py run --type grid-search --episodes 50
  python irt_experiments.py run --type 3way --episodes 200
  python irt_experiments.py run --type ablation --episodes 100

  # 결과 분석
  python irt_experiments.py analyze --type grid-search --input-dir logs/grid_search
  python irt_experiments.py analyze --type 3way --input-dir logs/3way_comparison

  # 시각화
  python irt_experiments.py visualize --type alpha-transition
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="명령어")

    # Run 명령
    run_parser = subparsers.add_parser("run", help="실험 실행")
    run_parser.add_argument(
        "--type",
        choices=["grid-search", "3way", "ablation"],
        required=True,
        help="실행할 실험 유형",
    )
    run_parser.add_argument(
        "--episodes", type=int, default=None, help="학습 에피소드 수"
    )
    run_parser.add_argument(
        "--output-dir", type=str, default=None, help="결과 출력 디렉토리"
    )

    # Analyze 명령
    analyze_parser = subparsers.add_parser("analyze", help="실험 결과 분석")
    analyze_parser.add_argument(
        "--type",
        choices=["grid-search", "3way"],
        required=True,
        help="수행할 분석 유형",
    )
    analyze_parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="실험 결과가 있는 디렉토리",
    )
    analyze_parser.add_argument(
        "--no-plot", action="store_true", help="플롯 생성 건너뛰기"
    )

    # Visualize 명령
    viz_parser = subparsers.add_parser("visualize", help="시각화 생성")
    viz_parser.add_argument(
        "--type",
        choices=["alpha-transition"],
        required=True,
        help="생성할 시각화 유형",
    )
    viz_parser.add_argument(
        "--output-dir", type=str, default="logs", help="플롯 출력 디렉토리"
    )

    args = parser.parse_args()

    # 명령어 처리
    if args.command == "run":
        if args.type == "grid-search":
            run_grid_search(
                output_dir=args.output_dir or "logs/grid_search",
                episodes=args.episodes or 50,
            )
        elif args.type == "3way":
            run_3way_comparison(
                output_dir=args.output_dir or "logs/3way_comparison",
                episodes=args.episodes or 200,
            )
        elif args.type == "ablation":
            run_ablation_study(
                output_dir=args.output_dir or "logs/ablation_study",
                episodes=args.episodes or 100,
            )

    elif args.command == "analyze":
        if args.type == "grid-search":
            analyze_grid_search(input_dir=args.input_dir, create_plot=not args.no_plot)
        elif args.type == "3way":
            analyze_3way(input_dir=args.input_dir, create_plot=not args.no_plot)

    elif args.command == "visualize":
        if args.type == "alpha-transition":
            visualize_alpha_transition(output_dir=args.output_dir)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
