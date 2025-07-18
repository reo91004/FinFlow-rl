# utils/rl_tracker.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from datetime import datetime
from collections import deque
from typing import Dict, List, Optional
import seaborn as sns

# 한글 폰트 문제 방지
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


class RLTracker:
    """강화학습 학습 과정 추적 및 시각화"""

    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or "."

        # 학습 데이터 저장
        self.episode_rewards = []
        self.episode_returns = []
        self.learning_rates = {"actor": [], "critic": []}
        self.losses = {"actor": [], "critic": [], "total": []}
        self.td_errors = []
        self.epsilon_values = []
        self.curriculum_levels = []
        self.bcell_rewards = {f"bcell_{i}": [] for i in range(5)}
        self.meta_rewards = []

        # 시간 추적
        self.episode_timestamps = []
        self.training_start_time = datetime.now()

        # 통계
        self.reward_stats = {"mean": [], "std": [], "max": [], "min": []}

    def log_episode(
        self,
        episode: int,
        reward: float,
        portfolio_return: float = None,
        learning_rates: Dict = None,
        losses: Dict = None,
        epsilon: float = None,
        curriculum_level: int = None,
        bcell_rewards: Dict = None,
        meta_reward: float = None,
        td_error: float = None,
    ):
        """에피소드별 학습 데이터 기록"""

        self.episode_rewards.append(reward)
        self.episode_timestamps.append(datetime.now())

        if portfolio_return is not None:
            self.episode_returns.append(portfolio_return)

        if learning_rates:
            self.learning_rates["actor"].append(learning_rates.get("actor", 0))
            self.learning_rates["critic"].append(learning_rates.get("critic", 0))

        if losses:
            self.losses["actor"].append(losses.get("actor", 0))
            self.losses["critic"].append(losses.get("critic", 0))
            self.losses["total"].append(losses.get("total", 0))

        if epsilon is not None:
            self.epsilon_values.append(epsilon)

        if curriculum_level is not None:
            self.curriculum_levels.append(curriculum_level)

        if bcell_rewards:
            for bcell_id, reward_val in bcell_rewards.items():
                if bcell_id in self.bcell_rewards:
                    self.bcell_rewards[bcell_id].append(reward_val)

        if meta_reward is not None:
            self.meta_rewards.append(meta_reward)

        if td_error is not None:
            self.td_errors.append(td_error)

        # 통계 업데이트
        if len(self.episode_rewards) >= 10:
            recent_rewards = self.episode_rewards[-10:]
            self.reward_stats["mean"].append(np.mean(recent_rewards))
            self.reward_stats["std"].append(np.std(recent_rewards))
            self.reward_stats["max"].append(np.max(recent_rewards))
            self.reward_stats["min"].append(np.min(recent_rewards))

    def extract_from_immune_system(self, immune_system, episode: int):
        """면역 시스템에서 자동으로 데이터 추출"""

        # B-Cell 데이터 추출
        bcell_data = {}
        learning_rates = {}
        losses = {}

        if hasattr(immune_system, "bcells") and immune_system.use_learning_bcells:
            for i, bcell in enumerate(immune_system.bcells):
                # 보상 데이터
                if (
                    hasattr(bcell, "specialist_performance")
                    and bcell.specialist_performance
                ):
                    bcell_data[f"bcell_{i}_{bcell.risk_type}"] = np.mean(
                        list(bcell.specialist_performance)[-5:]
                    )

                # 학습률 데이터
                if hasattr(bcell, "actor_optimizer"):
                    actor_lr = bcell.actor_optimizer.param_groups[0]["lr"]
                    critic_lr = bcell.critic_optimizer.param_groups[0]["lr"]
                    learning_rates[f"actor_{i}"] = actor_lr
                    learning_rates[f"critic_{i}"] = critic_lr

                # Epsilon 값
                epsilon = getattr(bcell, "epsilon", None)

        # 커리큘럼 레벨
        curriculum_level = None
        if (
            hasattr(immune_system, "curriculum_manager")
            and immune_system.curriculum_manager
        ):
            curriculum_level = immune_system.curriculum_manager.scheduler.current_level

        # 메타 컨트롤러 보상
        meta_reward = None
        if (
            hasattr(immune_system, "hierarchical_controller")
            and immune_system.hierarchical_controller
        ):
            if immune_system.hierarchical_controller.meta_level_rewards:
                meta_reward = immune_system.hierarchical_controller.meta_level_rewards[
                    -1
                ]

        # 전체 시스템 보상 (예: 최근 포트폴리오 수익률)
        system_reward = getattr(immune_system, "last_portfolio_return", 0.0)

        return {
            "reward": system_reward,
            "learning_rates": learning_rates,
            "losses": losses,
            "epsilon": epsilon,
            "curriculum_level": curriculum_level,
            "bcell_rewards": bcell_data,
            "meta_reward": meta_reward,
        }

    def create_comprehensive_plot(self, save_path: str = None, window_size: int = 50):
        """종합적인 강화학습 추적 플롯 생성"""

        if len(self.episode_rewards) < 10:
            print("[경고] 충분한 데이터가 없어서 시각화를 건너뜁니다.")
            return None

        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

        # 1. 에피소드별 보상 변화
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_reward_progression(ax1, window_size)

        # 2. Learning Rate 변화
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_learning_rates(ax2)

        # 3. 손실 함수 변화
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_losses(ax3, window_size)

        # 4. TD Error 및 Epsilon
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_td_error_epsilon(ax4, window_size)

        # 5. B-Cell별 성능
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_bcell_performance(ax5, window_size)

        # 6. 커리큘럼 학습 진행
        ax6 = fig.add_subplot(gs[2, 0])
        self._plot_curriculum_progress(ax6)

        # 7. 보상 분포
        ax7 = fig.add_subplot(gs[2, 1])
        self._plot_reward_distribution(ax7)

        # 8. 학습 안정성 지표
        ax8 = fig.add_subplot(gs[2, 2])
        self._plot_learning_stability(ax8, window_size)

        # 9. 메타 컨트롤러 성능 (계층적 학습)
        ax9 = fig.add_subplot(gs[3, 0])
        self._plot_meta_performance(ax9, window_size)

        # 10. 전체 학습 요약
        ax10 = fig.add_subplot(gs[3, 1:])
        self._plot_learning_summary(ax10)

        plt.suptitle(
            "BIPD Reinforcement Learning Training Progress",
            fontsize=16,
            fontweight="bold",
        )

        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(
                self.output_dir, f"rl_training_progress_{timestamp}.png"
            )

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"강화학습 추적 차트 저장: {save_path}")

        return fig

    def _plot_reward_progression(self, ax, window_size):
        """에피소드별 보상 진행 플롯"""
        episodes = range(len(self.episode_rewards))

        # 원본 보상
        ax.plot(
            episodes,
            self.episode_rewards,
            alpha=0.3,
            color="lightblue",
            label="Episode Rewards",
            linewidth=0.5,
        )

        # 이동평균
        if len(self.episode_rewards) > window_size:
            moving_avg = pd.Series(self.episode_rewards).rolling(window_size).mean()
            ax.plot(
                episodes,
                moving_avg,
                color="blue",
                linewidth=2,
                label=f"Moving Average ({window_size})",
            )

        # 추세선
        if len(episodes) > 10:
            z = np.polyfit(episodes, self.episode_rewards, 1)
            p = np.poly1d(z)
            ax.plot(
                episodes, p(episodes), "--", color="red", alpha=0.7, label="Trend Line"
            )

        ax.set_title("Episode Reward Progression")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_learning_rates(self, ax):
        """학습률 변화 플롯"""
        if self.learning_rates["actor"]:
            episodes = range(len(self.learning_rates["actor"]))
            ax.semilogy(
                episodes,
                self.learning_rates["actor"],
                label="Actor LR",
                color="green",
                marker="o",
                markersize=2,
            )
        if self.learning_rates["critic"]:
            episodes = range(len(self.learning_rates["critic"]))
            ax.semilogy(
                episodes,
                self.learning_rates["critic"],
                label="Critic LR",
                color="orange",
                marker="s",
                markersize=2,
            )

        ax.set_title("Learning Rate Schedule")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Learning Rate (log scale)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_losses(self, ax, window_size):
        """손실 함수 변화 플롯"""
        if self.losses["actor"]:
            episodes = range(len(self.losses["actor"]))
            if len(self.losses["actor"]) > window_size:
                actor_smooth = (
                    pd.Series(self.losses["actor"]).rolling(window_size // 5).mean()
                )
                ax.plot(episodes, actor_smooth, label="Actor Loss", color="red")

        if self.losses["critic"]:
            episodes = range(len(self.losses["critic"]))
            if len(self.losses["critic"]) > window_size:
                critic_smooth = (
                    pd.Series(self.losses["critic"]).rolling(window_size // 5).mean()
                )
                ax.plot(episodes, critic_smooth, label="Critic Loss", color="blue")

        ax.set_title("Training Losses")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_td_error_epsilon(self, ax, window_size):
        """TD Error와 Epsilon 플롯"""
        if self.td_errors and self.epsilon_values:
            ax2 = ax.twinx()

            # TD Error
            episodes = range(len(self.td_errors))
            if len(self.td_errors) > window_size:
                td_smooth = pd.Series(self.td_errors).rolling(window_size // 5).mean()
                ax.plot(episodes, td_smooth, color="purple", label="TD Error")

            # Epsilon
            eps_episodes = range(len(self.epsilon_values))
            ax2.plot(
                eps_episodes,
                self.epsilon_values,
                color="orange",
                label="Epsilon",
                linestyle="--",
                alpha=0.7,
            )

            ax.set_xlabel("Episode")
            ax.set_ylabel("TD Error", color="purple")
            ax2.set_ylabel("Epsilon", color="orange")
            ax.set_title("TD Error & Exploration Rate")
        else:
            ax.text(
                0.5,
                0.5,
                "No TD Error/Epsilon Data",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            ax.set_title("TD Error & Exploration Rate")

    def _plot_bcell_performance(self, ax, window_size):
        """B-Cell별 성능 플롯"""
        colors = ["red", "blue", "green", "orange", "purple"]

        for i, (bcell_name, rewards) in enumerate(self.bcell_rewards.items()):
            if rewards and len(rewards) > 5:
                episodes = range(len(rewards))
                if len(rewards) > window_size // 5:
                    smooth_rewards = pd.Series(rewards).rolling(window_size // 5).mean()
                    ax.plot(
                        episodes,
                        smooth_rewards,
                        color=colors[i % len(colors)],
                        label=bcell_name.replace("bcell_", "")
                        .replace("_", " ")
                        .title(),
                        linewidth=1.5,
                    )

        ax.set_title("B-Cell Expert Performance")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Performance")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_curriculum_progress(self, ax):
        """커리큘럼 학습 진행 플롯"""
        if self.curriculum_levels:
            episodes = range(len(self.curriculum_levels))
            ax.step(
                episodes,
                self.curriculum_levels,
                where="post",
                color="darkgreen",
                linewidth=2,
            )
            ax.fill_between(
                episodes,
                self.curriculum_levels,
                step="post",
                alpha=0.3,
                color="lightgreen",
            )

            # 레벨 변화 지점 표시
            level_changes = []
            for i in range(1, len(self.curriculum_levels)):
                if self.curriculum_levels[i] != self.curriculum_levels[i - 1]:
                    level_changes.append(i)

            for change_point in level_changes:
                ax.axvline(x=change_point, color="red", linestyle="--", alpha=0.7)

        ax.set_title("Curriculum Learning Progress")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Curriculum Level")
        ax.grid(True, alpha=0.3)

    def _plot_reward_distribution(self, ax):
        """보상 분포 히스토그램"""
        if len(self.episode_rewards) > 20:
            ax.hist(
                self.episode_rewards,
                bins=30,
                alpha=0.7,
                color="skyblue",
                edgecolor="black",
            )

            # 통계 정보 추가
            mean_reward = np.mean(self.episode_rewards)
            std_reward = np.std(self.episode_rewards)
            ax.axvline(
                mean_reward,
                color="red",
                linestyle="--",
                label=f"Mean: {mean_reward:.3f}",
            )
            ax.axvline(
                mean_reward + std_reward,
                color="orange",
                linestyle=":",
                label=f"+1σ: {mean_reward + std_reward:.3f}",
            )
            ax.axvline(
                mean_reward - std_reward,
                color="orange",
                linestyle=":",
                label=f"-1σ: {mean_reward - std_reward:.3f}",
            )

        ax.set_title("Reward Distribution")
        ax.set_xlabel("Reward Value")
        ax.set_ylabel("Frequency")
        ax.legend()

    def _plot_learning_stability(self, ax, window_size):
        """학습 안정성 지표"""
        if len(self.reward_stats["std"]) > 5:
            episodes = range(len(self.reward_stats["std"]))
            ax.plot(
                episodes,
                self.reward_stats["std"],
                color="red",
                label="Reward Std",
                linewidth=2,
            )

            # 안정성 임계값
            stability_threshold = np.mean(self.reward_stats["std"])
            ax.axhline(
                y=stability_threshold,
                color="green",
                linestyle="--",
                alpha=0.7,
                label=f"Avg Std: {stability_threshold:.3f}",
            )

        ax.set_title("Learning Stability (Reward Std)")
        ax.set_xlabel("Episode (windowed)")
        ax.set_ylabel("Standard Deviation")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_meta_performance(self, ax, window_size):
        """메타 컨트롤러 성능 (계층적 학습)"""
        if self.meta_rewards and len(self.meta_rewards) > 5:
            episodes = range(len(self.meta_rewards))
            ax.plot(episodes, self.meta_rewards, alpha=0.5, color="lightcoral")

            if len(self.meta_rewards) > window_size // 5:
                meta_smooth = (
                    pd.Series(self.meta_rewards).rolling(window_size // 5).mean()
                )
                ax.plot(
                    episodes,
                    meta_smooth,
                    color="darkred",
                    linewidth=2,
                    label="Meta Controller Reward",
                )
        else:
            ax.text(
                0.5,
                0.5,
                "No Meta Controller Data",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )

        ax.set_title("Meta Controller Performance")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Meta Reward")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_learning_summary(self, ax):
        """전체 학습 요약"""
        # 학습 진행 요약 테이블
        if len(self.episode_rewards) > 0:
            summary_data = {
                "Total Episodes": len(self.episode_rewards),
                "Avg Reward": f"{np.mean(self.episode_rewards):.3f}",
                "Best Reward": f"{np.max(self.episode_rewards):.3f}",
                "Worst Reward": f"{np.min(self.episode_rewards):.3f}",
                "Reward Std": f"{np.std(self.episode_rewards):.3f}",
                "Training Time": str(datetime.now() - self.training_start_time).split(
                    "."
                )[0],
            }

            if self.curriculum_levels:
                summary_data["Max Curriculum Level"] = max(self.curriculum_levels)

            if self.epsilon_values:
                summary_data["Final Epsilon"] = f"{self.epsilon_values[-1]:.3f}"

            # 텍스트로 요약 정보 표시
            summary_text = "\n".join([f"{k}: {v}" for k, v in summary_data.items()])
            ax.text(
                0.1,
                0.9,
                summary_text,
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7),
            )

            # 학습 곡선 트렌드 분석
            if len(self.episode_rewards) > 10:
                recent_rewards = self.episode_rewards[-len(self.episode_rewards) // 4 :]
                early_rewards = self.episode_rewards[: len(self.episode_rewards) // 4]

                improvement = np.mean(recent_rewards) - np.mean(early_rewards)
                trend_text = f"Learning Trend: {improvement:+.3f}"
                trend_color = "green" if improvement > 0 else "red"

                ax.text(
                    0.6,
                    0.5,
                    trend_text,
                    transform=ax.transAxes,
                    fontsize=14,
                    color=trend_color,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
                )

        ax.set_title("Training Summary")
        ax.axis("off")

    def save_training_data(self, filename: str = None):
        """학습 데이터를 CSV로 저장"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(
                self.output_dir, f"rl_training_data_{timestamp}.csv"
            )

        # 데이터프레임 생성
        max_length = (
            max(
                len(self.episode_rewards),
                len(self.learning_rates.get("actor", [])),
                len(self.epsilon_values),
            )
            if any(
                [
                    self.episode_rewards,
                    self.learning_rates.get("actor", []),
                    self.epsilon_values,
                ]
            )
            else 0
        )

        data = {}
        data["episode"] = list(range(max_length))

        # 각 데이터를 최대 길이에 맞춰 패딩
        def pad_list(lst, target_length):
            return lst + [np.nan] * (target_length - len(lst))

        data["reward"] = pad_list(self.episode_rewards, max_length)
        data["actor_lr"] = pad_list(self.learning_rates.get("actor", []), max_length)
        data["critic_lr"] = pad_list(self.learning_rates.get("critic", []), max_length)
        data["epsilon"] = pad_list(self.epsilon_values, max_length)
        data["curriculum_level"] = pad_list(self.curriculum_levels, max_length)

        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"학습 데이터 저장: {filename}")

        return filename
