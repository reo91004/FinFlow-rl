# finrl/agents/irt/alpha_scheduler.py

"""
Dynamic Alpha Scheduler for IRT

이론적 기반:
- Bengio et al. (2009): Curriculum Learning
  "Learning is more effective when examples are organized in a meaningful order."
- Loshchilov & Hutter (2017): SGDR - Cosine annealing

Alpha의 의미:
- w = (1-α)·Replicator + α·OT
- α=0.3 (Replicator 70%): 과거 성공 전략 선호 (exploitation)
- α=0.7 (OT 70%): 구조적 매칭 (exploration)

Curriculum 전략:
- Early training: α low → Replicator dominant (빠른 수렴)
- Late training: α high → OT dominant (구조적 매칭)

사용법:
    scheduler = AlphaScheduler(alpha_start=0.3, alpha_end=0.7)

    for step in range(total_steps):
        alpha = scheduler.get_alpha(step)
        model.policy.actor.irt_actor.irt.alpha = alpha
"""

import numpy as np
from typing import Literal


class AlphaScheduler:
    """
    Alpha scheduler for IRT mixing ratio.

    w = (1-α)·Replicator + α·OT

    Schedule types:
    - 'linear': α(t) = α_start + (α_end - α_start) * t/T
    - 'cosine': α(t) = α_start + 0.5*(α_end - α_start)*(1 - cos(πt/T))
    - 'exponential': α(t) = α_end - (α_end - α_start) * exp(-λt/T)

    권장: 'cosine' (smooth transition, fast early, slow late)
    """

    def __init__(
        self,
        alpha_start: float = 0.3,
        alpha_end: float = 0.7,
        total_steps: int = 50000,
        schedule_type: Literal['linear', 'cosine', 'exponential'] = 'cosine'
    ):
        """
        Args:
            alpha_start: Initial alpha (low for Replicator dominance)
            alpha_end: Final alpha (high for OT dominance)
            total_steps: Total training steps
            schedule_type: Schedule function ('cosine' recommended)
        """
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.total_steps = total_steps
        self.schedule_type = schedule_type

        assert 0 <= alpha_start <= 1, "alpha_start must be in [0, 1]"
        assert 0 <= alpha_end <= 1, "alpha_end must be in [0, 1]"
        assert alpha_start < alpha_end, "alpha should increase over time"

    def get_alpha(self, step: int) -> float:
        """
        Get alpha for current step.

        Args:
            step: Current training step

        Returns:
            alpha: Mixing ratio ∈ [alpha_start, alpha_end]
        """
        progress = min(step / self.total_steps, 1.0)
        delta = self.alpha_end - self.alpha_start

        if self.schedule_type == 'linear':
            # Linear interpolation
            alpha = self.alpha_start + delta * progress

        elif self.schedule_type == 'cosine':
            # Smooth transition (fast early, slow late)
            # Cosine annealing from Loshchilov & Hutter (2017)
            alpha = self.alpha_start + delta * 0.5 * (1 - np.cos(np.pi * progress))

        elif self.schedule_type == 'exponential':
            # Very fast early transition, slow late
            alpha = self.alpha_end - delta * np.exp(-5 * progress)

        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

        return float(alpha)

    def plot_schedule(self, save_path: str = None):
        """
        Visualize schedule (for debugging).

        Args:
            save_path: Path to save plot (optional)
        """
        import matplotlib.pyplot as plt

        steps = np.linspace(0, self.total_steps, 1000)
        alphas = [self.get_alpha(int(s)) for s in steps]

        plt.figure(figsize=(10, 6))
        plt.plot(steps, alphas, linewidth=2, label=f'{self.schedule_type} schedule')
        plt.axhline(self.alpha_start, color='gray', linestyle='--', alpha=0.5, label='Start')
        plt.axhline(self.alpha_end, color='gray', linestyle='--', alpha=0.5, label='End')
        plt.xlabel('Training Step')
        plt.ylabel('Alpha (OT weight)')
        plt.title(f'Alpha Schedule ({self.schedule_type})')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
