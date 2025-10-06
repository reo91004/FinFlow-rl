# finrl/agents/irt/alpha_scheduler.py

"""
Dynamic Alpha Scheduler for IRT

이론적 기반:
- Bengio et al. (2009): Curriculum Learning
  "Learning is more effective when examples are organized in a meaningful order."
- Loshchilov & Hutter (2017): SGDR - Cosine annealing
- Haarnoja et al. (2018): SAC - Entropy-regularized RL

Alpha의 의미:
- w = (1-α)·Replicator + α·OT
- α=0.3 (Replicator 70%): 과거 성공 전략 선호 (exploitation)
- α=0.7 (OT 70%): 구조적 매칭 (exploration)

Schedule types:
- 'linear': Linear interpolation
- 'cosine': Smooth transition (fast early, slow late)
- 'exponential': Very fast early transition
- 'adaptive': Entropy-based automatic tuning (SAC style)

사용법:
    # Step-based scheduling
    scheduler = AlphaScheduler(schedule_type='cosine', alpha_start=0.3, alpha_end=0.7)
    alpha = scheduler.get_alpha(step)

    # Adaptive (entropy-based) tuning
    scheduler = AlphaScheduler(schedule_type='adaptive', action_dim=30)
    new_alpha, alpha_loss = scheduler.update(step, log_prob)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Literal, Optional, Tuple, Union


class AlphaScheduler:
    """
    Unified Alpha Scheduler for IRT mixing ratio.

    Supports both step-based scheduling and entropy-based adaptive tuning.

    w = (1-α)·Replicator + α·OT

    Schedule types:
    - 'linear': α(t) = α_start + (α_end - α_start) * t/T
    - 'cosine': α(t) = α_start + 0.5*(α_end - α_start)*(1 - cos(πt/T))
    - 'exponential': α(t) = α_end - (α_end - α_start) * exp(-λt/T)
    - 'adaptive': SAC-style entropy-based automatic tuning

    권장: 'cosine' for simplicity, 'adaptive' for best performance
    """

    def __init__(
        self,
        schedule_type: Literal['linear', 'cosine', 'exponential', 'adaptive'] = 'cosine',
        # Step-based scheduling parameters
        alpha_start: float = 0.3,
        alpha_end: float = 0.7,
        total_steps: int = 50000,
        # Adaptive (entropy-based) parameters
        action_dim: Optional[int] = None,
        alpha_min: float = 0.05,
        alpha_max: float = 0.40,
        warmup_steps: int = 5000,
        lr: float = 3e-4
    ):
        """
        Args:
            schedule_type: Type of scheduling ('cosine' or 'adaptive')

            Step-based scheduling:
            alpha_start: Initial alpha (low for Replicator dominance)
            alpha_end: Final alpha (high for OT dominance)
            total_steps: Total training steps

            Adaptive (entropy-based):
            action_dim: Action space dimension (for target entropy)
            alpha_min: Minimum alpha value
            alpha_max: Maximum alpha value
            warmup_steps: Steps before alpha adaptation starts
            lr: Learning rate for alpha optimization
        """
        self.schedule_type = schedule_type

        if schedule_type in ['linear', 'cosine', 'exponential']:
            # Step-based scheduling
            self.alpha_start = alpha_start
            self.alpha_end = alpha_end
            self.total_steps = total_steps

            assert 0 <= alpha_start <= 1, "alpha_start must be in [0, 1]"
            assert 0 <= alpha_end <= 1, "alpha_end must be in [0, 1]"
            assert alpha_start < alpha_end, "alpha should increase over time"

        elif schedule_type == 'adaptive':
            # Adaptive (entropy-based) tuning
            assert action_dim is not None, "action_dim required for adaptive mode"

            self.target_entropy = -float(action_dim)  # SAC 표준: -dim(A)
            self.alpha_min = alpha_min
            self.alpha_max = alpha_max
            self.warmup_steps = warmup_steps

            # Log-space alpha parameter (SAC style)
            initial_log_alpha = math.log(0.3)  # log(0.3) ≈ -1.204
            # Use regular tensor with requires_grad instead of nn.Parameter
            self.log_alpha = torch.tensor(initial_log_alpha, dtype=torch.float32, requires_grad=True)
            self.optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

    @property
    def alpha(self) -> torch.Tensor:
        """Get current alpha with clamping (adaptive mode only)."""
        if self.schedule_type != 'adaptive':
            raise RuntimeError("alpha property only available in adaptive mode")

        with torch.no_grad():
            self.log_alpha.data.clamp_(
                math.log(self.alpha_min),
                math.log(self.alpha_max)
            )
        return self.log_alpha.exp()

    def get_alpha(self, step: int) -> float:
        """
        Get alpha for current step (step-based scheduling).

        Args:
            step: Current training step

        Returns:
            alpha: Mixing ratio ∈ [alpha_start, alpha_end]
        """
        if self.schedule_type == 'adaptive':
            # Adaptive mode uses update() method instead
            return self.alpha.item()

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

    def update(self, step: int, log_prob: Union[torch.Tensor, float]) -> Tuple[torch.Tensor, float]:
        """
        Update alpha based on policy entropy (adaptive mode only).

        Args:
            step: Current training step
            log_prob: Log probability from policy (detached tensor or scalar value)

        Returns:
            alpha: Current alpha value (detached tensor)
            alpha_loss: Alpha optimization loss (scalar float)
        """
        if self.schedule_type != 'adaptive':
            raise RuntimeError("update() method only available in adaptive mode")

        # Warmup period: fixed alpha
        if step < self.warmup_steps:
            return self.alpha.detach(), 0.0

        # Convert log_prob to scalar value (already detached from IRTPolicy)
        if isinstance(log_prob, torch.Tensor):
            log_prob_value = log_prob.mean().item() if log_prob.numel() > 1 else log_prob.item()
        else:
            log_prob_value = float(log_prob)

        # Compute alpha loss
        # SAC formula: L = -log(α) * (H_target - H_current)
        # where H_current = -E[log π(a|s)] ≈ -log_prob_value
        # Since log_prob is negative, adding it to target_entropy gives us the difference

        # Compute entropy difference
        entropy_diff = self.target_entropy + log_prob_value  # Both are scalars

        # Since log_prob_value is a scalar and self.log_alpha needs gradient,
        # we manually compute the gradient instead of using backward()
        # Gradient of -log_alpha * entropy_diff w.r.t log_alpha is -entropy_diff
        with torch.no_grad():
            # Manually compute gradient
            grad = -entropy_diff

            # Clear previous gradients
            self.optimizer.zero_grad()

            # Set gradient manually
            if self.log_alpha.grad is None:
                self.log_alpha.grad = torch.tensor(grad, dtype=self.log_alpha.dtype)
            else:
                self.log_alpha.grad.fill_(grad)

            # Optimizer step
            self.optimizer.step()

            # Clamp log_alpha to valid range
            self.log_alpha.data.clamp_(
                math.log(self.alpha_min),
                math.log(self.alpha_max)
            )

        # Compute loss for logging
        alpha_loss_value = -(self.log_alpha.item() * entropy_diff)

        return self.alpha.detach(), alpha_loss_value

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
