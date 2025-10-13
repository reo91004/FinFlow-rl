"""
FinFlow custom SAC implementation with gradient clipping and logging hooks.

This extends Stable-Baselines3 SAC to:
    - apply gradient clipping to actor/critic optimizers (Phase 2 requirement)
    - expose clipped gradient norms for debugging

The rest of the training loop remains identical to SB3's SAC implementation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch as th
from torch import nn
from torch.nn.utils import clip_grad_norm_

from stable_baselines3.sac.sac import SAC
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.type_aliases import MaybeCallback


class FinFlowSAC(SAC):
    """
    Soft Actor-Critic with gradient clipping and norm logging.

    Args:
        max_grad_norm: maximum gradient norm applied to actor and critic (default: 1.0)
    """

    def __init__(self, *args, max_grad_norm: float = 1.0, **kwargs):
        self.max_grad_norm = float(max_grad_norm) if max_grad_norm is not None else None
        super().__init__(*args, **kwargs)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (affects dropout/batch norm)
        self.policy.set_training_mode(True)

        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers.append(self.ent_coef_optimizer)

        # Update learning rate according to schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        critic_grad_norms, actor_grad_norms = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            if self.use_sde:
                self.actor.reset_noise()

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                assert isinstance(self.target_entropy, float)
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                if self.max_grad_norm and self.max_grad_norm > 0:
                    clip_grad_norm_([self.log_ent_coef], self.max_grad_norm)
                self.ent_coef_optimizer.step()

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(nn.functional.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            critic_losses.append(float(critic_loss.item()))

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            if self.max_grad_norm and self.max_grad_norm > 0:
                critic_norm = clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                critic_grad_norms.append(float(critic_norm))
            self.critic.optimizer.step()

            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(float(actor_loss.item()))

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            if self.max_grad_norm and self.max_grad_norm > 0:
                actor_norm = clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                actor_grad_norms.append(float(actor_norm))
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        # Logging
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
        if actor_grad_norms:
            self.logger.record("train/actor_grad_norm", np.mean(actor_grad_norms))
        if critic_grad_norms:
            self.logger.record("train/critic_grad_norm", np.mean(critic_grad_norms))

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "SAC",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> "FinFlowSAC":
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
