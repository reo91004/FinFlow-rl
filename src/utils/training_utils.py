# src/utils/training_utils.py

"""
학습 유틸리티: 최적화 도구 + 안정성 모니터링 통합

목적: 강화학습 최적화 및 학습 안정성 도구 제공
의존: torch, numpy
사용처: BCellIRTActor, IQL, REDQ 등 IRT 학습 모듈
역할: 그래디언트 처리, 네트워크 업데이트, 안정성 모니터링

구현 내용:
[최적화 도구]
- Polyak averaging (타겟 네트워크 소프트 업데이트)
- SAC 온도 자동 조정
- 그래디언트 클리핑
- CQL 페널티, GAE, TD(λ), Retrace
- KL divergence, 엔트로피 정규화

[안정성 모니터링]
- Q-value 폭발/붕괴 감지
- 그래디언트 발산 체크
- 엔트로피 급락 감지
- 자동 개입 및 체크포인트 롤백
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Union, Dict, List, Any
from collections import deque
from dataclasses import dataclass, field
import time
from src.utils.logger import FinFlowLogger

# ============================================================================
# 최적화 도구 (Optimization Utilities)
# ============================================================================

def polyak_update(target_network: nn.Module,
                 source_network: nn.Module,
                 tau: float = 0.005) -> None:
    """
    Polyak averaging for target network soft update

    target = tau * source + (1 - tau) * target

    Args:
        target_network: Target network to update
        source_network: Source network to copy from
        tau: Soft update coefficient (0 < tau <= 1)
    """
    for target_param, source_param in zip(target_network.parameters(),
                                          source_network.parameters()):
        target_param.data.copy_(
            tau * source_param.data + (1.0 - tau) * target_param.data
        )

def auto_tune_temperature(log_alpha: torch.Tensor,
                         log_prob: torch.Tensor,
                         target_entropy: float,
                         alpha_optimizer: torch.optim.Optimizer,
                         alpha_min: float = 5e-4,
                         alpha_max: float = 0.2) -> Tuple[float, float]:
    """
    Automatic temperature (alpha) tuning for SAC

    Args:
        log_alpha: Log of temperature parameter
        log_prob: Log probability of actions
        target_entropy: Target entropy for exploration
        alpha_optimizer: Optimizer for alpha
        alpha_min: Minimum alpha value
        alpha_max: Maximum alpha value

    Returns:
        alpha: Current temperature value
        alpha_loss: Temperature loss value
    """
    # Compute alpha loss
    alpha_loss = -(log_alpha * (log_prob + target_entropy).detach()).mean()

    # Optimize alpha
    alpha_optimizer.zero_grad()
    alpha_loss.backward()
    alpha_optimizer.step()

    # Clamp alpha to valid range
    with torch.no_grad():
        alpha = log_alpha.exp().clamp(alpha_min, alpha_max)
        log_alpha.copy_(alpha.log())

    return alpha.item(), alpha_loss.item()

def cql_penalty(q_values: torch.Tensor,
               policy_actions: torch.Tensor,
               dataset_actions: torch.Tensor,
               num_samples: int = 10,
               alpha_cql: float = 1.0) -> torch.Tensor:
    """
    Conservative Q-Learning (CQL) penalty

    Penalizes Q-values for OOD actions to prevent overestimation

    Args:
        q_values: Q-network
        policy_actions: Actions from current policy
        dataset_actions: Actions from dataset
        num_samples: Number of OOD samples
        alpha_cql: CQL penalty weight

    Returns:
        cql_loss: CQL penalty term
    """
    batch_size = dataset_actions.shape[0]
    action_dim = dataset_actions.shape[1]
    device = dataset_actions.device

    # Sample random actions (OOD)
    random_actions = torch.rand(batch_size, num_samples, action_dim).to(device)
    random_actions = random_actions / random_actions.sum(dim=-1, keepdim=True)

    # Compute Q-values for different action types
    q_dataset = q_values(dataset_actions)
    q_policy = q_values(policy_actions)

    # Compute Q-values for random actions
    q_random_list = []
    for i in range(num_samples):
        q_random = q_values(random_actions[:, i, :])
        q_random_list.append(q_random)
    q_random = torch.stack(q_random_list, dim=1)

    # CQL loss: maximize Q for dataset actions, minimize for others
    logsumexp_random = torch.logsumexp(q_random, dim=1)
    cql_loss = logsumexp_random.mean() - q_dataset.mean()

    return alpha_cql * cql_loss

def compute_gae(rewards: torch.Tensor,
                values: torch.Tensor,
                next_values: torch.Tensor,
                dones: torch.Tensor,
                gamma: float = 0.99,
                gae_lambda: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation (GAE)

    Args:
        rewards: Reward sequence
        values: Value estimates
        next_values: Next value estimates
        dones: Done flags
        gamma: Discount factor
        gae_lambda: GAE lambda parameter

    Returns:
        advantages: GAE advantages
        returns: Discounted returns
    """
    advantages = torch.zeros_like(rewards)
    last_advantage = 0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = next_values[t]
        else:
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        advantages[t] = delta + gamma * gae_lambda * (1 - dones[t]) * last_advantage
        last_advantage = advantages[t]

    returns = advantages + values

    return advantages, returns

def clip_gradients(model: nn.Module,
                  max_norm: float = 1.0,
                  norm_type: float = 2.0) -> float:
    """
    Clip gradients by norm

    Args:
        model: Model with gradients
        max_norm: Maximum gradient norm
        norm_type: Type of norm (1, 2, or inf)

    Returns:
        total_norm: Total gradient norm before clipping
    """
    parameters = [p for p in model.parameters() if p.grad is not None]
    if len(parameters) == 0:
        return 0.0

    total_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type)
    return total_norm.item()

def compute_td_lambda(rewards: torch.Tensor,
                     values: torch.Tensor,
                     next_values: torch.Tensor,
                     dones: torch.Tensor,
                     gamma: float = 0.99,
                     td_lambda: float = 0.95) -> torch.Tensor:
    """
    Compute TD(λ) returns

    Args:
        rewards: Reward sequence
        values: Value estimates
        next_values: Next value estimates
        dones: Done flags
        gamma: Discount factor
        td_lambda: TD lambda parameter

    Returns:
        td_returns: TD(λ) returns
    """
    td_returns = torch.zeros_like(rewards)
    last_return = next_values[-1]

    for t in reversed(range(len(rewards))):
        td_returns[t] = rewards[t] + gamma * (1 - dones[t]) * (
            td_lambda * last_return + (1 - td_lambda) * next_values[t]
        )
        last_return = td_returns[t]

    return td_returns

def compute_retrace(q_values: torch.Tensor,
                   target_q_values: torch.Tensor,
                   actions: torch.Tensor,
                   rewards: torch.Tensor,
                   dones: torch.Tensor,
                   behavior_probs: torch.Tensor,
                   target_probs: torch.Tensor,
                   gamma: float = 0.99,
                   lambda_: float = 0.95) -> torch.Tensor:
    """
    Compute Retrace targets for off-policy correction

    Args:
        q_values: Q-values from main network
        target_q_values: Q-values from target network
        actions: Selected actions
        rewards: Rewards
        dones: Done flags
        behavior_probs: Behavior policy probabilities
        target_probs: Target policy probabilities
        gamma: Discount factor
        lambda_: Trace decay parameter

    Returns:
        retrace_targets: Retrace target values
    """
    batch_size, seq_len = rewards.shape[0], rewards.shape[1]
    device = rewards.device

    # Importance sampling ratios (clamped)
    ratios = (target_probs / (behavior_probs + 1e-8)).clamp(max=1.0)

    # Initialize retrace targets
    retrace_targets = torch.zeros_like(rewards)
    next_retrace = target_q_values[:, -1]

    for t in reversed(range(seq_len)):
        # TD error
        td_error = rewards[:, t] + gamma * (1 - dones[:, t]) * next_retrace - q_values[:, t]

        # Retrace target
        retrace_targets[:, t] = q_values[:, t] + ratios[:, t] * td_error

        # Update next retrace
        next_retrace = (lambda_ * ratios[:, t] * retrace_targets[:, t] +
                       (1 - lambda_ * ratios[:, t]) * q_values[:, t])

    return retrace_targets

def update_lagrange_multiplier(lagrange: torch.Tensor,
                             constraint_value: float,
                             constraint_target: float,
                             lagrange_lr: float = 1e-3,
                             max_lagrange: float = 10.0) -> float:
    """
    Update Lagrange multiplier for constrained optimization

    Args:
        lagrange: Current Lagrange multiplier
        constraint_value: Current constraint value
        constraint_target: Target constraint value
        lagrange_lr: Learning rate for Lagrange multiplier
        max_lagrange: Maximum Lagrange multiplier value

    Returns:
        updated_lagrange: Updated Lagrange multiplier value
    """
    with torch.no_grad():
        # Gradient ascent on Lagrange multiplier
        constraint_violation = constraint_value - constraint_target
        lagrange += lagrange_lr * constraint_violation

        # Clamp to valid range
        lagrange.clamp_(min=0.0, max=max_lagrange)

    return lagrange.item()

def entropy_regularization(log_probs: torch.Tensor,
                          alpha: float = 0.01) -> torch.Tensor:
    """
    Compute entropy regularization term

    Args:
        log_probs: Log probabilities of actions
        alpha: Entropy coefficient

    Returns:
        entropy_loss: Entropy regularization loss
    """
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
    return -alpha * entropy.mean()

def compute_kl_divergence(p_logits: torch.Tensor,
                        q_logits: torch.Tensor,
                        reduction: str = 'mean') -> torch.Tensor:
    """
    Compute KL divergence between two distributions

    Args:
        p_logits: Logits of distribution P
        q_logits: Logits of distribution Q
        reduction: Reduction method ('mean', 'sum', 'none')

    Returns:
        kl_div: KL divergence KL(P||Q)
    """
    p_probs = torch.softmax(p_logits, dim=-1)
    p_log_probs = torch.log_softmax(p_logits, dim=-1)
    q_log_probs = torch.log_softmax(q_logits, dim=-1)

    kl_div = (p_probs * (p_log_probs - q_log_probs)).sum(dim=-1)

    if reduction == 'mean':
        return kl_div.mean()
    elif reduction == 'sum':
        return kl_div.sum()
    else:
        return kl_div


# ============================================================================
# 학습 안정성 모니터링 (Stability Monitoring)
# ============================================================================

@dataclass
class MetricHistory:
    """메트릭 히스토리 관리"""
    window_size: int = 100
    values: deque = field(default_factory=lambda: deque(maxlen=100))

    def add(self, value: float):
        self.values.append(value)

    def get_mean(self) -> float:
        return np.mean(self.values) if self.values else 0.0

    def get_std(self) -> float:
        return np.std(self.values) if len(self.values) > 1 else 0.0

    def get_trend(self, window: int = 10) -> float:
        """최근 window개 값의 트렌드 (증가: +, 감소: -)"""
        if len(self.values) < window:
            return 0.0
        recent = list(self.values)[-window:]
        x = np.arange(len(recent))
        y = np.array(recent)
        if np.std(x) == 0 or np.std(y) == 0:
            return 0.0
        return np.corrcoef(x, y)[0, 1] * np.std(y)

    def is_diverging(self, threshold: float = 3.0) -> bool:
        """발산 여부 체크 (평균에서 표준편차의 threshold배 이상 벗어남)"""
        if len(self.values) < 10:
            return False
        mean = self.get_mean()
        std = self.get_std()
        if std == 0:
            return False
        latest = self.values[-1]
        return abs(latest - mean) > threshold * std


@dataclass
class StabilityMetrics:
    """학습 안정성 메트릭"""
    q_values: MetricHistory = field(default_factory=lambda: MetricHistory())
    gradients: MetricHistory = field(default_factory=lambda: MetricHistory())
    entropy: MetricHistory = field(default_factory=lambda: MetricHistory())
    rewards: MetricHistory = field(default_factory=lambda: MetricHistory())
    actor_loss: MetricHistory = field(default_factory=lambda: MetricHistory())
    critic_loss: MetricHistory = field(default_factory=lambda: MetricHistory())


class StabilityMonitor:
    """
    학습 안정성 모니터링 및 자동 개입

    주요 기능:
    - Q-value 폭발/붕괴 감지
    - 그래디언트 발산 모니터링
    - 엔트로피 붕괴 감지
    - 보상 클리프 탐지
    - 자동 체크포인트 롤백
    """

    def __init__(self,
                 enable_intervention: bool = True,
                 checkpoint_dir: str = "checkpoints/stability",
                 log_interval: int = 100):
        """
        Args:
            enable_intervention: 자동 개입 활성화
            checkpoint_dir: 안정성 체크포인트 저장 경로
            log_interval: 로깅 주기
        """
        self.enable_intervention = enable_intervention
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval
        self.logger = FinFlowLogger("StabilityMonitor")

        # 메트릭 추적
        self.metrics = StabilityMetrics()
        self.step_count = 0
        self.intervention_count = 0
        self.last_stable_checkpoint = None

        # 안정성 임계값
        self.thresholds = {
            'q_value_max': 1000.0,      # Q값 최대치
            'q_value_min': -1000.0,     # Q값 최소치
            'gradient_max': 10.0,       # 그래디언트 최대 노름
            'entropy_min': 0.01,        # 최소 엔트로피
            'reward_cliff': 10.0,       # 보상 클리프 (평균의 10배)
            'loss_explosion': 100.0,    # 손실 폭발 임계값
        }

        # 발산 감지 윈도우
        self.divergence_window = 50
        self.divergence_patience = 5
        self.divergence_counter = 0

        # 안정성 점수 (0~1, 1이 완전히 안정)
        self.stability_score = 1.0
        self.stability_history = deque(maxlen=1000)

    def check_q_values(self, q_values: torch.Tensor) -> Tuple[bool, str]:
        """Q-value 안정성 체크"""
        q_mean = q_values.mean().item()
        q_max = q_values.max().item()
        q_min = q_values.min().item()

        self.metrics.q_values.add(q_mean)

        # NaN 체크
        if torch.isnan(q_values).any():
            return False, "Q-values contain NaN"

        # 무한대 체크
        if torch.isinf(q_values).any():
            return False, "Q-values contain Inf"

        # 범위 체크
        if q_max > self.thresholds['q_value_max']:
            return False, f"Q-value explosion detected: max={q_max:.2f}"

        if q_min < self.thresholds['q_value_min']:
            return False, f"Q-value collapse detected: min={q_min:.2f}"

        # 발산 체크
        if self.metrics.q_values.is_diverging():
            return False, "Q-values are diverging"

        return True, "Q-values stable"

    def check_gradients(self, model: nn.Module) -> Tuple[bool, str]:
        """그래디언트 안정성 체크"""
        total_norm = 0.0
        param_count = 0

        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                param_count += 1

                # NaN 체크
                if torch.isnan(p.grad).any():
                    return False, "Gradients contain NaN"

                # 무한대 체크
                if torch.isinf(p.grad).any():
                    return False, "Gradients contain Inf"

        if param_count == 0:
            return True, "No gradients to check"

        total_norm = np.sqrt(total_norm)
        self.metrics.gradients.add(total_norm)

        # 노름 체크
        if total_norm > self.thresholds['gradient_max']:
            return False, f"Gradient explosion detected: norm={total_norm:.2f}"

        # 발산 체크
        if self.metrics.gradients.is_diverging():
            return False, "Gradients are diverging"

        return True, f"Gradients stable (norm={total_norm:.2f})"

    def check_entropy(self, entropy: float) -> Tuple[bool, str]:
        """엔트로피 안정성 체크"""
        self.metrics.entropy.add(entropy)

        # 엔트로피 붕괴 체크
        if entropy < self.thresholds['entropy_min']:
            return False, f"Entropy collapse detected: {entropy:.4f}"

        # 급격한 감소 체크
        if len(self.metrics.entropy.values) > 10:
            recent_trend = self.metrics.entropy.get_trend()
            if recent_trend < -0.5:  # 급격한 감소
                return False, f"Rapid entropy decrease: trend={recent_trend:.4f}"

        return True, f"Entropy stable: {entropy:.4f}"

    def check_rewards(self, reward: float) -> Tuple[bool, str]:
        """보상 안정성 체크"""
        self.metrics.rewards.add(reward)

        if len(self.metrics.rewards.values) < 10:
            return True, "Insufficient reward history"

        mean_reward = self.metrics.rewards.get_mean()

        # 보상 클리프 체크
        if abs(reward) > abs(mean_reward) * self.thresholds['reward_cliff']:
            return False, f"Reward cliff detected: {reward:.4f} (mean={mean_reward:.4f})"

        return True, f"Rewards stable: {reward:.4f}"

    def check_losses(self,
                    actor_loss: Optional[float] = None,
                    critic_loss: Optional[float] = None) -> Tuple[bool, str]:
        """손실 안정성 체크"""
        messages = []

        if actor_loss is not None:
            self.metrics.actor_loss.add(actor_loss)

            # NaN/Inf 체크
            if np.isnan(actor_loss) or np.isinf(actor_loss):
                return False, "Actor loss is NaN or Inf"

            # 폭발 체크
            if abs(actor_loss) > self.thresholds['loss_explosion']:
                return False, f"Actor loss explosion: {actor_loss:.4f}"

            # 발산 체크
            if self.metrics.actor_loss.is_diverging():
                messages.append("Actor loss diverging")

        if critic_loss is not None:
            self.metrics.critic_loss.add(critic_loss)

            # NaN/Inf 체크
            if np.isnan(critic_loss) or np.isinf(critic_loss):
                return False, "Critic loss is NaN or Inf"

            # 폭발 체크
            if abs(critic_loss) > self.thresholds['loss_explosion']:
                return False, f"Critic loss explosion: {critic_loss:.4f}"

            # 발산 체크
            if self.metrics.critic_loss.is_diverging():
                messages.append("Critic loss diverging")

        if messages:
            return False, "; ".join(messages)

        return True, "Losses stable"

    def update_stability_score(self, checks: List[Tuple[bool, str]]) -> float:
        """전체 안정성 점수 업데이트"""
        stable_count = sum(1 for is_stable, _ in checks if is_stable)
        total_count = len(checks)

        if total_count == 0:
            return self.stability_score

        # 현재 안정성 비율
        current_stability = stable_count / total_count

        # 지수 이동 평균으로 부드럽게 업데이트
        alpha = 0.1
        self.stability_score = alpha * current_stability + (1 - alpha) * self.stability_score

        self.stability_history.append(self.stability_score)

        return self.stability_score

    def check_overall_stability(self,
                              q_values: Optional[torch.Tensor] = None,
                              model: Optional[nn.Module] = None,
                              entropy: Optional[float] = None,
                              reward: Optional[float] = None,
                              actor_loss: Optional[float] = None,
                              critic_loss: Optional[float] = None) -> Tuple[bool, Dict[str, str]]:
        """
        전체 안정성 체크

        Returns:
            is_stable: 안정 여부
            diagnostics: 진단 메시지 딕셔너리
        """
        self.step_count += 1
        checks = []
        diagnostics = {}

        # 각 항목 체크
        if q_values is not None:
            is_stable, msg = self.check_q_values(q_values)
            checks.append((is_stable, msg))
            diagnostics['q_values'] = msg

        if model is not None:
            is_stable, msg = self.check_gradients(model)
            checks.append((is_stable, msg))
            diagnostics['gradients'] = msg

        if entropy is not None:
            is_stable, msg = self.check_entropy(entropy)
            checks.append((is_stable, msg))
            diagnostics['entropy'] = msg

        if reward is not None:
            is_stable, msg = self.check_rewards(reward)
            checks.append((is_stable, msg))
            diagnostics['rewards'] = msg

        if actor_loss is not None or critic_loss is not None:
            is_stable, msg = self.check_losses(actor_loss, critic_loss)
            checks.append((is_stable, msg))
            diagnostics['losses'] = msg

        # 안정성 점수 업데이트
        self.update_stability_score(checks)
        diagnostics['stability_score'] = f"{self.stability_score:.3f}"

        # 전체 안정성 판단
        unstable_checks = [msg for is_stable, msg in checks if not is_stable]

        if unstable_checks:
            self.divergence_counter += 1

            if self.divergence_counter >= self.divergence_patience:
                # 심각한 불안정
                diagnostics['status'] = "CRITICAL INSTABILITY"

                if self.enable_intervention:
                    self.intervene(unstable_checks)

                return False, diagnostics
        else:
            self.divergence_counter = max(0, self.divergence_counter - 1)
            diagnostics['status'] = "STABLE"

        # 주기적 로깅
        if self.step_count % self.log_interval == 0:
            self.log_status(diagnostics)

        return len(unstable_checks) == 0, diagnostics

    def intervene(self, issues: List[str]):
        """자동 개입"""
        self.intervention_count += 1
        self.logger.warning(f"Intervention #{self.intervention_count}: {'; '.join(issues)}")

        # TODO: 실제 개입 로직
        # - 학습률 감소
        # - 체크포인트 롤백
        # - 하이퍼파라미터 조정

    def save_checkpoint(self, model: nn.Module, tag: str = "stable"):
        """안정 상태 체크포인트 저장"""
        import os
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        checkpoint_path = f"{self.checkpoint_dir}/{tag}_{self.step_count}.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'stability_score': self.stability_score,
            'step_count': self.step_count,
            'metrics': self.get_metrics_summary()
        }, checkpoint_path)

        self.last_stable_checkpoint = checkpoint_path
        self.logger.info(f"Stability checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, model: nn.Module, checkpoint_path: Optional[str] = None):
        """체크포인트 로드"""
        if checkpoint_path is None:
            checkpoint_path = self.last_stable_checkpoint

        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            self.logger.warning("No checkpoint to load")
            return False

        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        self.stability_score = checkpoint.get('stability_score', 1.0)

        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
        return True

    def get_metrics_summary(self) -> Dict[str, Any]:
        """메트릭 요약"""
        return {
            'q_values': {
                'mean': self.metrics.q_values.get_mean(),
                'std': self.metrics.q_values.get_std(),
                'trend': self.metrics.q_values.get_trend()
            },
            'gradients': {
                'mean': self.metrics.gradients.get_mean(),
                'std': self.metrics.gradients.get_std(),
                'trend': self.metrics.gradients.get_trend()
            },
            'entropy': {
                'mean': self.metrics.entropy.get_mean(),
                'std': self.metrics.entropy.get_std(),
                'trend': self.metrics.entropy.get_trend()
            },
            'rewards': {
                'mean': self.metrics.rewards.get_mean(),
                'std': self.metrics.rewards.get_std(),
                'trend': self.metrics.rewards.get_trend()
            },
            'stability_score': self.stability_score,
            'intervention_count': self.intervention_count
        }

    def log_status(self, diagnostics: Dict[str, str]):
        """상태 로깅"""
        self.logger.debug(f"Step {self.step_count} - Stability: {self.stability_score:.3f}")
        for key, value in diagnostics.items():
            self.logger.debug(f"  {key}: {value}")

    def reset_metrics(self):
        """메트릭 초기화"""
        self.metrics = StabilityMetrics()
        self.divergence_counter = 0
        self.stability_score = 1.0
        self.logger.info("Stability metrics reset")