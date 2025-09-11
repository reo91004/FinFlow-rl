# src/utils/optimizer_utils.py

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Union

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