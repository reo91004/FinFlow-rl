# src/core/distributional.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

class QuantileNetwork(nn.Module):
    """
    Quantile Regression Network for Distributional RL
    
    Returns N quantile values for Q-distribution
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, 
                 n_quantiles: int = 32):
        super().__init__()
        self.n_quantiles = n_quantiles
        
        # 네트워크 구조
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_quantiles)
        )
        
        # Quantile fractions (균등 분포)
        self.register_buffer(
            'tau', 
            torch.linspace(0, 1, n_quantiles + 1)[:-1] + 1/(2*n_quantiles)
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [batch_size, state_dim]
            action: [batch_size, action_dim]
            
        Returns:
            quantiles: [batch_size, n_quantiles]
        """
        x = torch.cat([state, action], dim=-1)
        return self.net(x)
    
    def get_quantiles(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        분위수와 tau 값 반환
        
        Returns:
            quantiles: [batch_size, n_quantiles]
            taus: [n_quantiles]
        """
        quantiles = self.forward(state, action)
        return quantiles, self.tau

class QuantileHuberLoss(nn.Module):
    """
    Quantile Huber Loss for Distributional RL
    
    Combines quantile regression with Huber loss for stability
    """
    
    def __init__(self, kappa: float = 1.0):
        super().__init__()
        self.kappa = kappa
    
    def forward(self, q_pred: torch.Tensor, q_target: torch.Tensor, 
                taus: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            q_pred: 예측 분위수 [batch_size, n_quantiles]
            q_target: 타겟 분위수 [batch_size, n_quantiles] or [batch_size, 1]
            taus: 분위수 위치 [n_quantiles]
            weights: Importance sampling weights [batch_size]
            
        Returns:
            loss: scalar
        """
        # Expand dimensions for broadcasting
        if q_target.dim() == 2 and q_target.shape[1] == 1:
            q_target = q_target.expand(-1, q_pred.shape[1])
        
        # Pairwise differences
        td_error = q_target.unsqueeze(-1) - q_pred.unsqueeze(1)  # [batch, n_tau', n_tau]
        
        # Huber loss
        huber_loss = torch.where(
            td_error.abs() <= self.kappa,
            0.5 * td_error.pow(2),
            self.kappa * (td_error.abs() - 0.5 * self.kappa)
        )
        
        # Quantile regression
        taus = taus.view(1, 1, -1)  # [1, 1, n_tau]
        quantile_weight = (taus - (td_error < 0).float()).abs()
        
        # Weighted loss
        loss = (quantile_weight * huber_loss).mean(dim=1).sum(dim=1)  # [batch]
        
        # Apply importance sampling weights if provided
        if weights is not None:
            loss = loss * weights
        
        return loss.mean()

class DistributionalCritic(nn.Module):
    """
    Twin Quantile Critics for Distributional SAC
    
    두 개의 Q-network로 과대추정 방지
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 n_quantiles: int = 32):
        super().__init__()
        
        # Twin Q-networks
        self.q1 = QuantileNetwork(state_dim, action_dim, hidden_dim, n_quantiles)
        self.q2 = QuantileNetwork(state_dim, action_dim, hidden_dim, n_quantiles)
        
        # Target networks
        self.q1_target = QuantileNetwork(state_dim, action_dim, hidden_dim, n_quantiles)
        self.q2_target = QuantileNetwork(state_dim, action_dim, hidden_dim, n_quantiles)
        
        # Copy parameters to target
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Freeze target networks
        for param in self.q1_target.parameters():
            param.requires_grad = False
        for param in self.q2_target.parameters():
            param.requires_grad = False
        
        self.n_quantiles = n_quantiles
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both Q-networks
        
        Returns:
            q1_quantiles: [batch_size, n_quantiles]
            q2_quantiles: [batch_size, n_quantiles]
        """
        q1 = self.q1(state, action)
        q2 = self.q2(state, action)
        return q1, q2
    
    def get_min_q(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Conservative Q-value (minimum of two networks)
        
        Returns:
            min_q: [batch_size] - mean of quantiles
        """
        q1, q2 = self.forward(state, action)
        q1_mean = q1.mean(dim=-1)
        q2_mean = q2.mean(dim=-1)
        return torch.min(q1_mean, q2_mean)
    
    def get_cvar(self, state: torch.Tensor, action: torch.Tensor, alpha: float = 0.05) -> torch.Tensor:
        """
        Conditional Value at Risk from quantile distribution
        
        Args:
            alpha: CVaR level (e.g., 0.05 for 5% CVaR)
            
        Returns:
            cvar: [batch_size]
        """
        q1, q2 = self.forward(state, action)
        
        # Use minimum Q for conservative estimate
        q_min = torch.min(q1, q2)
        
        # Get lower alpha% quantiles
        k = max(1, int(self.n_quantiles * alpha))
        cvar, _ = torch.topk(q_min, k, dim=-1, largest=False)
        
        return cvar.mean(dim=-1)
    
    def soft_update(self, tau: float = 0.005):
        """
        Soft update of target networks
        
        θ_target = τ * θ + (1 - τ) * θ_target
        """
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

def extract_cvar_from_quantiles(quantiles: torch.Tensor, alpha: float = 0.05) -> torch.Tensor:
    """
    Extract CVaR from quantile distribution
    
    Args:
        quantiles: [batch_size, n_quantiles]
        alpha: CVaR level
        
    Returns:
        cvar: [batch_size]
    """
    batch_size, n_quantiles = quantiles.shape
    k = max(1, int(n_quantiles * alpha))
    
    # Get bottom k quantiles
    bottom_k, _ = torch.topk(quantiles, k, dim=-1, largest=False)
    
    return bottom_k.mean(dim=-1)

def compute_quantile_td_target(rewards: torch.Tensor, next_quantiles: torch.Tensor, 
                               dones: torch.Tensor, gamma: float = 0.99) -> torch.Tensor:
    """
    Compute TD target for quantile regression
    
    Args:
        rewards: [batch_size, 1]
        next_quantiles: [batch_size, n_quantiles]
        dones: [batch_size, 1]
        gamma: discount factor
        
    Returns:
        td_target: [batch_size, n_quantiles]
    """
    # Broadcast rewards and dones
    if rewards.dim() == 1:
        rewards = rewards.unsqueeze(1)
    if dones.dim() == 1:
        dones = dones.unsqueeze(1)
    
    # TD target: r + γ * (1 - done) * Q_next
    td_target = rewards + gamma * (1 - dones) * next_quantiles
    
    return td_target