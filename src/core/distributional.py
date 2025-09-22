# src/core/distributional.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict

class QuantileNetwork(nn.Module):
    """
    Enhanced Quantile Regression Network for Distributional RL

    Returns N quantile values for Q-distribution with optional risk-sensitive weighting
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 n_quantiles: int = 64, quantile_embedding_dim: int = 64):
        super().__init__()
        self.n_quantiles = n_quantiles
        self.quantile_embedding_dim = quantile_embedding_dim

        # Quantile embedding network (IQN-style)
        self.quantile_embedding = nn.Sequential(
            nn.Linear(1, quantile_embedding_dim),
            nn.ReLU(),
            nn.Linear(quantile_embedding_dim, quantile_embedding_dim)
        )

        # Main network with quantile conditioning
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # Quantile head (combines features with quantile embedding)
        self.quantile_head = nn.Linear(hidden_dim + quantile_embedding_dim, 1)

        # Quantile fractions (균등 분포)
        self.register_buffer(
            'tau',
            torch.linspace(0, 1, n_quantiles + 1)[:-1] + 1/(2*n_quantiles)
        )

        # Risk-sensitive weights (emphasize lower quantiles for risk-averse behavior)
        risk_weights = torch.exp(-2 * self.tau)  # Exponential decay
        self.register_buffer('risk_weights', risk_weights / risk_weights.sum())
    
    def forward(self, state: torch.Tensor, action: torch.Tensor,
                risk_sensitive: bool = False) -> torch.Tensor:
        """
        Args:
            state: [batch_size, state_dim]
            action: [batch_size, action_dim]
            risk_sensitive: Apply risk-sensitive weighting

        Returns:
            quantiles: [batch_size, n_quantiles]
        """
        batch_size = state.shape[0]

        # Extract features
        x = torch.cat([state, action], dim=-1)
        features = self.feature_net(x)  # [batch, hidden_dim]

        # Compute quantile values with embedding
        quantiles_list = []
        for i in range(self.n_quantiles):
            # Quantile embedding
            tau_i = self.tau[i].view(1, 1).expand(batch_size, 1)
            tau_embedding = self.quantile_embedding(tau_i)  # [batch, embedding_dim]

            # Combine features and quantile embedding
            combined = torch.cat([features, tau_embedding], dim=-1)
            quantile_value = self.quantile_head(combined)  # [batch, 1]
            quantiles_list.append(quantile_value)

        quantiles = torch.cat(quantiles_list, dim=-1)  # [batch, n_quantiles]

        # Apply risk-sensitive weighting if requested
        if risk_sensitive:
            quantiles = quantiles * self.risk_weights.unsqueeze(0)

        return quantiles
    
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
    Enhanced Quantile Huber Loss for Distributional RL

    Combines quantile regression with Huber loss for stability
    Supports dynamic kappa adjustment based on training progress
    """

    def __init__(self, kappa: float = 1.0, kappa_min: float = 0.5,
                 kappa_max: float = 2.0, dynamic_kappa: bool = True):
        super().__init__()
        self.kappa = kappa
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max
        self.dynamic_kappa = dynamic_kappa
        self.training_steps = 0

    def update_kappa(self, td_errors: torch.Tensor):
        """
        Dynamically adjust kappa based on TD error magnitude
        """
        if not self.dynamic_kappa:
            return

        # Compute average TD error magnitude
        avg_td_error = td_errors.abs().mean().item()

        # Adjust kappa: smaller errors -> smaller kappa (more L2-like)
        # Larger errors -> larger kappa (more L1-like for robustness)
        if avg_td_error < 0.1:
            target_kappa = self.kappa_min
        elif avg_td_error > 1.0:
            target_kappa = self.kappa_max
        else:
            # Linear interpolation
            ratio = (avg_td_error - 0.1) / 0.9
            target_kappa = self.kappa_min + ratio * (self.kappa_max - self.kappa_min)

        # Smooth update
        self.kappa = 0.95 * self.kappa + 0.05 * target_kappa
        self.training_steps += 1
    
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

        # Dynamic kappa adjustment
        if self.dynamic_kappa:
            self.update_kappa(td_error.mean(dim=[1, 2]))

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
    Enhanced Twin Quantile Critics for Distributional SAC

    두 개의 Q-network로 과대추정 방지
    Supports risk-sensitive evaluation and dynamic quantile adjustment
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 n_quantiles: int = 64, quantile_embedding_dim: int = 64):
        super().__init__()

        # Twin Q-networks with enhanced architecture
        self.q1 = QuantileNetwork(state_dim, action_dim, hidden_dim, n_quantiles, quantile_embedding_dim)
        self.q2 = QuantileNetwork(state_dim, action_dim, hidden_dim, n_quantiles, quantile_embedding_dim)

        # Target networks
        self.q1_target = QuantileNetwork(state_dim, action_dim, hidden_dim, n_quantiles, quantile_embedding_dim)
        self.q2_target = QuantileNetwork(state_dim, action_dim, hidden_dim, n_quantiles, quantile_embedding_dim)
        
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

    def get_var(self, state: torch.Tensor, action: torch.Tensor, alpha: float = 0.05) -> torch.Tensor:
        """
        Value at Risk from quantile distribution

        Args:
            alpha: VaR level (e.g., 0.05 for 5% VaR)

        Returns:
            var: [batch_size]
        """
        q1, q2 = self.forward(state, action)
        q_min = torch.min(q1, q2)

        # Get the alpha-quantile
        k = max(1, int(self.n_quantiles * alpha))
        var = q_min[:, k-1]  # The k-th smallest value

        return var

    def get_risk_sensitive_q(self, state: torch.Tensor, action: torch.Tensor,
                             risk_measure: str = "cvar", alpha: float = 0.05) -> torch.Tensor:
        """
        Get risk-sensitive Q-value based on specified risk measure

        Args:
            risk_measure: "cvar", "var", or "mean"
            alpha: Risk level for CVaR/VaR

        Returns:
            q_value: [batch_size]
        """
        if risk_measure == "cvar":
            return self.get_cvar(state, action, alpha)
        elif risk_measure == "var":
            return self.get_var(state, action, alpha)
        else:  # mean
            return self.get_min_q(state, action)
    
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


def compute_risk_sensitive_td_loss(
    q_pred: torch.Tensor,
    q_target: torch.Tensor,
    taus: torch.Tensor,
    risk_measure: str = "cvar",
    alpha: float = 0.05,
    weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute risk-sensitive TD loss with emphasis on tail risk

    Args:
        q_pred: Predicted quantiles [batch_size, n_quantiles]
        q_target: Target quantiles [batch_size, n_quantiles]
        taus: Quantile positions [n_quantiles]
        risk_measure: Risk measure to emphasize ("cvar", "var", "mean")
        alpha: Risk level
        weights: Importance sampling weights

    Returns:
        loss: Weighted loss with risk emphasis
    """
    n_quantiles = q_pred.shape[1]

    # Create risk-sensitive weights
    if risk_measure == "cvar":
        # Emphasize lower quantiles
        k = max(1, int(n_quantiles * alpha))
        risk_weights = torch.zeros_like(taus)
        risk_weights[:k] = 1.0 / k
    elif risk_measure == "var":
        # Focus on specific quantile
        k = max(1, int(n_quantiles * alpha))
        risk_weights = torch.zeros_like(taus)
        risk_weights[k-1] = 1.0
    else:  # mean
        risk_weights = torch.ones_like(taus) / n_quantiles

    # Standard quantile regression loss
    td_error = q_target.unsqueeze(-1) - q_pred.unsqueeze(1)
    quantile_weight = (taus.view(1, 1, -1) - (td_error < 0).float()).abs()

    # Apply risk weighting
    risk_weights = risk_weights.view(1, 1, -1)
    weighted_loss = (quantile_weight * td_error.abs() * risk_weights).sum(dim=[1, 2])

    if weights is not None:
        weighted_loss = weighted_loss * weights

    return weighted_loss.mean()


def get_quantile_statistics(quantiles: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Compute comprehensive statistics from quantile distribution

    Args:
        quantiles: [batch_size, n_quantiles]

    Returns:
        stats: Dictionary with mean, std, skewness, kurtosis, etc.
    """
    mean = quantiles.mean(dim=-1)
    std = quantiles.std(dim=-1)

    # Compute higher moments
    centered = quantiles - mean.unsqueeze(-1)
    variance = (centered ** 2).mean(dim=-1)
    skewness = (centered ** 3).mean(dim=-1) / (variance ** 1.5 + 1e-8)
    kurtosis = (centered ** 4).mean(dim=-1) / (variance ** 2 + 1e-8) - 3

    # Risk metrics
    cvar_05 = extract_cvar_from_quantiles(quantiles, alpha=0.05)
    cvar_10 = extract_cvar_from_quantiles(quantiles, alpha=0.10)

    return {
        "mean": mean,
        "std": std,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "cvar_05": cvar_05,
        "cvar_10": cvar_10,
        "median": quantiles[:, quantiles.shape[1]//2],
        "iqr": quantiles[:, 3*quantiles.shape[1]//4] - quantiles[:, quantiles.shape[1]//4]
    }