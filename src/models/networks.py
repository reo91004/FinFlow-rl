# src/core/networks.py

"""
신경망 아키텍처 (Actor-Critic 네트워크)

목적: Dirichlet 정책, Q-네트워크, 가치 네트워크 구현
의존: PyTorch, Dirichlet 분포
사용처: IQLAgent (오프라인), BCell (온라인), TD3BC
역할: 포트폴리오 가중치 생성 및 가치 추정

구현 내용:
- DirichletActor: 유효한 포트폴리오 가중치 생성 (합=1)
- QNetwork: 상태-행동 Q값 추정
- ValueNetwork: 상태 가치 추정
- EnsembleQNetwork: Q-네트워크 앙상블 (불확실성 추정)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet
import numpy as np
from typing import Tuple, Optional

class DirichletActor(nn.Module):
    """
    Dirichlet Policy Network for Portfolio Allocation
    
    Outputs concentration parameters for Dirichlet distribution
    ensuring valid portfolio weights (simplex constraint)
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [256, 256],
                 min_concentration: float = 1.0, max_concentration: float = 50.0,  # 안정성 강화를 위해 하한 상향
                 dynamic_concentration: bool = False, crisis_scaling: float = 0.5,
                 base_concentration: float = 2.0, action_smoothing: bool = False,
                 smoothing_alpha: float = 0.95):
        super().__init__()

        self.action_dim = action_dim
        self.min_concentration = min_concentration  # 안정적인 하한값 (0.5)
        self.max_concentration = max_concentration
        self.base_concentration = base_concentration

        # Dynamic concentration settings
        self.dynamic_concentration = dynamic_concentration
        self.crisis_scaling = crisis_scaling

        # Action smoothing
        self.action_smoothing = action_smoothing
        self.smoothing_alpha = smoothing_alpha
        self.prev_action = None
        
        # Build network
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            input_dim = hidden_dim
        
        self.net = nn.Sequential(*layers)
        
        # Output layer for concentration parameters
        self.concentration_layer = nn.Linear(input_dim, action_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor, crisis_level: Optional[float] = None) -> torch.Tensor:
        """
        Forward pass to get concentration parameters with optional crisis adjustment

        Args:
            state: [batch_size, state_dim]
            crisis_level: Optional crisis level [0, 1] for dynamic adjustment

        Returns:
            concentration: [batch_size, action_dim]
        """
        features = self.net(state)

        # Softplus to ensure positive concentrations
        concentration = F.softplus(self.concentration_layer(features)) + self.min_concentration

        # Dynamic concentration adjustment based on crisis level
        if self.dynamic_concentration and crisis_level is not None:
            # Higher crisis -> lower concentration (more exploration)
            # Lower crisis -> higher concentration (more exploitation)
            adjustment_factor = 1.0 - (crisis_level * self.crisis_scaling)
            adjustment_factor = torch.clamp(torch.tensor(adjustment_factor), 0.5, 2.0)
            concentration = concentration * adjustment_factor
        elif self.dynamic_concentration:
            # Use base concentration when no crisis level provided
            concentration = concentration * (self.base_concentration / concentration.mean())

        # Clamp to max value for stability
        concentration = torch.clamp(concentration, self.min_concentration, self.max_concentration)

        return concentration
    
    def get_distribution(self, state: torch.Tensor, crisis_level: Optional[float] = None) -> Dirichlet:
        """
        Get Dirichlet distribution for the given state

        Args:
            state: [batch_size, state_dim]
            crisis_level: Optional crisis level for dynamic adjustment

        Returns:
            dist: Dirichlet distribution
        """
        concentration = self.forward(state, crisis_level)
        return Dirichlet(concentration)

    def sample(self, state: torch.Tensor, crisis_level: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action and compute log probability (for SAC/TD3BC compatibility)

        Args:
            state: [batch_size, state_dim]
            crisis_level: Optional crisis level for adjustment

        Returns:
            action: [batch_size, action_dim] - sampled portfolio weights
            log_prob: [batch_size, 1] - log probability of the action
        """
        dist = self.get_distribution(state, crisis_level)
        action = dist.rsample()  # Reparameterized sampling for backprop

        # Normalize to ensure sum = 1
        action = action / action.sum(dim=-1, keepdim=True)

        # Compute log probability
        log_prob = dist.log_prob(action).unsqueeze(-1)

        return action, log_prob

    def get_action(self, state: torch.Tensor, deterministic: bool = False,
                   crisis_level: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from Dirichlet distribution with optional smoothing

        Args:
            state: [batch_size, state_dim]
            deterministic: If True, return mode instead of sample
            crisis_level: Optional crisis level for dynamic adjustment

        Returns:
            action: [batch_size, action_dim] - portfolio weights
            log_prob: [batch_size, 1] - log probability (consistent shape with sample())
        """
        concentration = self.forward(state, crisis_level)

        # Create Dirichlet distribution
        dist = Dirichlet(concentration)

        if deterministic:
            # Mode of Dirichlet: (α_i - 1) / (Σα_j - K)
            # Only valid when all α_i > 1
            safe_conc = torch.clamp(concentration, min=1.01)
            action = (safe_conc - 1) / (safe_conc.sum(dim=-1, keepdim=True) - self.action_dim)
            # Normalize to ensure simplex
            action = action / action.sum(dim=-1, keepdim=True)
            log_prob = dist.log_prob(action).unsqueeze(-1)  # Shape 일치: [batch_size, 1]
        else:
            # Sample from distribution with stability
            action = dist.rsample()  # Reparameterized sampling

            # 강화된 안정화: 더 큰 엡실론으로 클램핑 및 재정규화
            action = torch.clamp(action, min=1e-5, max=1.0-1e-5)  # 더 큰 경계값 회피
            action = action / action.sum(dim=-1, keepdim=True)

            # log_prob 계산 시 추가 안정화
            # Simplex constraint를 만족하도록 action을 보정
            log_prob = dist.log_prob(action).unsqueeze(-1)  # Shape 일치: [batch_size, 1]

        # Apply action smoothing (EMA)
        if self.action_smoothing and not deterministic:
            if self.prev_action is not None and self.prev_action.shape == action.shape:
                action = self.smoothing_alpha * action + (1 - self.smoothing_alpha) * self.prev_action
                # Re-normalize to ensure simplex constraint
                action = action / action.sum(dim=-1, keepdim=True)
            self.prev_action = action.clone().detach()

        return action, log_prob
    
    def get_log_prob(self, state: torch.Tensor, action: torch.Tensor,
                     crisis_level: Optional[float] = None) -> torch.Tensor:
        """
        Compute log probability of action

        Args:
            state: [batch_size, state_dim]
            action: [batch_size, action_dim]
            crisis_level: Optional crisis level for dynamic adjustment

        Returns:
            log_prob: [batch_size]
        """
        concentration = self.forward(state, crisis_level)
        dist = Dirichlet(concentration)
        # Simplex constraint 보장
        action = torch.clamp(action, min=1e-5, max=1.0-1e-5)
        action = action / action.sum(dim=-1, keepdim=True)
        return dist.log_prob(action)  # 정규화된 action 사용
    
    def entropy(self, state: torch.Tensor, crisis_level: Optional[float] = None) -> torch.Tensor:
        """
        Compute entropy of the policy

        Args:
            state: [batch_size, state_dim]
            crisis_level: Optional crisis level for dynamic adjustment

        Returns:
            entropy: [batch_size]
        """
        concentration = self.forward(state, crisis_level)
        dist = Dirichlet(concentration)
        return dist.entropy()

    def reset_smoothing(self):
        """
        Reset action smoothing state
        """
        self.prev_action = None

class ValueNetwork(nn.Module):
    """
    State Value Network V(s) for IQL
    
    Estimates the value of being in a state
    """
    
    def __init__(self, state_dim: int, hidden_dims: list = [256, 256]):
        super().__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [batch_size, state_dim]
            
        Returns:
            value: [batch_size, 1]
        """
        return self.net(state)

class QNetwork(nn.Module):
    """
    Standard Q-Network Q(s,a) for IQL
    
    Used alongside ValueNetwork for advantage computation
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [256, 256]):
        super().__init__()
        
        layers = []
        input_dim = state_dim + action_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [batch_size, state_dim]
            action: [batch_size, action_dim]
            
        Returns:
            q_value: [batch_size, 1]
        """
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

class EnsembleQNetwork(nn.Module):
    """
    Ensemble of Q-Networks for uncertainty estimation
    
    Multiple Q-networks to reduce overestimation bias
    """
    
    def __init__(self, state_dim: int, action_dim: int, n_ensemble: int = 5,
                 hidden_dims: list = [256, 256]):
        super().__init__()
        
        self.n_ensemble = n_ensemble
        self.q_networks = nn.ModuleList([
            QNetwork(state_dim, action_dim, hidden_dims)
            for _ in range(n_ensemble)
        ])
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward through all Q-networks
        
        Args:
            state: [batch_size, state_dim]
            action: [batch_size, action_dim]
            
        Returns:
            q_values: [batch_size, n_ensemble]
        """
        q_values = []
        for q_net in self.q_networks:
            q = q_net(state, action)
            q_values.append(q)
        
        return torch.cat(q_values, dim=-1)
    
    def get_min_q(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Conservative Q-value (minimum across ensemble)
        
        Returns:
            min_q: [batch_size, 1]
        """
        q_values = self.forward(state, action)
        min_q, _ = torch.min(q_values, dim=-1, keepdim=True)
        return min_q
    
    def get_mean_q(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Mean Q-value across ensemble
        
        Returns:
            mean_q: [batch_size, 1]
        """
        q_values = self.forward(state, action)
        return q_values.mean(dim=-1, keepdim=True)


class QuantileNetwork(nn.Module):
    """
    Quantile Network for TQC (Truncated Quantile Critics)

    Implements quantile regression for distributional Q-values.
    Each critic outputs a distribution over returns using quantile regression.
    """

    def __init__(self, state_dim: int, action_dim: int, n_quantiles: int = 25,
                 hidden_dims: list = [256, 256], quantile_embedding_dim: int = 64):
        """
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            n_quantiles: Number of quantiles to estimate
            hidden_dims: Hidden layer dimensions
            quantile_embedding_dim: Dimension of quantile embedding
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_quantiles = n_quantiles
        self.quantile_embedding_dim = quantile_embedding_dim

        # Quantile fractions (τ) - centers of quantile bins
        quantiles = torch.linspace(0.0, 1.0, n_quantiles + 1)
        centers = (quantiles[:-1] + quantiles[1:]) / 2  # Center of each quantile bin
        self.register_buffer('quantile_fractions', centers)

        # Quantile embedding network (cos embedding)
        self.quantile_embedding = nn.Sequential(
            nn.Linear(quantile_embedding_dim, hidden_dims[-1]),  # Match base_net output dim
            nn.ReLU()
        )

        # Base network for state-action
        layers = []
        input_dim = state_dim + action_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            input_dim = hidden_dim

        self.base_net = nn.Sequential(*layers)

        # Output layer (combines base features with quantile embedding)
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_quantile_embedding(self, batch_size: int) -> torch.Tensor:
        """
        Generate cosine embedding for quantiles

        Args:
            batch_size: Batch size

        Returns:
            embedding: [batch_size * n_quantiles, quantile_embedding_dim]
        """
        # Create cosine features
        quantiles = self.quantile_fractions.unsqueeze(0).expand(batch_size, -1)
        quantiles = quantiles.reshape(-1, 1)  # [batch_size * n_quantiles, 1]

        # Cosine embedding
        i_pi = torch.arange(self.quantile_embedding_dim, device=quantiles.device).float()
        cos_embedding = torch.cos(quantiles * i_pi * np.pi)  # [batch_size * n_quantiles, embedding_dim]

        return cos_embedding

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing quantile Q-values

        Args:
            state: [batch_size, state_dim]
            action: [batch_size, action_dim]

        Returns:
            quantile_q_values: [batch_size, n_quantiles]
        """
        batch_size = state.shape[0]

        # Compute base features
        x = torch.cat([state, action], dim=-1)  # [batch_size, state_dim + action_dim]
        base_features = self.base_net(x)  # [batch_size, hidden_dim]

        # Expand for quantiles
        base_features = base_features.unsqueeze(1).expand(-1, self.n_quantiles, -1)
        base_features = base_features.reshape(batch_size * self.n_quantiles, -1)

        # Get quantile embedding
        quantile_embedding = self.get_quantile_embedding(batch_size)
        quantile_features = self.quantile_embedding(quantile_embedding)

        # Combine and output
        combined = base_features * quantile_features  # Element-wise multiplication
        quantile_q_values = self.output_layer(combined)  # [batch_size * n_quantiles, 1]

        # Reshape to [batch_size, n_quantiles]
        quantile_q_values = quantile_q_values.view(batch_size, self.n_quantiles)

        return quantile_q_values

    def get_truncated_quantiles(self, state: torch.Tensor, action: torch.Tensor,
                               top_quantiles_to_drop: int = 2) -> torch.Tensor:
        """
        Get truncated quantiles (drop top-k for conservative estimation)

        Args:
            state: [batch_size, state_dim]
            action: [batch_size, action_dim]
            top_quantiles_to_drop: Number of top quantiles to drop

        Returns:
            truncated_quantiles: [batch_size, n_quantiles - top_quantiles_to_drop]
        """
        quantile_q_values = self.forward(state, action)

        # Sort and drop top quantiles
        sorted_q, _ = torch.sort(quantile_q_values, dim=-1, descending=False)

        if top_quantiles_to_drop > 0:
            truncated = sorted_q[:, :-top_quantiles_to_drop]
        else:
            truncated = sorted_q

        return truncated