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
            log_prob: [batch_size] - log probability
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
            log_prob = dist.log_prob(action)
        else:
            # Sample from distribution with stability
            action = dist.rsample()  # Reparameterized sampling

            # 강화된 안정화: 더 큰 엡실론으로 클램핑 및 재정규화
            action = torch.clamp(action, min=1e-5, max=1.0-1e-5)  # 더 큰 경계값 회피
            action = action / action.sum(dim=-1, keepdim=True)

            # log_prob 계산 시 추가 안정화
            # Simplex constraint를 만족하도록 action을 보정
            log_prob = dist.log_prob(action)  # epsilon 없이 정규화된 action 사용

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