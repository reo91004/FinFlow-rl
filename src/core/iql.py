# src/agents/iql.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
from src.core.networks import DirichletActor
from src.utils.logger import FinFlowLogger

class ValueNetwork(nn.Module):
    """Value function V(s) for IQL"""
    
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # Layer norm for stability
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        return self.fc3(x)

class QNetwork(nn.Module):
    """Q-function Q(s, a) for IQL"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # Layer norm for stability
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        return self.fc3(x)

class IQLAgent:
    """
    Implicit Q-Learning (IQL) Agent for offline pretraining
    
    IQL learns from offline data without requiring importance sampling
    by using expectile regression for value function learning
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 expectile: float = 0.7,
                 temperature: float = 3.0,
                 discount: float = 0.99,
                 tau: float = 0.005,
                 learning_rate: float = 3e-4,
                 device: torch.device = torch.device("cpu")):
        """
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Hidden layer dimension
            expectile: Expectile for value function (0.5 = mean, 1.0 = max)
            temperature: Temperature for advantage weighting
            discount: Discount factor
            tau: Soft update coefficient
            learning_rate: Learning rate
            device: Device to use
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.expectile = expectile
        self.temperature = temperature
        self.discount = discount
        self.tau = tau
        self.device = device
        
        # Networks
        self.actor = DirichletActor(state_dim, action_dim, [hidden_dim, hidden_dim]).to(device)
        self.value = ValueNetwork(state_dim, hidden_dim).to(device)
        
        self.q1 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        
        # Target networks
        self.q1_target = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.q2_target = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        
        # Initialize targets
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=learning_rate)
        self.q_optimizer = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=learning_rate
        )
        
        # Statistics
        self.training_steps = 0
        self.losses = {
            'value_loss': [],
            'q_loss': [],
            'actor_loss': []
        }
        
        self.logger = FinFlowLogger("IQLAgent")
        self.logger.info(f"IQL Agent 초기화 - expectile={expectile}, temperature={temperature}")
    
    def update(self,
               states: torch.Tensor,
               actions: torch.Tensor,
               rewards: torch.Tensor,
               next_states: torch.Tensor,
               dones: torch.Tensor) -> Dict[str, float]:
        """
        Update IQL agent
        
        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of done flags
            
        Returns:
            losses: Dictionary of losses
        """
        # Update value function
        value_loss = self._update_value(states, actions)
        
        # Update Q-functions
        q_loss = self._update_q(states, actions, rewards, next_states, dones)
        
        # Update actor
        actor_loss = self._update_actor(states, actions)
        
        # Soft update target networks
        self._soft_update(self.q1_target, self.q1)
        self._soft_update(self.q2_target, self.q2)
        
        self.training_steps += 1
        
        # Track losses for statistics
        self.losses['value_loss'].append(value_loss)
        self.losses['q_loss'].append(q_loss)
        self.losses['actor_loss'].append(actor_loss)
        
        return {
            'value_loss': value_loss,
            'q_loss': q_loss,
            'actor_loss': actor_loss
        }
    
    def _update_value(self, states: torch.Tensor, actions: torch.Tensor) -> float:
        """
        Update value function using expectile regression
        
        V(s) ≈ E_τ[Q(s,a)] where τ is the expectile
        """
        with torch.no_grad():
            # Get Q-values for the given state-action pairs
            q1 = self.q1_target(states, actions)
            q2 = self.q2_target(states, actions)
            q = torch.min(q1, q2)
        
        # Current value estimate
        value = self.value(states)
        
        # Expectile regression loss
        diff = q - value
        weight = torch.where(diff > 0, self.expectile, 1 - self.expectile)
        value_loss = (weight * diff.pow(2)).mean()
        
        # Optimize
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value.parameters(), 1.0)
        self.value_optimizer.step()
        
        return value_loss.item()
    
    def _update_q(self,
                  states: torch.Tensor,
                  actions: torch.Tensor,
                  rewards: torch.Tensor,
                  next_states: torch.Tensor,
                  dones: torch.Tensor) -> float:
        """
        Update Q-functions
        
        Q(s,a) ← r + γ * V(s')
        """
        with torch.no_grad():
            # Next value (not next Q!)
            next_value = self.value(next_states)
            target = rewards + self.discount * (1 - dones) * next_value
        
        # Current Q estimates
        q1 = self.q1(states, actions)
        q2 = self.q2(states, actions)
        
        # MSE loss
        q1_loss = F.mse_loss(q1, target)
        q2_loss = F.mse_loss(q2, target)
        q_loss = q1_loss + q2_loss
        
        # Optimize
        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            1.0
        )
        self.q_optimizer.step()
        
        return q_loss.item()
    
    def _update_actor(self, states: torch.Tensor, actions: torch.Tensor) -> float:
        """
        Update actor using advantage weighted regression
        
        π(a|s) ∝ exp(A(s,a) / β) where A(s,a) = Q(s,a) - V(s)
        """
        with torch.no_grad():
            # Compute advantages
            q1 = self.q1(states, actions)
            q2 = self.q2(states, actions)
            q = torch.min(q1, q2)
            value = self.value(states)
            advantage = q - value
            
            # Advantage weighting
            weights = torch.exp(advantage / self.temperature)
            weights = torch.clamp(weights, max=100.0)  # Prevent overflow
        
        # Get log probabilities from actor
        action_dist = self.actor.get_distribution(states)
        log_probs = action_dist.log_prob(actions)
        
        # Weighted regression loss
        actor_loss = -(weights * log_probs).mean()
        
        # Optimize
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        return actor_loss.item()
    
    def _soft_update(self, target: nn.Module, source: nn.Module):
        """Soft update target network"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1 - self.tau) * target_param.data
            )
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Select action from policy
        
        Args:
            state: Current state
            deterministic: Whether to use deterministic policy
            
        Returns:
            action: Selected action
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, _ = self.actor.get_action(state_tensor, deterministic)
            return action.cpu().numpy().squeeze()
    
    def save(self, path: str):
        """Save model with metadata"""
        import datetime
        
        # 메타데이터 생성
        metadata = {
            'checkpoint_type': 'iql',  # IQL 체크포인트 표시
            'timestamp': datetime.datetime.now().isoformat(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'hidden_dim': self.hidden_dim,
            'expectile': self.expectile,
            'temperature': self.temperature,
            'discount': self.discount,
            'tau': self.tau,
            'framework_version': '2.0',
            'training_mode': 'iql',
            'total_steps': self.training_steps
        }
        
        torch.save({
            'actor': self.actor.state_dict(),
            'value': self.value.state_dict(),
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
            'q1_target': self.q1_target.state_dict(),
            'q2_target': self.q2_target.state_dict(),
            'training_steps': self.training_steps,
            'metadata': metadata  # 메타데이터 추가
        }, path)
        self.logger.info(f"모델 저장 (메타데이터 포함): {path}")
    
    def load(self, path: str):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.value.load_state_dict(checkpoint['value'])
        self.q1.load_state_dict(checkpoint['q1'])
        self.q2.load_state_dict(checkpoint['q2'])
        self.q1_target.load_state_dict(checkpoint['q1_target'])
        self.q2_target.load_state_dict(checkpoint['q2_target'])
        self.training_steps = checkpoint['training_steps']
        self.logger.info(f"모델 로드: {path}")