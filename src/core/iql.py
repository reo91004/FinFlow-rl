# src/core/iql.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
from src.core.networks import DirichletActor, ValueNetwork, QNetwork
from src.utils.logger import FinFlowLogger

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
                 device: torch.device = torch.device("cpu"),
                 warmup_steps: int = 0,
                 value_regularization: float = 0.0):
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

        # Learning rate warmup and regularization
        self.warmup_steps = warmup_steps
        self.base_lr = learning_rate
        self.value_regularization = value_regularization
        
        # Networks
        self.actor = DirichletActor(state_dim, action_dim, [hidden_dim, hidden_dim]).to(device)
        self.value = ValueNetwork(state_dim, [hidden_dim, hidden_dim]).to(device)

        self.q1 = QNetwork(state_dim, action_dim, [hidden_dim, hidden_dim]).to(device)
        self.q2 = QNetwork(state_dim, action_dim, [hidden_dim, hidden_dim]).to(device)

        # Target networks
        self.q1_target = QNetwork(state_dim, action_dim, [hidden_dim, hidden_dim]).to(device)
        self.q2_target = QNetwork(state_dim, action_dim, [hidden_dim, hidden_dim]).to(device)
        
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

        # Update learning rate if in warmup phase
        if self.training_steps < self.warmup_steps:
            self._adjust_learning_rate()

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

        # Add value regularization if specified
        if self.value_regularization > 0:
            value_loss = value_loss + self.value_regularization * value.pow(2).mean()
        
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
            
            # Advantage weighting with improved numerical stability
            # Clamp advantage before exponential to prevent overflow
            clamped_advantage = torch.clamp(advantage, min=-10.0, max=10.0)
            weights = torch.exp(clamped_advantage / self.temperature)
            weights = torch.clamp(weights, min=1e-8, max=100.0)  # Prevent overflow and underflow
        
        # Get log probabilities from actor
        action_dist = self.actor.get_distribution(states)
        log_probs = action_dist.log_prob(actions)

        # NaN 필터링 추가 - NaN이나 Inf가 있는 샘플 제외
        valid_mask = ~(torch.isnan(log_probs) | torch.isinf(log_probs))

        if valid_mask.sum() == 0:
            # 모든 샘플이 invalid하면 업데이트 스킵
            self.logger.warning(f"All log_probs are NaN/Inf, skipping update")
            return 0.0

        # 유효한 샘플만 사용
        valid_log_probs = log_probs[valid_mask]
        valid_weights = weights[valid_mask]

        # Clamp log probabilities to prevent extreme values
        # 30차원 Dirichlet의 정상 범위: 40-80, 여유있게 설정
        valid_log_probs = torch.clamp(valid_log_probs, min=-200, max=100)

        # Weighted regression loss with valid samples only
        actor_loss = -(valid_weights * valid_log_probs).mean()

        # Check for NaN and skip update if detected
        if torch.isnan(actor_loss) or torch.isinf(actor_loss):
            self.logger.warning(f"Actor loss is NaN/Inf: {actor_loss.item()}, skipping update")
            return 0.0  # Return 0 to indicate no update

        # Optimize
        self.actor_optimizer.zero_grad()
        actor_loss.backward()

        # Check for NaN gradients before step
        grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            self.logger.warning(f"Actor gradient is NaN/Inf: {grad_norm.item()}, skipping update")
            self.actor_optimizer.zero_grad()  # Clear gradients
            return 0.0

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

    def _adjust_learning_rate(self):
        """
        Adjust learning rate during warmup phase
        """
        progress = self.training_steps / max(1, self.warmup_steps)
        lr = self.base_lr * min(1.0, progress)

        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = lr
        for param_group in self.value_optimizer.param_groups:
            param_group['lr'] = lr
        for param_group in self.q_optimizer.param_groups:
            param_group['lr'] = lr
    
    def save(self, path: str):
        """Save model with SafeTensors"""
        import datetime
        import json
        from pathlib import Path
        from safetensors.torch import save_file

        # 경로 준비
        save_path = Path(path)
        if save_path.suffix == '.pt':
            save_path = save_path.parent / save_path.stem  # Remove .pt extension
        save_path.mkdir(exist_ok=True, parents=True)

        # 1. 모델 가중치를 safetensors로 저장
        model_tensors = {}

        # Actor
        for key, value in self.actor.state_dict().items():
            if isinstance(value, torch.Tensor):
                model_tensors[f"actor.{key}"] = value

        # Value
        for key, value in self.value.state_dict().items():
            if isinstance(value, torch.Tensor):
                model_tensors[f"value.{key}"] = value

        # Q networks
        for key, value in self.q1.state_dict().items():
            if isinstance(value, torch.Tensor):
                model_tensors[f"q1.{key}"] = value

        for key, value in self.q2.state_dict().items():
            if isinstance(value, torch.Tensor):
                model_tensors[f"q2.{key}"] = value

        # Q targets
        for key, value in self.q1_target.state_dict().items():
            if isinstance(value, torch.Tensor):
                model_tensors[f"q1_target.{key}"] = value

        for key, value in self.q2_target.state_dict().items():
            if isinstance(value, torch.Tensor):
                model_tensors[f"q2_target.{key}"] = value

        # 모델 저장
        save_file(model_tensors, save_path / "model.safetensors")

        # 2. 메타데이터를 JSON으로 저장
        metadata = {
            'checkpoint_type': 'iql',
            'timestamp': datetime.datetime.now().isoformat(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'expectile': self.expectile,
            'temperature': self.temperature,
            'discount': self.discount,
            'tau': self.tau,
            'framework_version': '3.0',  # SafeTensors version
            'training_mode': 'iql',
            'training_steps': self.training_steps
        }

        with open(save_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"모델 저장 (SafeTensors): {save_path}")
    
    def load(self, path: str):
        """Load model (SafeTensors or legacy format)"""
        from pathlib import Path
        from safetensors.torch import load_file
        import json

        load_path = Path(path)

        # SafeTensors 형식 체크
        if load_path.is_dir() and (load_path / "model.safetensors").exists():
            # SafeTensors 형식 로드
            self.logger.info(f"SafeTensors 체크포인트 로드: {load_path}")

            # 메타데이터 로드
            with open(load_path / "metadata.json", 'r') as f:
                metadata = json.load(f)

            self.training_steps = metadata.get('training_steps', 0)

            # 모델 가중치 로드
            model_tensors = load_file(load_path / "model.safetensors")

            # Actor
            actor_state = {}
            for key, value in model_tensors.items():
                if key.startswith("actor."):
                    actor_state[key.replace("actor.", "")] = value
            self.actor.load_state_dict(actor_state)

            # Value
            value_state = {}
            for key, value in model_tensors.items():
                if key.startswith("value."):
                    value_state[key.replace("value.", "")] = value
            self.value.load_state_dict(value_state)

            # Q networks
            q1_state = {}
            for key, value in model_tensors.items():
                if key.startswith("q1.") and not key.startswith("q1_target."):
                    q1_state[key.replace("q1.", "")] = value
            self.q1.load_state_dict(q1_state)

            q2_state = {}
            for key, value in model_tensors.items():
                if key.startswith("q2.") and not key.startswith("q2_target."):
                    q2_state[key.replace("q2.", "")] = value
            self.q2.load_state_dict(q2_state)

            # Q targets
            q1_target_state = {}
            for key, value in model_tensors.items():
                if key.startswith("q1_target."):
                    q1_target_state[key.replace("q1_target.", "")] = value
            self.q1_target.load_state_dict(q1_target_state)

            q2_target_state = {}
            for key, value in model_tensors.items():
                if key.startswith("q2_target."):
                    q2_target_state[key.replace("q2_target.", "")] = value
            self.q2_target.load_state_dict(q2_target_state)

            self.logger.info(f"SafeTensors 모델 로드 완료: {load_path}")

        else:
            # 기존 .pt 형식 로드 (호환성)
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            self.actor.load_state_dict(checkpoint['actor'])
            self.value.load_state_dict(checkpoint['value'])
            self.q1.load_state_dict(checkpoint['q1'])
            self.q2.load_state_dict(checkpoint['q2'])
            self.q1_target.load_state_dict(checkpoint['q1_target'])
            self.q2_target.load_state_dict(checkpoint['q2_target'])
            self.training_steps = checkpoint['training_steps']
            self.logger.info(f"레거시 모델 로드: {path}")