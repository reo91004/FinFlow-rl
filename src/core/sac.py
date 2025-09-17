# src/core/sac.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from src.core.networks import DirichletActor
from src.core.distributional import QuantileNetwork
from src.utils.logger import FinFlowLogger
from src.utils.optimizer_utils import polyak_update, auto_tune_temperature, clip_gradients

class DistributionalSAC:
    """
    Distributional Soft Actor-Critic
    
    Quantile Critics + Dirichlet Policy
    온라인 미세조정용
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 num_quantiles: int = 32,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 alpha: float = 0.2,
                 lr: float = 3e-4,
                 device: str = 'cpu'):
        """
        Args:
            state_dim: 상태 차원
            action_dim: 액션 차원
            hidden_dim: 은닉층 차원
            num_quantiles: 분위수 개수
            gamma: 할인율
            tau: 타겟 네트워크 업데이트 비율
            alpha: 엔트로피 계수
            lr: 학습률
            device: 디바이스
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.device = device
        
        # Networks
        self.actor = DirichletActor(state_dim, action_dim, hidden_dim).to(device)
        
        self.critic1 = QuantileNetwork(
            state_dim + action_dim, 1, hidden_dim, num_quantiles
        ).to(device)
        self.critic2 = QuantileNetwork(
            state_dim + action_dim, 1, hidden_dim, num_quantiles
        ).to(device)
        
        self.critic1_target = QuantileNetwork(
            state_dim + action_dim, 1, hidden_dim, num_quantiles
        ).to(device)
        self.critic2_target = QuantileNetwork(
            state_dim + action_dim, 1, hidden_dim, num_quantiles
        ).to(device)
        
        # Copy parameters
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=lr)
        
        # Entropy
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = alpha
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
        
        self.logger = FinFlowLogger("DistributionalSAC")
        self.logger.info(f"Distributional SAC 초기화 - quantiles={num_quantiles}")
    
    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> np.ndarray:
        """액션 선택"""
        with torch.no_grad():
            action, _ = self.actor(state, deterministic=deterministic)
            return action.cpu().numpy().squeeze()
    
    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """SAC 업데이트"""
        # Convert to tensors
        states = torch.FloatTensor(batch['states']).to(self.device)
        actions = torch.FloatTensor(batch['actions']).to(self.device)
        rewards = torch.FloatTensor(batch['rewards']).unsqueeze(-1).to(self.device)
        next_states = torch.FloatTensor(batch['next_states']).to(self.device)
        dones = torch.FloatTensor(batch['dones']).unsqueeze(-1).to(self.device)
        
        losses = {}
        
        # Update critics
        critic1_loss, critic2_loss = self._update_critics(
            states, actions, rewards, next_states, dones
        )
        losses['critic1_loss'] = critic1_loss
        losses['critic2_loss'] = critic2_loss
        
        # Update actor
        actor_loss, entropy = self._update_actor(states)
        losses['actor_loss'] = actor_loss
        losses['entropy'] = entropy
        
        # Update alpha
        alpha_loss = self._update_alpha(entropy)
        losses['alpha_loss'] = alpha_loss
        losses['alpha'] = self.alpha
        
        # Update target networks using polyak averaging
        polyak_update(self.critic1_target, self.critic1, self.tau)
        polyak_update(self.critic2_target, self.critic2, self.tau)
        
        return losses
    
    def _update_critics(self, states, actions, rewards, next_states, dones):
        """Critic 업데이트"""
        with torch.no_grad():
            # Sample next actions
            next_actions, next_log_probs = self.actor(next_states)
            
            # Get target quantiles
            next_sa = torch.cat([next_states, next_actions], dim=-1)
            target_q1 = self.critic1_target(next_sa)  # [batch, num_quantiles]
            target_q2 = self.critic2_target(next_sa)  # [batch, num_quantiles]
            
            # Min over critics
            target_q = torch.min(target_q1, target_q2)
            
            # Add entropy bonus
            target_q = target_q - self.alpha * next_log_probs.unsqueeze(-1)
            
            # Compute targets with distributional Bellman
            targets = rewards.unsqueeze(-1) + self.gamma * (1 - dones.unsqueeze(-1)) * target_q
        
        # Current Q estimates
        sa = torch.cat([states, actions], dim=-1)
        current_q1 = self.critic1(sa)
        current_q2 = self.critic2(sa)
        
        # Quantile Huber loss
        critic1_loss = self._quantile_huber_loss(current_q1, targets.detach())
        critic2_loss = self._quantile_huber_loss(current_q2, targets.detach())
        
        # Optimize with gradient clipping
        self.critic1_optimizer.zero_grad(set_to_none=True)
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=1.0)  # 5.0 → 1.0
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad(set_to_none=True)
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=1.0)  # 5.0 → 1.0
        self.critic2_optimizer.step()
        
        return critic1_loss.item(), critic2_loss.item()
    
    def _update_actor(self, states):
        """Actor 업데이트"""
        # Sample actions
        actions, log_probs = self.actor(states)
        
        # Get Q values
        sa = torch.cat([states, actions], dim=-1)
        q1 = self.critic1(sa).mean(dim=-1, keepdim=True)  # Mean over quantiles
        q2 = self.critic2(sa).mean(dim=-1, keepdim=True)
        q = torch.min(q1, q2)
        
        # Actor loss (maximize Q - alpha * log_prob)
        actor_loss = -(q - self.alpha * log_probs).mean()
        
        # Optimize with gradient clipping
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)  # 5.0 → 1.0
        self.actor_optimizer.step()
        
        return actor_loss.item(), log_probs.mean().item()
    
    def _update_alpha(self, entropy):
        """엔트로피 계수 업데이트 (auto-tuning)"""
        # Use auto temperature tuning
        log_prob = torch.tensor(entropy, device=self.device)
        self.alpha, alpha_loss = auto_tune_temperature(
            self.log_alpha,
            log_prob,
            self.target_entropy,
            self.alpha_optimizer,
            alpha_min=5e-4,
            alpha_max=0.2
        )
        
        return alpha_loss
    
    def _quantile_huber_loss(self, quantiles, targets, kappa=1.0):
        """Quantile Huber Loss"""
        # Expand for pairwise differences
        quantiles_expanded = quantiles.unsqueeze(2)  # [batch, N, 1]
        targets_expanded = targets.unsqueeze(1)  # [batch, 1, N]
        
        # Compute TD errors
        td_error = targets_expanded - quantiles_expanded  # [batch, N, N]
        
        # Huber loss
        huber_loss = torch.where(
            td_error.abs() <= kappa,
            0.5 * td_error.pow(2),
            kappa * (td_error.abs() - 0.5 * kappa)
        )
        
        # Quantile regression loss
        num_quantiles = quantiles.shape[1]
        tau = torch.arange(num_quantiles, device=self.device).float() / num_quantiles + 1 / (2 * num_quantiles)
        tau = tau.view(1, -1, 1)
        
        loss = torch.where(td_error < 0, (1 - tau) * huber_loss, tau * huber_loss)
        loss = loss.mean()
        
        return loss
    
    def _soft_update(self, source, target):
        """소프트 타겟 업데이트 (deprecated - use polyak_update instead)"""
        polyak_update(target, source, self.tau)
    
    def save(self, path: str):
        """모델 저장"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'critic1_target': self.critic1_target.state_dict(),
            'critic2_target': self.critic2_target.state_dict(),
            'log_alpha': self.log_alpha,
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict()
        }, path)
        self.logger.info(f"모델 저장: {path}")
    
    def load(self, path: str):
        """모델 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target'])
        self.log_alpha = checkpoint['log_alpha']
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        
        self.alpha = self.log_alpha.exp().item()
        self.logger.info(f"모델 로드: {path}")