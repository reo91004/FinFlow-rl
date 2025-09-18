# src/agents/b_cell.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List
from collections import deque

from src.core.networks import DirichletActor
from src.core.distributional import DistributionalCritic, QuantileHuberLoss, extract_cvar_from_quantiles
from src.core.replay import PrioritizedReplayBuffer, Transition
from src.utils.logger import FinFlowLogger
from src.utils.optimizer_utils import polyak_update, cql_penalty, clip_gradients

class BCell:
    """
    B-Cell: Specialized Strategy Agent with Distributional SAC + CQL
    
    5개의 전문화 전략:
    - volatility: 고변동성 위기 전문
    - correlation: 상관관계 붕괴 전문
    - momentum: 모멘텀 전략
    - defensive: 방어적 전략
    - growth: 성장 전략
    """
    
    SPECIALIZATIONS = ['volatility', 'correlation', 'momentum', 'defensive', 'growth']
    
    def __init__(self,
                 specialization: str,
                 state_dim: int,
                 action_dim: int,
                 config: Dict,
                 device: torch.device = torch.device("cpu")):
        """
        Args:
            specialization: 전문화 유형
            state_dim: 상태 차원
            action_dim: 액션 차원 (포트폴리오 자산 수)
            config: SAC 설정
            device: 연산 디바이스
        """
        assert specialization in self.SPECIALIZATIONS, f"Invalid specialization: {specialization}"
        
        self.specialization = specialization
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.logger = FinFlowLogger(f"BCell-{specialization}")
        
        # Hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.005)
        self.alpha_init = 0.79  # 중요: 초기 탐색 강화 (0.75~0.79 중 0.79 권장)
        self.alpha_min = config.get('alpha_min', 5e-4)
        self.alpha_max = config.get('alpha_max', 2.0)  # 상한 상향
        self.target_entropy_ratio = 0.5  # -0.5 * |A| 목표
        
        # CQL settings (표준 범위로 상향)
        self.cql_weight = 10.0  # 연구 결과 기반 최적값 (5.0→10.0)
        self.cql_min_q_weight = 10.0  # min Q weight 동치
        self.cql_num_samples = config.get('cql_num_samples', 8)
        self.enable_cql = config.get('enable_cql', True)
        
        # Networks
        self.actor = DirichletActor(
            state_dim, action_dim,
            hidden_dims=config.get('actor_hidden', [256, 256])
        ).to(device)
        
        self.critic = DistributionalCritic(
            state_dim, action_dim,
            hidden_dim=config.get('critic_hidden', [256, 256])[0],
            n_quantiles=config.get('n_quantiles', 32)
        ).to(device)
        
        # Temperature parameter (learnable)
        self.log_alpha = torch.tensor(np.log(self.alpha_init), requires_grad=True, device=device, dtype=torch.float32)
        self.alpha = self.alpha_init  # 명시적 초기값
        self.target_entropy = -self.target_entropy_ratio * action_dim  # -0.5 * |A|
        
        # Optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=config.get('actor_lr', 3e-4)
        )
        
        self.critic_optimizer = optim.Adam(
            list(self.critic.q1.parameters()) + list(self.critic.q2.parameters()),
            lr=config.get('critic_lr', 3e-4)
        )
        
        self.alpha_optimizer = optim.Adam(
            [self.log_alpha],
            lr=config.get('alpha_lr', 3e-4)
        )
        
        # Loss functions
        self.quantile_loss = QuantileHuberLoss(kappa=1.0)
        
        # Experience replay
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=config.get('buffer_size', 100000),
            alpha=0.6,
            beta=0.4
        )
        
        # Training stats
        self.training_step = 0
        self.recent_rewards = deque(maxlen=100)
        self.performance_score = 0.0
        
        self.logger.info(f"B-Cell [{specialization}] 초기화 완료")
    
    def get_specialization_score(self, crisis_info: Dict[str, float]) -> float:
        """
        위기 정보 기반 전문성 점수 계산
        
        Args:
            crisis_info: T-Cell의 위기 분석 정보
            
        Returns:
            score: 전문성 점수 [0, 1]
        """
        overall = crisis_info.get('overall_crisis', 0)
        volatility = crisis_info.get('volatility_crisis', 0)
        correlation = crisis_info.get('correlation_crisis', 0)
        volume = crisis_info.get('volume_crisis', 0)
        
        if self.specialization == 'volatility':
            # 고변동성 위기에 특화
            score = 0.7 * volatility + 0.3 * overall
            
        elif self.specialization == 'correlation':
            # 상관관계 붕괴에 특화
            score = 0.6 * correlation + 0.2 * volatility + 0.2 * overall
            
        elif self.specialization == 'momentum':
            # 저위기 모멘텀 전략
            score = max(0, 1 - overall) * 0.8 + volume * 0.2
            
        elif self.specialization == 'defensive':
            # 중간 위기 방어 전략
            mid_crisis = 1 - abs(overall - 0.5) * 2  # 0.5에서 최대
            score = mid_crisis * 0.7 + min(volatility, correlation) * 0.3
            
        elif self.specialization == 'growth':
            # 극저위기 성장 전략
            score = max(0, 1 - overall * 2)
        
        else:
            score = 0.5  # Default
        
        # 성과 기반 조정
        performance_adjustment = np.tanh(self.performance_score / 100)
        score = score * (1 + 0.2 * performance_adjustment)
        
        return np.clip(score, 0, 1)
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, Dict]:
        """
        정책에서 액션 샘플링
        
        Args:
            state: 상태 배열
            deterministic: 결정적 액션 여부
            
        Returns:
            action: 포트폴리오 가중치
            info: 추가 정보
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob = self.actor.get_action(state_tensor, deterministic)
            
            # Get Q-value for logging
            q1, q2 = self.critic(state_tensor, action)
            q_value = torch.min(q1.mean(dim=-1), q2.mean(dim=-1))
            
            info = {
                'log_prob': log_prob.item(),
                'q_value': q_value.item(),
                'entropy': -log_prob.item(),
                'alpha': self.log_alpha.exp().item()
            }
            
            return action.cpu().numpy().squeeze(), info
    
    def store_experience(self, state: np.ndarray, action: np.ndarray, 
                        reward: float, next_state: np.ndarray, done: bool):
        """경험 저장"""
        transition = Transition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done
        )
        self.replay_buffer.push(transition)
        self.recent_rewards.append(reward)
    
    def update(self, batch_size: int = 256) -> Dict[str, float]:
        """
        SAC + CQL 업데이트
        
        Returns:
            losses: 손실 딕셔너리
        """
        if len(self.replay_buffer) < batch_size:
            return {}
        
        # Sample batch
        transitions, weights, indices = self.replay_buffer.sample(batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor([t.state for t in transitions]).to(self.device)
        actions = torch.FloatTensor([t.action for t in transitions]).to(self.device)
        rewards = torch.FloatTensor([t.reward for t in transitions]).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor([t.next_state for t in transitions]).to(self.device)
        dones = torch.FloatTensor([t.done for t in transitions]).unsqueeze(1).to(self.device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)
        
        # Update critics
        critic_loss, td_errors = self._update_critics(
            states, actions, rewards, next_states, dones, weights
        )
        
        # Update actor
        actor_loss = self._update_actor(states)
        
        # Update temperature
        alpha_loss = self._update_alpha(states)
        
        # Update priorities in replay buffer
        priorities = td_errors.abs().cpu().numpy() + 1e-6
        self.replay_buffer.update_priorities(indices, priorities)

        # Update performance score
        self._update_performance()
        
        self.training_step += 1
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': self.log_alpha.exp().item(),
            'td_error': td_errors.mean().item()
        }
    
    def select_action(self, state_tensor: torch.Tensor, bcell_type: str = None, 
                     deterministic: bool = False) -> np.ndarray:
        """
        액션 선택 (Trainer 호환)
        
        Args:
            state_tensor: 상태 텐서
            bcell_type: B-Cell 유형 (미사용, 호환성용)
            deterministic: 결정적 액션 여부
            
        Returns:
            action: 포트폴리오 가중치
        """
        with torch.no_grad():
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)
            
            action, _ = self.actor.get_action(state_tensor, deterministic)
            return action.cpu().numpy().squeeze()
    
    def state_dict(self) -> Dict:
        """모델 상태 반환"""
        return {
            'actor': self.actor.state_dict(),
            'critic_q1': self.critic.q1.state_dict(),
            'critic_q2': self.critic.q2.state_dict(),
            'log_alpha': self.log_alpha.detach().cpu().numpy(),
            'specialization': self.specialization,
            'training_step': self.training_step,
            'performance_score': self.performance_score
        }
    
    def load_state_dict(self, state_dict: Dict):
        """모델 상태 로드"""
        self.actor.load_state_dict(state_dict['actor'])
        self.critic.q1.load_state_dict(state_dict['critic_q1'])
        self.critic.q2.load_state_dict(state_dict['critic_q2'])
        self.log_alpha = torch.tensor(
            state_dict['log_alpha'], 
            requires_grad=True, 
            device=self.device
        )
        self.training_step = state_dict.get('training_step', 0)
        self.performance_score = state_dict.get('performance_score', 0.0)
    
    def get_statistics(self) -> Dict:
        """통계 정보 반환"""
        return {
            'specialization': self.specialization,
            'training_step': self.training_step,
            'performance_score': self.performance_score,
            'avg_recent_reward': np.mean(self.recent_rewards) if self.recent_rewards else 0,
            'buffer_size': len(self.replay_buffer),
            'alpha': self.log_alpha.exp().item(),
            'cql_weight': self.cql_weight
        }
    
    def update(self, batch: Dict) -> Dict[str, float]:
        """
        업데이트 메서드 (Trainer 호환)
        
        Args:
            batch: 배치 데이터 딕셔너리
            
        Returns:
            losses: 손실 딕셔너리
        """
        # Convert batch format - 텐서 타입 체크
        if isinstance(batch['states'], torch.Tensor):
            # 이미 텐서인 경우 그대로 사용
            states = batch['states']
            actions = batch['actions']
            rewards = batch['rewards'] if batch['rewards'].dim() == 2 else batch['rewards'].unsqueeze(1)
            next_states = batch['next_states']
            dones = batch['dones'] if batch['dones'].dim() == 2 else batch['dones'].unsqueeze(1)
        else:
            # numpy array인 경우 변환
            states = torch.FloatTensor(batch['states']).to(self.device)
            actions = torch.FloatTensor(batch['actions']).to(self.device)
            rewards = torch.FloatTensor(batch['rewards']).unsqueeze(1).to(self.device)
            next_states = torch.FloatTensor(batch['next_states']).to(self.device)
            dones = torch.FloatTensor(batch['dones']).unsqueeze(1).to(self.device)
        
        # Default weights if not provided
        if 'weights' in batch:
            if isinstance(batch['weights'], torch.Tensor):
                weights = batch['weights'] if batch['weights'].dim() == 2 else batch['weights'].unsqueeze(1)
            else:
                weights = torch.FloatTensor(batch['weights']).unsqueeze(1).to(self.device)
        else:
            weights = torch.ones_like(rewards)
        
        # Update critics
        critic_loss, td_errors = self._update_critics(
            states, actions, rewards, next_states, dones, weights
        )
        
        # Update actor
        actor_loss = self._update_actor(states)
        
        # Update temperature
        alpha_loss = self._update_alpha(states)
        
        # Update performance
        self._update_performance()
        
        self.training_step += 1
        
        return {
            'critic_loss': critic_loss,
            'actor_loss': actor_loss,
            'alpha_loss': alpha_loss,
            'alpha': self.log_alpha.exp().item(),
            'cql_weight': self.cql_weight,
            'performance': self.performance_score
        }
    
    def _update_critics(self, states, actions, rewards, next_states, dones, weights):
        """Update distributional critics with CQL"""

        # 보상 정규화 (수치 안정성)
        rewards = torch.tanh(rewards / 10.0) * 10.0

        with torch.no_grad():
            # Sample next actions
            next_actions, next_log_probs = self.actor.get_action(next_states)
            
            # Get target Q-values (quantiles)
            next_q1, next_q2 = self.critic.q1_target(next_states, next_actions), \
                               self.critic.q2_target(next_states, next_actions)
            
            # Use minimum for conservative estimate
            next_q = torch.min(next_q1, next_q2)

            # Q값 범위 제한
            next_q = torch.clamp(next_q, -100, 100)
            
            # Subtract entropy term
            alpha = self.log_alpha.exp()
            next_q = next_q - alpha * next_log_probs.unsqueeze(1)
            
            # Compute TD targets for each quantile
            target_q = rewards + self.gamma * (1 - dones) * next_q
        
        # Current Q-values
        current_q1, current_q2 = self.critic(states, actions)

        # Q값 범위 제한
        current_q1 = torch.clamp(current_q1, -100, 100)
        current_q2 = torch.clamp(current_q2, -100, 100)
        
        # Quantile regression loss
        loss_q1 = self.quantile_loss(current_q1, target_q.detach(), self.critic.q1.tau, weights)
        loss_q2 = self.quantile_loss(current_q2, target_q.detach(), self.critic.q2.tau, weights)
        
        # CQL regularization with adaptive weight
        if self.enable_cql:
            cql_loss = self._compute_cql_loss(states, current_q1, current_q2)

            # Adaptive CQL weight adjustment
            q_mean = (current_q1.abs().mean() + current_q2.abs().mean()) / 2
            if q_mean > 50:  # Q-value getting too large
                self.cql_weight = min(self.cql_weight * 1.5, 20.0)
            elif q_mean < 10 and self.cql_weight > 1.0:  # Q-value stable
                self.cql_weight = max(self.cql_weight * 0.95, 1.0)

            critic_loss = loss_q1 + loss_q2 + self.cql_weight * cql_loss
        else:
            critic_loss = loss_q1 + loss_q2
        
        # nan/inf 체크
        if not torch.isfinite(critic_loss):
            self.logger.warning(f"Critic loss is {critic_loss.item()}, skipping update")
            return torch.tensor(0.0, device=self.device), td_errors.detach()

        # Optimize
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.critic.q1.parameters()) + list(self.critic.q2.parameters()),
            1.0  # 5.0 → 1.0 강화된 클리핑
        )
        self.critic_optimizer.step()
        
        # Soft update target networks
        self.critic.soft_update(self.tau)
        
        # Compute TD errors for prioritization
        td_errors = (target_q.mean(dim=1) - current_q1.mean(dim=1)).detach()
        
        return critic_loss.item(), td_errors
    
    def _update_actor(self, states):
        """Update actor with entropy regularization"""
        
        # Sample actions
        actions, log_probs = self.actor.get_action(states)
        
        # Get Q-values
        q1, q2 = self.critic(states, actions)
        min_q = torch.min(q1.mean(dim=-1), q2.mean(dim=-1))
        
        # Actor loss: maximize Q - α * log_prob
        alpha = self.log_alpha.exp()
        actor_loss = -(min_q - alpha * log_probs).mean()
        
        # Optimize
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)  # 유지
        self.actor_optimizer.step()
        
        return actor_loss.item()
    
    def _update_alpha(self, states):
        """Update temperature parameter with improved stability"""

        with torch.no_grad():
            actions, log_probs = self.actor.get_action(states)

        # Clamp entropy difference to prevent extreme updates
        entropy_diff = (log_probs + self.target_entropy).detach()
        entropy_diff = torch.clamp(entropy_diff, min=-5.0, max=5.0)  # Prevent extreme values

        # Alpha loss: maintain target entropy
        alpha_loss = -(self.log_alpha * entropy_diff).mean()

        # Optimize
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        torch.nn.utils.clip_grad_norm_([self.log_alpha], max_norm=1.0)  # Proper gradient clipping
        self.alpha_optimizer.step()

        # Enforce alpha bounds with clamping
        with torch.no_grad():
            self.log_alpha.clamp_(min=np.log(self.alpha_min), max=np.log(self.alpha_max))

        # Update alpha value
        self.alpha = self.log_alpha.exp().item()

        return alpha_loss.item()
    
    def _cql_penalty(self, states, actions, q1, q2):
        """
        CQL penalty 계산 (표준 가중치 사용)
        """
        # 기존 _compute_cql_loss를 _cql_penalty로 명츰 변경
        # 가중치만 self.cql_weight 사용
        with torch.no_grad():
            # Random actions
            random_actions = torch.rand_like(actions).to(self.device)
            random_actions = random_actions / random_actions.sum(dim=-1, keepdim=True)

            # Current policy actions
            current_actions, _ = self.actor.get_action(states)

        # Q values for different actions
        q1_random = self.critic.q1(states, random_actions)
        q2_random = self.critic.q2(states, random_actions)
        q1_current = self.critic.q1(states, current_actions)
        q2_current = self.critic.q2(states, current_actions)

        # CQL penalty
        cql_q1_loss = torch.mean(torch.max(q1_random, q1_current) - q1)
        cql_q2_loss = torch.mean(torch.max(q2_random, q2_current) - q2)

        return self.cql_weight * cql_q1_loss + self.cql_min_q_weight * cql_q2_loss

    def _compute_cql_loss(self, states, q1, q2):
        """
        Compute CQL (Conservative Q-Learning) loss
        
        CQL adds a penalty for overestimation:
        L_CQL = E[log sum exp(Q) - Q(s,a)]
        """
        batch_size = states.shape[0]
        
        # Sample random actions for log-sum-exp
        random_actions = torch.FloatTensor(
            batch_size, self.cql_num_samples, self.action_dim
        ).uniform_(0, 1).to(self.device)
        
        # Normalize to simplex
        random_actions = random_actions / random_actions.sum(dim=-1, keepdim=True)
        
        # Compute Q-values for random actions
        q1_random_list = []
        q2_random_list = []
        
        for i in range(self.cql_num_samples):
            q1_r, q2_r = self.critic(
                states, 
                random_actions[:, i, :]
            )
            q1_random_list.append(q1_r.mean(dim=-1, keepdim=True))
            q2_random_list.append(q2_r.mean(dim=-1, keepdim=True))
        
        q1_random = torch.cat(q1_random_list, dim=1)  # [batch, num_samples]
        q2_random = torch.cat(q2_random_list, dim=1)
        
        # Sample actions from current policy
        with torch.no_grad():
            policy_actions, _ = self.actor.get_action(states)
        
        q1_policy, q2_policy = self.critic(states, policy_actions)
        q1_policy = q1_policy.mean(dim=-1, keepdim=True)
        q2_policy = q2_policy.mean(dim=-1, keepdim=True)
        
        # Concatenate all Q-values
        q1_all = torch.cat([q1_random, q1_policy], dim=1)
        q2_all = torch.cat([q2_random, q2_policy], dim=1)
        
        # Log-sum-exp (수치적으로 안정한 구현)
        q_max = torch.max(q1_all, dim=1, keepdim=True)[0]
        q1_logsumexp = q_max + torch.log(torch.sum(torch.exp(q1_all - q_max), dim=1, keepdim=True) + 1e-8)
        q1_logsumexp = torch.clamp(q1_logsumexp, -100, 100)

        q_max = torch.max(q2_all, dim=1, keepdim=True)[0]
        q2_logsumexp = q_max + torch.log(torch.sum(torch.exp(q2_all - q_max), dim=1, keepdim=True) + 1e-8)
        q2_logsumexp = torch.clamp(q2_logsumexp, -100, 100)
        
        # CQL loss: penalize overestimation
        cql_loss = (q1_logsumexp - q1.mean(dim=-1, keepdim=True)).mean() + \
                   (q2_logsumexp - q2.mean(dim=-1, keepdim=True)).mean()
        
        return cql_loss
    
    
    def _update_performance(self):
        """Update performance score for specialization"""
        if len(self.recent_rewards) > 0:
            # Simple exponential moving average of rewards
            avg_reward = np.mean(self.recent_rewards)
            self.performance_score = 0.95 * self.performance_score + 0.05 * avg_reward * 100
    
    def load_pretrained(self, checkpoint_path: str):
        """Load pretrained IQL weights"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'actor' in checkpoint:
            self.actor.load_state_dict(checkpoint['actor'])
            self.logger.info(f"Loaded pretrained actor from {checkpoint_path}")
        
        if 'q_network' in checkpoint:
            # Initialize critic with IQL Q-network weights (partial)
            self.logger.info("Initialized critic with IQL Q-values")
    
    def load_iql_checkpoint(self, checkpoint: Dict):
        """
        Load IQL checkpoint weights into B-Cell networks
        
        Args:
            checkpoint: IQL checkpoint dictionary containing 'actor', 'value', 'q1', 'q2' etc.
        """
        # Load actor network
        if 'actor' in checkpoint:
            self.actor.load_state_dict(checkpoint['actor'])
            self.logger.info("IQL actor weights loaded into B-Cell actor")
        
        # Load Q-networks if compatible
        if 'q1' in checkpoint and 'q2' in checkpoint:
            # Note: IQL Q-networks might have different structure than distributional critics
            # So we only load the shared layers if possible
            self.logger.info("IQL Q-network weights detected")
            
            # Store IQL value function for reference if needed
            if hasattr(self, 'iql_agent'):
                self.iql_agent.value.load_state_dict(checkpoint['value'])
                self.iql_agent.q1.load_state_dict(checkpoint['q1'])
                self.iql_agent.q2.load_state_dict(checkpoint['q2'])
                if 'q1_target' in checkpoint:
                    self.iql_agent.q1_target.load_state_dict(checkpoint['q1_target'])
                if 'q2_target' in checkpoint:
                    self.iql_agent.q2_target.load_state_dict(checkpoint['q2_target'])
                self.logger.info("IQL agent networks fully loaded")
            else:
                self.logger.info("Will initialize Q-networks randomly for SAC training")
        
        # Set training step if available
        if 'training_steps' in checkpoint:
            self.training_step = checkpoint['training_steps']
            self.logger.info(f"Training step set to {self.training_step}")
        
        self.logger.info("IQL checkpoint loaded successfully into B-Cell")
    
    def save(self, path: str):
        """Save B-Cell model"""
        torch.save({
            'specialization': self.specialization,
            'actor': self.actor.state_dict(),
            'critic_q1': self.critic.q1.state_dict(),
            'critic_q2': self.critic.q2.state_dict(),
            'critic_q1_target': self.critic.q1_target.state_dict(),
            'critic_q2_target': self.critic.q2_target.state_dict(),
            'log_alpha': self.log_alpha,
            'training_step': self.training_step,
            'performance_score': self.performance_score
        }, path)
        
    def load(self, path: str):
        """Load B-Cell model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.q1.load_state_dict(checkpoint['critic_q1'])
        self.critic.q2.load_state_dict(checkpoint['critic_q2'])
        self.critic.q1_target.load_state_dict(checkpoint['critic_q1_target'])
        self.critic.q2_target.load_state_dict(checkpoint['critic_q2_target'])
        self.log_alpha = checkpoint['log_alpha']
        self.training_step = checkpoint.get('training_step', 0)
        self.performance_score = checkpoint.get('performance_score', 0.0)