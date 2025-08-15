# bipd/agents/bcell.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from collections import deque
import random
from utils.logger import BIPDLogger
from config import DEVICE

class ActorNetwork(nn.Module):
    """Actor 네트워크: 포트폴리오 가중치 생성"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        # 탐험용 온도 파라미터
        self.temperature = 1.0
        
        # 가중치 초기화
        self._init_weights()
    
    def _init_weights(self):
        """가중치 초기화"""
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, state, temperature=None):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        
        # 온도 조절된 Softmax로 가중치 합이 1이 되도록 보장
        temp = temperature if temperature is not None else self.temperature
        weights = F.softmax(logits / temp, dim=-1)
        return weights
    
    def set_temperature(self, temperature):
        """탐험을 위한 온도 설정"""
        self.temperature = max(0.1, temperature)  # 최소값 0.1로 제한

class CriticNetwork(nn.Module):
    """Critic 네트워크: 상태 가치 함수"""
    
    def __init__(self, state_dim, hidden_dim=128):
        super(CriticNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # 가중치 초기화
        self._init_weights()
    
    def _init_weights(self):
        """가중치 초기화"""
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value

class ReplayBuffer:
    """경험 재생 버퍼"""
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """경험 저장"""
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """배치 샘플링"""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class BCell:
    """
    B-세포: 특정 위험 유형에 특화된 포트폴리오 전략 실행
    
    Actor-Critic 강화학습을 사용하여 포트폴리오 가중치를 학습
    각 B-Cell은 특정 시장 상황(volatility, correlation, momentum)에 특화
    """
    
    def __init__(self, risk_type, state_dim, action_dim, 
                 actor_lr=3e-4, critic_lr=6e-4, hidden_dim=128):
        self.risk_type = risk_type
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 신경망 초기화 및 GPU로 이동
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim).to(DEVICE)
        self.critic = CriticNetwork(state_dim, hidden_dim).to(DEVICE)
        
        # 타겟 네트워크 (안정적 학습용)
        self.target_critic = CriticNetwork(state_dim, hidden_dim).to(DEVICE)
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # 옵티마이저
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # 경험 재생 버퍼
        self.replay_buffer = ReplayBuffer(capacity=10000)
        
        # 학습 파라미터
        self.gamma = 0.99
        self.tau = 0.005
        self.batch_size = 64
        self.update_frequency = 4
        
        
        # 탐험 파라미터
        self.epsilon = 0.9
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        
        # 학습 통계
        self.update_count = 0
        self.actor_losses = []
        self.critic_losses = []
        
        # 로거
        self.logger = BIPDLogger(f"BCell-{risk_type}")
        
        self.logger.info(
            f"{risk_type} B-Cell이 초기화되었습니다. "
            f"상태차원={state_dim}, 행동차원={action_dim}, "
            f"Device={DEVICE}"
        )
    
    def get_action(self, state, deterministic=False):
        """
        포트폴리오 가중치 생성
        
        Args:
            state: np.array of shape (state_dim,)
            deterministic: bool, True면 탐험 없이 결정적 행동
            
        Returns:
            weights: np.array of shape (action_dim,)
        """
        self.actor.eval()
        
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            weights = self.actor(state_tensor).squeeze(0).cpu().numpy()
        
        self.actor.train()
        
        # 탐험 (훈련 시에만) - 온도 조절 방식
        if not deterministic and random.random() < self.epsilon:
            # 온도를 높여서 더 균등한 분포로 탐험
            exploration_temp = 2.0 + random.random() * 3.0  # 2.0~5.0 사이
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                weights = self.actor(state_tensor, temperature=exploration_temp).squeeze(0).cpu().numpy()
        
        # 가중치 정규화 (안전장치)
        weights = weights / weights.sum()
        weights = np.clip(weights, 0.001, 0.999)  # 극단값 방지
        weights = weights / weights.sum()  # 재정규화
        
        return weights
    
    def store_experience(self, state, action, reward, next_state, done):
        """경험 저장 (CUDA 호환성을 위한 타입 변환)"""
        # NumPy 타입을 Python native 타입으로 안전하게 변환
        done = bool(done)
        reward = float(reward)
        
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self):
        """네트워크 업데이트"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # 배치 샘플링
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.tensor(np.array(states), dtype=torch.float32).to(DEVICE)
        actions = torch.tensor(np.array(actions), dtype=torch.float32).to(DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(DEVICE)
        dones = torch.tensor(dones, dtype=torch.bool).to(DEVICE)
        
        # Critic 업데이트
        current_q_values = self.critic(states).squeeze()
        
        with torch.no_grad():
            next_q_values = self.target_critic(next_states).squeeze()
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        critic_loss = F.mse_loss(current_q_values, target_q_values)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Actor 업데이트
        predicted_actions = self.actor(states)
        
        # 정책 그래디언트
        advantages = target_q_values - current_q_values.detach()
        
        # 로그 확률 계산 (포트폴리오 가중치용)
        log_probs = torch.log(predicted_actions + 1e-8)
        action_log_probs = torch.sum(log_probs * actions, dim=1)
        
        actor_loss = -torch.mean(action_log_probs * advantages)
        
        # 엔트로피 보너스 (다양성 증진)
        entropy = -torch.mean(torch.sum(predicted_actions * log_probs, dim=1))
        actor_loss -= 0.01 * entropy
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # 타겟 네트워크 소프트 업데이트
        self._soft_update_target()
        
        # 통계 기록
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())
        self.update_count += 1
        
        # 엡실론 감소
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # 주기적 로깅
        if self.update_count % 100 == 0:
            avg_actor_loss = np.mean(self.actor_losses[-100:])
            avg_critic_loss = np.mean(self.critic_losses[-100:])
            
            self.logger.debug(
                f"[{self.risk_type}] 업데이트 {self.update_count}: "
                f"Actor 손실={avg_actor_loss:.4f}, "
                f"Critic 손실={avg_critic_loss:.4f}, "
                f"탐험률={self.epsilon:.3f}"
            )
    
    def _soft_update_target(self):
        """타겟 네트워크 소프트 업데이트"""
        for target_param, param in zip(self.target_critic.parameters(), 
                                     self.critic.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
    
    def get_specialization_score(self, crisis_level):
        """
        현재 상황에 대한 전문성 점수 계산
        
        Args:
            crisis_level: float [0, 1]
            
        Returns:
            score: float [0, 1] - 높을수록 현재 상황에 특화됨
        """
        if self.risk_type == 'volatility':
            # 고위기 상황에 특화 (crisis_level > 0.7)
            return crisis_level
        elif self.risk_type == 'correlation':
            # 중상위기 상황에 특화 (crisis_level 0.4~0.7)
            return 1 - abs(crisis_level - 0.55) * 2.5
        elif self.risk_type == 'momentum':
            # 저중위기 상황에 특화 (crisis_level < 0.4)
            return max(0, 1 - crisis_level * 2.5)
        elif self.risk_type == 'defensive':
            # 중고위기 상황에 특화 (crisis_level 0.5~0.8)
            return 1 - abs(crisis_level - 0.65) * 2
        elif self.risk_type == 'growth':
            # 저위기 안정 상황에 특화 (crisis_level < 0.3)
            return max(0, 1 - crisis_level * 3)
        else:
            return 0.5  # 기본값
    
    def get_explanation(self, state):
        """
        의사결정에 대한 설명 생성 (XAI)
        
        Returns:
            dict: 의사결정 설명
        """
        try:
            self.actor.eval()
            
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                weights = self.actor(state_tensor).squeeze(0)
                value = self.critic(state_tensor).squeeze(0)
            
            self.actor.train()
            
            # 입력 특성 분석
            features = state[:12]  # 시장 특성
            crisis_level = state[12]  # 위기 수준
            prev_weights = state[13:]  # 이전 가중치
            
            explanation = {
                'risk_type': self.risk_type,
                'predicted_weights': weights.cpu().numpy().tolist(),
                'predicted_value': float(value.cpu()),
                'specialization_score': self.get_specialization_score(crisis_level),
                'crisis_level': float(crisis_level),
                'max_weight_asset': int(weights.argmax().cpu()),
                'min_weight_asset': int(weights.argmin().cpu()),
                'weight_concentration': float((weights ** 2).sum().cpu()),
                'epsilon': self.epsilon,
                'update_count': self.update_count
            }
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"설명 생성 실패: {e}")
            return {'error': str(e)}
    
    def save_model(self, filepath):
        """모델 저장"""
        try:
            # 저장 디렉토리 생성 보장
            base_dir = os.path.dirname(filepath)
            if base_dir:
                os.makedirs(base_dir, exist_ok=True)
            
            torch.save({
                'actor_state_dict': self.actor.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'target_critic_state_dict': self.target_critic.state_dict(),
                'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                'risk_type': self.risk_type,
                'epsilon': self.epsilon,
                'update_count': self.update_count
            }, filepath)
            
            self.logger.info(f"{self.risk_type} B-Cell 모델이 저장되었습니다: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"모델 저장 실패: {e}")
            return False
    
    def load_model(self, filepath):
        """모델 로드"""
        try:
            checkpoint = torch.load(filepath, map_location=DEVICE)
            
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            
            self.epsilon = checkpoint['epsilon']
            self.update_count = checkpoint['update_count']
            
            self.logger.info(f"{self.risk_type} B-Cell 모델이 로드되었습니다: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"모델 로드 실패: {e}")
            return False