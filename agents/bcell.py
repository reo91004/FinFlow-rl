# bipd/agents/bcell.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
from utils.logger import BIPDLogger

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
    """Critic 네트워크: Q(s,a) 가치 함수"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(CriticNetwork, self).__init__()
        
        # state 처리용
        self.state_fc = nn.Linear(state_dim, hidden_dim)
        # action 처리용  
        self.action_fc = nn.Linear(action_dim, hidden_dim)
        # 결합 처리용
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
        # 가중치 초기화
        self._init_weights()
    
    def _init_weights(self):
        """가중치 초기화"""
        for layer in [self.state_fc, self.action_fc, self.fc1, self.fc2]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, state, action=None):
        # action이 제공되지 않으면 state만으로 계산 (이전 버전 호환성)
        if action is None:
            # 단순한 상태 가치 함수로 작동
            state_value = F.relu(self.state_fc(state))
            x = F.relu(self.fc1(torch.cat([state_value, torch.zeros_like(state_value)], dim=1)))
            return self.fc2(x)
        
        # state와 action을 결합한 Q(s,a) 계산
        state_value = F.relu(self.state_fc(state))
        action_value = F.relu(self.action_fc(action))
        x = F.relu(self.fc1(torch.cat([state_value, action_value], dim=1)))
        q_value = self.fc2(x)
        return q_value

class PrioritizedReplayBuffer:
    """우선순위 경험 재생 버퍼 (PER)"""
    
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha  # 우선순위 지수
        self.beta = beta    # 중요도 샘플링 지수
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.buffer = []
        self.pos = 0
        
    def push(self, state, action, reward, next_state, done):
        """경험 저장 (최대 우선순위로 초기 설정)"""
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size):
        """우선순위 기반 배치 샘플링"""
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.pos]
        
        # 우선순위 기반 확률 계산
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # 샘플 인덱스 선택
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]
        
        # 중요도 샘플링 가중치 계산
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # 정규화
        
        return samples, weights, indices
    
    def update_priorities(self, indices, priorities):
        """TD-error 기반 우선순위 업데이트"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
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
        
        # 신경망 초기화
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim)
        
        # Twin Critics (TD3)
        self.critic1 = CriticNetwork(state_dim, action_dim, hidden_dim)
        self.critic2 = CriticNetwork(state_dim, action_dim, hidden_dim)
        
        # 타겟 네트워크들
        self.target_actor = ActorNetwork(state_dim, action_dim, hidden_dim)
        self.target_critic1 = CriticNetwork(state_dim, action_dim, hidden_dim)
        self.target_critic2 = CriticNetwork(state_dim, action_dim, hidden_dim)
        
        # 타겟 네트워크 초기화 (메인 네트워크 복사)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # 옵티마이저
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)
        
        # 경험 재생 버퍼 (PER)
        self.replay_buffer = PrioritizedReplayBuffer(capacity=10000, alpha=0.6, beta=0.4)
        
        # 학습 파라미터
        self.gamma = 0.99
        self.tau = 0.005
        self.batch_size = 64
        self.update_frequency = 4
        
        # TD3 파라미터
        self.target_noise = 0.2  # Target Policy Smoothing 노이즈
        self.noise_clip = 0.5    # 노이즈 클리핑
        self.policy_delay = 2    # Actor 업데이트 지연
        self.policy_update_counter = 0
        
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
            f"상태차원={state_dim}, 행동차원={action_dim}"
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
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            weights = self.actor(state_tensor).squeeze(0).numpy()
        
        self.actor.train()
        
        # 탐험 (훈련 시에만) - 온도 조절 방식
        if not deterministic and random.random() < self.epsilon:
            # 온도를 높여서 더 균등한 분포로 탐험
            exploration_temp = 2.0 + random.random() * 3.0  # 2.0~5.0 사이
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                weights = self.actor(state_tensor, temperature=exploration_temp).squeeze(0).numpy()
        
        # 가중치 정규화 (안전장치)
        weights = weights / weights.sum()
        weights = np.clip(weights, 0.001, 0.999)  # 극단값 방지
        weights = weights / weights.sum()  # 재정규화
        
        return weights
    
    def store_experience(self, state, action, reward, next_state, done):
        """경험 저장"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self):
        """네트워크 업데이트 (TD3 + PER)"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # PER 배치 샘플링
        batch, is_weights, indices = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.BoolTensor(dones)
        is_weights = torch.FloatTensor(is_weights)
        
        # ===== Twin Critics 업데이트 =====
        with torch.no_grad():
            # Target Policy Smoothing: 타겟 액션에 노이즈 추가
            noise = torch.clamp(
                torch.randn_like(actions) * self.target_noise,
                -self.noise_clip, self.noise_clip
            )
            next_actions = torch.clamp(
                self.target_actor(next_states) + noise,
                0.001, 0.999  # 포트폴리오 가중치 범위
            )
            
            # Twin Q-values에서 최소값 선택 (과대평가 방지)
            target_q1 = self.target_critic1(next_states, next_actions).squeeze()
            target_q2 = self.target_critic2(next_states, next_actions).squeeze()
            target_q = torch.min(target_q1, target_q2)
            
            target_q_values = rewards + (self.gamma * target_q * ~dones)
        
        # Critic 1 업데이트
        current_q1 = self.critic1(states, actions).squeeze()
        critic1_loss = (is_weights * F.mse_loss(current_q1, target_q_values, reduction='none')).mean()
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
        self.critic1_optimizer.step()
        
        # Critic 2 업데이트
        current_q2 = self.critic2(states, actions).squeeze()
        critic2_loss = (is_weights * F.mse_loss(current_q2, target_q_values, reduction='none')).mean()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
        self.critic2_optimizer.step()
        
        # TD-error 계산 (PER 우선순위 업데이트용)
        td_errors = torch.min(
            (current_q1 - target_q_values).abs(),
            (current_q2 - target_q_values).abs()
        ).detach().cpu().numpy()
        
        priorities = td_errors + 1e-6
        self.replay_buffer.update_priorities(indices, priorities)
        
        # ===== Delayed Policy Updates =====
        self.policy_update_counter += 1
        if self.policy_update_counter % self.policy_delay != 0:
            return  # Actor 업데이트 건너뛰기
        
        # ===== Actor 업데이트 (TD3 방식) =====
        predicted_actions = self.actor(states)
        
        # Critic 1을 통한 정책 그래디언트 (TD3에서는 한 개만 사용)
        actor_loss = -self.critic1(states, predicted_actions).mean()
        
        # 엔트로피 보너스 (다양성 증진)
        log_probs = torch.log(predicted_actions + 1e-8)
        entropy = -torch.mean(torch.sum(predicted_actions * log_probs, dim=1))
        actor_loss -= 0.01 * entropy
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # 타겟 네트워크 소프트 업데이트 (TD3)
        self._soft_update_targets()
        
        # 통계 기록
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append((critic1_loss.item() + critic2_loss.item()) / 2)
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
    
    def _soft_update_targets(self):
        """타겟 네트워크들 소프트 업데이트 (TD3)"""
        # Target Actor 업데이트
        for target_param, param in zip(self.target_actor.parameters(), 
                                     self.actor.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
        
        # Target Critic 1 업데이트
        for target_param, param in zip(self.target_critic1.parameters(), 
                                     self.critic1.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
        
        # Target Critic 2 업데이트
        for target_param, param in zip(self.target_critic2.parameters(), 
                                     self.critic2.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
    
    def get_specialization_score(self, crisis_info):
        """
        다차원 위기 정보에 대한 전문성 점수 계산
        
        Args:
            crisis_info: dict or float - 다차원 위기 정보 또는 기존 crisis_level
            
        Returns:
            score: float [0, 1] - 높을수록 현재 상황에 특화됨
        """
        # 하위 호환성: 기존 float crisis_level 지원
        if isinstance(crisis_info, (int, float)):
            return self._get_legacy_specialization_score(crisis_info)
        
        # 다차원 위기 벡터 기반 전문성 계산
        if isinstance(crisis_info, dict):
            overall_crisis = crisis_info.get('overall_crisis', 0.0)
            volatility_crisis = crisis_info.get('volatility_crisis', 0.0)
            correlation_crisis = crisis_info.get('correlation_crisis', 0.0)
            volume_crisis = crisis_info.get('volume_crisis', 0.0)
            
            if self.risk_type == 'volatility':
                # 변동성 전문가: 변동성 위기와 전체 위기에 특화
                volatility_score = volatility_crisis * 1.5  # 변동성 위기에 높은 가중치
                overall_score = overall_crisis * 0.8
                return np.clip(volatility_score + overall_score * 0.5, 0.0, 1.0)
                
            elif self.risk_type == 'correlation':
                # 상관관계 전문가: 상관관계 위기와 중간 수준 전체 위기에 특화
                correlation_score = correlation_crisis * 1.8
                optimal_overall = 1 - abs(overall_crisis - 0.55) * 2.0  # 중간 위기 수준 선호
                return np.clip(correlation_score + optimal_overall * 0.3, 0.0, 1.0)
                
            elif self.risk_type == 'momentum':
                # 모멘텀 전문가: 낮은 위기 상황과 거래량 이상에 특화
                momentum_score = max(0, 1 - overall_crisis * 2.5)  # 낮은 위기 선호
                volume_score = volume_crisis * 1.2  # 거래량 이상 활용
                return np.clip(momentum_score + volume_score * 0.4, 0.0, 1.0)
                
            elif self.risk_type == 'defensive':
                # 방어 전문가: 중고위기와 모든 위기 유형에 균형 있게 대응
                defensive_score = 1 - abs(overall_crisis - 0.65) * 1.8  # 중고위기 선호
                multi_crisis = (volatility_crisis + correlation_crisis + volume_crisis) / 3
                return np.clip(defensive_score + multi_crisis * 0.6, 0.0, 1.0)
                
            elif self.risk_type == 'growth':
                # 성장 전문가: 매우 낮은 위기 상황에 특화
                growth_score = max(0, 1 - overall_crisis * 3.5)  # 매우 낮은 위기만
                stability_bonus = max(0, 1 - volatility_crisis * 2)  # 낮은 변동성 보너스
                return np.clip(growth_score + stability_bonus * 0.3, 0.0, 1.0)
                
            else:
                return 0.5  # 기본값
        
        return 0.5  # 예외 상황
    
    def _get_legacy_specialization_score(self, crisis_level):
        """
        기존 단일 crisis_level 기반 전문성 점수 (하위 호환성)
        """
        if self.risk_type == 'volatility':
            return crisis_level
        elif self.risk_type == 'correlation':
            return 1 - abs(crisis_level - 0.55) * 2.5
        elif self.risk_type == 'momentum':
            return max(0, 1 - crisis_level * 2.5)
        elif self.risk_type == 'defensive':
            return 1 - abs(crisis_level - 0.65) * 2
        elif self.risk_type == 'growth':
            return max(0, 1 - crisis_level * 3)
        else:
            return 0.5
    
    def get_explanation(self, state):
        """
        의사결정에 대한 설명 생성 (XAI)
        
        Returns:
            dict: 의사결정 설명
        """
        try:
            self.actor.eval()
            
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                weights = self.actor(state_tensor).squeeze(0)
                value = self.critic(state_tensor).squeeze(0)
            
            self.actor.train()
            
            # 입력 특성 분석
            features = state[:12]  # 시장 특성
            crisis_level = state[12]  # 위기 수준
            prev_weights = state[13:]  # 이전 가중치
            
            explanation = {
                'risk_type': self.risk_type,
                'predicted_weights': weights.numpy().tolist(),
                'predicted_value': float(value),
                'specialization_score': self.get_specialization_score(crisis_level),
                'crisis_level': float(crisis_level),
                'max_weight_asset': int(weights.argmax()),
                'min_weight_asset': int(weights.argmin()),
                'weight_concentration': float((weights ** 2).sum()),
                'epsilon': self.epsilon,
                'update_count': self.update_count
            }
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"설명 생성 실패: {e}")
            return {'error': str(e)}
    
    def save_model(self, filepath):
        """모델 저장 (TD3)"""
        try:
            torch.save({
                'actor_state_dict': self.actor.state_dict(),
                'critic1_state_dict': self.critic1.state_dict(),
                'critic2_state_dict': self.critic2.state_dict(),
                'target_actor_state_dict': self.target_actor.state_dict(),
                'target_critic1_state_dict': self.target_critic1.state_dict(),
                'target_critic2_state_dict': self.target_critic2.state_dict(),
                'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
                'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
                'risk_type': self.risk_type,
                'epsilon': self.epsilon,
                'update_count': self.update_count,
                'policy_update_counter': self.policy_update_counter
            }, filepath)
            
            self.logger.info(f"{self.risk_type} B-Cell 모델이 저장되었습니다: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"모델 저장 실패: {e}")
            return False
    
    def load_model(self, filepath):
        """모델 로드 (TD3)"""
        try:
            checkpoint = torch.load(filepath)
            
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
            self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
            self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
            self.target_critic1.load_state_dict(checkpoint['target_critic1_state_dict'])
            self.target_critic2.load_state_dict(checkpoint['target_critic2_state_dict'])
            
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
            self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
            
            self.epsilon = checkpoint['epsilon']
            self.update_count = checkpoint['update_count']
            self.policy_update_counter = checkpoint.get('policy_update_counter', 0)
            
            self.logger.info(f"{self.risk_type} B-Cell 모델이 로드되었습니다: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"모델 로드 실패: {e}")
            return False