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

class SACActorNetwork(nn.Module):
    """SAC Actor 네트워크: 확률적 포트폴리오 가중치 생성"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(SACActorNetwork, self).__init__()
        
        self.action_dim = action_dim
        
        # 공통 특성 추출 레이어
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Dirichlet 분포를 위한 concentration 파라미터 출력
        # 포트폴리오 가중치는 simplex 위에 있어야 하므로 Dirichlet이 적합
        self.concentration_head = nn.Linear(hidden_dim, action_dim)
        
        # 가중치 초기화
        self._init_weights()
    
    def _init_weights(self):
        """가중치 초기화"""
        for layer in [self.fc1, self.fc2, self.concentration_head]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, state):
        """
        확률적 정책 출력
        Returns:
            concentration: Dirichlet 분포의 concentration 파라미터
            weights: 샘플링된 포트폴리오 가중치
            log_prob: 로그 확률
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        # Concentration 파라미터 (안정성 강화)
        x_clamped = torch.clamp(self.concentration_head(x), min=-10.0, max=10.0)
        concentration = F.softplus(x_clamped) + 2.0  # 1.0 → 2.0 변경 (안정성 향상)
        
        # Dirichlet 분포에서 샘플링
        if self.training:
            # 훈련 시: 확률적 샘플링
            dist = torch.distributions.Dirichlet(concentration)
            weights = dist.rsample()  # reparameterization trick 사용
            log_prob = dist.log_prob(weights)
        else:
            # 평가 시: 결정적 출력 (평균 사용)
            # Dirichlet의 평균은 concentration / concentration.sum()
            weights = concentration / concentration.sum(dim=-1, keepdim=True)
            log_prob = torch.zeros(weights.shape[0], device=weights.device)
        
        return concentration, weights, log_prob
    
    def get_action_and_log_prob(self, state):
        """행동과 로그 확률을 함께 반환"""
        concentration, weights, log_prob = self.forward(state)
        return weights, log_prob

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
    
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha  # 우선순위 지수
        self.beta = beta    # 중요도 샘플링 지수 (초기값)
        self.beta_increment = beta_increment  # 베타 증가율
        self.max_beta = 1.0  # 베타 최대값
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
        # Beta annealing: 점진적으로 1.0에 수렴
        self.beta = min(self.max_beta, self.beta + self.beta_increment)
        
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
    
    SAC (Soft Actor-Critic) 강화학습을 사용하여 포트폴리오 가중치를 학습
    각 B-Cell은 특정 시장 상황(volatility, correlation, momentum)에 특화
    """
    
    def __init__(self, risk_type, state_dim, action_dim, 
                 actor_lr=3e-4, critic_lr=6e-4, alpha_lr=3e-4, hidden_dim=128):
        self.risk_type = risk_type
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # SAC 신경망 초기화 및 GPU로 이동
        self.actor = SACActorNetwork(state_dim, action_dim, hidden_dim).to(DEVICE)
        
        # Twin Critics (SAC에서도 사용)
        self.critic1 = CriticNetwork(state_dim, action_dim, hidden_dim).to(DEVICE)
        self.critic2 = CriticNetwork(state_dim, action_dim, hidden_dim).to(DEVICE)
        
        # 타겟 네트워크들 (Critic만 필요, SAC에서는 Actor 타겟 없음)
        self.target_critic1 = CriticNetwork(state_dim, action_dim, hidden_dim).to(DEVICE)
        self.target_critic2 = CriticNetwork(state_dim, action_dim, hidden_dim).to(DEVICE)
        
        # 타겟 네트워크 초기화 (Critic만 복사)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # SAC 엔트로피 계수 (탐험 강화)
        self.target_entropy = -float(action_dim) * 0.25  # 0.5 → 0.25로 변경하여 탐험 강화
        self.log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
        self.alpha = self.log_alpha.exp()
        
        # 옵티마이저
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        
        # 경험 재생 버퍼 (PER)
        self.replay_buffer = PrioritizedReplayBuffer(capacity=10000, alpha=0.6, beta=0.4)
        
        # SAC 학습 파라미터
        self.gamma = 0.99
        self.tau = 0.005
        self.batch_size = 64
        self.update_frequency = 4
        
        # 학습 통계
        self.update_count = 0
        self.actor_losses = []
        self.critic_losses = []
        
        # 모니터링 통계 (축약형)
        self.monitoring_stats = {
            'q_value_range': {'min': [], 'max': []},
            'gradient_norms': {'actor': [], 'critic': []},
            'alpha_history': [],
            'td_error_stats': {'mean': [], 'max': []},
            'last_report': 0
        }
        
        # 로거
        self.logger = BIPDLogger(f"BCell-{risk_type}")
        
        self.logger.info(
            f"{risk_type} SAC B-Cell이 초기화되었습니다. "
            f"상태차원={state_dim}, 행동차원={action_dim}, "
            f"Target Entropy={self.target_entropy}, "
            f"Device={DEVICE}"
        )
    
    def get_action(self, state, deterministic=False):
        """
        SAC 기반 포트폴리오 가중치 생성
        
        Args:
            state: np.array of shape (state_dim,)
            deterministic: bool, True면 탐험 없이 결정적 행동
            
        Returns:
            weights: np.array of shape (action_dim,)
        """
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            # SAC Actor에서 확률적 또는 결정적 행동 샘플링
            if deterministic:
                self.actor.eval()
                concentration, weights, _ = self.actor(state_tensor)
                weights = weights.squeeze(0).cpu().numpy()
                self.actor.train()
            else:
                # 훈련 모드에서는 확률적 샘플링 (엔트로피 최대화)
                concentration, weights, _ = self.actor(state_tensor)
                weights = weights.squeeze(0).cpu().numpy()
        
        # 가중치 정규화 (안전장치)
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
        """네트워크 업데이트 (SAC + PER)"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # PER 배치 샘플링
        batch, is_weights, indices = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # NumPy 배열을 안전하게 변환하여 CUDA 호환성 확보
        states = torch.tensor(np.array(states, dtype=np.float32), dtype=torch.float32).to(DEVICE)
        actions = torch.tensor(np.array(actions, dtype=np.float32), dtype=torch.float32).to(DEVICE)
        rewards = torch.tensor([float(r) for r in rewards], dtype=torch.float32).to(DEVICE)
        next_states = torch.tensor(np.array(next_states, dtype=np.float32), dtype=torch.float32).to(DEVICE)
        dones = torch.tensor([bool(d) for d in dones], dtype=torch.bool).to(DEVICE)
        is_weights = torch.tensor(np.array(is_weights, dtype=np.float32), dtype=torch.float32).to(DEVICE)
        
        # 현재 alpha 값 업데이트
        self.alpha = self.log_alpha.exp()
        
        # ===== SAC Twin Critics 업데이트 =====
        with torch.no_grad():
            # SAC에서는 다음 상태에서의 정책으로부터 행동과 로그 확률을 샘플링
            _, next_actions, next_log_probs = self.actor(next_states)
            
            # Twin Q-values 계산 (SAC는 엔트로피 항 포함)
            target_q1 = self.target_critic1(next_states, next_actions).squeeze()
            target_q2 = self.target_critic2(next_states, next_actions).squeeze()
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            
            # 타겟 Q-value 계산 (보상 범위와 일치하도록 조정)
            target_q_values = rewards + self.gamma * target_q * (~dones)
            target_q_values = torch.clamp(target_q_values, min=-10.0, max=10.0)  # 기존 [-50, 50]에서 조정
        
        # Critic 1 업데이트
        current_q1 = self.critic1(states, actions).squeeze()
        critic1_loss = (is_weights * F.mse_loss(current_q1, target_q_values, reduction='none')).mean()
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        critic1_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
        self.critic1_optimizer.step()
        
        # Critic 2 업데이트
        current_q2 = self.critic2(states, actions).squeeze()
        critic2_loss = (is_weights * F.mse_loss(current_q2, target_q_values, reduction='none')).mean()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        critic2_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
        self.critic2_optimizer.step()
        
        # TD-error 계산 (PER 우선순위 업데이트용)
        td_errors = torch.min(
            (current_q1 - target_q_values).abs(),
            (current_q2 - target_q_values).abs()
        ).detach().cpu().numpy()
        
        # Priority clipping으로 극단값 방지
        priorities = np.clip(td_errors + 1e-6, a_min=0.0, a_max=100.0)
        self.replay_buffer.update_priorities(indices, priorities)
        
        # ===== SAC Actor 업데이트 =====
        _, current_actions, current_log_probs = self.actor(states)
        
        # Q-values for current actions
        q1_current = self.critic1(states, current_actions).squeeze()
        q2_current = self.critic2(states, current_actions).squeeze()
        q_current = torch.min(q1_current, q2_current)
        
        # SAC Actor 손실 (엔트로피 정규화 포함)
        actor_loss = (self.alpha * current_log_probs - q_current).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # ===== Alpha (엔트로피 계수) 자동 튜닝 =====
        alpha_loss = -(self.log_alpha * (current_log_probs + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # 타겟 네트워크 소프트 업데이트
        self._soft_update_targets()
        
        # 통계 기록
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append((critic1_loss.item() + critic2_loss.item()) / 2)
        self.update_count += 1
        
        # 모니터링 통계 수집 (메모리 효율적인 방식)
        with torch.no_grad():
            # Q-value 범위
            self.monitoring_stats['q_value_range']['min'].append(target_q.min().item())
            self.monitoring_stats['q_value_range']['max'].append(target_q.max().item())
            
            # Gradient norms
            self.monitoring_stats['gradient_norms']['actor'].append(actor_grad_norm.item())
            self.monitoring_stats['gradient_norms']['critic'].append(
                (critic1_grad_norm.item() + critic2_grad_norm.item()) / 2
            )
            
            # Alpha 추적
            self.monitoring_stats['alpha_history'].append(self.alpha.item())
            
            # TD error 통계
            self.monitoring_stats['td_error_stats']['mean'].append(td_errors.mean())
            self.monitoring_stats['td_error_stats']['max'].append(td_errors.max())
        
        # 주기적 요약 로깅 (100회마다)
        if self.update_count % 100 == 0:
            self._log_monitoring_summary()
            
        # Q-value 범위 조정 확인 로깅 (첫 실행 시)
        if self.update_count == 1:
            self.logger.info("Q-value 범위가 [-10, 10]으로 조정되었습니다")
            
        # 메모리 관리: 오래된 통계 제거 (최근 1000개만 유지)
        if self.update_count % 500 == 0:
            self._cleanup_monitoring_stats()
    
    def _log_monitoring_summary(self):
        """모니터링 통계 요약 로깅"""
        window = min(100, len(self.monitoring_stats['q_value_range']['min']))
        
        if window == 0:
            return
            
        # 최근 통계 계산
        q_min = np.mean(self.monitoring_stats['q_value_range']['min'][-window:])
        q_max = np.mean(self.monitoring_stats['q_value_range']['max'][-window:])
        grad_actor = np.mean(self.monitoring_stats['gradient_norms']['actor'][-window:])
        grad_critic = np.mean(self.monitoring_stats['gradient_norms']['critic'][-window:])
        alpha_current = self.monitoring_stats['alpha_history'][-1] if self.monitoring_stats['alpha_history'] else 0
        td_mean = np.mean(self.monitoring_stats['td_error_stats']['mean'][-window:])
        td_max = np.mean(self.monitoring_stats['td_error_stats']['max'][-window:])
        
        # 손실 통계
        avg_actor_loss = np.mean(self.actor_losses[-window:])
        avg_critic_loss = np.mean(self.critic_losses[-window:])
        
        # 안정성 체크
        is_stable = (
            not np.isnan(avg_actor_loss) and 
            not np.isnan(avg_critic_loss) and
            avg_critic_loss < 1e6 and
            q_max < 100
        )
        
        stability_marker = "✓" if is_stable else "⚠"
        
        self.logger.debug(
            f"[{self.risk_type}] {stability_marker} 업데이트 {self.update_count}: "
            f"손실(A:{avg_actor_loss:.2f}/C:{avg_critic_loss:.2f}) "
            f"Q범위[{q_min:.1f},{q_max:.1f}] "
            f"Grad(A:{grad_actor:.2f}/C:{grad_critic:.2f}) "
            f"α={alpha_current:.3f} "
            f"TD({td_mean:.2f}/{td_max:.2f})"
        )
        
        # 위험 신호 감지
        if avg_critic_loss > 1e5:
            self.logger.warning(
                f"[{self.risk_type}] Critic 손실 급증 감지: {avg_critic_loss:.2e}"
            )
        if q_max > 50:
            self.logger.warning(
                f"[{self.risk_type}] Q-value 범위 확대 감지: max={q_max:.2f}"
            )
    
    def _cleanup_monitoring_stats(self):
        """오래된 모니터링 통계 정리 (메모리 효율성)"""
        max_history = 1000
        
        for key in ['q_value_range', 'gradient_norms', 'td_error_stats']:
            for subkey in self.monitoring_stats[key]:
                if len(self.monitoring_stats[key][subkey]) > max_history:
                    self.monitoring_stats[key][subkey] = self.monitoring_stats[key][subkey][-max_history:]
        
        if len(self.monitoring_stats['alpha_history']) > max_history:
            self.monitoring_stats['alpha_history'] = self.monitoring_stats['alpha_history'][-max_history:]
        
        # 손실 히스토리도 정리
        if len(self.actor_losses) > max_history:
            self.actor_losses = self.actor_losses[-max_history:]
        if len(self.critic_losses) > max_history:
            self.critic_losses = self.critic_losses[-max_history:]
    
    def _soft_update_targets(self):
        """타겟 네트워크들 소프트 업데이트 (SAC - Critic만)"""
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
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                concentration, weights, _ = self.actor(state_tensor)
                weights = weights.squeeze(0)
                value = self.critic1(state_tensor).squeeze(0)
            
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
                'alpha': float(self.alpha.item()),
                'update_count': self.update_count
            }
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"설명 생성 실패: {e}")
            return {'error': str(e)}
    
    def save_model(self, filepath):
        """모델 저장 (SAC)"""
        try:
            # 저장 디렉토리 생성 보장
            base_dir = os.path.dirname(filepath)
            if base_dir:
                os.makedirs(base_dir, exist_ok=True)
            
            torch.save({
                'actor_state_dict': self.actor.state_dict(),
                'critic1_state_dict': self.critic1.state_dict(),
                'critic2_state_dict': self.critic2.state_dict(),
                'target_critic1_state_dict': self.target_critic1.state_dict(),
                'target_critic2_state_dict': self.target_critic2.state_dict(),
                'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
                'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
                'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict(),
                'log_alpha': self.log_alpha,
                'target_entropy': self.target_entropy,
                'risk_type': self.risk_type,
                'update_count': self.update_count
            }, filepath)
            
            self.logger.info(f"{self.risk_type} SAC B-Cell 모델이 저장되었습니다: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"모델 저장 실패: {e}")
            return False
    
    def load_model(self, filepath):
        """모델 로드 (SAC)"""
        try:
            checkpoint = torch.load(filepath, map_location=DEVICE)
            
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
            self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
            self.target_critic1.load_state_dict(checkpoint['target_critic1_state_dict'])
            self.target_critic2.load_state_dict(checkpoint['target_critic2_state_dict'])
            
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
            self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
            
            self.log_alpha = checkpoint['log_alpha'].to(DEVICE)
            self.target_entropy = checkpoint['target_entropy']
            self.alpha = self.log_alpha.exp()
            self.update_count = checkpoint['update_count']
            
            self.logger.info(f"{self.risk_type} SAC B-Cell 모델이 로드되었습니다: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"모델 로드 실패: {e}")
            return False