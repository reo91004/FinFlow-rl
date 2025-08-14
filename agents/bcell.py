# agents/bcell.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
import random
from collections import deque
from datetime import datetime
from typing import Dict
from .base import ImmuneCell
from constant import *
from utils.logger import BIPDLogger


class ExperienceReplayBuffer:
    """올바른 Experience Replay Buffer - (s,a,r,s',done) 튜플 저장"""

    def __init__(self, capacity=EXPERIENCE_BUFFER_SIZE):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """완전한 transition 저장"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """랜덤 배치 샘플링"""
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class ActorNetwork(nn.Module):
    """Actor 네트워크: 정책 결정"""

    def __init__(self, input_size, n_assets, hidden_size=BCELL_ACTOR_HIDDEN_SIZE):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, n_assets)
        self.dropout = nn.Dropout(DEFAULT_DROPOUT_RATE)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.softmax(x, dim=-1)


class CriticNetwork(nn.Module):
    """Critic 네트워크: 가치 함수 평가"""

    def __init__(self, input_size, hidden_size=BCELL_CRITIC_HIDDEN_SIZE):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(DEFAULT_DROPOUT_RATE)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        value = self.fc3(x)
        return value


class AttentionMechanism(nn.Module):
    """어텐션 메커니즘: T-Cell 특성 기여도를 B-Cell에 연결"""

    def __init__(self, feature_dim, hidden_dim=32):
        super(AttentionMechanism, self).__init__()
        self.feature_dim = feature_dim
        self.attention_weights = nn.Linear(feature_dim, hidden_dim)
        self.attention_output = nn.Linear(hidden_dim, feature_dim)

    def forward(self, features, tcell_contributions):
        """
        features: 시장 특성 벡터
        tcell_contributions: T-Cell에서 제공한 특성별 기여도
        """
        # 어텐션 가중치 계산
        attention_scores = torch.softmax(
            self.attention_output(F.tanh(self.attention_weights(features))), dim=-1
        )

        # T-Cell 기여도와 결합
        tcell_weights = torch.FloatTensor(list(tcell_contributions.values()))
        if len(tcell_weights) < len(features):
            # 패딩 처리
            padding = torch.zeros(len(features) - len(tcell_weights))
            tcell_weights = torch.cat([tcell_weights, padding])
        elif len(tcell_weights) > len(features):
            tcell_weights = tcell_weights[: len(features)]

        # 어텐션과 T-Cell 기여도 결합 (detach 제거 - gradient flow 보장)
        combined_attention = attention_scores * tcell_weights
        attended_features = features * combined_attention

        return attended_features, combined_attention


class BCell(ImmuneCell):
    """B-세포: 실제 작동하는 강화학습 기반 전문화된 대응 전략 생성"""

    def __init__(self, cell_id, risk_type, input_size, n_assets):
        super().__init__(cell_id)
        self.risk_type = risk_type
        self.n_assets = n_assets
        self.feature_dim = 12
        
        # 로거 초기화 (DEBUG 로그는 파일만, INFO는 터미널도)
        self.logger = BIPDLogger().get_learning_logger()

        # constant.py에서 모든 설정 가져오기
        self.batch_size = DEFAULT_BATCH_SIZE
        self.learning_rate = DEFAULT_LEARNING_RATE
        self.gamma = DEFAULT_GAMMA
        self.tau = DEFAULT_TAU
        self.update_frequency = DEFAULT_UPDATE_FREQUENCY
        self.epsilon = DEFAULT_EPSILON
        self.epsilon_decay = DEFAULT_EPSILON_DECAY
        self.min_epsilon = DEFAULT_MIN_EPSILON

        # 네트워크 초기화
        self.critic_network = CriticNetwork(input_size)
        self.critic_target_network = CriticNetwork(input_size)
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        self.actor_network = ActorNetwork(input_size, n_assets)
        self.attention_mechanism = AttentionMechanism(self.feature_dim)

        # 옵티마이저
        self.critic_optimizer = optim.Adam(
            self.critic_network.parameters(), lr=DEFAULT_CRITIC_LR
        )
        self.actor_optimizer = optim.Adam(
            self.actor_network.parameters(), lr=DEFAULT_ACTOR_LR
        )
        self.attention_optimizer = optim.Adam(
            self.attention_mechanism.parameters(), lr=DEFAULT_ATTENTION_LR
        )

        # Experience Replay Buffer - 완전한 구현
        self.experience_buffer = ExperienceReplayBuffer(EXPERIENCE_BUFFER_SIZE)
        
        # MDP 전이 보존을 위한 에피소드 버퍼
        self.episode_buffer = []
        self._pending_experience = None  # 다음 상태값을 기다리는 경험
        
        # 현재 상태 저장 (next_state_value 계산용)
        self.last_combined_input = None

        # 학습 카운터
        self.update_counter = 0
        self.step_counter = 0
        
        # 엔트로피 계수 추가
        self.entropy_coef = 0.01

        # 전문화 관련 속성
        self.specialization_buffer = deque(maxlen=BCELL_SPECIALIZATION_BUFFER_SIZE)
        self.general_buffer = deque(maxlen=BCELL_GENERAL_BUFFER_SIZE)
        self.specialization_strength = 0.1

        # 전문 분야별 특화 기준
        self.specialization_criteria = self._initialize_specialization_criteria()

        # 성과 추적
        self.specialist_performance = deque(maxlen=BCELL_PERFORMANCE_BUFFER_SIZE)
        self.general_performance = deque(maxlen=BCELL_PERFORMANCE_BUFFER_SIZE)

        # 가치 함수 추적
        self.value_estimates = deque(maxlen=BCELL_DECISION_BUFFER_SIZE)
        self.td_errors = deque(maxlen=BCELL_DECISION_BUFFER_SIZE)

        # 전문화 가중치
        self.specialization_weights = self._initialize_specialization(
            risk_type, n_assets
        )

        # 학습률 스케줄러
        self.actor_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.actor_optimizer,
            mode="max",
            factor=SCHEDULER_FACTOR,
            patience=SCHEDULER_PATIENCE,
        )
        self.critic_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.critic_optimizer,
            mode="max",
            factor=SCHEDULER_FACTOR,
            patience=SCHEDULER_PATIENCE,
        )

        # 손실 추적용 변수
        self.last_actor_loss = 0.0
        self.last_critic_loss = 0.0
        
        # 학습 통계 누적
        self.learning_stats = {
            'total_updates': 0,
            'actor_loss_sum': 0.0,
            'critic_loss_sum': 0.0,
            'weight_change_sum': 0.0,
            'last_summary_update': 0
        }

        # 활성화 임계값 실제 구현
        self.activation_threshold = ACTIVATION_THRESHOLDS["bcell"]

    def should_activate(self, stimulus_level):
        """활성화 임계값 기반 판단"""
        return stimulus_level > self.activation_threshold

    def _initialize_specialization(self, risk_type, n_assets):
        """위험 유형별 초기 특화 설정"""
        weights = torch.ones(n_assets) * 0.1

        if risk_type == "volatility":
            safe_indices = [6, 7, 8] if n_assets >= 9 else [n_assets - 1]
            for idx in safe_indices:
                if idx < n_assets:
                    weights[idx] = 0.3
        elif risk_type == "correlation":
            weights = torch.ones(n_assets) * (0.8 / n_assets)
        elif risk_type == "momentum":
            weights = torch.ones(n_assets) * 0.5
        elif risk_type == "liquidity":
            large_cap_indices = [0, 1, 2, 3] if n_assets >= 4 else list(range(n_assets))
            for idx in large_cap_indices:
                if idx < n_assets:
                    weights[idx] = 0.25

        return weights

    def _initialize_specialization_criteria(self):
        """위험 유형별 전문화 기준 설정"""
        criteria = {
            "volatility": {
                "feature_indices": [0, 5],
                "thresholds": [0.4, 0.3],
                "crisis_range": (0.3, 0.9),
            },
            "correlation": {
                "feature_indices": [1],
                "thresholds": [0.6],
                "crisis_range": (0.4, 1.0),
            },
            "momentum": {
                "feature_indices": [2],
                "thresholds": [0.2],
                "crisis_range": (0.2, 0.8),
            },
            "liquidity": {
                "feature_indices": [6],
                "thresholds": [0.4],
                "crisis_range": (0.3, 0.9),
            },
            "macro": {
                "feature_indices": [3, 4, 7],
                "thresholds": [0.5, 1.0, 0.5],
                "crisis_range": (0.4, 1.0),
            },
        }
        return criteria.get(
            self.risk_type,
            {"feature_indices": [0], "thresholds": [0.5], "crisis_range": (0.3, 0.8)},
        )

    def is_my_specialty_situation(self, market_features, crisis_level):
        """현재 상황이 전문 분야인지 판단"""
        criteria = self.specialization_criteria

        # 위기 수준 확인
        min_crisis, max_crisis = criteria["crisis_range"]
        if not (min_crisis <= crisis_level <= max_crisis):
            return False

        # 시장 특성 확인
        feature_indices = criteria["feature_indices"]
        thresholds = criteria["thresholds"]

        specialty_signals = 0
        for idx, threshold in zip(feature_indices, thresholds):
            if idx < len(market_features):
                if abs(market_features[idx]) >= threshold:
                    specialty_signals += 1

        required_signals = max(1, len(feature_indices) // 2)
        is_specialty = specialty_signals >= required_signals

        return is_specialty

    def queue_experience(self, market_features, crisis_level, action, reward, 
                        tcell_contributions=None, done=False):
        """RJH 브랜치의 핵심: MDP 전이를 보존하면서 경험을 큐에 쌓는다"""
        
        # 현재 상태 구성
        features_tensor = torch.FloatTensor(market_features)
        crisis_tensor = torch.FloatTensor([crisis_level])
        
        if tcell_contributions:
            attended_features, _ = self.attention_mechanism(features_tensor, tcell_contributions)
            current_combined_input = torch.cat([attended_features, crisis_tensor, self.specialization_weights])
        else:
            current_combined_input = torch.cat([features_tensor, crisis_tensor, self.specialization_weights])
        
        # 이전 보류 경험이 있으면 next_state_value 채워서 저장
        if self._pending_experience is not None and self.last_combined_input is not None:
            with torch.no_grad():
                next_v = self.critic_target_network(current_combined_input.unsqueeze(0)).item()
            self._pending_experience["next_state_value"] = float(next_v)
            self._pending_experience["next_state"] = market_features.copy()
            self.episode_buffer.append(self._pending_experience)
            
        # 현재 경험 생성
        experience = {
            "state": market_features.copy(),
            "action": action.copy(),
            "reward": float(reward),
            "next_state": None,  # 다음 스텝에서 채움
            "next_state_value": None,  # 다음 스텝에서 채움
            "done": done,
            "timestamp": datetime.now(),
            "is_specialty": self.is_my_specialty_situation(market_features, crisis_level),
            "tcell_contributions": tcell_contributions or {},
            "combined_input": current_combined_input.numpy()
        }
        
        # 터미널 상태면 즉시 저장
        if done:
            experience["next_state_value"] = 0.0  # 터미널 상태
            experience["next_state"] = market_features.copy()  # 터미널에서는 자기 자신
            self.episode_buffer.append(experience)
            self._pending_experience = None
        else:
            # 다음 스텝까지 보류
            self._pending_experience = experience
            
        # 현재 상태 저장
        self.last_combined_input = current_combined_input
        
        # 전문성에 따른 분류
        if experience["is_specialty"]:
            self.specialist_performance.append(reward)
        else:
            self.general_performance.append(reward)

    def add_experience(self, state, action, reward, next_state, done, tcell_contributions=None):
        """기존 메소드 - 하위 호환성 유지"""
        self.experience_buffer.push(state, action, reward, next_state, done)
        
        # 전문성 여부 판단을 위한 추가 저장
        crisis_level = np.mean(np.abs(state[-3:]))
        experience = {
            "state": state.copy(),
            "action": action.copy(),
            "reward": reward,
            "next_state": next_state.copy(),
            "done": done,
            "timestamp": datetime.now(),
            "is_specialty": self.is_my_specialty_situation(state, crisis_level),
            "tcell_contributions": tcell_contributions or {},
        }

        if experience["is_specialty"]:
            self.specialization_buffer.append(experience)
            self.specialist_performance.append(reward)
        else:
            self.general_buffer.append(experience)
            self.general_performance.append(reward)

    def learn_from_episode_buffer(self):
        """RJH 브랜치 방식: episode_buffer를 사용한 올바른 TD 학습"""
        
        # 최소 경험 조건 완화
        min_experiences = max(4, self.batch_size // 2)
        if len(self.episode_buffer) < min_experiences:
            return False
            
        try:
            # 동적 배치 크기
            available_experiences = len(self.episode_buffer)
            actual_batch_size = min(self.batch_size, available_experiences)
            
            # 경험 샘플링
            if available_experiences >= self.batch_size:
                indices = np.random.choice(available_experiences, actual_batch_size, replace=False)
                batch = [self.episode_buffer[i] for i in indices]
            else:
                batch = self.episode_buffer[-actual_batch_size:]
            
            # 배치 데이터 준비 (combined_input 사용)
            states = torch.FloatTensor([exp["combined_input"] for exp in batch])
            actions = torch.FloatTensor([exp["action"] for exp in batch])
            rewards = torch.FloatTensor([exp["reward"] for exp in batch])
            next_state_values = torch.FloatTensor([exp["next_state_value"] for exp in batch])
            terminals = torch.FloatTensor([1.0 if exp["done"] else 0.0 for exp in batch])
            
            # TD Target 계산 (terminal 상태 처리 포함)
            td_targets = rewards + self.gamma * next_state_values * (1.0 - terminals)
            
            # Critic 업데이트
            current_values = self.critic_network(states).squeeze()
            critic_loss = F.mse_loss(current_values, td_targets.detach())
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(), 1.0)
            self.critic_optimizer.step()
            
            # Actor 업데이트
            action_probs = self.actor_network(states)
            log_probs = torch.log(action_probs + 1e-8)
            
            with torch.no_grad():
                advantages = td_targets - current_values
                # Advantage 정규화
                if len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            policy_loss = -torch.mean(torch.sum(log_probs * actions, dim=1) * advantages)
            
            # 엔트로피 정규화 (과도한 균등화 억제)
            entropy = -torch.mean(torch.sum(action_probs * torch.log(action_probs + 1e-8), dim=1))
            total_actor_loss = policy_loss - self.entropy_coef * entropy
            
            self.actor_optimizer.zero_grad()
            total_actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_network.parameters(), 1.0)
            self.actor_optimizer.step()
            
            # 타겟 네트워크 소프트 업데이트
            self.update_counter += 1
            if self.update_counter % TARGET_UPDATE_FREQUENCY == 0:
                self._soft_update_target_network()
            
            # Epsilon 감소
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            
            # TD Error 저장
            td_error = torch.mean(torch.abs(td_targets - current_values)).item()
            self.td_errors.append(td_error)
            
            # 학습 통계 누적
            self.learning_stats['total_updates'] += 1
            self.learning_stats['actor_loss_sum'] += total_actor_loss.item()
            self.learning_stats['critic_loss_sum'] += critic_loss.item()
            
            # 50번마다 학습 요약 로깅
            if self.learning_stats['total_updates'] % 50 == 0:
                avg_actor_loss = self.learning_stats['actor_loss_sum'] / 50
                avg_critic_loss = self.learning_stats['critic_loss_sum'] / 50
                avg_td_error = np.mean(list(self.td_errors)[-50:]) if self.td_errors else 0.0
                
                self.logger.debug(f"{self.risk_type} B-Cell 학습 요약 (50회): "
                                f"평균 Actor 손실={avg_actor_loss:.6f}, "
                                f"평균 Critic 손실={avg_critic_loss:.6f}, "
                                f"평균 TD 오차={avg_td_error:.6f}")
                
                # 통계 초기화
                self.learning_stats['actor_loss_sum'] = 0.0
                self.learning_stats['critic_loss_sum'] = 0.0
            
            return True
            
        except Exception as e:
            self.logger.error(f"{self.risk_type} B-Cell 에피소드 버퍼 학습 오류: {e}")
            return False
    
    def learn_from_batch(self):
        """기존 방식 유지 - 하위 호환성"""
        
        # 새로운 방식이 가능하면 사용
        if len(self.episode_buffer) >= self.batch_size // 2:
            return self.learn_from_episode_buffer()
        
        # 기존 방식 폴백
        min_experiences = self.batch_size // 2
        if len(self.experience_buffer) < min_experiences:
            return False
            
        try:
            available_experiences = len(self.experience_buffer)
            actual_batch_size = min(self.batch_size, available_experiences)
            
            if available_experiences >= self.batch_size:
                indices = np.random.choice(available_experiences, actual_batch_size, replace=False)
                batch = [self.experience_buffer.buffer[i] for i in indices]
            else:
                batch = list(self.experience_buffer.buffer)[-actual_batch_size:]
            
            # 차원 호환성을 위해 state를 combined_input으로 변환
            processed_states = []
            processed_next_states = []
            
            for exp in batch:
                state = exp[0]
                next_state = exp[3]
                
                # state가 12차원이면 combined_input으로 확장
                if len(state) == 12:
                    # market_features(12) + crisis_level(1) + specialization_weights(n_assets)
                    crisis_level = 0.0  # 기본값
                    combined_state = np.concatenate([state, [crisis_level], self.specialization_weights.numpy()])
                    processed_states.append(combined_state)
                else:
                    processed_states.append(state)
                    
                # next_state도 동일하게 처리
                if len(next_state) == 12:
                    crisis_level = 0.0
                    combined_next_state = np.concatenate([next_state, [crisis_level], self.specialization_weights.numpy()])
                    processed_next_states.append(combined_next_state)
                else:
                    processed_next_states.append(next_state)
            
            states = torch.FloatTensor(processed_states)
            actions = torch.FloatTensor([exp[1] for exp in batch])
            rewards = torch.FloatTensor([exp[2] for exp in batch])
            next_states = torch.FloatTensor(processed_next_states)
            dones = torch.BoolTensor([exp[4] for exp in batch])
            
            current_q = self.critic_network(states).squeeze()
            with torch.no_grad():
                next_q = self.critic_target_network(next_states).squeeze()
                if len(next_q.shape) == 0:
                    next_q = next_q.unsqueeze(0)
                target_q = rewards + self.gamma * next_q * (~dones)
            
            critic_loss = F.mse_loss(current_q, target_q)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(), 1.0)
            self.critic_optimizer.step()
            
            action_probs = self.actor_network(states)
            log_probs = torch.log(action_probs + 1e-8)
            
            with torch.no_grad():
                advantages = target_q - current_q
                if len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            policy_loss = -torch.mean(torch.sum(log_probs * actions, dim=1) * advantages)
            entropy = -torch.mean(torch.sum(action_probs * torch.log(action_probs + 1e-8), dim=1))
            total_actor_loss = policy_loss - ENTROPY_BONUS * entropy
            
            self.actor_optimizer.zero_grad()
            total_actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_network.parameters(), 1.0)
            self.actor_optimizer.step()
            
            self.update_counter += 1
            if self.update_counter % TARGET_UPDATE_FREQUENCY == 0:
                self._soft_update_target_network()
            
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            
            self.learning_stats['total_updates'] += 1
            self.learning_stats['actor_loss_sum'] += total_actor_loss.item()
            self.learning_stats['critic_loss_sum'] += critic_loss.item()
            
            if self.learning_stats['total_updates'] % 50 == 0:
                avg_actor_loss = self.learning_stats['actor_loss_sum'] / 50
                avg_critic_loss = self.learning_stats['critic_loss_sum'] / 50
                
                self.logger.debug(f"{self.risk_type} B-Cell 기존 방식 학습 요약: "
                                f"평균 Actor 손실={avg_actor_loss:.6f}, "
                                f"평균 Critic 손실={avg_critic_loss:.6f}")
                
                self.learning_stats['actor_loss_sum'] = 0.0
                self.learning_stats['critic_loss_sum'] = 0.0
            
            return True
            
        except Exception as e:
            self.logger.error(f"{self.risk_type} B-Cell 기존 방식 학습 오류: {e}")
            return False

    def _soft_update_target_network(self):
        """타겟 네트워크 소프트 업데이트"""
        tau = self.tau
        for target_param, param in zip(
            self.critic_target_network.parameters(), 
            self.critic_network.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def finish_episode(self):
        """에피소드 종료 시 호출 - 보류된 경험 처리 및 학습"""
        
        # 보류된 경험이 있으면 터미널로 처리
        if self._pending_experience is not None:
            self._pending_experience["next_state_value"] = 0.0
            self._pending_experience["next_state"] = self._pending_experience["state"].copy()
            self._pending_experience["done"] = True
            self.episode_buffer.append(self._pending_experience)
            self._pending_experience = None
        
        # 에피소드 종료 후 학습
        if len(self.episode_buffer) >= self.batch_size // 2:
            learning_success = self.learn_from_episode_buffer()
            if learning_success:
                self.logger.debug(f"{self.risk_type} B-Cell 에피소드 종료 후 학습 완료")
        
        # 에피소드 버퍼에서 Experience Replay Buffer로 데이터 이전
        for exp in self.episode_buffer:
            if exp.get("next_state") is not None:
                self.experience_buffer.push(
                    exp["state"], 
                    exp["action"], 
                    exp["reward"],
                    exp["next_state"], 
                    exp["done"]
                )
        
        # 에피소드 버퍼 초기화
        self.episode_buffer.clear()
        self.last_combined_input = None
        
        return len(self.episode_buffer) == 0  # 성공적으로 정리되었는지 반환


    def _get_available_gpu_memory(self):
        """실제 GPU 메모리 관리 구현"""
        if not torch.cuda.is_available():
            return CPU_DEFAULT_MEMORY

        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory
        allocated_memory = torch.cuda.memory_allocated(device)
        cached_memory = torch.cuda.memory_reserved(device)

        free_memory = total_memory - allocated_memory - cached_memory
        return free_memory // (1024 * 1024)  # MB 단위

    def adjust_batch_size_dynamically(self):
        """메모리 기반 배치 크기 동적 조정"""
        available_memory = self._get_available_gpu_memory()

        if available_memory < GPU_MEMORY_THRESHOLD_1GB:  # 1GB 미만
            self.batch_size = max(16, self.batch_size // 4)
        elif available_memory < GPU_MEMORY_THRESHOLD_2GB:  # 2GB 미만
            self.batch_size = max(32, self.batch_size // 2)
        else:
            self.batch_size = min(DEFAULT_BATCH_SIZE, self.batch_size * 2)

    def produce_antibody(
        self, market_features, crisis_level, tcell_contributions=None, training=True
    ):
        """Actor-Critic 기반 전략 생성"""
        try:
            # 활성화 임계값 확인
            stimulus_level = np.mean(np.abs(market_features))
            if not self.should_activate(stimulus_level):
                return np.ones(self.n_assets) / self.n_assets, 0.1

            features_tensor = torch.FloatTensor(market_features)
            crisis_tensor = torch.FloatTensor([crisis_level])

            # 어텐션 메커니즘 적용 (gradient flow 보장)
            if tcell_contributions:
                attended_features, attention_weights = self.attention_mechanism(
                    features_tensor, tcell_contributions
                )
                combined_input = torch.cat(
                    [attended_features, crisis_tensor, self.specialization_weights]
                )
            else:
                combined_input = torch.cat(
                    [features_tensor, crisis_tensor, self.specialization_weights]
                )

            # Actor 네트워크로 정책 생성
            with torch.no_grad():
                action_probs = self.actor_network(combined_input.unsqueeze(0))
                strategy_tensor = action_probs.squeeze(0)

                # Critic 네트워크로 가치 평가
                state_value = self.critic_network(combined_input.unsqueeze(0))
                self.last_state_value = state_value.item()

            # 전문 상황 여부에 따른 조정
            is_specialty = self.is_my_specialty_situation(market_features, crisis_level)

            if is_specialty:
                strategy_tensor = self._apply_specialist_strategy(
                    strategy_tensor, market_features, crisis_level
                )
                confidence_multiplier = 1.0 + self.specialization_strength
            else:
                strategy_tensor = self._apply_conservative_adjustment(strategy_tensor)
                confidence_multiplier = 0.7

            # 탐험/활용 (training 모드에서만)
            if training and np.random.random() < self.epsilon:
                exploration_strength = 0.05 if is_specialty else 0.1
                noise = torch.randn_like(strategy_tensor) * exploration_strength
                strategy_tensor = strategy_tensor + noise
                strategy_tensor = F.softmax(strategy_tensor, dim=0)

            # 항체 강도 계산
            base_confidence = 1.0 - float(torch.std(strategy_tensor))
            final_strength = max(0.1, base_confidence * confidence_multiplier)

            # 가치 추정 저장
            self.value_estimates.append(self.last_state_value)

            return strategy_tensor.numpy(), final_strength

        except Exception as e:
            print(f"[경고] {self.risk_type} B-세포 전략 생성 오류: {e}")
            default_strategy = np.ones(self.n_assets) / self.n_assets
            return default_strategy, 0.1

    def _apply_specialist_strategy(
        self, strategy_tensor, market_features, crisis_level
    ):
        """전문가 전략 적용"""
        specialized_strategy = strategy_tensor.clone()

        if self.risk_type == "volatility" and crisis_level > 0.5:
            safe_indices = [6, 7, 8]
            for idx in safe_indices:
                if idx < len(specialized_strategy):
                    specialized_strategy[idx] *= 1.0 + self.specialization_strength

        elif self.risk_type == "correlation" and market_features[1] > 0.7:
            uniform_weight = torch.ones_like(specialized_strategy) / len(
                specialized_strategy
            )
            blend_ratio = 0.3 + self.specialization_strength * 0.2
            specialized_strategy = (
                1 - blend_ratio
            ) * specialized_strategy + blend_ratio * uniform_weight

        # 추가 전문 전략들...

        specialized_strategy = F.softmax(specialized_strategy, dim=0)
        return specialized_strategy

    def _apply_conservative_adjustment(self, strategy_tensor):
        """보수적 조정"""
        uniform_weight = torch.ones_like(strategy_tensor) / len(strategy_tensor)
        conservative_blend = 0.3
        conservative_strategy = (
            1 - conservative_blend
        ) * strategy_tensor + conservative_blend * uniform_weight
        return F.softmax(conservative_strategy, dim=0)

    def get_expertise_metrics(self):
        """전문성 지표 반환"""
        specialist_avg = (
            np.mean(self.specialist_performance) if self.specialist_performance else 0
        )
        general_avg = (
            np.mean(self.general_performance) if self.general_performance else 0
        )
        expertise_advantage = specialist_avg - general_avg if general_avg != 0 else 0

        return {
            "specialization_strength": self.specialization_strength,
            "specialist_experiences": len(self.specialization_buffer),
            "general_experiences": len(self.general_buffer),
            "specialist_avg_reward": specialist_avg,
            "general_avg_reward": general_avg,
            "expertise_advantage": expertise_advantage,
            "risk_type": self.risk_type,
            "avg_value_estimate": (
                np.mean(self.value_estimates) if self.value_estimates else 0.0
            ),
            "avg_td_error": (
                np.mean([abs(e) for e in self.td_errors]) if self.td_errors else 0.0
            ),
            "buffer_size": len(self.experience_buffer),
            "update_counter": self.update_counter,
        }
    
    def pretrain_with_imitation(self, expert_states: torch.Tensor, 
                               expert_actions: torch.Tensor) -> float:
        """모방학습을 통한 사전훈련 - 전문가 전략 모방"""
        self.actor_network.train()
        
        try:
            # 에이전트가 예측한 행동
            predicted_actions = self.actor_network(expert_states)
            
            # 전문가 행동과의 차이를 최소화 (MSE Loss)
            imitation_loss = F.mse_loss(predicted_actions, expert_actions)
            
            # 추가적인 정규화: 가중치 합이 1이 되도록 소프트맥스 적용
            predicted_actions_normalized = F.softmax(predicted_actions, dim=-1)
            expert_actions_normalized = F.softmax(expert_actions, dim=-1)
            
            # KL Divergence 추가 (분포 간 차이)
            kl_loss = F.kl_div(
                torch.log(predicted_actions_normalized + 1e-8),
                expert_actions_normalized,
                reduction='batchmean'
            )
            
            # 총 손실
            total_loss = imitation_loss + 0.1 * kl_loss
            
            # 역전파
            self.actor_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_network.parameters(), 1.0)
            self.actor_optimizer.step()
            
            # 개별 스텝 로그 제거 - 배치 통계만 기록
            
            return total_loss.item()
            
        except Exception as e:
            self.logger.error(f"{self.risk_type} B-Cell 모방학습 실패: {e}")
            return float('inf')
    
    def validate_pretrained_policy(self, test_states: torch.Tensor) -> Dict:
        """사전훈련된 정책 검증"""
        self.actor_network.eval()
        
        with torch.no_grad():
            predictions = self.actor_network(test_states)
            
            # 정책 검증 지표
            policy_entropy = -torch.sum(
                predictions * torch.log(predictions + 1e-8), dim=-1
            ).mean()
            
            max_weights = predictions.max(dim=-1)[0].mean()
            min_weights = predictions.min(dim=-1)[0].mean()
            weight_variance = predictions.var(dim=-1).mean()
            
            # 가중치 합 검증
            weight_sums = predictions.sum(dim=-1)
            weight_sum_error = torch.abs(weight_sums - 1.0).mean()
            
        return {
            "policy_entropy": policy_entropy.item(),
            "max_weight": max_weights.item(),
            "min_weight": min_weights.item(),
            "weight_variance": weight_variance.item(),
            "weight_sum_error": weight_sum_error.item(),
            "is_reasonable": (
                policy_entropy.item() > 0.5 and  # 적당한 다양성
                weight_sum_error.item() < 0.1 and  # 가중치 합이 1에 가까움
                max_weights.item() < 0.8  # 극단적 집중 방지
            )
        }
