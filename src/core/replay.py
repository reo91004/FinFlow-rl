# src/core/replay.py

"""
경험 재생 버퍼 (온라인 학습)

목적: 우선순위/하이브리드 재생 버퍼 구현
의존: numpy, torch
사용처: BCell (온라인 학습)
역할: 실시간 경험 저장 및 효율적 샘플링

구현 내용:
- PrioritizedReplayBuffer: TD 오차 기반 우선순위
- ReservoirBuffer: 균등 샘플링 (다양성 유지)
- HybridReplayBuffer: 우선순위 + 균등 샘플링
- Importance sampling 보정
"""

import numpy as np
import torch
from typing import Tuple, Optional, Dict, Any
from collections import deque
import random
from dataclasses import dataclass

@dataclass
class Transition:
    """경험 저장 단위"""
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool
    info: Dict[str, Any] = None

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer
    
    TD 오차가 큰 샘플을 우선적으로 학습
    """
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        """
        Args:
            capacity: 버퍼 크기
            alpha: 우선순위 지수 (0=uniform, 1=full priority)
            beta: Importance sampling 보정 (0=no correction, 1=full)
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.001
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0
    
    def push(self, transition: Transition):
        """경험 저장"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        
        # 새 경험은 최대 우선순위
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[list, np.ndarray, np.ndarray]:
        """
        우선순위 기반 샘플링
        
        Returns:
            transitions: 샘플된 경험들
            weights: Importance sampling weights
            indices: 샘플 인덱스 (우선순위 업데이트용)
        """
        if len(self.buffer) == 0:
            return [], np.array([]), np.array([])
        
        # 유효한 우선순위만 사용
        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # 샘플링
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        transitions = [self.buffer[idx] for idx in indices]
        
        # Importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # 정규화
        
        # Beta 점진적 증가
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return transitions, weights, indices
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """TD 오차 기반 우선순위 업데이트"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6  # 0 방지
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)

class ReservoirBuffer:
    """
    Reservoir Sampling Buffer
    
    균등한 확률로 과거 경험 유지 (다양성 확보)
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.total_seen = 0
    
    def push(self, transition: Transition):
        """Reservoir sampling으로 경험 저장"""
        self.total_seen += 1
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            # Reservoir sampling: 확률적 교체
            j = random.randint(0, self.total_seen - 1)
            if j < self.capacity:
                self.buffer[j] = transition
    
    def sample(self, batch_size: int) -> list:
        """균등 샘플링"""
        if len(self.buffer) < batch_size:
            return self.buffer
        return random.sample(self.buffer, batch_size)
    
    def get_all(self) -> list:
        """모든 경험 반환 (오프라인 학습용)"""
        return self.buffer.copy()
    
    def __len__(self):
        return len(self.buffer)

class HybridReplayBuffer:
    """
    Hybrid Replay Buffer: PER + Reservoir Sampling
    
    80% 우선순위 샘플링 + 20% 균등 샘플링으로 
    효율성과 다양성을 동시에 확보
    """
    
    def __init__(self, 
                 capacity: int,
                 reservoir_ratio: float = 0.2,
                 alpha: float = 0.6,
                 beta: float = 0.4):
        """
        Args:
            capacity: 전체 버퍼 크기
            reservoir_ratio: Reservoir 버퍼 비율 (0.2 = 20%)
            alpha: PER 우선순위 지수
            beta: PER importance sampling 보정
        """
        # 용량 분할
        self.reservoir_capacity = int(capacity * reservoir_ratio)
        self.prioritized_capacity = capacity - self.reservoir_capacity
        
        # 두 버퍼 초기화
        self.prioritized_buffer = PrioritizedReplayBuffer(
            self.prioritized_capacity, alpha, beta
        )
        self.reservoir_buffer = ReservoirBuffer(self.reservoir_capacity)
        
        # 샘플링 비율
        self.reservoir_ratio = reservoir_ratio
        
        # 통계
        self.total_pushed = 0
        self.total_sampled = 0
        
        from src.utils.logger import FinFlowLogger
        self.logger = FinFlowLogger("HybridReplayBuffer")
        self.logger.info(f"Hybrid Buffer 초기화 - PER: {self.prioritized_capacity}, Reservoir: {self.reservoir_capacity}")
    
    def push(self, transition: Transition):
        """
        경험 저장 (두 버퍼에 모두 저장)
        """
        # 두 버퍼에 모두 추가
        self.prioritized_buffer.push(transition)
        self.reservoir_buffer.push(transition)
        
        self.total_pushed += 1
        
        if self.total_pushed % 10000 == 0:
            self.logger.debug(f"Hybrid Buffer 상태 - PER: {len(self.prioritized_buffer)}, Reservoir: {len(self.reservoir_buffer)}")
    
    def sample(self, batch_size: int) -> Tuple[list, np.ndarray, np.ndarray]:
        """
        하이브리드 샘플링
        
        Returns:
            transitions: 샘플된 경험들
            weights: Importance sampling weights
            indices: 샘플 인덱스 (PER 부분만)
        """
        # 샘플 수 계산
        n_reservoir = int(batch_size * self.reservoir_ratio)
        n_prioritized = batch_size - n_reservoir
        
        # PER 샘플링
        per_transitions, per_weights, per_indices = [], np.array([]), np.array([])
        if len(self.prioritized_buffer) > 0 and n_prioritized > 0:
            actual_n_prioritized = min(n_prioritized, len(self.prioritized_buffer))
            per_transitions, per_weights, per_indices = \
                self.prioritized_buffer.sample(actual_n_prioritized)
        
        # Reservoir 샘플링
        reservoir_transitions = []
        if len(self.reservoir_buffer) > 0 and n_reservoir > 0:
            actual_n_reservoir = min(n_reservoir, len(self.reservoir_buffer))
            reservoir_transitions = self.reservoir_buffer.sample(actual_n_reservoir)
        
        # 병합
        all_transitions = per_transitions + reservoir_transitions
        
        # Reservoir 샘플의 weight는 1.0 (균등)
        reservoir_weights = np.ones(len(reservoir_transitions))
        all_weights = np.concatenate([per_weights, reservoir_weights]) if len(per_weights) > 0 else reservoir_weights
        
        # 정규화
        if len(all_weights) > 0:
            all_weights = all_weights / all_weights.max()
        
        self.total_sampled += len(all_transitions)
        
        # indices는 PER 부분만 반환 (우선순위 업데이트용)
        return all_transitions, all_weights, per_indices
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """
        PER 우선순위 업데이트
        """
        if len(indices) > 0:
            self.prioritized_buffer.update_priorities(indices, priorities)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        버퍼 통계 반환
        """
        stats = {
            'total_pushed': self.total_pushed,
            'total_sampled': self.total_sampled,
            'prioritized_size': len(self.prioritized_buffer),
            'reservoir_size': len(self.reservoir_buffer),
            'prioritized_capacity': self.prioritized_capacity,
            'reservoir_capacity': self.reservoir_capacity,
            'reservoir_ratio': self.reservoir_ratio,
            'beta': self.prioritized_buffer.beta
        }
        
        if self.total_pushed > 0:
            stats['diversity_score'] = len(self.reservoir_buffer) / self.reservoir_capacity
            stats['efficiency_score'] = len(self.prioritized_buffer) / self.prioritized_capacity
        
        return stats
    
    def __len__(self):
        # 전체 유니크 경험 수 (중복 제거는 복잡하므로 최대값 사용)
        return max(len(self.prioritized_buffer), len(self.reservoir_buffer))

