# src/agents/memory.py

"""
Memory Cell: 위기별 성공 전략 기억 시스템

목적: k-NN 기반 경험 검색으로 위기 대응 전략 재사용
의존: sklearn (NearestNeighbors), logger.py
사용처: FinFlowTrainer._train_episode() (위기시 메모리 활용)
역할: 성공적인 위기 대응 경험 저장 및 유사 상황시 검색

구현 내용:
- 상위 30% 보상 경험만 선별 저장
- k-NN으로 유사 상태 검색 (default k=5)
- 위기 수준 유사도 필터링 (tolerance=0.2)
- 보상 가중 평균으로 추천 행동 생성
- 위기시 B-Cell 행동과 50:50 블렌딩
"""

import numpy as np
from typing import Dict, Optional, List
from collections import deque
from sklearn.neighbors import NearestNeighbors
from src.utils.logger import FinFlowLogger

class MemoryCell:
    """
    Memory Cell: 위기별 성공 전략 기억
    단순한 k-NN 기반 검색
    """

    def __init__(self, capacity: int = 1000, k_neighbors: int = 5):
        """
        Args:
            capacity: 메모리 크기
            k_neighbors: 검색할 이웃 수
        """
        self.capacity = capacity
        self.k_neighbors = k_neighbors
        self.logger = FinFlowLogger("MemoryCell")

        # 위기 상황별 메모리
        self.memories = deque(maxlen=capacity)

        # k-NN 검색기
        self.knn = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean')
        self.fitted = False

        # 메모리 통계
        self.memory_stats = {
            'total_stored': 0,
            'total_recalled': 0,
            'successful_recalls': 0,
            'average_reward': 0.0,
        }

        self.logger.info(f"Memory Cell 초기화 완료 (capacity={capacity}, k_neighbors={k_neighbors})")

    def store(self,
              state: np.ndarray,
              action: np.ndarray,
              reward: float,
              crisis_level: float):
        """
        성공적인 경험 저장

        Args:
            state: 상태
            action: 행동 (포트폴리오)
            reward: 보상
            crisis_level: 위기 수준
        """
        # 보상 임계값 계산 (동적)
        if len(self.memories) > 0:
            rewards = [m['reward'] for m in self.memories]
            threshold = np.percentile(rewards, 70)  # 상위 30%
        else:
            threshold = 0.0

        # 성공적인 경험만 저장
        if reward > threshold:
            self.memories.append({
                'state': state.copy(),
                'action': action.copy(),
                'reward': reward,
                'crisis_level': crisis_level
            })

            self.memory_stats['total_stored'] += 1
            self._update_average_reward()

            # k-NN 재학습
            if len(self.memories) >= self.k_neighbors:
                states = np.array([m['state'] for m in self.memories])
                self.knn.fit(states)
                self.fitted = True

            self.logger.debug(f"경험 저장: reward={reward:.4f}, crisis={crisis_level:.2f}")

    def recall(self, state: np.ndarray, crisis_level: float) -> Optional[np.ndarray]:
        """
        유사한 과거 성공 전략 검색

        Args:
            state: 현재 상태
            crisis_level: 현재 위기 수준

        Returns:
            추천 행동 또는 None
        """
        if not self.fitted or len(self.memories) < self.k_neighbors:
            return None

        self.memory_stats['total_recalled'] += 1

        # 유사한 상태 검색
        distances, indices = self.knn.kneighbors(state.reshape(1, -1))

        # 위기 수준이 비슷한 경험 필터링
        similar_memories = []
        crisis_tolerance = 0.2

        for idx in indices[0]:
            memory = self.memories[idx]
            if abs(memory['crisis_level'] - crisis_level) < crisis_tolerance:
                similar_memories.append(memory)

        if not similar_memories:
            # 위기 수준 조건 완화
            crisis_tolerance = 0.5
            for idx in indices[0]:
                memory = self.memories[idx]
                if abs(memory['crisis_level'] - crisis_level) < crisis_tolerance:
                    similar_memories.append(memory)

        if not similar_memories:
            return None

        # 보상 기반 가중 평균 행동
        rewards = np.array([m['reward'] for m in similar_memories])

        # 보상 정규화 (softmax)
        rewards_normalized = np.exp(rewards - np.max(rewards))
        weights = rewards_normalized / rewards_normalized.sum()

        recommended_action = np.average(
            [m['action'] for m in similar_memories],
            weights=weights,
            axis=0
        )

        self.memory_stats['successful_recalls'] += 1
        self.logger.debug(f"메모리 검색 성공: {len(similar_memories)}개 유사 경험 발견")

        return recommended_action

    def get_best_action_for_crisis(self, crisis_level: float) -> Optional[np.ndarray]:
        """
        특정 위기 수준에서 최고 성과 행동 반환

        Args:
            crisis_level: 위기 수준

        Returns:
            최고 성과 행동 또는 None
        """
        if not self.memories:
            return None

        # 해당 위기 수준의 경험 필터링
        crisis_tolerance = 0.3
        relevant_memories = [
            m for m in self.memories
            if abs(m['crisis_level'] - crisis_level) < crisis_tolerance
        ]

        if not relevant_memories:
            return None

        # 최고 보상 경험 찾기
        best_memory = max(relevant_memories, key=lambda x: x['reward'])
        return best_memory['action'].copy()

    def clear(self):
        """메모리 초기화"""
        self.memories.clear()
        self.fitted = False
        self.memory_stats = {
            'total_stored': 0,
            'total_recalled': 0,
            'successful_recalls': 0,
            'average_reward': 0.0,
        }
        self.logger.info("Memory Cell 초기화")

    def get_stats(self) -> Dict:
        """메모리 통계 반환"""
        stats = self.memory_stats.copy()

        if len(self.memories) > 0:
            rewards = [m['reward'] for m in self.memories]
            crisis_levels = [m['crisis_level'] for m in self.memories]

            stats.update({
                'memory_size': len(self.memories),
                'max_reward': max(rewards),
                'min_reward': min(rewards),
                'std_reward': np.std(rewards),
                'avg_crisis_level': np.mean(crisis_levels),
                'recall_success_rate': (
                    stats['successful_recalls'] / max(stats['total_recalled'], 1)
                ),
            })

        return stats

    def _update_average_reward(self):
        """평균 보상 업데이트"""
        if self.memories:
            rewards = [m['reward'] for m in self.memories]
            self.memory_stats['average_reward'] = np.mean(rewards)

    def get_crisis_distribution(self) -> Dict[str, int]:
        """저장된 경험의 위기 수준 분포"""
        distribution = {
            'low': 0,     # crisis < 0.3
            'medium': 0,  # 0.3 <= crisis < 0.7
            'high': 0     # crisis >= 0.7
        }

        for memory in self.memories:
            crisis = memory['crisis_level']
            if crisis < 0.3:
                distribution['low'] += 1
            elif crisis < 0.7:
                distribution['medium'] += 1
            else:
                distribution['high'] += 1

        return distribution

    def save(self, path: str):
        """메모리 저장"""
        import pickle
        save_dict = {
            'memories': list(self.memories),
            'memory_stats': self.memory_stats,
            'capacity': self.capacity,
            'k_neighbors': self.k_neighbors,
        }

        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)

        self.logger.info(f"Memory Cell 저장: {path}")

    def load(self, path: str):
        """메모리 로드"""
        import pickle
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)

        self.memories = deque(save_dict['memories'], maxlen=self.capacity)
        self.memory_stats = save_dict['memory_stats']
        self.capacity = save_dict['capacity']
        self.k_neighbors = save_dict['k_neighbors']

        # k-NN 재학습
        if len(self.memories) >= self.k_neighbors:
            states = np.array([m['state'] for m in self.memories])
            self.knn.fit(states)
            self.fitted = True

        self.logger.info(f"Memory Cell 로드: {path}, {len(self.memories)}개 메모리")