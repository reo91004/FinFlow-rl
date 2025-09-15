# src/agents/memory.py

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import deque
from typing import Dict, List, Tuple, Optional
import pickle
from src.utils.logger import FinFlowLogger

class MemoryCell:
    """
    Memory Cell: k-NN 기반 경험 저장 및 검색
    
    과거 유사 상황을 찾아 의사결정 가이드 제공
    위기 수준별 계층화 및 시간적 다양성 고려
    """
    
    def __init__(self,
                 capacity: int = 50000,  # 500에서 50000으로 통일
                 embedding_dim: int = 32,
                 k_neighbors: int = 5,
                 similarity_threshold: float = 0.7):
        """
        Args:
            capacity: 메모리 용량
            embedding_dim: 임베딩 차원
            k_neighbors: 검색할 이웃 수
            similarity_threshold: 유사도 임계값
        """
        self.capacity = capacity
        self.embedding_dim = embedding_dim
        self.k_neighbors = k_neighbors
        self.similarity_threshold = similarity_threshold
        
        # Memory storage (Reservoir sampling을 위해 list 사용)
        self.memories = []  # deque 대신 list로 변경
        self.embeddings = []
        self.total_seen = 0  # Reservoir sampling용 카운터

        # Crisis-stratified storage
        self.stratified_memories = {
            'low': [],     # crisis < 0.3
            'medium': [],  # 0.3 <= crisis < 0.7
            'high': []     # crisis >= 0.7
        }
        
        # Performance tracking
        self.memory_stats = {
            'total_stored': 0,
            'total_recalled': 0,
            'successful_recalls': 0,
            'avg_similarity': 0.0
        }
        
        # Temporal diversity
        self.temporal_weights = np.exp(-np.linspace(0, 2, capacity))
        
        self.logger = FinFlowLogger("MemoryCell")
        self.logger.info(f"Memory Cell 초기화 - capacity={capacity}, k={k_neighbors}")
    
    def store(self, 
              state: np.ndarray,
              action: np.ndarray,
              reward: float,
              crisis_level: float,
              bcell_type: str,
              additional_info: Optional[Dict] = None):
        """
        경험 저장
        
        Args:
            state: 상태 벡터
            action: 포트폴리오 가중치
            reward: 보상
            crisis_level: 위기 수준
            bcell_type: 사용된 B-Cell 유형
            additional_info: 추가 정보
        """
        # Create embedding
        embedding = self._create_embedding(state, crisis_level)
        
        # Create memory object
        memory = {
            'state': state.copy(),
            'action': action.copy(),
            'reward': reward,
            'crisis_level': crisis_level,
            'bcell_type': bcell_type,
            'embedding': embedding,
            'timestamp': self.memory_stats['total_stored'],
            'info': additional_info or {}
        }
        
        # Reservoir sampling으로 저장 (기존 store 내부 로직만 교체)
        if len(self.memories) < self.capacity:
            self.memories.append(memory)
        else:
            # Reservoir sampling
            j = np.random.randint(0, self.total_seen + 1)
            if j < self.capacity:
                self.memories[j] = memory

        # Store in stratified memory (Reservoir sampling 적용)
        stratum = 'low' if crisis_level < 0.3 else 'medium' if crisis_level < 0.7 else 'high'
        stratum_cap = self.capacity // 3

        if len(self.stratified_memories[stratum]) < stratum_cap:
            self.stratified_memories[stratum].append(memory)
        else:
            # Reservoir sampling for stratum
            j = np.random.randint(0, self.total_seen + 1)
            if j < stratum_cap:
                self.stratified_memories[stratum][j] = memory

        self.total_seen += 1
        self.memory_stats['total_stored'] = self.total_seen
        
        # Update embeddings cache
        self._update_embeddings_cache()
        
        if self.memory_stats['total_stored'] % 100 == 0:
            self.logger.debug(f"메모리 저장: {len(self.memories)} / {self.capacity}")
    
    def recall(self, 
               current_state: np.ndarray,
               current_crisis: float,
               k: Optional[int] = None,
               use_stratified: bool = True) -> List[Dict]:
        """
        유사 경험 검색
        
        Args:
            current_state: 현재 상태
            current_crisis: 현재 위기 수준
            k: 검색할 메모리 수
            use_stratified: 계층적 검색 사용 여부
            
        Returns:
            similar_memories: 유사 경험 리스트
        """
        if len(self.memories) == 0:
            return []
        
        k = k or self.k_neighbors
        self.memory_stats['total_recalled'] += 1
        
        # Create query embedding
        query_embedding = self._create_embedding(current_state, current_crisis)
        
        if use_stratified:
            # Stratified sampling based on crisis level
            memories_to_search = self._get_stratified_candidates(current_crisis)
        else:
            memories_to_search = list(self.memories)
        
        if len(memories_to_search) == 0:
            return []
        
        # Compute similarities
        similarities = []
        for memory in memories_to_search:
            sim = self._compute_similarity(query_embedding, memory['embedding'])
            
            # Apply temporal diversity weight
            time_diff = self.memory_stats['total_stored'] - memory['timestamp']
            temporal_weight = np.exp(-time_diff / self.capacity)
            
            # Combine similarity with temporal weight
            weighted_sim = sim * (0.7 + 0.3 * temporal_weight)
            
            similarities.append((weighted_sim, memory))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # Filter by threshold and get top-k
        similar_memories = []
        total_sim = 0
        
        for sim, memory in similarities[:k]:
            if sim >= self.similarity_threshold:
                memory_copy = memory.copy()
                memory_copy['similarity'] = sim
                similar_memories.append(memory_copy)
                total_sim += sim
        
        # Update stats
        if len(similar_memories) > 0:
            self.memory_stats['successful_recalls'] += 1
            self.memory_stats['avg_similarity'] = \
                0.9 * self.memory_stats['avg_similarity'] + 0.1 * (total_sim / len(similar_memories))
        
        return similar_memories
    
    def get_memory_guidance(self, 
                           current_state: np.ndarray,
                           current_crisis: float) -> Dict:
        """
        메모리 기반 의사결정 가이드
        
        Args:
            current_state: 현재 상태
            current_crisis: 현재 위기 수준
            
        Returns:
            guidance: 의사결정 가이드 정보
        """
        similar_memories = self.recall(current_state, current_crisis)
        
        if len(similar_memories) == 0:
            return {
                'has_guidance': False,
                'confidence': 0.0,
                'recommended_action': None,
                'expected_reward': 0.0,
                'similar_count': 0
            }
        
        # Aggregate information from similar experiences
        total_weight = sum(m['similarity'] for m in similar_memories)
        
        # Weighted average of actions
        weighted_action = np.zeros_like(similar_memories[0]['action'])
        weighted_reward = 0
        bcell_votes = {}
        
        for memory in similar_memories:
            weight = memory['similarity'] / total_weight
            weighted_action += weight * memory['action']
            weighted_reward += weight * memory['reward']
            
            # Vote for B-Cell type
            bcell = memory['bcell_type']
            bcell_votes[bcell] = bcell_votes.get(bcell, 0) + weight
        
        # Normalize action to simplex
        weighted_action = weighted_action / weighted_action.sum()
        
        # Most voted B-Cell
        recommended_bcell = max(bcell_votes.items(), key=lambda x: x[1])[0]
        
        # Confidence based on similarity and count
        confidence = min(1.0, total_weight / len(similar_memories) * len(similar_memories) / self.k_neighbors)
        
        guidance = {
            'has_guidance': True,
            'confidence': confidence,
            'recommended_action': weighted_action,
            'recommended_bcell': recommended_bcell,
            'expected_reward': weighted_reward,
            'similar_count': len(similar_memories),
            'avg_similarity': total_weight / len(similar_memories),
            'bcell_distribution': bcell_votes
        }
        
        return guidance
    
    def _create_embedding(self, state: np.ndarray, crisis_level: float) -> np.ndarray:
        """
        상태와 위기 수준으로 임베딩 생성
        
        Simple linear projection for now
        """
        # Concatenate state and crisis
        full_state = np.concatenate([state, [crisis_level]])
        
        # Random projection (fixed for consistency)
        if not hasattr(self, '_projection_matrix'):
            np.random.seed(42)
            self._projection_matrix = np.random.randn(len(full_state), self.embedding_dim)
            self._projection_matrix /= np.linalg.norm(self._projection_matrix, axis=0)
        
        # Project to embedding space
        embedding = np.dot(full_state, self._projection_matrix)
        
        # L2 normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        return embedding
    
    def _compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """코사인 유사도 계산"""
        return float(cosine_similarity(
            embedding1.reshape(1, -1),
            embedding2.reshape(1, -1)
        )[0, 0])
    
    def _get_stratified_candidates(self, crisis_level: float) -> List[Dict]:
        """
        위기 수준에 따른 계층적 후보 선택
        """
        candidates = []
        
        # Primary stratum (same crisis level)
        if crisis_level < 0.3:
            primary = list(self.stratified_memories['low'])
            secondary = list(self.stratified_memories['medium'])
        elif crisis_level < 0.7:
            primary = list(self.stratified_memories['medium'])
            secondary = list(self.stratified_memories['low']) + \
                       list(self.stratified_memories['high'])
        else:
            primary = list(self.stratified_memories['high'])
            secondary = list(self.stratified_memories['medium'])
        
        # 70% from primary, 30% from secondary
        n_primary = int(0.7 * self.k_neighbors * 3)
        n_secondary = int(0.3 * self.k_neighbors * 3)
        
        candidates.extend(primary[-n_primary:] if len(primary) > n_primary else primary)
        candidates.extend(secondary[-n_secondary:] if len(secondary) > n_secondary else secondary)
        
        return candidates
    
    def _update_embeddings_cache(self):
        """임베딩 캐시 업데이트"""
        self.embeddings = [m['embedding'] for m in self.memories]
    
    def get_statistics(self) -> Dict:
        """메모리 통계 반환"""
        stats = self.memory_stats.copy()
        stats['memory_usage'] = len(self.memories) / self.capacity
        stats['stratified_counts'] = {
            k: len(v) for k, v in self.stratified_memories.items()
        }
        
        if stats['total_recalled'] > 0:
            stats['recall_success_rate'] = stats['successful_recalls'] / stats['total_recalled']
        else:
            stats['recall_success_rate'] = 0.0
        
        return stats
    
    def save(self, path: str):
        """메모리 저장"""
        with open(path, 'wb') as f:
            pickle.dump({
                'memories': list(self.memories),
                'stratified_memories': {
                    k: list(v) for k, v in self.stratified_memories.items()
                },
                'memory_stats': self.memory_stats,
                'projection_matrix': getattr(self, '_projection_matrix', None)
            }, f)
        self.logger.info(f"메모리 저장: {path}")
    
    def load(self, path: str):
        """메모리 로드"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.memories = deque(data['memories'], maxlen=self.capacity)
        
        for k, v in data['stratified_memories'].items():
            self.stratified_memories[k] = deque(v, maxlen=self.capacity // 3)
        
        self.memory_stats = data['memory_stats']
        
        if data['projection_matrix'] is not None:
            self._projection_matrix = data['projection_matrix']
        
        self._update_embeddings_cache()
        
        self.logger.info(f"메모리 로드: {path}")
    
    def get_memory_guidance(self, state: np.ndarray, crisis_level: float) -> Dict:
        """
        메모리 가이던스 생성 (Trainer 호환)
        
        Args:
            state: 현재 상태
            crisis_level: 현재 위기 수준
            
        Returns:
            guidance: 메모리 기반 가이던스
        """
        # Recall similar memories
        similar_memories = self.recall(state, crisis_level)
        
        # Create guidance (inline to avoid missing method)
        if similar_memories:
            # 유사 경험들의 가중 평균 액션
            weights = np.array([m['similarity'] for m in similar_memories])
            weights = weights / weights.sum()
            
            actions = np.array([m['action'] for m in similar_memories])
            recommended_action = np.average(actions, axis=0, weights=weights)
            
            # 평균 보상
            avg_reward = np.mean([m['reward'] for m in similar_memories])
            
            guidance = {
                'has_guidance': True,
                'recommended_action': recommended_action,
                'confidence': float(np.mean(weights)),
                'num_memories': len(similar_memories),
                'avg_reward': avg_reward
            }
        else:
            guidance = {
                'has_guidance': False,
                'recommended_action': None,
                'confidence': 0.0,
                'num_memories': 0,
                'avg_reward': 0.0
            }
        
        # Add tensor format for Trainer
        import torch
        if guidance['has_guidance']:
            guidance['tensor'] = torch.FloatTensor(guidance['recommended_action'])
        else:
            guidance['tensor'] = None
        
        return guidance
    
    def clear(self):
        """메모리 초기화"""
        self.memories.clear()
        for v in self.stratified_memories.values():
            v.clear()
        self.embeddings = []
        self.memory_stats = {
            'total_stored': 0,
            'total_recalled': 0,
            'successful_recalls': 0,
            'avg_similarity': 0.0
        }
        self.logger.info("메모리 초기화 완료")