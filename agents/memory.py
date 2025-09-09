# bipd/agents/memory.py

import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from collections import deque
import pickle
from utils.logger import BIPDLogger

class MemoryCell:
    """
    Memory 세포: 과거 경험 저장 및 유사 상황 회상
    
    유사한 시장 상황에서의 과거 의사결정과 그 결과를 저장하여
    현재 의사결정에 참고할 수 있도록 함
    """
    
    def __init__(self, capacity=500, embedding_dim=32, similarity_threshold=0.7):
        self.capacity = capacity
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold
        
        # 메모리 저장소
        self.memories = deque(maxlen=capacity)
        
        # 메모리 인덱싱용
        self.embeddings = deque(maxlen=capacity)
        
        # 다양성 개선 시스템
        self.diversity_config = {
            'similarity_relaxation_rate': 0.8,  # 유사도 완화: 원래 임계값의 80%
            'stratified_sampling': True,  # 계층적 샘플링 활성화
            'crisis_strata': [0.0, 0.3, 0.6, 1.0],  # 위기 수준별 계층 구분
            'temporal_diversity_weight': 0.3,  # 시간적 다양성 가중치
            'max_similar_memories': 3,  # 유사한 메모리 최대 개수 제한
        }
        
        # 계층별 메모리 인덱스 (위기 수준별 분류)
        self.stratified_indices = {
            'low_crisis': [],      # 0.0 <= crisis < 0.3
            'medium_crisis': [],   # 0.3 <= crisis < 0.6
            'high_crisis': [],     # 0.6 <= crisis <= 1.0
        }
        
        # 회상 통계
        self.recall_stats = {
            'total_recalls': 0,
            'successful_recalls': 0,
            'similarity_sum': 0.0,
            'diversity_recalls': 0,  # 다양성 기반 회상 횟수
            'stratified_recalls': 0,  # 계층적 회상 횟수
        }
        
        # 저장 통계
        self.storage_stats = {
            'total_stored': 0,
            'last_logged_count': 0
        }
        
        # 로거
        self.logger = BIPDLogger("MemoryCell")
        
        self.logger.info(
            f"Memory Cell이 초기화되었습니다. "
            f"용량={capacity}, 임베딩차원={embedding_dim}"
        )
    
    def store(self, state, action, reward, crisis_level, additional_info=None):
        """
        경험 저장 (CUDA 호환성을 위한 타입 변환)
        
        Args:
            state: np.array, 시장 상태
            action: np.array, 포트폴리오 가중치
            reward: float, 보상
            crisis_level: float, 위기 수준
            additional_info: dict, 추가 정보
        """
        # NumPy 타입을 Python native 타입으로 안전하게 변환
        reward = float(reward)
        crisis_level = float(crisis_level)
        try:
            # 임베딩 생성
            embedding = self._create_embedding(state, crisis_level)
            
            # 메모리 객체 생성
            memory = {
                'state': state.copy(),
                'action': action.copy(),
                'reward': float(reward),
                'crisis_level': float(crisis_level),
                'embedding': embedding,
                'timestamp': len(self.memories),  # 간단한 시간 정보
                'additional_info': additional_info or {}
            }
            
            # 저장
            self.memories.append(memory)
            self.embeddings.append(embedding)
            self.storage_stats['total_stored'] += 1
            
            # 계층적 인덱싱 업데이트
            memory_idx = len(self.memories) - 1
            self._update_stratified_indices(memory_idx, crisis_level)
            
            # 주기적 로깅 (500개 단위로, 중복 방지)
            current_count = len(self.memories)
            if (current_count % 500 == 0 and 
                current_count != self.storage_stats['last_logged_count']):
                self.storage_stats['last_logged_count'] = current_count
                self.logger.debug(
                    f"메모리 저장: {current_count}개 경험 보유 "
                    f"(총 {self.storage_stats['total_stored']}개 저장됨)"
                )
                
        except Exception as e:
            self.logger.error(f"메모리 저장 실패: {e}")
    
    def recall(self, current_state, current_crisis, k=5, return_similarities=False):
        """
        다양성 개선된 메모리 회상 시스템
        
        - 유사도 완화 (similarity relaxation)
        - 계층적 샘플링 (stratified sampling)
        - 시간적 다양성 (temporal diversity)
        
        Args:
            current_state: np.array, 현재 시장 상태
            current_crisis: float, 현재 위기 수준
            k: int, 반환할 유사 경험 수
            return_similarities: bool, 유사도도 함께 반환
            
        Returns:
            similar_memories: list of dict, 유사한 과거 경험들
        """
        if len(self.memories) < k:
            if return_similarities:
                return [], []
            return []
        
        try:
            # 다양성 기반 회상 vs 기존 방식 선택 (70% 다양성)
            use_diversity_recall = np.random.random() < 0.7
            
            if use_diversity_recall and self.diversity_config['stratified_sampling']:
                return self._diverse_recall(current_state, current_crisis, k, return_similarities)
            else:
                return self._traditional_recall(current_state, current_crisis, k, return_similarities)
                
        except Exception as e:
            self.logger.error(f"메모리 회상 실패: {e}")
            if return_similarities:
                return [], []
            return []
    
    def _diverse_recall(self, current_state, current_crisis, k, return_similarities):
        """다양성 기반 회상"""
        current_embedding = self._create_embedding(current_state, current_crisis)
        
        # 계층적 샘플링: 위기 수준별 메모리 풀 구성
        crisis_stratum = self._get_crisis_stratum(current_crisis)
        candidate_indices = self.stratified_indices.get(crisis_stratum, [])
        
        # 후보가 부족하면 다른 계층에서도 선택
        if len(candidate_indices) < k:
            for stratum_name, indices in self.stratified_indices.items():
                if stratum_name != crisis_stratum:
                    candidate_indices.extend(indices)
        
        if not candidate_indices:
            return ([], []) if return_similarities else []
        
        # 유사도 완화: 원래 임계값보다 낮은 기준 적용
        relaxed_threshold = self.similarity_threshold * self.diversity_config['similarity_relaxation_rate']
        
        # 유사도와 시간적 다양성 계산
        memory_scores = []
        current_timestamp = len(self.memories)
        
        for idx in candidate_indices:
            if idx >= len(self.memories):  # 인덱스 범위 확인
                continue
                
            memory = self.memories[idx]
            embedding = self.embeddings[idx]
            
            # 코사인 유사도 계산
            sim = cosine_similarity(
                current_embedding.reshape(1, -1),
                embedding.reshape(1, -1)
            )[0, 0]
            
            # 시간적 다양성 점수 (최근 메모리에 페널티)
            temporal_penalty = abs(current_timestamp - memory['timestamp']) / current_timestamp
            temporal_diversity = self.diversity_config['temporal_diversity_weight'] * temporal_penalty
            
            # 복합 점수 = 유사도 + 시간적 다양성
            composite_score = sim + temporal_diversity
            
            if sim >= relaxed_threshold:  # 완화된 유사도 임계값
                memory_scores.append((composite_score, sim, idx))
        
        # 복합 점수 순으로 정렬
        memory_scores.sort(key=lambda x: x[0], reverse=True)
        
        # 다양성 제한: 너무 유사한 메모리 개수 제한
        selected_memories = []
        selected_similarities = []
        similar_count = 0
        max_similar = self.diversity_config['max_similar_memories']
        
        for composite_score, sim, idx in memory_scores[:k * 2]:  # 여유분 확보
            if len(selected_memories) >= k:
                break
                
            # 높은 유사도 메모리 개수 제한
            if sim > 0.9 and similar_count >= max_similar:
                continue
                
            selected_memories.append(self.memories[idx])
            selected_similarities.append(sim)
            
            if sim > 0.9:
                similar_count += 1
        
        # 통계 업데이트
        self.recall_stats['total_recalls'] += 1
        self.recall_stats['diversity_recalls'] += 1
        self.recall_stats['stratified_recalls'] += 1
        
        if len(selected_memories) > 0:
            self.recall_stats['successful_recalls'] += 1
            self.recall_stats['similarity_sum'] += max(selected_similarities)
        
        # 주기적 로깅
        if self.recall_stats['total_recalls'] % 1000 == 0:
            self._log_diversity_stats()
        
        if return_similarities:
            return selected_memories, selected_similarities
        else:
            return selected_memories
    
    def _traditional_recall(self, current_state, current_crisis, k, return_similarities):
        """기존 방식의 회상 (호환성)"""
        current_embedding = self._create_embedding(current_state, current_crisis)
        
        # 모든 과거 경험과의 유사도 계산
        similarities = []
        for i, past_embedding in enumerate(self.embeddings):
            sim = cosine_similarity(
                current_embedding.reshape(1, -1),
                past_embedding.reshape(1, -1)
            )[0, 0]
            similarities.append((sim, i))
        
        # 유사도 순으로 정렬
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # 상위 k개 선택
        top_k = similarities[:k]
        
        # 임계값 이상인 것만 선택
        filtered_memories = []
        filtered_similarities = []
        
        for sim, idx in top_k:
            if sim >= self.similarity_threshold:
                filtered_memories.append(self.memories[idx])
                filtered_similarities.append(sim)
        
        # 통계 업데이트
        self.recall_stats['total_recalls'] += 1
        if len(filtered_memories) > 0:
            self.recall_stats['successful_recalls'] += 1
            self.recall_stats['similarity_sum'] += max(filtered_similarities)
        
        if return_similarities:
            return filtered_memories, filtered_similarities
        else:
            return filtered_memories
    
    def get_memory_guidance(self, current_state, current_crisis, k=3):
        """
        메모리 기반 의사결정 가이던스
        
        Returns:
            dict: 메모리 기반 권장사항
        """
        similar_memories, similarities = self.recall(
            current_state, current_crisis, k, return_similarities=True
        )
        
        if not similar_memories:
            return {
                'has_guidance': False,
                'recommended_action': None,
                'confidence': 0.0,
                'memory_count': 0,
                'message': '유사한 과거 경험이 없습니다.'
            }
        
        # 유사도 가중 평균으로 권장 행동 계산
        weighted_actions = []
        total_weight = 0
        
        for memory, sim in zip(similar_memories, similarities):
            weight = sim * (1 + memory['reward'])  # 보상이 높은 경험에 더 큰 가중치
            weighted_actions.append(memory['action'] * weight)
            total_weight += weight
        
        if total_weight > 0:
            recommended_action = np.sum(weighted_actions, axis=0) / total_weight
            # 정규화
            recommended_action = recommended_action / recommended_action.sum()
        else:
            recommended_action = None
        
        # 과거 성과 분석
        rewards = [mem['reward'] for mem in similar_memories]
        avg_reward = np.mean(rewards)
        success_rate = np.mean([r > 0 for r in rewards])
        
        guidance = {
            'has_guidance': True,
            'recommended_action': recommended_action,
            'confidence': float(np.mean(similarities)),
            'memory_count': len(similar_memories),
            'avg_past_reward': float(avg_reward),
            'success_rate': float(success_rate),
            'similar_situations': [
                {
                    'reward': float(mem['reward']),
                    'crisis_level': float(mem['crisis_level']),
                    'similarity': float(sim)
                }
                for mem, sim in zip(similar_memories, similarities)
            ],
            # XAI 확장 정보
            'similar_episodes': [f"Episode {mem.get('episode', i)}" for i, mem in enumerate(similar_memories)],
            'similarity_scores': [float(sim) for sim in similarities]
        }
        
        return guidance
    
    def _create_embedding(self, state, crisis_level):
        """
        상태를 저차원 임베딩으로 변환
        
        현재는 간단한 특성 선택 방식 사용
        향후 오토인코더나 PCA로 개선 가능
        """
        try:
            # 시장 특성 (첫 12개)
            market_features = state[:12] if len(state) >= 12 else state
            
            # 핵심 특성 선택 (embedding_dim에 맞춰)
            if len(market_features) >= self.embedding_dim - 1:
                # 가장 변동성이 큰 특성들 선택
                selected_features = market_features[:(self.embedding_dim-1)]
            else:
                # 패딩
                selected_features = np.zeros(self.embedding_dim - 1)
                selected_features[:len(market_features)] = market_features
            
            # 위기 수준 추가
            embedding = np.concatenate([selected_features, [crisis_level]])
            
            # 정규화
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"임베딩 생성 실패: {e}")
            return np.zeros(self.embedding_dim, dtype=np.float32)
    
    def get_statistics(self):
        """메모리 통계 정보"""
        if len(self.memories) == 0:
            return {'memory_count': 0}
        
        rewards = [mem['reward'] for mem in self.memories]
        crisis_levels = [mem['crisis_level'] for mem in self.memories]
        
        stats = {
            'memory_count': len(self.memories),
            'avg_reward': float(np.mean(rewards)),
            'reward_std': float(np.std(rewards)),
            'positive_experiences': int(np.sum([r > 0 for r in rewards])),
            'avg_crisis_level': float(np.mean(crisis_levels)),
            'crisis_std': float(np.std(crisis_levels)),
            'capacity_usage': float(len(self.memories) / self.capacity)
        }
        
        return stats
    
    def clear_memory(self):
        """메모리 초기화"""
        self.memories.clear()
        self.embeddings.clear()
        
        # 통계 초기화
        self.recall_stats = {
            'total_recalls': 0,
            'successful_recalls': 0,
            'similarity_sum': 0.0
        }
        self.storage_stats = {
            'total_stored': 0,
            'last_logged_count': 0
        }
        
        self.logger.info("메모리가 초기화되었습니다.")
    
    def save_memory(self, filepath):
        """메모리 저장"""
        try:
            # 저장 디렉토리 생성 보장
            base_dir = os.path.dirname(filepath)
            if base_dir:
                os.makedirs(base_dir, exist_ok=True)
            
            memory_data = {
                'memories': list(self.memories),
                'embeddings': list(self.embeddings),
                'capacity': self.capacity,
                'embedding_dim': self.embedding_dim,
                'similarity_threshold': self.similarity_threshold
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(memory_data, f)
            
            self.logger.info(f"메모리가 저장되었습니다: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"메모리 저장 실패: {e}")
            return False
    
    def load_memory(self, filepath):
        """메모리 로드"""
        try:
            with open(filepath, 'rb') as f:
                memory_data = pickle.load(f)
            
            self.memories = deque(memory_data['memories'], maxlen=self.capacity)
            self.embeddings = deque(memory_data['embeddings'], maxlen=self.capacity)
            self.capacity = memory_data['capacity']
            self.embedding_dim = memory_data['embedding_dim']
            self.similarity_threshold = memory_data['similarity_threshold']
            
            self.logger.info(
                f"메모리가 로드되었습니다: {filepath}, "
                f"{len(self.memories)}개 경험 복원"
            )
            return True
            
        except Exception as e:
            self.logger.error(f"메모리 로드 실패: {e}")
            return False
    
    # ===== 다양성 개선 시스템 헬퍼 메소드 =====
    
    def _update_stratified_indices(self, memory_idx: int, crisis_level: float) -> None:
        """계층적 인덱스 업데이트"""
        crisis_stratum = self._get_crisis_stratum(crisis_level)
        
        # 인덱스 추가
        self.stratified_indices[crisis_stratum].append(memory_idx)
        
        # deque 크기 제한에 따른 인덱스 정리 (메모리가 오래되어 제거된 경우)
        for stratum_name, indices in self.stratified_indices.items():
            # 유효하지 않은 인덱스 제거 (메모리 용량 초과로 제거된 경우)
            valid_indices = [idx for idx in indices if idx < len(self.memories)]
            self.stratified_indices[stratum_name] = valid_indices[-self.capacity:]  # 최신 인덱스만 유지
    
    def _get_crisis_stratum(self, crisis_level: float) -> str:
        """위기 수준을 계층으로 분류"""
        if crisis_level < 0.3:
            return 'low_crisis'
        elif crisis_level < 0.6:
            return 'medium_crisis'
        else:
            return 'high_crisis'
    
    def _log_diversity_stats(self) -> None:
        """다양성 시스템 통계 로깅"""
        total_recalls = self.recall_stats['total_recalls']
        diversity_recalls = self.recall_stats['diversity_recalls']
        stratified_recalls = self.recall_stats['stratified_recalls']
        successful_recalls = self.recall_stats['successful_recalls']
        
        if total_recalls > 0:
            success_rate = successful_recalls / total_recalls
            diversity_rate = diversity_recalls / total_recalls
            stratified_rate = stratified_recalls / total_recalls
            
            avg_similarity = (self.recall_stats['similarity_sum'] / successful_recalls 
                            if successful_recalls > 0 else 0)
            
            # 계층별 메모리 분포
            stratum_counts = {k: len(v) for k, v in self.stratified_indices.items()}
            
            self.logger.debug(
                f"Memory 다양성 통계 (1000회): 성공률={success_rate:.1%}, "
                f"다양성 회상={diversity_rate:.1%}, 계층적 회상={stratified_rate:.1%}, "
                f"평균 유사도={avg_similarity:.3f}, "
                f"계층 분포={stratum_counts}, 총 메모리={len(self.memories)}"
            )
    
    def get_diversity_stats(self) -> dict:
        """다양성 시스템 상세 통계 반환"""
        total_recalls = self.recall_stats['total_recalls']
        
        stats = {
            'total_recalls': total_recalls,
            'success_rate': self.recall_stats['successful_recalls'] / total_recalls if total_recalls > 0 else 0,
            'diversity_recall_rate': self.recall_stats['diversity_recalls'] / total_recalls if total_recalls > 0 else 0,
            'stratified_recall_rate': self.recall_stats['stratified_recalls'] / total_recalls if total_recalls > 0 else 0,
            'avg_similarity': (self.recall_stats['similarity_sum'] / self.recall_stats['successful_recalls'] 
                             if self.recall_stats['successful_recalls'] > 0 else 0),
            'stratum_distribution': {k: len(v) for k, v in self.stratified_indices.items()},
            'total_memories': len(self.memories),
            'diversity_config': self.diversity_config.copy()
        }
        
        return stats
    
    def configure_diversity(self, **kwargs) -> None:
        """다양성 설정 업데이트"""
        for key, value in kwargs.items():
            if key in self.diversity_config:
                old_value = self.diversity_config[key]
                self.diversity_config[key] = value
                self.logger.info(f"다양성 설정 업데이트: {key} = {old_value} → {value}")
            else:
                self.logger.warning(f"알 수 없는 다양성 설정: {key}")
    
    def reset_diversity_stats(self) -> None:
        """다양성 통계 초기화"""
        self.recall_stats = {
            'total_recalls': 0,
            'successful_recalls': 0,
            'similarity_sum': 0.0,
            'diversity_recalls': 0,
            'stratified_recalls': 0,
        }
        
        self.stratified_indices = {
            'low_crisis': [],
            'medium_crisis': [],
            'high_crisis': [],
        }
        
        self.logger.info("다양성 통계가 초기화되었습니다.")