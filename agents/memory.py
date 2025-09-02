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
<<<<<<< HEAD

        # 기억 저장소
        self.crisis_memories = []
        self.memory_embeddings = []
        self.memory_effectiveness = []
        self.memory_timestamps = []
        self.memory_contexts = []

        # 신경망 컴포넌트
        self.memory_embedding_net = MemoryEmbedding(
            24, embedding_dim
        )  # 12 features + 12 strategy
        self.memory_retrieval_net = MemoryRetrieval(
            12, 24
        )  # query: 12 features, memory: 24 (features + strategy)

        # 기억 강도 관리
        self.memory_strengths = []
        self.memory_access_counts = []
        self.decay_rate = 0.05

        # 성과 추적
        self.retrieval_success_rate = deque(maxlen=50)
        self.memory_utilization_rate = deque(maxlen=50)

        # 옵티마이저
        self.optimizer = torch.optim.Adam(
            list(self.memory_embedding_net.parameters())
            + list(self.memory_retrieval_net.parameters()),
            lr=DEFAULT_MEMORY_LR,
=======
        
        # 메모리 저장소
        self.memories = deque(maxlen=capacity)
        
        # 메모리 인덱싱용
        self.embeddings = deque(maxlen=capacity)
        
        # 회상 통계
        self.recall_stats = {
            'total_recalls': 0,
            'successful_recalls': 0,
            'similarity_sum': 0.0
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
>>>>>>> origin/dev
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
        유사한 과거 경험 회상
        
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
            # 현재 상황 임베딩
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
                
                # 매 1000회마다 통계 로깅
                if self.recall_stats['total_recalls'] % 1000 == 0:
                    success_rate = self.recall_stats['successful_recalls'] / self.recall_stats['total_recalls']
                    avg_similarity = self.recall_stats['similarity_sum'] / self.recall_stats['successful_recalls'] if self.recall_stats['successful_recalls'] > 0 else 0
                    self.logger.debug(
                        f"Memory 회상 통계 (1000회): 성공률={success_rate:.1%}, "
                        f"평균 유사도={avg_similarity:.3f}, 메모리 수={len(self.memories)}"
                    )
            
            if return_similarities:
                return filtered_memories, filtered_similarities
            else:
                return filtered_memories
                
        except Exception as e:
            self.logger.error(f"메모리 회상 실패: {e}")
            if return_similarities:
                return [], []
            return []
    
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
<<<<<<< HEAD
            # 현재 패턴을 쿼리로 사용
            query_tensor = torch.FloatTensor(current_pattern[:12]).unsqueeze(0)

            # 기억 은행 구성
            if len(self.memory_embeddings) == 0:
                return None, 0.0, None

            memory_bank = torch.stack(self.memory_embeddings)

            # 기억 검색
            retrieved_memory, attention_weights = self.memory_retrieval_net(
                query_tensor, memory_bank
            )

            # 계산 그래프에서 분리
            attention_weights = attention_weights.detach()
            retrieved_memory = retrieved_memory.detach()

            # 가장 유사한 기억 찾기
            best_memory_idx = torch.argmax(attention_weights[0]).item()
            best_similarity = attention_weights[0][best_memory_idx].item()

            # 임계값 확인
            if best_similarity < self.similarity_threshold:
                # 명시적 해제
                del query_tensor, memory_bank, attention_weights, retrieved_memory
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                return None, 0.0, None

            # 기억 강화
            self._strengthen_memory(best_memory_idx, best_similarity)

            best_memory = self.crisis_memories[best_memory_idx].copy()

            # 다중 기억 반환 옵션
            if return_multiple:
                # 상위 3개 기억 반환
                top_indices = torch.topk(
                    attention_weights[0], min(3, len(self.crisis_memories))
                )[1]
                multiple_memories = []
                for idx in top_indices:
                    idx = idx.item()
                    if (
                        attention_weights[0][idx].item()
                        > self.similarity_threshold * 0.7
                    ):
                        multiple_memories.append(
                            {
                                "memory": self.crisis_memories[idx].copy(),
                                "similarity": attention_weights[0][idx].item(),
                                "context": self.memory_contexts[idx],
                            }
                        )

                # 사용 후 텐서 정리
                del query_tensor, memory_bank
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                return best_memory, best_similarity, multiple_memories

            # 사용 후 텐서 정리
            del query_tensor, memory_bank
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return best_memory, best_similarity, multiple_memories

        except Exception as e:
            # 예외 발생 시에도 메모리 정리
            if "query_tensor" in locals():
                del query_tensor
            if "memory_bank" in locals():
                del memory_bank
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            print(f"기억 회상 중 오류 발생: {e}")
            return None, 0.0, None

    def get_memory_augmented_features(
        self, current_pattern: np.ndarray
    ) -> Optional[np.ndarray]:
        """기억 기반 특성 보강"""

        recalled_memory, similarity, _ = self.recall_memory(current_pattern)

        if recalled_memory is None or similarity < 0.5:
            return None

        try:
            # 현재 패턴과 기억된 패턴 결합
            memory_pattern = recalled_memory["pattern"][:12]
            current_pattern_truncated = current_pattern[:12]

            # 가중 평균으로 특성 보강
            weight = min(similarity, 0.8)  # 최대 80% 영향
            augmented_features = (
                1 - weight
            ) * current_pattern_truncated + weight * memory_pattern

            # 기억 기반 추가 특성
            memory_indicators = np.array(
                [
                    similarity,  # 기억 유사도
                    recalled_memory["effectiveness"],  # 과거 효과
                    recalled_memory["strength"],  # 기억 강도
                    len(self.crisis_memories) / self.max_memories,  # 기억 은행 포화도
                ]
            )

            # 결합된 특성 반환
            return np.concatenate([augmented_features, memory_indicators])

        except Exception as e:
            print(f"기억 기반 특성 보강 중 오류 발생: {e}")
            return None

    def update_memory_effectiveness(
        self, recent_pattern: np.ndarray, actual_effectiveness: float
    ):
        """최근 사용된 기억의 효과 업데이트"""

        recalled_memory, similarity, _ = self.recall_memory(recent_pattern)

        if recalled_memory is None:
            return

        try:
            # 해당 기억 찾기
            for i, memory in enumerate(self.crisis_memories):
                if np.allclose(
                    memory["pattern"], recalled_memory["pattern"], atol=1e-6
                ):
                    # 효과성 업데이트 (지수 이동 평균)
                    alpha = 0.3
                    old_effectiveness = self.memory_effectiveness[i]
                    new_effectiveness = (
                        alpha * actual_effectiveness + (1 - alpha) * old_effectiveness
                    )

                    self.memory_effectiveness[i] = new_effectiveness
                    self.crisis_memories[i]["effectiveness"] = new_effectiveness

                    # 성공/실패 기록
                    self.retrieval_success_rate.append(
                        1.0 if actual_effectiveness > 0.5 else 0.0
                    )
                    break

        except Exception as e:
            print(f"기억 효과성 업데이트 중 오류 발생: {e}")

    def _is_duplicate_memory(
        self,
        new_embedding: torch.Tensor,
        crisis_pattern: np.ndarray,
        response_strategy: np.ndarray,
    ) -> bool:
        """중복 기억 확인"""

        if len(self.memory_embeddings) == 0:
            return False

        try:
            # 임베딩 유사도 확인
            existing_embeddings = torch.stack(self.memory_embeddings)
            similarities = F.cosine_similarity(
                new_embedding.unsqueeze(0), existing_embeddings
            )
            max_similarity = torch.max(similarities).item()

            if max_similarity > 0.95:  # 매우 높은 유사도
                return True

            # 패턴 유사도 확인
            for memory in self.crisis_memories:
                pattern_similarity = cosine_similarity(
                    [crisis_pattern], [memory["pattern"]]
                )[0][0]
                strategy_similarity = cosine_similarity(
                    [response_strategy], [memory["strategy"]]
                )[0][0]

                if pattern_similarity > 0.9 and strategy_similarity > 0.9:
                    return True

            return False

        except Exception as e:
            print(f"중복 기억 확인 중 오류 발생: {e}")
            return False

    def _strengthen_memory(self, memory_index: int, similarity: float):
        """기억 강화"""

        if 0 <= memory_index < len(self.memory_strengths):
            # 접근 횟수 증가
            self.memory_access_counts[memory_index] += 1

            # 강도 증가 (상한선 있음)
            strength_boost = similarity * MEMORY_STRENGTH_FACTOR
            self.memory_strengths[memory_index] = min(
                2.0, self.memory_strengths[memory_index] + strength_boost
            )

            # 기억 객체 업데이트
            self.crisis_memories[memory_index]["strength"] = self.memory_strengths[
                memory_index
            ]

    def _manage_memory_capacity(self):
        """메모리 용량 관리"""

        if len(self.crisis_memories) <= self.max_memories:
            return

        # 점수 계산 (효과성, 강도, 최근성 고려)
        scores = []
        current_time = datetime.now()

        for i in range(len(self.crisis_memories)):
            effectiveness = self.memory_effectiveness[i]
            strength = self.memory_strengths[i]
            recency = 1.0 / (1.0 + (current_time - self.memory_timestamps[i]).days)
            access_frequency = self.memory_access_counts[i] / 10.0

            score = (
                effectiveness * 0.4
                + strength * 0.3
                + recency * 0.2
                + access_frequency * 0.1
            )
            scores.append(score)

        # 하위 기억들 제거
        removal_count = len(self.crisis_memories) - self.max_memories
        indices_to_remove = np.argsort(scores)[:removal_count]

        # 역순으로 제거 (인덱스 문제 방지)
        for idx in sorted(indices_to_remove, reverse=True):
            del self.crisis_memories[idx]
            del self.memory_embeddings[idx]
            del self.memory_effectiveness[idx]
            del self.memory_timestamps[idx]
            del self.memory_contexts[idx]
            del self.memory_strengths[idx]
            del self.memory_access_counts[idx]

    def decay_memories(self):
        """시간에 따른 기억 감쇠"""

        for i in range(len(self.memory_strengths)):
            # 시간 경과에 따른 감쇠
            self.memory_strengths[i] *= 1 - self.decay_rate
            self.crisis_memories[i]["strength"] = self.memory_strengths[i]

            # 최소 강도 보장
            if self.memory_strengths[i] < 0.1:
                self.memory_strengths[i] = 0.1

    def get_memory_statistics(self) -> Dict:
        """기억 시스템 통계"""

        if not self.crisis_memories:
            return {"total_memories": 0}

        return {
            "total_memories": len(self.crisis_memories),
            "avg_effectiveness": np.mean(self.memory_effectiveness),
            "avg_strength": np.mean(self.memory_strengths),
            "memory_utilization": len(self.crisis_memories) / self.max_memories,
            "recent_success_rate": (
                np.mean(self.retrieval_success_rate)
                if self.retrieval_success_rate
                else 0.0
            ),
            "most_accessed_memory_count": (
                max(self.memory_access_counts) if self.memory_access_counts else 0
            ),
            "oldest_memory_age": (
                (datetime.now() - min(self.memory_timestamps)).days
                if self.memory_timestamps
                else 0
            ),
        }

    def save_memory_bank(self, filepath: str):
        """기억 은행 저장"""

        memory_data = {
            "crisis_memories": self.crisis_memories,
            "memory_effectiveness": self.memory_effectiveness,
            "memory_timestamps": self.memory_timestamps,
            "memory_contexts": self.memory_contexts,
            "memory_strengths": self.memory_strengths,
            "memory_access_counts": self.memory_access_counts,
            "network_state": {
                "embedding_net": self.memory_embedding_net.state_dict(),
                "retrieval_net": self.memory_retrieval_net.state_dict(),
            },
        }

        with open(filepath, "wb") as f:
            pickle.dump(memory_data, f)

    def load_memory_bank(self, filepath: str):
        """기억 은행 로드"""

        try:
            with open(filepath, "rb") as f:
=======
            with open(filepath, 'rb') as f:
>>>>>>> origin/dev
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