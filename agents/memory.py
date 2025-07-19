# agents/memory.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import pickle
from constant import *


class MemoryEmbedding(nn.Module):
    """기억 임베딩 네트워크"""

    def __init__(self, input_dim, embedding_dim=64):
        super(MemoryEmbedding, self).__init__()
        self.embedding_net = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, x):
        return self.embedding_net(x)


class MemoryRetrieval(nn.Module):
    """기억 검색 네트워크"""

    def __init__(self, query_dim, memory_dim, hidden_dim=64):
        super(MemoryRetrieval, self).__init__()
        self.query_projection = nn.Linear(query_dim, hidden_dim)
        self.memory_projection = nn.Linear(memory_dim, hidden_dim)
        self.attention_weights = nn.Linear(hidden_dim, 1)

    def forward(self, query, memory_bank):
        """
        query: 현재 상황 쿼리 (batch_size, query_dim)
        memory_bank: 기억 은행 (num_memories, memory_dim)
        """
        batch_size = query.size(0)
        num_memories = memory_bank.size(0)

        # 쿼리와 메모리 프로젝션
        query_proj = self.query_projection(query)  # (batch_size, hidden_dim)
        memory_proj = self.memory_projection(memory_bank)  # (num_memories, hidden_dim)

        # 어텐션 스코어 계산
        query_expanded = query_proj.unsqueeze(1).expand(
            -1, num_memories, -1
        )  # (batch_size, num_memories, hidden_dim)
        memory_expanded = memory_proj.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # (batch_size, num_memories, hidden_dim)

        combined = query_expanded * memory_expanded  # Element-wise multiplication
        attention_scores = self.attention_weights(combined).squeeze(
            -1
        )  # (batch_size, num_memories)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 가중 기억 검색
        retrieved_memories = torch.bmm(
            attention_weights.unsqueeze(1), memory_expanded
        ).squeeze(1)

        return retrieved_memories, attention_weights


class MemoryCell:
    """기억 기반 의사결정 지원 세포"""

    def __init__(self, max_memories=100, embedding_dim=64, similarity_threshold=0.8):
        self.max_memories = max_memories
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold

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

        # 최적화
        self.optimizer = torch.optim.Adam(
            list(self.memory_embedding_net.parameters())
            + list(self.memory_retrieval_net.parameters()),
            lr=DEFAULT_MEMORY_LR,
        )

    def store_memory(
        self,
        crisis_pattern: np.ndarray,
        response_strategy: np.ndarray,
        effectiveness: float,
        context: Dict = None,
    ):
        """기억 저장 (임베딩 기반)"""

        # 입력 검증
        if len(crisis_pattern) < 12 or len(response_strategy) < len(crisis_pattern):
            return False

        try:
            # 패턴과 전략 결합
            combined_input = np.concatenate(
                [
                    crisis_pattern[:12],  # 시장 특성
                    (
                        response_strategy[:12]
                        if len(response_strategy) >= 12
                        else np.pad(
                            response_strategy, (0, max(0, 12 - len(response_strategy)))
                        )
                    ),  # 전략
                ]
            )

            # 임베딩 생성
            input_tensor = torch.FloatTensor(combined_input)
            with torch.no_grad():
                embedding = self.memory_embedding_net(input_tensor)

            # 유사한 기억이 이미 있는지 확인
            if self._is_duplicate_memory(embedding, crisis_pattern, response_strategy):
                return False

            # 새 기억 추가
            memory = {
                "pattern": crisis_pattern.copy(),
                "strategy": response_strategy.copy(),
                "effectiveness": effectiveness,
                "strength": 1.0,
                "created_at": datetime.now(),
                "context": context or {},
            }

            self.crisis_memories.append(memory)
            self.memory_embeddings.append(embedding)
            self.memory_effectiveness.append(effectiveness)
            self.memory_timestamps.append(datetime.now())
            self.memory_contexts.append(context or {})
            self.memory_strengths.append(1.0)
            self.memory_access_counts.append(0)

            # 메모리 크기 관리
            if len(self.crisis_memories) > self.max_memories:
                self._manage_memory_capacity()

            return True

        except Exception as e:
            print(f"기억 저장 중 오류 발생: {e}")
            return False

    def recall_memory(
        self, current_pattern: np.ndarray, return_multiple: bool = False
    ) -> Tuple[Optional[Dict], float, Optional[List]]:
        """기억 회상 (신경망 기반 검색)"""

        if not self.crisis_memories or len(current_pattern) < 12:
            return None, 0.0, None

        try:
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
            strength_boost = similarity * 0.1
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
                memory_data = pickle.load(f)

            self.crisis_memories = memory_data["crisis_memories"]
            self.memory_effectiveness = memory_data["memory_effectiveness"]
            self.memory_timestamps = memory_data["memory_timestamps"]
            self.memory_contexts = memory_data["memory_contexts"]
            self.memory_strengths = memory_data["memory_strengths"]
            self.memory_access_counts = memory_data["memory_access_counts"]

            # 임베딩 재생성
            self.memory_embeddings = []
            for memory in self.crisis_memories:
                combined_input = np.concatenate(
                    [
                        memory["pattern"][:12],
                        (
                            memory["strategy"][:12]
                            if len(memory["strategy"]) >= 12
                            else np.pad(
                                memory["strategy"],
                                (0, max(0, 12 - len(memory["strategy"]))),
                            )
                        ),
                    ]
                )
                input_tensor = torch.FloatTensor(combined_input)
                with torch.no_grad():
                    embedding = self.memory_embedding_net(input_tensor)
                self.memory_embeddings.append(embedding)

            # 네트워크 상태 복원
            if "network_state" in memory_data:
                self.memory_embedding_net.load_state_dict(
                    memory_data["network_state"]["embedding_net"]
                )
                self.memory_retrieval_net.load_state_dict(
                    memory_data["network_state"]["retrieval_net"]
                )

            return True

        except Exception as e:
            print(f"기억 은행 로드 중 오류 발생: {e}")
            return False
