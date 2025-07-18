# agents/memory.py

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class MemoryCell:
    """기억 세포"""

    def __init__(self, max_memories=20):
        self.max_memories = max_memories
        self.crisis_memories = []

    def store_memory(self, crisis_pattern, response_strategy, effectiveness):
        """기억 저장"""
        memory = {
            "pattern": crisis_pattern.copy(),
            "strategy": response_strategy.copy(),
            "effectiveness": effectiveness,
            "strength": 1.0,
        }

        self.crisis_memories.append(memory)

        if len(self.crisis_memories) > self.max_memories:
            self.crisis_memories.sort(key=lambda x: x["effectiveness"])
            self.crisis_memories.pop(0)

    def recall_memory(self, current_pattern):
        """기억 회상"""
        if not self.crisis_memories:
            return None, 0.0

        similarities = []
        for memory in self.crisis_memories:
            similarity = cosine_similarity([current_pattern], [memory["pattern"]])[0][0]
            similarities.append(similarity * memory["effectiveness"])

        best_memory_idx = np.argmax(similarities)
        best_similarity = similarities[best_memory_idx]

        if best_similarity > 0.7:
            return self.crisis_memories[best_memory_idx], best_similarity

        return None, 0.0
