# src/agents/gating.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, List, Any
from dataclasses import dataclass
from collections import deque
from src.utils.logger import FinFlowLogger

@dataclass
class GatingDecision:
    """게이팅 결정 결과"""
    selected_bcell: str
    confidence: float
    weights: Dict[str, float]
    reasoning: str

class GatingNetwork(nn.Module):
    """
    Gating Network: B-Cell 선택을 위한 메타 컨트롤러
    
    위기 수준과 메모리 가이던스를 기반으로 최적 B-Cell 선택
    """
    
    def __init__(self,
                 state_dim: int,
                 hidden_dim: int = 256,
                 num_experts: int = 5,
                 temperature: float = 1.5,  # 1.0 → 1.5 (탐색 강화)
                 min_dwell_steps: int = 2,  # 5 → 2 (빠른 전환)
                 performance_maxlen: int = 100):  # deque maxlen for performance history
        """
        Args:
            state_dim: 상태 차원
            hidden_dim: 은닉층 차원
            num_experts: B-Cell 전문가 수
            temperature: 소프트맥스 온도
            min_dwell_steps: 최소 유지 스텝
            performance_maxlen: 성과 기록 최대 길이
        """
        super().__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.temperature = temperature
        self.min_dwell_steps = min_dwell_steps

        # B-Cell types
        self.bcell_types = ['volatility', 'correlation', 'momentum', 'defensive', 'growth']

        # Network layers
        self.fc1 = nn.Linear(state_dim + 11, hidden_dim)  # +11 for crisis(4) + memory(7)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_experts)

        # LayerNorm for stability
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(0.1)

        # Performance tracking
        self.performance_history = {bcell: deque(maxlen=performance_maxlen) for bcell in self.bcell_types}
        self.selection_count = {bcell: 0 for bcell in self.bcell_types}
        self.current_bcell = None
        self.dwell_counter = 0
        
        # Statistics
        self.total_decisions = 0
        self.switch_count = 0
        
        self.logger = FinFlowLogger("GatingNetwork")
        self.logger.info(f"Gating Network 초기화 - experts={num_experts}")
    
    def forward(self, 
                state: torch.Tensor,
                memory_guidance: Dict,
                crisis_level: float) -> GatingDecision:
        """
        B-Cell 선택
        
        Args:
            state: 상태 텐서
            memory_guidance: 메모리 가이던스
            crisis_level: 위기 수준
            
        Returns:
            decision: 게이팅 결정
        """
        # Prepare input
        crisis_features = self._encode_crisis(crisis_level)
        memory_features = self._encode_memory(memory_guidance)
        
        # Concatenate all features
        full_input = torch.cat([
            state,
            torch.FloatTensor(crisis_features).unsqueeze(0).to(state.device),
            torch.FloatTensor(memory_features).unsqueeze(0).to(state.device)
        ], dim=1)
        
        # Forward pass
        x = F.relu(self.ln1(self.fc1(full_input)))
        x = self.dropout(x)
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        logits = self.fc3(x)
        
        # Apply temperature
        weights = F.softmax(logits / self.temperature, dim=1)
        weights_dict = {
            self.bcell_types[i]: float(weights[0, i])
            for i in range(self.num_experts)
        }
        
        # Consider dwell time (sticky selection)
        if self.current_bcell and self.dwell_counter < self.min_dwell_steps:
            # Boost current B-Cell
            weights_dict[self.current_bcell] *= 1.5
            total = sum(weights_dict.values())
            weights_dict = {k: v/total for k, v in weights_dict.items()}
            self.dwell_counter += 1
        else:
            self.dwell_counter = 0
        
        # Select B-Cell
        selected_bcell = max(weights_dict.items(), key=lambda x: x[1])[0]
        
        # Track switches
        if self.current_bcell and self.current_bcell != selected_bcell:
            self.switch_count += 1
            self.dwell_counter = 0
        
        self.current_bcell = selected_bcell
        self.selection_count[selected_bcell] += 1
        self.total_decisions += 1
        
        # Generate reasoning
        reasoning = self._generate_reasoning(crisis_level, memory_guidance, weights_dict)
        
        decision = GatingDecision(
            selected_bcell=selected_bcell,
            confidence=weights_dict[selected_bcell],
            weights=weights_dict,
            reasoning=reasoning
        )
        
        return decision
    
    def update_performance(self, 
                          bcell_type: str,
                          reward: float,
                          info: Optional[Dict] = None):
        """
        B-Cell 성능 업데이트
        
        Args:
            bcell_type: B-Cell 유형
            reward: 획득 보상
            info: 추가 정보
        """
        if bcell_type in self.performance_history:
            self.performance_history[bcell_type].append(reward)
    
    def _encode_crisis(self, crisis_level: float) -> np.ndarray:
        """위기 수준을 특성으로 인코딩"""
        # 4D crisis encoding
        return np.array([
            crisis_level,
            crisis_level ** 2,  # Non-linear
            float(crisis_level > 0.3),  # Binary threshold
            float(crisis_level > 0.7)   # High crisis
        ])
    
    def _encode_memory(self, memory_guidance: Dict) -> np.ndarray:
        """메모리 가이던스를 특성으로 인코딩"""
        # 7D memory encoding
        if memory_guidance.get('has_guidance', False):
            bcell_dist = memory_guidance.get('bcell_distribution', {})
            return np.array([
                memory_guidance.get('confidence', 0),
                memory_guidance.get('expected_reward', 0),
                memory_guidance.get('avg_similarity', 0),
                memory_guidance.get('similar_count', 0) / 10,  # Normalize
                bcell_dist.get('volatility', 0),
                bcell_dist.get('momentum', 0),
                bcell_dist.get('defensive', 0)
            ])
        else:
            return np.zeros(7)
    
    def _generate_reasoning(self, 
                          crisis_level: float,
                          memory_guidance: Dict,
                          weights: Dict[str, float]) -> str:
        """선택 이유 생성"""
        selected = max(weights.items(), key=lambda x: x[1])[0]
        confidence = weights[selected]
        
        if crisis_level > 0.7:
            crisis_str = "높은 위기"
        elif crisis_level > 0.3:
            crisis_str = "중간 위기"
        else:
            crisis_str = "정상"
        
        reasoning = f"{crisis_str} 상황에서 {selected} 전략 선택 (신뢰도: {confidence:.2%})"
        
        if memory_guidance.get('has_guidance', False):
            reasoning += f", 유사 경험 {memory_guidance.get('similar_count', 0)}개 참조"
        
        return reasoning
    
    def get_statistics(self) -> Dict:
        """통계 정보 반환"""
        avg_performance = {}
        for bcell, history in self.performance_history.items():
            if history:
                avg_performance[bcell] = float(np.mean(history))
            else:
                avg_performance[bcell] = 0.0
        
        return {
            'total_decisions': self.total_decisions,
            'switch_count': self.switch_count,
            'switch_rate': self.switch_count / max(1, self.total_decisions),
            'selection_distribution': self.selection_count,
            'avg_performance': avg_performance,
            'current_bcell': self.current_bcell,
            'dwell_counter': self.dwell_counter
        }
    
    def reset(self):
        """상태 초기화"""
        self.current_bcell = None
        self.dwell_counter = 0
        self.logger.debug("Gating Network 상태 초기화")