# core/hierarchical.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
from collections import deque
from datetime import datetime
from constant import *


class MetaController(nn.Module):
    """Meta-Controller: 상위 제어기 - 어떤 B-Cell 전문가를 선택할지 결정"""

    def __init__(self, input_size, num_experts, hidden_size=128):
        super(MetaController, self).__init__()
        self.input_size = input_size
        self.num_experts = num_experts

        # 상황 분석 네트워크
        self.situation_analyzer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
        )

        # 전문가 선택 네트워크
        self.expert_selector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_experts),
        )

        # 가치 추정 네트워크
        self.value_estimator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, state):
        situation_features = self.situation_analyzer(state)
        expert_logits = self.expert_selector(situation_features)
        state_value = self.value_estimator(situation_features)

        expert_probs = F.softmax(expert_logits, dim=-1)

        return expert_probs, state_value, situation_features


class HierarchicalController:
    """계층적 제어 시스템"""

    def __init__(
        self,
        meta_input_size: int,
        num_experts: int,
        expert_names: List[str],
        learning_rate: float = 0.001,
    ):

        self.num_experts = num_experts
        self.expert_names = expert_names

        # Meta-Controller 초기화
        self.meta_controller = MetaController(meta_input_size, num_experts)
        self.meta_optimizer = optim.Adam(
            self.meta_controller.parameters(), lr=learning_rate or DEFAULT_META_LR
        )

        # 학습 파라미터
        self.gamma = DEFAULT_GAMMA
        self.meta_batch_size = DEFAULT_BATCH_SIZE
        self.experience_buffer = deque(maxlen=1000)

        # 전문가 성과 추적
        self.expert_performance = {name: deque(maxlen=100) for name in expert_names}
        self.expert_selection_history = deque(maxlen=200)
        self.expert_transition_matrix = np.zeros((num_experts, num_experts))

        # 상황별 전문가 매핑 학습
        self.situation_expert_mapping = {}
        self.situation_clusters = []

        # 메타 레벨 성과 지표
        self.meta_level_rewards = deque(maxlen=100)
        self.expert_utilization = np.zeros(num_experts)

    def select_expert(
        self,
        market_features: np.ndarray,
        crisis_level: float,
        tcell_analysis: Dict,
        training: bool = True,
    ) -> Tuple[int, str, float, Dict]:
        """전문가 선택"""

        try:
            # 상태 벡터 구성
            state_vector = self._construct_meta_state(
                market_features, crisis_level, tcell_analysis
            )
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)

            # Meta-Controller로 전문가 선택
            with torch.no_grad() if not training else torch.enable_grad():
                expert_probs, state_value, situation_features = self.meta_controller(
                    state_tensor
                )

            # 전문가 선택 (훈련 시 확률적, 평가 시 결정적)
            if training:
                expert_dist = torch.distributions.Categorical(expert_probs)
                selected_expert_idx = expert_dist.sample().item()
            else:
                selected_expert_idx = torch.argmax(expert_probs).item()

            selected_expert_name = self.expert_names[selected_expert_idx]
            selection_confidence = expert_probs[0][selected_expert_idx].item()

            # 선택 이력 업데이트
            self._update_selection_history(
                selected_expert_idx, market_features, crisis_level
            )

            # 메타 정보 구성
            meta_info = {
                "expert_probabilities": expert_probs[0].detach().numpy(),
                "state_value": state_value.item(),
                "situation_features": situation_features[0].detach().numpy(),
                "crisis_classification": self._classify_crisis_situation(
                    tcell_analysis
                ),
                "selection_reasoning": self._generate_selection_reasoning(
                    selected_expert_name, expert_probs[0], tcell_analysis
                ),
            }

            return (
                selected_expert_idx,
                selected_expert_name,
                selection_confidence,
                meta_info,
            )

        except Exception as e:
            print(f"전문가 선택 중 오류 발생: {e}")
            # 폴백: 위기 수준 기반 휴리스틱 선택
            return self._fallback_expert_selection(crisis_level, tcell_analysis)

    def _construct_meta_state(
        self, market_features: np.ndarray, crisis_level: float, tcell_analysis: Dict
    ) -> np.ndarray:
        """메타 상태 벡터 구성"""

        # 모든 입력의 타입 검증 및 변환
        if not isinstance(market_features, np.ndarray):
            market_features = np.array(market_features, dtype=np.float32)

        if not isinstance(crisis_level, (int, float)):
            crisis_level = float(crisis_level) if crisis_level is not None else 0.0

        # 기본 시장 특성 (12차원)
        base_features = (
            market_features[:12]
            if len(market_features) >= 12
            else np.pad(
                market_features,
                (0, max(0, 12 - len(market_features))),
                mode="constant",
                constant_values=0.0,
            ).astype(np.float32)
        )

        # 위기 정보 (4차원)
        def safe_get(d, key, default=0.0):
            try:
                value = d.get(key, default) if isinstance(d, dict) else default
                return float(value) if value is not None else default
            except (TypeError, ValueError):
                return default

        crisis_features = np.array(
            [
                float(crisis_level),
                safe_get(tcell_analysis, "dominant_risk_intensity"),
                safe_get(tcell_analysis, "risk_diversity"),
                min(5.0, len(tcell_analysis.get("detected_risks", []))) / 5.0,
            ],
            dtype=np.float32,
        )

        # 전문가 성과 히스토리 (5차원)
        expert_performance_features = []
        for name in self.expert_names:
            try:
                performances = list(self.expert_performance.get(name, []))
                if performances:
                    avg_performance = np.mean(performances[-10:])
                    expert_performance_features.append(float(avg_performance))
                else:
                    expert_performance_features.append(0.0)
            except (TypeError, ValueError):
                expert_performance_features.append(0.0)

        expert_performance_features = np.array(
            expert_performance_features, dtype=np.float32
        )

        # 배열 길이 검증
        def ensure_length(arr, target_length, fill_value=0.0):
            if len(arr) == target_length:
                return arr
            elif len(arr) > target_length:
                return arr[:target_length]
            else:
                padding = np.full(
                    target_length - len(arr), fill_value, dtype=np.float32
                )
                return np.concatenate([arr, padding])

        # 모든 구성 요소의 길이 보장
        base_features = ensure_length(base_features, 12)
        crisis_features = ensure_length(crisis_features, 4)
        expert_performance_features = ensure_length(expert_performance_features, 5)

        # 최근 전문가 선택 패턴 (5차원)
        selection_distribution = np.zeros(5, dtype=np.float32)
        try:
            recent_selections = list(self.expert_selection_history)[-10:]
            if recent_selections:
                for selection in recent_selections:
                    expert_idx = selection.get("expert_idx", 0)
                    if 0 <= expert_idx < 5:
                        selection_distribution[expert_idx] += 1
                selection_distribution /= len(recent_selections)
        except Exception:
            pass  # 실패 시 0 배열 유지

        # 시간적 특성 (3차원)
        current_time = datetime.now()
        temporal_features = np.array(
            [
                current_time.hour / 24.0,
                current_time.weekday() / 7.0,
                (current_time.month - 1) / 12.0,
            ],
            dtype=np.float32,
        )

        # 안전한 concatenation
        try:
            meta_state = np.concatenate(
                [
                    base_features,  # 12차원
                    crisis_features,  # 4차원
                    expert_performance_features,  # 5차원
                    selection_distribution,  # 5차원
                    temporal_features,  # 3차원
                ]
            )  # 총 29차원

            # 최종 검증
            assert (
                meta_state.shape[0] == 29
            ), f"메타 상태 차원 오류: {meta_state.shape[0]} != 29"
            assert (
                meta_state.dtype == np.float32
            ), f"메타 상태 타입 오류: {meta_state.dtype}"

            return meta_state

        except Exception as e:
            print(f"[경고] 메타 상태 구성 실패: {e}")
            # 폴백: 기본 상태 반환
            return np.zeros(29, dtype=np.float32)

    def _classify_crisis_situation(self, tcell_analysis: Dict) -> str:
        """위기 상황 분류"""

        crisis_level = tcell_analysis.get("crisis_level", 0.0)
        dominant_risk = tcell_analysis.get("dominant_risk", "unknown")

        if crisis_level > 0.8:
            return f"severe_{dominant_risk}"
        elif crisis_level > 0.5:
            return f"moderate_{dominant_risk}"
        elif crisis_level > 0.2:
            return f"mild_{dominant_risk}"
        else:
            return "normal"

    def _generate_selection_reasoning(
        self, selected_expert: str, expert_probs: torch.Tensor, tcell_analysis: Dict
    ) -> List[str]:
        """선택 근거 생성"""

        reasoning = []

        # 선택 확신도 분석
        max_prob = torch.max(expert_probs).item()
        if max_prob > 0.7:
            reasoning.append(f"High confidence selection (probability: {max_prob:.3f})")
        elif max_prob > 0.4:
            reasoning.append(
                f"Moderate confidence selection (probability: {max_prob:.3f})"
            )
        else:
            reasoning.append(f"Low confidence selection (probability: {max_prob:.3f})")

        # 위기 상황 기반 근거
        crisis_level = tcell_analysis.get("crisis_level", 0.0)
        dominant_risk = tcell_analysis.get("dominant_risk", "unknown")

        if crisis_level > 0.5:
            reasoning.append(
                f"Selected {selected_expert} expert for {dominant_risk} risk management in crisis situation"
            )
        else:
            reasoning.append(
                f"Selected {selected_expert} expert for normal market conditions with {dominant_risk} focus"
            )

        # 전문가별 특화 근거
        specialist_reasoning = {
            "volatility": "to manage market volatility through defensive positioning",
            "correlation": "to enhance portfolio diversification during correlation breakdown",
            "momentum": "to capitalize on or defend against momentum shifts",
            "liquidity": "to ensure portfolio liquidity and reduce transaction costs",
            "macro": "to position for macroeconomic risk factors",
        }

        if selected_expert in specialist_reasoning:
            reasoning.append(specialist_reasoning[selected_expert])

        return reasoning

    def add_meta_experience(
        self,
        state_vector: np.ndarray,
        selected_expert_idx: int,
        expert_performance: float,
        next_state_vector: np.ndarray = None,
    ):
        """메타 레벨 경험 추가"""

        experience = {
            "state": state_vector.copy(),
            "action": selected_expert_idx,
            "reward": expert_performance,
            "next_state": (
                next_state_vector.copy() if next_state_vector is not None else None
            ),
            "timestamp": datetime.now(),
        }

        self.experience_buffer.append(experience)

        # 전문가 성과 업데이트
        expert_name = self.expert_names[selected_expert_idx]
        self.expert_performance[expert_name].append(expert_performance)

        # 메타 레벨 보상 추가
        self.meta_level_rewards.append(expert_performance)

        # 전문가 활용도 업데이트
        self.expert_utilization[selected_expert_idx] += 1

    def learn_meta_policy(self):
        """메타 정책 학습"""

        if len(self.experience_buffer) < self.meta_batch_size:
            return

        try:
            # 배치 샘플링
            batch_indices = np.random.choice(
                len(self.experience_buffer),
                size=min(self.meta_batch_size, len(self.experience_buffer)),
                replace=False,
            )

            batch = [self.experience_buffer[i] for i in batch_indices]

            # 배치 데이터 구성
            states = torch.FloatTensor([exp["state"] for exp in batch])
            actions = torch.LongTensor([exp["action"] for exp in batch])
            rewards = torch.FloatTensor([exp["reward"] for exp in batch])

            # 다음 상태가 있는 경우만 처리
            next_states = []
            has_next_state = []
            for exp in batch:
                if exp["next_state"] is not None:
                    next_states.append(exp["next_state"])
                    has_next_state.append(True)
                else:
                    next_states.append(np.zeros_like(exp["state"]))
                    has_next_state.append(False)

            next_states = torch.FloatTensor(next_states)
            has_next_state = torch.BoolTensor(has_next_state)

            # 현재 상태 가치 및 정책
            expert_probs, state_values, _ = self.meta_controller(states)
            current_values = state_values.squeeze()

            # 다음 상태 가치 (있는 경우만)
            with torch.no_grad():
                _, next_state_values, _ = self.meta_controller(next_states)
                next_values = next_state_values.squeeze()
                next_values = next_values * has_next_state.float()

            # TD Target 계산
            td_targets = rewards + self.gamma * next_values

            # Advantage 계산
            advantages = td_targets - current_values.detach()

            # 정책 손실 (Actor)
            log_probs = torch.log(expert_probs + 1e-8)
            selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()
            policy_loss = -torch.mean(selected_log_probs * advantages)

            # 가치 손실 (Critic)
            value_loss = F.mse_loss(current_values, td_targets)

            # 엔트로피 정규화
            entropy = -torch.mean(
                torch.sum(expert_probs * torch.log(expert_probs + 1e-8), dim=1)
            )
            entropy_bonus = 0.01 * entropy

            # 총 손실
            total_loss = policy_loss + 0.5 * value_loss - entropy_bonus

            # 역전파
            self.meta_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.meta_controller.parameters(), 0.5)
            self.meta_optimizer.step()

        except Exception as e:
            print(f"메타 정책 학습 중 오류 발생: {e}")

    def _update_selection_history(
        self, expert_idx: int, market_features: np.ndarray, crisis_level: float
    ):
        """선택 이력 업데이트"""

        selection_record = {
            "expert_idx": expert_idx,
            "expert_name": self.expert_names[expert_idx],
            "market_features": market_features.copy(),
            "crisis_level": crisis_level,
            "timestamp": datetime.now(),
        }

        self.expert_selection_history.append(selection_record)

        # 전문가 전환 매트릭스 업데이트
        if len(self.expert_selection_history) > 1:
            prev_expert = self.expert_selection_history[-2]["expert_idx"]
            self.expert_transition_matrix[prev_expert][expert_idx] += 1

    def _fallback_expert_selection(
        self, crisis_level: float, tcell_analysis: Dict
    ) -> Tuple[int, str, float, Dict]:
        """폴백 전문가 선택"""

        dominant_risk = tcell_analysis.get("dominant_risk", "volatility")

        # 휴리스틱 기반 선택
        if dominant_risk == "volatility" or crisis_level > 0.7:
            expert_idx = 0  # volatility expert
        elif dominant_risk == "correlation":
            expert_idx = 1  # correlation expert
        elif dominant_risk == "momentum":
            expert_idx = 2  # momentum expert
        elif dominant_risk == "liquidity":
            expert_idx = 3  # liquidity expert
        else:
            expert_idx = 4  # macro expert

        return expert_idx, self.expert_names[expert_idx], 0.5, {"fallback": True}

    def get_hierarchical_metrics(self) -> Dict:
        """계층적 시스템 메트릭"""

        # 전문가 활용 분포
        total_selections = np.sum(self.expert_utilization)
        utilization_distribution = self.expert_utilization / (total_selections + 1e-8)

        # 전문가 성과 통계
        expert_performance_stats = {}
        for name in self.expert_names:
            performances = list(self.expert_performance[name])
            if performances:
                expert_performance_stats[name] = {
                    "avg_performance": np.mean(performances),
                    "std_performance": np.std(performances),
                    "recent_performance": (
                        np.mean(performances[-10:])
                        if len(performances) >= 10
                        else np.mean(performances)
                    ),
                }
            else:
                expert_performance_stats[name] = {
                    "avg_performance": 0.0,
                    "std_performance": 0.0,
                    "recent_performance": 0.0,
                }

        # 메타 레벨 성과
        meta_performance = {
            "avg_meta_reward": (
                np.mean(self.meta_level_rewards) if self.meta_level_rewards else 0.0
            ),
            "meta_reward_trend": (
                np.mean(list(self.meta_level_rewards)[-20:])
                - np.mean(list(self.meta_level_rewards)[-40:-20])
                if len(self.meta_level_rewards) >= 40
                else 0.0
            ),
        }

        return {
            "expert_utilization_distribution": utilization_distribution.tolist(),
            "expert_performance_stats": expert_performance_stats,
            "meta_performance": meta_performance,
            "total_expert_selections": int(total_selections),
            "selection_diversity": (
                1.0 - np.max(utilization_distribution) if total_selections > 0 else 0.0
            ),
            "expert_transition_entropy": self._calculate_transition_entropy(),
        }

    def _calculate_transition_entropy(self) -> float:
        """전문가 전환 엔트로피 계산"""

        if np.sum(self.expert_transition_matrix) == 0:
            return 0.0

        # 전환 확률 계산
        transition_probs = self.expert_transition_matrix / (
            np.sum(self.expert_transition_matrix, axis=1, keepdims=True) + 1e-8
        )

        # 엔트로피 계산
        entropy = 0.0
        for i in range(self.num_experts):
            for j in range(self.num_experts):
                if transition_probs[i, j] > 0:
                    entropy -= transition_probs[i, j] * np.log(transition_probs[i, j])

        return entropy / (
            self.num_experts * np.log(self.num_experts)
        )  # 정규화된 엔트로피
