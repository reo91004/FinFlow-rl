# agents/bcell.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from collections import deque
from datetime import datetime
from .base import ImmuneCell


class StrategyNetwork(nn.Module):
    """전략 생성 신경망"""

    def __init__(self, input_size, n_assets, hidden_size=64):
        super(StrategyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, n_assets)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.softmax(x, dim=-1)


class BCell(ImmuneCell):
    """B-세포: 전문화된 대응 전략 생성"""

    def __init__(self, cell_id, risk_type, input_size, n_assets, learning_rate=0.001):
        super().__init__(cell_id)
        self.risk_type = risk_type
        self.n_assets = n_assets

        # 신경망 초기화
        self.strategy_network = StrategyNetwork(input_size, n_assets)
        self.optimizer = optim.Adam(
            self.strategy_network.parameters(), lr=learning_rate
        )

        # 강화학습 파라미터
        self.experience_buffer = []
        self.episode_buffer = []
        self.antibody_strength = 0.1
        self.epsilon = 0.3
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.05

        # 학습 설정
        self.batch_size = 32
        self.update_frequency = 10
        self.experience_count = 0

        # 전문화 관련 속성
        self.specialization_buffer = deque(maxlen=1000)
        self.general_buffer = deque(maxlen=500)
        self.specialization_strength = 0.1

        # 전문 분야별 특화 기준
        self.specialization_criteria = self._initialize_specialization_criteria()

        # 적응형 학습률
        self.adaptive_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.8, patience=15, verbose=False
        )

        # 성과 추적
        self.specialist_performance = deque(maxlen=50)
        self.general_performance = deque(maxlen=50)

        # 전문화 가중치
        self.specialization_weights = self._initialize_specialization(
            risk_type, n_assets
        )

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

        confidence_boost = 1.0 + self.specialization_strength * 0.5

        return is_specialty and (
            specialty_signals * confidence_boost >= required_signals
        )

    def produce_antibody(self, market_features, crisis_level, training=True):
        """전략 생성"""

        try:
            features_tensor = torch.FloatTensor(market_features)
            crisis_tensor = torch.FloatTensor([crisis_level])
            combined_input = torch.cat(
                [features_tensor, crisis_tensor, self.specialization_weights]
            )

            with torch.no_grad():
                raw_strategy = self.strategy_network(combined_input.unsqueeze(0))
                strategy_tensor = raw_strategy.squeeze(0)

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

            # 탐험/활용
            if training and np.random.random() < self.epsilon:
                exploration_strength = 0.05 if is_specialty else 0.1
                noise = torch.randn_like(strategy_tensor) * exploration_strength
                strategy_tensor = strategy_tensor + noise
                strategy_tensor = F.softmax(strategy_tensor, dim=0)

            # 마지막 행동 저장
            self.last_strategy = strategy_tensor

            # 강도 계산
            base_confidence = 1.0 - float(torch.std(strategy_tensor))
            final_strength = max(0.1, base_confidence * confidence_multiplier)
            self.antibody_strength = final_strength

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

        elif self.risk_type == "momentum" and abs(market_features[2]) > 0.3:
            if market_features[2] > 0:
                growth_indices = [0, 1, 4]
                for idx in growth_indices:
                    if idx < len(specialized_strategy):
                        specialized_strategy[idx] *= (
                            1.0 + self.specialization_strength * 0.5
                        )
            else:
                defensive_indices = [6, 7, 8]
                for idx in defensive_indices:
                    if idx < len(specialized_strategy):
                        specialized_strategy[idx] *= (
                            1.0 + self.specialization_strength * 0.8
                        )

        elif self.risk_type == "liquidity" and market_features[6] > 0.5:
            large_cap_indices = [0, 1, 2, 3]
            boost_factor = 1.0 + self.specialization_strength * 0.6
            for idx in large_cap_indices:
                if idx < len(specialized_strategy):
                    specialized_strategy[idx] *= boost_factor

        elif self.risk_type == "macro":
            defensive_indices = [7, 8, 9]
            boost_factor = 1.0 + self.specialization_strength * 0.7
            for idx in defensive_indices:
                if idx < len(specialized_strategy):
                    specialized_strategy[idx] *= boost_factor

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

    def add_experience(self, market_features, crisis_level, action, reward):
        """경험 저장"""

        experience = {
            "state": market_features.copy(),
            "crisis_level": crisis_level,
            "action": action.copy(),
            "reward": reward,
            "timestamp": datetime.now(),
            "is_specialty": self.is_my_specialty_situation(
                market_features, crisis_level
            ),
        }

        if experience["is_specialty"]:
            self.specialization_buffer.append(experience)
            self.specialist_performance.append(reward)
            self.specialization_strength = min(
                1.0, self.specialization_strength + 0.005
            )
        else:
            self.general_buffer.append(experience)
            self.general_performance.append(reward)

        self.episode_buffer.append(experience)
        self.experience_count += 1

    def learn_from_batch(self):
        """배치 학습"""
        if len(self.episode_buffer) < self.batch_size:
            return

        try:
            batch_size = min(self.batch_size, len(self.episode_buffer))
            batch = np.random.choice(self.episode_buffer, batch_size, replace=False)

            states = []
            actions = []
            rewards = []

            for exp in batch:
                features_tensor = torch.FloatTensor(exp["state"])
                crisis_tensor = torch.FloatTensor([exp["crisis_level"]])
                combined_state = torch.cat(
                    [features_tensor, crisis_tensor, self.specialization_weights]
                )
                states.append(combined_state)

                actions.append(torch.FloatTensor(exp["action"]))
                rewards.append(exp["reward"])

            states = torch.stack(states)
            actions = torch.stack(actions)
            rewards = torch.FloatTensor(rewards)

            if len(rewards) > 1:
                rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

            predicted_actions = self.strategy_network(states)

            log_probs = torch.log(predicted_actions + 1e-8)
            policy_loss = -torch.mean(
                log_probs * actions.unsqueeze(1) * rewards.unsqueeze(1)
            )

            entropy = -torch.mean(
                predicted_actions * torch.log(predicted_actions + 1e-8)
            )
            entropy_bonus = 0.01 * entropy

            total_loss = policy_loss - entropy_bonus

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.strategy_network.parameters(), 0.5)
            self.optimizer.step()

            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        except Exception as e:
            print(f"[경고] {self.risk_type} B-세포 배치 학습 오류: {e}")

    def learn_from_specialized_experience(self):
        """전문 분야 집중 학습"""

        if len(self.specialization_buffer) < self.batch_size:
            return False

        try:
            specialist_batch = list(self.specialization_buffer)[-self.batch_size :]

            states = []
            actions = []
            rewards = []

            for exp in specialist_batch:
                features_tensor = torch.FloatTensor(exp["state"])
                crisis_tensor = torch.FloatTensor([exp["crisis_level"]])
                combined_state = torch.cat(
                    [features_tensor, crisis_tensor, self.specialization_weights]
                )
                states.append(combined_state)

                actions.append(torch.FloatTensor(exp["action"]))
                rewards.append(exp["reward"])

            states = torch.stack(states)
            actions = torch.stack(actions)
            rewards = torch.FloatTensor(rewards)

            predicted_actions = self.strategy_network(states)
            log_probs = torch.log(predicted_actions + 1e-8)

            specialist_weight = 3.0
            specialist_loss = -torch.mean(
                log_probs
                * actions.unsqueeze(1)
                * rewards.unsqueeze(1)
                * specialist_weight
            )

            self.optimizer.zero_grad()
            specialist_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.strategy_network.parameters(), 0.5)
            self.optimizer.step()

            avg_specialist_reward = torch.mean(rewards).item()
            self.adaptive_scheduler.step(avg_specialist_reward)

            return True

        except Exception as e:
            print(f"[경고] {self.risk_type} B-세포 전문가 학습 오류: {e}")
            return False

    def end_episode(self):
        """에피소드 종료"""
        if len(self.episode_buffer) > 0:
            self.experience_buffer.extend(self.episode_buffer)

            if len(self.episode_buffer) >= self.batch_size:
                self.learn_from_batch()

            self.episode_buffer = []

            if len(self.experience_buffer) > 1000:
                self.experience_buffer = self.experience_buffer[-1000:]

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
            "specialization_ratio": len(self.specialization_buffer)
            / max(1, len(self.specialization_buffer) + len(self.general_buffer)),
            "risk_type": self.risk_type,
        }

    def learn_from_experience(self, market_features, crisis_level, effectiveness):
        """호환성 래퍼"""
        if len(market_features) >= 8:
            dummy_action = np.ones(self.n_assets) / self.n_assets
            self.add_experience(
                market_features, crisis_level, dummy_action, effectiveness
            )

            if self.experience_count % self.update_frequency == 0:
                self.learn_from_batch()

    def adapt_response(self, antigen_pattern, effectiveness):
        """호환성 래퍼"""
        if len(antigen_pattern) >= 8:
            crisis_level = 0.5
            self.learn_from_experience(antigen_pattern, crisis_level, effectiveness)


class LegacyBCell(ImmuneCell):
    """규칙 기반 B-세포"""

    def __init__(self, cell_id, risk_type, response_strategy):
        super().__init__(cell_id)
        self.risk_type = risk_type
        self.response_strategy = response_strategy
        self.antibody_strength = 0.1

    def produce_antibody(self, antigen_pattern):
        """전략 생성"""
        if hasattr(self, "learned_patterns"):
            similarities = [
                cosine_similarity([antigen_pattern], [pattern])[0][0]
                for pattern in self.learned_patterns
            ]
            max_similarity = max(similarities) if similarities else 0
        else:
            max_similarity = 0

        self.antibody_strength = min(1.0, max_similarity + 0.1)
        return self.antibody_strength

    def adapt_response(self, antigen_pattern, effectiveness):
        """적응적 학습"""
        if not hasattr(self, "learned_patterns"):
            self.learned_patterns = []

        if effectiveness > 0.6:
            self.learned_patterns.append(antigen_pattern.copy())
            if len(self.learned_patterns) > 10:
                self.learned_patterns.pop(0)
