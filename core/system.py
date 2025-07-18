# core/system.py

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import warnings
from agents import TCell, BCell, LegacyBCell, MemoryCell
from core.reward import RewardCalculator
from core.hierarchical import HierarchicalController
from xai import DecisionAnalyzer
from constant import *

warnings.filterwarnings("ignore")


class ImmunePortfolioSystem:
    """면역 포트폴리오 시스템"""

    def __init__(
        self,
        n_assets,
        n_tcells=DEFAULT_N_TCELLS,
        n_bcells=DEFAULT_N_BCELLS,
        random_state=None,
        use_learning_bcells=True,
        use_hierarchical=True,
        logging_level="full",
        output_dir=None,
    ):
        self.n_assets = n_assets
        self.use_learning_bcells = use_learning_bcells
        self.use_hierarchical = use_hierarchical
        self.logging_level = logging_level

        # T-세포 초기화
        self.tcells = [
            TCell(
                f"T{i}",
                sensitivity=0.05 + i * 0.02,
                random_state=None if random_state is None else random_state + i,
            )
            for i in range(n_tcells)
        ]

        # B-세포 초기화
        if use_learning_bcells:
            feature_size = FEATURE_SIZE
            input_size = feature_size + 1 + n_assets

            self.bcells = [
                BCell("B1-Vol", "volatility", input_size, n_assets),
                BCell("B2-Corr", "correlation", input_size, n_assets),
                BCell("B3-Mom", "momentum", input_size, n_assets),
                BCell("B4-Liq", "liquidity", input_size, n_assets),
                BCell("B5-Macro", "macro", input_size, n_assets),
            ]
            print("시스템 유형: 적응형 신경망 기반 BIPD 모델")

            # 계층적 제어기 초기화
            if use_hierarchical:
                expert_names = [bcell.risk_type for bcell in self.bcells]
                meta_input_size = feature_size + 4 + 5 + 5 + 3  # 29차원
                self.hierarchical_controller = HierarchicalController(
                    meta_input_size=meta_input_size,
                    num_experts=len(self.bcells),
                    expert_names=expert_names,
                    learning_rate=0.001,
                )
                print("계층적 제어 시스템이 활성화되었습니다.")
            else:
                self.hierarchical_controller = None
        else:
            self.bcells = [
                LegacyBCell("LB1-Vol", "volatility", self._volatility_response),
                LegacyBCell("LB2-Corr", "correlation", self._correlation_response),
                LegacyBCell("LB3-Mom", "momentum", self._momentum_response),
                LegacyBCell("LB4-Liq", "liquidity", self._liquidity_response),
                LegacyBCell("LB5-Macro", "macro", self._macro_response),
            ]
            print("시스템 유형: 규칙 기반 레거시 BIPD 모델")
            self.hierarchical_controller = None

        # 기억 세포
        self.memory_cell = MemoryCell()

        # 보상 계산기 초기화
        self.reward_calculator = RewardCalculator(
            lookback_window=20,
            transaction_cost_rate=0.001,
            target_volatility=0.15,
            target_max_drawdown=0.1,
        )

        # 포트폴리오 가중치
        self.base_weights = np.ones(n_assets) / n_assets
        self.current_weights = self.base_weights.copy()
        self.previous_weights = None

        # 시스템 상태
        self.immune_activation = 0.0
        self.crisis_level = 0.0

        # 분석 시스템
        self.analyzer = DecisionAnalyzer(output_dir=output_dir or ".")
        self.enable_logging = True

        # 로깅 레벨에 따른 설정
        if logging_level == "full":
            print("분석 시스템이 활성화되었습니다. (전체 로깅)")
        elif logging_level == "sample":
            print("분석 시스템이 활성화되었습니다. (샘플링 로깅)")
        elif logging_level == "minimal":
            print("분석 시스템이 활성화되었습니다. (최소 로깅)")
        else:
            print("분석 시스템이 활성화되었습니다.")

    def extract_market_features(self, market_data, lookback=DEFAULT_LOOKBACK):
        """시장 특성 추출"""
        if len(market_data) < lookback:
            return np.zeros(FEATURE_SIZE)

        return self._extract_technical_features(market_data, lookback)

    def _extract_basic_features(self, market_data, lookback=DEFAULT_LOOKBACK):
        """기본 특성 추출"""
        returns = market_data.pct_change().dropna()
        if len(returns) == 0:
            return np.zeros(8)

        recent_returns = returns.iloc[-lookback:]
        if len(recent_returns) == 0:
            return np.zeros(8)

        def safe_mean(x):
            if len(x) == 0 or x.isnull().all():
                return 0.0
            return x.mean() if not np.isnan(x.mean()) else 0.0

        def safe_std(x):
            if len(x) == 0 or x.isnull().all():
                return 0.0
            return x.std() if not np.isnan(x.std()) else 0.0

        def safe_corr(x):
            try:
                if len(x) <= 1 or x.isnull().all().all():
                    return 0.0
                corr_matrix = np.corrcoef(x.T)
                if np.isnan(corr_matrix).any():
                    return 0.0
                return np.mean(corr_matrix[~np.eye(corr_matrix.shape[0], dtype=bool)])
            except:
                return 0.0

        def safe_skew(x):
            try:
                skew_vals = x.skew()
                if skew_vals.isnull().all():
                    return 0.0
                return skew_vals.mean() if not np.isnan(skew_vals.mean()) else 0.0
            except:
                return 0.0

        def safe_kurtosis(x):
            try:
                kurt_vals = x.kurtosis()
                if kurt_vals.isnull().all():
                    return 0.0
                return kurt_vals.mean() if not np.isnan(kurt_vals.mean()) else 0.0
            except:
                return 0.0

        features = [
            safe_std(recent_returns.std()),
            safe_corr(recent_returns),
            safe_mean(recent_returns.mean()),
            safe_skew(recent_returns),
            safe_kurtosis(recent_returns),
            safe_std(recent_returns.std()),
            len(recent_returns[recent_returns.sum(axis=1) < -0.02])
            / max(len(recent_returns), 1),
            (
                max(recent_returns.max().max() - recent_returns.min().min(), 0)
                if not recent_returns.empty
                else 0
            ),
        ]

        features = np.array(features)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return features

    def _extract_technical_features(self, market_data, lookback=DEFAULT_LOOKBACK):
        """기술적 지표 기반 특성 추출"""
        if not hasattr(self, "train_features") or not hasattr(self, "test_features"):
            basic_features = self._extract_basic_features(market_data, lookback)
            return self._expand_to_12_features(basic_features)

        current_date = market_data.index[-1]

        if current_date in self.train_features.index:
            feature_data = self.train_features.loc[current_date]
        elif current_date in self.test_features.index:
            feature_data = self.test_features.loc[current_date]
        else:
            basic_features = self._extract_basic_features(market_data, lookback)
            return self._expand_to_12_features(basic_features)

        selected_features = []

        market_volatility = feature_data.get("market_volatility", 0.0)
        selected_features.append(np.clip(market_volatility * 5, 0, 1))

        market_correlation = feature_data.get("market_correlation", 0.5)
        selected_features.append(np.clip(abs(market_correlation), 0, 1))

        market_return = feature_data.get("market_return", 0.0)
        selected_features.append(np.clip(market_return * 10, -1, 1))

        vix_proxy = feature_data.get("vix_proxy", 0.1)
        selected_features.append(np.clip(vix_proxy * 3, 0, 1))

        market_stress = feature_data.get("market_stress", 0.0)
        selected_features.append(np.clip(market_stress / 10, 0, 1))

        rsi_cols = [col for col in feature_data.index if "_rsi" in col]
        if rsi_cols:
            avg_rsi = np.mean(
                [
                    feature_data[col]
                    for col in rsi_cols
                    if not pd.isna(feature_data[col])
                ]
            )
            rsi_risk = abs(avg_rsi - 50) / 50
            selected_features.append(np.clip(rsi_risk, 0, 1))
        else:
            selected_features.append(0.0)

        momentum_cols = [col for col in feature_data.index if "_momentum" in col]
        if momentum_cols:
            avg_momentum = np.mean(
                [
                    feature_data[col]
                    for col in momentum_cols
                    if not pd.isna(feature_data[col])
                ]
            )
            selected_features.append(np.clip(abs(avg_momentum), 0, 1))
        else:
            selected_features.append(0.0)

        bb_cols = [col for col in feature_data.index if "_bb_position" in col]
        if bb_cols:
            avg_bb_position = np.mean(
                [feature_data[col] for col in bb_cols if not pd.isna(feature_data[col])]
            )
            bb_risk = abs(avg_bb_position - 0.5) * 2
            selected_features.append(np.clip(bb_risk, 0, 1))
        else:
            selected_features.append(0.0)

        volume_cols = [col for col in feature_data.index if "_volume_ratio" in col]
        if volume_cols:
            avg_volume_ratio = np.mean(
                [
                    feature_data[col]
                    for col in volume_cols
                    if not pd.isna(feature_data[col])
                ]
            )
            volume_risk = abs(avg_volume_ratio - 1.0) / 2
            selected_features.append(np.clip(volume_risk, 0, 1))
        else:
            selected_features.append(0.0)

        range_cols = [col for col in feature_data.index if "_price_range" in col]
        if range_cols:
            avg_range = np.mean(
                [
                    feature_data[col]
                    for col in range_cols
                    if not pd.isna(feature_data[col])
                ]
            )
            selected_features.append(np.clip(avg_range * 2, 0, 1))
        else:
            selected_features.append(0.1)

        sma_cols = [col for col in feature_data.index if "_price_sma20_ratio" in col]
        if sma_cols:
            avg_sma_ratio = np.mean(
                [
                    feature_data[col]
                    for col in sma_cols
                    if not pd.isna(feature_data[col])
                ]
            )
            sma_risk = abs(avg_sma_ratio - 1.0)
            selected_features.append(np.clip(sma_risk, 0, 1))
        else:
            selected_features.append(0.0)

        vol_cols = [col for col in feature_data.index if "_volatility" in col]
        if vol_cols:
            avg_volatility = np.mean(
                [
                    feature_data[col]
                    for col in vol_cols
                    if not pd.isna(feature_data[col])
                ]
            )
            selected_features.append(np.clip(avg_volatility * 5, 0, 1))
        else:
            selected_features.append(0.1)

        while len(selected_features) < FEATURE_SIZE:
            selected_features.append(0.0)

        features = np.array(selected_features[:FEATURE_SIZE])
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return features

    def _expand_to_12_features(self, basic_features):
        """8개 기본 특성을 12개로 확장"""
        if len(basic_features) >= FEATURE_SIZE:
            return basic_features[:FEATURE_SIZE]

        expanded_features = list(basic_features)

        additional_features = [
            0.5,
            0.0,
            0.5,
            1.0,
        ]

        for i in range(FEATURE_SIZE - len(expanded_features)):
            if i < len(additional_features):
                expanded_features.append(additional_features[i])
            else:
                expanded_features.append(0.0)

        return np.array(expanded_features[:FEATURE_SIZE])

    def _get_dominant_risk(self, market_features):
        """지배적 위험 유형 계산"""
        risk_features = market_features[:5]
        dominant_risk_idx = np.argmax(np.abs(risk_features - np.mean(risk_features)))
        risk_map = {
            0: "volatility",
            1: "correlation",
            2: "momentum",
            3: "liquidity",
            4: "macro",
        }
        return risk_map.get(dominant_risk_idx, "volatility")

    def _extract_tcell_contributions(self, market_features):
        """T-Cell 분석에서 특성 기여도 추출"""
        tcell_contributions = {}

        if hasattr(self, "detailed_tcell_analysis") and self.detailed_tcell_analysis:
            detailed_logs = self.detailed_tcell_analysis.get("detailed_crisis_logs", [])

            for log in detailed_logs:
                feature_contributions = log.get("feature_contributions", {})
                for feature, contribution in feature_contributions.items():
                    if feature not in tcell_contributions:
                        tcell_contributions[feature] = []
                    tcell_contributions[feature].append(contribution)

            for feature in tcell_contributions:
                tcell_contributions[feature] = np.mean(tcell_contributions[feature])

        if not tcell_contributions and len(market_features) >= 5:
            risk_features = market_features[:5]
            mean_risk = np.mean(risk_features)

            tcell_contributions = {
                "volatility": abs(risk_features[0] - mean_risk),
                "correlation": abs(risk_features[1] - mean_risk),
                "momentum": abs(risk_features[2] - mean_risk),
                "liquidity": (
                    abs(risk_features[3] - mean_risk) if len(risk_features) > 3 else 0.0
                ),
                "macro": (
                    abs(risk_features[4] - mean_risk) if len(risk_features) > 4 else 0.0
                ),
            }

        return tcell_contributions

    def immune_response(self, market_features, training=False):
        """면역 반응 실행 (계층적 제어 적용)"""

        # T-세포 활성화 및 상세 위기 감지 로그 수집
        tcell_activations = []
        detailed_crisis_logs = []

        for tcell in self.tcells:
            activation = tcell.detect_anomaly(market_features)
            tcell_activations.append(activation)

            if hasattr(tcell, "last_crisis_detection") and tcell.last_crisis_detection:
                detailed_crisis_logs.append(tcell.last_crisis_detection)

        self.crisis_level = np.mean(tcell_activations)

        # 상세 T-cell 분석 정보 저장
        self.detailed_tcell_analysis = {
            "crisis_level": self.crisis_level,
            "detailed_crisis_logs": detailed_crisis_logs,
        }

        # T-Cell 특성 기여도 추출
        tcell_contributions = self._extract_tcell_contributions(market_features)

        # 기억 세포 확인
        memory_augmented_features = self.memory_cell.get_memory_augmented_features(
            market_features
        )
        if memory_augmented_features is not None:
            market_features = memory_augmented_features

        recalled_memory, memory_strength, multiple_memories = self.memory_cell.recall_memory(
            market_features, return_multiple=True
        )

        if recalled_memory and memory_strength > 0.8:
            bcell_decisions = [
                {
                    "id": "Memory",
                    "risk_type": "memory_recall",
                    "activation_level": memory_strength,
                    "antibody_strength": memory_strength,
                    "strategy_contribution": 1.0,
                    "specialized_for_today": True,
                    "multiple_memories_count": len(multiple_memories) if multiple_memories else 0,
                    "memory_diversity": (
                        len(set(m["memory"].get("context", {}).get("crisis_type", "unknown") 
                               for m in multiple_memories)) 
                        if multiple_memories else 0
                    ),
                }
            ]
            return recalled_memory["strategy"], "memory_response", bcell_decisions

        # B-세포 활성화
        if self.crisis_level > 0.15:
            if self.use_learning_bcells:
                # 계층적 제어 사용 여부에 따른 분기
                if self.use_hierarchical and self.hierarchical_controller:
                    return self._hierarchical_immune_response(
                        market_features, tcell_contributions, training
                    )
                else:
                    return self._ensemble_immune_response(
                        market_features, tcell_contributions, training
                    )
            else:
                return self._legacy_immune_response(market_features)

        return self.base_weights, "normal", []

    def _hierarchical_immune_response(
        self, market_features, tcell_contributions, training
    ):
        """계층적 제어를 사용한 면역 반응"""

        # Meta-Controller로 전문가 선택
        selected_expert_idx, selected_expert_name, selection_confidence, meta_info = (
            self.hierarchical_controller.select_expert(
                market_features=market_features,
                crisis_level=self.crisis_level,
                tcell_analysis=self.detailed_tcell_analysis,
                training=training,
            )
        )

        # 선택된 B-Cell 전문가로 전략 생성
        selected_bcell = self.bcells[selected_expert_idx]
        strategy, antibody_strength = selected_bcell.produce_antibody(
            market_features,
            self.crisis_level,
            tcell_contributions=tcell_contributions,
            training=training,
        )

        self.immune_activation = antibody_strength

        # B-세포 결정 정보 구성
        bcell_decisions = [
            {
                "id": selected_bcell.cell_id,
                "risk_type": selected_bcell.risk_type,
                "activation_level": 1.0,  # 선택된 전문가는 100% 활성화
                "antibody_strength": float(antibody_strength),
                "strategy_contribution": 1.0,
                "specialized_for_today": True,
                "selection_confidence": float(selection_confidence),
                "meta_reasoning": meta_info.get("selection_reasoning", []),
                "expert_probabilities": meta_info.get(
                    "expert_probabilities", []
                ).tolist(),
            }
        ]

        # 가중치 업데이트
        self.previous_weights = self.current_weights.copy()
        self.current_weights = strategy

        response_type = f"hierarchical_{selected_expert_name}"

        return strategy, response_type, bcell_decisions

    def _ensemble_immune_response(self, market_features, tcell_contributions, training):
        """앙상블을 사용한 면역 반응 (기존 방식)"""

        response_weights = []
        antibody_strengths = []

        for bcell in self.bcells:
            strategy, antibody_strength = bcell.produce_antibody(
                market_features,
                self.crisis_level,
                tcell_contributions=tcell_contributions,
                training=training,
            )
            response_weights.append(strategy)
            antibody_strengths.append(antibody_strength)

        if len(antibody_strengths) > 0 and sum(antibody_strengths) > 0:
            normalized_strengths = np.array(antibody_strengths) / sum(
                antibody_strengths
            )

            ensemble_strategy = np.zeros(self.n_assets)
            for i, (strategy, weight) in enumerate(
                zip(response_weights, normalized_strengths)
            ):
                ensemble_strategy += strategy * weight

            ensemble_strategy = ensemble_strategy / np.sum(ensemble_strategy)
            self.immune_activation = np.mean(antibody_strengths)

            # B-세포 정보 수집
            bcell_decisions = []
            dominant_risk = self._get_dominant_risk(market_features)

            for i, bcell in enumerate(self.bcells):
                bcell_decisions.append(
                    {
                        "id": bcell.cell_id,
                        "risk_type": bcell.risk_type,
                        "activation_level": float(normalized_strengths[i]),
                        "antibody_strength": float(antibody_strengths[i]),
                        "strategy_contribution": float(normalized_strengths[i]),
                        "specialized_for_today": bcell.risk_type == dominant_risk,
                    }
                )

            dominant_bcell_idx = np.argmax(antibody_strengths)
            response_type = f"ensemble_{self.bcells[dominant_bcell_idx].risk_type}"

            # 가중치 업데이트
            self.previous_weights = self.current_weights.copy()
            self.current_weights = ensemble_strategy

            return ensemble_strategy, response_type, bcell_decisions
        else:
            return self.base_weights, "fallback", []

    def _legacy_immune_response(self, market_features):
        """규칙 기반 면역 반응"""

        response_weights = []
        antibody_strengths = []

        for bcell in self.bcells:
            antibody_strength = bcell.produce_antibody(market_features)
            response_weight = bcell.response_strategy(
                self.crisis_level * antibody_strength
            )
            response_weights.append(response_weight)
            antibody_strengths.append(antibody_strength)

        best_response_idx = np.argmax(antibody_strengths)
        self.immune_activation = antibody_strengths[best_response_idx]

        bcell_decisions = [
            {
                "id": self.bcells[best_response_idx].cell_id,
                "risk_type": self.bcells[best_response_idx].risk_type,
                "activation_level": float(self.immune_activation),
                "antibody_strength": float(self.immune_activation),
                "strategy_contribution": 1.0,
                "specialized_for_today": True,
            }
        ]

        strategy = response_weights[best_response_idx]
        self.previous_weights = self.current_weights.copy()
        self.current_weights = strategy

        return (
            strategy,
            f"legacy_{self.bcells[best_response_idx].risk_type}",
            bcell_decisions,
        )

    def calculate_reward(self, portfolio_return):
        """고도화된 보상 계산 사용"""
        if self.previous_weights is None:
            self.previous_weights = self.base_weights.copy()

        reward_details = self.reward_calculator.calculate_comprehensive_reward(
            current_return=portfolio_return,
            previous_weights=self.previous_weights,
            current_weights=self.current_weights,
            market_features=getattr(self, "last_market_features", np.zeros(12)),
            crisis_level=self.crisis_level,
        )

        return reward_details["total_reward"]

    def update_hierarchical_learning(self, expert_performance):
        """계층적 제어기 학습 업데이트"""
        if self.use_hierarchical and self.hierarchical_controller:
            # 현재 상태 벡터 구성
            if hasattr(self, "last_market_features"):
                state_vector = self.hierarchical_controller._construct_meta_state(
                    self.last_market_features,
                    self.crisis_level,
                    self.detailed_tcell_analysis,
                )

                # 마지막 선택된 전문가 인덱스 가져오기
                if hasattr(self.hierarchical_controller, "last_selected_expert"):
                    selected_expert_idx = (
                        self.hierarchical_controller.last_selected_expert
                    )

                    # 메타 경험 추가
                    self.hierarchical_controller.add_meta_experience(
                        state_vector=state_vector,
                        selected_expert_idx=selected_expert_idx,
                        expert_performance=expert_performance,
                    )

                    # 주기적 메타 정책 학습
                    if len(self.hierarchical_controller.experience_buffer) >= 32:
                        self.hierarchical_controller.learn_meta_policy()

    def _volatility_response(self, activation_level):
        """변동성 위험 대응"""
        risk_reduction = activation_level * 0.3
        weights = self.base_weights * (1 - risk_reduction)
        safe_indices = [6, 7, 8]
        for idx in safe_indices:
            if idx < len(weights):
                weights[idx] += risk_reduction / len(safe_indices)
        return weights / np.sum(weights)

    def _correlation_response(self, activation_level):
        """상관관계 위험 대응"""
        diversification_boost = activation_level * 0.2
        weights = self.base_weights.copy()
        weights = weights * (1 - diversification_boost) + diversification_boost / len(
            weights
        )
        return weights / np.sum(weights)

    def _momentum_response(self, activation_level):
        """모멘텀 위험 대응"""
        neutral_adjustment = activation_level * 0.25
        weights = self.base_weights * (1 - neutral_adjustment) + (
            self.base_weights * neutral_adjustment
        )
        return weights / np.sum(weights)

    def _liquidity_response(self, activation_level):
        """유동성 위험 대응"""
        large_cap_boost = activation_level * 0.2
        weights = self.base_weights.copy()
        large_cap_indices = [0, 1, 2, 3]
        for idx in large_cap_indices:
            if idx < len(weights):
                weights[idx] += large_cap_boost / len(large_cap_indices)
        return weights / np.sum(weights)

    def _macro_response(self, activation_level):
        """거시경제 위험 대응"""
        defensive_boost = activation_level * 0.3
        weights = self.base_weights * (1 - defensive_boost)
        defensive_indices = [7, 8, 9]
        for idx in defensive_indices:
            if idx < len(weights):
                weights[idx] += defensive_boost / len(defensive_indices)
        return weights / np.sum(weights)

    def pretrain_bcells(self, market_data, episodes=500):
        """B-세포 사전 훈련"""
        if not self.use_learning_bcells:
            return

        print(f"B-세포 네트워크 사전 훈련을 시작합니다. (에피소드: {episodes})")

        expert_policy_functions = {
            "volatility": self._volatility_response,
            "correlation": self._correlation_response,
            "momentum": self._momentum_response,
            "liquidity": self._liquidity_response,
            "macro": self._macro_response,
        }

        loss_function = nn.MSELoss()

        for episode in tqdm(range(episodes), desc="사전 훈련 진행률"):
            start_idx = np.random.randint(
                20, len(market_data.pct_change().dropna()) - 50
            )
            current_data = market_data.iloc[:start_idx]
            market_features = self.extract_market_features(current_data)
            crisis_level = np.random.uniform(0.2, 0.8)

            for bcell in self.bcells:
                if bcell.risk_type in expert_policy_functions:
                    expert_action = expert_policy_functions[bcell.risk_type](
                        crisis_level
                    )
                    target_policy = torch.FloatTensor(expert_action)

                    features_tensor = torch.FloatTensor(market_features)
                    crisis_tensor = torch.FloatTensor([crisis_level])
                    specialization_tensor = bcell.specialization_weights
                    combined_input = torch.cat(
                        [features_tensor, crisis_tensor, specialization_tensor]
                    )
                    current_policy = bcell.actor_network(
                        combined_input.unsqueeze(0)
                    ).squeeze(0)

                    bcell.actor_optimizer.zero_grad()
                    loss = loss_function(current_policy, target_policy)
                    loss.backward()
                    bcell.actor_optimizer.step()

        print("B-세포 네트워크 사전 훈련이 완료되었습니다.")

    def update_memory(self, crisis_pattern, response_strategy, effectiveness):
        """기억 업데이트"""
        self.memory_cell.store_memory(crisis_pattern, response_strategy, effectiveness)

        if self.use_learning_bcells:
            for bcell in self.bcells:
                bcell.learn_from_experience(
                    crisis_pattern, self.crisis_level, effectiveness
                )
        else:
            for bcell in self.bcells:
                bcell.adapt_response(crisis_pattern, effectiveness)

    def get_hierarchical_metrics(self):
        """계층적 시스템 메트릭 반환"""
        if self.use_hierarchical and self.hierarchical_controller:
            return self.hierarchical_controller.get_hierarchical_metrics()
        else:
            return {"hierarchical_system": "disabled"}
