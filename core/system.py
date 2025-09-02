# core/system.py

import numpy as np
import os
from typing import Dict, Tuple, List, Optional
from agents import TCell, BCell, MemoryCell
from utils.logger import BIPDLogger
from config import *


class ImmunePortfolioSystem:
    """
    면역 시스템 기반 포트폴리오 관리자

    T-Cell, B-Cell, Memory Cell을 통합하여
    시장 상황에 적응적으로 반응하는 포트폴리오 전략 실행
    """

    def __init__(
        self, n_assets: int, state_dim: int, symbols: Optional[List[str]] = None
    ):
        self.n_assets = n_assets
        self.state_dim = state_dim
        self.symbols = symbols or SYMBOLS  # config.py의 SYMBOLS 사용

        # T-Cell (위기 감지)
        self.tcell = TCell(
            contamination=TCELL_CONTAMINATION,
            sensitivity=TCELL_SENSITIVITY,
            random_state=GLOBAL_SEED,
        )

        # B-Cell 전문가 그룹 (위험 유형별 특화)
        self.bcells = {
            "volatility": BCell(
                "volatility", state_dim, n_assets, ACTOR_LR, CRITIC_LR, ALPHA_LR
            ),  # 고변동성 시장
            "correlation": BCell(
                "correlation", state_dim, n_assets, ACTOR_LR, CRITIC_LR, ALPHA_LR
            ),  # 상관관계 변화
            "momentum": BCell(
                "momentum", state_dim, n_assets, ACTOR_LR, CRITIC_LR, ALPHA_LR
            ),  # 모멘텀 추세
            "defensive": BCell(
                "defensive", state_dim, n_assets, ACTOR_LR, CRITIC_LR, ALPHA_LR
            ),  # 방어적 전략
            "growth": BCell(
                "growth", state_dim, n_assets, ACTOR_LR, CRITIC_LR, ALPHA_LR
            ),  # 성장 중심
        }

        # Memory Cell (경험 저장 및 회상)
        self.memory = MemoryCell(
            capacity=MEMORY_CAPACITY,
            embedding_dim=EMBEDDING_DIM,
            similarity_threshold=0.7,
        )

        # 시스템 상태
        self.is_trained = False
        self.training_steps = 0
        self.decision_count = 0

        # 의사결정 히스토리
        self.decision_history = []
        self.performance_history = []

<<<<<<< HEAD
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
            return np.zeros(FEATURE_SIZE, dtype=np.float32)  # 명시적 타입 지정

        features = self._extract_technical_features(market_data, lookback)

        # 타입 검증 및 변환
        if not isinstance(features, np.ndarray):
            features = np.array(features, dtype=np.float32)

        # NaN/Inf 값 안전 처리
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # 크기 및 타입 검증
        if features.shape[0] != FEATURE_SIZE:
            if features.shape[0] > FEATURE_SIZE:
                features = features[:FEATURE_SIZE]
            else:
                padding = np.zeros(FEATURE_SIZE - features.shape[0], dtype=np.float32)
                features = np.concatenate([features, padding])

        return features.astype(np.float32)

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
        selected_features.append(np.clip(market_volatility * MARKET_VOLATILITY_SCALE, 0, 1))

        market_correlation = feature_data.get("market_correlation", 0.5)
        selected_features.append(np.clip(abs(market_correlation), 0, 1))

        market_return = feature_data.get("market_return", 0.0)
        selected_features.append(np.clip(market_return * MARKET_RETURN_SCALE, -1, 1))

        vix_proxy = feature_data.get("vix_proxy", 0.1)
        selected_features.append(np.clip(vix_proxy * VIX_PROXY_SCALE, 0, 1))

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
            bb_risk = abs(avg_bb_position - 0.5) * BOLLINGER_SCALE
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
            selected_features.append(np.clip(avg_range * PRICE_RANGE_SCALE, 0, 1))
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
            selected_features.append(np.clip(avg_volatility * MARKET_VOLATILITY_SCALE, 0, 1))
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
=======
        # 메모리 가이던스 통계
        self.memory_stats = {
            "guidance_applied": 0,
            "total_guidance_attempts": 0,
            "confidence_sum": 0.0,
            "last_report_step": 0,
>>>>>>> origin/dev
        }

        # B-Cell 성과 추적 시스템
        self.bcell_performance = {
            name: {
                "recent_rewards": [],  # 최근 10회 성과
                "consecutive_selections": 0,
                "total_selections": 0,
            }
            for name in self.bcells.keys()
        }

        # 로깅 통계 (장황한 로그 줄이기)
        self.logging_stats = {
            "random_selections": 0,
            "penalty_applications": 0,
            "last_log_report": 0,
            "log_interval": 100,  # 100번마다 통계 요약 출력
        }

        self.logger = BIPDLogger("ImmuneSystem")

        self.logger.info(
            f"면역 포트폴리오 시스템이 초기화되었습니다. "
            f"자산수={n_assets}, 상태차원={state_dim}, "
            f"B-Cell={len(self.bcells)}개, "
            f"Device={get_device_info()}"
        )

    def fit_tcell(self, historical_features: np.ndarray) -> bool:
        """T-Cell 학습 (정상 시장 패턴)"""
        success = self.tcell.fit(historical_features)
        if success:
            self.logger.info("T-Cell 학습이 완료되었습니다.")
        return success

    def decide(
        self, state: np.ndarray, training: bool = False
    ) -> Tuple[np.ndarray, Dict]:
        """
        포트폴리오 의사결정

        Args:
            state: [market_features(12), crisis_level(1), prev_weights(n_assets)]
            training: 훈련 모드 여부

        Returns:
            weights: 포트폴리오 가중치
            info: 의사결정 상세 정보
        """
        # 상태 분해
        market_features = state[:FEATURE_DIM]
        crisis_level = state[FEATURE_DIM]
        prev_weights = state[FEATURE_DIM + 1 :]

        # T-Cell 다차원 위기 감지
        crisis_detection = self.tcell.detect_crisis(market_features)

        # 기존 호환성을 위한 단일 위기 수준 계산
        if isinstance(crisis_detection, dict):
            detected_crisis = crisis_detection["overall_crisis"]
            final_crisis_level = max(crisis_level, detected_crisis)
            crisis_info = crisis_detection  # 다차원 정보 보존
        else:
            # 하위 호환성 (단일 float 반환 시)
            final_crisis_level = max(crisis_level, crisis_detection)
            crisis_info = final_crisis_level

        # B-Cell 선택 (다차원 위기 정보 기반)
        selected_bcell_name = self._select_bcell(crisis_info)
        selected_bcell = self.bcells[selected_bcell_name]

        # Phase 3: 적응형 엔트로피 업데이트 (선택된 B-Cell에 T-Cell 신호 전달)
        if hasattr(selected_bcell, "update_adaptive_target_entropy"):
            # 시장 안정도 추정 (과거 위기 레벨의 역수)
            market_stability = 1.0 - final_crisis_level
            regime_info = selected_bcell.update_adaptive_target_entropy(
                crisis_level=final_crisis_level, market_stability=market_stability
            )

        # Memory 회상
        memory_guidance = self.memory.get_memory_guidance(
            market_features, final_crisis_level, k=MEMORY_K
        )

        # 포트폴리오 가중치 생성
        weights = selected_bcell.get_action(state, deterministic=not training)

        # Memory 가이던스 적용 (있는 경우)
        self.memory_stats["total_guidance_attempts"] += 1
        if (
            memory_guidance["has_guidance"]
            and memory_guidance["confidence"] > 0.8
            and memory_guidance["recommended_action"] is not None
        ):

            memory_weight = min(0.3, memory_guidance["confidence"] - 0.5)
            weights = (1 - memory_weight) * weights + memory_weight * memory_guidance[
                "recommended_action"
            ]
            weights = weights / weights.sum()  # 정규화

            # 통계 업데이트
            self.memory_stats["guidance_applied"] += 1
            self.memory_stats["confidence_sum"] += memory_guidance["confidence"]

            # 간헐적 로깅 (매 100회마다)
            if self.memory_stats["guidance_applied"] % 100 == 0:
                avg_confidence = (
                    self.memory_stats["confidence_sum"]
                    / self.memory_stats["guidance_applied"]
                )
                usage_rate = (
                    self.memory_stats["guidance_applied"]
                    / self.memory_stats["total_guidance_attempts"]
                )
                self.logger.debug(
                    f"Memory 가이던스 통계 (최근 100회): 사용률={usage_rate:.1%}, "
                    f"평균 신뢰도={avg_confidence:.3f}"
                )

        # 의사결정 정보 수집 (XAI 강화)
        decision_info = {
            "crisis_level": float(final_crisis_level),
            "selected_bcell": selected_bcell_name,
            "memory_guidance": memory_guidance["has_guidance"],
            "memory_confidence": memory_guidance.get("confidence", 0.0),
            "specialization_scores": {
                name: bcell.get_specialization_score(crisis_info)
                for name, bcell in self.bcells.items()
            },
            "weights_concentration": float(np.sum(weights**2)),
            "decision_count": self.decision_count,
            # XAI 확장 데이터
            "xai_data": {
                "tcell_data": (
                    crisis_info
                    if isinstance(crisis_info, dict)
                    else {
                        "overall_crisis": crisis_info,
                        "volatility_crisis": 0.0,
                        "correlation_crisis": 0.0,
                        "volume_crisis": 0.0,
                    }
                ),
                "bcell_data": {
                    "strategy_scores": {
                        name: bcell.get_specialization_score(crisis_info)
                        for name, bcell in self.bcells.items()
                    },
                    "selected_strategy": selected_bcell_name,
                    "selection_reason": self._get_selection_reason(
                        selected_bcell_name, crisis_info
                    ),
                },
                "memory_data": {
                    "has_guidance": memory_guidance["has_guidance"],
                    "confidence": memory_guidance.get("confidence", 0.0),
                    "similar_episodes": memory_guidance.get("similar_episodes", []),
                    "similarity_scores": memory_guidance.get("similarity_scores", []),
                },
                "portfolio_data": {
                    "weights": (
                        {self.symbols[i]: float(w) for i, w in enumerate(weights)}
                        if hasattr(self, "symbols")
                        else {}
                    ),
                    "top_holdings": self._get_top_holdings(weights, 5),
                    "concentration": float(np.sum(weights**2)),
                    "diversification": float(
                        1.0 / np.sum(weights**2)
                    ),  # 역집중도 = 다양성
                },
                "step": self.decision_count,
            },
        }

        # 의사결정 히스토리 저장
        self.decision_history.append(
            {
                "step": self.decision_count,
                "crisis_level": final_crisis_level,
                "selected_bcell": selected_bcell_name,
                "weights": weights.copy(),
                "memory_used": memory_guidance["has_guidance"],
            }
        )

        self.decision_count += 1

        # 로깅 (주기적으로만)
        if self.decision_count % 50 == 0:
            self.logger.debug(
                f"의사결정 #{self.decision_count}: "
                f"위기수준={final_crisis_level:.3f}, "
                f"선택전략={selected_bcell_name}, "
                f"메모리={memory_guidance['has_guidance']}"
            )

        return weights, decision_info

    def _select_bcell(self, crisis_info) -> str:
        """
        다차원 위기 정보에 따른 B-Cell 선택 (다양성 확보)

        성능 기반 동적 페널티 + 확률적 선택을 통한 편향성 해결
        """
        # 각 B-Cell의 기본 전문성 점수 계산 (다차원 위기 정보 사용)
        scores = {}
        for name, bcell in self.bcells.items():
            base_score = bcell.get_specialization_score(crisis_info)

            # 최근 성과 기반 페널티 계산
            performance_data = self.bcell_performance[name]
            recent_rewards = performance_data["recent_rewards"]

            if len(recent_rewards) > 5:
                avg_reward = np.mean(recent_rewards)
                # 성과가 나쁠수록 페널티 적용
                performance_penalty = max(0, -avg_reward * 0.5)
                base_score -= performance_penalty

            # 연속 선택 페널티 (강제 순환)
            consecutive_count = performance_data["consecutive_selections"]
            if consecutive_count >= 5:  # 5회 연속 선택시
                base_score *= 0.3  # 점수 대폭 감소
                self.logging_stats["penalty_applications"] += 1

            scores[name] = base_score

        # 확률적 선택 (완전 greedy 방지)
        if np.random.random() < 0.2:  # 20% 확률로 랜덤 선택
            selected = np.random.choice(list(self.bcells.keys()))
            self.logging_stats["random_selections"] += 1
        else:
            # 최고 점수 전략 선택
            selected = max(scores, key=scores.get)

        # 선택 통계 업데이트
        self._update_selection_stats(selected)

        # 주기적 로깅 통계 보고
        decisions_since_last_report = (
            self.decision_count - self.logging_stats["last_log_report"]
        )
        if decisions_since_last_report >= self.logging_stats["log_interval"]:
            self._log_selection_statistics()
            self.logging_stats["last_log_report"] = self.decision_count

        return selected

    def _update_selection_stats(self, selected_bcell: str) -> None:
        """B-Cell 선택 통계 업데이트"""
        # 선택된 B-Cell 통계 업데이트
        self.bcell_performance[selected_bcell]["total_selections"] += 1

        # 연속 선택 카운트 업데이트
        if (
            self.decision_history
            and self.decision_history[-1]["selected_bcell"] == selected_bcell
        ):
            self.bcell_performance[selected_bcell]["consecutive_selections"] += 1
        else:
            self.bcell_performance[selected_bcell]["consecutive_selections"] = 1

        # 다른 B-Cell들의 연속 선택 카운트 리셋
        for name, data in self.bcell_performance.items():
            if name != selected_bcell:
                data["consecutive_selections"] = 0

    def _get_recent_rewards(self, bcell_name: str, window: int = 10) -> list:
        """특정 B-Cell의 최근 성과 반환"""
        return self.bcell_performance[bcell_name]["recent_rewards"][-window:]

    def _get_consecutive_count(self, bcell_name: str) -> int:
        """특정 B-Cell의 연속 선택 횟수 반환"""
        return self.bcell_performance[bcell_name]["consecutive_selections"]

    def _log_selection_statistics(self) -> None:
        """B-Cell 선택 통계 요약 로깅 (장황한 로그 대신)"""
        interval = self.logging_stats["log_interval"]
        random_count = self.logging_stats["random_selections"]
        penalty_count = self.logging_stats["penalty_applications"]

        # 현재 연속 선택 상황 체크
        consecutive_issues = []
        for name, data in self.bcell_performance.items():
            consecutive = data["consecutive_selections"]
            if consecutive >= 3:  # 3회 이상 연속시 보고
                consecutive_issues.append(f"{name}:{consecutive}회")

        # 통계 요약 로깅
        self.logger.debug(
            f"B-Cell 선택 통계 (최근 {interval}회): "
            f"탐험적 선택 {random_count}회 ({random_count/interval:.1%}), "
            f"페널티 적용 {penalty_count}회, "
            f"연속선택: [{', '.join(consecutive_issues) if consecutive_issues else '정상'}]"
        )

        # 통계 리셋
        self.logging_stats["random_selections"] = 0
        self.logging_stats["penalty_applications"] = 0

    def update(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done,
    ) -> None:
        """
        시스템 업데이트 (학습)

        Args:
            state: 현재 상태
            action: 선택된 행동 (포트폴리오 가중치)
            reward: 받은 보상
            next_state: 다음 상태
            done: 에피소드 종료 여부
        """
        # NumPy 타입을 Python native 타입으로 안전하게 변환
        done = bool(done)  # numpy.bool을 Python bool로 변환
        reward = float(reward)  # numpy.float을 Python float로 변환

        # 상태 분해
        market_features = state[:FEATURE_DIM]
        crisis_level = float(state[FEATURE_DIM])  # numpy scalar을 Python float로 변환

        # Memory에 경험 저장
        self.memory.store(
            state=market_features,
            action=action,
            reward=reward,
            crisis_level=crisis_level,
            additional_info={
                "step": int(self.training_steps),  # numpy int를 Python int로 변환
                "done": done,
            },
        )

        # B-Cell 업데이트 (모든 전문가 학습)
        for bcell in self.bcells.values():
            bcell.store_experience(state, action, reward, next_state, done)

            # 주기적 학습
            if self.training_steps % UPDATE_FREQUENCY == 0:
                bcell.update()

        # 성과 히스토리 업데이트
        self.performance_history.append(
            {
                "step": self.training_steps,
                "reward": reward,
                "crisis_level": crisis_level,
            }
        )

        # 마지막 선택된 B-Cell의 성과 추적 업데이트
        if self.decision_history:
            last_decision = self.decision_history[-1]
            selected_bcell = last_decision["selected_bcell"]

            # 최근 성과 리스트에 추가 (최대 10개 유지)
            recent_rewards = self.bcell_performance[selected_bcell]["recent_rewards"]
            recent_rewards.append(reward)
            if len(recent_rewards) > 10:
                recent_rewards.pop(0)  # 가장 오래된 것 제거

        self.training_steps += 1

        # 주기적 로깅
        if self.training_steps % (LOG_INTERVAL * 10) == 0:
            self.logger.info(
                f"시스템 업데이트: 스텝={self.training_steps}, "
                f"보상={reward:.4f}, 메모리={len(self.memory.memories)}개"
            )

    def get_system_explanation(self, state: np.ndarray) -> Dict:
        """
        현재 의사결정에 대한 종합 설명 (XAI)
        """
        market_features = state[:FEATURE_DIM]
        crisis_level = state[FEATURE_DIM]

        # T-Cell 다차원 위기 감지
        crisis_detection = self.tcell.detect_crisis(market_features)

        # 기존 호환성을 위한 처리
        if isinstance(crisis_detection, dict):
            crisis_info = crisis_detection
            overall_crisis = crisis_detection["overall_crisis"]
        else:
            crisis_info = crisis_level
            overall_crisis = crisis_level

        # T-Cell 설명
        tcell_explanation = self.tcell.get_anomaly_explanation(market_features)

        # 선택된 B-Cell 설명 (다차원 위기 정보 기반)
        selected_bcell_name = self._select_bcell(crisis_info)
        bcell_explanation = self.bcells[selected_bcell_name].get_explanation(state)

        # Memory 통계
        memory_stats = self.memory.get_statistics()

        # 전체 설명
        explanation = {
            "system_overview": {
                "training_steps": self.training_steps,
                "decision_count": self.decision_count,
                "tcell_fitted": self.tcell.is_fitted,
            },
            "crisis_detection": tcell_explanation,
            "multidimensional_crisis": (
                crisis_detection if isinstance(crisis_detection, dict) else None
            ),
            "strategy_selection": {
                "selected_strategy": selected_bcell_name,
                "selection_reason": f"다차원 위기 분석 결과에 최적화됨 (전체: {overall_crisis:.3f})",
                "all_specialization_scores": {
                    name: bcell.get_specialization_score(crisis_info)
                    for name, bcell in self.bcells.items()
                },
            },
            "portfolio_generation": bcell_explanation,
            "memory_system": memory_stats,
            "recent_decisions": (
                self.decision_history[-5:] if self.decision_history else []
            ),
        }

        return explanation

    def get_performance_summary(self) -> Dict:
        """성과 요약 통계"""
        if not self.performance_history:
            return {"message": "성과 데이터가 없습니다."}

        rewards = [p["reward"] for p in self.performance_history]
        crisis_levels = [p["crisis_level"] for p in self.performance_history]

        # B-Cell 사용 통계
        bcell_usage = {}
        for decision in self.decision_history:
            bcell = decision["selected_bcell"]
            bcell_usage[bcell] = bcell_usage.get(bcell, 0) + 1

        # B-Cell 다양성 통계
        bcell_diversity_stats = {}
        total_selections = sum(
            self.bcell_performance[name]["total_selections"]
            for name in self.bcells.keys()
        )

        for name, data in self.bcell_performance.items():
            recent_rewards = data["recent_rewards"]
            selection_rate = data["total_selections"] / max(total_selections, 1)

            bcell_diversity_stats[name] = {
                "total_selections": data["total_selections"],
                "selection_rate": selection_rate,
                "consecutive_selections": data["consecutive_selections"],
                "avg_recent_reward": np.mean(recent_rewards) if recent_rewards else 0.0,
                "recent_performance_count": len(recent_rewards),
            }

<<<<<<< HEAD
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

            # Division by zero 방지
            total_weight = np.sum(ensemble_strategy)
            if total_weight > WEIGHT_NORMALIZATION_MIN:
                ensemble_strategy = ensemble_strategy / total_weight
            else:
                # 모든 가중치가 0에 가까울 경우, 균등 분배로 폴백
                ensemble_strategy = np.ones(self.n_assets) / self.n_assets
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
=======
        # 다양성 지수 계산 (Shannon Entropy)
        diversity_index = 0.0
        if total_selections > 0:
            for name in self.bcells.keys():
                selection_rate = (
                    self.bcell_performance[name]["total_selections"] / total_selections
>>>>>>> origin/dev
                )
                if selection_rate > 0:
                    diversity_index -= selection_rate * np.log(selection_rate)

<<<<<<< HEAD
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
        risk_reduction = activation_level * RISK_REDUCTION_FACTOR
        weights = self.base_weights * (1 - risk_reduction)
        safe_indices = [6, 7, 8]
        for idx in safe_indices:
            if idx < len(weights):
                weights[idx] += risk_reduction / len(safe_indices)
        # Division by zero 방지
        total_weight = np.sum(weights)
        if total_weight > WEIGHT_NORMALIZATION_MIN:
            return weights / total_weight
        else:
            return np.ones(len(weights)) / len(weights)

    def _correlation_response(self, activation_level):
        """상관관계 위험 대응"""
        diversification_boost = activation_level * DIVERSIFICATION_FACTOR
        weights = self.base_weights.copy()
        weights = weights * (1 - diversification_boost) + diversification_boost / len(
            weights
        )
        # Division by zero 방지
        total_weight = np.sum(weights)
        if total_weight > WEIGHT_NORMALIZATION_MIN:
            return weights / total_weight
        else:
            return np.ones(len(weights)) / len(weights)

    def _momentum_response(self, activation_level):
        """모멘텀 위험 대응"""
        neutral_adjustment = activation_level * NEUTRAL_ADJUSTMENT_FACTOR
        weights = self.base_weights * (1 - neutral_adjustment) + (
            self.base_weights * neutral_adjustment
        )
        # Division by zero 방지
        total_weight = np.sum(weights)
        if total_weight > WEIGHT_NORMALIZATION_MIN:
            return weights / total_weight
        else:
            return np.ones(len(weights)) / len(weights)

    def _liquidity_response(self, activation_level):
        """유동성 위험 대응"""
        large_cap_boost = activation_level * LARGE_CAP_FACTOR
        weights = self.base_weights.copy()
        large_cap_indices = [0, 1, 2, 3]
        for idx in large_cap_indices:
            if idx < len(weights):
                weights[idx] += large_cap_boost / len(large_cap_indices)
        # Division by zero 방지
        total_weight = np.sum(weights)
        if total_weight > WEIGHT_NORMALIZATION_MIN:
            return weights / total_weight
        else:
            return np.ones(len(weights)) / len(weights)

    def _macro_response(self, activation_level):
        """거시경제 위험 대응"""
        defensive_boost = activation_level * DEFENSIVE_FACTOR
        weights = self.base_weights * (1 - defensive_boost)
        defensive_indices = [7, 8, 9]
        for idx in defensive_indices:
            if idx < len(weights):
                weights[idx] += defensive_boost / len(defensive_indices)
        # Division by zero 방지
        total_weight = np.sum(weights)
        if total_weight > WEIGHT_NORMALIZATION_MIN:
            return weights / total_weight
        else:
            return np.ones(len(weights)) / len(weights)

    def pretrain_bcells(self, market_data, episodes=PRETRAIN_EPISODES):
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
=======
        summary = {
            "training_steps": self.training_steps,
            "avg_reward": np.mean(rewards),
            "reward_std": np.std(rewards),
            "total_decisions": self.decision_count,
            "avg_crisis_level": np.mean(crisis_levels),
            "crisis_std": np.std(crisis_levels),
            "bcell_usage": bcell_usage,
            "bcell_diversity_stats": bcell_diversity_stats,
            "bcell_diversity_index": diversity_index,
            "memory_size": len(self.memory.memories),
            "recent_performance": (
                np.mean(rewards[-50:]) if len(rewards) >= 50 else np.mean(rewards)
            ),
>>>>>>> origin/dev
        }

        return summary

    def save_system(self, base_path: str) -> bool:
        """전체 시스템을 단일 파일로 저장"""
        try:
            # 저장 디렉토리 생성 보장
            base_dir = os.path.dirname(base_path)
            if base_dir:
                os.makedirs(base_dir, exist_ok=True)

            # 전체 시스템을 하나의 딕셔너리로 통합
            system_data = {
                "tcell": {
                    "detector": self.tcell.detector if self.tcell.is_fitted else None,
                    "scaler": self.tcell.scaler if self.tcell.is_fitted else None,
                    "contamination": self.tcell.contamination,
                    "sensitivity": self.tcell.sensitivity,
                    "is_fitted": self.tcell.is_fitted,
                },
                "bcells": {},
                "memory": {
                    "memories": list(self.memory.memories),
                    "embeddings": list(self.memory.embeddings),
                    "capacity": self.memory.capacity,
                    "embedding_dim": self.memory.embedding_dim,
                    "similarity_threshold": self.memory.similarity_threshold,
                },
                "system_config": {
                    "n_assets": self.n_assets,
                    "state_dim": self.state_dim,
                    "training_steps": self.training_steps,
                    "decision_count": self.decision_count,
                },
            }

            # B-Cell 데이터 추가 (SAC 버전)
            for name, bcell in self.bcells.items():
                system_data["bcells"][name] = {
                    "actor_state_dict": bcell.actor.state_dict(),
                    "critic1_state_dict": bcell.critic1.state_dict(),
                    "critic2_state_dict": bcell.critic2.state_dict(),
                    "target_critic1_state_dict": bcell.target_critic1.state_dict(),
                    "target_critic2_state_dict": bcell.target_critic2.state_dict(),
                    "log_alpha": bcell.log_alpha,
                    "target_entropy": bcell.target_entropy,
                    "risk_type": bcell.risk_type,
                    "update_count": bcell.update_count,
                }

            # 단일 파일로 저장
            import pickle

            with open(f"{base_path}.pkl", "wb") as f:
                pickle.dump(system_data, f)

            self.logger.info(f"면역 시스템이 저장되었습니다: {base_path}.pkl")
            return True

        except Exception as e:
            self.logger.error(f"시스템 저장 실패: {e}")
            return False

    def load_system(self, base_path: str) -> bool:
        """전체 시스템 로드"""
        try:
            # T-Cell 로드
            tcell_path = f"{base_path}_tcell.pkl"
            self.tcell.load_model(tcell_path)

            # B-Cell 로드
            for name, bcell in self.bcells.items():
                bcell_path = f"{base_path}_bcell_{name}.pth"
                bcell.load_model(bcell_path)

            # Memory 로드
            memory_path = f"{base_path}_memory.pkl"
            self.memory.load_memory(memory_path)

            self.is_trained = True
            self.logger.info(f"면역 시스템이 로드되었습니다: {base_path}")
            return True

        except Exception as e:
            self.logger.error(f"시스템 로드 실패: {e}")
            return False

    def _get_selection_reason(self, selected_bcell: str, crisis_info) -> str:
        """B-Cell 선택 이유 설명 생성"""

        if isinstance(crisis_info, dict):
            crisis_level = crisis_info.get("overall_crisis", 0)
            crisis_types = []

            if crisis_info.get("volatility_crisis", 0) > 0.5:
                crisis_types.append("high volatility")
            if crisis_info.get("correlation_crisis", 0) > 0.5:
                crisis_types.append("correlation shift")
            if crisis_info.get("volume_crisis", 0) > 0.5:
                crisis_types.append("volume anomaly")
        else:
            crisis_level = crisis_info
            crisis_types = []

        if crisis_level > 0.7:
            crisis_desc = "High crisis detected"
        elif crisis_level > 0.4:
            crisis_desc = "Moderate crisis detected"
        else:
            crisis_desc = "Normal market conditions"

        type_desc = f" ({', '.join(crisis_types)})" if crisis_types else ""

        strategy_reasons = {
            "volatility": f"High volatility strategy selected for unstable market{type_desc}",
            "correlation": f"Correlation strategy selected for regime change{type_desc}",
            "momentum": f"Momentum strategy selected for trending market{type_desc}",
            "defensive": f"Defensive strategy selected for risk management{type_desc}",
            "growth": f"Growth strategy selected for opportunity capture{type_desc}",
        }

        return strategy_reasons.get(selected_bcell, f"{crisis_desc}{type_desc}")

    def _get_top_holdings(self, weights: np.ndarray, top_n: int = 5) -> List[str]:
        """상위 보유 종목 리스트 반환"""

        if hasattr(self, "symbols") and len(self.symbols) == len(weights):
            # 가중치가 높은 순으로 정렬
            sorted_indices = np.argsort(weights)[::-1]
            return [self.symbols[i] for i in sorted_indices[:top_n]]
        else:
            # 심볼이 없으면 인덱스로 표시
            sorted_indices = np.argsort(weights)[::-1]
            return [f"Asset_{i}" for i in sorted_indices[:top_n]]
