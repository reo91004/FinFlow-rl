# agents/tcell.py

import numpy as np
from sklearn.ensemble import IsolationForest
from collections import deque
from datetime import datetime
from .base import ImmuneCell
from constant import *


class TCell(ImmuneCell):
    """T-세포: 위험 탐지"""

    def __init__(self, cell_id, sensitivity=DEFAULT_SENSITIVITY, random_state=None):
        super().__init__(cell_id)
        self.sensitivity = sensitivity
        self.detector = IsolationForest(
            contamination=sensitivity, random_state=random_state
        )
        self.is_trained = False
        self.training_data = deque(maxlen=TCELL_TRAINING_BUFFER_SIZE)  # 훈련 데이터 저장
        self.historical_scores = deque(maxlen=TCELL_PATTERN_BUFFER_SIZE)
        self.market_state_history = deque(maxlen=TCELL_REWARD_BUFFER_SIZE)  # 시장 상태 히스토리

    def detect_anomaly(self, market_features):
        """시장 이상 상황 탐지"""
        # 입력 특성 크기 확인 및 조정
        if len(market_features.shape) == 1:
            market_features = market_features.reshape(1, -1)

        # 훈련 데이터 축적
        self.training_data.append(market_features[0])

        if not self.is_trained:
            if len(self.training_data) >= TCELL_TRAINING_BUFFER_SIZE:  # 충분한 데이터가 쌓인 후 훈련
                training_matrix = np.array(list(self.training_data))
                self.detector.fit(training_matrix)
                self.is_trained = True
                self.expected_features = training_matrix.shape[1]
                print(
                    f"[정보] T-cell {self.cell_id} 훈련 완료 (데이터: {len(self.training_data)}개)"
                )
            return 0.0

        # 특성 크기 확인
        if market_features.shape[1] != self.expected_features:
            print(
                f"[경고] T-cell 특성 크기 불일치: 기대 {self.expected_features}, 실제 {market_features.shape[1]}"
            )
            min_features = min(self.expected_features, market_features.shape[1])
            market_features = market_features[:, :min_features]

            if market_features.shape[1] < self.expected_features:
                padding = np.zeros(
                    (
                        market_features.shape[0],
                        self.expected_features - market_features.shape[1],
                    )
                )
                market_features = np.hstack([market_features, padding])

        # 이상 점수 계산
        anomaly_scores = self.detector.decision_function(market_features)
        current_score = np.mean(anomaly_scores)

        # 시장 상태 분석
        market_state = self._analyze_market_state(market_features[0])
        self.market_state_history.append(market_state)

        # 히스토리 업데이트
        self.historical_scores.append(current_score)

        # 위기 감지 로직 및 상세 분석
        crisis_detection = self._detailed_crisis_analysis(current_score, market_state)
        self.activation_level = crisis_detection["activation_level"]

        # 위기 감지 로그 저장 (활성화 레벨이 0.15 이상일 때)
        if self.activation_level > 0.15:
            self.last_crisis_detection = crisis_detection

        return self.activation_level

    def _detailed_crisis_analysis(self, current_score, market_state):
        """위기 감지 상세 분석"""
        crisis_info = {
            "tcell_id": self.cell_id,
            "timestamp": datetime.now().isoformat(),
            "raw_anomaly_score": current_score,
            "market_state": market_state,
            "activation_level": 0.0,
            "crisis_indicators": [],
            "feature_contributions": {},
            "decision_reasoning": [],
        }

        if len(self.historical_scores) >= 10:
            historical_mean = np.mean(self.historical_scores)
            historical_std = np.std(self.historical_scores)

            # Z-score 기반 이상 탐지
            z_score = (current_score - historical_mean) / (historical_std + 1e-8)

            # 시장 상태 기반 조정
            market_volatility = market_state["volatility"]
            market_stress = market_state["stress"]
            market_correlation = market_state["correlation"]

            # 기본 활성화 레벨 계산 및 근거 기록
            base_activation = 0.0
            if z_score < -1.5:
                base_activation = TCELL_ACTIVATION_THRESHOLDS[0]
                crisis_info["crisis_indicators"].append(
                    {
                        "type": "extreme_anomaly",
                        "value": z_score,
                        "threshold": -1.5,
                        "contribution": TCELL_ACTIVATION_THRESHOLDS[0],
                        "description": f"매우 이상한 이상 점수 (Z-score: {z_score:.3f})",
                    }
                )
                crisis_info["decision_reasoning"].append(
                    f"이상 점수가 과거 평균보다 {abs(z_score):.1f} 표준편차 낮음 (매우 이상)"
                )
            elif z_score < -1.0:
                base_activation = TCELL_ACTIVATION_THRESHOLDS[1]
                crisis_info["crisis_indicators"].append(
                    {
                        "type": "high_anomaly",
                        "value": z_score,
                        "threshold": -1.0,
                        "contribution": TCELL_ACTIVATION_THRESHOLDS[1],
                        "description": f"상당히 이상한 이상 점수 (Z-score: {z_score:.3f})",
                    }
                )
                crisis_info["decision_reasoning"].append(
                    f"이상 점수가 과거 평균보다 {abs(z_score):.1f} 표준편차 낮음 (상당히 이상)"
                )
            elif z_score < -0.5:
                base_activation = TCELL_ACTIVATION_THRESHOLDS[2]
                crisis_info["crisis_indicators"].append(
                    {
                        "type": "moderate_anomaly",
                        "value": z_score,
                        "threshold": -0.5,
                        "contribution": TCELL_ACTIVATION_THRESHOLDS[2],
                        "description": f"약간 이상한 이상 점수 (Z-score: {z_score:.3f})",
                    }
                )
                crisis_info["decision_reasoning"].append(
                    f"이상 점수가 과거 평균보다 {abs(z_score):.1f} 표준편차 낮음 (약간 이상)"
                )
            elif z_score < 0.0:
                base_activation = TCELL_ACTIVATION_THRESHOLDS[3]
                crisis_info["crisis_indicators"].append(
                    {
                        "type": "mild_anomaly",
                        "value": z_score,
                        "threshold": 0.0,
                        "contribution": TCELL_ACTIVATION_THRESHOLDS[3],
                        "description": f"주의 수준 이상 점수 (Z-score: {z_score:.3f})",
                    }
                )
                crisis_info["decision_reasoning"].append(
                    f"이상 점수가 과거 평균보다 낮음 (주의 필요)"
                )

            # 시장 상태 기반 조정 및 근거 기록
            volatility_boost = 0.0
            if market_volatility > 0.3:
                volatility_boost = RISK_BOOST_FACTORS["volatility"]
                base_activation += volatility_boost
                crisis_info["crisis_indicators"].append(
                    {
                        "type": "high_volatility",
                        "value": market_volatility,
                        "threshold": 0.3,
                        "contribution": volatility_boost,
                        "description": f"높은 시장 변동성 ({market_volatility:.3f})",
                    }
                )
                crisis_info["decision_reasoning"].append(
                    f"시장 변동성이 임계값 0.3을 초과함 ({market_volatility:.3f})"
                )

            stress_boost = 0.0
            if market_stress > 0.5:
                stress_boost = RISK_BOOST_FACTORS["stress"]
                base_activation += stress_boost
                crisis_info["crisis_indicators"].append(
                    {
                        "type": "high_stress",
                        "value": market_stress,
                        "threshold": 0.5,
                        "contribution": stress_boost,
                        "description": f"높은 시장 스트레스 ({market_stress:.3f})",
                    }
                )
                crisis_info["decision_reasoning"].append(
                    f"시장 스트레스 지수가 임계값 0.5를 초과함 ({market_stress:.3f})"
                )

            # 상관관계 위험 분석
            corr_boost = 0.0
            if market_correlation > 0.8:
                corr_boost = RISK_BOOST_FACTORS["correlation"]
                base_activation += corr_boost
                crisis_info["crisis_indicators"].append(
                    {
                        "type": "high_correlation",
                        "value": market_correlation,
                        "threshold": 0.8,
                        "contribution": corr_boost,
                        "description": f"높은 시장 상관관계 ({market_correlation:.3f})",
                    }
                )
                crisis_info["decision_reasoning"].append(
                    f"시장 상관관계가 과도하게 높음 ({market_correlation:.3f}) - 시스템적 위험"
                )

            # 최근 시장 상태 변화 고려
            trend_boost = 0.0
            if len(self.market_state_history) >= 5:
                recent_volatility_change = np.mean(
                    [s["volatility"] for s in list(self.market_state_history)[-5:]]
                )
                if recent_volatility_change > 0.4:
                    trend_boost = 0.1
                    base_activation += trend_boost
                    crisis_info["crisis_indicators"].append(
                        {
                            "type": "volatility_trend",
                            "value": recent_volatility_change,
                            "threshold": 0.4,
                            "contribution": trend_boost,
                            "description": f"지속적인 높은 변동성 ({recent_volatility_change:.3f})",
                        }
                    )
                    crisis_info["decision_reasoning"].append(
                        f"최근 5일 평균 변동성이 지속적으로 높음 ({recent_volatility_change:.3f})"
                    )

            # 특성별 기여도 분석
            crisis_info["feature_contributions"] = {
                "z_score_base": base_activation
                - volatility_boost
                - stress_boost
                - trend_boost
                - corr_boost,
                "volatility_boost": volatility_boost,
                "stress_boost": stress_boost,
                "correlation_boost": corr_boost,
                "trend_boost": trend_boost,
                "total_score": base_activation,
            }

            crisis_info["activation_level"] = np.clip(base_activation, 0.0, 1.0)

            # 위기 수준 분류
            if crisis_info["activation_level"] > 0.7:
                crisis_info["crisis_level"] = "severe"
            elif crisis_info["activation_level"] > 0.5:
                crisis_info["crisis_level"] = "high"
            elif crisis_info["activation_level"] > 0.3:
                crisis_info["crisis_level"] = "moderate"
            elif crisis_info["activation_level"] > 0.15:
                crisis_info["crisis_level"] = "mild"
            else:
                crisis_info["crisis_level"] = "normal"

        else:
            # 초기 학습 기간
            raw_score = max(0, min(1, (1 - current_score) / 1.5))
            crisis_info["activation_level"] = raw_score * 0.5
            crisis_info["crisis_level"] = "learning"
            crisis_info["decision_reasoning"].append("초기 학습 기간 - 보수적 설정")

        return crisis_info

    def _analyze_market_state(self, features):
        """시장 상태 분석"""
        # 특성 기반 시장 상태 분석
        volatility = features[0] if len(features) > 0 else 0.0
        correlation = features[1] if len(features) > 1 else 0.0
        returns = features[2] if len(features) > 2 else 0.0

        # 스트레스 지수 계산
        stress_indicators = []
        if len(features) > 4:
            stress_indicators.append(abs(features[4]))  # 왜도
        if len(features) > 5:
            stress_indicators.append(abs(features[5]))  # 첨도
        if len(features) > 6:
            stress_indicators.append(features[6])  # 하락일 비율

        stress_level = np.mean(stress_indicators) if stress_indicators else 0.0

        return {
            "volatility": abs(volatility),
            "correlation": abs(correlation),
            "returns": returns,
            "stress": stress_level,
        }
