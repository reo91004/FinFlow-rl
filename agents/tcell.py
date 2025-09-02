# bipd/agents/tcell.py

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pickle
import os
from collections import deque
<<<<<<< HEAD
from datetime import datetime
from .base import ImmuneCell


class TCell(ImmuneCell):
    """T-세포: 위험 탐지"""

    def __init__(self, cell_id, sensitivity=0.1, random_state=None):
        super().__init__(cell_id)
        self.sensitivity = sensitivity
        self.detector = IsolationForest(
            contamination=sensitivity, random_state=random_state
        )
        self.is_trained = False
        self.training_data = deque(maxlen=200)  # 훈련 데이터 저장
        self.historical_scores = deque(maxlen=100)
        self.market_state_history = deque(maxlen=50)  # 시장 상태 히스토리

    def detect_anomaly(self, market_features):
        """시장 이상 상황 탐지"""
        # 입력 특성 크기 확인 및 조정
        if len(market_features.shape) == 1:
            market_features = market_features.reshape(1, -1)

        # 훈련 데이터 축적
        self.training_data.append(market_features[0])

        if not self.is_trained:
            if len(self.training_data) >= 200:  # 충분한 데이터가 쌓인 후 훈련
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
        raw_activation_level = crisis_detection["activation_level"]
        
        # Activation Threshold 적용 - 임계값 이상일 때만 활성화
        if raw_activation_level >= self.activation_threshold:
            self.activation_level = raw_activation_level
            
            # 활성화 로그 저장
            self.last_crisis_detection = crisis_detection
            self.last_crisis_detection["threshold_triggered"] = True
        else:
            # 임계값 미달 시 비활성화
            self.activation_level = 0.0
            self.last_crisis_detection = {
                "tcell_id": self.cell_id,
                "timestamp": datetime.now().isoformat(),
                "activation_level": 0.0,
                "raw_activation_level": raw_activation_level,
                "threshold_triggered": False,
                "threshold_value": self.activation_threshold,
                "reason": f"Activation level {raw_activation_level:.3f} below threshold {self.activation_threshold:.3f}"
            }

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
                base_activation = 0.8
                crisis_info["crisis_indicators"].append(
                    {
                        "type": "extreme_anomaly",
                        "value": z_score,
                        "threshold": -1.5,
                        "contribution": 0.8,
                        "description": f"매우 이상한 이상 점수 (Z-score: {z_score:.3f})",
                    }
                )
                crisis_info["decision_reasoning"].append(
                    f"이상 점수가 과거 평균보다 {abs(z_score):.1f} 표준편차 낮음 (매우 이상)"
                )
            elif z_score < -1.0:
                base_activation = 0.6
                crisis_info["crisis_indicators"].append(
                    {
                        "type": "high_anomaly",
                        "value": z_score,
                        "threshold": -1.0,
                        "contribution": 0.6,
                        "description": f"상당히 이상한 이상 점수 (Z-score: {z_score:.3f})",
                    }
                )
                crisis_info["decision_reasoning"].append(
                    f"이상 점수가 과거 평균보다 {abs(z_score):.1f} 표준편차 낮음 (상당히 이상)"
                )
            elif z_score < -0.5:
                base_activation = 0.4
                crisis_info["crisis_indicators"].append(
                    {
                        "type": "moderate_anomaly",
                        "value": z_score,
                        "threshold": -0.5,
                        "contribution": 0.4,
                        "description": f"약간 이상한 이상 점수 (Z-score: {z_score:.3f})",
                    }
                )
                crisis_info["decision_reasoning"].append(
                    f"이상 점수가 과거 평균보다 {abs(z_score):.1f} 표준편차 낮음 (약간 이상)"
                )
            elif z_score < 0.0:
                base_activation = 0.2
                crisis_info["crisis_indicators"].append(
                    {
                        "type": "mild_anomaly",
                        "value": z_score,
                        "threshold": 0.0,
                        "contribution": 0.2,
                        "description": f"주의 수준 이상 점수 (Z-score: {z_score:.3f})",
                    }
                )
                crisis_info["decision_reasoning"].append(
                    f"이상 점수가 과거 평균보다 낮음 (주의 필요)"
                )

            # 시장 상태 기반 조정 및 근거 기록
            volatility_boost = 0.0
            if market_volatility > 0.3:
                volatility_boost = 0.2
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
                stress_boost = 0.15
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
                corr_boost = 0.1
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
=======
from typing import Dict
from utils.logger import BIPDLogger
from config import (
    THRESHOLD_WINDOW_SIZE, VOLATILITY_CRISIS_QUANTILE, CORRELATION_CRISIS_QUANTILE,
    VOLUME_CRISIS_QUANTILE, OVERALL_CRISIS_QUANTILE,
    VOLATILITY_CRISIS_RATE, CORRELATION_CRISIS_RATE, VOLUME_CRISIS_RATE, OVERALL_CRISIS_RATE
)
>>>>>>> origin/dev

class AdaptiveThresholdDetector:
    """분위수 기반 적응형 임계값 감지기"""
    
    def __init__(self, window_size=THRESHOLD_WINDOW_SIZE, target_quantile=0.975, target_crisis_rate=0.15):
        self.window_size = window_size
        self.target_quantile = target_quantile
        self.target_crisis_rate = target_crisis_rate  # 목표 위기율 (15%)
        self.buffer = deque(maxlen=window_size)
        self.crisis_history = deque(maxlen=100)  # 최근 100회 위기 판정 기록
        
    def update_and_detect(self, metric_value):
        """메트릭 값 업데이트 및 위기 감지"""
        self.buffer.append(float(metric_value))
        
        # 충분한 데이터가 없으면 기본값 반환
        if len(self.buffer) < 50:
            is_crisis = metric_value > 0.7  # 기본 임계값
            threshold = 0.7
        else:
            # 분위수 기반 임계값 계산
            current_quantile = self.target_quantile
            
            # 위기율 피드백을 통한 임계값 조정
            if len(self.crisis_history) >= 50:
                recent_crisis_rate = sum(self.crisis_history) / len(self.crisis_history)
                
                # 위기율이 너무 높으면 임계값 상향 조정
                if recent_crisis_rate > self.target_crisis_rate + 0.1:
                    current_quantile = min(0.99, current_quantile + 0.01)
                # 위기율이 너무 낮으면 임계값 하향 조정
                elif recent_crisis_rate < self.target_crisis_rate - 0.1:
                    current_quantile = max(0.8, current_quantile - 0.01)
            
            threshold = np.quantile(list(self.buffer), current_quantile)
            is_crisis = metric_value >= threshold
        
        # 위기 히스토리 기록
        self.crisis_history.append(is_crisis)
        
        return is_crisis, threshold
    
    def get_crisis_rate(self):
        """최근 위기율 반환"""
        if not self.crisis_history:
            return 0.0
        return sum(self.crisis_history) / len(self.crisis_history)

class TCell:
    """
    T-세포: 시장 이상 감지를 통한 위기 수준 측정
    
    Isolation Forest를 사용하여 정상 시장 패턴을 학습하고
    현재 시장 상황의 이상도를 기반으로 위기 수준을 산출
    """
    
    def __init__(self, contamination=0.1, sensitivity=1.0, random_state=42):
        self.contamination = contamination
        self.sensitivity = sensitivity
        self.random_state = random_state
        
        # Isolation Forest 모델
        self.detector = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        
        # 특성 정규화
        self.scaler = StandardScaler()
        
        # 학습 상태
        self.is_fitted = False
        self.training_features = []
        
        # 적응형 임계값 감지기들 (config.py에서 관리)
        self.volatility_detector = AdaptiveThresholdDetector(
            window_size=THRESHOLD_WINDOW_SIZE, 
            target_quantile=VOLATILITY_CRISIS_QUANTILE, 
            target_crisis_rate=VOLATILITY_CRISIS_RATE
        )
        self.correlation_detector = AdaptiveThresholdDetector(
            window_size=THRESHOLD_WINDOW_SIZE, 
            target_quantile=CORRELATION_CRISIS_QUANTILE, 
            target_crisis_rate=CORRELATION_CRISIS_RATE
        )
        self.volume_detector = AdaptiveThresholdDetector(
            window_size=THRESHOLD_WINDOW_SIZE, 
            target_quantile=VOLUME_CRISIS_QUANTILE, 
            target_crisis_rate=VOLUME_CRISIS_RATE
        )
        self.overall_detector = AdaptiveThresholdDetector(
            window_size=THRESHOLD_WINDOW_SIZE, 
            target_quantile=OVERALL_CRISIS_QUANTILE, 
            target_crisis_rate=OVERALL_CRISIS_RATE
        )
        
        # 로거
        self.logger = BIPDLogger("TCell")
        
        self.logger.info(f"T-Cell이 초기화되었습니다. 오염율={contamination}, 민감도={sensitivity}")
    
    def fit(self, historical_features):
        """
        정상 시장 패턴 학습
        
        Args:
            historical_features: np.array of shape (n_samples, feature_dim)
        """
        if len(historical_features) < 50:
            self.logger.warning(f"학습 데이터가 부족합니다: {len(historical_features)}개")
            return False
        
        try:
            # 특성 정규화
            normalized_features = self.scaler.fit_transform(historical_features)
            
            # Isolation Forest 학습
            self.detector.fit(normalized_features)
            
            self.is_fitted = True
            self.training_features = normalized_features.copy()
            
            # 학습 성능 검증
            train_scores = self.detector.decision_function(normalized_features)
            normal_ratio = (train_scores > 0).mean()
            
            self.logger.info(
                f"T-Cell 학습 완료: "
                f"샘플수={len(historical_features)}, "
                f"정상비율={normal_ratio:.2%}, "
                f"적응형 임계값 감지기 활성화"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"T-Cell 학습 실패: {e}")
            return False
    
    def detect_crisis(self, features):
        """
        다차원 위기 감지 시스템
        
        Args:
            features: np.array of shape (feature_dim,) - 시장 특성
            
        Returns:
            crisis_vector: dict containing multiple crisis dimensions
        """
        if not self.is_fitted:
            self.logger.warning("T-Cell이 학습되지 않았습니다.")
            return {
                'overall_crisis': 0.0,
                'volatility_crisis': 0.0,
                'correlation_crisis': 0.0,
                'volume_crisis': 0.0,
                'crisis_vector': np.array([0.0, 0.0, 0.0])
            }
        
        try:
            # 특성 정규화
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # 전체 이상치 점수 계산 (기존 방식)
            anomaly_score = self.detector.decision_function(features_scaled)[0]
            
            # 적응형 임계값 기반 전체 위기 감지
            raw_crisis = 1 / (1 + np.exp(anomaly_score * self.sensitivity))
            is_overall_crisis, overall_threshold = self.overall_detector.update_and_detect(raw_crisis)
            
            if is_overall_crisis:
                overall_crisis = np.clip(raw_crisis, 0.0, 1.0)
            else:
                overall_crisis = np.clip(raw_crisis * 0.5, 0.0, 0.5)  # 위기 아니면 최대 0.5로 제한
            
            # 다차원 위기 분석
            volatility_crisis = self._detect_volatility_crisis(features)
            correlation_crisis = self._detect_correlation_crisis(features)
            volume_crisis = self._detect_volume_crisis(features)
            
            # 위기 벡터 생성
            crisis_vector = np.array([volatility_crisis, correlation_crisis, volume_crisis])
            
            return {
                'overall_crisis': float(overall_crisis),
                'volatility_crisis': float(volatility_crisis),
                'correlation_crisis': float(correlation_crisis),
                'volume_crisis': float(volume_crisis),
                'crisis_vector': crisis_vector
            }
            
        except Exception as e:
            self.logger.error(f"다차원 위기 감지 실패: {e}")
            return {
                'overall_crisis': 0.0,
                'volatility_crisis': 0.0,
                'correlation_crisis': 0.0,
                'volume_crisis': 0.0,
                'crisis_vector': np.array([0.0, 0.0, 0.0])
            }
    
    def _detect_volatility_crisis(self, features):
        """
        변동성 위기 감지 (적응형 임계값)
        
        특성 벡터에서 변동성 관련 지표들을 추출하여 위기 수준 계산
        """
        try:
            # features 구조 가정: [returns, volatilities, correlations, volumes, ...]
            # 변동성 관련 특성 추출 (인덱스는 FeatureExtractor 구조에 따라 조정 필요)
            if len(features) >= 4:
                volatility_indicators = features[1:4]  # 변동성 관련 특성들
                
                # 평균 변동성 계산
                avg_volatility = np.mean(volatility_indicators)
                
                # 적응형 임계값 감지
                is_crisis, threshold = self.volatility_detector.update_and_detect(avg_volatility)
                
                if is_crisis:
                    # 0과 threshold 사이의 비율로 위기 수준 계산
                    if threshold > 0:
                        crisis_level = min(1.0, avg_volatility / threshold)
                        return np.tanh(crisis_level) # 부드러운 스케일링
                    else:
                        return 0.5  # 기본값
                else:
                    return 0.0
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"변동성 위기 감지 실패: {e}")
            return 0.0
    
    def _detect_correlation_crisis(self, features):
        """
        상관관계 위기 감지 (적응형 임계값)
        
        자산 간 상관관계 변화를 통한 위기 감지
        """
        try:
            # 상관관계 관련 특성 추출
            if len(features) >= 8:
                correlation_indicators = features[4:8]  # 상관관계 관련 특성들
                
                # 상관관계 변화 계산
                correlation_change = np.std(correlation_indicators)
                
                # 적응형 임계값 감지
                is_crisis, threshold = self.correlation_detector.update_and_detect(correlation_change)
                
                if is_crisis:
                    # 0과 threshold 사이의 비율로 위기 수준 계산
                    if threshold > 0:
                        crisis_level = min(1.0, correlation_change / threshold)
                        return np.tanh(crisis_level)
                    else:
                        return 0.5  # 기본값
                else:
                    return 0.0
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"상관관계 위기 감지 실패: {e}")
            return 0.0
    
    def _detect_volume_crisis(self, features):
        """
        거래량 위기 감지 (적응형 임계값)
        
        비정상적인 거래량 변화를 통한 위기 감지
        """
        try:
            # 거래량 관련 특성 추출
            if len(features) >= 12:
                volume_indicators = features[8:12]  # 거래량 관련 특성들
                
                # 거래량 변화 계산
                volume_change = np.mean(np.abs(volume_indicators))
                
                # 적응형 임계값 감지
                is_crisis, threshold = self.volume_detector.update_and_detect(volume_change)
                
                if is_crisis:
                    # 0과 threshold 사이의 비율로 위기 수준 계산
                    if threshold > 0:
                        crisis_level = min(1.0, volume_change / threshold)
                        return np.tanh(crisis_level)
                    else:
                        return 0.5  # 기본값
                else:
                    return 0.0
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"거래량 위기 감지 실패: {e}")
            return 0.0
    
    def get_anomaly_explanation(self, features):
        """
        위기 감지 결과에 대한 설명 생성 (XAI)
        
        Returns:
            dict: 위기 감지 상세 정보
        """
        if not self.is_fitted:
            return {'error': 'T-Cell not fitted'}
        
        try:
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            anomaly_score = self.detector.decision_function(features_scaled)[0]
            crisis_level = 1 / (1 + np.exp(anomaly_score * self.sensitivity))
            
            # 과거 데이터와의 거리 계산
            if len(self.training_features) > 0:
                distances = np.linalg.norm(
                    self.training_features - features_scaled, axis=1
                )
                avg_distance = distances.mean()
                min_distance = distances.min()
            else:
                avg_distance = 0.0
                min_distance = 0.0
            
            # 특성별 기여도 (간단한 방법)
            feature_importance = np.abs(features_scaled[0] - self.training_features.mean(axis=0))
            top_features = np.argsort(feature_importance)[-3:]  # 상위 3개
            
            explanation = {
                'crisis_level': float(crisis_level),
                'anomaly_score': float(anomaly_score),
                'is_anomaly': anomaly_score < 0,
                'avg_distance_to_normal': float(avg_distance),
                'min_distance_to_normal': float(min_distance),
                'top_anomaly_features': top_features.tolist(),
                'feature_importance': feature_importance.tolist()
            }
            
            return explanation

        except Exception as e:
            self.logger.error(f"Explanation generation failed: {e}")
            return {'error': f"Explanation generation failed: {e}"}
    def get_crisis_statistics(self):
        """위기 감지 통계 요약"""
        return {
            'volatility_crisis_rate': self.volatility_detector.get_crisis_rate(),
            'correlation_crisis_rate': self.correlation_detector.get_crisis_rate(),
            'volume_crisis_rate': self.volume_detector.get_crisis_rate(),
            'overall_crisis_rate': self.overall_detector.get_crisis_rate()
        }
    
    def reset_detectors(self):
        """모든 감지기 재설정"""
        self.volatility_detector = AdaptiveThresholdDetector(
            window_size=512, target_quantile=0.95, target_crisis_rate=0.3
        )
        self.correlation_detector = AdaptiveThresholdDetector(
            window_size=512, target_quantile=0.97, target_crisis_rate=0.25
        )
        self.volume_detector = AdaptiveThresholdDetector(
            window_size=512, target_quantile=0.96, target_crisis_rate=0.35
        )
        self.overall_detector = AdaptiveThresholdDetector(
            window_size=512, target_quantile=0.975, target_crisis_rate=0.4
        )
        self.logger.info("적응형 임계값 감지기들이 재설정되었습니다.")
    
    def save_model(self, filepath):
        """모델 저장"""
        if not self.is_fitted:
            self.logger.warning("학습되지 않은 모델은 저장할 수 없습니다.")
            return False
        
        try:
            # 저장 디렉토리 생성 보장
            base_dir = os.path.dirname(filepath)
            if base_dir:
                os.makedirs(base_dir, exist_ok=True)
            
            model_data = {
                'detector': self.detector,
                'scaler': self.scaler,
                'contamination': self.contamination,
                'sensitivity': self.sensitivity,
                'is_fitted': self.is_fitted
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.logger.info(f"T-Cell 모델이 저장되었습니다: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"모델 저장 실패: {e}")
            return False
    
    def load_model(self, filepath):
        """모델 로드"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.detector = model_data['detector']
            self.scaler = model_data['scaler']
            self.contamination = model_data['contamination']
            self.sensitivity = model_data['sensitivity']
            self.is_fitted = model_data['is_fitted']
            
            self.logger.info(f"T-Cell 모델이 로드되었습니다: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"모델 로드 실패: {e}")
            return False