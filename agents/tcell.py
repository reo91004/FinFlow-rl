# bipd/agents/tcell.py

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pickle
import os
from collections import deque
from typing import Dict
from utils.logger import BIPDLogger
from config import (
    THRESHOLD_WINDOW_SIZE, VOLATILITY_CRISIS_QUANTILE, CORRELATION_CRISIS_QUANTILE,
    VOLUME_CRISIS_QUANTILE, OVERALL_CRISIS_QUANTILE,
    VOLATILITY_CRISIS_RATE, CORRELATION_CRISIS_RATE, VOLUME_CRISIS_RATE, OVERALL_CRISIS_RATE
)

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
        
        # EMA(지수이동평균) 안정화 시스템
        self.ema_alpha = 0.2  # EMA 계수 (0.2 = 최근 20% 반영)
        self.ema_volatility = None
        self.ema_correlation = None
        self.ema_volume = None
        self.ema_overall = None
        
        # 히스테리시스 시스템 (급격한 변동 방지)
        self.hysteresis_margin = 0.1  # 10% 마진
        self.previous_crisis_state = {
            'volatility': False,
            'correlation': False,
            'volume': False,
            'overall': False
        }
        
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
            
            # 다차원 위기 분석 (원시 점수)
            raw_volatility_crisis = self._detect_volatility_crisis(features)
            raw_correlation_crisis = self._detect_correlation_crisis(features)
            raw_volume_crisis = self._detect_volume_crisis(features)
            
            # EMA 안정화 적용
            volatility_crisis = self._apply_ema_stabilization('volatility', raw_volatility_crisis)
            correlation_crisis = self._apply_ema_stabilization('correlation', raw_correlation_crisis)
            volume_crisis = self._apply_ema_stabilization('volume', raw_volume_crisis)
            overall_crisis_ema = self._apply_ema_stabilization('overall', overall_crisis)
            
            # 히스테리시스 적용 (급격한 상태 변화 방지)
            volatility_crisis = self._apply_hysteresis('volatility', volatility_crisis)
            correlation_crisis = self._apply_hysteresis('correlation', correlation_crisis)
            volume_crisis = self._apply_hysteresis('volume', volume_crisis)
            overall_crisis_final = self._apply_hysteresis('overall', overall_crisis_ema)
            
            # 위기 벡터 생성
            crisis_vector = np.array([volatility_crisis, correlation_crisis, volume_crisis])
            
            return {
                'overall_crisis': float(overall_crisis_final),
                'volatility_crisis': float(volatility_crisis),
                'correlation_crisis': float(correlation_crisis),
                'volume_crisis': float(volume_crisis),
                'volatility_score': raw_volatility_crisis,  # 원시 점수 추가 (디버깅용)
                'correlation_score': raw_correlation_crisis,
                'volume_score': raw_volume_crisis,
                'overall_score': raw_crisis,
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

    # ===== EMA 안정화 및 히스테리시스 시스템 =====
    
    def _apply_ema_stabilization(self, crisis_type: str, raw_score: float) -> float:
        """
        지수이동평균(EMA)을 통한 위기 점수 안정화
        
        Args:
            crisis_type: 'volatility', 'correlation', 'volume', 'overall'
            raw_score: 원시 위기 점수
            
        Returns:
            float: EMA 적용된 안정화된 점수
        """
        ema_attr = f'ema_{crisis_type}'
        
        # 첫 번째 관측치인 경우 초기화
        if getattr(self, ema_attr) is None:
            setattr(self, ema_attr, raw_score)
            return raw_score
        
        # EMA 계산: EMA_t = α * X_t + (1-α) * EMA_{t-1}
        current_ema = getattr(self, ema_attr)
        new_ema = self.ema_alpha * raw_score + (1 - self.ema_alpha) * current_ema
        
        # EMA 값 업데이트
        setattr(self, ema_attr, new_ema)
        
        return new_ema
    
    def _apply_hysteresis(self, crisis_type: str, ema_score: float, threshold: float = 0.5) -> float:
        """
        히스테리시스를 통한 위기 상태 안정화 (급격한 상태 변화 방지)
        
        Args:
            crisis_type: 위기 유형
            ema_score: EMA 적용된 점수
            threshold: 위기 판정 기준선
            
        Returns:
            float: 히스테리시스 적용된 최종 점수
        """
        previous_state = self.previous_crisis_state[crisis_type]
        
        if previous_state:  # 이전에 위기 상태였다면
            # 위기 해제를 위해서는 임계값보다 마진만큼 낮아야 함
            if ema_score < (threshold - self.hysteresis_margin):
                self.previous_crisis_state[crisis_type] = False
                return ema_score
            else:
                # 위기 상태 유지
                return max(ema_score, threshold + self.hysteresis_margin * 0.5)
        
        else:  # 이전에 정상 상태였다면
            # 위기 진입을 위해서는 임계값보다 마진만큼 높아야 함
            if ema_score > (threshold + self.hysteresis_margin):
                self.previous_crisis_state[crisis_type] = True
                return ema_score
            else:
                # 정상 상태 유지
                return min(ema_score, threshold - self.hysteresis_margin * 0.5)
    
    def get_stabilization_stats(self) -> Dict[str, float]:
        """
        안정화 시스템 통계 정보 반환
        
        Returns:
            dict: EMA 값들과 히스테리시스 상태
        """
        return {
            'ema_volatility': self.ema_volatility,
            'ema_correlation': self.ema_correlation,
            'ema_volume': self.ema_volume,
            'ema_overall': self.ema_overall,
            'hysteresis_states': self.previous_crisis_state.copy(),
            'ema_alpha': self.ema_alpha,
            'hysteresis_margin': self.hysteresis_margin
        }
    
    def reset_stabilization(self) -> None:
        """안정화 시스템 리셋"""
        self.ema_volatility = None
        self.ema_correlation = None
        self.ema_volume = None
        self.ema_overall = None
        self.previous_crisis_state = {
            'volatility': False,
            'correlation': False,
            'volume': False,
            'overall': False
        }
        self.logger.info("T-Cell 안정화 시스템이 리셋되었습니다.")