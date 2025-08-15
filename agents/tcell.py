# bipd/agents/tcell.py

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pickle
import os
from utils.logger import BIPDLogger

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
                f"정상비율={normal_ratio:.2%}"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"T-Cell 학습 실패: {e}")
            return False
    
    def detect_crisis(self, features):
        """
        현재 시장 상황의 위기 수준 감지
        
        Args:
            features: np.array of shape (feature_dim,)
            
        Returns:
            crisis_level: float [0, 1] - 0: 정상, 1: 극심한 위기
        """
        if not self.is_fitted:
            self.logger.warning("T-Cell이 학습되지 않았습니다.")
            return 0.0
        
        try:
            # 특성 정규화
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Isolation Forest 이상치 점수 계산
            # decision_function: 양수(정상), 음수(이상)
            anomaly_score = self.detector.decision_function(features_scaled)[0]
            
            # 위기 수준으로 변환 [0, 1]
            # sigmoid 변환으로 부드러운 확률값 생성
            crisis_level = 1 / (1 + np.exp(anomaly_score * self.sensitivity))
            
            # 안전한 범위로 클리핑
            crisis_level = np.clip(crisis_level, 0.0, 1.0)
            
            return float(crisis_level)
            
        except Exception as e:
            self.logger.error(f"위기 감지 실패: {e}")
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
            self.logger.error(f"설명 생성 실패: {e}")
            return {'error': str(e)}
    
    def save_model(self, filepath):
        """모델 저장"""
        if not self.is_fitted:
            self.logger.warning("학습되지 않은 모델은 저장할 수 없습니다.")
            return False
        
        try:
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