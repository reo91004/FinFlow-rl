# src/agents/t_cell.py

"""
T-Cell: 시장 위기 감지 시스템

목적: Isolation Forest 기반 이상치 탐지로 시장 위기 조기 감지
의존: sklearn (IsolationForest, SHAP), logger.py
사용처: FinFlowTrainer, BCell (위기 수준 전달)
역할: 실시간 시장 이상 패턴 감지 및 위기 타입 분류

구현 내용:
- Isolation Forest로 정상 시장 패턴 학습
- SHAP으로 위기 원인 설명 (XAI)
- 6가지 위기 타입 분류 (volatility, correlation, volume, momentum, technical, returns)
- 적응형 임계값으로 위기 수준 동적 조정
- B-Cell에 crisis_level(0~1) 전달하여 리스크 회피도 조정
"""

import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Optional, List
from collections import deque
import shap
from src.utils.logger import FinFlowLogger

class TCell:
    """
    T-Cell: 시장 위기 감지 시스템
    Isolation Forest + SHAP 설명
    """

    def __init__(self,
                 feature_dim: int = 12,
                 contamination: float = 0.1,
                 n_estimators: int = 100,
                 window_size: int = 100):
        """
        Args:
            feature_dim: 특성 차원
            contamination: 이상치 비율
            n_estimators: Isolation Forest 트리 개수
            window_size: 적응형 임계값 윈도우
        """
        self.feature_dim = feature_dim
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.window_size = window_size
        self.logger = FinFlowLogger("TCell")

        # Isolation Forest 감지기
        self.detector = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42
        )

        # 정규화기
        self.scaler = StandardScaler()

        # 특성 이름 및 타입 매핑
        self.feature_names = [
            'returns_1d', 'returns_5d', 'returns_20d',  # returns type
            'volatility',                                 # volatility type
            'rsi', 'macd',                               # technical type
            'volume_ratio',                               # volume type
            'correlation', 'beta',                        # structure type
            'drawdown',                                   # risk type
            'momentum_short', 'momentum_long'             # momentum type
        ][:feature_dim]  # feature_dim에 맞춰 조정

        # 특성 타입 매핑
        self.feature_types = {
            'volatility': ['volatility', 'drawdown'],
            'correlation': ['correlation', 'beta'],
            'volume': ['volume_ratio'],
            'momentum': ['momentum_short', 'momentum_long'],
            'technical': ['rsi', 'macd'],
            'returns': ['returns_1d', 'returns_5d', 'returns_20d']
        }

        # 위기 기록
        self.crisis_history = deque(maxlen=window_size)
        self.crisis_type_history = deque(maxlen=window_size)
        self.explainer = None
        self.is_fitted = False

        # 위기 수준 통계
        self.crisis_stats = {
            'total_detections': 0,
            'crisis_count': 0,
            'normal_count': 0,
            'crisis_types': {'volatility': 0, 'correlation': 0, 'volume': 0, 'momentum': 0}
        }

        self.logger.info(f"T-Cell 초기화 완료 (feature_dim={feature_dim}, contamination={contamination})")

    def fit(self, historical_features: np.ndarray):
        """
        정상 시장 패턴 학습

        Args:
            historical_features: [n_samples, feature_dim]
        """
        if len(historical_features) < 10:
            self.logger.warning("학습 데이터가 너무 적습니다. 최소 10개 샘플이 필요합니다.")
            return

        # 정규화
        normalized_features = self.scaler.fit_transform(historical_features)

        # Isolation Forest 학습
        self.detector.fit(normalized_features)
        self.is_fitted = True

        # SHAP explainer 초기화
        self.explainer = shap.TreeExplainer(self.detector)

        self.logger.info(f"T-Cell 학습 완료: {len(historical_features)} 샘플로 학습")

    def detect_crisis(self, features: np.ndarray) -> Tuple[float, Dict]:
        """
        위기 수준 감지

        Args:
            features: 현재 시장 특성 [feature_dim]

        Returns:
            crisis_level: 0 (정상) ~ 1 (극단 위기)
            explanation: 위기 원인 설명
        """
        if not self.is_fitted:
            self.logger.warning("T-Cell이 학습되지 않았습니다. 기본값 반환")
            return 0.0, {}

        # 입력 차원 확인
        if features.shape[-1] != self.feature_dim:
            self.logger.error(f"특성 차원 불일치: 예상 {self.feature_dim}, 실제 {features.shape[-1]}")
            return 0.0, {}

        # 정규화
        features_normalized = self.scaler.transform(features.reshape(1, -1))

        # 이상치 점수 (-1: 이상, 1: 정상)
        anomaly_prediction = self.detector.predict(features_normalized)[0]
        anomaly_score = self.detector.decision_function(features_normalized)[0]

        # 0~1로 정규화 (낮을수록 이상)
        # decision_function이 음수일수록 이상치
        crisis_level = 1.0 / (1.0 + np.exp(anomaly_score * 2))

        # 극단값 보정
        crisis_level = np.clip(crisis_level, 0.0, 1.0)

        # 위기 설명 생성
        explanation = self._generate_explanation(features_normalized, crisis_level)

        # 기록
        self.crisis_history.append(crisis_level)
        if explanation.get('crisis_type'):
            self.crisis_type_history.append(explanation['crisis_type'])

        self.crisis_stats['total_detections'] += 1
        if crisis_level > 0.5:
            self.crisis_stats['crisis_count'] += 1
            self.logger.debug(f"위기 감지: level={crisis_level:.3f}, type={explanation.get('crisis_type', 'unknown')}")
        else:
            self.crisis_stats['normal_count'] += 1

        return crisis_level, explanation

    def _generate_explanation(self, features_normalized: np.ndarray, crisis_level: float) -> Dict:
        """
        위기 원인 설명 생성

        Args:
            features_normalized: 정규화된 특성
            crisis_level: 위기 수준

        Returns:
            설명 딕셔너리
        """
        explanation = {
            'crisis_level': crisis_level,
            'status': self._get_status_text(crisis_level),
            'crisis_type': None,
            'type_scores': {}
        }

        # 위기 상황에서만 SHAP 설명 생성
        if crisis_level > 0.5 and self.explainer is not None:
            shap_values = self.explainer.shap_values(features_normalized)[0]

            # 상위 3개 기여 요인
            top_indices = np.argsort(np.abs(shap_values))[-3:][::-1]
            explanation['top_factors'] = {}
            for idx in top_indices:
                if idx < len(self.feature_names):
                    explanation['top_factors'][self.feature_names[idx]] = float(shap_values[idx])

            # 위기 타입 분류
            crisis_type, type_scores = self._classify_crisis_type(shap_values)
            explanation['crisis_type'] = crisis_type
            explanation['type_scores'] = type_scores

            self.logger.debug(f"위기 주요 요인: {list(explanation['top_factors'].keys())}, 타입: {crisis_type}")

        return explanation

    def _classify_crisis_type(self, shap_values: np.ndarray) -> Tuple[str, Dict[str, float]]:
        """
        SHAP 값을 기반으로 위기 타입 분류

        Args:
            shap_values: SHAP importance values

        Returns:
            (주요 위기 타입, 각 타입별 점수)
        """
        type_scores = {}

        for crisis_type, features in self.feature_types.items():
            # 해당 타입의 특성들의 SHAP 값 합산
            score = 0
            for feature in features:
                if feature in self.feature_names:
                    idx = self.feature_names.index(feature)
                    if idx < len(shap_values):
                        score += abs(shap_values[idx])
            type_scores[crisis_type] = score

        # 가장 높은 점수의 타입 선택
        primary_type = max(type_scores, key=type_scores.get) if type_scores else 'unknown'

        # 정규화
        total = sum(type_scores.values())
        if total > 0:
            type_scores = {k: v/total for k, v in type_scores.items()}

        return primary_type, type_scores

    def _get_status_text(self, crisis_level: float) -> str:
        """위기 수준에 따른 상태 텍스트"""
        if crisis_level < 0.3:
            return "정상"
        elif crisis_level < 0.5:
            return "주의"
        elif crisis_level < 0.7:
            return "경고"
        else:
            return "위기"

    def get_crisis_stats(self) -> Dict:
        """위기 통계 반환"""
        if not self.crisis_history:
            return self.crisis_stats

        recent_levels = list(self.crisis_history)
        self.crisis_stats.update({
            'mean_crisis': np.mean(recent_levels),
            'max_crisis': np.max(recent_levels),
            'min_crisis': np.min(recent_levels),
            'std_crisis': np.std(recent_levels),
            'crisis_frequency': np.mean([l > 0.5 for l in recent_levels]),
            'current_crisis': recent_levels[-1] if recent_levels else 0.0,
        })

        # 위기 타입별 빈도 업데이트
        if self.crisis_type_history:
            for crisis_type in self.crisis_stats['crisis_types']:
                type_count = sum(1 for t in self.crisis_type_history if t == crisis_type)
                self.crisis_stats['crisis_types'][crisis_type] = type_count / len(self.crisis_type_history)

        return self.crisis_stats

    def reset_stats(self):
        """통계 초기화"""
        self.crisis_history.clear()
        self.crisis_stats = {
            'total_detections': 0,
            'crisis_count': 0,
            'normal_count': 0,
        }
        self.logger.info("T-Cell 통계 초기화")

    def get_adaptive_threshold(self) -> float:
        """적응형 위기 임계값 계산"""
        if len(self.crisis_history) < 10:
            return 0.5  # 기본값

        # 최근 위기 수준의 평균 + 1 표준편차
        recent_levels = list(self.crisis_history)
        mean = np.mean(recent_levels)
        std = np.std(recent_levels)
        threshold = mean + std

        return np.clip(threshold, 0.3, 0.8)  # 0.3 ~ 0.8 범위로 제한

    def save(self, path: str):
        """모델 저장"""
        import pickle
        save_dict = {
            'detector': self.detector,
            'scaler': self.scaler,
            'feature_dim': self.feature_dim,
            'contamination': self.contamination,
            'is_fitted': self.is_fitted,
            'crisis_stats': self.crisis_stats,
        }

        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)

        self.logger.info(f"T-Cell 모델 저장: {path}")

    def load(self, path: str):
        """모델 로드"""
        import pickle
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)

        self.detector = save_dict['detector']
        self.scaler = save_dict['scaler']
        self.feature_dim = save_dict['feature_dim']
        self.contamination = save_dict['contamination']
        self.is_fitted = save_dict['is_fitted']
        self.crisis_stats = save_dict['crisis_stats']

        if self.is_fitted:
            self.explainer = shap.TreeExplainer(self.detector)

        self.logger.info(f"T-Cell 모델 로드: {path}")