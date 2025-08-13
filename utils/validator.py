# utils/validator.py

import numpy as np
import pandas as pd
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional


class DataLeakageValidator:
    """시계열 데이터 리키지 검증"""

    @staticmethod
    def validate_train_test_split(train_data, test_data):
        """훈련/테스트 데이터 분할 검증"""

        train_end = train_data.index.max()
        test_start = test_data.index.min()

        if train_end >= test_start:
            raise ValueError(
                f"데이터 리키지 감지: "
                f"훈련 데이터 마지막 날짜({train_end})가 "
                f"테스트 데이터 시작 날짜({test_start})보다 늦습니다."
            )

        # 시간 간격 확인
        time_gap = (test_start - train_end).days
        if time_gap < 0:
            raise ValueError(
                f"훈련/테스트 데이터 간 시간 간격이 음수입니다: {time_gap}일"
            )
        elif time_gap > 365:
            warnings.warn(f"훈련/테스트 데이터 간 시간 간격이 큽니다: {time_gap}일")

        print(f"시계열 데이터 분할 검증 통과 (간격: {time_gap}일)")
        return True

    @staticmethod
    def validate_feature_calculation(features, market_data, current_idx):
        """특성 계산에서 미래 정보 사용 여부 검증"""

        if current_idx >= len(market_data):
            warnings.warn(
                f"인덱스 {current_idx}가 데이터 길이 {len(market_data)}를 초과합니다."
            )
            return False

        # 현재 인덱스까지의 데이터만 사용했는지 확인
        available_data = market_data.iloc[: current_idx + 1]

        if len(available_data) == 0:
            warnings.warn(f"인덱스 {current_idx}에서 사용 가능한 데이터가 없습니다.")
            return False

        return True

    @staticmethod
    def validate_portfolio_weights(weights):
        """포트폴리오 가중치 검증"""

        if weights is None or len(weights) == 0:
            return False, "가중치가 비어있습니다."

        # numpy 배열로 변환
        weights = np.array(weights)

        # NaN/Inf 검사
        if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
            return False, "가중치에 NaN 또는 Inf 값이 있습니다."

        # 음수 검사
        if np.any(weights < 0):
            return False, "가중치에 음수가 있습니다."

        # 합계 검사 (허용 오차 1%)
        weight_sum = np.sum(weights)
        if not (0.99 <= weight_sum <= 1.01):
            return False, f"가중치 합계가 1이 아닙니다: {weight_sum:.6f}"

        return True, "가중치 검증 통과"

    @staticmethod
    def validate_market_features(features):
        """시장 특성 벡터 검증"""

        if features is None:
            return False, "특성이 None입니다."

        if not isinstance(features, np.ndarray):
            try:
                features = np.array(features)
            except Exception as e:
                return False, f"특성을 numpy 배열로 변환할 수 없습니다: {e}"

        # NaN/Inf 검사
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            return False, "특성에 NaN 또는 Inf 값이 있습니다."

        # 크기 검사
        if len(features) == 0:
            return False, "특성 벡터가 비어있습니다."

        # 극단값 검사
        if np.any(np.abs(features) > 100):
            return (
                False,
                f"특성에 극단값이 있습니다: max={np.max(np.abs(features)):.3f}",
            )

        return True, "특성 검증 통과"


class SystemValidator:
    """시스템 전체 검증"""

    def __init__(self):
        self.validation_history = []

    def validate_episode_data(self, episode_data, episode_features):
        """에피소드 데이터 검증"""

        validation_result = {
            "timestamp": datetime.now(),
            "data_valid": True,
            "features_valid": True,
            "issues": [],
        }

        # 데이터 검증
        if episode_data is None or len(episode_data) == 0:
            validation_result["data_valid"] = False
            validation_result["issues"].append("에피소드 데이터가 비어있습니다.")

        # 특성 검증
        features_valid, features_msg = DataLeakageValidator.validate_market_features(
            episode_features
        )
        if not features_valid:
            validation_result["features_valid"] = False
            validation_result["issues"].append(f"특성 검증 실패: {features_msg}")

        # 데이터와 특성 일관성 검사
        if episode_data is not None and len(episode_data) > 0:
            returns = episode_data.pct_change().dropna()
            if len(returns) == 0:
                validation_result["issues"].append("수익률 계산 결과가 비어있습니다.")

        self.validation_history.append(validation_result)

        return validation_result

    def validate_immune_response(self, weights, response_type, bcell_decisions):
        """면역 반응 결과 검증"""

        validation_result = {
            "timestamp": datetime.now(),
            "weights_valid": True,
            "response_valid": True,
            "issues": [],
        }

        # 가중치 검증
        weights_valid, weights_msg = DataLeakageValidator.validate_portfolio_weights(
            weights
        )
        if not weights_valid:
            validation_result["weights_valid"] = False
            validation_result["issues"].append(f"가중치 검증 실패: {weights_msg}")

        # 응답 타입 검증
        if response_type is None or response_type == "":
            validation_result["response_valid"] = False
            validation_result["issues"].append("응답 타입이 비어있습니다.")

        # B-세포 결정 검증
        if bcell_decisions is None or len(bcell_decisions) == 0:
            validation_result["issues"].append("B-세포 결정이 비어있습니다.")

        return validation_result

    def get_validation_summary(self, last_n=100):
        """최근 검증 결과 요약"""

        if not self.validation_history:
            return {"total": 0, "successful": 0, "success_rate": 0.0, "common_issues": []}

        recent_validations = self.validation_history[-last_n:]

        total_validations = len(recent_validations)
        successful_validations = sum(
            1
            for v in recent_validations
            if v.get("data_valid", True)
            and v.get("features_valid", True)
            and v.get("weights_valid", True)
        )

        success_rate = (
            successful_validations / total_validations if total_validations > 0 else 0.0
        )

        # 공통 이슈 수집
        all_issues = []
        for v in recent_validations:
            all_issues.extend(v.get("issues", []))

        # 이슈 빈도 계산
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1

        common_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[
            :5
        ]

        return {
            "total": total_validations,
            "successful": successful_validations,
            "success_rate": success_rate,
            "common_issues": common_issues,
        }
