# utils/portfolio_utils.py

import numpy as np
from typing import Optional
from utils.logger import BIPDLogger

def project_to_capped_simplex(weights: np.ndarray, 
                             target_sum: float = 1.0,
                             w_min: float = 1e-4,
                             w_max: float = 1.0,
                             max_iterations: int = 100) -> np.ndarray:
    """
    제약 조건을 만족하는 심플렉스로 Euclidean projection 수행
    
    제약 조건:
    - sum(w) = target_sum (레버리지 제약)
    - w_min <= w_i <= w_max for all i (개별 가중치 제약)
    
    Args:
        weights: 입력 가중치 벡터
        target_sum: 목표 합계 (레버리지)
        w_min: 최소 가중치
        w_max: 최대 가중치  
        max_iterations: 최대 반복 횟수
        
    Returns:
        projected_weights: 제약 조건을 만족하는 투영된 가중치
    """
    weights = np.asarray(weights, dtype=np.float64)
    n = len(weights)
    
    # 초기 클리핑
    w = np.clip(weights, w_min, w_max)
    
    # 이미 조건을 만족하는지 확인
    current_sum = w.sum()
    if abs(current_sum - target_sum) < 1e-12:
        return w.astype(np.float32)
    
    # 이분 탐색을 통한 라그랑주 승수 찾기
    # minimize ||w - weights_original||^2 subject to sum(clip(w - lambda, w_min, w_max)) = target_sum
    
    # 더 넓은 탐색 범위 설정
    lambda_low = -10.0  # 충분히 낮은 값
    lambda_high = 10.0  # 충분히 높은 값
    
    # 초기 범위 조정
    for _ in range(10):
        w_low = np.clip(weights - lambda_high, w_min, w_max)
        w_high = np.clip(weights - lambda_low, w_min, w_max)
        
        if w_low.sum() <= target_sum <= w_high.sum():
            break
        
        if w_low.sum() > target_sum:
            lambda_high += 10.0
        if w_high.sum() < target_sum:
            lambda_low -= 10.0
    
    # 이분 탐색 수행
    for iteration in range(max_iterations):
        lambda_mid = (lambda_low + lambda_high) / 2.0
        
        # 라그랑주 조건 적용
        w_candidate = np.clip(weights - lambda_mid, w_min, w_max)
        candidate_sum = w_candidate.sum()
        
        if abs(candidate_sum - target_sum) < 1e-10:
            return w_candidate.astype(np.float32)
        
        if candidate_sum > target_sum:
            lambda_low = lambda_mid
        else:
            lambda_high = lambda_mid
        
        # 수렴 체크
        if abs(lambda_high - lambda_low) < 1e-12:
            break
    
    # 최종 결과 반환
    final_weights = np.clip(weights - lambda_mid, w_min, w_max)
    
    # 합계 보정 (필요시)
    current_sum = final_weights.sum()
    if abs(current_sum - target_sum) > 1e-8:
        # 스케일링으로 합계 보정
        if current_sum > 0:
            final_weights = final_weights * (target_sum / current_sum)
            final_weights = np.clip(final_weights, w_min, w_max)
        else:
            # 모든 가중치가 0인 경우 균등 분배
            final_weights = np.full(n, target_sum / n)
            final_weights = np.clip(final_weights, w_min, w_max)
    
    return final_weights.astype(np.float32)


def validate_portfolio_constraints(weights: np.ndarray,
                                 target_sum: float = 1.0,
                                 w_min: float = 1e-4, 
                                 w_max: float = 1.0,
                                 tolerance: float = 1e-6) -> dict:
    """
    포트폴리오 제약 조건 검증
    
    Returns:
        dict: 검증 결과 딕셔너리
    """
    weights = np.asarray(weights)
    
    # 기본 통계
    w_sum = weights.sum()
    w_min_actual = weights.min()
    w_max_actual = weights.max()
    
    # 제약 조건 체크
    sum_constraint_satisfied = abs(w_sum - target_sum) <= tolerance
    min_constraint_satisfied = w_min_actual >= (w_min - tolerance)
    max_constraint_satisfied = w_max_actual <= (w_max + tolerance)
    
    # 수치 안정성 체크
    has_nan = np.isnan(weights).any()
    has_inf = np.isinf(weights).any()
    
    return {
        'sum_constraint_satisfied': sum_constraint_satisfied,
        'min_constraint_satisfied': min_constraint_satisfied,
        'max_constraint_satisfied': max_constraint_satisfied,
        'all_constraints_satisfied': (sum_constraint_satisfied and 
                                    min_constraint_satisfied and 
                                    max_constraint_satisfied and 
                                    not has_nan and not has_inf),
        'actual_sum': float(w_sum),
        'actual_min': float(w_min_actual),
        'actual_max': float(w_max_actual),
        'target_sum': target_sum,
        'has_numerical_issues': has_nan or has_inf
    }


def adaptive_leverage_projection(weights: np.ndarray,
                               min_leverage: float = 0.5,
                               max_leverage: float = 2.0,
                               w_min: float = 1e-4,
                               w_max: float = 1.0) -> tuple:
    """
    적응형 레버리지 투영
    
    현재 가중치에 가장 가까우면서 제약 조건을 만족하는 레버리지를 찾음
    
    Returns:
        tuple: (투영된 가중치, 사용된 레버리지)
    """
    weights = np.asarray(weights, dtype=np.float64)
    current_sum = weights.sum()
    
    # 목표 레버리지 결정
    if min_leverage <= current_sum <= max_leverage:
        target_leverage = current_sum
    else:
        # 가장 가까운 유효 레버리지로 클리핑
        target_leverage = np.clip(current_sum, min_leverage, max_leverage)
    
    # 심플렉스 투영 수행
    projected_weights = project_to_capped_simplex(
        weights, target_leverage, w_min, w_max
    )
    
    return projected_weights, float(target_leverage)


class PortfolioConstraintValidator:
    """포트폴리오 제약 조건 검증 및 통계 추적 클래스"""
    
    def __init__(self, logger_name: str = "PortfolioValidator"):
        self.logger = BIPDLogger(logger_name)
        self.validation_stats = {
            'total_validations': 0,
            'constraint_violations': 0,
            'numerical_issues': 0,
            'projection_applied': 0
        }
    
    def validate_and_project(self, weights: np.ndarray,
                           target_sum: float = 1.0,
                           w_min: float = 1e-4,
                           w_max: float = 1.0,
                           apply_projection: bool = True) -> tuple:
        """
        검증 및 필요시 투영 수행
        
        Returns:
            tuple: (최종 가중치, 검증 결과 딕셔너리)
        """
        self.validation_stats['total_validations'] += 1
        
        # 초기 검증
        validation_result = validate_portfolio_constraints(
            weights, target_sum, w_min, w_max
        )
        
        final_weights = weights.copy()
        
        # 제약 조건 위반 시 투영 적용
        if not validation_result['all_constraints_satisfied'] and apply_projection:
            final_weights = project_to_capped_simplex(
                weights, target_sum, w_min, w_max
            )
            self.validation_stats['projection_applied'] += 1
            
            # 투영 후 재검증
            validation_result = validate_portfolio_constraints(
                final_weights, target_sum, w_min, w_max
            )
        
        # 통계 업데이트
        if not validation_result['all_constraints_satisfied']:
            self.validation_stats['constraint_violations'] += 1
        
        if validation_result['has_numerical_issues']:
            self.validation_stats['numerical_issues'] += 1
        
        return final_weights, validation_result
    
    def get_stats(self) -> dict:
        """검증 통계 반환"""
        stats = self.validation_stats.copy()
        if stats['total_validations'] > 0:
            stats['violation_rate'] = stats['constraint_violations'] / stats['total_validations']
            stats['numerical_issue_rate'] = stats['numerical_issues'] / stats['total_validations']
            stats['projection_rate'] = stats['projection_applied'] / stats['total_validations']
        
        return stats


# 테스트 함수들
def test_simplex_projection():
    """심플렉스 투영 기능 테스트"""
    logger = BIPDLogger("PortfolioUtilsTest")
    
    # 테스트 케이스들
    test_cases = [
        # (입력 가중치, 목표합, 최소값, 최대값, 설명)
        (np.array([0.4, 0.3, 0.2, 0.1]), 1.0, 0.0, 1.0, "정상 케이스"),
        (np.array([1.5, -0.3, 0.2, 0.1]), 1.0, 0.01, 0.5, "음수/최대값 초과"),
        (np.array([0.1, 0.1, 0.1, 0.1]), 2.0, 0.01, 1.0, "레버리지 2.0"),
        (np.array([0.0, 0.0, 0.0, 1.0]), 1.0, 0.1, 0.4, "집중도 제약"),
    ]
    
    logger.info("심플렉스 투영 테스트 시작")
    
    for i, (weights, target_sum, w_min, w_max, description) in enumerate(test_cases):
        logger.info(f"테스트 케이스 {i+1}: {description}")
        
        # 투영 수행
        projected = project_to_capped_simplex(weights, target_sum, w_min, w_max)
        
        # 검증
        validation = validate_portfolio_constraints(projected, target_sum, w_min, w_max)
        
        logger.info(f"  입력: {weights}")
        logger.info(f"  투영 결과: {projected}")
        logger.info(f"  제약 조건 만족: {validation['all_constraints_satisfied']}")
        logger.info(f"  합계: {validation['actual_sum']:.6f} (목표: {target_sum})")
        logger.info("")
    
    logger.info("심플렉스 투영 테스트 완료")


if __name__ == "__main__":
    test_simplex_projection()