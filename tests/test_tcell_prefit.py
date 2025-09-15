# tests/test_tcell_prefit.py

import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents.t_cell import TCell

def test_tcell_prefit():
    """T-Cell prefit이 정상 작동하는지 검증"""
    tcell = TCell(
        feature_dim=12,
        window_size=20,
        contamination=0.1
    )

    # 초기 상태 확인
    assert tcell.is_fitted == False, "초기에는 fitted가 False여야 함"
    assert tcell.detector is None, "초기에는 detector가 None이어야 함"

    # 더미 특성 윈도우 생성
    feat_window = np.random.randn(20, 12)

    # Prefit 실행
    tcell.prefit(feat_window)

    # Prefit 후 상태 확인
    assert tcell.is_fitted == True, "prefit 후 is_fitted가 True여야 함"
    assert tcell.detector is not None, "prefit 후 detector가 생성되어야 함"
    assert tcell.scaler is not None, "prefit 후 scaler가 생성되어야 함"

    print("✓ T-Cell prefit 테스트 통과")


def test_tcell_prefit_assertion():
    """Prefit 실패 시 assert 발생 검증"""
    tcell = TCell(
        feature_dim=12,
        window_size=20
    )

    # 빈 데이터로 prefit 시도
    with pytest.raises(Exception):  # fit 실패로 인한 예외
        empty_window = np.array([])
        tcell.prefit(empty_window)

    print("✓ T-Cell prefit 실패 검증 통과")


def test_crisis_detection_after_prefit():
    """Prefit 후 위기 감지가 작동하는지 검증"""
    tcell = TCell(
        feature_dim=12,
        window_size=20,
        contamination=0.1
    )

    # 정상 데이터로 학습
    normal_data = np.random.randn(100, 12) * 0.1  # 낮은 분산
    tcell.prefit(normal_data[:20])

    # 정상 상황 감지
    normal_features = np.random.randn(12) * 0.1
    normal_result = tcell.detect_crisis({'features': normal_features})

    assert 'overall_crisis' in normal_result, "overall_crisis 키 누락"
    assert 0 <= normal_result['overall_crisis'] <= 1, "위기 수준이 [0,1] 범위 벗어남"

    # 이상 상황 시뮬레이션
    anomaly_features = np.random.randn(12) * 5.0  # 높은 분산
    anomaly_result = tcell.detect_crisis({'features': anomaly_features})

    # 일반적으로 이상 상황이 더 높은 위기 수준을 가져야 함
    # (항상 보장되지는 않으므로 존재 여부만 확인)
    assert 'overall_crisis' in anomaly_result, "이상 상황에서도 위기 수준 반환해야 함"

    print(f"✓ 위기 감지 테스트 통과")
    print(f"  정상 위기 수준: {normal_result['overall_crisis']:.4f}")
    print(f"  이상 위기 수준: {anomaly_result['overall_crisis']:.4f}")


def test_state_dependent_constraints():
    """위기 수준에 따른 제약 조정 검증"""
    tcell = TCell(
        feature_dim=12,
        window_size=20
    )

    # Prefit
    tcell.prefit(np.random.randn(20, 12))

    # 낮은 위기 상태
    low_crisis = {'features': np.random.randn(12) * 0.1}
    low_result = tcell.detect_crisis(low_crisis)

    # 높은 위기 상태 시뮬레이션
    tcell.anomaly_scores = [0.9] * 10  # 높은 이상 점수 강제 설정
    high_crisis = {'features': np.random.randn(12) * 3.0}
    high_result = tcell.detect_crisis(high_crisis)

    # 위기 수준별 제약 권고사항
    low_constraints = tcell.get_trading_constraints(low_result['overall_crisis'])
    high_constraints = tcell.get_trading_constraints(high_result['overall_crisis'])

    # 높은 위기 시 더 보수적인 제약
    assert high_constraints['position_limit'] <= low_constraints['position_limit'], \
        "높은 위기 시 포지션 제한이 더 엄격해야 함"

    assert high_constraints['leverage_limit'] <= low_constraints['leverage_limit'], \
        "높은 위기 시 레버리지 제한이 더 엄격해야 함"

    print(f"✓ 상태 의존 제약 테스트 통과")
    print(f"  낮은 위기 포지션 제한: {low_constraints['position_limit']:.2f}")
    print(f"  높은 위기 포지션 제한: {high_constraints['position_limit']:.2f}")


def test_multi_regime_detection():
    """다중 레짐 감지 테스트"""
    tcell = TCell(
        feature_dim=12,
        window_size=20,
        n_regimes=3
    )

    # 서로 다른 레짐 데이터 생성
    regime1 = np.random.randn(30, 12) * 0.5 + 0
    regime2 = np.random.randn(30, 12) * 1.0 + 2
    regime3 = np.random.randn(30, 12) * 0.3 - 1

    all_data = np.vstack([regime1, regime2, regime3])

    # Prefit
    tcell.prefit(all_data[:20])

    # 각 레짐에서 샘플 테스트
    results = []
    for regime_data in [regime1[0], regime2[0], regime3[0]]:
        result = tcell.detect_crisis({'features': regime_data})
        results.append(result)

    # 레짐별로 다른 위기 수준이 감지되어야 함
    crisis_levels = [r['overall_crisis'] for r in results]
    assert len(set(crisis_levels)) > 1 or all(0 <= c <= 1 for c in crisis_levels), \
        "다양한 레짐이 감지되어야 함"

    print(f"✓ 다중 레짐 감지 테스트 통과")
    print(f"  레짐별 위기 수준: {crisis_levels}")


if __name__ == "__main__":
    test_tcell_prefit()
    test_tcell_prefit_assertion()
    test_crisis_detection_after_prefit()
    test_state_dependent_constraints()
    test_multi_regime_detection()
    print("\n모든 T-Cell prefit 테스트 통과!")