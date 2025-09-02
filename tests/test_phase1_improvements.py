# tests/test_phase1_improvements.py

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from typing import Dict, Any

from data.features import FeatureExtractor
from utils.portfolio_utils import (
    project_to_capped_simplex, validate_portfolio_constraints, 
    adaptive_leverage_projection, PortfolioConstraintValidator
)
from core.environment import PortfolioEnvironment
from config import SYMBOLS
from utils.logger import BIPDLogger


class TestFeatureConsistency:
    """Phase 1 개선사항: 피처 기준 일관성 테스트"""
    
    def get_sample_data(self):
        """테스트용 샘플 데이터 생성"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        n_assets = 10
        
        # 트렌드와 노이즈가 있는 가격 데이터 생성
        price_data = pd.DataFrame(
            np.random.randn(100, n_assets).cumsum(axis=0) * 0.5 + 100,
            index=dates,
            columns=SYMBOLS[:n_assets]
        )
        
        return price_data
    
    def test_feature_extraction_consistency(self, sample_data):
        """피처 추출 일관성 검증"""
        extractor = FeatureExtractor(lookback_window=20)
        
        # 여러 시점에서 피처 추출
        features_list = []
        for i in range(25, min(50, len(sample_data))):
            features = extractor.extract_features(sample_data, current_idx=i)
            features_list.append(features)
        
        # 기본 검증
        for features in features_list:
            assert len(features) == 12, f"피처 차원이 12가 아님: {len(features)}"
            assert not np.isnan(features).any(), "NaN 값이 포함됨"
            assert np.isfinite(features).all(), "무한값이 포함됨"
        
        # 일관성 검증: 같은 구조의 데이터에서 비슷한 분포
        features_array = np.array(features_list)
        feature_stds = np.std(features_array, axis=0)
        
        # 극단적인 변동성이 없는지 확인 (임계값: 표준편차 < 10)
        assert np.all(feature_stds < 10), f"일부 피처의 변동성이 너무 큼: {feature_stds}"
    
    def test_market_index_based_features(self, sample_data):
        """시장 인덱스 기반 피처 계산 검증"""
        extractor = FeatureExtractor(lookback_window=20)
        features = extractor.extract_features(sample_data)
        
        # MACD (인덱스 4번째 피처)와 베타 (인덱스 8번째 피처) 검증
        macd_feature = features[4]
        beta_feature = features[8]
        
        # 합리적인 범위 체크
        assert -1 <= macd_feature <= 1, f"MACD 피처가 범위를 벗어남: {macd_feature}"
        assert 0 <= beta_feature <= 3, f"베타 피처가 범위를 벗어남: {beta_feature}"


class TestSimplexProjection:
    """Phase 1 개선사항: 심플렉스 투영 테스트"""
    
    def test_basic_projection_constraints(self):
        """기본 투영 제약 조건 테스트"""
        test_cases = [
            (np.array([0.4, 0.3, 0.2, 0.1]), 1.0, 0.0, 1.0),
            (np.array([1.5, -0.3, 0.2, 0.1]), 1.0, 0.01, 0.5),
            (np.array([0.1, 0.1, 0.1, 0.1]), 2.0, 0.01, 1.0),
            (np.array([0.0, 0.0, 0.0, 1.0]), 1.0, 0.1, 0.4),
        ]
        
        for weights, target_sum, w_min, w_max in test_cases:
            projected = project_to_capped_simplex(weights, target_sum, w_min, w_max)
            validation = validate_portfolio_constraints(projected, target_sum, w_min, w_max)
            
            assert validation['all_constraints_satisfied'], \
                f"제약 조건 미충족: {validation}"
    
    def test_projection_convergence(self):
        """투영 수렴성 테스트"""
        np.random.seed(123)
        
        for _ in range(50):  # 50번 랜덤 테스트
            n = np.random.randint(3, 15)
            weights = np.random.randn(n) * 2  # 큰 변동성
            target_sum = np.random.uniform(0.5, 2.5)
            w_min = np.random.uniform(1e-4, 0.1)
            w_max = np.random.uniform(0.3, 1.0)
            
            projected = project_to_capped_simplex(weights, target_sum, w_min, w_max)
            validation = validate_portfolio_constraints(projected, target_sum, w_min, w_max)
            
            assert validation['all_constraints_satisfied'], \
                f"랜덤 케이스에서 제약 조건 미충족: weights={weights}, projected={projected}"
    
    def test_adaptive_leverage_projection(self):
        """적응형 레버리지 투영 테스트"""
        test_weights = np.array([0.3, 0.2, 0.4, 0.1, 0.2])  # sum=1.2
        
        projected, leverage = adaptive_leverage_projection(
            test_weights, min_leverage=0.5, max_leverage=2.0
        )
        
        # 합계가 선택된 레버리지와 일치하는지 확인
        assert abs(projected.sum() - leverage) < 1e-6
        assert 0.5 <= leverage <= 2.0
    
    def test_constraint_validator_stats(self):
        """제약 조건 검증기 통계 추적 테스트"""
        validator = PortfolioConstraintValidator()
        
        # 여러 테스트 케이스 실행
        test_cases = [
            np.array([0.5, 0.3, 0.2]),  # 정상
            np.array([1.2, -0.1, 0.1]), # 위반
            np.array([0.25, 0.25, 0.25, 0.25]), # 정상
        ]
        
        for weights in test_cases:
            validator.validate_and_project(weights, apply_projection=True)
        
        stats = validator.get_stats()
        assert stats['total_validations'] == 3
        assert 'violation_rate' in stats
        assert 'projection_rate' in stats


class TestAdaptiveNoTradeBand:
    """Phase 2 개선사항: 적응형 노-트레이드 밴드 테스트"""
    
    def get_test_environment(self):
        """테스트용 환경 생성"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        price_data = pd.DataFrame(
            np.random.randn(50, 3).cumsum(axis=0) + 100,
            index=dates,
            columns=SYMBOLS[:3]
        )
        
        from data.features import FeatureExtractor
        feature_extractor = FeatureExtractor(lookback_window=20)
        
        env = PortfolioEnvironment(price_data, feature_extractor)
        return env
    
    def test_band_adaptation_to_volatility(self, test_environment):
        """변동성에 따른 밴드 적응 테스트"""
        env = test_environment
        
        # 초기 밴드 값 저장
        initial_band = env.no_trade_band
        
        # 시뮬레이션: 높은 변동성 수익률 주입
        high_vol_returns = [0.05, -0.04, 0.06, -0.03, 0.04]
        for ret in high_vol_returns:
            env.return_history.append(ret)
        
        # 밴드 업데이트
        env._update_adaptive_no_trade_band()
        
        # 높은 변동성 시 밴드가 증가해야 함
        assert env.no_trade_band >= initial_band, \
            f"높은 변동성 시 밴드가 증가하지 않음: {env.no_trade_band} vs {initial_band}"
        
        # 낮은 변동성 수익률 주입
        low_vol_returns = [0.001, 0.0005, -0.0008, 0.0012, -0.0003]
        for ret in low_vol_returns:
            env.return_history.append(ret)
        
        # 밴드 업데이트
        env._update_adaptive_no_trade_band()
        
        # 낮은 변동성 시 밴드가 감소할 수 있음
        # (단, 최소값 제한 때문에 크게 줄지 않을 수 있음)
        assert env.base_no_trade_band * 0.3 <= env.no_trade_band <= env.base_no_trade_band * 3.0, \
            f"밴드가 허용 범위를 벗어남: {env.no_trade_band}"
    
    def test_band_constraints(self, test_environment):
        """밴드 제약 조건 테스트"""
        env = test_environment
        
        # 극단적 변동성 테스트
        extreme_returns = [0.5, -0.3, 0.4, -0.6, 0.2]  # 매우 높은 변동성
        for ret in extreme_returns:
            env.return_history.append(ret)
        
        env._update_adaptive_no_trade_band()
        
        # 밴드가 최대 제한을 넘지 않아야 함
        max_allowed = env.base_no_trade_band * 3.0
        assert env.no_trade_band <= max_allowed, \
            f"밴드가 최대값을 초과함: {env.no_trade_band} > {max_allowed}"
        
        # 밴드가 최소 제한보다 작지 않아야 함
        min_allowed = env.base_no_trade_band * 0.3
        assert env.no_trade_band >= min_allowed, \
            f"밴드가 최소값보다 작음: {env.no_trade_band} < {min_allowed}"


class TestIntegrationStability:
    """통합 안정성 테스트"""
    
    def test_environment_with_improvements(self):
        """개선사항이 적용된 환경의 안정성 테스트"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        price_data = pd.DataFrame(
            np.random.randn(30, 5).cumsum(axis=0) + 100,
            index=dates,
            columns=SYMBOLS[:5]
        )
        
        from data.features import FeatureExtractor
        feature_extractor = FeatureExtractor(lookback_window=20)
        env = PortfolioEnvironment(price_data, feature_extractor)
        
        # 환경 초기화 확인
        initial_state = env.reset()
        assert len(initial_state) == 12 + 1 + 5  # 피처 + 위기레벨 + 자산수
        
        # 몇 스텝 실행해보기
        for step in range(5):
            action = np.random.dirichlet([1.0] * 5)  # 랜덤 액션
            state, reward, done, info = env.step(action)
            
            # 기본 체크
            assert isinstance(reward, (int, float, np.number))
            assert isinstance(done, bool)
            assert 'portfolio_value' in info
            assert 'final_weights' in info
            
            # 가중치 유효성 체크
            final_weights = info['final_weights']
            assert abs(final_weights.sum() - 1.0) < 1e-3, \
                f"가중치 합이 1이 아님: {final_weights.sum()}"
            assert np.all(final_weights >= 0), "음수 가중치 존재"
        
        # 통계 정보 확인
        stats = env.get_portfolio_metrics()
        assert 'avg_concentration' in stats
        assert 'avg_turnover' in stats


def run_all_tests():
    """모든 테스트 실행"""
    logger = BIPDLogger("Phase1ImprovementsTest")
    
    logger.info("=== Phase 1&2 개선사항 테스트 시작 ===")
    
    # 테스트 클래스들
    test_classes = [
        TestFeatureConsistency,
        TestSimplexProjection, 
        TestAdaptiveNoTradeBand,
        TestIntegrationStability
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        class_name = test_class.__name__
        logger.info(f"\n--- {class_name} 테스트 ---")
        
        # 테스트 인스턴스 생성
        test_instance = test_class()
        
        # 픽스처가 필요한 테스트들을 위한 데이터 준비
        if hasattr(test_instance, 'sample_data'):
            # 샘플 데이터 생성
            np.random.seed(42)
            dates = pd.date_range('2023-01-01', periods=100, freq='D')
            sample_data = pd.DataFrame(
                np.random.randn(100, 10).cumsum(axis=0) * 0.5 + 100,
                index=dates,
                columns=SYMBOLS[:10]
            )
        
        # 클래스의 모든 테스트 메소드 실행
        for method_name in dir(test_instance):
            if method_name.startswith('test_'):
                total_tests += 1
                try:
                    method = getattr(test_instance, method_name)
                    
                    # 매개변수 처리
                    import inspect
                    sig = inspect.signature(method)
                    params = list(sig.parameters.keys())[1:]  # self 제외
                    
                    if 'sample_data' in params:
                        if hasattr(test_instance, 'get_sample_data'):
                            data = test_instance.get_sample_data()
                        else:
                            data = sample_data
                        method(data)
                    elif 'test_environment' in params:
                        if hasattr(test_instance, 'get_test_environment'):
                            env = test_instance.get_test_environment()
                        else:
                            # 환경 생성
                            np.random.seed(42)
                            dates = pd.date_range('2023-01-01', periods=50, freq='D')
                            price_data = pd.DataFrame(
                                np.random.randn(50, 3).cumsum(axis=0) + 100,
                                index=dates,
                                columns=SYMBOLS[:3]
                            )
                            from data.features import FeatureExtractor
                            feature_extractor = FeatureExtractor(lookback_window=20)
                            from core.environment import PortfolioEnvironment
                            env = PortfolioEnvironment(price_data, feature_extractor)
                        method(env)
                    else:
                        method()
                    
                    logger.info(f"  ✓ {method_name}: PASS")
                    passed_tests += 1
                    
                except Exception as e:
                    logger.error(f"  ✗ {method_name}: FAIL - {str(e)}")
                    import traceback
                    logger.error(f"    상세 오류: {traceback.format_exc()}")
    
    # 결과 요약
    logger.info(f"\n=== 테스트 결과 요약 ===")
    logger.info(f"총 테스트: {total_tests}")
    logger.info(f"성공: {passed_tests}, 실패: {total_tests - passed_tests}")
    logger.info(f"성공률: {passed_tests/total_tests*100:.1f}%")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)