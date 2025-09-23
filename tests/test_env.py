# tests/test_env.py

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

from src.environments.portfolio_env import PortfolioEnv
from src.data.feature_extractor import FeatureExtractor

def create_dummy_price_data(n_assets=5, n_days=100):
    """테스트용 더미 가격 데이터 생성"""
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
    
    # 랜덤 워크 가격 생성
    prices = {}
    for i in range(n_assets):
        returns = np.random.randn(n_days) * 0.02  # 2% 일일 변동성
        price = 100 * np.exp(np.cumsum(returns))
        prices[f'ASSET_{i}'] = price
    
    return pd.DataFrame(prices, index=dates)

class TestPortfolioEnv:
    """포트폴리오 환경 테스트"""
    
    def setup_method(self):
        """각 테스트 전 실행"""
        self.price_data = create_dummy_price_data(n_assets=5, n_days=100)
        self.feature_extractor = FeatureExtractor(window=20)
        self.env = PortfolioEnv(
            price_data=self.price_data,
            feature_extractor=self.feature_extractor,
            initial_capital=1000000,
            transaction_cost=0.001
        )
    
    def test_reset(self):
        """환경 초기화 테스트"""
        state, info = self.env.reset()
        
        # 상태 차원 확인 (12 features + 5 weights + 1 crisis = 18)
        assert state.shape == (18,)
        
        # 초기 포트폴리오 가치
        assert info['portfolio_value'] == 1000000
        
        # 초기 가중치 (모두 0)
        assert np.allclose(info['current_weights'], 0)
    
    def test_step_with_uniform_weights(self):
        """균등 가중치 액션 테스트"""
        state, info = self.env.reset()
        
        # 균등 가중치 액션
        action = np.ones(5) / 5
        
        next_state, reward, done, truncated, info = self.env.step(action)
        
        # 상태 차원
        assert next_state.shape == (18,)
        
        # 가중치 합 = 1
        assert np.abs(info['current_weights'].sum() - 1.0) < 1e-6
        
        # 종료 조건
        assert isinstance(done, bool)
        assert isinstance(truncated, bool)
    
    def test_action_normalization(self):
        """액션 정규화 테스트"""
        state, info = self.env.reset()
        
        # 정규화되지 않은 액션
        action = np.array([2, 3, 1, 4, 2])
        
        next_state, reward, done, truncated, info = self.env.step(action)
        
        # 가중치 합 = 1 (정규화됨)
        assert np.abs(info['current_weights'].sum() - 1.0) < 1e-6
        
        # 모든 가중치 >= 0
        assert np.all(info['current_weights'] >= 0)
    
    def test_turnover_constraint(self):
        """턴오버 제약 테스트"""
        env = PortfolioEnv(
            price_data=self.price_data,
            feature_extractor=self.feature_extractor,
            initial_capital=1000000,
            transaction_cost=0.001,
            max_turnover=0.2  # 20% 최대 턴오버
        )
        
        state, info = env.reset()
        
        # 첫 번째 스텝: [1,0,0,0,0] 가중치
        action1 = np.array([1, 0, 0, 0, 0])
        env.step(action1)
        
        # 두 번째 스텝: [0,0,0,0,1] 가중치 (200% 턴오버 시도)
        action2 = np.array([0, 0, 0, 0, 1])
        next_state, reward, done, truncated, info = env.step(action2)
        
        # 실제 턴오버가 제약 이하
        actual_turnover = np.abs(info['current_weights'] - action1).sum()
        assert actual_turnover <= 0.2 + 1e-6
    
    def test_t_plus_1_settlement(self):
        """T+1 체결 규칙 테스트"""
        state, info = self.env.reset()
        
        initial_value = info['portfolio_value']
        
        # 액션 실행
        action = np.ones(5) / 5
        next_state, reward, done, truncated, info = self.env.step(action)
        
        # T+1: 다음 날 수익률이 적용됨
        # (정확한 검증은 수익률 데이터 접근 필요)
        assert 'portfolio_value' in info
        assert info['portfolio_value'] != initial_value  # 가치 변화
    
    def test_crisis_level_update(self):
        """위기 수준 업데이트 테스트"""
        state, info = self.env.reset()
        
        # 위기 수준 설정
        self.env.set_crisis_level(0.8)
        
        # 다음 상태에 반영
        next_state = self.env._get_state()
        
        # 마지막 요소가 위기 수준
        assert next_state.to_array()[-1] == 0.8

if __name__ == "__main__":
    pytest.main([__file__, "-v"])