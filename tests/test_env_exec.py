# tests/test_env_exec.py

import pytest
import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.core.env import PortfolioEnv
from src.data.features import FeatureExtractor

def test_trading_execution():
    """거래 체결 테스트"""
    # 더미 가격 데이터 생성
    dates = pd.date_range('2020-01-01', periods=200, freq='D')
    prices = pd.DataFrame(
        np.random.randn(200, 10).cumsum(axis=0) + 100,
        index=dates,
        columns=[f'Asset_{i}' for i in range(10)]
    )

    # 환경 생성 (완화된 설정)
    env = PortfolioEnv(
        price_data=prices,
        feature_extractor=FeatureExtractor(window=20),
        initial_capital=1000000,
        turnover_cost=0.001,
        slip_coeff=0.0005,
        no_trade_band=0.0005,  # 완화
        max_turnover=0.9  # 상향
    )

    # 환경 리셋
    state, _ = env.reset()

    # 균등가중치로 시작했는지 확인
    assert np.allclose(env.weights, 1.0 / env.n_assets), "균등가중치 초기화 실패"

    total_turnover = 0.0
    for _ in range(100):
        # 랜덤 액션 (Dirichlet 분포)
        action = np.random.dirichlet(np.ones(env.n_assets))
        next_state, reward, terminated, truncated, info = env.step(action)

        # 턴오버 누적
        if 'avg_turnover' in info:
            total_turnover += info['avg_turnover']

    # 100스텝 내 거래 발생 확인
    assert total_turnover > 0.0, "100스텝 내 turnover 발생 필수"
    assert env.portfolio_value != env.initial_capital, "포트폴리오 가치 변화 필수"

    print(f"✓ 거래 체결 테스트 통과: 총 턴오버 = {total_turnover:.4f}")

def test_no_trade_band():
    """무거래 밴드 테스트"""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    prices = pd.DataFrame(
        np.ones((100, 5)) * 100,  # 고정 가격
        index=dates,
        columns=[f'Asset_{i}' for i in range(5)]
    )

    env = PortfolioEnv(
        price_data=prices,
        initial_capital=1000000,
        no_trade_band=0.0005
    )

    state, _ = env.reset()
    current_weights = env.weights.copy()

    # 작은 변화 액션
    small_change = current_weights.copy()
    small_change[0] += 0.0001  # 밴드보다 작은 변화
    small_change = small_change / small_change.sum()

    _, _, _, _, info = env.step(small_change)

    # 무거래 확인
    assert np.allclose(env.weights, current_weights), "작은 변화는 무시되어야 함"

    print("✓ 무거래 밴드 테스트 통과")

if __name__ == "__main__":
    test_trading_execution()
    test_no_trade_band()
    print("\n모든 환경 테스트 통과!")