# src/core/env.py

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass
import gymnasium as gym
from gymnasium import spaces

from src.utils.logger import FinFlowLogger
from src.data.features import FeatureExtractor

@dataclass
class PortfolioState:
    """포트폴리오 상태 정보"""
    features: np.ndarray  # 시장 특성 (동적 차원)
    weights: np.ndarray   # 현재 포트폴리오 가중치
    crisis_level: float   # T-Cell 위기 수준
    portfolio_value: float
    cash: float
    step: int
    
    def to_array(self) -> np.ndarray:
        """상태를 단일 배열로 변환 (동적 차원)"""
        return np.concatenate([
            self.features,
            self.weights,
            [self.crisis_level]
        ])

class PortfolioEnv(gym.Env):
    """
    T+1 체결 규칙을 적용한 포트폴리오 거래 환경
    
    액션 w_t는 t일 종가로 집행 → t+1일 수익률 적용
    Gymnasium API 준수
    """
    
    def __init__(self,
                 price_data: pd.DataFrame,
                 feature_extractor: Optional[FeatureExtractor] = None,
                 initial_capital: float = 1000000,
                 turnover_cost: float = 0.001,  # transaction_cost 대신 turnover_cost 사용
                 slip_coeff: float = 0.0005,  # slippage 대신 slip_coeff 사용
                 no_trade_band: float = 0.0005,  # 0.002에서 완화
                 max_leverage: float = 1.0,
                 max_turnover: float = 0.9,  # 0.5에서 상향
                 max_steps: Optional[int] = None):
        """
        Args:
            price_data: 가격 데이터 (DataFrame)
            feature_extractor: 특성 추출기
            initial_capital: 초기 자본금
            transaction_cost: 거래 수수료
            slippage: 슬리피지
            no_trade_band: 무거래 밴드 (작은 변화 무시)
            max_leverage: 최대 레버리지
            max_turnover: 일일 최대 턴오버
            max_steps: 에피소드 최대 길이
        """
        super().__init__()
        
        self.price_data = price_data
        self.n_assets = len(price_data.columns)
        self.asset_names = list(price_data.columns)
        
        # 특성 추출기
        self.feature_extractor = feature_extractor or FeatureExtractor(window=20)
        
        # 환경 파라미터
        self.initial_capital = initial_capital
        self.turnover_cost = turnover_cost  # 비용 키 통일
        self.slip_coeff = slip_coeff  # 비용 키 통일
        self.no_trade_band = no_trade_band
        self.max_leverage = max_leverage
        self.max_turnover = max_turnover
        self.max_steps = max_steps or len(price_data) - 1

        # 잔량 누적 버퍼 (fractional shares 처리)
        self._residual = np.zeros(self.n_assets)
        
        # 로거
        self.logger = FinFlowLogger("PortfolioEnv")
        
        # 수익률 계산 (T+1 체결용)
        self.returns = price_data.pct_change().fillna(0).values
        
        # Gym spaces - 동적 feature 차원
        feature_dim = self.feature_extractor.total_dim if hasattr(self.feature_extractor, 'total_dim') else 12
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(feature_dim + self.n_assets + 1,),  # features + weights + crisis
            dtype=np.float32
        )
        
        # 액션 공간: 포트폴리오 가중치 (심플렉스)
        self.action_space = spaces.Box(
            low=0, high=1, 
            shape=(self.n_assets,),
            dtype=np.float32
        )
        
        # 상태 변수는 reset()에서 초기화
        self.current_step = 0
        self.portfolio_value = initial_capital
        self.cash = 0.0  # 균등가중치로 시작하므로 현금 0
        self.weights = np.ones(self.n_assets) / self.n_assets  # 균등가중치 초기화
        self.crisis_level = 0.0
        self.market_data_cache = None
        self.holdings = np.zeros(self.n_assets)  # 실제 보유 주식 수
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """환경 초기화 (Gymnasium API)"""
        super().reset(seed=seed, options=options)

        self.current_step = self.feature_extractor.window  # 충분한 데이터 확보
        self.portfolio_value = self.initial_capital

        # 균등 가중치로 시작하여 현금 고착 방지
        self.weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets

        # 초기 가격으로 보유 주식 수 계산
        initial_prices = self.price_data.iloc[self.current_step].values
        self.holdings = (self.weights * self.portfolio_value / initial_prices).astype(np.float32)
        self.cash = 0.0  # 모든 자금을 주식에 투자

        self.crisis_level = 0.0
        self._residual = np.zeros(self.n_assets)  # 잔량 버퍼 초기화
        
        # 성과 추적
        self.portfolio_values = [self.initial_capital]
        self.all_weights = [self.weights.copy()]
        self.all_returns = []
        self.all_turnovers = []
        self.all_costs = []
        
        # 초기 상태 생성
        state = self._get_state()
        info = self._get_info()
        
        return state.to_array(), info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        한 스텝 실행 (T+1 체결, Gymnasium API)
        
        Args:
            action: 목표 포트폴리오 가중치 (심플렉스)
            
        Returns:
            next_state: 다음 상태
            reward: 보상
            terminated: 에피소드 종료 여부 (목표 달성 또는 실패)
            truncated: 최대 스텝 도달로 인한 종료
            info: 추가 정보
        """
        # 액션 정규화 (심플렉스 투영)
        action = self._normalize_weights(action)
        
        # No-trade band 적용 (완화된 기준)
        weight_change = np.abs(action - self.weights)
        if np.sum(weight_change) < self.no_trade_band:  # max 대신 sum 사용
            action = self.weights.copy()
        
        # 최대 턴오버 제약
        turnover = np.sum(np.abs(action - self.weights))
        if turnover > self.max_turnover:
            # 턴오버 제한에 맞게 조정
            scale = self.max_turnover / turnover
            action = self.weights + scale * (action - self.weights)
            action = self._normalize_weights(action)
            turnover = self.max_turnover
        
        # 거래 비용 계산 (통일된 키 사용)
        trade_cost = self.turnover_cost * turnover
        slippage_cost = self.slip_coeff * turnover
        total_cost = trade_cost + slippage_cost
        
        # T+1 수익률 적용 (다음 날 수익률)
        if self.current_step < len(self.returns) - 1:
            next_returns = self.returns[self.current_step + 1]
            portfolio_return = np.dot(action, next_returns)
        else:
            portfolio_return = 0.0
        
        # 포트폴리오 가치 업데이트
        self.portfolio_value *= (1 + portfolio_return - total_cost)
        self.weights = action.copy()
        
        # 기록
        self.portfolio_values.append(self.portfolio_value)
        self.all_weights.append(self.weights.copy())
        self.all_returns.append(portfolio_return)
        self.all_turnovers.append(turnover)
        self.all_costs.append(total_cost)
        
        # 스텝 증가
        self.current_step += 1
        
        # 보상 계산
        reward = self._calculate_reward(portfolio_return, total_cost, turnover)
        
        # 종료 조건
        terminated = self.portfolio_value <= self.initial_capital * 0.5  # 50% 손실
        truncated = self.current_step >= min(self.max_steps, len(self.price_data) - 2)
        
        # 다음 상태
        next_state = self._get_state()
        info = self._get_info()
        
        return next_state.to_array(), reward, terminated, truncated, info
    
    def _get_state(self) -> PortfolioState:
        """현재 상태 생성"""
        # 시장 특성 추출 (12D)
        features = self.feature_extractor.extract_features(
            self.price_data, 
            current_idx=self.current_step
        )
        
        return PortfolioState(
            features=features,
            weights=self.weights,
            crisis_level=self.crisis_level,
            portfolio_value=self.portfolio_value,
            cash=self.cash,
            step=self.current_step
        )
    
    def _calculate_reward(self, portfolio_return: float, cost: float, turnover: float) -> float:
        """
        보상 계산 (단순 버전 - objectives.py에서 고도화)
        
        여기서는 기본 수익률만 반환
        """
        return portfolio_return - cost
    
    def _normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        """가중치를 심플렉스로 정규화"""
        weights = np.clip(weights, 0, 1)
        weights_sum = np.sum(weights)
        
        if weights_sum > 0:
            return weights / weights_sum
        else:
            # 균등 가중치
            return np.ones(self.n_assets) / self.n_assets
    
    def _get_info(self) -> Dict[str, Any]:
        """추가 정보 반환"""
        if len(self.all_returns) > 0:
            returns_array = np.array(self.all_returns)
            
            # 샤프 비율 계산 (연율화)
            if len(returns_array) > 1 and returns_array.std() > 0:
                sharpe = np.sqrt(252) * returns_array.mean() / returns_array.std()
            else:
                sharpe = 0.0
            
            # 최대 낙폭
            cum_returns = np.cumprod(1 + returns_array)
            running_max = np.maximum.accumulate(cum_returns)
            drawdown = (cum_returns - running_max) / running_max
            max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0
            
            # 평균 턴오버
            avg_turnover = np.mean(self.all_turnovers) if self.all_turnovers else 0.0
            
            return {
                'portfolio_value': self.portfolio_value,
                'total_return': (self.portfolio_value / self.initial_capital - 1),
                'sharpe_ratio': sharpe,
                'max_drawdown': max_drawdown,
                'avg_turnover': avg_turnover,
                'current_weights': self.weights.copy(),
                'step': self.current_step
            }
        else:
            return {
                'portfolio_value': self.portfolio_value,
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'avg_turnover': 0.0,
                'current_weights': self.weights.copy(),
                'step': self.current_step
            }
    
    def set_crisis_level(self, crisis_level: float):
        """T-Cell로부터 위기 수준 업데이트"""
        self.crisis_level = np.clip(crisis_level, 0, 1)

        # 위기 점수에 따른 제약 조정
        if crisis_level >= 0.7:
            self.max_turnover = max(0.3, self.max_turnover * 0.7)  # 높은 위기 시 턴오버 제한
            self.no_trade_band = min(0.01, self.no_trade_band * 1.5)  # 무거래 밴드 확대
        else:
            # 기본값 유지 (이미 완화된 값)
            self.max_turnover = 0.9
            self.no_trade_band = 0.0005

    def _apply_trading_rules(self, target_weights: np.ndarray) -> float:
        """
        거래 규칙 적용 및 체결 처리

        Returns:
            executed_turnover: 실제 체결된 턴오버
        """
        # 무거래 밴드 적용 (완화된 기준)
        delta = target_weights - self.weights
        if np.sum(np.abs(delta)) < self.no_trade_band:
            return 0.0  # 체결 없음

        # 가중치 -> 수량 변환 (min_trade_size 완화)
        min_trade_size = 1  # 기존 100에서 크게 완화
        current_prices = self.price_data.iloc[self.current_step].values
        desired_shares = target_weights * self.portfolio_value / current_prices
        trade_shares = desired_shares - self.holdings

        # 라운딩 후 잔량 누적
        executed = np.zeros_like(trade_shares)
        for i in range(self.n_assets):
            trade_value = abs(trade_shares[i]) * current_prices[i]
            if trade_value >= min_trade_size:
                executed[i] = trade_shares[i]
            else:
                # 잔량 누적 버퍼에 저장
                self._residual[i] += trade_shares[i]
                # 누적된 잔량이 충분하면 체결
                if abs(self._residual[i]) * current_prices[i] >= min_trade_size:
                    executed[i] = self._residual[i]
                    self._residual[i] = 0.0

        # 턴오버 제한 적용
        turnover = np.sum(np.abs(executed)) / np.maximum(np.sum(np.abs(self.holdings)), 1e-12)
        if turnover > self.max_turnover:
            scale = self.max_turnover / (turnover + 1e-12)
            executed *= scale
            turnover = self.max_turnover

        # 체결 반영
        self.holdings += executed
        self.weights = (self.holdings * current_prices) / np.maximum(self.portfolio_value, 1e-12)
        self.weights = self._normalize_weights(self.weights)

        return float(np.sum(np.abs(executed * current_prices)) / self.portfolio_value)


    def get_market_data(self) -> Dict[str, np.ndarray]:
        """현재 시장 데이터 반환 (T-Cell용)"""
        if self.current_step < self.feature_extractor.window:
            # 데이터가 충분하지 않은 경우
            return {
                'prices': np.zeros((self.feature_extractor.window, self.n_assets)),
                'returns': np.zeros((self.feature_extractor.window, self.n_assets)),
                'volumes': np.ones((self.feature_extractor.window, self.n_assets)),
                'features': np.zeros(self.feature_extractor.total_dim if hasattr(self.feature_extractor, 'total_dim') else 12)
            }
        
        # 현재 윈도우의 데이터 추출
        start_idx = max(0, self.current_step - self.feature_extractor.window)
        end_idx = self.current_step + 1
        
        window_prices = self.price_data.iloc[start_idx:end_idx].values
        window_returns = self.returns[start_idx:end_idx]
        
        # 간단한 거래량 프록시 (변동성 기반)
        window_volumes = np.abs(window_returns) * 1000000  # 가상 거래량
        
        # 현재 특성
        current_features = self.feature_extractor.extract_features(
            self.price_data, 
            current_idx=self.current_step
        )
        
        return {
            'prices': window_prices,
            'returns': window_returns,
            'volumes': window_volumes,
            'features': current_features,
            'current_step': self.current_step,
            'asset_names': self.asset_names
        }