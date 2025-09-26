# src/core/env.py

"""
포트폴리오 거래 환경 (T+1 결제)

목적: Gymnasium 호환 포트폴리오 최적화 환경 구현
의존: FeatureExtractor (특성 추출), FinFlowLogger (로깅)
사용처: FinFlowTrainer (메인 학습), OfflineDataset (데이터 수집), 평가 스크립트
역할: 오프라인 데이터 수집과 온라인 학습의 핵심 환경

구현 내용:
- T+1 결제 규칙 (시간 t 행동 → t 종가 실행 → t+1 수익률 적용)
- 거래 비용 및 슬리피지 모델링
- 상태공간: 시장특성(12D) + 포트폴리오 가중치(30D) + 위기레벨(1D)
- 행동공간: 심플렉스 위 포트폴리오 가중치
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass
import gymnasium as gym
from gymnasium import spaces
import torch

from src.utils.logger import FinFlowLogger
from src.data.feature_extractor import FeatureExtractor
from src.environments.reward_functions import PortfolioObjective, RewardNormalizer

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
            self.features.flatten() if self.features.ndim > 1 else self.features,
            self.weights.flatten() if self.weights.ndim > 1 else self.weights,
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
                 transaction_cost: float = 0.001,
                 slippage: float = 0.0005,
                 no_trade_band: float = 0.002,
                 max_leverage: float = 1.0,
                 max_turnover: float = 0.5,
                 max_steps: Optional[int] = None,
                 objective_config: Optional[Dict] = None,
                 use_advanced_reward: bool = False):
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
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.no_trade_band = no_trade_band
        self.max_leverage = max_leverage
        self.max_turnover = max_turnover
        self.max_steps = max_steps or len(price_data) - 1
        
        # 로거
        self.logger = FinFlowLogger("PortfolioEnv")

        # 고급 보상 함수 초기화
        self.use_advanced_reward = use_advanced_reward
        if use_advanced_reward and objective_config:
            self.objective = PortfolioObjective(objective_config)
            self.reward_normalizer = RewardNormalizer()
        else:
            self.objective = None
            self.reward_normalizer = None

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
        self.cash = initial_capital
        self.weights = np.zeros(self.n_assets)
        self.prev_weights = np.zeros(self.n_assets)  # 이전 가중치 추적
        self.crisis_level = 0.0
        self.market_data_cache = None
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """환경 초기화 (Gymnasium API)"""
        super().reset(seed=seed, options=options)
        
        self.current_step = self.feature_extractor.window  # 충분한 데이터 확보
        self.portfolio_value = self.initial_capital
        # 균등 배분으로 시작 (무거래 루프 방지)
        self.weights = np.ones(self.n_assets) / self.n_assets
        self.prev_weights = np.ones(self.n_assets) / self.n_assets  # 이전 가중치도 초기화
        self.cash = 0  # 현금 0으로 시작 (모두 투자)
        self.crisis_level = 0.0

        # 무거래 카운터 추가
        self.no_trade_counter = 0

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
        
        # No-trade band 적용
        weight_change = np.abs(action - self.weights)
        if np.max(weight_change) < self.no_trade_band:
            action = self.weights.copy()

        # 최대 턴오버 제약
        turnover = np.sum(np.abs(action - self.weights))
        if turnover > self.max_turnover:
            # 턴오버 제한에 맞게 조정
            scale = self.max_turnover / turnover
            action = self.weights + scale * (action - self.weights)
            action = self._normalize_weights(action)
            turnover = self.max_turnover

        # 무거래 검증 (무거래 루프 감지)
        actual_weight_change = np.abs(action - self.weights).sum()
        if actual_weight_change < 1e-6:
            self.no_trade_counter += 1
            if self.no_trade_counter > 20:  # 20회부터 경고
                self.logger.warning(f"무거래 {self.no_trade_counter}회 연속 감지 - 포트폴리오 고착")

            # 30회 이상 무거래시 강제 거래 트리거
            if self.no_trade_counter >= 30:
                # 작은 랜덤 노이즈를 추가하여 거래 유도
                noise = np.random.randn(self.n_assets) * 0.005
                action = action + noise
                action = np.maximum(action, 0)
                action = action / (action.sum() + 1e-8)
                self.logger.info("무거래 30회 - 강제 거래 트리거 활성화")

            if self.no_trade_counter > 50:  # 100 → 50으로 단축
                self.logger.error("무거래 50회 초과 - 에피소드 종료")
                # terminated 플래그는 나중에 설정
        else:
            self.no_trade_counter = 0
        
        # 거래 비용 계산
        trade_cost = self.transaction_cost * turnover
        slippage_cost = self.slippage * turnover
        total_cost = trade_cost + slippage_cost
        
        # T+1 수익률 적용 (다음 날 수익률)
        if self.current_step < len(self.returns) - 1:
            next_returns = self.returns[self.current_step + 1]
            portfolio_return = np.dot(action, next_returns)
        else:
            portfolio_return = 0.0
        
        # 포트폴리오 가치 업데이트
        self.portfolio_value *= (1 + portfolio_return - total_cost)
        self.prev_weights = self.weights.copy()  # 이전 가중치 저장
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
        terminated = (
            self.portfolio_value <= self.initial_capital * 0.5 or  # 50% 손실
            self.no_trade_counter > 50  # 무거래 50회 초과 (100 → 50)
        )
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
        보상 계산

        고급 보상 함수가 활성화되면 PortfolioObjective 사용,
        그렇지 않으면 기본 수익률 반환
        """
        if self.objective and self.use_advanced_reward:
            # 고급 보상 함수 사용
            returns_tensor = torch.tensor([portfolio_return], dtype=torch.float32)
            weights_tensor = torch.tensor([self.weights], dtype=torch.float32)
            prev_weights_tensor = torch.tensor([self.prev_weights], dtype=torch.float32)

            objective_value, metrics = self.objective(returns_tensor, weights_tensor, prev_weights_tensor)
            reward = objective_value.item()

            # 정규화 적용
            if self.reward_normalizer:
                reward = self.reward_normalizer.normalize(reward)

            return reward
        else:
            # 기본 보상 (수익률 - 비용)
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
    
    def get_market_data(self) -> Dict[str, np.ndarray]:
        """현재 시장 데이터 반환 (T-Cell용)"""
        if self.current_step < self.feature_extractor.window:
            # 데이터가 충분하지 않은 경우
            return {
                'prices': np.zeros((self.feature_extractor.window, self.n_assets)),
                'returns': np.zeros((self.feature_extractor.window, self.n_assets)),
                'volumes': np.ones((self.feature_extractor.window, self.n_assets)),
                'features': np.zeros(feature_dim)
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