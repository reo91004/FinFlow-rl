# utils/expert_strategies.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from utils.logger import BIPDLogger

class ExpertStrategies:
    """사전훈련을 위한 전문가 전략 구현"""
    
    def __init__(self, n_assets: int):
        self.n_assets = n_assets
        self.logger = BIPDLogger().get_logger('expert_strategies')
        
    def buy_and_hold_strategy(self, market_data: pd.DataFrame) -> np.ndarray:
        """Buy & Hold 전략: 시장 시가총액 가중 또는 동일 비중"""
        # 동일 비중 Buy & Hold
        weights = np.ones(self.n_assets) / self.n_assets
        return weights
    
    def equal_weight_strategy(self, market_data: pd.DataFrame) -> np.ndarray:
        """Equal Weight 전략: 모든 자산에 동일한 비중"""
        return np.ones(self.n_assets) / self.n_assets
    
    def momentum_strategy(self, market_data: pd.DataFrame, lookback: int = 20) -> np.ndarray:
        """간단한 모멘텀 전략: 최근 수익률 기반"""
        if len(market_data) < lookback + 1:
            return self.equal_weight_strategy(market_data)
            
        # 최근 lookback일 수익률 계산
        returns = market_data.pct_change().iloc[-lookback:].mean()
        
        # 수익률이 높은 자산에 더 많은 비중
        momentum_scores = returns.fillna(0)
        
        # 음수 방지를 위해 최소값을 0으로 조정
        momentum_scores = momentum_scores - momentum_scores.min() + 0.1
        
        # 정규화하여 가중치 생성
        weights = momentum_scores / momentum_scores.sum()
        
        # NaN 처리 및 유효성 검사
        weights = weights.fillna(1.0 / self.n_assets)
        if not np.isclose(weights.sum(), 1.0):
            weights = weights / weights.sum()
            
        return weights.values
    
    def mean_reversion_strategy(self, market_data: pd.DataFrame, lookback: int = 60) -> np.ndarray:
        """평균 회귀 전략: 최근 성과가 나쁜 자산에 더 투자"""
        if len(market_data) < lookback + 1:
            return self.equal_weight_strategy(market_data)
            
        # 장기 평균 수익률 대비 최근 성과
        long_term_returns = market_data.pct_change().iloc[-lookback:].mean()
        short_term_returns = market_data.pct_change().iloc[-5:].mean()
        
        # 상대적으로 저조한 성과를 보인 자산에 더 투자
        reversion_scores = long_term_returns - short_term_returns
        reversion_scores = reversion_scores.fillna(0)
        
        # 점수를 양수로 만들고 정규화
        reversion_scores = reversion_scores - reversion_scores.min() + 0.1
        weights = reversion_scores / reversion_scores.sum()
        
        weights = weights.fillna(1.0 / self.n_assets)
        if not np.isclose(weights.sum(), 1.0):
            weights = weights / weights.sum()
            
        return weights.values
    
    def volatility_based_strategy(self, market_data: pd.DataFrame, lookback: int = 30) -> np.ndarray:
        """변동성 기반 전략: 낮은 변동성 자산에 더 투자"""
        if len(market_data) < lookback + 1:
            return self.equal_weight_strategy(market_data)
            
        # 최근 변동성 계산
        returns = market_data.pct_change().iloc[-lookback:]
        volatilities = returns.std()
        
        # 변동성의 역수에 비례하여 가중치 부여 (낮은 변동성에 더 투자)
        inv_volatilities = 1.0 / (volatilities + 1e-6)  # 0으로 나누기 방지
        inv_volatilities = inv_volatilities.fillna(1.0)
        
        weights = inv_volatilities / inv_volatilities.sum()
        
        if not np.isclose(weights.sum(), 1.0):
            weights = weights / weights.sum()
            
        return weights.values
    
    def generate_expert_data(self, market_data: pd.DataFrame, 
                           strategy_name: str = "mixed") -> List[Tuple[np.ndarray, np.ndarray]]:
        """전문가 데이터 생성: (state, action) 쌍들"""
        expert_data = []
        
        # 최소 데이터 요구사항 확인
        if len(market_data) < 100:
            self.logger.warning(f"시장 데이터가 부족함: {len(market_data)}일 < 100일")
            return expert_data
            
        for i in range(100, len(market_data)):  # 충분한 히스토리가 있는 시점부터
            # 현재 시점까지의 데이터
            current_data = market_data.iloc[:i+1]
            
            # 시장 특성 추출 (이는 실제 시스템의 extract_market_features와 동일해야 함)
            market_features = self._extract_simple_features(current_data)
            
            # 전문가 전략에 따른 행동 결정
            if strategy_name == "buy_hold":
                action = self.buy_and_hold_strategy(current_data)
            elif strategy_name == "equal_weight":
                action = self.equal_weight_strategy(current_data)
            elif strategy_name == "momentum":
                action = self.momentum_strategy(current_data)
            elif strategy_name == "mean_reversion":
                action = self.mean_reversion_strategy(current_data)
            elif strategy_name == "volatility":
                action = self.volatility_based_strategy(current_data)
            else:  # mixed strategy
                # 여러 전략을 랜덤하게 섞어서 사용
                strategies = [
                    self.buy_and_hold_strategy,
                    self.equal_weight_strategy,
                    self.momentum_strategy,
                    self.mean_reversion_strategy,
                    self.volatility_based_strategy
                ]
                chosen_strategy = np.random.choice(strategies)
                action = chosen_strategy(current_data)
            
            # (state, action) 쌍 저장
            expert_data.append((market_features, action))
            
        self.logger.info(f"전문가 데이터 생성 완료: {len(expert_data)}개 샘플 ({strategy_name} 전략)")
        return expert_data
    
    def _extract_simple_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """간단한 시장 특성 추출 (system.py의 extract_market_features와 호환)"""
        if len(market_data) < 20:
            return np.zeros(12, dtype=np.float32)
            
        returns = market_data.pct_change().dropna()
        if len(returns) < 10:
            return np.zeros(12, dtype=np.float32)
            
        # 기본 통계
        recent_returns = returns.tail(20)
        
        try:
            features = [
                recent_returns.std().mean(),  # 평균 변동성
                np.corrcoef(recent_returns.T).mean(),  # 평균 상관관계
                recent_returns.mean().mean(),  # 평균 수익률
                recent_returns.skew().mean(),  # 왜도
                recent_returns.kurtosis().mean(),  # 첨도
                len(recent_returns[recent_returns.sum(axis=1) < -0.02]) / len(recent_returns),  # 하락일 비율
                recent_returns.max().max() - recent_returns.min().min(),  # 수익률 범위
                recent_returns.std().std(),  # 변동성의 변동성
                # 추가 특성들 (총 12개 맞추기)
                recent_returns.quantile(0.25).mean(),
                recent_returns.quantile(0.75).mean(),
                (recent_returns > 0).sum().sum() / (len(recent_returns) * len(recent_returns.columns)),
                recent_returns.abs().mean().mean()
            ]
            
            # NaN 처리
            features = [float(f) if not np.isnan(f) and not np.isinf(f) else 0.0 for f in features]
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            self.logger.warning(f"특성 추출 실패: {e}")
            return np.zeros(12, dtype=np.float32)