# bipd/core/environment.py

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from collections import deque
from data.features import FeatureExtractor
from utils.logger import BIPDLogger
from utils.metrics import calculate_concentration_index
from config import (
    SHARPE_WINDOW, SHARPE_SCALE, REWARD_BUFFER_SIZE, 
    REWARD_OUTLIER_SIGMA, REWARD_CLIP_MIN, REWARD_CLIP_MAX
)

class RunningNormalizer:
    """실시간 평균/표준편차 정규화"""
    
    def __init__(self, feature_dim: int, momentum: float = 0.99):
        self.momentum = momentum
        self.running_mean = np.zeros(feature_dim, dtype=np.float32)
        self.running_var = np.ones(feature_dim, dtype=np.float32)
        self.count = 0
    
    def update_and_normalize(self, features: np.ndarray) -> np.ndarray:
        """특성 업데이트 및 정규화"""
        if self.count == 0:
            self.running_mean = features.copy()
            self.running_var = np.ones_like(features)
        else:
            # Exponential moving average
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * features
            diff = features - self.running_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * (diff ** 2)
        
        self.count += 1
        
        # 정규화 (0으로 나누기 방지)
        std = np.sqrt(self.running_var + 1e-8)
        normalized = (features - self.running_mean) / std
        
        return np.clip(normalized, -5.0, 5.0)  # 극단값 클리핑

class PortfolioEnvironment:
    """
    포트폴리오 관리 환경
    
    강화학습 에이전트가 상호작용할 수 있는 포트폴리오 환경을 제공
    매 스텝마다 포트폴리오 리밸런싱을 수행하고 보상을 계산
    """
    
    def __init__(self, price_data: pd.DataFrame, feature_extractor: FeatureExtractor,
                 initial_capital: float = 1000000, transaction_cost: float = 0.001):
        self.price_data = price_data
        self.feature_extractor = feature_extractor
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        
        self.n_assets = len(price_data.columns)
        self.max_steps = len(price_data) - feature_extractor.lookback_window - 1
        
        # 환경 상태
        self.current_step = 0
        self.portfolio_value = initial_capital
        self.cash = initial_capital
        self.holdings = np.zeros(self.n_assets)  # 주식 보유량
        self.weights = np.ones(self.n_assets) / self.n_assets
        
        # 성과 추적
        self.portfolio_history = []
        self.weight_history = []
        self.return_history = []
        self.cost_history = []
        
        # 가중치 검증 통계
        self.validation_stats = {
            'negative_weights': 0,
            'invalid_weights': 0,
            'total_validations': 0,
            'last_log_step': 0
        }
        
        # 상태 정규화기 추가
        self.state_normalizer = RunningNormalizer(feature_dim=12)  # 시장 특성 12차원
        
        self.logger = BIPDLogger("Environment")
        
        self.logger.info(
            f"포트폴리오 환경이 초기화되었습니다. "
            f"자산수={self.n_assets}, 최대스텝={self.max_steps}, "
            f"초기자본={initial_capital:,.0f}"
        )
    
    def reset(self) -> np.ndarray:
        """환경 초기화"""
        self.current_step = 0
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.holdings = np.zeros(self.n_assets)
        self.weights = np.ones(self.n_assets) / self.n_assets
        
        # 히스토리 초기화
        self.portfolio_history = [self.initial_capital]
        self.weight_history = [self.weights.copy()]
        self.return_history = []
        self.cost_history = []
        
        # 통계 초기화
        self.validation_stats = {
            'negative_weights': 0,
            'invalid_weights': 0,
            'total_validations': 0,
            'last_log_step': 0
        }
        
        return self._get_state()
    
    def step(self, new_weights: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        환경 스텝 실행
        
        Args:
            new_weights: 새로운 포트폴리오 가중치
            
        Returns:
            next_state, reward, done, info
        """
        if self.current_step >= self.max_steps:
            return self._get_state(), 0.0, True, {}
        
        # 가중치 검증 및 정규화
        new_weights = self._validate_weights(new_weights)
        
        # 리밸런싱 비용 계산
        weight_change = np.abs(new_weights - self.weights).sum()
        transaction_cost = self.portfolio_value * weight_change * self.transaction_cost
        
        # 시장 수익률 적용
        current_prices = self.price_data.iloc[self.current_step + self.feature_extractor.lookback_window]
        next_prices = self.price_data.iloc[self.current_step + self.feature_extractor.lookback_window + 1]
        
        # 안전한 수익률 계산 (극단값 클리핑)
        asset_returns = (next_prices - current_prices) / current_prices
        asset_returns = np.clip(asset_returns, -0.5, 0.5)  # 일일 수익률 50% 제한
        portfolio_return = np.dot(new_weights, asset_returns)
        
        # 포트폴리오 가치 업데이트
        old_value = self.portfolio_value
        self.portfolio_value *= (1 + portfolio_return)
        self.portfolio_value -= transaction_cost
        
        # 실제 수익률 (비용 포함)
        actual_return = (self.portfolio_value - old_value) / old_value
        
        # 보상 계산
        reward = self._calculate_reward(actual_return, new_weights, asset_returns)
        
        # Phase 2: 단체 재투영 (거래비용 후 최종 정규화)
        # 거래비용과 슬리피지 적용 후 음수/합≠1 문제 해결
        if new_weights.sum() == 0:
            new_weights = np.ones(self.n_assets) / self.n_assets
        else:
            # 음수 제거 후 재정규화
            new_weights = np.maximum(new_weights, 0.0)
            new_weights = new_weights / new_weights.sum()
        
        # 상태 업데이트
        self.weights = new_weights.copy()
        self.current_step += 1
        
        # 히스토리 업데이트
        self.portfolio_history.append(self.portfolio_value)
        self.weight_history.append(new_weights.copy())
        self.return_history.append(actual_return)
        self.cost_history.append(transaction_cost)
        
        # 종료 조건 (Python native bool로 변환)
        done = bool((self.current_step >= self.max_steps) or (self.portfolio_value <= 0))
        
        # 정보 딕셔너리
        info = {
            'portfolio_value': self.portfolio_value,
            'portfolio_return': actual_return,
            'transaction_cost': transaction_cost,
            'weight_change': weight_change,
            'concentration': calculate_concentration_index(new_weights)
        }
        
        return self._get_state(), reward, done, info
    
    def _get_state(self) -> np.ndarray:
        """현재 상태 벡터 생성"""
        if self.current_step >= self.max_steps:
            # 마지막 상태
            return np.zeros(12 + 1 + self.n_assets, dtype=np.float32)
        
        # 시장 특성 추출
        current_idx = self.current_step + self.feature_extractor.lookback_window
        market_features = self.feature_extractor.extract_features(
            self.price_data, current_idx
        )
        
        # 시장 특성 정규화 적용
        normalized_features = self.state_normalizer.update_and_normalize(market_features)
        
        # 위기 수준 계산
        if len(self.return_history) >= 5:
            recent_volatility = np.std(self.return_history[-5:])
            crisis_level = np.clip(recent_volatility * 50, 0, 1)
        else:
            crisis_level = 0.0
        
        # 상태 벡터 구성
        state = np.concatenate([
            normalized_features,  # 정규화된 시장 특성
            [crisis_level],
            self.weights
        ]).astype(np.float32)
        
        if not hasattr(self, '_normalization_logged'):
            self.logger.info("상태 정규화 활성화")
            self._normalization_logged = True
        
        return state
    
    def _validate_weights(self, weights: np.ndarray) -> np.ndarray:
        """가중치 검증 및 정규화"""
        weights = np.array(weights, dtype=np.float32)
        
        # 가중치 검증 통계 업데이트
        self.validation_stats['total_validations'] += 1
        
        # NaN/Inf 체크
        if np.any(~np.isfinite(weights)):
            self.validation_stats['invalid_weights'] += 1
            self.logger.debug("가중치에 NaN/Inf가 포함되어 균등 가중치로 대체합니다.")
            weights = np.ones(self.n_assets) / self.n_assets
        
        # 음수 체크 (통계적 로깅)
        if np.any(weights < 0):
            self.validation_stats['negative_weights'] += 1
            weights = np.maximum(weights, 0)
            
            # 주기적 통계 로깅 (100회마다)
            if (self.validation_stats['total_validations'] - self.validation_stats['last_log_step']) >= 100:
                negative_rate = self.validation_stats['negative_weights'] / self.validation_stats['total_validations']
                self.logger.debug(
                    f"가중치 검증 통계 (최근 100회): "
                    f"음수 가중치 {self.validation_stats['negative_weights']}회 ({negative_rate:.1%}), "
                    f"총 검증 {self.validation_stats['total_validations']}회"
                )
                self.validation_stats['last_log_step'] = self.validation_stats['total_validations']
        
        # 정규화
        if weights.sum() == 0:
            weights = np.ones(self.n_assets) / self.n_assets
        else:
            weights = weights / weights.sum()
        
        # 극단값 제한
        min_weight = 0.001
        max_weight = 0.8
        weights = np.clip(weights, min_weight, max_weight)
        weights = weights / weights.sum()  # 재정규화
        
        return weights
    
    def _calculate_reward(self, portfolio_return: float, weights: np.ndarray, 
                         asset_returns: np.ndarray) -> float:
        """
        Sharpe ratio 기반 보상 함수
        
        위험 조정 수익률을 직접 최적화하여 안정적인 학습 유도
        """
        # 1. 기본 수익률 보상
        base_reward = portfolio_return
        
        # 2. Sharpe ratio 보상
        sharpe_reward = 0.0
        
        if len(self.return_history) >= SHARPE_WINDOW:
            recent_returns = np.array(self.return_history[-SHARPE_WINDOW:])
            mean_return = recent_returns.mean()
            std_return = recent_returns.std()
            
            if std_return > 1e-8:
                sharpe_ratio = mean_return / std_return
                sharpe_reward = np.clip(sharpe_ratio * SHARPE_SCALE, -1.0, 1.0)
            else:
                sharpe_reward = 0.0
        
        # 3. 최종 보상 계산
        final_reward = base_reward + sharpe_reward
        
        # 4. 3-sigma 필터링 및 robust 정규화
        if not hasattr(self, 'reward_buffer'):
            self.reward_buffer = deque(maxlen=REWARD_BUFFER_SIZE)
        
        self.reward_buffer.append(final_reward)
        
        # 3-sigma 아웃라이어 필터링
        if len(self.reward_buffer) >= 20:
            reward_array = np.array(self.reward_buffer)
            mean = reward_array.mean()
            std = reward_array.std()
            
            if std > 1e-8:
                z_score = abs((final_reward - mean) / std)
                if z_score > REWARD_OUTLIER_SIGMA:
                    original_reward = final_reward  # 원본 보상 저장
                    final_reward = mean + REWARD_OUTLIER_SIGMA * std * np.sign(final_reward - mean)
                    if not hasattr(self, '_outlier_count'):
                        self._outlier_count = 0
                    self._outlier_count += 1
                    
                    if self._outlier_count % 10 == 1:
                        self.logger.warning(f"극단적 보상 감지 및 필터링: 원본={original_reward:.2f}, 조정후={final_reward:.2f}")
        
        # MAD 기반 robust 정규화
        if len(self.reward_buffer) >= 10:
            reward_array = np.array(self.reward_buffer)
            median = np.median(reward_array)
            mad = np.median(np.abs(reward_array - median))
            
            if mad > 1e-8:
                final_reward = (final_reward - median) / (1.4826 * mad)
            
        # 최종 클리핑
        final_reward = np.clip(final_reward, REWARD_CLIP_MIN, REWARD_CLIP_MAX)
        
        if not hasattr(self, '_reward_logged'):
            self.logger.info("보상 함수 초기화: 3-sigma 필터링, MAD 정규화, Sharpe 기반")
            self._reward_logged = True
        
        return float(final_reward)
    
    def get_portfolio_metrics(self) -> Dict[str, float]:
        """포트폴리오 성과 메트릭 계산"""
        if len(self.return_history) == 0:
            return {}
        
        returns = np.array(self.return_history)
        
        # 기본 메트릭
        total_return = (self.portfolio_value / self.initial_capital) - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        
        # 샤프 비율
        if returns.std() > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # 최대 낙폭
        cumulative = np.cumprod(1 + returns)
        rolling_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        # 거래 비용
        total_cost = sum(self.cost_history)
        cost_ratio = total_cost / self.initial_capital
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_cost': total_cost,
            'cost_ratio': cost_ratio,
            'final_value': self.portfolio_value
        }
    
    def get_weight_statistics(self) -> Dict[str, float]:
        """가중치 통계"""
        if len(self.weight_history) == 0:
            return {}
        
        weights_array = np.array(self.weight_history)
        
        # 평균 가중치
        avg_weights = weights_array.mean(axis=0)
        
        # 가중치 변동성
        weight_volatility = weights_array.std(axis=0).mean()
        
        # 평균 집중도
        concentrations = [calculate_concentration_index(w) for w in self.weight_history]
        avg_concentration = np.mean(concentrations)
        
        # 턴오버 (가중치 변화율)
        if len(self.weight_history) > 1:
            turnovers = []
            for i in range(1, len(self.weight_history)):
                turnover = np.abs(self.weight_history[i] - self.weight_history[i-1]).sum()
                turnovers.append(turnover)
            avg_turnover = np.mean(turnovers)
        else:
            avg_turnover = 0.0
        
        return {
            'avg_concentration': avg_concentration,
            'weight_volatility': weight_volatility,
            'avg_turnover': avg_turnover,
            'max_weight': avg_weights.max(),
            'min_weight': avg_weights.min()
        }
    
    def render_summary(self) -> str:
        """환경 요약 정보"""
        portfolio_metrics = self.get_portfolio_metrics()
        weight_stats = self.get_weight_statistics()
        
        summary = f"""
포트폴리오 환경 요약:
==================
진행도: {self.current_step}/{self.max_steps} ({self.current_step/self.max_steps:.1%})
포트폴리오 가치: {self.portfolio_value:,.0f} (초기: {self.initial_capital:,.0f})

성과 메트릭:
- 총 수익률: {portfolio_metrics.get('total_return', 0):.2%}
- 연율화 수익률: {portfolio_metrics.get('annual_return', 0):.2%}
- 변동성: {portfolio_metrics.get('volatility', 0):.2%}
- 샤프 비율: {portfolio_metrics.get('sharpe_ratio', 0):.3f}
- 최대 낙폭: {portfolio_metrics.get('max_drawdown', 0):.2%}

가중치 통계:
- 평균 집중도: {weight_stats.get('avg_concentration', 0):.3f}
- 가중치 변동성: {weight_stats.get('weight_volatility', 0):.3f}
- 평균 턴오버: {weight_stats.get('avg_turnover', 0):.3f}
        """
        
        return summary.strip()
    
    def get_validation_summary(self) -> str:
        """가중치 검증 통계 요약"""
        if self.validation_stats['total_validations'] == 0:
            return "가중치 검증 통계 없음"
        
        negative_rate = self.validation_stats['negative_weights'] / self.validation_stats['total_validations']
        invalid_rate = self.validation_stats['invalid_weights'] / self.validation_stats['total_validations']
        
        return (f"가중치 검증 통계: "
                f"음수 {self.validation_stats['negative_weights']}회 ({negative_rate:.1%}), "
                f"무효값 {self.validation_stats['invalid_weights']}회 ({invalid_rate:.1%}), "
                f"총 {self.validation_stats['total_validations']}회 검증")