# bipd/core/environment.py

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from collections import deque
import collections
from data.features import FeatureExtractor
from utils.logger import BIPDLogger
from utils.metrics import calculate_concentration_index, calculate_comprehensive_metrics
from config import (
    SHARPE_WINDOW, SHARPE_SCALE, REWARD_BUFFER_SIZE, 
    REWARD_OUTLIER_SIGMA, REWARD_CLIP_MIN, REWARD_CLIP_MAX,
    REWARD_EMPIRICAL_MEAN, REWARD_EMPIRICAL_STD, VOLATILITY_TARGET, VOLATILITY_WINDOW,
    MIN_LEVERAGE, MAX_LEVERAGE, NO_TRADE_BAND, MAX_TURNOVER,
    RISK_PENALTY_WEIGHT, TRANSACTION_PENALTY_WEIGHT, HHI_PENALTY_WEIGHT,
    WEIGHT_VALIDATION_WINDOW, CORRELATION_CHECK_INTERVAL, DEBUG_LOG_INTERVAL
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

class FixedRewardNormalizer:
    """고정 스케일러 기반 보상 정규화 (에피소드 일관성 확보)"""
    
    def __init__(self, target_mean: float = 0.0, target_std: float = 1.0):
        self.target_mean = target_mean
        self.target_std = target_std
        # 훈련 데이터 기반 사전 추정값 (config.py에서 관리)
        self.empirical_mean = REWARD_EMPIRICAL_MEAN  # 일일 평균 수익률 추정
        self.empirical_std = REWARD_EMPIRICAL_STD   # 일일 변동성 추정
        self.count = 0
    
    def normalize(self, reward: float) -> float:
        """고정 스케일러 기반 정규화"""
        self.count += 1
        
        # 고정 스케일링 (에피소드 간 일관성 보장)
        normalized = (reward - self.empirical_mean) / self.empirical_std
        
        # 하드 클리핑 (±3 시그마)
        return np.clip(normalized, -3.0, 3.0)
    
    def reset(self):
        """정규화기 초기화 (고정 파라미터는 유지)"""
        self.count = 0

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
        
        # 설정 스냅샷 저장 (드리프트 방지)
        self._config_snapshot = {
            'initial_capital': initial_capital,
            'transaction_cost': transaction_cost,
            'target_volatility': VOLATILITY_TARGET,
            'volatility_window': VOLATILITY_WINDOW,
            'min_leverage': MIN_LEVERAGE,
            'max_leverage': MAX_LEVERAGE,
            'no_trade_band': NO_TRADE_BAND,
            'max_turnover': MAX_TURNOVER,
            'data_shape': price_data.shape,
            'feature_lookback': feature_extractor.lookback_window
        }
        
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
        
        # 슬라이딩 윈도우 기반 가중치 검증 통계 (config.py에서 관리)
        self.weight_validation_window = collections.deque(maxlen=WEIGHT_VALIDATION_WINDOW)  # 최근 N회 기록
        self.weight_validation_stats = {
            'total_validations': 0,  # 누적 검증 횟수
            'last_report_step': 0   # 마지막 리포트 시점
        }
        
        # 보상 정규화기 (고정 스케일러로 변경)
        self.reward_normalizer = FixedRewardNormalizer()
        
        # Sharpe 비율 EMA 추적기 (개선사항)
        self.sharpe_tracker = {
            'return_ema': 0.0,
            'volatility_ema': 1e-8,  # 0 방지
            'count': 0,
            'decay': 0.9  # 더 빠른 반응을 위해 감소
        }
        
        # 보상-성과 상관성 추적
        self.reward_performance_tracker = {
            'rewards': [],
            'returns': [],
            'last_correlation_check': 0
        }
        
        # 상태 정규화기 추가
        self.state_normalizer = RunningNormalizer(feature_dim=12)  # 시장 특성 12차원
        
        # 변동성 타깃팅 파라미터 (config.py에서 관리)
        self.target_volatility = VOLATILITY_TARGET      # 연간 목표 변동성
        self.volatility_window = VOLATILITY_WINDOW      # 변동성 추정 윈도우
        self.min_leverage = MIN_LEVERAGE                # 최소 레버리지
        self.max_leverage = MAX_LEVERAGE                # 최대 레버리지
        
        # EMA 기반 변동성 추적 (점진적 조정)
        self.volatility_ema = None                      # 변동성 EMA
        self.leverage_ema = 1.0                         # 레버리지 EMA
        self.volatility_decay = 0.9                     # EMA 감쇠율 (10일 반감기)
        self.leverage_decay = 0.95                      # 레버리지 EMA 감쇠율 (20일 반감기)
        
        # 거래비용 최적화 파라미터 (config.py에서 관리)
        self.no_trade_band = NO_TRADE_BAND              # 노-트레이드 밴드
        self.max_turnover = MAX_TURNOVER                # 최대 턴오버
        
        self.logger = BIPDLogger("Environment")
        
        self.logger.info(
            f"포트폴리오 환경이 초기화되었습니다. "
            f"자산수={self.n_assets}, 최대스텝={self.max_steps}, "
            f"초기자본={initial_capital:,.0f}"
        )
        
        # 초기 설정 로깅 (재현성)
        self.logger.debug(f"환경 설정 스냅샷: {self._config_snapshot}")
    
    def _verify_config_integrity(self) -> None:
        """설정 무결성 검증 (드리프트 감지)"""
        current_config = {
            'initial_capital': self.initial_capital,
            'transaction_cost': self.transaction_cost,
            'target_volatility': self.target_volatility,
            'volatility_window': self.volatility_window,
            'min_leverage': self.min_leverage,
            'max_leverage': self.max_leverage,
            'no_trade_band': self.no_trade_band,
            'max_turnover': self.max_turnover,
            'data_shape': self.price_data.shape,
            'feature_lookback': self.feature_extractor.lookback_window
        }
        
        # 드리프트 감지
        drifted_keys = []
        for key, original_value in self._config_snapshot.items():
            current_value = current_config[key]
            if original_value != current_value:
                drifted_keys.append(f"{key}: {original_value} → {current_value}")
        
        # 연구용 표준: 드리프트 발견 시 즉시 실패
        assert not drifted_keys, (
            f"환경 설정 드리프트 감지 (연구 무결성 위반): {drifted_keys}. "
            f"동일한 실험 세션에서 환경 설정이 변경되면 안 됩니다."
        )
    
    def reset(self) -> np.ndarray:
        """환경 초기화 (설정 무결성 검증)"""
        # 설정 드리프트 검증 (연구 표준)
        self._verify_config_integrity()
        
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
        
        # 검증 통계 초기화
        self.weight_validation_window.clear()
        self.weight_validation_stats = {
            'total_validations': 0,
            'last_report_step': 0
        }
        
        # 보상 정규화기 초기화
        self.reward_normalizer.reset()
        
        # 상관성 추적 초기화
        self.reward_performance_tracker = {
            'rewards': [],
            'returns': [],
            'last_correlation_check': 0
        }
        
        # 변동성 EMA 초기화
        self.volatility_ema = None
        self.leverage_ema = 1.0
        
        return self._get_state()
    
    def get_current_state(self) -> np.ndarray:
        """
        5단계: 환경 이중 초기화 방지 - 현재 상태 반환 (reset 불필요)
        
        Returns:
            current_state: 현재 환경 상태
        """
        if not hasattr(self, 'current_step') or self.current_step is None:
            # 환경이 아직 초기화되지 않은 경우 None 반환 (fallback trigger)
            return None
        return self._get_state()
    
    def step(self, new_weights: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        환경 스텝 실행 (시계열 정합성 검증 포함)
        
        Args:
            new_weights: 새로운 포트폴리오 가중치
            
        Returns:
            next_state, reward, done, info
        """
        if self.current_step >= self.max_steps:
            return self._get_state(), 0.0, True, {}
        
        # 시계열 정합성 검증 (사전)
        self._verify_temporal_consistency()
        
        # 가중치 검증 및 정규화
        new_weights = self._validate_weights(new_weights)
        
        # 변동성 타깃팅 적용
        new_weights = self._apply_volatility_targeting(new_weights)
        
        # 거래비용 최적화 적용 (노-트레이드 밴드 & 턴오버 캡)
        new_weights = self._apply_trading_constraints(new_weights)
        
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
        
        # Phase 2: 최종 실행 포트폴리오 준비 (거래비용 후 최종 정규화)
        # 거래비용과 슬리피지 적용 후 음수/합≠1 문제 해결
        final_weights = self._prepare_final_portfolio(new_weights)
        
        # 최종 실행 포트폴리오로 검증 및 로깅
        self._validate_and_log_weights(final_weights, "FINAL_EXECUTION")
        
        # 상태 업데이트 (최종 실행 가중치 사용)
        self.weights = final_weights.copy()
        self.current_step += 1
        
        # 히스토리 업데이트 (최종 실행 가중치 사용)
        self.portfolio_history.append(self.portfolio_value)
        self.weight_history.append(final_weights.copy())
        self.return_history.append(actual_return)
        self.cost_history.append(transaction_cost)
        
        # 종료 조건 (Python native bool로 변환)
        done = bool((self.current_step >= self.max_steps) or (self.portfolio_value <= 0))
        
        # 에피소드 종료 시 시계열 일관성 검증
        if done:
            try:
                self._verify_episode_metrics_consistency()
            except ValueError as e:
                self.logger.error(f"에피소드 종료 시 일관성 검증 실패: {e}")
                # 연구용 표준: 일관성 오류 시 즉시 실패
                raise e
        
        # 정보 딕셔너리 (최종 실행 가중치 기준)
        info = {
            'portfolio_value': self.portfolio_value,
            'portfolio_return': actual_return,
            'transaction_cost': transaction_cost,
            'weight_change': weight_change,
            'concentration': calculate_concentration_index(final_weights),
            'final_weights': final_weights.copy(),  # 최종 실행 가중치 추가
            'temporal_verification': 'passed'  # 시계열 검증 통과 표시
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
        """기본 가중치 검증 및 정규화 (초기 전처리용)"""
        weights = np.array(weights, dtype=np.float32)
        
        # NaN/Inf 체크
        if np.any(~np.isfinite(weights)):
            self.logger.debug("가중치에 NaN/Inf가 포함되어 균등 가중치로 대체합니다.")
            weights = np.ones(self.n_assets) / self.n_assets
        
        # 기본 정규화 (음수 및 합 처리)
        weights = np.maximum(weights, 0.0)  # 음수 제거
        
        if weights.sum() == 0:
            weights = np.ones(self.n_assets) / self.n_assets
        else:
            weights = weights / weights.sum()
        
        return weights
    
    def _prepare_final_portfolio(self, weights: np.ndarray) -> np.ndarray:
        """최종 실행 포트폴리오 준비 (거래비용 적용 후 완전한 비음수 보장)"""
        final_weights = np.array(weights, dtype=np.float32)
        
        # 1. 음수 제거
        final_weights = np.maximum(final_weights, 0.0)
        
        # 2. 영 벡터 방지
        if final_weights.sum() <= 1e-12:
            final_weights = np.ones(self.n_assets) / self.n_assets
        else:
            final_weights = final_weights / final_weights.sum()
        
        # 3. 극단값 방지 (최소 가중치 보장)
        min_weight = 1e-6  # 매우 작은 최소값
        final_weights = np.maximum(final_weights, min_weight)
        final_weights = final_weights / final_weights.sum()  # 재정규화
        
        # 4. 최대 가중치 제한 (80%)
        max_weight = 0.8
        final_weights = np.minimum(final_weights, max_weight)
        final_weights = final_weights / final_weights.sum()  # 재정규화
        
        return final_weights
    
    def _apply_volatility_targeting(self, weights: np.ndarray) -> np.ndarray:
        """시간 정렬 강화된 변동성 타깃팅 (피드백 루프 차단)"""
        if len(self.return_history) < 5:  # 최소 5일 이상
            return weights
        
        # 시간 정렬: 오직 과거 데이터만 사용
        historical_returns = np.array(self.return_history)  # t-1까지의 수익률만
        current_volatility_daily = np.std(historical_returns[-min(self.volatility_window, len(historical_returns)):])
        
        # EMA 기반 변동성 추정 (과거 정보만)
        if self.volatility_ema is None:
            # 초기화: 과거 윈도우의 표준편차만
            self.volatility_ema = current_volatility_daily
        else:
            # 느린 게인으로 EMA 업데이트 (피드백 억제)
            slow_decay = 0.95  # 기존 0.9 → 0.95 (더 보수적)
            self.volatility_ema = (slow_decay * self.volatility_ema + 
                                  (1 - slow_decay) * current_volatility_daily)
        
        # 연간 변동성으로 환산
        annualized_volatility = self.volatility_ema * np.sqrt(252)
        
        # 목표 레버리지 계산
        if annualized_volatility > 1e-6:
            target_leverage = self.target_volatility / annualized_volatility
        else:
            target_leverage = 1.0
        
        # 레버리지 클리핑
        target_leverage = np.clip(target_leverage, self.min_leverage, self.max_leverage)
        
        # 레버리지 변화율 제한 (|ΔL_t| ≤ δ)
        delta_max = 0.05  # 최대 변화율 5%
        leverage_change = target_leverage - self.leverage_ema
        leverage_change_clipped = np.clip(leverage_change, -delta_max, delta_max)
        
        # 제한된 변화율로 레버리지 업데이트
        self.leverage_ema += leverage_change_clipped
        self.leverage_ema = np.clip(self.leverage_ema, self.min_leverage, self.max_leverage)
        
        # 가중치 스케일링 (제한된 레버리지 사용)
        scaled_weights = weights * self.leverage_ema
        
        # 재정규화 (합이 1이 되도록)
        if scaled_weights.sum() > 0:
            scaled_weights = scaled_weights / scaled_weights.sum()
        else:
            scaled_weights = weights  # 백업
        
        # 변동성 타깃팅 로그 (config.py 간격)
        if self.current_step % CORRELATION_CHECK_INTERVAL == 0:
            self.logger.debug(
                f"제어된 변동성 타깃팅 (step {self.current_step}): "
                f"EMA_vol={annualized_volatility:.3f}, 목표_vol={self.target_volatility:.3f}, "
                f"목표_lev={target_leverage:.2f}, 최종_lev={self.leverage_ema:.2f}, "
                f"변화율={leverage_change_clipped:.3f} (제한={delta_max})"
            )
        
        if not hasattr(self, '_volatility_logged'):
            self.logger.info(f"시간 정렬 변동성 타깃팅 활성화: 목표={self.target_volatility:.1%}, "
                           f"느린 게인(0.95), 변화율 제한(±{delta_max})")
            self._volatility_logged = True
        
        return scaled_weights
    
    def _soft_trading_penalty(self, new_weights: np.ndarray) -> float:
        """소프트 거래 제약 패널티 계산"""
        current_weights = self.weights
        
        # 노-트레이드 밴드 패널티
        weight_changes = np.abs(new_weights - current_weights)
        band_violations = np.maximum(0, weight_changes - self.no_trade_band)
        band_penalty = 0.1 * band_violations.sum()  # 라그랑지안 승수 λ = 0.1
        
        # 턴오버 캡 패널티
        total_turnover = weight_changes.sum()
        turnover_violation = max(0, total_turnover - self.max_turnover)
        turnover_penalty = 0.2 * turnover_violation  # 라그랑지안 승수 λ = 0.2
        
        return band_penalty + turnover_penalty
    
    def _apply_trading_constraints(self, new_weights: np.ndarray) -> np.ndarray:
        """점진적 거래 제약 적용 (소프트 제약으로 전환)"""
        current_weights = self.weights
        
        # 기존 하드 제약을 완화된 형태로 유지 (점진적 전환)
        # 1. 극단적인 턴오버만 제한 (기존의 2배 허용)
        total_turnover = np.abs(new_weights - current_weights).sum()
        soft_turnover_limit = self.max_turnover * 2.0  # 더 관대한 제한
        
        if total_turnover > soft_turnover_limit:
            # 점진적 스케일링 (완전히 막지 않고 점진 조정)
            scale_factor = soft_turnover_limit / total_turnover
            scaled_changes = (new_weights - current_weights) * scale_factor
            constrained_weights = current_weights + scaled_changes
        else:
            constrained_weights = new_weights  # 자유로운 거래 허용
        
        # 2. 재정규화 (합이 1이 되도록)
        if constrained_weights.sum() > 0:
            constrained_weights = constrained_weights / constrained_weights.sum()
        else:
            constrained_weights = new_weights  # 백업
        
        # 3. 로깅 (config.py 간격)
        if self.current_step % CORRELATION_CHECK_INTERVAL == 0:
            original_turnover = np.abs(new_weights - current_weights).sum()
            final_turnover = np.abs(constrained_weights - current_weights).sum()
            soft_penalty = self._soft_trading_penalty(new_weights)
            
            self.logger.debug(
                f"소프트 거래 제약 (step {self.current_step}): "
                f"원래_턴오버={original_turnover:.3f}, 최종_턴오버={final_turnover:.3f}, "
                f"소프트_패널티={soft_penalty:.4f}, 제한배수={soft_turnover_limit/self.max_turnover:.1f}x"
            )
        
        if not hasattr(self, '_trading_constraints_logged'):
            self.logger.info(f"소프트 거래 제약 활성화: 기존 제한의 {soft_turnover_limit/self.max_turnover:.1f}배 허용, 초과시 패널티")
            self._trading_constraints_logged = True
        
        return constrained_weights
    
    def _validate_and_log_weights(self, weights: np.ndarray, context: str = "UNKNOWN"):
        """최종 가중치 검증 및 로깅 (슬라이딩 윈도우 기반)"""
        # 검증 대상을 가중치로 제한
        if context != "FINAL_EXECUTION":
            return  # 최종 실행 가중치만 검증
        
        # 검증 수행
        has_negative = np.any(weights < 0)
        has_invalid = np.any(~np.isfinite(weights))
        sum_close_to_one = abs(weights.sum() - 1.0) < 1e-6
        
        # 슬라이딩 윈도우에 기록
        validation_result = {
            'has_negative': has_negative,
            'has_invalid': has_invalid,
            'sum_valid': sum_close_to_one,
            'min_weight': float(weights.min()),
            'max_weight': float(weights.max())
        }
        self.weight_validation_window.append(validation_result)
        
        # 누적 카운터 업데이트
        self.weight_validation_stats['total_validations'] += 1
        
        # 주기적 리포트 (100회마다)
        if (self.weight_validation_stats['total_validations'] - 
            self.weight_validation_stats['last_report_step']) >= 100:
            self._report_weight_validation_statistics()
            self.weight_validation_stats['last_report_step'] = self.weight_validation_stats['total_validations']
    
    def _report_weight_validation_statistics(self):
        """가중치 검증 통계 리포트"""
        if not self.weight_validation_window:
            return
        
        # 슬라이딩 윈도우 통계 계산
        recent_count = len(self.weight_validation_window)
        negative_count = sum(1 for r in self.weight_validation_window if r['has_negative'])
        invalid_count = sum(1 for r in self.weight_validation_window if r['has_invalid'])
        sum_invalid_count = sum(1 for r in self.weight_validation_window if not r['sum_valid'])
        
        negative_rate = negative_count / recent_count if recent_count > 0 else 0.0
        invalid_rate = invalid_count / recent_count if recent_count > 0 else 0.0
        
        # 누적 통계
        total_validations = self.weight_validation_stats['total_validations']
        
        self.logger.debug(
            f"최종 실행 가중치 검증 통계: "
            f"슬라이딩(최근 {recent_count}회) - "
            f"음수: {negative_count}회 ({negative_rate:.1%}), "
            f"무효값: {invalid_count}회 ({invalid_rate:.1%}), "
            f"합불일치: {sum_invalid_count}회 | "
            f"누적: {total_validations}회"
        )
        
        # 경고 발생
        if negative_rate > 0.01:  # 1% 초과 시 경고
            self.logger.warning(f"최종 실행 가중치에서 음수값 비율이 높음: {negative_rate:.1%}")
        if invalid_rate > 0.01:
            self.logger.warning(f"최종 실행 가중치에서 무효값 비율이 높음: {invalid_rate:.1%}")
    
    def _calculate_reward(self, portfolio_return: float, weights: np.ndarray, 
                         asset_returns: np.ndarray) -> float:
        """
        Potential-based reward shaping으로 재설계된 보상 함수
        
        r_t = log(V_{t+1}/V_t) + Φ(s_{t+1}) - Φ(s_t) (정책 불변성 보장)
        """
        # 1. 기본 로그 수익률 보상 (진짜 보상)
        log_return = np.log(max(1 + portfolio_return, 1e-12))
        
        # 2. Potential-based shaping 성분들 계산
        current_potential = self._calculate_state_potential(weights, portfolio_return)
        
        # 이전 상태의 potential (첫 번째 스텝은 0)
        if len(self.return_history) > 0 and len(self.weight_history) > 0:
            prev_weights = self.weight_history[-1]
            prev_return = self.return_history[-1]
            prev_potential = self._calculate_state_potential(prev_weights, prev_return)
        else:
            prev_potential = 0.0
        
        # 3. Potential-based shaping 항 계산 (Φ(s') - Φ(s))
        shaping_reward = current_potential - prev_potential
        
        # 4. 단일 스케일 관문: 진짜 보상 + shaping만 결합
        raw_reward = log_return + shaping_reward
        
        # 5. 소프트 클리핑만 적용 (부호 보존)
        r_max = 0.1  # 스케일링 계수 증가
        final_reward = np.tanh(raw_reward / r_max) * r_max
        
        # 6. 분리 로깅용 진짜 보상 저장
        self._true_reward = log_return
        self._shaping_reward = shaping_reward
        
        # 7. 보상-성과 상관성 추적
        self.reward_performance_tracker['rewards'].append(final_reward)
        self.reward_performance_tracker['returns'].append(portfolio_return)
        
        # 주기적 상관성 체크 (config.py 간격)
        if len(self.reward_performance_tracker['rewards']) >= CORRELATION_CHECK_INTERVAL:
            if (len(self.reward_performance_tracker['rewards']) - 
                self.reward_performance_tracker['last_correlation_check']) >= CORRELATION_CHECK_INTERVAL:
                self._check_reward_performance_correlation()
                self.reward_performance_tracker['last_correlation_check'] = \
                    len(self.reward_performance_tracker['rewards'])
        
        # 디버그 정보 (config.py 간격)
        if self.current_step % DEBUG_LOG_INTERVAL == 0:
            self.logger.debug(
                f"Potential-based 보상 (step {self.current_step}): "
                f"true_rew={log_return:.4f}, shaping_rew={shaping_reward:.4f}, "
                f"curr_pot={current_potential:.4f}, prev_pot={prev_potential:.4f}, "
                f"raw_rew={raw_reward:.4f}, final_rew={final_reward:.4f}"
            )
        
        if not hasattr(self, '_reward_logged'):
            self.logger.info(
                "보상 함수 재설계: Potential-based shaping, 정책 불변성 보장, "
                f"단일 스케일 관문, 소프트 클리핑 (r_max={r_max})"
            )
            self._reward_logged = True
        
        # 안전한 타입 변환 (스칼라 값 보장)
        if hasattr(final_reward, 'item'):
            return float(final_reward.item())
        else:
            return float(final_reward)
    
    def _verify_temporal_consistency(self) -> None:
        """
        시계열 정합성 검증 (연구 무결성 보장)
        """
        # 1. 타임스탬프 정렬 검증
        if self.current_step > 0:
            current_idx = self.current_step + self.feature_extractor.lookback_window
            if current_idx >= len(self.price_data):
                raise ValueError(
                    f"시계열 인덱스 오버플로우: step={self.current_step}, "
                    f"idx={current_idx}, data_len={len(self.price_data)}"
                )
            
            # 가격 데이터 유효성 검증
            current_prices = self.price_data.iloc[current_idx]
            if current_prices.isna().any():
                raise ValueError(f"Step {self.current_step}에서 NaN 가격 데이터 감지")
                
            # 타임스탬프 순서 검증 (가능한 경우)
            if hasattr(self.price_data.index, 'is_monotonic_increasing'):
                if not self.price_data.index.is_monotonic_increasing:
                    raise ValueError("가격 데이터의 타임스탬프가 단조증가하지 않습니다")
        
        # 2. 히스토리 길이 일관성 검증
        expected_length = self.current_step
        histories = {
            'portfolio_history': len(self.portfolio_history) - 1,  # 초기값 제외
            'weight_history': len(self.weight_history) - 1,       # 초기값 제외  
            'return_history': len(self.return_history),
            'cost_history': len(self.cost_history)
        }
        
        for name, actual_length in histories.items():
            if actual_length != expected_length:
                raise ValueError(
                    f"히스토리 길이 불일치: {name} = {actual_length}, "
                    f"expected = {expected_length} (step {self.current_step})"
                )
    
    def _verify_episode_metrics_consistency(self) -> None:
        """
        에피소드 종료 시 지표 교차일관성 검사
        """
        if len(self.return_history) == 0:
            return
        
        # 1. 누적 수익률 계산 방식 비교
        total_return_from_history = sum(self.return_history)
        total_return_from_values = (self.portfolio_value - self.initial_capital) / self.initial_capital
        
        relative_error = abs(total_return_from_history - total_return_from_values) / (abs(total_return_from_values) + 1e-8)
        if relative_error > 0.01:  # 1% 오차 허용
            raise ValueError(
                f"누적 수익률 불일치: history={total_return_from_history:.6f}, "
                f"values={total_return_from_values:.6f}, error={relative_error:.6f}"
            )
        
        # 2. 가중치 히스토리 합 검증
        for i, weights in enumerate(self.weight_history):
            weight_sum = weights.sum()
            if abs(weight_sum - 1.0) > 1e-6:
                raise ValueError(
                    f"가중치 합 불일치 (step {i}): sum={weight_sum:.8f}, expected=1.0"
                )
        
        # 3. 비용 히스토리 비음수 검증
        if any(cost < 0 for cost in self.cost_history):
            negative_costs = [i for i, cost in enumerate(self.cost_history) if cost < 0]
            raise ValueError(f"음의 거래비용 감지: steps {negative_costs}")
        
        self.logger.debug(f"에피소드 지표 일관성 검증 통과: {len(self.return_history)}단계")
    
    def _calculate_state_potential(self, weights: np.ndarray, portfolio_return: float) -> float:
        """
        Potential-based shaping을 위한 상태 퍼텐셜 계산
        
        Φ(s) = -λ_vol*σ(s) - λ_hhi*HHI(w) - λ_tc*turnover(w) + λ_sharpe*sharpe_proxy(s)
        
        Args:
            weights: 포트폴리오 가중치
            portfolio_return: 포트폴리오 수익률
        
        Returns:
            potential: 상태 퍼텐셜 값
        """
        potential = 0.0
        
        # 1. 변동성 퍼텐셜 (낮은 변동성이 높은 퍼텐셜)
        if len(self.return_history) >= 5:
            recent_volatility = np.std(self.return_history[-5:])
        else:
            recent_volatility = abs(portfolio_return)
        lambda_vol = 0.1
        volatility_potential = -lambda_vol * recent_volatility
        potential += volatility_potential
        
        # 2. 집중도(HHI) 퍼텐셜 (분산투자가 높은 퍼텐셜)
        hhi = np.sum(weights ** 2)
        equal_weight_hhi = 1.0 / self.n_assets
        lambda_hhi = 0.05
        concentration_potential = -lambda_hhi * max(0, hhi - equal_weight_hhi)
        potential += concentration_potential
        
        # 3. 거래비용 퍼텐셜 (낮은 회전율이 높은 퍼텐셜)
        if len(self.weight_history) > 0:
            prev_weights = self.weight_history[-1]
            turnover = np.abs(weights - prev_weights).sum()
        else:
            turnover = 0.0
        lambda_tc = 0.02
        turnover_potential = -lambda_tc * turnover
        potential += turnover_potential
        
        # 4. Sharpe proxy 퍼텐셜 (높은 샤프 비율이 높은 퍼텐셜)
        sharpe_reward = self._update_and_get_sharpe_reward(portfolio_return)
        lambda_sharpe = 0.1
        sharpe_potential = lambda_sharpe * sharpe_reward
        potential += sharpe_potential
        
        return potential
    
    def _check_reward_performance_correlation(self):
        """보상-성과 상관성 검증"""
        if len(self.reward_performance_tracker['rewards']) < 50:
            return
        
        # 최근 100개 데이터로 상관관계 계산
        recent_rewards = np.array(self.reward_performance_tracker['rewards'][-100:])
        recent_returns = np.array(self.reward_performance_tracker['returns'][-100:])
        
        # 상관계수 계산
        if recent_rewards.std() > 1e-8 and recent_returns.std() > 1e-8:
            correlation = np.corrcoef(recent_rewards, recent_returns)[0, 1]
            
            if not np.isnan(correlation):
                self.logger.debug(f"보상-성과 상관성: {correlation:.3f} (목표: >0)")
                
                # 경고 발생 (음의 상관관계)
                if correlation < -0.1:
                    self.logger.warning(f"보상-성과 역상관 감지: {correlation:.3f}")
                elif correlation > 0.3:
                    self.logger.info(f"보상-성과 정렬 양호: {correlation:.3f}")
        
        return
    
    def get_portfolio_metrics(self, num_trials: int = 1) -> Dict[str, float]:
        """포트폴리오 성과 메트릭 계산 (DSR 포함)"""
        if len(self.return_history) == 0:
            return {}
        
        returns = np.array(self.return_history)
        
        # DSR을 포함한 종합적인 메트릭 계산
        comprehensive_metrics = calculate_comprehensive_metrics(returns, num_trials)
        
        # 추가적인 포트폴리오 특화 메트릭
        additional_metrics = {
            'portfolio_value': self.portfolio_value,
            'total_cost': sum(self.cost_history) if self.cost_history else 0.0,
            'average_concentration': np.mean([calculate_concentration_index(w) for w in self.weight_history]) if self.weight_history else 0.0
        }
        
        # 메트릭 결합
        all_metrics = {**comprehensive_metrics, **additional_metrics}
        
        return all_metrics
        
    
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
        """최종 실행 가중치 검증 통계 요약"""
        if not self.weight_validation_window:
            return "최종 실행 가중치 검증 통계 없음"
        
        # 슬라이딩 윈도우 통계
        recent_count = len(self.weight_validation_window)
        negative_count = sum(1 for r in self.weight_validation_window if r['has_negative'])
        invalid_count = sum(1 for r in self.weight_validation_window if r['has_invalid'])
        sum_invalid_count = sum(1 for r in self.weight_validation_window if not r['sum_valid'])
        
        negative_rate = negative_count / recent_count if recent_count > 0 else 0.0
        invalid_rate = invalid_count / recent_count if recent_count > 0 else 0.0
        
        # 누적 통계
        total_validations = self.weight_validation_stats['total_validations']
        
        return (f"최종 실행 가중치 검증 통계: "
                f"슬라이딩(최근 {recent_count}회) - "
                f"음수: {negative_count}회 ({negative_rate:.1%}), "
                f"무효값: {invalid_count}회 ({invalid_rate:.1%}), "
                f"합불일치: {sum_invalid_count}회 | "
                f"누적: {total_validations}회")
    
    def _update_and_get_sharpe_reward(self, portfolio_return: float) -> float:
        """
        EMA 기반 Sharpe 비율 추정 및 보상 항목 계산
        
        Args:
            portfolio_return: 현재 포트폴리오 수익률
            
        Returns:
            sharpe_reward: Sharpe 기반 보상 항목
        """
        tracker = self.sharpe_tracker
        decay = tracker['decay']
        
        if tracker['count'] == 0:
            # 초기화
            tracker['return_ema'] = portfolio_return
            tracker['volatility_ema'] = (portfolio_return ** 2)
        else:
            # EMA 업데이트
            tracker['return_ema'] = decay * tracker['return_ema'] + (1 - decay) * portfolio_return
            return_deviation = (portfolio_return - tracker['return_ema'])
            tracker['volatility_ema'] = decay * tracker['volatility_ema'] + (1 - decay) * (return_deviation ** 2)
        
        tracker['count'] += 1
        
        # Sharpe 비율 추정 (분모 안정화)
        mu_t = tracker['return_ema']
        sigma_t = max(np.sqrt(tracker['volatility_ema']), 1e-6)  # 0 방지
        sharpe_estimate = mu_t / sigma_t
        
        # Sharpe 보상 (가중치 상향으로 성과 정렬 강화)
        lambda_s = 0.05  # 가중치 상향 (0.02 → 0.05)
        sharpe_reward = lambda_s * np.clip(sharpe_estimate, -2.0, 2.0)
        
        return sharpe_reward