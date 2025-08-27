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
        
        # 거래비용 최적화 파라미터 (config.py에서 관리)
        self.no_trade_band = NO_TRADE_BAND              # 노-트레이드 밴드
        self.max_turnover = MAX_TURNOVER                # 최대 턴오버
        
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
        
        # 정보 딕셔너리 (최종 실행 가중치 기준)
        info = {
            'portfolio_value': self.portfolio_value,
            'portfolio_return': actual_return,
            'transaction_cost': transaction_cost,
            'weight_change': weight_change,
            'concentration': calculate_concentration_index(final_weights),
            'final_weights': final_weights.copy()  # 최종 실행 가중치 추가
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
        """변동성 타깃팅 오버레이 적용"""
        if len(self.return_history) < self.volatility_window:
            # 초기에는 변동성 타깃팅 미적용
            return weights
        
        # 최근 변동성 추정 (일일 기준)
        recent_returns = np.array(self.return_history[-self.volatility_window:])
        current_volatility = recent_returns.std()
        
        # 연간 변동성으로 환산
        annualized_volatility = current_volatility * np.sqrt(252)
        
        # 레버리지 스케일 계산
        if annualized_volatility > 1e-6:
            leverage_scale = self.target_volatility / annualized_volatility
        else:
            leverage_scale = 1.0
        
        # 레버리지 클리핑
        leverage_scale = np.clip(leverage_scale, self.min_leverage, self.max_leverage)
        
        # 가중치 스케일링
        scaled_weights = weights * leverage_scale
        
        # 재정규화 (합이 1이 되도록)
        if scaled_weights.sum() > 0:
            scaled_weights = scaled_weights / scaled_weights.sum()
        else:
            scaled_weights = weights  # 백업
        
        # 변동성 타깃팅 로그 (config.py 간격)
        if self.current_step % CORRELATION_CHECK_INTERVAL == 0:
            self.logger.debug(
                f"변동성 타깃팅 (step {self.current_step}): "
                f"현재_vol={annualized_volatility:.3f}, 목표_vol={self.target_volatility:.3f}, "
                f"레버리지={leverage_scale:.2f}"
            )
        
        if not hasattr(self, '_volatility_logged'):
            self.logger.info(f"변동성 타깃팅 활성화: 목표={self.target_volatility:.1%}, 윈도우={self.volatility_window}일")
            self._volatility_logged = True
        
        return scaled_weights
    
    def _apply_trading_constraints(self, new_weights: np.ndarray) -> np.ndarray:
        """노-트레이드 밴드 및 턴오버 캡 적용"""
        current_weights = self.weights
        
        # 1. 노-트레이드 밴드 적용
        weight_changes = new_weights - current_weights
        
        # 각 자산별로 노-트레이드 밴드 적용
        small_change_mask = np.abs(weight_changes) <= self.no_trade_band
        constrained_weights = np.where(small_change_mask, current_weights, new_weights)
        
        # 2. 턴오버 캡 적용
        total_turnover = np.abs(constrained_weights - current_weights).sum()
        
        if total_turnover > self.max_turnover:
            # 턴오버가 제한을 초과하면 스케일링
            scale_factor = self.max_turnover / total_turnover
            # 변화량을 스케일링하고 현재 가중치에 더함
            scaled_changes = (constrained_weights - current_weights) * scale_factor
            constrained_weights = current_weights + scaled_changes
        
        # 3. 재정규화 (합이 1이 되도록)
        if constrained_weights.sum() > 0:
            constrained_weights = constrained_weights / constrained_weights.sum()
        else:
            constrained_weights = new_weights  # 백업
        
        # 4. 로깅 (config.py 간격)
        if self.current_step % CORRELATION_CHECK_INTERVAL == 0:
            original_turnover = np.abs(new_weights - current_weights).sum()
            final_turnover = np.abs(constrained_weights - current_weights).sum()
            band_applied = np.sum(small_change_mask)
            
            self.logger.debug(
                f"거래 제약 (step {self.current_step}): "
                f"원래_턴오버={original_turnover:.3f}, 최종_턴오버={final_turnover:.3f}, "
                f"밴드적용={band_applied}/{len(new_weights)}"
            )
        
        if not hasattr(self, '_trading_constraints_logged'):
            self.logger.info(f"거래 제약 활성화: 노트레이드밴드={self.no_trade_band:.1%}, 최대턴오버={self.max_turnover:.1%}")
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
        보상 함수 (Sharpe proxy 통합)
        
        r_t = log(V_{t+1}/V_t) - λ_risk*σ_t - λ_tc*||Δw_t||_1 + λ_S*Sharpe_EMA
        """
        # 1. 기본 로그 수익률 보상 (안전한 로그 계산)
        log_return = np.log(max(1 + portfolio_return, 1e-12))
        
        # 2. Sharpe proxy EMA 업데이트 및 보상 항목
        sharpe_reward = self._update_and_get_sharpe_reward(portfolio_return)
        
        # 3. 리스크 페널티 (최근 변동성 추정)
        if len(self.return_history) >= 5:
            recent_volatility = np.std(self.return_history[-5:])
        else:
            recent_volatility = abs(portfolio_return)  # 단일 관측치로 추정
        
        risk_penalty = RISK_PENALTY_WEIGHT * recent_volatility  # λ_risk (config.py)
        
        # 4. 거래비용 페널티 (가중치 변화)
        if len(self.weight_history) > 0:
            prev_weights = self.weight_history[-1]
            weight_change = np.abs(weights - prev_weights).sum()
        else:
            weight_change = 0.0
        
        transaction_penalty = TRANSACTION_PENALTY_WEIGHT * weight_change  # λ_tc (config.py)
        
        # 5. 집중도(HHI) 페널티 추가
        hhi = np.sum(weights ** 2)  # Herfindahl-Hirschman Index
        equal_weight_hhi = 1.0 / self.n_assets  # 균등분산 시 HHI
        concentration_penalty = HHI_PENALTY_WEIGHT * max(0, hhi - equal_weight_hhi)  # λ_hhi (config.py)
        
        # 6. 원시 보상 계산 (HHI 페널티 포함)
        raw_reward = log_return - risk_penalty - transaction_penalty - concentration_penalty + sharpe_reward
        
        # 7. tanh 바운딩 적용 (Q-value 폭주 방지)
        r_max, tau = 0.05, 0.05  # 보수적 스케일 설정
        bounded_reward = r_max * np.tanh(raw_reward / tau)
        
        # 8. 고정 스케일러 정규화 (바운딩된 보상 사용)
        normalized_reward = self.reward_normalizer.normalize(bounded_reward)
        
        # 8. 보상-성과 상관성 추적
        self.reward_performance_tracker['rewards'].append(normalized_reward)
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
            current_sharpe_estimate = self.sharpe_tracker['return_ema'] / max(np.sqrt(self.sharpe_tracker['volatility_ema']), 1e-6)
            self.logger.debug(
                f"보상 구성 (step {self.current_step}): "
                f"log_ret={log_return:.4f}, risk_pen={risk_penalty:.4f}, "
                f"tc_pen={transaction_penalty:.4f}, hhi_pen={concentration_penalty:.4f}, "
                f"sharpe_rew={sharpe_reward:.4f}, HHI={hhi:.3f}, "
                f"raw_rew={raw_reward:.4f}, bounded_rew={bounded_reward:.4f}, norm_rew={normalized_reward:.4f}"
            )
        
        if not hasattr(self, '_reward_logged'):
            self.logger.info(
                "보상 함수 대폭 개선: log-return 기반, tanh 바운딩 (r_max=0.05), "
                "고정 정규화, HHI 페널티, Sharpe proxy 통합"
            )
            self._reward_logged = True
        
        # 안전한 타입 변환 (스칼라 값 보장)
        if hasattr(normalized_reward, 'item'):
            return float(normalized_reward.item())
        else:
            return float(normalized_reward)
    
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