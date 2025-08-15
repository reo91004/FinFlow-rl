# core/reward.py

import numpy as np
import pandas as pd
from typing import Dict, Optional
from collections import deque
from constant import *
from utils.logger import BIPDLogger

class RewardCalculator:
    """단순화된 보상 함수 - 클리핑 100% 문제 해결"""
    
    def __init__(self, 
                 lookback_window=20,
                 transaction_cost_rate=0.001):
        self.lookback_window = lookback_window
        self.transaction_cost_rate = transaction_cost_rate
        
        # 로거 초기화
        self.logger = BIPDLogger().get_reward_logger()
        
        # 성과 추적
        self.return_history = deque(maxlen=252)
        self.weight_history = deque(maxlen=5)
        
        # 보상 정규화 (매우 보수적)
        self.reward_history = deque(maxlen=1000)
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.update_count = 0
        
        # 에피소드별 보상 통계 (로깅 최적화)
        self.episode_rewards = []
        self.episode_step_count = 0

    def calculate_comprehensive_reward(self, 
                                     current_return: float,
                                     previous_weights: np.ndarray,
                                     current_weights: np.ndarray,
                                     market_features: np.ndarray,
                                     crisis_level: float) -> Dict[str, float]:
        """
        단순화된 3-컴포넌트 보상 (스케일 균형)
        목표: 클리핑 비율 20% 미만
        """
        
        # 1. 수익률 보상 (더욱 축소)
        return_reward = current_return * 0.5  # 2→0.5로 추가 축소
        
        # 2. 위험 조정 보상 (샤프 기반, 스케일 축소)
        risk_adjusted_reward = self._calculate_simple_sharpe() * 0.1  # 추가 스케일 축소
        
        # 3. 거래 비용 패널티 (스케일 축소)
        transaction_penalty = self._calculate_simple_transaction_cost(
            previous_weights, current_weights
        ) * 0.1  # 추가 스케일 축소
        
        # 단순 가중 평균 (복잡한 계산 제거)
        total_reward = (
            0.7 * return_reward +           # 수익률 중심
            0.2 * risk_adjusted_reward +    # 위험 조정
            0.1 * transaction_penalty       # 거래 비용
        )
        
        # 정규화 완전 제거 (클리핑 문제 방지)
        normalized_reward = total_reward
        
        # 매우 관대한 클리핑 (거의 클리핑되지 않도록)
        final_reward = np.clip(normalized_reward, -0.5, 0.5)
        
        # 에피소드 보상 통계 누적
        self.episode_rewards.append({
            'return': return_reward,
            'risk_adjusted': risk_adjusted_reward,
            'transaction': transaction_penalty,
            'total': total_reward,
            'final': final_reward
        })
        self.episode_step_count += 1
        
        # 50스텝마다 보상 요약 로깅
        if self.episode_step_count % 50 == 0:
            self._log_reward_summary()
        
        # 이력 업데이트
        self._update_history(current_return, current_weights, final_reward)
        
        return {
            "total_reward": float(final_reward),
            "raw_total_reward": float(total_reward),
            "return_reward": float(return_reward),
            "risk_adjusted_reward": float(risk_adjusted_reward),
            "transaction_cost_penalty": float(transaction_penalty),
            "target_reward": 0.0,  # 제거
            "adaptation_reward": 0.0,  # 제거
            "concentration_penalty": 0.0,  # 제거
        }

    def _calculate_simple_sharpe(self) -> float:
        """단순화된 샤프 보상 (스케일 축소)"""
        if len(self.return_history) < self.lookback_window:
            return 0.0
            
        returns = np.array(list(self.return_history)[-self.lookback_window:])
        if len(returns) < 2:
            return 0.0
            
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return < 1e-8:
            return 0.0
            
        sharpe = mean_return / std_return * np.sqrt(252)
        return np.tanh(sharpe) * 1.0  # 10→1로 10배 축소

    def _calculate_simple_transaction_cost(self, prev_weights, curr_weights) -> float:
        """단순화된 거래 비용 (스케일 축소)"""
        if prev_weights is None or len(prev_weights) != len(curr_weights):
            return 0.0
            
        turnover = np.sum(np.abs(curr_weights - prev_weights)) / 2
        return -turnover * self.transaction_cost_rate * 100  # 1000→100으로 10배 축소

    def _conservative_normalize(self, reward: float) -> float:
        """매우 보수적 정규화 (클리핑 방지)"""
        try:
            alpha = min(0.05, 1.0 / (self.update_count + 1))  # 매우 느린 업데이트
            
            self.reward_mean = (1 - alpha) * self.reward_mean + alpha * reward
            
            if self.update_count > 1:
                variance_update = alpha * (reward - self.reward_mean) ** 2
                current_variance = self.reward_std ** 2
                new_variance = (1 - alpha) * current_variance + variance_update
                self.reward_std = np.sqrt(max(new_variance, 1e-6))
            
            # 보수적 Z-score (±1.5σ)
            if self.reward_std > 1e-6:
                normalized = (reward - self.reward_mean) / self.reward_std
                return np.clip(normalized, -1.5, 1.5)
            else:
                return reward
                
        except (ValueError, ZeroDivisionError):
            return reward

    def _update_history(self, current_return, current_weights, reward):
        """이력 업데이트"""
        self.return_history.append(current_return)
        if current_weights is not None:
            self.weight_history.append(current_weights.copy())
        self.reward_history.append(reward)
        self.update_count += 1
    
    def _log_reward_summary(self):
        """50스텝마다 보상 통계 요약 로깅"""
        if not self.episode_rewards:
            return
            
        recent_rewards = self.episode_rewards[-50:] if len(self.episode_rewards) >= 50 else self.episode_rewards
        
        # 각 컴포넌트별 통계
        returns = [r['return'] for r in recent_rewards]
        finals = [r['final'] for r in recent_rewards]
        totals = [r['total'] for r in recent_rewards]
        
        self.logger.debug(f"보상 요약 (최근 {len(recent_rewards)}스텝): "
                         f"평균 최종보상={np.mean(finals):.3f}, "
                         f"평균 수익보상={np.mean(returns):.3f}, "
                         f"범위=[{np.min(finals):.2f}, {np.max(finals):.2f}], "
                         f"클리핑 비율={sum(1 for t, f in zip(totals, finals) if abs(t-f) > 0.01) / len(totals):.2%}")
    
    def reset_episode_stats(self):
        """에피소드 통계 초기화 (에피소드 종료 시 호출)"""
        self.episode_rewards = []
        self.episode_step_count = 0

    # 기존 호환성을 위한 메서드들
    def get_performance_metrics(self):
        if len(self.return_history) < 2:
            return {}

        returns = np.array(list(self.return_history))

        return {
            "annualized_return": np.mean(returns) * 252,
            "annualized_volatility": np.std(returns) * np.sqrt(252),
            "sharpe_ratio": np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252),
            "max_drawdown": 0.0,  # 단순화
            "total_return": np.prod(1 + returns) - 1,
            "win_rate": (
                len(returns[returns > 0]) / len(returns) if len(returns) > 0 else 0
            ),
        }