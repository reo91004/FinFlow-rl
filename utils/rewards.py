# utils/rewards.py

import numpy as np
from typing import Optional, Dict, Tuple, Union


def downside_vol(returns: np.ndarray, ref: float = 0.0) -> float:
    """
    다운사이드 변동성 계산 (Handover v2)
    
    Args:
        returns: 수익률 배열
        ref: 기준 수익률 (기본값: 0.0)
        
    Returns:
        float: 다운사이드 변동성
    """
    neg = np.minimum(returns - ref, 0.0)
    return np.sqrt(np.mean(neg**2) + 1e-12)


def compute_hhi(w: np.ndarray) -> float:
    """
    Herfindahl-Hirschman Index (HHI) 계산
    
    Args:
        w: 가중치 벡터
        
    Returns:
        float: HHI 값
    """
    w = np.clip(w, 1e-12, 1.0)
    w = w / w.sum()  # 정규화
    return np.sum(w**2)


def estimate_cvar(samples: np.ndarray, alpha: float = 0.05) -> float:
    """
    조건부 위험값(CVaR) 추정
    
    Args:
        samples: 손실 샘플 (음수 수익률)
        alpha: CVaR 분위수 (기본값: 0.05 = 5%)
        
    Returns:
        float: CVaR 값 (양수 페널티로 반환)
    """
    if len(samples) == 0:
        return 0.0
        
    q = np.quantile(samples, alpha)
    tail_losses = samples[samples <= q]
    
    if len(tail_losses) == 0:
        return 0.0
        
    return -np.mean(tail_losses)  # 손실을 양수 페널티로 변환


def compute_max_drawdown(returns: np.ndarray) -> float:
    """
    최대 낙폭(MDD) 계산
    
    Args:
        returns: 수익률 배열
        
    Returns:
        float: 최대 낙폭 (양수)
    """
    if len(returns) == 0:
        return 0.0
        
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return float(-np.min(drawdown))  # 양수로 반환


def compose_reward(
    log_ret: float,
    w: np.ndarray,
    w_prev: np.ndarray,
    vol_down: float,
    mdd: float,
    cvar_alpha: float,
    lambdas: Tuple[float, float, float, float],
    cvar_samples: Optional[np.ndarray] = None,
    return_components: bool = False
) -> Union[float, Tuple[float, Dict[str, float]]]:
    """
    통합 보상 함수 (Handover v2)
    
    Args:
        log_ret: 로그 수익률
        w: 현재 가중치
        w_prev: 이전 가중치
        vol_down: 다운사이드 변동성
        mdd: 최대 낙폭
        cvar_alpha: CVaR 분위수
        lambdas: 페널티 가중치 튜플 (dd, vol, turn, hhi)
        cvar_samples: CVaR 계산용 샘플 (선택적)
        return_components: 구성 요소별 반환 여부
        
    Returns:
        float or (float, dict): 총 보상 또는 (총 보상, 구성요소 딕셔너리)
    """
    lam_dd, lam_vol, lam_turn, lam_hhi = lambdas
    
    # 턴오버 계산
    turn = np.sum(np.abs(w - w_prev))
    
    # HHI 계산
    hhi = compute_hhi(w)
    
    # CVaR 페널티 계산
    if cvar_samples is not None and len(cvar_samples) > 0:
        cvar_pen = estimate_cvar(cvar_samples, cvar_alpha)
    else:
        cvar_pen = 0.0
    
    # 총 보상 계산
    reward = log_ret - lam_vol * vol_down - lam_dd * mdd - lam_turn * turn - lam_hhi * hhi - cvar_pen
    
    if return_components:
        components = {
            'log_return': log_ret,
            'vol_penalty': lam_vol * vol_down,
            'dd_penalty': lam_dd * mdd,
            'turnover_penalty': lam_turn * turn,
            'hhi_penalty': lam_hhi * hhi,
            'cvar_penalty': cvar_pen,
            'turnover': turn,
            'hhi': hhi,
            'vol_down': vol_down,
            'mdd': mdd,
            'cvar': cvar_pen
        }
        return reward, components
    else:
        return reward


def lagrangian_constraint_update(
    constraint_value: float,
    constraint_limit: float,
    current_lambda: float,
    learning_rate: float = 1e-3,
    lambda_min: float = 0.0,
    lambda_max: float = 1.0
) -> float:
    """
    라그랑주 제약 조건 업데이트 (CPO 스타일)
    
    Args:
        constraint_value: 현재 제약 조건 값
        constraint_limit: 제약 조건 한계값
        current_lambda: 현재 라그랑주 승수
        learning_rate: 학습률
        lambda_min: 라그랑주 승수 하한
        lambda_max: 라그랑주 승수 상한
        
    Returns:
        float: 업데이트된 라그랑주 승수
    """
    # 제약 조건 위반 정도 계산
    constraint_violation = max(0, constraint_value - constraint_limit)
    
    # 라그랑주 승수 업데이트: λ ← [λ + η·(metric - limit)]₊
    new_lambda = current_lambda + learning_rate * constraint_violation
    
    # 범위 제한
    new_lambda = np.clip(new_lambda, lambda_min, lambda_max)
    
    return float(new_lambda)


class RewardComponentTracker:
    """보상 구성 요소 추적기"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.reset()
    
    def reset(self):
        """통계 초기화"""
        self.components_history = []
        self.total_rewards = []
    
    def update(self, reward: float, components: Dict[str, float]) -> None:
        """보상 구성 요소 업데이트"""
        self.total_rewards.append(reward)
        self.components_history.append(components.copy())
        
        # 윈도우 크기 제한
        if len(self.components_history) > self.window_size:
            self.components_history.pop(0)
            self.total_rewards.pop(0)
    
    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """구성 요소별 통계 반환"""
        if not self.components_history:
            return {}
        
        stats = {}
        component_names = self.components_history[0].keys()
        
        for comp_name in component_names:
            values = [comp[comp_name] for comp in self.components_history]
            stats[comp_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'recent_mean': np.mean(values[-100:]) if len(values) >= 100 else np.mean(values)
            }
        
        # 전체 보상 통계 추가
        stats['total_reward'] = {
            'mean': np.mean(self.total_rewards),
            'std': np.std(self.total_rewards),
            'min': np.min(self.total_rewards),
            'max': np.max(self.total_rewards),
            'recent_mean': np.mean(self.total_rewards[-100:]) if len(self.total_rewards) >= 100 else np.mean(self.total_rewards)
        }
        
        return stats
    
    def get_component_contribution(self) -> Dict[str, float]:
        """각 구성 요소의 기여도 계산 (절댓값 기준)"""
        if not self.components_history:
            return {}
        
        contributions = {}
        component_names = self.components_history[0].keys()
        
        for comp_name in component_names:
            values = [abs(comp[comp_name]) for comp in self.components_history]
            contributions[comp_name] = np.mean(values)
        
        # 정규화 (비율로 변환)
        total_contribution = sum(contributions.values())
        if total_contribution > 0:
            for comp_name in contributions:
                contributions[comp_name] /= total_contribution
        
        return contributions


# 편의 함수들
def calculate_downside_metrics(returns: np.ndarray) -> Dict[str, float]:
    """다운사이드 리스크 메트릭 계산"""
    if len(returns) == 0:
        return {'downside_vol': 0.0, 'max_drawdown': 0.0}
    
    return {
        'downside_vol': downside_vol(returns),
        'max_drawdown': compute_max_drawdown(returns)
    }


def safe_cvar_estimation(samples: np.ndarray, alpha: float = 0.05, min_samples: int = 20) -> float:
    """안전한 CVaR 추정 (충분한 샘플이 없으면 0 반환)"""
    if len(samples) < min_samples:
        return 0.0
    return estimate_cvar(samples, alpha)