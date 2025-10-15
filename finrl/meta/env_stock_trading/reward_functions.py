# finrl/meta/env_stock_trading/reward_functions.py

"""
리스크 민감 보상 함수 모듈

Phase 3: DSR (Differential Sharpe Ratio) + CVaR 기반 보상
Sharpe ≥ 1.0 달성을 위한 직접적 신호 제공

이론적 근거:
- DSR: Moody & Saffell (1998), "Learning to Trade via Direct RL"
- CVaR: Tamar et al. (2015), "Policy Gradient with CVaR Constraints"
- 온라인 추정: Welford (1962) 온라인 평균/분산 알고리즘
"""

import numpy as np
from collections import deque


class DifferentialSharpeRatio:
    """
    온라인 Differential Sharpe Ratio (DSR) 계산

    Sharpe의 시간 미분 근사를 통해 직접 Sharpe 상승 방향 그라디언트 제공.

    수학적 정의:
    - μ_t = β·μ_{t-1} + (1-β)·R_t (이동평균)
    - σ²_t = β·σ²_{t-1} + (1-β)·(R_t - μ_t)² (이동분산)
    - S_t = μ_t / σ_t (Sharpe)
    - ΔS_t ≈ z_t - (S_{t-1}/2)·z_t², z_t = (R_t - μ_{t-1}) / σ_{t-1}

    Args:
        beta: EMA 계수 (Phase 1: 0.95 for faster response to market changes)
        epsilon: 분모 안정화 (σ=0 방지)
    """

    def __init__(self, beta: float = 0.95, epsilon: float = 1e-8):  # Phase 1: 0.99 → 0.95
        self.beta = beta
        self.epsilon = epsilon

        # 상태 변수
        self.mu = 0.0  # 평균 수익
        self.sigma_sq = 1.0  # 분산 (초기값 1.0)
        self.sharpe = 0.0  # 현재 Sharpe

    def update(self, return_t: float) -> float:
        """
        새 리턴 관찰 시 DSR 증분 계산

        Args:
            return_t: 시점 t의 로그수익 (예: log(V_t/V_{t-1}))

        Returns:
            dsr: Sharpe 증분 (보상 보너스로 사용)
        """
        # 이전 통계 저장
        mu_prev = self.mu
        sigma_prev = np.sqrt(self.sigma_sq + self.epsilon)

        # 온라인 평균 업데이트
        self.mu = self.beta * self.mu + (1.0 - self.beta) * return_t

        # 온라인 분산 업데이트 (Welford)
        delta = return_t - self.mu
        self.sigma_sq = self.beta * self.sigma_sq + (1.0 - self.beta) * delta**2

        # 현재 Sharpe
        sigma_curr = np.sqrt(self.sigma_sq + self.epsilon)
        self.sharpe = self.mu / sigma_curr

        # DSR 증분 계산
        if sigma_prev > self.epsilon:
            z = (return_t - mu_prev) / sigma_prev
            dsr = z - 0.5 * self.sharpe * z**2
        else:
            # 초기 단계 (통계 불안정)
            dsr = 0.0

        return dsr

    def reset(self):
        """에피소드 종료 시 상태 초기화"""
        self.mu = 0.0
        self.sigma_sq = 1.0
        self.sharpe = 0.0


class CVaREstimator:
    """
    온라인 Conditional Value at Risk (CVaR) 추정

    최근 H 스텝의 리턴 분포에서 α% 하위 평균 계산.
    꼬리 위험 제어를 통해 Sharpe 개선 및 downside 보호.

    수학적 정의:
    - VaR_α = percentile(returns, α)
    - CVaR_α = E[R | R ≤ VaR_α]

    Args:
        alpha: CVaR 분위수 (0.05 = 5% 하위)
        window: 추정 윈도우 크기 (스텝)
    """

    def __init__(self, alpha: float = 0.05, window: int = 50):
        self.alpha = alpha
        self.window = window
        self.buffer = deque(maxlen=window)

    def update(self, return_t: float) -> float:
        """
        새 리턴 관찰 시 CVaR 추정

        Args:
            return_t: 시점 t의 로그수익

        Returns:
            cvar: CVaR_α 추정값 (음수 = 손실)
        """
        self.buffer.append(return_t)

        # 초기 워밍업 (충분한 샘플 확보 전)
        if len(self.buffer) < self.window // 2:
            return 0.0

        # α% VaR/CVaR 계산
        sorted_returns = np.sort(self.buffer)
        cutoff_idx = max(1, int(self.alpha * len(sorted_returns)))
        cvar = np.mean(sorted_returns[:cutoff_idx])

        return cvar

    def reset(self):
        """에피소드 종료 시 버퍼 초기화"""
        self.buffer.clear()


class RiskSensitiveReward:
    """
    DSR + CVaR 혼합 보상 래퍼 (Phase 3.5: 스케일 정규화)

    기본 로그수익에 Sharpe 지향 보너스(DSR)와 꼬리위험 패널티(CVaR) 혼합.

    Phase 3.5 개선:
    - EMA 기반 정규화: DSR/CVaR를 각자의 평균 절댓값으로 나눠 스케일 통일
    - 보상 클리핑: [-0.01, 0.01] 범위로 제한

    최종 보상:
    r'_t = r_t + λ_S·(ΔS_t / ||ΔS||_EMA) - λ_CVaR·(|CVaR| / ||CVaR||_EMA)
    클리핑: r'_t ∈ [-0.01, 0.01]

    Args:
        lambda_dsr: DSR 가중치 (권장: 0.1~0.2)
        lambda_cvar: CVaR 가중치 (권장: 0.05~0.1)
        dsr_beta: DSR 이동평균 계수
        cvar_alpha: CVaR 분위수
        cvar_window: CVaR 추정 윈도우
        normalization_window: EMA 정규화 윈도우 (default: 30)
    """

    def __init__(
        self,
        lambda_dsr: float = 0.1,
        lambda_cvar: float = 0.05,
        dsr_beta: float = 0.99,
        cvar_alpha: float = 0.05,
        cvar_window: int = 50,
        normalization_window: int = 30,
    ):
        self.lambda_dsr = lambda_dsr
        self.lambda_cvar = lambda_cvar

        self.dsr = DifferentialSharpeRatio(beta=dsr_beta)
        self.cvar = CVaREstimator(alpha=cvar_alpha, window=cvar_window)

        # Phase 3.5: 정규화 분모 (EMA)
        self.norm_window = normalization_window
        self.norm_beta = 1.0 - 1.0 / normalization_window

        # 각 신호의 평균 절댓값 (초기값 1.0)
        self.log_return_ema = 1.0
        self.dsr_ema = 1.0
        self.cvar_ema = 1.0

    def compute(self, basic_return: float) -> tuple[float, dict]:
        """
        리스크 민감 보상 계산 (Phase 3.5: 정규화 + 클리핑)

        Args:
            basic_return: 기본 로그수익 (log(V_t/V_{t-1}))

        Returns:
            reward: 혼합 보상 (정규화 + 클리핑 적용)
            info: 디버깅 정보 (dsr, cvar, 정규화 상태)
        """
        # DSR 보너스
        dsr_bonus = self.dsr.update(basic_return)

        # CVaR 패널티
        cvar_value = self.cvar.update(basic_return)
        cvar_penalty = abs(cvar_value) if cvar_value < 0 else 0.0

        # Phase 3.5: EMA 정규화 분모 업데이트
        self.log_return_ema = (
            self.norm_beta * self.log_return_ema
            + (1.0 - self.norm_beta) * abs(basic_return)
        )
        self.dsr_ema = (
            self.norm_beta * self.dsr_ema + (1.0 - self.norm_beta) * abs(dsr_bonus)
        )
        self.cvar_ema = (
            self.norm_beta * self.cvar_ema
            + (1.0 - self.norm_beta) * abs(cvar_value)
        )

        # Phase 3.5: 정규화 (스케일 통일)
        # 분모에 1e-6 추가하여 0으로 나누기 방지
        norm_dsr = dsr_bonus / (self.dsr_ema + 1e-6)
        norm_cvar = cvar_penalty / (self.cvar_ema + 1e-6)

        # 혼합 보상 (정규화된 신호 사용)
        reward = basic_return + self.lambda_dsr * norm_dsr - self.lambda_cvar * norm_cvar

        # Phase 3.5: 클리핑 [-0.01, 0.01]
        reward_clipped = np.clip(reward, -0.01, 0.01)

        # 디버깅 정보
        info = {
            "dsr_bonus": dsr_bonus,
            "cvar_value": cvar_value,
            "cvar_penalty": cvar_penalty,
            "sharpe_online": self.dsr.sharpe,
            # Phase 3.5 추가 정보
            "norm_dsr": norm_dsr,
            "norm_cvar": norm_cvar,
            "reward_pre_clip": reward,
            "reward_clipped": reward_clipped,
            "log_return_ema": self.log_return_ema,
            "dsr_ema": self.dsr_ema,
            "cvar_ema": self.cvar_ema,
        }

        return reward_clipped, info

    def reset(self):
        """에피소드 종료 시 상태 초기화 (Phase 3.5: EMA도 리셋)"""
        self.dsr.reset()
        self.cvar.reset()

        # Phase 3.5: 정규화 분모 초기화
        self.log_return_ema = 1.0
        self.dsr_ema = 1.0
        self.cvar_ema = 1.0


class AdaptiveRiskReward:
    """
    Phase-H1: Adaptive Risk-Aware Reward with Crisis Sensitivity

    Log-return base + Adaptive risk bonus + Transaction cost penalty

    수식:
    r_t = log(V_t/V_{t-1}) + κ_S(c)·ΔSharpe - κ_C(c)·CVaR - μ·turnover

    여기서:
    - log(V_t/V_{t-1}): 항상 non-zero gradient를 보장하는 base return
    - κ_S(c) = λ_S + g_S·c (Sharpe 보너스 게이팅, g_S < 0)
    - κ_C(c) = λ_C + g_C·c (CVaR 패널티 게이팅, g_C > 0 권장)
    - ΔSharpe: DSR (Differential Sharpe Ratio)
    - CVaR: 꼬리 위험 (음수)
    - turnover: 거래 비용 (실행가중 기반)

    특징:
    1. Base log-return이 항상 존재 → gradient vanishing 해결
    2. Crisis-aware κ → 위기 시 자동으로 risk-averse 강화
    3. Direct CVaR penalty → Risk-aware value learning

    Args:
        lambda_sharpe: ΔSharpe 기본 가중치 (κ_S, default: 0.20)
        lambda_cvar: CVaR penalty 기본 가중치 (κ_C, default: 0.40)
        lambda_turnover: Turnover penalty 가중치 (μ, default: 0.0)
        crisis_gain_sharpe: Sharpe 게이팅에 대한 위기 계수 g_S (default: -0.15)
        crisis_gain_cvar: CVaR 게이팅에 대한 위기 계수 g_C (default: 0.25)
        dsr_beta: DSR 이동평균 계수 (default: 0.92)
        cvar_alpha: CVaR 분위수 (default: 0.05)
        cvar_window: CVaR 추정 윈도우 (default: 40)
    """

    def __init__(
        self,
        lambda_sharpe: float = 0.20,
        lambda_cvar: float = 0.40,
        lambda_turnover: float = 0.0,
        crisis_gain_sharpe: float = -0.15,
        crisis_gain_cvar: float = 0.25,
        dsr_beta: float = 0.92,
        cvar_alpha: float = 0.05,
        cvar_window: int = 40,
    ):
        self.lambda_sharpe_base = lambda_sharpe
        self.lambda_cvar_base = lambda_cvar
        self.lambda_cvar = lambda_cvar  # backward compatibility (logging)
        self.lambda_turnover = lambda_turnover
        self.crisis_gain_sharpe = crisis_gain_sharpe
        self.crisis_gain_cvar = crisis_gain_cvar

        # DSR and CVaR estimators
        self.dsr = DifferentialSharpeRatio(beta=dsr_beta)
        self.cvar = CVaREstimator(alpha=cvar_alpha, window=cvar_window)

        # Crisis level (updated externally via callback)
        self.crisis_level = 0.5  # Default: neutral

        # Turnover tracking
        self.prev_weights = None

    def set_crisis_level(self, crisis_level: float):
        """
        Policy callback에서 crisis_level을 업데이트

        Args:
            crisis_level: 0 (평시) ~ 1 (위기)
        """
        self.crisis_level = np.clip(crisis_level, 0.0, 1.0)

    def compute(
        self,
        basic_return: float,
        current_weights: np.ndarray = None
    ) -> tuple[float, dict]:
        """
        Phase-H1 Adaptive Risk-Aware Reward 계산

        Args:
            basic_return: log(V_t/V_{t-1})
            current_weights: 현재 포트폴리오 가중치 (turnover 계산용, optional)

        Returns:
            reward: 최종 보상
            info: 디버깅 정보
        """
        # 1. Base log-return (항상 non-zero gradient)
        log_return = basic_return

        # 2. ΔSharpe (DSR)
        delta_sharpe = self.dsr.update(basic_return)

        # 3. CVaR (꼬리 위험)
        cvar_value = self.cvar.update(basic_return)
        cvar_penalty = abs(cvar_value) if cvar_value < 0 else 0.0

        # 4. Adaptive 게이팅 (Sharpe ↘, CVaR ↗ when crisis ↑)
        kappa_sharpe = max(
            0.0, self.lambda_sharpe_base + self.crisis_gain_sharpe * self.crisis_level
        )
        kappa_cvar = max(
            0.0, self.lambda_cvar_base + self.crisis_gain_cvar * self.crisis_level
        )

        # 5. Risk bonus (Sharpe contribution minus CVaR penalty)
        sharpe_term = kappa_sharpe * delta_sharpe
        cvar_term = -kappa_cvar * cvar_penalty
        risk_bonus = sharpe_term + cvar_term

        # 6. Turnover penalty
        turnover_penalty = 0.0
        if current_weights is not None and self.prev_weights is not None:
            # Turnover = 0.5 * Σ|w_t - w_{t-1}|
            turnover = 0.5 * np.sum(np.abs(current_weights - self.prev_weights))
            turnover_penalty = self.lambda_turnover * turnover

        # Update prev_weights
        if current_weights is not None:
            self.prev_weights = current_weights.copy()

        # 7. 최종 보상
        reward = log_return + risk_bonus - turnover_penalty
        
        # Phase 1: Reward clipping for gradient stability
        # Prevents exploding/vanishing gradients in Q-network
        reward_pre_clip = reward
        reward = np.clip(reward, -1.0, 1.0)

        # 디버깅 정보
        info = {
            "log_return": log_return,
            "delta_sharpe": delta_sharpe,
            "cvar_value": cvar_value,
            "cvar_penalty": cvar_penalty,
            "crisis_level": self.crisis_level,
            "kappa": kappa_sharpe,
            "kappa_sharpe": kappa_sharpe,
            "kappa_cvar": kappa_cvar,
            "risk_bonus": risk_bonus,
            "turnover_penalty": turnover_penalty,
            "sharpe_online": self.dsr.sharpe,
            "reward_pre_clip": reward_pre_clip,  # Phase 1: clipping info
            "reward_total": reward,
            "components": {
                "log_return": log_return,
                "sharpe_term": sharpe_term,
                "cvar_term": cvar_term,
                "turnover": -turnover_penalty,
                "kappa_sharpe": kappa_sharpe,
                "kappa_cvar": kappa_cvar,
            },
        }

        return reward, info

    def reset(self):
        """에피소드 종료 시 상태 초기화"""
        self.dsr.reset()
        self.cvar.reset()
        self.crisis_level = 0.5
        self.prev_weights = None
