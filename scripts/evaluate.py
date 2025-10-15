# scripts/evaluate.py

"""
모델 상세 평가 및 시각화 스크립트

두 가지 평가 방식 지원:
- direct: SB3 모델 직접 사용 (scripts/train.py 결과용)
- drlagent: DRLAgent.DRL_prediction() 사용 (scripts/train_finrl_standard.py 결과용)

Usage:
    # Direct 방식 (기본)
    python scripts/evaluate.py --model logs/sac/20251004_120000/sac_final.zip --save-plot

    # DRLAgent 방식 (FinRL 표준)
    python scripts/evaluate.py --model trained_models/sac_50k.zip --method drlagent --save-json
"""

import argparse
import contextlib
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import Optional, Iterable, Dict, Any

from finrl.config_tickers import DOW_30_TICKER
from finrl.config import INDICATORS, TEST_START_DATE, TEST_END_DATE
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3 import SAC, PPO, A2C, TD3, DDPG
from sb3_contrib.tqc import TQC


def _unwrap_env(env):
    """Extract the underlying base environment from VecEnv/wrappers."""
    env_ref = env
    visited = set()
    while True:
        if id(env_ref) in visited:
            break
        visited.add(id(env_ref))
        if hasattr(env_ref, "envs") and getattr(env_ref, "envs"):
            env_ref = env_ref.envs[0]
            continue
        if hasattr(env_ref, "env"):
            env_ref = env_ref.env
            continue
        break
    return env_ref


def create_env(
    df,
    stock_dim,
    tech_indicators,
    reward_type="basic",
    lambda_dsr=0.1,
    lambda_cvar=0.05,
    use_weighted_action: bool = True,
    weight_slippage: float = 0.001,
    weight_transaction_cost: float = 0.0005,
    reward_scaling: float = 1e-4,
    adaptive_lambda_sharpe: float = 0.20,
    adaptive_lambda_cvar: float = 0.40,
    adaptive_lambda_turnover: float = 0.0,
    adaptive_crisis_gain_sharpe: float = -0.15,
    adaptive_crisis_gain_cvar: float = 0.25,
    adaptive_dsr_beta: float = 0.92,
    adaptive_cvar_window: int = 40,
):
    """환경 생성 (Phase 3.5: reward_type 지원)"""
    # State space: balance(1) + prices(N) + shares(N) + tech_indicators(K*N)
    # Phase 3.5: reward_type='dsr_cvar'일 때 환경 내부에서 +2 (DSR/CVaR)
    state_space = 1 + (len(tech_indicators) + 2) * stock_dim

    env_kwargs = {
        "df": df,
        "stock_dim": stock_dim,
        "hmax": 100,
        "initial_amount": 1000000,
        "num_stock_shares": [0] * stock_dim,
        "buy_cost_pct": [0.001] * stock_dim,
        "sell_cost_pct": [0.001] * stock_dim,
        "reward_scaling": reward_scaling,
        "state_space": state_space,
        "action_space": stock_dim,
        "tech_indicator_list": tech_indicators,
        "print_verbosity": 500,
        # Phase 3.5: 리스크 민감 보상
        "reward_type": reward_type,
        "lambda_dsr": lambda_dsr,
        "lambda_cvar": lambda_cvar,
        "use_weighted_action": use_weighted_action,
        "weight_slippage": weight_slippage,
        "weight_transaction_cost": weight_transaction_cost,
        "adaptive_lambda_sharpe": adaptive_lambda_sharpe,
        "adaptive_lambda_cvar": adaptive_lambda_cvar,
        "adaptive_lambda_turnover": adaptive_lambda_turnover,
        "adaptive_crisis_gain_sharpe": adaptive_crisis_gain_sharpe,
        "adaptive_crisis_gain_cvar": adaptive_crisis_gain_cvar,
        "adaptive_dsr_beta": adaptive_dsr_beta,
        "adaptive_cvar_window": adaptive_cvar_window,
    }

    return StockTradingEnv(**env_kwargs)


def calculate_metrics(
    portfolio_values,
    initial_amount=1000000,
    weights_history=None,
    returns=None,
    executed_weights_history=None,
    turnover_target_series=None,
):
    """
    성능 지표 계산 (상세 메트릭 포함)

    Args:
        portfolio_values: 포트폴리오 가치 배열
        initial_amount: 초기 자본
        weights_history: 포트폴리오 가중치 히스토리 (optional, turnover 계산용)

    Returns:
        dict: 성능 지표 딕셔너리
    """
    from finrl.evaluation.metrics import (
        calculate_sharpe_ratio,
        calculate_sortino_ratio,
        calculate_calmar_ratio,
        calculate_max_drawdown,
        calculate_var,
        calculate_cvar,
        calculate_turnover,
    )

    pv = np.asarray(portfolio_values, dtype=np.float64).reshape(-1)

    # Daily returns
    if returns is None:
        if pv.size > 1:
            prev_values = np.clip(pv[:-1], 1e-8, None)
            returns = (pv[1:] - pv[:-1]) / prev_values
        else:
            returns = np.array([], dtype=np.float64)
    else:
        returns = np.asarray(returns, dtype=np.float64).reshape(-1)
        expected_len = max(pv.size - 1, 0)
        if expected_len and returns.size != expected_len:
            min_len = min(expected_len, returns.size)
            if min_len == 0:
                returns = np.array([], dtype=np.float64)
            else:
                returns = returns[-min_len:]
                pv = pv[-(min_len + 1) :]
        elif expected_len == 0:
            returns = np.array([], dtype=np.float64)

    # Total return
    total_return = (pv[-1] - initial_amount) / initial_amount

    # Annualized return (assuming 252 trading days)
    n_days = returns.size
    annualized_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0

    # Volatility (annualized)
    volatility = np.std(returns) * np.sqrt(252)

    # Sharpe Ratio (using detailed calculation from metrics.py)
    sharpe_ratio = calculate_sharpe_ratio(
        returns, risk_free_rate=0.02, periods_per_year=252
    )

    # Maximum Drawdown (using detailed calculation from metrics.py)
    max_drawdown = calculate_max_drawdown(pv)

    # Calmar Ratio (using detailed calculation from metrics.py)
    calmar_ratio = calculate_calmar_ratio(returns, periods_per_year=252)

    # Sortino Ratio (using detailed calculation from metrics.py)
    sortino_ratio = calculate_sortino_ratio(
        returns, target_return=0.02, periods_per_year=252
    )

    # VaR and CVaR (5% level)
    var_5 = calculate_var(returns, alpha=0.05)
    cvar_5 = calculate_cvar(returns, alpha=0.05)

    # Downside deviation
    downside_returns = returns[returns < 0]
    downside_deviation = (
        np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
    )

    # Turnover (target vs executed)
    turnover_target = 0.0
    turnover_actual = 0.0
    if turnover_target_series is not None and len(turnover_target_series) > 0:
        turnover_target = float(np.mean(turnover_target_series))
    elif weights_history is not None and len(weights_history) > 1:
        turnover_target = calculate_turnover(np.array(weights_history))
    if executed_weights_history is not None and len(executed_weights_history) > 1:
        turnover_actual = calculate_turnover(np.array(executed_weights_history))
    else:
        turnover_actual = turnover_target

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "calmar_ratio": calmar_ratio,
        "max_drawdown": max_drawdown,
        "var_5": var_5,
        "cvar_5": cvar_5,
        "downside_deviation": downside_deviation,
        "avg_turnover": turnover_actual,
        "avg_turnover_target": turnover_target,
        "turnover_gap_abs": abs(turnover_actual - turnover_target),
        "final_value": pv[-1],
        "n_steps": n_days,
    }


def plot_results(
    portfolio_values,
    returns,
    output_dir,
    model_name="Model",
    model_path=None,
    irt_data=None,
    cumulative_mode: str = "log",
):
    """
    시각화 생성 (finrl.evaluation.visualizer 사용)

    일반 모델: 3개 시각화
    IRT 모델: 14개 시각화

    Args:
        portfolio_values: 포트폴리오 가치 배열
        returns: 실행 기반 수익률 배열 (decimal)
        output_dir: 출력 디렉토리
        model_name: 모델 이름
        model_path: 모델 경로 (IRT 감지용)
        irt_data: IRT 중간 데이터 (optional)
    """
    from finrl.evaluation.visualizer import plot_all

    plot_all(
        portfolio_values=np.array(portfolio_values),
        dates=None,
        output_dir=output_dir,
        irt_data=irt_data,
        returns=np.array(returns) if returns is not None else None,
        cumulative_mode=cumulative_mode,
    )


def detect_model_type(model_path):
    """모델 파일명에서 모델 타입 자동 감지"""

    filename = os.path.basename(model_path).lower()

    if "tqc" in filename:
        return "tqc", TQC
    if "sac" in filename:
        return "sac", SAC
    elif "ppo" in filename:
        return "ppo", PPO
    elif "a2c" in filename:
        return "a2c", A2C
    elif "td3" in filename:
        return "td3", TD3
    elif "ddpg" in filename:
        return "ddpg", DDPG
    else:
        # 경로에서 찾기
        path_lower = model_path.lower()
        order = [
            ("tqc", TQC),
            ("sac", SAC),
            ("ppo", PPO),
            ("a2c", A2C),
            ("td3", TD3),
            ("ddpg", DDPG),
        ]
        for name, cls in order:
            if name in path_lower:
                return name, cls

        raise ValueError(
            f"모델 타입을 자동 감지할 수 없습니다: {model_path}\n"
            f"파일명 또는 경로에 모델명(tqc/sac/ppo/a2c/td3/ddpg)을 포함하거나 --model-type 인자를 명시하세요."
        )


def evaluate_model(
    model_path,
    model_class,
    test_start,
    test_end,
    stock_tickers=DOW_30_TICKER,
    tech_indicators=INDICATORS,
    initial_amount=1000000,
    verbose=True,
    reward_type="basic",
    lambda_dsr=0.1,
    lambda_cvar=0.05,
    expected_obs_dim=None,
    use_weighted_action: bool = True,
    weight_slippage: float = 0.001,
    weight_transaction_cost: float = 0.0005,
    reward_scaling: float = 1e-4,
    adaptive_lambda_sharpe: float = 0.20,
    adaptive_lambda_cvar: float = 0.40,
    adaptive_lambda_turnover: float = 0.0,
    adaptive_crisis_gain_sharpe: float = -0.15,
    adaptive_crisis_gain_cvar: float = 0.25,
    adaptive_dsr_beta: float = 0.92,
    adaptive_cvar_window: int = 40,
    *,
    cap_metrics: bool = False,
):
    """
    범용 모델 평가 함수

    Args:
        model_path: 모델 파일 경로 (.zip)
        model_class: SB3 모델 클래스 (SAC, PPO 등)
        test_start: 테스트 시작 날짜 (YYYY-MM-DD)
        test_end: 테스트 종료 날짜 (YYYY-MM-DD)
        stock_tickers: 주식 티커 리스트
        tech_indicators: 기술 지표 리스트
        initial_amount: 초기 자본
        verbose: 진행 상황 출력 여부
        reward_type: 보상 함수 타입 (basic/dsr_cvar/adaptive_risk). IRT 모델은 자동 감지.
        lambda_dsr: DSR 보상 가중치
        lambda_cvar: CVaR 보상 가중치
        expected_obs_dim: 학습 시점 관측 공간 차원 (검증용)
        use_weighted_action: 가중치 기반 행동 모드 사용 여부
        weight_slippage: 가중치 슬리피지 계수
        weight_transaction_cost: 가중치 거래 비용 계수
        reward_scaling: 환경 보상 스케일링 계수
        adaptive_lambda_sharpe: Adaptive risk reward Sharpe 가중치
        adaptive_lambda_cvar: Adaptive risk reward CVaR 가중치
        adaptive_lambda_turnover: Adaptive risk reward turnover 가중치
        adaptive_crisis_gain_sharpe: Adaptive risk reward Sharpe 위기 계수
        adaptive_crisis_gain_cvar: Adaptive risk reward CVaR 위기 계수
        adaptive_dsr_beta: Adaptive risk reward DSR EMA 계수
        adaptive_cvar_window: Adaptive risk reward CVaR 추정 윈도우
        cap_metrics: 메트릭 산출 전에 수익률 클리핑 여부 (False 권장)

    Returns:
        portfolio_values: 포트폴리오 가치 배열
        execution_returns: 거래 비용 반영 실행 기반 수익률 (정제됨)
        value_returns: 포트폴리오 가치 기반 보조 수익률
        irt_data: IRT 중간 데이터 (IRT 모델인 경우, 아니면 None)
        metrics: 성능 지표 딕셔너리
    """
    if verbose:
        print(f"\n[Method: Direct - SB3 predict()]")

    # 1. 데이터 준비
    if verbose:
        print(f"\n[1/4] Downloading test data...")
    df = YahooDownloader(
        start_date=test_start, end_date=test_end, ticker_list=stock_tickers
    ).fetch_data()
    if verbose:
        print(f"  Downloaded: {df.shape[0]} rows")

    # 2. Feature Engineering
    if verbose:
        print(f"\n[2/4] Feature Engineering...")
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=tech_indicators,
        use_turbulence=False,
        user_defined_feature=False,
    )
    df_processed = fe.preprocess_data(df)
    test_df = data_split(df_processed, test_start, test_end)
    if verbose:
        print(f"  Test rows: {len(test_df)}")

    # 3. 환경 및 모델 로드
    if verbose:
        print(f"\n[3/4] Loading model...")
    stock_dim = len(test_df.tic.unique())
    if verbose:
        print(f"  실제 주식 수: {stock_dim}")

    # Phase 3.5 & H1: IRT 모델은 학습 시점 보상 타입을 그대로 사용해야 함
    is_irt = "irt" in model_path.lower()
    requested_reward_type = reward_type

    # Phase 1.5: Load checkpoint metadata first (reward scale, action mode, obs dim, etc.)
    env_meta = {}
    meta_candidates = []
    try:
        model_path_obj = Path(model_path).expanduser().resolve()
        meta_candidates.append(model_path_obj.parent / "env_meta.json")
        # Many checkpoints live in .../best_model/best_model.zip
        meta_candidates.append(model_path_obj.parent.parent / "env_meta.json")
    except Exception:
        model_path_obj = None  # type: ignore[assignment]

    for candidate in meta_candidates:
        if candidate is None:
            continue
        try:
            if candidate.exists():
                with candidate.open("r") as meta_fp:
                    env_meta = json.load(meta_fp)
                break
        except (OSError, json.JSONDecodeError):
            continue

    if not env_meta:
        raise FileNotFoundError(
            "env_meta.json not found alongside the checkpoint; evaluation requires metadata "
            "to verify reward configuration, action mode, and observation layout."
        )

    if env_meta:
        meta_reward_type = env_meta.get("reward_type")
        if meta_reward_type:
            if requested_reward_type and requested_reward_type != meta_reward_type and verbose:
                print(
                    f"  Overriding requested reward_type '{requested_reward_type}' with checkpoint value '{meta_reward_type}'"
                )
            requested_reward_type = str(meta_reward_type)
        if expected_obs_dim is None and env_meta.get("obs_dim") is not None:
            expected_obs_dim = int(env_meta["obs_dim"])
        use_weighted_action = bool(env_meta.get("use_weighted_action", use_weighted_action))
        weight_slippage = float(env_meta.get("weight_slippage", weight_slippage))
        weight_transaction_cost = float(
            env_meta.get("weight_transaction_cost", weight_transaction_cost)
        )
        reward_scaling = float(env_meta.get("reward_scaling", reward_scaling))
        adaptive_lambda_sharpe = float(
            env_meta.get("adaptive_lambda_sharpe", adaptive_lambda_sharpe)
        )
        adaptive_lambda_cvar = float(
            env_meta.get("adaptive_lambda_cvar", adaptive_lambda_cvar)
        )
        adaptive_lambda_turnover = float(
            env_meta.get("adaptive_lambda_turnover", adaptive_lambda_turnover)
        )
        adaptive_crisis_gain_sharpe = float(
            env_meta.get(
                "adaptive_crisis_gain_sharpe",
                env_meta.get("adaptive_crisis_gain", adaptive_crisis_gain_sharpe),
            )
        )
        adaptive_crisis_gain_cvar = float(
            env_meta.get("adaptive_crisis_gain_cvar", adaptive_crisis_gain_cvar)
        )
        adaptive_dsr_beta = float(
            env_meta.get("adaptive_dsr_beta", adaptive_dsr_beta)
        )
        adaptive_cvar_window = int(
            env_meta.get("adaptive_cvar_window", adaptive_cvar_window)
        )

    # Auto-detect reward type when loading IRT models (fallback if mismatch occurs)
    if is_irt:
        candidate_reward_types = []
        if requested_reward_type:
            candidate_reward_types.append(requested_reward_type)
        for candidate in ["adaptive_risk", "dsr_cvar", "basic"]:
            if candidate not in candidate_reward_types:
                candidate_reward_types.append(candidate)
    else:
        candidate_reward_types = [requested_reward_type or "basic"]

    model = None
    test_env = None
    env_reward_type = None
    load_error = None
    mismatch_reasons: list[str] = []

    for candidate_reward_type in candidate_reward_types:
        env_candidate = create_env(
            test_df,
            stock_dim,
            tech_indicators,
            reward_type=candidate_reward_type,
            lambda_dsr=lambda_dsr,
            lambda_cvar=lambda_cvar,
            use_weighted_action=use_weighted_action,
            weight_slippage=weight_slippage,
            weight_transaction_cost=weight_transaction_cost,
            reward_scaling=reward_scaling,
            adaptive_lambda_sharpe=adaptive_lambda_sharpe,
            adaptive_lambda_cvar=adaptive_lambda_cvar,
            adaptive_lambda_turnover=adaptive_lambda_turnover,
            adaptive_crisis_gain_sharpe=adaptive_crisis_gain_sharpe,
            adaptive_crisis_gain_cvar=adaptive_crisis_gain_cvar,
            adaptive_dsr_beta=adaptive_dsr_beta,
            adaptive_cvar_window=adaptive_cvar_window,
        )
        if expected_obs_dim is not None:
            obs_dim = env_candidate.observation_space.shape[0]
            if obs_dim != expected_obs_dim:
                reason = (
                    f"reward_type={candidate_reward_type}: observation dimension mismatch"
                    f" (expected {expected_obs_dim}, got {obs_dim})"
                )
                mismatch_reasons.append(reason)
                load_error = ValueError(reason)
                if hasattr(env_candidate, "close"):
                    with contextlib.suppress(AttributeError):
                        env_candidate.close()
                continue
        try:
            model = model_class.load(model_path, env=env_candidate)
            test_env = env_candidate
            env_reward_type = candidate_reward_type
            break
        except ValueError as exc:
            # Observation space mismatch → try next reward type
            if "Observation spaces do not match" not in str(exc):
                raise
            load_error = exc
            if hasattr(env_candidate, "close"):
                with contextlib.suppress(AttributeError):
                    env_candidate.close()
    if model is None or test_env is None or env_reward_type is None:
        detail = "\n".join(mismatch_reasons) if mismatch_reasons else "<none>"
        raise (
            load_error
            if load_error is not None
            else ValueError(
                "Failed to load model with any compatible reward type."
                f"\nTried reward types: {candidate_reward_types}\nReasons:\n{detail}"
            )
        )

    if verbose:
        print(f"  Model loaded successfully")
        if is_irt:
            if env_reward_type != requested_reward_type:
                print(
                    f"  Detected reward type from model: {env_reward_type} "
                    f"(requested: {requested_reward_type})"
                )
            else:
                print(f"  Reward type: {env_reward_type}")

    base_env = _unwrap_env(test_env) if test_env is not None else None

    if verbose:
        print(
            f"  Env settings → reward_scale={getattr(base_env, 'reward_scaling', None)}, "
            f"use_weighted_action={getattr(base_env, 'use_weighted_action', False)}, "
            f"slippage={getattr(base_env, 'weight_slippage', None)}, "
            f"tx_cost={getattr(base_env, 'weight_transaction_cost', None)}"
        )

    def _apply_crisis_level(env_obj, level_value) -> None:
        """Synchronize crisis level into the evaluation environment."""
        if env_obj is None:
            return
        try:
            crisis_float = float(np.array(level_value).item())
        except (TypeError, ValueError):
            return

        risk_reward = getattr(env_obj, "risk_reward", None)
        if risk_reward is not None and hasattr(risk_reward, "set_crisis_level"):
            with contextlib.suppress(TypeError, ValueError):
                risk_reward.set_crisis_level(crisis_float)

        if hasattr(env_obj, "_crisis_level"):
            env_obj._crisis_level = crisis_float
        if hasattr(env_obj, "last_crisis_level"):
            env_obj.last_crisis_level = crisis_float

    class _CrisisBridgeMonitor:
        """Track crisis thresholds, regime state, and sync with environment."""

        def __init__(
            self,
            vec_env=None,
            base_env=None,
            default_up: float = 0.55,
            default_down: float = 0.45,
        ):
            self.vec_env = vec_env
            self.base_env = base_env
            self.hysteresis_up = float(default_up)
            self.hysteresis_down = float(default_down)
            self.prev_regime = None  # type: ignore[assignment]
            self.latest_level = None  # type: ignore[assignment]
            self.steps_crisis = 0
            self.steps_normal = 0

        @staticmethod
        def _to_float(value, fallback=None):
            try:
                import torch  # type: ignore

                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu().numpy()
            except Exception:
                pass
            if value is None:
                return fallback
            try:
                return float(np.asarray(value).astype(np.float64).reshape(-1)[0])
            except (TypeError, ValueError, IndexError):
                return fallback

        def observe_policy(self, info_dict) -> None:
            """Ingest policy-side statistics prior to env update."""
            if not info_dict or "crisis_level" not in info_dict:
                self.latest_level = None
                return

            self.latest_level = self._to_float(info_dict.get("crisis_level"), None)

            hysteresis_up_val = self._to_float(info_dict.get("hysteresis_up"), None)
            hysteresis_down_val = self._to_float(info_dict.get("hysteresis_down"), None)
            if hysteresis_up_val is not None:
                self.hysteresis_up = hysteresis_up_val
            if hysteresis_down_val is not None:
                self.hysteresis_down = hysteresis_down_val

        def inject_env(self) -> None:
            """Push latest crisis level into the environment via bridge."""
            if self.latest_level is None:
                return
            env_target = self.base_env if self.base_env is not None else self.vec_env
            _apply_crisis_level(env_target, self.latest_level)

        def classify(self):
            """Compute hysteresis-based regime label and update counters."""
            if self.latest_level is None:
                return None

            if self.prev_regime is None:
                regime = 1 if self.latest_level > 0.5 else 0
            elif self.prev_regime == 0:
                regime = 1 if self.latest_level > self.hysteresis_up else 0
            else:
                regime = 0 if self.latest_level < self.hysteresis_down else 1

            self.prev_regime = regime
            if regime == 1:
                self.steps_crisis += 1
            else:
                self.steps_normal += 1
            return regime

        def stats(self):
            return {
                "steps_crisis": self.steps_crisis,
                "steps_normal": self.steps_normal,
            }

    # 4. 평가 실행
    if verbose:
        print(f"\n[4/4] Running evaluation...")
    obs, _ = test_env.reset()
    if is_irt and env_reward_type == "adaptive_risk":
        target_env = base_env if base_env is not None else test_env
        _apply_crisis_level(target_env, getattr(target_env, "last_crisis_level", 0.5))
    done = False
    portfolio_values = [initial_amount]
    value_returns = []
    execution_returns = []
    transaction_costs = []
    turnover_executed = []
    turnover_target_series = []

    prev_prices = None
    if stock_dim > 0 and isinstance(obs, (np.ndarray, list, tuple)):
        obs_array = np.asarray(obs, dtype=np.float64).reshape(-1)
        if obs_array.size >= stock_dim + 1:
            prev_prices = obs_array[1 : stock_dim + 1]

    # IRT 모델 감지 (이미 위에서 설정됨, 중복 제거)

    # IRT 데이터 수집 준비
    if is_irt:
        irt_data_list = {
            "w": [],
            "w_rep": [],
            "w_ot": [],
            "crisis_levels": [],
            "crisis_levels_pre_guard": [],
            "crisis_raw": [],
            "crisis_bias": [],
            "crisis_temperature": [],
            "crisis_guard_rate": [],
            "crisis_types": [],
            "cost_matrices": [],
            "weights": [],
            "actual_weights": [],  # Phase-1: 실행가중 기록
            "eta": [],
            "alpha_c": [],
            "alpha_c_raw": [],  # Phase 1.5: raw alpha before clamp
            "alpha_c_prev": [],  # Phase 1.5: previous alpha
            "alpha_c_decay_factor": [],
            "alpha_crisis_input": [],
            "hysteresis_up": [],
            "hysteresis_down": [],
            "turnover_target": [],
            "top_snapshots": [],
            "crisis_regime": [],  # Phase 1.5: 이진 레짐 분류 결과 (0/1)
            "reward_components": {},
            "reward_components_scaled": {},
        }
        if env_reward_type == "adaptive_risk":
            irt_data_list["env_crisis_levels"] = []
            irt_data_list["env_kappa"] = []
            irt_data_list["env_delta_sharpe"] = []
    else:
        irt_data_list = {}

    bridge_monitor = (
        _CrisisBridgeMonitor(vec_env=test_env, base_env=base_env)
        if is_irt
        else None
    )

    step = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)

        info_dict = None
        # IRT info 수집
        if is_irt and hasattr(model.policy, "get_irt_info"):
            info_dict = model.policy.get_irt_info()
            if info_dict is not None:
                if bridge_monitor is not None:
                    bridge_monitor.observe_policy(info_dict)
                # Batch=1이므로 [0] 인덱스로 추출
                irt_data_list["w"].append(info_dict["w"][0].cpu().numpy())
                irt_data_list["w_rep"].append(info_dict["w_rep"][0].cpu().numpy())
                irt_data_list["w_ot"].append(info_dict["w_ot"][0].cpu().numpy())
                irt_data_list["crisis_levels"].append(
                    info_dict["crisis_level"][0].cpu().numpy()
                )
                if "crisis_level_pre_guard" in info_dict:
                    irt_data_list["crisis_levels_pre_guard"].append(
                        info_dict["crisis_level_pre_guard"][0].cpu().numpy()
                    )
                if "crisis_raw" in info_dict:
                    irt_data_list["crisis_raw"].append(
                        info_dict["crisis_raw"][0].cpu().numpy()
                    )
                if "crisis_bias" in info_dict:
                    irt_data_list["crisis_bias"].append(
                        info_dict["crisis_bias"].cpu().numpy()
                    )
                if "crisis_temperature" in info_dict:
                    irt_data_list["crisis_temperature"].append(
                        info_dict["crisis_temperature"].cpu().numpy()
                    )
                if "crisis_guard_rate" in info_dict:
                    irt_data_list["crisis_guard_rate"].append(
                        info_dict["crisis_guard_rate"].cpu().numpy()
                    )
                irt_data_list["crisis_types"].append(
                    info_dict["crisis_types"][0].cpu().numpy()
                )
                irt_data_list["cost_matrices"].append(
                    info_dict["cost_matrix"][0].cpu().numpy()
                )
                irt_data_list["eta"].append(info_dict["eta"][0].cpu().numpy())
                irt_data_list["alpha_c"].append(info_dict["alpha_c"][0].cpu().numpy())

                # Phase 1.5: alpha_c 상세 정보 수집
                if "alpha_c_raw" in info_dict:
                    irt_data_list["alpha_c_raw"].append(
                        info_dict["alpha_c_raw"][0].cpu().numpy()
                    )
                if "alpha_c_prev" in info_dict:
                    irt_data_list["alpha_c_prev"].append(
                        info_dict["alpha_c_prev"][0].cpu().numpy()
                    )
                if "alpha_c_decay_factor" in info_dict:
                    irt_data_list["alpha_c_decay_factor"].append(
                        info_dict["alpha_c_decay_factor"][0].cpu().numpy()
                    )
                if "alpha_crisis_input" in info_dict:
                    irt_data_list["alpha_crisis_input"].append(
                        info_dict["alpha_crisis_input"][0].cpu().numpy()
                    )

                # Phase 1.5: 히스테리시스 임계치 수집
                hysteresis_up_val = 0.55  # 기본값
                hysteresis_down_val = 0.45  # 기본값
                if "hysteresis_up" in info_dict:
                    hysteresis_up_val = float(
                        np.asarray(info_dict["hysteresis_up"], dtype=np.float64).item()
                    )
                    irt_data_list["hysteresis_up"].append(hysteresis_up_val)
                if "hysteresis_down" in info_dict:
                    hysteresis_down_val = float(
                        np.asarray(
                            info_dict["hysteresis_down"], dtype=np.float64
                        ).item()
                    )
                    irt_data_list["hysteresis_down"].append(hysteresis_down_val)
                # Action을 weight로 변환 (simplex 정규화)
                weights = action / (action.sum() + 1e-8)
                irt_data_list["weights"].append(weights.copy())

        if bridge_monitor is not None and info_dict is None:
            bridge_monitor.observe_policy(None)

        next_obs, reward, done_step, truncated, info = test_env.step(action)
        done = done_step or truncated

        # Phase-H1: Crisis bridge during evaluation (policy → env) & hysteresis classify
        if bridge_monitor is not None:
            if env_reward_type == "adaptive_risk":
                bridge_monitor.inject_env()
            regime = bridge_monitor.classify()
            if regime is not None:
                irt_data_list["crisis_regime"].append(int(regime))

        # Portfolio value 계산
        state = np.asarray(test_env.state, dtype=np.float64)
        cash = float(state[0])
        prices = np.asarray(state[1 : stock_dim + 1], dtype=np.float64)
        holdings = np.asarray(
            state[stock_dim + 1 : 2 * stock_dim + 1], dtype=np.float64
        )
        prev_value = float(portfolio_values[-1])
        pv = float(cash + np.dot(prices, holdings))
        portfolio_values.append(pv)

        denom = max(prev_value, 1e-8)
        value_return = (pv - prev_value) / denom
        value_returns.append(value_return)

        executed_weights = info.get("executed_weights")
        if executed_weights is not None:
            executed_weights = np.asarray(executed_weights, dtype=np.float64).reshape(
                -1
            )

        turnover_value = float(info.get("turnover_target", 0.0) or 0.0)
        turnover_target_series.append(turnover_value)
        if is_irt:
            irt_data_list["turnover_target"].append(turnover_value)
            reward_components = info.get("reward_components")
            if reward_components:
                scaling_factor = float(getattr(test_env, "reward_scaling", 1.0))
                for key, value in reward_components.items():
                    component_value = float(value)
                    bucket = irt_data_list["reward_components"].setdefault(key, [])
                    bucket.append(component_value)
                    bucket_scaled = irt_data_list["reward_components_scaled"].setdefault(
                        key, []
                    )
                    bucket_scaled.append(component_value * scaling_factor)

        # Phase-1: 실행가중(actual_weights) 계산 및 기록
        actual_weights = None
        if is_irt:
            # w^{exec}_t = (p_t ⊙ h_t) / (cash_t + Σ p_{t,i} h_{t,i} + ε)
            total_equity = pv + 1e-8
            actual_weights = (prices * holdings) / total_equity
            irt_data_list["actual_weights"].append(actual_weights)
            if env_reward_type == "adaptive_risk":
                irt_data_list["env_crisis_levels"].append(test_env.last_crisis_level)
                irt_data_list["env_kappa"].append(test_env.last_kappa)
                irt_data_list["env_delta_sharpe"].append(test_env.last_delta_sharpe)

            if len(irt_data_list["top_snapshots"]) < 50 and actual_weights is not None:
                sorted_idx = np.argsort(actual_weights)[::-1]
                top_k = int(min(5, sorted_idx.size))
                top_symbols = [stock_tickers[i] for i in sorted_idx[:top_k]]
                top_exec_weights = actual_weights[sorted_idx[:top_k]].tolist()
                target_slice = None
                if irt_data_list["weights"]:
                    last_target = np.asarray(irt_data_list["weights"][-1])
                    target_slice = last_target[sorted_idx[:top_k]].tolist()
                current_date = None
                if hasattr(test_env, "date_memory") and test_env.date_memory:
                    current_date = test_env.date_memory[-1]
                snapshot = {
                    "step": step,
                    "date": str(current_date) if current_date is not None else None,
                    "tickers": top_symbols,
                    "weights": top_exec_weights,
                    "target_weights": target_slice,
                }
                irt_data_list["top_snapshots"].append(snapshot)

        tc_value = float(info.get("transaction_cost", 0.0) or 0.0)
        transaction_costs.append(tc_value)

        current_prices = None
        if stock_dim > 0 and isinstance(next_obs, (np.ndarray, list, tuple)):
            next_obs_array = np.asarray(next_obs, dtype=np.float64).reshape(-1)
            if next_obs_array.size >= stock_dim + 1:
                current_prices = next_obs_array[1 : stock_dim + 1]

        exec_return = value_return
        if prev_prices is not None and current_prices is not None:
            price_denominator = np.clip(prev_prices, 1e-8, None)
            price_relatives = (current_prices - price_denominator) / price_denominator
            weight_vector = executed_weights
            if weight_vector is None and actual_weights is not None:
                weight_vector = actual_weights
            if weight_vector is not None and weight_vector.size == price_relatives.size:
                exec_return = float(
                    np.dot(weight_vector, price_relatives) - tc_value / denom
                )
        execution_returns.append(exec_return)

        if hasattr(test_env, "_last_turnover_executed"):
            turnover_executed.append(
                float(getattr(test_env, "_last_turnover_executed"))
            )
        else:
            turnover_executed.append(0.0)

        obs = next_obs
        prev_prices = current_prices
        step += 1

    if verbose:
        print(f"  Evaluation completed: {step} steps")

    if bridge_monitor is not None:
        stats = bridge_monitor.stats()
        total_reg = stats.get("steps_crisis", 0) + stats.get("steps_normal", 0)
        expected_reg = len(irt_data_list.get("crisis_levels", []))
        if total_reg == 0:
            raise RuntimeError(
                "Crisis regime classification produced zero samples. Verify bridge wiring."
            )
        if expected_reg and total_reg != expected_reg:
            print(
                f"⚠️  Crisis regime length mismatch (classified {total_reg} vs {expected_reg}).",
            )

    # 수익률 시리즈 (raw + sanitize)
    from finrl.evaluation.visualizer import sanitize_returns

    execution_returns_raw = np.array(execution_returns, dtype=np.float64)
    value_returns_raw = np.array(value_returns, dtype=np.float64)
    execution_returns_array = sanitize_returns(execution_returns_raw)
    value_returns_array = sanitize_returns(value_returns_raw)
    transaction_costs_array = np.array(transaction_costs, dtype=np.float64)
    turnover_executed_array = np.array(turnover_executed, dtype=np.float64)
    turnover_target_array = np.array(turnover_target_series, dtype=np.float64)
    if turnover_target_array.size and turnover_target_array.size != turnover_executed_array.size:
        raise RuntimeError(
            "Turnover series length mismatch between target and executed. "
            f"target={turnover_target_array.size}, executed={turnover_executed_array.size}"
        )

    # IRT 데이터 변환
    irt_data = None
    if is_irt and irt_data_list["w"]:
        irt_data = {
            "w_rep": np.array(irt_data_list["w_rep"]),  # [T, M]
            "w_ot": np.array(irt_data_list["w_ot"]),  # [T, M]
            "weights": np.array(irt_data_list["weights"]),  # [T, N] 목표가중
            "actual_weights": np.array(
                irt_data_list["actual_weights"]
            ),  # [T, N] Phase-1: 실행가중
            "crisis_levels": np.array(irt_data_list["crisis_levels"]).squeeze(),  # [T]
            "crisis_levels_pre_guard": np.array(
                irt_data_list["crisis_levels_pre_guard"]
            ).squeeze()
            if irt_data_list["crisis_levels_pre_guard"]
            else np.array([], dtype=np.float64),
            "crisis_regime": np.array(
                irt_data_list["crisis_regime"], dtype=np.int32
            ),  # Phase 1.5: [T] 이진 분류
            "crisis_types": np.array(irt_data_list["crisis_types"]),  # [T, K]
            "prototype_weights": np.array(irt_data_list["w"]),  # [T, M]
            "cost_matrices": np.array(irt_data_list["cost_matrices"]),  # [T, m, M]
            "eta": np.array(irt_data_list["eta"]).squeeze(),  # [T]
            "alpha_c": np.array(irt_data_list["alpha_c"]).squeeze(),  # [T]
            "alpha_crisis_input": np.array(
                irt_data_list["alpha_crisis_input"]
            ).squeeze()
            if irt_data_list["alpha_crisis_input"]
            else np.array([], dtype=np.float64),
            "hysteresis_up": np.array(irt_data_list["hysteresis_up"], dtype=np.float64),
            "hysteresis_down": np.array(
                irt_data_list["hysteresis_down"], dtype=np.float64
            ),
            "symbols": stock_tickers[:stock_dim],  # 실제 주식 수만큼
            "metrics": None,  # 호출자가 calculate_metrics()로 계산
            "returns": execution_returns_array,
            "returns_exec": execution_returns_array,
            "returns_value": value_returns_array,
            "transaction_costs": transaction_costs_array,
            "turnover_executed": turnover_executed_array,
            "turnover_target": np.array(
                irt_data_list["turnover_target"], dtype=np.float64
            ),
            "returns_raw": execution_returns_raw,
            "returns_sanitized": execution_returns_array,
            "value_returns_raw": value_returns_raw,
            "value_returns_sanitized": value_returns_array,
            "reward_components": {
                key: np.array(values, dtype=np.float64)
                for key, values in irt_data_list["reward_components"].items()
            },
            "reward_components_scaled": {
                key: np.array(values, dtype=np.float64)
                for key, values in irt_data_list["reward_components_scaled"].items()
            },
            "top_snapshots": irt_data_list["top_snapshots"],
        }
        # Phase 1.5: alpha_c 상세 정보 추가
        if irt_data_list["alpha_c_raw"]:
            irt_data["alpha_c_raw"] = np.array(irt_data_list["alpha_c_raw"]).squeeze()
        if irt_data_list["alpha_c_prev"]:
            irt_data["alpha_c_prev"] = np.array(irt_data_list["alpha_c_prev"]).squeeze()
        if irt_data_list["alpha_c_decay_factor"]:
            irt_data["alpha_c_decay_factor"] = np.array(
                irt_data_list["alpha_c_decay_factor"]
            ).squeeze()
        if env_reward_type == "adaptive_risk" and irt_data_list.get(
            "env_crisis_levels"
        ):
            irt_data["env_crisis_levels"] = np.array(
                irt_data_list["env_crisis_levels"]
            ).squeeze()
            irt_data["env_kappa"] = np.array(irt_data_list["env_kappa"]).squeeze()
            irt_data["env_delta_sharpe"] = np.array(
                irt_data_list["env_delta_sharpe"]
            ).squeeze()
        if irt_data_list["crisis_levels_pre_guard"]:
            irt_data["crisis_levels_pre_guard"] = np.array(
                irt_data_list["crisis_levels_pre_guard"]
            ).squeeze()
        if irt_data_list["crisis_raw"]:
            irt_data["crisis_raw"] = np.array(irt_data_list["crisis_raw"]).squeeze()
        if irt_data_list["crisis_bias"]:
            irt_data["crisis_bias"] = np.array(irt_data_list["crisis_bias"]).squeeze()
        if irt_data_list["crisis_temperature"]:
            irt_data["crisis_temperature"] = np.array(
                irt_data_list["crisis_temperature"]
            ).squeeze()
        if irt_data_list["crisis_guard_rate"]:
            irt_data["crisis_guard_rate"] = np.array(
                irt_data_list["crisis_guard_rate"]
            ).squeeze()

        env_info = {
            "reward_type": env_reward_type,
            "reward_scaling": getattr(base_env, "reward_scaling", None),
            "use_weighted_action": getattr(base_env, "use_weighted_action", None),
            "weight_slippage": getattr(base_env, "weight_slippage", None),
            "weight_transaction_cost": getattr(
                base_env, "weight_transaction_cost", None
            ),
            "adaptive_lambda_sharpe": getattr(
                base_env, "adaptive_lambda_sharpe", None
            ),
            "adaptive_lambda_cvar": getattr(base_env, "adaptive_lambda_cvar", None),
            "adaptive_lambda_turnover": getattr(
                base_env, "adaptive_lambda_turnover", None
            ),
            "adaptive_crisis_gain_sharpe": getattr(
                base_env, "adaptive_crisis_gain_sharpe", None
            ),
            "adaptive_crisis_gain_cvar": getattr(
                base_env, "adaptive_crisis_gain_cvar", None
            ),
            "adaptive_dsr_beta": getattr(base_env, "adaptive_dsr_beta", None),
            "adaptive_cvar_window": getattr(base_env, "adaptive_cvar_window", None),
        }
        irt_data["env_info"] = env_info
        if env_meta:
            irt_data["env_meta"] = env_meta

    # 성능 지표 계산
    weights_history = irt_data["weights"] if irt_data else None
    executed_weights_history = irt_data["actual_weights"] if irt_data else None
    returns_for_metrics = (
        execution_returns_array if cap_metrics else execution_returns_raw
    )
    metrics = calculate_metrics(
        portfolio_values,
        initial_amount,
        weights_history,
        returns=returns_for_metrics,
        executed_weights_history=executed_weights_history,
        turnover_target_series=turnover_target_array,
    )

    avg_turnover_exec = (
        float(np.mean(turnover_executed_array)) if turnover_executed_array.size else 0.0
    )
    avg_turnover_target = (
        float(np.mean(turnover_target_array)) if turnover_target_array.size else 0.0
    )

    metrics["avg_turnover"] = avg_turnover_exec
    metrics["avg_turnover_executed"] = avg_turnover_exec
    metrics["avg_turnover_target_env"] = avg_turnover_target
    metrics["avg_turnover_target"] = avg_turnover_target
    metrics["turnover_executed_std"] = (
        float(np.std(turnover_executed_array)) if turnover_executed_array.size else 0.0
    )
    metrics["turnover_target_std"] = (
        float(np.std(turnover_target_array)) if turnover_target_array.size else 0.0
    )
    metrics["turnover_transfer_ratio"] = (
        float(avg_turnover_exec / (avg_turnover_target + 1e-8))
        if avg_turnover_exec or avg_turnover_target
        else 0.0
    )

    metrics["returns_capped_for_metrics"] = bool(cap_metrics)
    if execution_returns_array.size and execution_returns_raw.size:
        diff = execution_returns_array - execution_returns_raw
        diff_abs = np.abs(diff)
        metrics["sanitize_gap_mean"] = float(np.mean(diff_abs))
        metrics["sanitize_gap_max"] = float(np.max(diff_abs))
    else:
        metrics["sanitize_gap_mean"] = 0.0
        metrics["sanitize_gap_max"] = 0.0

    if irt_data is not None:

        def _safe_array(name):
            value = irt_data.get(name)
            if value is None:
                return None
            arr = np.asarray(value)
            return arr if arr.size else None

        alpha_vals = _safe_array("alpha_c")
        if alpha_vals is not None:
            metrics["alpha_c_mean_eval"] = float(np.mean(alpha_vals))
            metrics["alpha_c_std_eval"] = float(np.std(alpha_vals))
            metrics["alpha_c_min_eval"] = float(np.min(alpha_vals))
            metrics["alpha_c_max_eval"] = float(np.max(alpha_vals))

        # Phase 1.5: alpha_c 상세 정보 (raw, prev 포함)
        alpha_raw_vals = _safe_array("alpha_c_raw")
        alpha_prev_vals = _safe_array("alpha_c_prev")
        if alpha_raw_vals is not None:
            metrics["alpha_c_raw_mean_eval"] = float(np.mean(alpha_raw_vals))
            metrics["alpha_c_raw_std_eval"] = float(np.std(alpha_raw_vals))
        if alpha_prev_vals is not None:
            metrics["alpha_c_prev_mean_eval"] = float(np.mean(alpha_prev_vals))
        if (
            alpha_vals is not None
            and alpha_prev_vals is not None
            and len(alpha_vals) == len(alpha_prev_vals)
        ):
            delta_alpha = alpha_vals - alpha_prev_vals
            metrics["delta_alpha_c_mean"] = float(np.mean(delta_alpha))
            metrics["delta_alpha_c_std"] = float(np.std(delta_alpha))
            metrics["delta_alpha_c_abs_mean"] = float(np.mean(np.abs(delta_alpha)))

        crisis_vals = _safe_array("crisis_levels")
        if crisis_vals is not None:
            metrics["crisis_level_max_eval"] = float(np.max(crisis_vals))
            metrics["crisis_level_p90_eval"] = float(np.quantile(crisis_vals, 0.9))
            metrics["crisis_level_p99_eval"] = float(np.quantile(crisis_vals, 0.99))
            metrics["crisis_level_median_eval"] = float(np.median(crisis_vals))
            metrics["crisis_activation_rate"] = float(np.mean(crisis_vals >= 0.55))

        # Phase 1.5: 레짐 기반 분리 통계
        crisis_regime_vals = _safe_array("crisis_regime")
        if crisis_regime_vals is not None and len(crisis_regime_vals) > 0:
            crisis_mask = crisis_regime_vals == 1
            normal_mask = crisis_regime_vals == 0
            crisis_steps = int(np.sum(crisis_mask))
            normal_steps = int(np.sum(normal_mask))
            total_steps = len(crisis_regime_vals)

            metrics["crisis_regime_pct"] = float(crisis_steps / max(total_steps, 1))
            metrics["crisis_steps"] = crisis_steps
            metrics["normal_steps"] = normal_steps

            # 레짐별 수익률/샤프 분리 (execution_returns_array 사용)
            if len(execution_returns_array) == len(crisis_regime_vals):
                if crisis_steps > 0:
                    crisis_returns = execution_returns_array[crisis_mask]
                    metrics["crisis_mean_return"] = float(np.mean(crisis_returns))
                    metrics["crisis_volatility"] = float(np.std(crisis_returns))
                    if metrics["crisis_volatility"] > 1e-8:
                        metrics["crisis_sharpe"] = float(
                            metrics["crisis_mean_return"]
                            / metrics["crisis_volatility"]
                            * np.sqrt(252)
                        )
                if normal_steps > 0:
                    normal_returns = execution_returns_array[normal_mask]
                    metrics["normal_mean_return"] = float(np.mean(normal_returns))
                    metrics["normal_volatility"] = float(np.std(normal_returns))
                    if metrics["normal_volatility"] > 1e-8:
                        metrics["normal_sharpe"] = float(
                            metrics["normal_mean_return"]
                            / metrics["normal_volatility"]
                            * np.sqrt(252)
                        )

        reward_components_dict = irt_data.get("reward_components")
        if reward_components_dict:
            component_items = [
                (key, np.asarray(values, dtype=np.float64))
                for key, values in sorted(reward_components_dict.items())
                if len(values) > 0
            ]
            if component_items:
                component_matrix = np.vstack([values for _, values in component_items])
                abs_sum = np.sum(np.abs(component_matrix), axis=0)
                abs_sum = np.where(abs_sum <= 1e-12, 1e-12, abs_sum)
                for idx, (key, values) in enumerate(component_items):
                    metrics[f"reward_component_{key}_mean"] = float(np.mean(values))
                    metrics[f"reward_component_{key}_std"] = float(np.std(values))
                    shares = np.abs(component_matrix[idx]) / abs_sum
                    metrics[f"reward_component_{key}_abs_share"] = float(np.mean(shares))

        hyst_up_vals = _safe_array("hysteresis_up")
        hyst_down_vals = _safe_array("hysteresis_down")
        if (
            hyst_up_vals is not None
            and hyst_down_vals is not None
            and len(hyst_up_vals) > 0
            and len(hyst_down_vals) > 0
        ):
            metrics["hysteresis_up_mean"] = float(np.mean(hyst_up_vals))
            metrics["hysteresis_down_mean"] = float(np.mean(hyst_down_vals))
            # Phase 1.5: 히스테리시스 범위 추적
            metrics["hysteresis_up_min"] = float(np.min(hyst_up_vals))
            metrics["hysteresis_up_max"] = float(np.max(hyst_up_vals))
            metrics["hysteresis_down_min"] = float(np.min(hyst_down_vals))
            metrics["hysteresis_down_max"] = float(np.max(hyst_down_vals))
            metrics["hysteresis_width_mean"] = float(
                np.mean(hyst_up_vals - hyst_down_vals)
            )

        proto_weights = _safe_array("prototype_weights")
        if proto_weights is not None:
            weights_clipped = np.clip(proto_weights, 1e-12, 1.0)
            proto_entropy = -np.sum(weights_clipped * np.log(weights_clipped), axis=1)
            proto_max = np.max(weights_clipped, axis=1)
            proto_var = np.var(weights_clipped, axis=1)
            metrics["prototype_entropy_mean_eval"] = float(np.mean(proto_entropy))
            metrics["prototype_entropy_std_eval"] = float(np.std(proto_entropy))
            metrics["prototype_entropy_min_eval"] = float(
                np.min(proto_entropy)
            )  # Phase 1.5
            metrics["prototype_entropy_max_eval"] = float(
                np.max(proto_entropy)
            )  # Phase 1.5
            metrics["prototype_max_weight_mean_eval"] = float(np.mean(proto_max))
            metrics["prototype_max_weight_max_eval"] = float(
                np.max(proto_max)
            )  # Phase 1.5
            metrics["prototype_var_mean_eval"] = float(np.mean(proto_var))

    # IRT 데이터에 metrics 추가
    if irt_data is not None:
        irt_data["metrics"] = metrics

    return (
        portfolio_values,
        execution_returns_array,
        value_returns_array,
        irt_data,
        metrics,
    )


def evaluate_direct(args, model_name, model_class):
    """Direct 방식: SB3 모델 직접 사용 (args 객체 wrapper)"""

    return evaluate_model(
        model_path=args.model,
        model_class=model_class,
        test_start=args.test_start,
        test_end=args.test_end,
        stock_tickers=DOW_30_TICKER,
        tech_indicators=INDICATORS,
        initial_amount=1000000,
        verbose=True,
        adaptive_lambda_sharpe=args.adaptive_lambda_sharpe,
        adaptive_lambda_cvar=args.adaptive_lambda_cvar,
        adaptive_lambda_turnover=args.adaptive_lambda_turnover,
        adaptive_crisis_gain_sharpe=args.adaptive_crisis_gain_sharpe,
        adaptive_crisis_gain_cvar=args.adaptive_crisis_gain_cvar,
        adaptive_dsr_beta=args.adaptive_dsr_beta,
        adaptive_cvar_window=args.adaptive_cvar_window,
        cap_metrics=not args.no_cap_metrics,
    )


def evaluate_drlagent(args, model_name, model_class):
    """DRLAgent 방식: DRLAgent.DRL_prediction() 사용"""

    print(f"\n[Method: DRLAgent - DRL_prediction()]")

    # 1. 데이터 준비
    print(f"\n[1/4] Downloading test data...")
    df = YahooDownloader(
        start_date=args.test_start, end_date=args.test_end, ticker_list=DOW_30_TICKER
    ).fetch_data()
    print(f"  Downloaded: {df.shape[0]} rows")

    # 2. Feature Engineering
    print(f"\n[2/4] Feature Engineering...")
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_turbulence=False,
        user_defined_feature=False,
    )
    df_processed = fe.preprocess_data(df)
    test_df = data_split(df_processed, args.test_start, args.test_end)
    print(f"  Test rows: {len(test_df)}")

    # 3. 환경 생성 및 모델 로드
    print(f"\n[3/4] Loading model...")
    stock_dim = len(test_df.tic.unique())
    print(f"  실제 주식 수: {stock_dim}")

    # Phase 3.5: IRT 모델은 dsr_cvar 보상 사용
    is_irt = "irt" in args.model.lower()
    reward_type = "dsr_cvar" if is_irt else "basic"

    # State space: balance(1) + prices(N) + shares(N) + tech_indicators(K*N)
    # Phase 3.5: reward_type='dsr_cvar'일 때 환경 내부에서 +2 (DSR/CVaR)
    state_space = 1 + (len(INDICATORS) + 2) * stock_dim

    env_kwargs = {
        "df": test_df,
        "stock_dim": stock_dim,
        "hmax": 100,
        "initial_amount": 1000000,
        "num_stock_shares": [0] * stock_dim,
        "buy_cost_pct": [0.001] * stock_dim,
        "sell_cost_pct": [0.001] * stock_dim,
        "reward_scaling": 1e-4,
        "state_space": state_space,
        "action_space": stock_dim,
        "tech_indicator_list": INDICATORS,
        "print_verbosity": 500,
        # Phase 3.5: 리스크 민감 보상
        "reward_type": reward_type,
        "lambda_dsr": 0.1,
        "lambda_cvar": 0.05,
    }

    e_test_gym = StockTradingEnv(**env_kwargs)

    model = model_class.load(args.model)
    print(f"  Model loaded successfully")
    if is_irt:
        print(f"  Reward type: {reward_type}")

    # 4. DRL_prediction 실행
    print(f"\n[4/4] Running DRL_prediction...")
    account_memory, actions_memory = DRLAgent.DRL_prediction(
        model=model, environment=e_test_gym, deterministic=True
    )

    print(f"  Evaluation completed: {len(account_memory)} steps")

    # account_memory에서 portfolio_values 추출
    portfolio_values = account_memory["account_value"].tolist()

    # DRLAgent 방식은 IRT 데이터 수집 불가
    pv_array = np.asarray(portfolio_values, dtype=np.float64)
    if pv_array.size > 1:
        prev_vals = np.clip(pv_array[:-1], 1e-8, None)
        raw_returns = (pv_array[1:] - pv_array[:-1]) / prev_vals
    else:
        raw_returns = np.array([], dtype=np.float64)
    from finrl.evaluation.visualizer import sanitize_returns

    exec_returns = sanitize_returns(raw_returns)
    value_returns = sanitize_returns(raw_returns)

    cap_metrics = not args.no_cap_metrics
    returns_for_metrics = exec_returns if cap_metrics else raw_returns
    metrics = calculate_metrics(
        portfolio_values,
        returns=returns_for_metrics,
    )
    metrics["returns_capped_for_metrics"] = bool(cap_metrics)
    if exec_returns.size and raw_returns.size:
        diff = exec_returns - raw_returns
        diff_abs = np.abs(diff)
        metrics["sanitize_gap_mean"] = float(np.mean(diff_abs))
        metrics["sanitize_gap_max"] = float(np.max(diff_abs))
    else:
        metrics["sanitize_gap_mean"] = 0.0
        metrics["sanitize_gap_max"] = 0.0

    return portfolio_values, exec_returns, value_returns, None, metrics


def main(args):
    print("=" * 70)
    print("Model Evaluation")
    print("=" * 70)

    # 모델 타입 감지
    if args.model_type is None:
        model_name, model_class = detect_model_type(args.model)
        print(f"\n  Auto-detected model type: {model_name.upper()}")
    else:
        model_name = args.model_type
        model_class = {"sac": SAC, "ppo": PPO, "a2c": A2C, "td3": TD3, "ddpg": DDPG}[
            model_name
        ]

    print(f"\n[Config]")
    print(f"  Model: {args.model}")
    print(f"  Type: {model_name.upper()}")
    print(f"  Method: {args.method}")
    print(f"  Test: {args.test_start} ~ {args.test_end}")

    # 평가 방식 선택
    if args.method == "direct":
        portfolio_values, exec_returns, value_returns, irt_data, metrics = (
            evaluate_direct(args, model_name, model_class)
        )
    elif args.method == "drlagent":
        (
            portfolio_values,
            exec_returns,
            value_returns,
            irt_data,
            metrics,
        ) = evaluate_drlagent(args, model_name, model_class)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    # 6. 결과 출력
    print(f"\n" + "=" * 70)
    print(f"Performance Metrics")
    print("=" * 70)
    print(f"\n[Period]")
    print(f"  Start: {args.test_start}")
    print(f"  End: {args.test_end}")
    print(f"  Steps: {metrics['n_steps']}")

    print(f"\n[Returns]")
    print(f"  Total Return: {metrics['total_return']*100:.2f}%")
    print(f"  Annualized Return: {metrics['annualized_return']*100:.2f}%")

    print(f"\n[Risk Metrics]")
    print(f"  Volatility (annualized): {metrics['volatility']*100:.2f}%")
    print(f"  Maximum Drawdown: {metrics['max_drawdown']*100:.2f}%")

    print(f"\n[Risk-Adjusted Returns]")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"  Sortino Ratio: {metrics['sortino_ratio']:.3f}")
    print(f"  Calmar Ratio: {metrics['calmar_ratio']:.3f}")

    print(f"\n[Portfolio Value]")
    print(f"  Initial: ${1000000:,.2f}")
    print(f"  Final: ${metrics['final_value']:,.2f}")
    print(f"  Profit/Loss: ${metrics['final_value'] - 1000000:,.2f}")

    print(f"\n" + "=" * 70)

    # 7. 시각화 (기본 활성화)
    if not args.no_plot:
        output_dir = args.output or os.path.join(
            os.path.dirname(args.model), "evaluation_plots"
        )
        plot_results(
            portfolio_values,
            exec_returns,
            output_dir,
            model_name=model_name.upper(),
            model_path=args.model,
            irt_data=irt_data,
        )

    env_info = {}
    env_meta = None
    if isinstance(irt_data, dict):
        env_info = irt_data.pop("env_info", {}) or {}
        env_meta = irt_data.pop("env_meta", None)

    env_reward_type = env_info.get("reward_type")
    reward_scaling = env_info.get("reward_scaling")
    use_weighted_action = env_info.get("use_weighted_action")
    weight_slippage = env_info.get("weight_slippage")
    weight_transaction_cost = env_info.get("weight_transaction_cost")
    adaptive_lambda_sharpe = env_info.get("adaptive_lambda_sharpe")
    adaptive_lambda_cvar = env_info.get("adaptive_lambda_cvar")
    adaptive_lambda_turnover = env_info.get("adaptive_lambda_turnover")
    adaptive_crisis_gain_sharpe = env_info.get("adaptive_crisis_gain_sharpe")
    adaptive_crisis_gain_cvar = env_info.get("adaptive_crisis_gain_cvar")
    adaptive_dsr_beta = env_info.get("adaptive_dsr_beta")
    adaptive_cvar_window = env_info.get("adaptive_cvar_window")

    if adaptive_lambda_sharpe is None:
        adaptive_lambda_sharpe = args.adaptive_lambda_sharpe
    if adaptive_lambda_cvar is None:
        adaptive_lambda_cvar = args.adaptive_lambda_cvar
    if adaptive_lambda_turnover is None:
        adaptive_lambda_turnover = args.adaptive_lambda_turnover
    if adaptive_crisis_gain_sharpe is None:
        adaptive_crisis_gain_sharpe = args.adaptive_crisis_gain_sharpe
    if adaptive_crisis_gain_cvar is None:
        adaptive_crisis_gain_cvar = args.adaptive_crisis_gain_cvar
    if adaptive_dsr_beta is None:
        adaptive_dsr_beta = args.adaptive_dsr_beta
    if adaptive_cvar_window is None:
        adaptive_cvar_window = args.adaptive_cvar_window

    # 8. JSON 저장 (기본 활성화)
    if not args.no_json:
        from finrl.evaluation.visualizer import save_evaluation_results
        from pathlib import Path

        output_dir = args.output_json or os.path.dirname(args.model)

        # IRT 데이터가 있으면 상세 JSON 저장, 없으면 기본 메트릭만 저장
        if irt_data is not None:
            # 상세 evaluation_results.json + evaluation_insights.json 저장
            results = {
                "returns": exec_returns,
                "returns_raw": irt_data.get("returns_raw"),
                "returns_sanitized": irt_data.get("returns_sanitized"),
                "returns_value": value_returns,
                "value_returns_raw": irt_data.get("value_returns_raw"),
                "value_returns_sanitized": irt_data.get("value_returns_sanitized"),
                "values": np.array(portfolio_values),
                "weights": irt_data.get("weights"),
                "actual_weights": irt_data.get("actual_weights"),
                "crisis_levels": irt_data.get("crisis_levels"),
                "crisis_levels_pre_guard": irt_data.get("crisis_levels_pre_guard"),
                "crisis_regime": irt_data.get("crisis_regime"),
                "crisis_types": irt_data.get("crisis_types"),
                "prototype_weights": irt_data.get("prototype_weights"),
                "w_rep": irt_data.get("w_rep"),
                "w_ot": irt_data.get("w_ot"),
                "eta": irt_data.get("eta"),
                "alpha_c": irt_data.get("alpha_c"),
                "alpha_c_raw": irt_data.get("alpha_c_raw"),
                "alpha_c_prev": irt_data.get("alpha_c_prev"),
                "alpha_c_decay_factor": irt_data.get("alpha_c_decay_factor"),
                "alpha_crisis_input": irt_data.get("alpha_crisis_input"),
                "cost_matrices": irt_data.get("cost_matrices"),
                "symbols": irt_data.get("symbols"),
                "transaction_costs": irt_data.get("transaction_costs"),
                "turnover_executed": irt_data.get("turnover_executed"),
                "hysteresis_up": irt_data.get("hysteresis_up"),
                "hysteresis_down": irt_data.get("hysteresis_down"),
                "turnover_target_series": irt_data.get("turnover_target"),
                "reward_components": irt_data.get("reward_components"),
                "reward_components_scaled": irt_data.get("reward_components_scaled"),
                "metrics": metrics,
            }

            # Config (IRT Policy의 경우)
            config = {
                "env": {
                    "reward_type": env_reward_type,
                    "reward_scaling": reward_scaling,
                    "use_weighted_action": use_weighted_action,
                    "weight_slippage": weight_slippage,
                    "weight_transaction_cost": weight_transaction_cost,
                    "adaptive_lambda_sharpe": adaptive_lambda_sharpe,
                    "adaptive_lambda_cvar": adaptive_lambda_cvar,
                    "adaptive_lambda_turnover": adaptive_lambda_turnover,
                    "adaptive_crisis_gain_sharpe": adaptive_crisis_gain_sharpe,
                    "adaptive_crisis_gain_cvar": adaptive_crisis_gain_cvar,
                    "adaptive_dsr_beta": adaptive_dsr_beta,
                    "adaptive_cvar_window": adaptive_cvar_window,
                },
                "meta": env_meta,
            }

            print(f"\n  Saving detailed JSON (IRT data)...")
            save_evaluation_results(results, Path(output_dir), config)

        else:
            # 기본 메트릭만 저장
            output_file = os.path.join(output_dir, "evaluation_results.json")
            os.makedirs(output_dir, exist_ok=True)

            results = {
                "model_path": args.model,
                "model_type": model_name,
                "evaluation_method": args.method,
                "test_period": {
                    "start": args.test_start,
                    "end": args.test_end,
                    "steps": metrics["n_steps"],
                },
                "metrics": {
                    "total_return": float(metrics["total_return"]),
                    "annualized_return": float(metrics["annualized_return"]),
                    "volatility": float(metrics["volatility"]),
                    "sharpe_ratio": float(metrics["sharpe_ratio"]),
                    "sortino_ratio": float(metrics["sortino_ratio"]),
                    "calmar_ratio": float(metrics["calmar_ratio"]),
                    "max_drawdown": float(metrics["max_drawdown"]),
                    "final_value": float(metrics["final_value"]),
                    "profit_loss": float(metrics["final_value"] - 1000000),
                    "returns_capped_for_metrics": bool(
                        metrics.get("returns_capped_for_metrics", False)
                    ),
                    "sanitize_gap_mean": float(metrics.get("sanitize_gap_mean", 0.0)),
                    "sanitize_gap_max": float(metrics.get("sanitize_gap_max", 0.0)),
                },
                "timestamp": datetime.now().isoformat(),
            }

            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)

            print(f"\n  Results saved to: {output_file}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="모델 상세 평가")

    parser.add_argument(
        "--model", type=str, required=True, help="모델 파일 경로 (.zip)"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        choices=["sac", "ppo", "a2c", "td3", "ddpg"],
        help="모델 타입 (자동 감지 실패 시 명시)",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="direct",
        choices=["direct", "drlagent"],
        help="평가 방식 (default: direct)",
    )

    parser.add_argument(
        "--test-start",
        type=str,
        default=TEST_START_DATE,
        help=f"Test start date (default: {TEST_START_DATE})",
    )
    parser.add_argument(
        "--test-end",
        type=str,
        default=TEST_END_DATE,
        help=f"Test end date (default: {TEST_END_DATE})",
    )
    parser.add_argument(
        "--adaptive-lambda-sharpe",
        type=float,
        default=0.20,
        help="Adaptive risk reward Sharpe weight λ_S (default: 0.20)",
    )
    parser.add_argument(
        "--adaptive-lambda-cvar",
        type=float,
        default=0.40,
        help="Adaptive risk reward CVaR weight β (default: 0.40)",
    )
    parser.add_argument(
        "--adaptive-lambda-turnover",
        type=float,
        default=0.0,
        help="Adaptive risk reward turnover penalty μ (default: 0.0; set >0 only if NAV lacks transaction costs)",
    )
    parser.add_argument(
        "--adaptive-crisis-gain",
        dest="adaptive_crisis_gain_sharpe",
        type=float,
        default=-0.15,
        help="Adaptive risk reward Sharpe crisis gain g_S (default: -0.15, keep negative)",
    )
    parser.add_argument(
        "--adaptive-crisis-gain-cvar",
        type=float,
        default=0.25,
        help="Adaptive risk reward CVaR crisis gain g_C (default: 0.25)",
    )
    parser.add_argument(
        "--adaptive-dsr-beta",
        type=float,
        default=0.92,
        help="Adaptive risk reward DSR EMA β (default: 0.92)",
    )
    parser.add_argument(
        "--adaptive-cvar-window",
        type=int,
        default=40,
        help="Adaptive risk reward CVaR window length (default: 40)",
    )

    parser.set_defaults(no_cap_metrics=True)
    parser.add_argument(
        "--no-cap-metrics",
        dest="no_cap_metrics",
        action="store_true",
        help="메트릭 계산 시 수익률 클리핑 비활성화 (기본값)",
    )
    parser.add_argument(
        "--cap-metrics",
        dest="no_cap_metrics",
        action="store_false",
        help="메트릭 계산 전에 수익률을 클리핑합니다 (추천하지 않음).",
    )

    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="시각화 결과 저장 비활성화 (기본: 활성화)",
    )
    parser.add_argument(
        "--no-json", action="store_true", help="JSON 결과 저장 비활성화 (기본: 활성화)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Plot 출력 디렉토리 (기본: 모델 디렉토리/evaluation_plots)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="JSON 출력 파일 (기본: 모델 디렉토리/evaluation_results.json)",
    )

    args = parser.parse_args()

    metrics = main(args)
