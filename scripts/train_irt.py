# scripts/train_irt.py

"""
IRT Policy 학습 스크립트

위기 적응형 포트폴리오 관리를 위한 SAC + IRT Policy

사용법:
    # 기본 설정으로 학습 및 평가
    python scripts/train_irt.py --mode both

    # 커스텀 alpha로 학습만 수행
    python scripts/train_irt.py --mode train --alpha 0.5 --episodes 200

    # 저장된 모델 평가
    python scripts/train_irt.py --mode eval --checkpoint logs/irt/.../irt_final.zip
"""

import argparse
import json
import os
import random
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from finrl.config import (
    INDICATORS,
    SAC_PARAMS,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
)
from finrl.config_tickers import DOW_30_TICKER
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import torch
from stable_baselines3.common.logger import (
    HumanOutputFormat,
    Logger,
    TensorBoardOutputFormat,
)

# Import IRT Policy
from finrl.agents.irt import IRTPolicy

# Import evaluation function
import sys

sys.path.insert(0, os.path.dirname(__file__))
from evaluate import evaluate_model
from finrl.agents.stablebaselines3 import StrictEvalCallback


def set_global_seed(seed: int) -> None:
    """
    Configure all known PRNGs to use the same seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_environment(env, seed: int) -> None:
    """
    Try to seed gym environments and their spaces if supported.
    """
    target_envs: Iterable[Any]
    if hasattr(env, "envs") and hasattr(env, "seed"):
        num_envs = len(getattr(env, "envs", []))
        env.seed(seed)
        target_envs = getattr(env, "envs")
    else:
        if hasattr(env, "seed"):
            env.seed(seed)
        elif hasattr(env, "_seed"):  # legacy gym
            env._seed(seed)
        target_envs = (env,)
    if hasattr(env, "reset"):
        try:
            env.reset(seed=seed)
        except TypeError:
            env.reset()

    for idx, sub_env in enumerate(target_envs):
        sub_seed = seed + idx * 997
        if hasattr(sub_env, "reset"):
            try:
                sub_env.reset(seed=sub_seed)
            except TypeError:
                sub_env.reset()
        action_space = getattr(sub_env, "action_space", None)
        if action_space is not None and hasattr(action_space, "seed"):
            action_space.seed(sub_seed)
        observation_space = getattr(sub_env, "observation_space", None)
        if observation_space is not None and hasattr(observation_space, "seed"):
            observation_space.seed(sub_seed)


def _unwrap_env(env):
    """
    Recursively unwrap VecEnv/Gym wrappers to obtain the base environment.
    """
    env_ref = env
    visited = set()
    while True:
        if id(env_ref) in visited:
            break
        visited.add(id(env_ref))
        if hasattr(env_ref, "envs") and len(getattr(env_ref, "envs")) > 0:
            env_ref = env_ref.envs[0]
            continue
        if hasattr(env_ref, "env"):
            env_ref = env_ref.env
            continue
        break
    return env_ref


ALLOWED_TENSORBOARD_TAGS = {
    "returns/episode_sharpe",
    "returns/episode_sortino",
    "returns/annualized_return",
    "risk/max_drawdown",
    "risk/cvar_p05",
    "risk/crisis_level_avg",
    "risk/kappa_sharpe",
    "risk/kappa_cvar",
    "reward/log_return",
    "reward/sharpe_term",
    "reward/cvar_term",
    "reward/total",
    "action/turnover_executed",
    "action/cash_weight",
    "train/actor_loss",
    "train/critic_loss",
    "train/entropy",
    "train/alpha",
    "xai/proto_entropy",
}


class TensorboardWhitelistOutputFormat(TensorBoardOutputFormat):
    """
    TensorBoard output that drops any keys outside the allowed FinFlow telemetry set.
    """

    def __init__(self, folder: str):
        os.makedirs(folder, exist_ok=True)
        super().__init__(folder)

    def _is_allowed(self, key: str) -> bool:
        return key in ALLOWED_TENSORBOARD_TAGS

    def write(
        self,
        key_values: dict[str, Any],
        key_excluded: dict[str, tuple[str, ...]],
        step: int = 0,
    ) -> None:
        filtered_keys = {}
        filtered_excluded = {}
        for k, v in key_values.items():
            if self._is_allowed(k):
                filtered_keys[k] = v
                filtered_excluded[k] = key_excluded.get(k, tuple())

        if not filtered_keys:
            return

        super().write(filtered_keys, filtered_excluded, step)


class IRTLoggingCallback(BaseCallback):
    """Minimal TensorBoard logging for FinFlow IRT training."""

    _METRIC_KEYS = {
        "reward/log_return": "reward_log_return",
        "reward/sharpe_term": ("reward_components", "sharpe_term"),
        "reward/cvar_term": ("reward_components", "cvar_term"),
        "reward/total": "reward_total",
        "risk/cvar_p05": "cvar_value",
        "risk/crisis_level_avg": "crisis_level",
        "risk/kappa_sharpe": "kappa_sharpe",
        "risk/kappa_cvar": "kappa_cvar",
        "action/turnover_executed": "turnover_executed",
        "action/cash_weight": "cash_weight",
    }

    def __init__(self, log_freq: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = max(1, int(log_freq))
        self._reset_buffers()

    def _reset_buffers(self) -> None:
        self._metrics = {key: [0.0, 0] for key in self._METRIC_KEYS}

    @staticmethod
    def _to_float(value):
        if value is None:
            return None
        try:
            val = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(val):
            return None
        return val

    def _accumulate_metric(self, key: str, value) -> None:
        val = self._to_float(value)
        if val is None:
            return
        bucket = self._metrics[key]
        bucket[0] += val
        bucket[1] += 1

    def _value_from_info(self, info: dict, descriptor):
        if isinstance(descriptor, tuple):
            container = info.get(descriptor[0]) or {}
            return container.get(descriptor[1])
        return info.get(descriptor)

    def _on_training_start(self) -> None:
        self._reset_buffers()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos") if isinstance(self.locals, dict) else None
        if infos:
            for info in infos:
                if not info:
                    continue
                for tb_key, desc in self._METRIC_KEYS.items():
                    value = self._value_from_info(info, desc)
                    self._accumulate_metric(tb_key, value)
        if self.n_calls % self.log_freq == 0:
            self._flush()
        return True

    def _on_training_end(self) -> None:
        self._flush()

    def _flush(self) -> None:
        for key, (total, count) in self._metrics.items():
            if count:
                self.logger.record(key, total / count)
        proto_entropy = self._compute_proto_entropy()
        if proto_entropy is not None:
            self.logger.record("xai/proto_entropy", proto_entropy)
        alpha = self._current_alpha()
        if alpha is not None:
            self.logger.record("train/alpha", alpha)
        self._reset_buffers()

    def _compute_proto_entropy(self) -> Optional[float]:
        model = getattr(self, "model", None)
        if model is None:
            return None
        policy = getattr(model, "policy", None)
        if policy is None or not hasattr(policy, "get_irt_info"):
            return None
        try:
            info = policy.get_irt_info()
        except AttributeError:
            return None
        if not info:
            return None
        weights = info.get("w")
        if weights is None:
            return None
        if isinstance(weights, torch.Tensor):
            tensor = weights.detach().float()
        else:
            tensor = torch.as_tensor(weights, dtype=torch.float32)
        if tensor.ndim < 2:
            return None
        probs = torch.clamp(tensor, min=1e-8)
        entropy = -(probs * probs.log()).sum(dim=-1).mean()
        if not torch.isfinite(entropy):
            return None
        return float(entropy.item())

    def _current_alpha(self) -> Optional[float]:
        model = getattr(self, "model", None)
        if model is None or not hasattr(model, "log_ent_coef"):
            return None
        log_ent_coef = getattr(model, "log_ent_coef")
        if log_ent_coef is None:
            return None
        if isinstance(log_ent_coef, torch.Tensor):
            value = torch.exp(log_ent_coef.detach()).cpu().item()
        elif isinstance(log_ent_coef, (float, int)):
            value = float(math.exp(log_ent_coef))
        else:
            return None
        if not np.isfinite(value):
            return None
        return float(value)



class CrisisBridgeCallback(BaseCallback):
    """
    Phase-H1: Policy → Environment Crisis Level Bridge

    IRT Policy에서 계산된 crisis_level을 환경의 AdaptiveRiskReward로 전달하는 브릿지.

    동작:
    1. Policy에서 get_irt_info()를 통해 crisis_level 추출
    2. Environment의 risk_reward.set_crisis_level() 호출
    3. Adaptive κ(c) 업데이트로 위기 반응성 향상

    사용 조건:
    - reward_type='adaptive_risk'인 환경에서만 작동
    - IRT Policy 사용 시 필수
    """

    def __init__(self, learning_starts: int = 5000, verbose: int = 0):
        super().__init__(verbose)
        self.learning_starts = learning_starts
        self._warned_once = False
        self._connected_once = False

    def _on_step(self) -> bool:
        if self.model is None:
            return True

        # Phase 2.1: Remove learning_starts check - connect immediately
        # Policy outputs crisis_level from step 0, should sync immediately
        # Old code (removed):
        # if getattr(self.model, "num_timesteps", 0) < self.learning_starts:
        #     return True

        info = None
        policy = getattr(self.model, "policy", None)
        if policy is not None and hasattr(policy, "get_irt_info"):
            try:
                info = policy.get_irt_info()
            except AttributeError as exc:
                if not self._warned_once:
                    print(f"⚠️  CrisisBridgeCallback: Failed to fetch IRT info ({exc})")
                    self._warned_once = True
                return True

        if not info or "crisis_level" not in info:
            if not self._warned_once:
                print("⚠️  CrisisBridgeCallback: IRT info not available, skipping")
                self._warned_once = True
            return True

        crisis_tensor = info["crisis_level"]
        if isinstance(crisis_tensor, torch.Tensor):
            crisis_level = float(crisis_tensor.detach().cpu().mean().item())
        elif isinstance(crisis_tensor, (np.ndarray, list)):
            crisis_level = float(np.mean(crisis_tensor))
        else:
            crisis_level = float(crisis_tensor)

        base_env = _unwrap_env(self.model.get_env())
        setattr(base_env, "_crisis_level", crisis_level)

        risk_reward = getattr(base_env, "risk_reward", None)
        if risk_reward is not None and hasattr(risk_reward, "set_crisis_level"):
            try:
                risk_reward.set_crisis_level(crisis_level)
            except (TypeError, ValueError) as exc:
                if not self._warned_once:
                    print(f"⚠️  CrisisBridgeCallback: Failed to update crisis level ({exc})")
                    self._warned_once = True
                return True

        if hasattr(base_env, "_crisis_history"):
            base_env._crisis_history.append((int(self.model.num_timesteps), crisis_level))

        if not self._connected_once:
            print(f"✅ CrisisBridgeCallback: Successfully connected (t={self.model.num_timesteps})")
            self._connected_once = True
            self._warned_once = False

        return True


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
    """StockTradingEnv 생성 (Phase 3: DSR + CVaR 보상 지원)"""

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
        # Phase 3: 리스크 민감 보상
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


@dataclass
class DiversificationConfig:
    start_index: int
    start_date: str
    end_index: int
    end_date: str
    window_days: int
    tx_scale: float
    slippage_scale: float
    row_count: int

    def to_meta(self) -> Dict[str, Any]:
        return {
            "start_index": int(self.start_index),
            "start_date": self.start_date,
            "end_index": int(self.end_index),
            "end_date": self.end_date,
            "window_days": int(self.window_days),
            "tx_scale": float(self.tx_scale),
            "slippage_scale": float(self.slippage_scale),
            "row_count": int(self.row_count),
        }


def _sorted_unique_dates(df) -> List[Any]:
    dates = df["date"].unique()
    return sorted(dates.tolist())


def _slice_df_by_dates(df, start_date, end_date) -> Tuple[Any, Any, pd.DataFrame]:
    mask = (df["date"] >= start_date) & (df["date"] <= end_date)
    window = df.loc[mask].copy()
    if window.empty:
        raise ValueError("Selected diversification window is empty.")
    window.sort_values(by=["date", "tic"], inplace=True)
    day_codes = pd.factorize(window["date"], sort=False)[0]
    window.index = day_codes
    window.index.name = None
    return start_date, end_date, window


def _sample_scales(rng: np.random.Generator, magnitude: float) -> float:
    if magnitude <= 0:
        return 1.0
    sample = float(rng.uniform(-magnitude, magnitude))
    scale = 1.0 + sample
    return max(scale, 0.05)


def _choose_offsets(
    mode: str,
    n_envs: int,
    start_mode: str,
    max_start: int,
    rng: np.random.Generator,
    window_step: int,
) -> List[int]:
    if mode == "off":
        return [0]
    total = max(1, n_envs)
    if max_start <= 0:
        return [0 for _ in range(total)]
    if start_mode == "random":
        return [int(rng.integers(0, max_start + 1)) for _ in range(total)]
    if start_mode == "uniform":
        if total == 1:
            return [int(max_start // 2)]
        lin = np.linspace(0, max_start, num=total)
        return [int(round(v)) for v in lin]
    if start_mode == "rolling":
        step = max(1, min(window_step, max_start if max_start > 0 else window_step))
        offsets: List[int] = []
        cursor = 0
        for _ in range(total):
            offsets.append(min(cursor, max_start))
            cursor += step
            if cursor > max_start:
                cursor = cursor % (max_start + 1)
        return offsets
    raise ValueError(f"Unsupported start_offset_mode: {start_mode}")


def build_training_env_with_diversification(
    df,
    stock_dim: int,
    tech_indicators: Sequence[str],
    args,
) -> Tuple[Any, Dict[str, Any]]:
    mode = args.env_diversify
    rolling_offsets = args.start_offset_mode == "rolling"
    rng = np.random.default_rng(args.seed + 17041)
    unique_dates = _sorted_unique_dates(df)
    if not unique_dates:
        raise ValueError("Training dataframe does not contain any dates.")
    total_days = len(unique_dates)
    min_window = max(1, min(total_days, max(32, int(0.1 * total_days))))
    window_step = max(1, min(int(args.window_step), total_days))

    if rolling_offsets:
        window_days = max(min_window, window_step)
        max_start = max(0, total_days - window_days)
    else:
        window_days = total_days
        max_start = max(0, total_days - min_window)

    n_envs = 1 if mode != "vector" else max(1, int(args.n_envs))
    offsets = _choose_offsets(
        mode=mode,
        n_envs=n_envs,
        start_mode=args.start_offset_mode,
        max_start=max_start,
        rng=rng,
        window_step=window_step,
    )

    configs: List[DiversificationConfig] = []
    env_fns: List[Any] = []

    base_tx = 0.0005
    base_slippage = 0.001

    for offset in offsets:
        start_idx = max(0, min(offset, total_days - 1))
        if rolling_offsets:
            end_idx = min(total_days, start_idx + window_days)
        else:
            end_idx = total_days
        end_idx = max(start_idx + 1, end_idx)
        start_date = unique_dates[start_idx]
        end_date = unique_dates[end_idx - 1]
        _, _, window_df = _slice_df_by_dates(df, start_date, end_date)
        tx_scale = 1.0 if mode == "off" else _sample_scales(rng, float(args.domain_rand_tx))
        slip_scale = 1.0 if mode == "off" else _sample_scales(rng, float(args.domain_rand_slippage))
        config = DiversificationConfig(
            start_index=int(start_idx),
            start_date=str(start_date),
            end_index=int(end_idx - 1),
            end_date=str(end_date),
            window_days=int(end_idx - start_idx),
            tx_scale=float(tx_scale),
            slippage_scale=float(slip_scale),
            row_count=int(window_df.shape[0]),
        )
        configs.append(config)

        env_kwargs = {
            "df": window_df,
            "stock_dim": stock_dim,
            "tech_indicators": tech_indicators,
            "reward_type": args.reward_type,
            "lambda_dsr": args.lambda_dsr,
            "lambda_cvar": args.lambda_cvar,
            "reward_scaling": args.reward_scale,
            "adaptive_lambda_sharpe": args.adaptive_lambda_sharpe,
            "adaptive_lambda_cvar": args.adaptive_lambda_cvar,
            "adaptive_lambda_turnover": args.adaptive_lambda_turnover,
            "adaptive_crisis_gain_sharpe": args.adaptive_crisis_gain_sharpe,
            "adaptive_crisis_gain_cvar": args.adaptive_crisis_gain_cvar,
            "adaptive_dsr_beta": args.adaptive_dsr_beta,
            "adaptive_cvar_window": args.adaptive_cvar_window,
            "use_weighted_action": True,
            "weight_slippage": base_slippage * slip_scale,
            "weight_transaction_cost": base_tx * tx_scale,
        }

        def _make_env(env_kwargs=env_kwargs):
            return create_env(**env_kwargs)

        env_fns.append(_make_env)

    if mode == "vector" and len(env_fns) > 1:
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)

    effective_mode = mode
    if mode == "vector" and len(env_fns) == 1:
        effective_mode = "basic"

    diversify_meta: Dict[str, Any] = {
        "mode": effective_mode,
        "requested_mode": mode,
        "params": {
            "n_envs": len(env_fns),
            "start_offset_mode": args.start_offset_mode,
            "window_step": int(window_step),
            "min_window_days": int(min_window),
            "domain_rand_tx": float(args.domain_rand_tx),
            "domain_rand_slippage": float(args.domain_rand_slippage),
        },
        "configs": [cfg.to_meta() for cfg in configs],
    }

    return vec_env, diversify_meta


def train_irt(args):
    """IRT 모델 학습"""

    print("=" * 70)
    print(f"IRT Training - Dow Jones 30")
    print("=" * 70)

    set_global_seed(args.seed)

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.output, "irt", timestamp)
    os.makedirs(log_dir, exist_ok=True)

    print(f"\n[Config]")
    print(f"  Model: SAC + IRT Policy")
    print(f"  Stocks: Dow Jones 30 ({len(DOW_30_TICKER)} tickers)")
    print(f"  Train: {args.train_start} ~ {args.train_end}")
    print(f"  Test: {args.test_start} ~ {args.test_end}")
    print(f"  Episodes: {args.episodes}")
    print(f"  IRT alpha: {args.alpha}")
    print(f"  Output: {log_dir}")

    # 1. Download data
    print(f"\n[1/5] Downloading data...")
    df = YahooDownloader(
        start_date=args.train_start, end_date=args.test_end, ticker_list=DOW_30_TICKER
    ).fetch_data()
    print(f"  Downloaded: {df.shape[0]} rows")

    # 2. Feature Engineering
    print(f"\n[2/5] Feature Engineering...")
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_turbulence=False,
        user_defined_feature=False,
    )
    df_processed = fe.preprocess_data(df)
    print(f"  Features: {df_processed.shape[1]} columns")

    # 3. Train/Test Split
    print(f"\n[3/5] Splitting data...")
    train_df = data_split(df_processed, args.train_start, args.train_end)
    test_df = data_split(df_processed, args.test_start, args.test_end)
    print(f"  Train: {len(train_df)} rows")
    print(f"  Test: {len(test_df)} rows")

    # 4. Create environments
    print(f"\n[4/5] Creating environments...")
    stock_dim = len(train_df.tic.unique())

    # 데이터 누락 경고
    if stock_dim != len(DOW_30_TICKER):
        removed_count = len(DOW_30_TICKER) - stock_dim
        print(f"  ⚠️  주의: {removed_count}개 주식이 데이터 부족으로 제외됨")
        print(f"      (2008년 초 데이터가 없는 종목: Visa (V) 등)")

    print(f"  실제 주식 수: {stock_dim}")

    # Phase 3: 리스크 민감 보상 환경 생성
    train_env, diversify_meta = build_training_env_with_diversification(
        train_df,
        stock_dim,
        INDICATORS,
        args,
    )
    seed_environment(train_env, args.seed)
    base_train_env = _unwrap_env(train_env)
    print(
        f"  Train env: reward_scale={getattr(base_train_env, 'reward_scaling', None)}, "
        f"use_weighted_action={getattr(base_train_env, 'use_weighted_action', False)}, "
        f"slippage={getattr(base_train_env, 'weight_slippage', None)}, "
        f"tx_cost={getattr(base_train_env, 'weight_transaction_cost', None)}"
    )
    if diversify_meta["mode"] != "off":
        print(
            f"    Diversify: mode={diversify_meta['mode']}, "
            f"n_envs={diversify_meta['params']['n_envs']}, "
            f"window_step={diversify_meta['params']['window_step']}, "
            f"domain_rand_tx={diversify_meta['params']['domain_rand_tx']}, "
            f"domain_rand_slippage={diversify_meta['params']['domain_rand_slippage']}"
        )
        if diversify_meta.get("requested_mode") != diversify_meta["mode"]:
            print("      (requested mode: {req}, downcast to {eff} because n_envs=1)".format(
                req=diversify_meta.get("requested_mode"), eff=diversify_meta["mode"]
            ))
        for idx, cfg in enumerate(diversify_meta["configs"]):
            print(
                "      Env[{idx}] start={start} end={end} days={days} tx×{tx:.3f} slip×{sl:.3f}".format(
                    idx=idx,
                    start=cfg["start_date"],
                    end=cfg["end_date"],
                    days=cfg["window_days"],
                    tx=cfg["tx_scale"],
                    sl=cfg["slippage_scale"],
                )
            )

    tech_indicator_count = len(getattr(base_train_env, "tech_indicator_list", []))
    has_dsr_cvar = args.reward_type == "dsr_cvar"
    test_env = create_env(
        test_df,
        stock_dim,
        INDICATORS,
        reward_type=args.reward_type,
        lambda_dsr=args.lambda_dsr,
        lambda_cvar=args.lambda_cvar,
        reward_scaling=args.reward_scale,
        adaptive_lambda_sharpe=args.adaptive_lambda_sharpe,
        adaptive_lambda_cvar=args.adaptive_lambda_cvar,
        adaptive_lambda_turnover=args.adaptive_lambda_turnover,
        adaptive_crisis_gain_sharpe=args.adaptive_crisis_gain_sharpe,
        adaptive_crisis_gain_cvar=args.adaptive_crisis_gain_cvar,
        adaptive_dsr_beta=args.adaptive_dsr_beta,
        adaptive_cvar_window=args.adaptive_cvar_window,
    )
    seed_environment(test_env, args.seed)
    print(
        f"  Test env:  reward_scale={test_env.reward_scaling}, "
        f"use_weighted_action={getattr(test_env, 'use_weighted_action', False)}, "
        f"slippage={getattr(test_env, 'weight_slippage', None)}, "
        f"tx_cost={getattr(test_env, 'weight_transaction_cost', None)}"
    )

    train_obs_dim = train_env.observation_space.shape[0]
    test_obs_dim = test_env.observation_space.shape[0]
    if train_obs_dim != test_obs_dim:
        raise ValueError(
            f"Train/Test observation space mismatch: train={train_obs_dim}, test={test_obs_dim}"
        )

    print(f"  State space: {getattr(base_train_env, 'state_space', train_obs_dim)}")
    print(f"  Action space: {base_train_env.action_space}")
    print(f"  Reward type: {args.reward_type}")
    if args.reward_type == "dsr_cvar":
        print(f"    λ_dsr: {args.lambda_dsr}, λ_cvar: {args.lambda_cvar}")
    elif args.reward_type == "adaptive_risk":
        rr = getattr(base_train_env, "risk_reward", None)
        lambda_sharpe = getattr(rr, "lambda_sharpe_base", None)
        lambda_cvar = getattr(rr, "lambda_cvar", None)
        lambda_turnover = getattr(rr, "lambda_turnover", None)
        crisis_gain_sharpe = getattr(rr, "crisis_gain_sharpe", None)
        crisis_gain_cvar = getattr(rr, "crisis_gain_cvar", None)
        print(f"    Phase-H1: Adaptive Risk-Aware Reward")
        print(
            f"    λ_sharpe: {lambda_sharpe}, λ_cvar: {lambda_cvar}, "
            f"λ_turnover: {lambda_turnover}, "
            f"g_S: {crisis_gain_sharpe}, g_C: {crisis_gain_cvar}"
        )

    # 5. Train IRT model
    print(f"\n[5/5] Training SAC + IRT Policy...")

    # Callback 디렉토리 미리 생성 (FileNotFoundError 방지)
    os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "best_model"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "eval"), exist_ok=True)

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=os.path.join(log_dir, "checkpoints"),
        name_prefix="irt_model",
    )

    eval_callback = StrictEvalCallback(
        Monitor(test_env),
        best_model_save_path=os.path.join(log_dir, "best_model"),
        log_path=os.path.join(log_dir, "eval"),
        eval_freq=5000,
        deterministic=True,
        render=False,
    )

    # Phase A: IRT 계측 callback
    irt_logging_callback = IRTLoggingCallback(log_freq=100, verbose=0)

    # IRT Policy kwargs
    effective_market_dim = (
        args.market_feature_dim
        if args.market_feature_dim is not None
        else 4 + tech_indicator_count
    )

    policy_kwargs = {
        "emb_dim": args.emb_dim,
        "m_tokens": args.m_tokens,
        "M_proto": args.M_proto,
        "alpha": args.alpha,
        "alpha_min": args.alpha_min,
        "alpha_max": args.alpha_max if args.alpha_max else args.alpha,
        "ema_beta": args.ema_beta,
        "market_feature_dim": effective_market_dim,
        "stock_dim": stock_dim,
        "tech_indicator_count": tech_indicator_count,
        "has_dsr_cvar": has_dsr_cvar,
        # Phase 3.5: Dirichlet 및 온도 파라미터 (훈련 경로 전달 필수)
        "dirichlet_min": args.dirichlet_min,
        "dirichlet_max": args.dirichlet_max,
        "action_temp": args.action_temp,
        # IRT Operator 파라미터
        "eps": args.eps,
        "max_iters": args.max_iters,
        "replicator_temp": args.replicator_temp,
        "eta_0": args.eta_0,
        "eta_1": args.eta_1,
        "gamma": args.gamma,
        "alpha_update_rate": args.alpha_update_rate,
        "alpha_feedback_gain": args.alpha_feedback_gain,
        "alpha_feedback_bias": args.alpha_feedback_bias,
        "directional_decay_min": args.directional_decay_min,
        "alpha_noise_std": args.alpha_noise_std,
        "alpha_crisis_source": args.alpha_crisis_source,
        # Phase 3.5 Step 2: 다중 신호 위기 감지
        "w_r": args.w_r,
        "w_s": args.w_s,
        "w_c": args.w_c,
        # Phase B: 바이어스 EMA 보정
        "eta_b": args.eta_b,
        "eta_b_min": args.eta_b_min,
        "eta_b_decay_steps": args.eta_b_decay_steps,
        "eta_T": args.eta_T,
        "p_star": args.p_star,
        "temperature_min": args.temperature_min,
        "temperature_max": args.temperature_max,
        "stat_momentum": args.stat_momentum,
        "eta_b_warmup_steps": args.eta_b_warmup_steps,
        "eta_b_warmup_value": args.eta_b_warmup_value,
        "crisis_guard_rate_init": args.crisis_guard_rate_init,
        "crisis_guard_rate_final": args.crisis_guard_rate_final,
        "crisis_guard_warmup_steps": args.crisis_guard_warmup_steps,
        "hysteresis_up": args.hysteresis_up,
        "hysteresis_down": args.hysteresis_down,
        "hysteresis_quantile": args.hysteresis_quantile,
        "k_s": args.k_s,
        "k_c": args.k_c,
        "k_b": args.k_b,
        "crisis_guard_rate": args.crisis_guard_rate,
    }

    # SAC parameters
    sac_params = SAC_PARAMS.copy()

    # Phase 2.2a: Increase entropy coefficient for exploration
    # Simplex entropy H ∈ [0, log(30)] = [0, 3.40]
    # Portfolio allocation requires higher exploration than typical continuous control
    # 0.3 (Phase 2.1) was too conservative → uniform freeze
    # 0.5 balances exploration and stability (comparable to portfolio RL literature)
    sac_params["ent_coef"] = 0.5  # Phase 2.2a: 0.3 → 0.5 (enable diversity)
    sac_params["learning_starts"] = 1000  # Phase 2: 5000 → 1000 (faster warmup)

    # Phase-H1: Critic learning rate 상향 (adaptive_risk 전용)
    if args.reward_type == "adaptive_risk":
        sac_params["learning_rate"] = 5e-4  # 1e-4 → 5e-4 (5배 증가)
        print("  Phase-H1: SAC learning_rate=5e-4 for gradient amplification")

    print(f"  SAC params: {sac_params}")
    print(f"  IRT params: {policy_kwargs}")

    # Phase-H1: Crisis Bridge callback (adaptive_risk 전용)
    crisis_bridge_callback = None
    if args.reward_type == "adaptive_risk":
        crisis_bridge_callback = CrisisBridgeCallback(
            learning_starts=0,  # Phase 2.1: Start immediately (was: sac_params["learning_starts"])
            verbose=1,  # Phase 2.1: Enable verbose logging
        )
        print("  Phase-H1: CrisisBridgeCallback enabled (Policy→Env crisis signal)")

    sac_params["seed"] = args.seed

    # Create SAC model with IRT Policy
    model = SAC(
        policy=IRTPolicy,
        env=train_env,
        policy_kwargs=policy_kwargs,
        **sac_params,
        verbose=1,
        tensorboard_log=None,
    )

    tb_path = os.path.join(log_dir, "tensorboard")
    logger_formats = [
        HumanOutputFormat(sys.stdout),
        TensorboardWhitelistOutputFormat(tb_path),
    ]
    model.set_logger(Logger(log_dir, logger_formats))
    
    # Phase 1.5: 파라미터 그룹 검증 (프로토타입 포함 확인)
    print("\n[Phase 1.5] Verifying optimizer parameter groups...")
    actor_params = list(model.actor.parameters())
    actor_param_count = sum(p.numel() for p in actor_params)
    actor_requires_grad = sum(p.numel() for p in actor_params if p.requires_grad)
    print(f"  Actor total params: {actor_param_count:,}")
    print(f"  Actor trainable params: {actor_requires_grad:,}")
    
    # Prototype decoder 파라미터 확인
    if hasattr(model.actor, 'irt_actor'):
        irt_actor = model.actor.irt_actor
        if hasattr(irt_actor, 'decoders'):
            decoder_params = sum(p.numel() for d in irt_actor.decoders for p in d.parameters())
            decoder_trainable = sum(p.numel() for d in irt_actor.decoders for p in d.parameters() if p.requires_grad)
            print(f"  Prototype decoders: {decoder_params:,} params ({decoder_trainable:,} trainable)")
            
            # Gradient flow 검증용 플래그
            if decoder_trainable == 0:
                print("  WARNING: Prototype decoders have 0 trainable parameters!")
            elif decoder_trainable < decoder_params:
                print(f"  WARNING: Some prototype parameters are frozen ({decoder_params - decoder_trainable} params)")

    # Total timesteps
    total_timesteps = 250 * args.episodes
    print(f"\n  Total timesteps: {total_timesteps}")
    print(f"  Starting training...")

    # Train
    callbacks = [checkpoint_callback, eval_callback, irt_logging_callback]
    if crisis_bridge_callback is not None:
        callbacks.append(crisis_bridge_callback)

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    # Save final model
    final_model_path = os.path.join(log_dir, "irt_final.zip")
    model.save(final_model_path)

    print(f"\n" + "=" * 70)
    print(f"Training completed!")
    print("=" * 70)
    print(f"  Final model: {final_model_path}")
    print(f"  Best model: {os.path.join(log_dir, 'best_model', 'best_model.zip')}")
    print(f"  Logs: {log_dir}")

    env_meta = {
        "obs_dim": int(train_obs_dim),
        "action_dim": int(train_env.action_space.shape[0]),
        "feature_columns": df_processed.columns.tolist(),
        "tech_indicators": list(INDICATORS),
        "reward_type": args.reward_type,
        "use_weighted_action": getattr(base_train_env, "use_weighted_action", True),
        "weight_slippage": getattr(base_train_env, "weight_slippage", 0.001),
        "weight_transaction_cost": getattr(base_train_env, "weight_transaction_cost", 0.0005),
        "reward_scaling": args.reward_scale,
        "stock_dim": stock_dim,
        "tech_indicator_count": tech_indicator_count,
        "has_dsr_cvar": has_dsr_cvar,
        "stat_momentum": args.stat_momentum,
        "adaptive_lambda_sharpe": args.adaptive_lambda_sharpe,
        "adaptive_lambda_cvar": args.adaptive_lambda_cvar,
        "adaptive_lambda_turnover": args.adaptive_lambda_turnover,
        "adaptive_crisis_gain_sharpe": args.adaptive_crisis_gain_sharpe,
        "adaptive_crisis_gain_cvar": args.adaptive_crisis_gain_cvar,
        "adaptive_dsr_beta": args.adaptive_dsr_beta,
        "adaptive_cvar_window": args.adaptive_cvar_window,
        "train_start": args.train_start,
        "train_end": args.train_end,
        "test_start": args.test_start,
        "test_end": args.test_end,
        "env_diversify": diversify_meta["mode"],
        "env_diversify_requested": diversify_meta.get("requested_mode"),
        "env_diversify_params": diversify_meta["params"],
        "env_diversify_configs": diversify_meta["configs"],
    }
    with open(os.path.join(log_dir, "env_meta.json"), "w") as meta_fp:
        json.dump(env_meta, meta_fp, indent=2)

    return log_dir, final_model_path


def _log_returns_to_tensorboard(log_dir: str, metrics: Dict[str, Any]) -> None:
    candidates: list[str] = []
    if log_dir:
        candidates.append(os.path.join(log_dir, "tensorboard"))
        parent = os.path.dirname(log_dir.rstrip(os.sep)) if log_dir.rstrip(os.sep) else ""
        if parent and parent != log_dir:
            candidates.append(os.path.join(parent, "tensorboard"))

    tb_dir = next((path for path in candidates if os.path.isdir(path)), None)
    if not tb_dir:
        return

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        return

    step = int(metrics.get("n_steps", 0) or 0)
    writer = SummaryWriter(tb_dir)
    try:
        sharpe = metrics.get("sharpe_ratio")
        if sharpe is not None:
            writer.add_scalar("returns/episode_sharpe", float(sharpe), step)

        sortino = metrics.get("sortino_ratio")
        if sortino is not None:
            writer.add_scalar("returns/episode_sortino", float(sortino), step)

        annual_return = metrics.get("annualized_return")
        if annual_return is not None:
            writer.add_scalar("returns/annualized_return", float(annual_return), step)

        max_dd = metrics.get("max_drawdown")
        if max_dd is not None:
            writer.add_scalar("risk/max_drawdown", float(max_dd), step)

        writer.flush()
    finally:
        writer.close()


def test_irt(args, model_path=None):
    """IRT 모델 평가"""

    print("=" * 70)
    print(f"IRT Evaluation")
    print("=" * 70)

    set_global_seed(args.seed)

    # Model path
    if model_path is None:
        if args.checkpoint is None:
            raise ValueError("--checkpoint required for --mode test")
        model_path = args.checkpoint

    print(f"\n[Config]")
    print(f"  Model: {model_path}")
    print(f"  Test: {args.test_start} ~ {args.test_end}")

    # 메타데이터 로드 (관측 공간 검증 및 설정 유지)
    # Phase 1.5: 체크포인트 config 우선
    env_meta = {}
    expected_obs_dim = None
    use_weighted_action = True
    weight_slippage = 0.001
    weight_transaction_cost = 0.0005
    reward_scaling = args.reward_scale
    adaptive_lambda_sharpe = args.adaptive_lambda_sharpe
    adaptive_lambda_cvar = args.adaptive_lambda_cvar
    adaptive_lambda_turnover = args.adaptive_lambda_turnover
    adaptive_crisis_gain_sharpe = args.adaptive_crisis_gain_sharpe
    adaptive_crisis_gain_cvar = args.adaptive_crisis_gain_cvar
    adaptive_dsr_beta = args.adaptive_dsr_beta
    adaptive_cvar_window = args.adaptive_cvar_window
    meta_path = os.path.join(os.path.dirname(model_path), "env_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as meta_fp:
            env_meta = json.load(meta_fp)
        expected_obs_dim = env_meta.get("obs_dim")
        use_weighted_action = env_meta.get("use_weighted_action", use_weighted_action)
        weight_slippage = env_meta.get("weight_slippage", weight_slippage)
        weight_transaction_cost = env_meta.get("weight_transaction_cost", weight_transaction_cost)
        reward_scaling = env_meta.get("reward_scaling", reward_scaling)
        adaptive_lambda_sharpe = env_meta.get("adaptive_lambda_sharpe", adaptive_lambda_sharpe)
        adaptive_lambda_cvar = env_meta.get("adaptive_lambda_cvar", adaptive_lambda_cvar)
        adaptive_lambda_turnover = env_meta.get("adaptive_lambda_turnover", adaptive_lambda_turnover)
        adaptive_crisis_gain_sharpe = env_meta.get(
            "adaptive_crisis_gain_sharpe",
            env_meta.get("adaptive_crisis_gain", adaptive_crisis_gain_sharpe),
        )
        adaptive_crisis_gain_cvar = env_meta.get(
            "adaptive_crisis_gain_cvar",
            adaptive_crisis_gain_cvar,
        )
        adaptive_dsr_beta = env_meta.get("adaptive_dsr_beta", adaptive_dsr_beta)
        adaptive_cvar_window = env_meta.get("adaptive_cvar_window", adaptive_cvar_window)

        # Phase 1.5: 체크포인트 reward_type 우선 (CLI보다 우선)
        if env_meta.get("reward_type"):
            if args.reward_type and args.reward_type != env_meta["reward_type"]:
                print(f"  Warning: Overriding CLI reward_type ({args.reward_type}) with checkpoint ({env_meta['reward_type']})")
            args.reward_type = env_meta["reward_type"]
        args.adaptive_lambda_sharpe = adaptive_lambda_sharpe
        args.adaptive_lambda_cvar = adaptive_lambda_cvar
        args.adaptive_lambda_turnover = adaptive_lambda_turnover
        args.adaptive_crisis_gain_sharpe = adaptive_crisis_gain_sharpe
        args.adaptive_crisis_gain_cvar = adaptive_crisis_gain_cvar
        args.adaptive_dsr_beta = adaptive_dsr_beta
        args.adaptive_cvar_window = adaptive_cvar_window

    # evaluate.py의 evaluate_model() 재사용
    portfolio_values, exec_returns, value_returns, irt_data, metrics = evaluate_model(
        model_path=model_path,
        model_class=SAC,
        test_start=args.test_start,
        test_end=args.test_end,
        stock_tickers=DOW_30_TICKER,
        tech_indicators=INDICATORS,
        initial_amount=1000000,
        verbose=True,
        reward_type=args.reward_type,  # Phase-H1: reward_type 전달
        lambda_dsr=args.lambda_dsr,
        lambda_cvar=args.lambda_cvar,
        expected_obs_dim=expected_obs_dim,
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

    # 결과 추출
    final_value = portfolio_values[-1]
    total_return = metrics["total_return"]
    step = metrics["n_steps"]

    print(f"\n" + "=" * 70)
    print(f"Evaluation Results")
    print("=" * 70)
    print(f"\n[Period]")
    print(f"  Start: {args.test_start}")
    print(f"  End: {args.test_end}")
    print(f"  Steps: {step}")

    print(f"\n[Returns]")
    print(f"  Total Return: {total_return*100:.2f}%")
    print(f"  Annualized Return: {metrics['annualized_return']*100:.2f}%")

    print(f"\n[Risk Metrics]")
    print(f"  Volatility (annualized): {metrics['volatility']*100:.2f}%")
    print(f"  Maximum Drawdown: {metrics['max_drawdown']*100:.2f}%")

    print(f"\n[Risk-Adjusted Returns]")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"  Sortino Ratio: {metrics['sortino_ratio']:.3f}")
    print(f"  Calmar Ratio: {metrics['calmar_ratio']:.3f}")

    print(f"\n[Portfolio Value]")
    print(f"  Initial: $1,000,000.00")
    print(f"  Final: ${final_value:,.2f}")
    print(f"  Profit/Loss: ${final_value - 1000000:,.2f}")

    print(f"\n" + "=" * 70)

    _log_returns_to_tensorboard(os.path.dirname(model_path), metrics)

    # 6. 시각화 (기본 활성화)
    if not args.no_plot and irt_data is not None:
        from finrl.evaluation.visualizer import plot_all

        # log_dir 결정
        if model_path:
            log_dir = os.path.dirname(model_path)
        elif args.checkpoint:
            log_dir = os.path.dirname(args.checkpoint)
        else:
            log_dir = "evaluation_results"

        output_dir = args.output_plot or os.path.join(log_dir, "evaluation_plots")

        print(f"\n[시각화 생성]")
        print(f"  Output: {output_dir}")

        plot_all(
            portfolio_values=np.array(portfolio_values),
            dates=None,
            output_dir=output_dir,
            irt_data=irt_data,
        )

    # 7. JSON 저장 (기본 활성화)
    if not args.no_json and irt_data is not None:
        from finrl.evaluation.visualizer import save_evaluation_results
        from pathlib import Path

        # log_dir 결정
        if model_path:
            log_dir = os.path.dirname(model_path)
        elif args.checkpoint:
            log_dir = os.path.dirname(args.checkpoint)
        else:
            log_dir = "evaluation_results"

        # evaluation_results 구조 구축
        returns = np.diff(portfolio_values) / portfolio_values[:-1]

        results = {
            "returns": returns,
            "values": np.array(portfolio_values),
            "weights": irt_data["weights"],  # 목표가중
            "actual_weights": irt_data["actual_weights"],  # Phase-1: 실행가중
            "crisis_levels": irt_data["crisis_levels"],
            "crisis_types": irt_data["crisis_types"],
            "prototype_weights": irt_data["prototype_weights"],
            "w_rep": irt_data["w_rep"],
            "w_ot": irt_data["w_ot"],
            "eta": irt_data["eta"],
            "alpha_c": irt_data["alpha_c"],
            "cost_matrices": irt_data["cost_matrices"],
            "symbols": irt_data["symbols"],
            "metrics": metrics,
        }

        # Config (IRT 파라미터 포함)
        config = {
            "irt": {
                "alpha": args.alpha,
                "alpha_min": args.alpha_min,
                "alpha_max": args.alpha_max if args.alpha_max else args.alpha,
                "eps": args.eps,
                "max_iters": args.max_iters,
                "replicator_temp": args.replicator_temp,
                "ema_beta": args.ema_beta,
                "eta_0": args.eta_0,
                "eta_1": args.eta_1,
                "gamma": args.gamma,
                "alpha_update_rate": args.alpha_update_rate,
                "alpha_feedback_gain": args.alpha_feedback_gain,
                "alpha_feedback_bias": args.alpha_feedback_bias,
                "directional_decay_min": args.directional_decay_min,
                "alpha_noise_std": args.alpha_noise_std,
                "alpha_crisis_source": args.alpha_crisis_source,
                # Phase B
                "w_r": args.w_r,
                "w_s": args.w_s,
                "w_c": args.w_c,
                "eta_b": args.eta_b,
            }
        }

        print(f"\n[JSON 저장]")
        print(f"  Output: {log_dir}")
        save_evaluation_results(results, Path(log_dir), config)

    return {
        "portfolio_values": portfolio_values,
        "final_value": final_value,
        "total_return": total_return,
        "metrics": metrics,
        "steps": step,
    }


def main():
    parser = argparse.ArgumentParser(description="IRT Policy Training/Evaluation")

    # Mode
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval", "both", "test"],
        help="Execution mode (default: train; use 'eval' for evaluation only)",
    )

    # Data period (defaults from config.py)
    parser.add_argument(
        "--train-start",
        type=str,
        default=TRAIN_START_DATE,
        help=f"Training start date (default: {TRAIN_START_DATE})",
    )
    parser.add_argument(
        "--train-end",
        type=str,
        default=TRAIN_END_DATE,
        help=f"Training end date (default: {TRAIN_END_DATE})",
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

    # Training settings
    parser.add_argument(
        "--episodes", type=int, default=60, help="Number of training episodes (default: 60)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output", type=str, default="logs", help="Output directory (default: logs)"
    )
    parser.add_argument(
        "--reward-scale",
        type=float,
        default=1.0,  # Phase 1: 1e-2 → 1.0 (100x increase for proper gradient flow)
        help="Reward scaling factor applied inside the trading environment (default: 1.0)",
    )
    parser.add_argument(
        "--env-diversify",
        choices=["off", "basic", "vector"],
        default="vector",
        help="Environment diversification strategy (default: vector)",
    )
    parser.add_argument(
        "--advanced",
        action="store_true",
        help="Display advanced configuration options and exit",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=None,
        help="Number of parallel environments when env-diversify=vector (default: auto)",
    )
    parser.add_argument(
        "--start-offset-mode",
        choices=["random", "uniform", "rolling"],
        default="random",
        help="Offset sampling mode for diversified training environments (default: random)",
    )
    parser.add_argument(
        "--window-step",
        type=int,
        default=252,
        help="Window step (days) when env-diversify uses rolling offsets (default: 252)",
    )
    parser.add_argument(
        "--domain-rand-tx",
        type=float,
        default=0.25,
        help="Uniform domain randomization magnitude for transaction cost scaling (default: 0.25)",
    )
    parser.add_argument(
        "--domain-rand-slippage",
        type=float,
        default=0.25,
        help="Uniform domain randomization magnitude for slippage scaling (default: 0.25)",
    )

    # IRT parameters
    parser.add_argument(
        "--emb-dim",
        type=int,
        default=128,
        help="IRT embedding dimension (default: 128)",
    )
    parser.add_argument(
        "--m-tokens", type=int, default=6, help="Number of epitope tokens (default: 6)"
    )
    parser.add_argument(
        "--M-proto", type=int, default=8, help="Number of prototypes (default: 8)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Base OT-Replicator mixing ratio (default: 0.45, Phase F)",
    )
    parser.add_argument(
        "--alpha-min",
        type=float,
        default=0.05,
        help="Crisis minimum alpha (default: 0.05, Phase 3.5)",
    )
    parser.add_argument(
        "--alpha-max",
        type=float,
        default=0.75,  # Phase 2: 0.55 → 0.75 (allow more OT usage, prevent saturation)
        help="Normal maximum alpha (Phase 2: increased for IRT flexibility)",
    )
    parser.add_argument(
        "--alpha-update-rate",
        type=float,
        default=0.85,
        help="Update rate for alpha_c state adaptation (default: 0.85)",
    )
    parser.add_argument(
        "--alpha-feedback-gain",
        type=float,
        default=0.25,
        help="Sharpe feedback gain for alpha_c (default: 0.25)",
    )
    parser.add_argument(
        "--alpha-feedback-bias",
        type=float,
        default=0.0,
        help="Sharpe feedback bias for alpha_c (default: 0.0)",
    )
    parser.add_argument(
        "--directional-decay-min",
        type=float,
        default=0.05,
        help="Minimum directional decay factor near alpha bounds (default: 0.05)",
    )
    parser.add_argument(
        "--alpha-noise-std",
        type=float,
        default=0.02,
        help="Gaussian noise std added to alpha during training (default: 0.02)",
    )
    parser.add_argument(
        "--alpha-crisis-source",
        type=str,
        default="pre_guard",
        choices=["pre_guard", "post_guard"],
        help="Which crisis signal source controls alpha_c (default: pre_guard)",
    )
    parser.add_argument(
        "--ema-beta",
        type=float,
        default=0.50,
        help="EMA memory coefficient (default: 0.50, Phase 1.5)",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.05,
        help="Sinkhorn entropy (default: 0.05, Phase 1)",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=30,
        help="Sinkhorn max iterations (default: 30, Phase 1)",
    )
    parser.add_argument(
        "--replicator-temp",
        type=float,
        default=0.4,
        help="Replicator softmax temperature (default: 0.4 for stronger selection)",
    )
    parser.add_argument(
        "--eta-0",
        type=float,
        default=0.05,
        help="Base learning rate (Replicator) (default: 0.05)",
    )
    parser.add_argument(
        "--eta-1",
        type=float,
        default=0.12,
        help="Crisis increase (Replicator) (default: 0.12, Phase E)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.90,
        help="Co-stimulation weight in cost function (default: 0.90, Phase 1.5)",
    )
    parser.add_argument(
        "--market-feature-dim",
        type=int,
        default=12,
        help="Market feature dimension (default: 12)",
    )
    # Phase 2: Dirichlet 및 온도 파라미터
    parser.add_argument(
        "--dirichlet-min",
        type=float,
        default=0.05,
        help="Dirichlet concentration minimum (default: 0.05 for sharper selectivity)"
    )
    parser.add_argument(
        "--dirichlet-max",
        type=float,
        default=6.0,
        help="Dirichlet concentration maximum (encourages sharper prototypes)"
    )
    parser.add_argument(
        "--action-temp",
        type=float,
        default=0.50,
        help="Action softmax temperature (default: 0.50 for smoother regime transitions)"
    )

    # Phase 3: DSR + CVaR 보상 파라미터
    parser.add_argument(
        "--reward-type",
        type=str,
        default="adaptive_risk",
        choices=["basic", "dsr_cvar", "adaptive_risk"],
        help="Reward function (default: adaptive_risk)",
    )
    parser.add_argument(
        "--lambda-dsr",
        type=float,
        default=0.15,
        help="DSR bonus weight (default: 0.15, Phase C)",
    )
    parser.add_argument(
        "--lambda-cvar",
        type=float,
        default=0.05,
        help="CVaR penalty weight (default: 0.05, Phase 3)",
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
        help="Adaptive risk reward CVaR penalty weight β (default: 0.40)",
    )
    parser.add_argument(
        "--adaptive-lambda-turnover",
        type=float,
        default=0.0,
        help="Adaptive risk reward turnover penalty μ (default: 0.0; set >0 only if not already deducted in NAV)",
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
        help="Adaptive risk reward CVaR crisis gain g_C (default: 0.25, increase penalty in crises)",
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

    # Phase E: 위기신호 가중 재균형
    parser.add_argument(
        "--w-r",
        type=float,
        default=0.55,
        help="Market crisis signal weight (T-Cell output) (Phase 1 default: 0.55)",
    )
    parser.add_argument(
        "--w-s",
        type=float,
        default=-0.25,
        help="Sharpe signal weight (DSR bonus) (Phase 1 default: -0.25)",
    )
    parser.add_argument(
        "--w-c",
        type=float,
        default=0.20,
        help="CVaR signal weight (Phase 1 default: 0.20)",
    )
    parser.add_argument(
        "--eta-b",
        type=float,
        default=2e-2,
        help="Initial bias learning rate for crisis calibration (Phase 1 default: 2e-2)",
    )
    parser.add_argument(
        "--eta-b-min",
        type=float,
        default=2e-3,
        help="Minimum bias learning rate after decay (Phase 1 default: 2e-3)",
    )
    parser.add_argument(
        "--eta-b-decay-steps",
        type=int,
        default=30000,
        help="Cosine decay horizon for bias learning rate (Phase 1 default: 30000)",
    )
    parser.add_argument(
        "--eta-b-warmup-steps",
        type=int,
        default=10000,
        help="Warmup steps to keep bias learning rate at warmup value (Phase 1 default: 10000)",
    )
    parser.add_argument(
        "--eta-b-warmup-value",
        type=float,
        default=0.05,
        help="Warmup bias learning rate before cosine decay (Phase 1 default: 0.05)",
    )
    parser.add_argument(
        "--eta-T",
        dest="eta_T",
        type=float,
        default=1e-2,
        help="Temperature adaptation rate (Phase 1 default: 0.01)",
    )
    parser.add_argument(
        "--p-star",
        type=float,
        default=0.35,
        help="Target crisis prevalence p* (Phase 1 default: 0.35)",
    )
    parser.add_argument(
        "--temperature-min",
        type=float,
        default=0.9,
        help="Minimum adaptive temperature clamp (Phase 1 default: 0.9)",
    )
    parser.add_argument(
        "--temperature-max",
        type=float,
        default=1.2,
        help="Maximum adaptive temperature clamp (Phase 1 default: 1.2)",
    )
    parser.add_argument(
        "--stat-momentum",
        type=float,
        default=0.92,
        help="EMA momentum for crisis signal statistics (default: 0.92)",
    )
    parser.add_argument(
        "--crisis-guard-rate-init",
        type=float,
        default=0.07,
        help="Initial crisis guard rate during warmup (default: 0.07)",
    )
    parser.add_argument(
        "--crisis-guard-rate-final",
        type=float,
        default=0.02,
        help="Final crisis guard rate after warmup (default: 0.02)",
    )
    parser.add_argument(
        "--crisis-guard-warmup-steps",
        type=int,
        default=7500,
        help="Warmup steps for crisis guard rate schedule (default: 7500)",
    )
    parser.add_argument(
        "--hysteresis-up",
        type=float,
        default=0.52,
        help="Crisis entry threshold when previous regime is normal (default: 0.52)",
    )
    parser.add_argument(
        "--hysteresis-down",
        type=float,
        default=0.42,
        help="Crisis exit threshold when previous regime is crisis (default: 0.42)",
    )
    parser.add_argument(
        "--hysteresis-quantile",
        type=float,
        default=0.72,
        help="Quantile target for adaptive hysteresis thresholds (default: 0.72)",
    )
    parser.add_argument(
        "--k-s",
        dest="k_s",
        type=float,
        default=6.0,
        help="Sigmoid slope for Sharpe z-score transform (Phase 1 default: 6.0)",
    )
    parser.add_argument(
        "--k-c",
        dest="k_c",
        type=float,
        default=6.0,
        help="Sigmoid slope for CVaR z-score transform (Phase 1 default: 6.0)",
    )
    parser.add_argument(
        "--k-b",
        dest="k_b",
        type=float,
        default=4.0,
        help="Sigmoid slope for base crisis transform (Phase 1 default: 4.0)",
    )
    parser.add_argument(
        "--crisis-guard-rate",
        type=float,
        default=None,
        help="Legacy static crisis guard rate override (optional)",
    )

    # Evaluation only
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path for evaluation (required for --mode test)",
    )

    # Visualization (기본값: 모두 저장)
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable saving evaluation plots (default: enabled)",
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        default=None,
        help="Plot output directory (default: log_dir/evaluation_plots)",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Disable saving evaluation results as JSON (default: enabled)",
    )

    base_visible = {"--mode", "--reward-type", "--episodes", "--env-diversify", "--advanced"}
    advanced_descriptions: List[Tuple[str, str]] = []
    for action in list(parser._actions):
        option_strings = tuple(action.option_strings)
        if not option_strings:
            continue
        if any(opt in ("-h", "--help") for opt in option_strings):
            continue
        if any(opt in base_visible for opt in option_strings):
            continue
        if action.help not in (None, argparse.SUPPRESS):
            label = ", ".join(option_strings)
            advanced_descriptions.append((label, action.help))
        action.help = argparse.SUPPRESS

    args = parser.parse_args()

    if getattr(args, "advanced", False):
        print("Advanced options:\n")
        for label, description in sorted(advanced_descriptions, key=lambda item: item[0]):
            print(f"  {label}\n    {description}")
        return

    if args.mode == "test":
        args.mode = "eval"

    if args.env_diversify == "vector":
        if args.n_envs is None:
            args.n_envs = 8
    elif args.n_envs is None:
        args.n_envs = 1

    banner = (
        f"reward={args.reward_type} | "
        f"episodes={args.episodes} | "
        f"diversify={args.env_diversify} | "
        f"n_envs={args.n_envs}"
    )
    print(f"[Run] {banner}")

    # Execute
    if args.mode == "train":
        log_dir, model_path = train_irt(args)
        print(f"\nCompleted! Results: {log_dir}")

    elif args.mode == "eval":
        results = test_irt(args)
        print(f"\nCompleted! Final return: {results['total_return']*100:.2f}%")

    elif args.mode == "both":
        log_dir, model_path = train_irt(args)
        print(f"\nTraining completed. Starting evaluation...\n")
        results = test_irt(args, model_path=model_path)
        print(f"\nAll tasks completed!")
        print(f"  Training results: {log_dir}")
        print(f"  Final return: {results['total_return']*100:.2f}%")


if __name__ == "__main__":
    main()
