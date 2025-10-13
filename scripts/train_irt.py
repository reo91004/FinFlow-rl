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
    python scripts/train_irt.py --mode test --checkpoint logs/irt/.../irt_final.zip
"""

import argparse
import json
import os
import random
from datetime import datetime
import numpy as np

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
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import torch

# Import IRT Policy
from finrl.agents.irt import IRTPolicy

# Import evaluation function
import sys

sys.path.insert(0, os.path.dirname(__file__))
from evaluate import evaluate_model


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
    if hasattr(env, "seed"):
        try:
            env.seed(seed)
        except TypeError:
            env.seed()
    elif hasattr(env, "_seed"):
        env._seed(seed)

    # Gymnasium style reset(seed=seed)
    if hasattr(env, "reset"):
        try:
            env.reset(seed=seed)
        except TypeError:
            env.reset()

    if hasattr(env, "action_space") and hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    if hasattr(env, "observation_space") and hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)


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


class IRTLoggingCallback(BaseCallback):
    """
    Phase D: IRT 중간 변수 텐서보드 로깅 (교정)

    100 스텝마다 IRT 내부 변수를 기록:
    - crisis_level: 위기 감지 레벨
    - alpha_c: 동적 OT-Replicator 혼합 비율
    - rep_contribution/ot_contribution: 혼합 후 실제 기여도
    - entropy: 프로토타입 혼합 다양성
    - prototype_max_weight: 프로토타입 과점 지표

    Phase 3.5 추가:
    - alpha_c_raw, pct_clamped_min/max: α clamp 검증
    - action_temp, ema_beta: 하이퍼파라미터 확인
    - turnover 관련 메트릭 (구현 예정)
    """

    def __init__(self, verbose: int = 0, log_freq: int = 100, ema_beta: float = 0.95):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.ema_beta = ema_beta

        # 타임축 통계 추적 (EMA)
        self.alpha_c_ema_mean = None  # EMA(alpha_c)
        self.alpha_c_ema_var = None   # EMA((alpha_c - mean)^2)

        # Phase 3.5: 이전 action 저장 (turnover 계산용)
        self.prev_action = None

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            # IRT info 가져오기
            info = self.model.policy.get_irt_info()
            if info is not None:
                # ===== 위기 레벨 =====
                self.logger.record("irt/avg_crisis_level", info['crisis_level'].mean().item())

                # ===== alpha_c (OT-Replicator 혼합 비율) =====
                alpha_c_val = info['alpha_c'].mean().item()
                self.logger.record("irt/avg_alpha_c", alpha_c_val)

                # Phase D: 타임축 EMA 분산 (B=1 가드 제거)
                if self.alpha_c_ema_mean is None:
                    # 초기화
                    self.alpha_c_ema_mean = alpha_c_val
                    self.alpha_c_ema_var = 0.0
                else:
                    # EMA 업데이트
                    delta = alpha_c_val - self.alpha_c_ema_mean
                    self.alpha_c_ema_mean = self.ema_beta * self.alpha_c_ema_mean + (1 - self.ema_beta) * alpha_c_val
                    self.alpha_c_ema_var = self.ema_beta * self.alpha_c_ema_var + (1 - self.ema_beta) * (delta ** 2)

                std_alpha_c = self.alpha_c_ema_var ** 0.5 if self.alpha_c_ema_var is not None else 0.0
                self.logger.record("irt/std_alpha_c", std_alpha_c)

                # ===== Phase D 교정: Replicator vs OT 실제 기여도 =====
                # w = (1 - alpha_c) * tilde_w + alpha_c * p_mass
                # 따라서 혼합 후 기여도는:
                # rep_contribution = (1 - alpha_c).mean()
                # ot_contribution = alpha_c.mean()
                alpha_c_tensor = info['alpha_c']  # [B, 1]
                rep_contribution = (1 - alpha_c_tensor).mean().item()
                ot_contribution = alpha_c_tensor.mean().item()

                self.logger.record("irt/rep_contribution", rep_contribution)
                self.logger.record("irt/ot_contribution", ot_contribution)

                # 검증: rep + ot ≈ 1.0
                contribution_sum = rep_contribution + ot_contribution
                self.logger.record("irt/contribution_sum", contribution_sum)

                # ===== 프로토타입 혼합 통계 =====
                w = info['w']  # [B, M]

                # 엔트로피 (다양성 지표)
                entropy = -torch.sum(w * torch.log(w + 1e-8), dim=-1).mean().item()
                self.logger.record("irt/avg_entropy", entropy)

                # Phase D: 프로토타입 과점 지표
                # 평균 최대 가중치 (배치 평균)
                max_proto_weight = w.max(dim=-1)[0].mean().item()
                self.logger.record("irt/max_proto_weight", max_proto_weight)

                # 프로토타입별 평균 가중치 (M개 프로토타입)
                prototype_avg_weights = w.mean(dim=0)  # [M]
                prototype_max_weight_value = prototype_avg_weights.max().item()
                self.logger.record("irt/prototype_max_weight", prototype_max_weight_value)

                # 프로토타입 가중치 엔트로피 (프로토타입 축 다양성)
                prototype_entropy = -torch.sum(prototype_avg_weights * torch.log(prototype_avg_weights + 1e-8)).item()
                self.logger.record("irt/prototype_entropy", prototype_entropy)

                # ===== Phase 3.5: α clamp 검증 =====
                if 'alpha_c_raw' in info:
                    alpha_c_raw_val = info['alpha_c_raw'].mean().item()
                    self.logger.record("irt/alpha_c_raw", alpha_c_raw_val)

                    pct_clamped_min = info['pct_clamped_min'].item() if isinstance(info['pct_clamped_min'], torch.Tensor) else info['pct_clamped_min']
                    pct_clamped_max = info['pct_clamped_max'].item() if isinstance(info['pct_clamped_max'], torch.Tensor) else info['pct_clamped_max']

                    self.logger.record("irt/pct_clamped_min", pct_clamped_min)
                    self.logger.record("irt/pct_clamped_max", pct_clamped_max)

                # ===== Phase 3.5: 정책 하이퍼파라미터 확인 =====
                try:
                    actor = self.model.policy.actor
                    if hasattr(actor, 'action_temp'):
                        self.logger.record("policy/action_temp", actor.action_temp)
                    if hasattr(actor, 'ema_beta'):
                        self.logger.record("policy/ema_beta", actor.ema_beta)
                except Exception:
                    pass  # IRTPolicy 구조에 따라 실패 가능

                # ===== Phase 3.5: Turnover 메트릭 (목표 가중치 기준) =====
                # 현재 action은 self.locals에 저장되어 있음
                if 'actions' in self.locals and self.prev_action is not None:
                    current_action = self.locals['actions']
                    if isinstance(current_action, torch.Tensor):
                        current_action_np = current_action.detach().cpu().numpy()
                    else:
                        current_action_np = current_action

                    # Turnover = 0.5 * sum(|w_t - w_{t-1}|) (metrics.py와 동일)
                    # current_action_np shape: [B, N] (B=1 for DummyVecEnv)
                    turnover_target = 0.5 * np.sum(np.abs(current_action_np - self.prev_action))
                    self.logger.record("irt/turnover_target", turnover_target)

                    # 업데이트
                    self.prev_action = current_action_np
                elif 'actions' in self.locals:
                    # 첫 스텝: 초기화
                    current_action = self.locals['actions']
                    if isinstance(current_action, torch.Tensor):
                        self.prev_action = current_action.detach().cpu().numpy()
                    else:
                        self.prev_action = current_action

        return True


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
        try:
            policy = getattr(self.model, "policy", None)
            if policy is not None and hasattr(policy, "get_irt_info"):
                info = policy.get_irt_info()
        except Exception as exc:
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

        try:
            base_env = _unwrap_env(self.model.get_env())
            setattr(base_env, "_crisis_level", crisis_level)

            risk_reward = getattr(base_env, "risk_reward", None)
            if risk_reward is not None and hasattr(risk_reward, "set_crisis_level"):
                risk_reward.set_crisis_level(crisis_level)

            if hasattr(base_env, "_crisis_history"):
                base_env._crisis_history.append((int(self.model.num_timesteps), crisis_level))

            if not self._connected_once:
                print(f"✅ CrisisBridgeCallback: Successfully connected (t={self.model.num_timesteps})")
                self._connected_once = True
                self._warned_once = False
        except Exception as exc:
            if not self._warned_once:
                print(f"⚠️  CrisisBridgeCallback: Failed to update crisis level ({exc})")
                self._warned_once = True

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
    }

    return StockTradingEnv(**env_kwargs)


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
    train_env = create_env(
        train_df, stock_dim, INDICATORS,
        reward_type=args.reward_type,
        lambda_dsr=args.lambda_dsr,
        lambda_cvar=args.lambda_cvar,
        reward_scaling=args.reward_scale,
    )
    seed_environment(train_env, args.seed)
    print(
        f"  Train env: reward_scale={train_env.reward_scaling}, "
        f"use_weighted_action={getattr(train_env, 'use_weighted_action', False)}, "
        f"slippage={getattr(train_env, 'weight_slippage', None)}, "
        f"tx_cost={getattr(train_env, 'weight_transaction_cost', None)}"
    )
    test_env = create_env(
        test_df, stock_dim, INDICATORS,
        reward_type=args.reward_type,
        lambda_dsr=args.lambda_dsr,
        lambda_cvar=args.lambda_cvar,
        reward_scaling=args.reward_scale,
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

    print(f"  State space: {train_env.state_space}")
    print(f"  Action space: {train_env.action_space}")
    print(f"  Reward type: {args.reward_type}")
    if args.reward_type == "dsr_cvar":
        print(f"    λ_dsr: {args.lambda_dsr}, λ_cvar: {args.lambda_cvar}")
    elif args.reward_type == "adaptive_risk":
        print(f"    Phase-H1: Adaptive Risk-Aware Reward")
        print(f"    λ_sharpe: 0.15, λ_cvar: 0.5, λ_turnover: 0.002, crisis_gain: 0.25")

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

    eval_callback = EvalCallback(
        Monitor(test_env),
        best_model_save_path=os.path.join(log_dir, "best_model"),
        log_path=os.path.join(log_dir, "eval"),
        eval_freq=5000,
        deterministic=True,
        render=False,
    )

    # Phase A: IRT 계측 callback
    irt_logging_callback = IRTLoggingCallback(verbose=0, log_freq=100)

    # IRT Policy kwargs
    policy_kwargs = {
        "emb_dim": args.emb_dim,
        "m_tokens": args.m_tokens,
        "M_proto": args.M_proto,
        "alpha": args.alpha,
        "alpha_min": args.alpha_min,
        "alpha_max": args.alpha_max if args.alpha_max else args.alpha,
        "ema_beta": args.ema_beta,
        "market_feature_dim": args.market_feature_dim,
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
        # Phase 3.5 Step 2: 다중 신호 위기 감지
        "w_r": args.w_r,
        "w_s": args.w_s,
        "w_c": args.w_c,
        # Phase B: 바이어스 EMA 보정
        "eta_b": args.eta_b,
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
        tensorboard_log=os.path.join(log_dir, "tensorboard"),
    )

    # Total timesteps
    total_timesteps = 250 * args.episodes
    print(f"  Total timesteps: {total_timesteps}")
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
        "use_weighted_action": getattr(train_env, "use_weighted_action", True),
        "weight_slippage": getattr(train_env, "weight_slippage", 0.001),
        "weight_transaction_cost": getattr(train_env, "weight_transaction_cost", 0.0005),
        "reward_scaling": args.reward_scale,
        "train_start": args.train_start,
        "train_end": args.train_end,
        "test_start": args.test_start,
        "test_end": args.test_end,
    }
    with open(os.path.join(log_dir, "env_meta.json"), "w") as meta_fp:
        json.dump(env_meta, meta_fp, indent=2)

    return log_dir, final_model_path


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
    env_meta = {}
    expected_obs_dim = None
    use_weighted_action = True
    weight_slippage = 0.001
    weight_transaction_cost = 0.0005
    reward_scaling = args.reward_scale
    meta_path = os.path.join(os.path.dirname(model_path), "env_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as meta_fp:
            env_meta = json.load(meta_fp)
        expected_obs_dim = env_meta.get("obs_dim")
        use_weighted_action = env_meta.get("use_weighted_action", use_weighted_action)
        weight_slippage = env_meta.get("weight_slippage", weight_slippage)
        weight_transaction_cost = env_meta.get("weight_transaction_cost", weight_transaction_cost)
        reward_scaling = env_meta.get("reward_scaling", reward_scaling)
        if env_meta.get("reward_type") and args.reward_type is None:
            args.reward_type = env_meta["reward_type"]

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
        default="both",
        choices=["train", "test", "both"],
        help="Execution mode (default: both)",
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
        "--episodes", type=int, default=600, help="Number of episodes (default: 600, Phase 3.5: 150k steps)"
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
        "--ema-beta",
        type=float,
        default=0.65,
        help="EMA memory coefficient (default: 0.65, Phase 3.5)",
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
        default=1.4,
        help="Replicator softmax temperature (default: 1.4, Phase F, >1 for diversity)",
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
        default=0.65,
        help="Co-stimulation weight in cost function (default: 0.65, Phase 3.5)",
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
        default=0.1,  # Phase 2: 0.8 → 0.1 (allow sparsity for exploration)
        help="Dirichlet concentration minimum (Phase 2: enables sparse exploration)"
    )
    parser.add_argument(
        "--dirichlet-max",
        type=float,
        default=5.0,  # Phase 2: 2.0 → 5.0 (keep current, only for stochastic)
        help="Dirichlet concentration maximum (Phase 2: stochastic path only)"
    )
    parser.add_argument(
        "--action-temp",
        type=float,
        default=0.3,  # Phase 2: 0.9 → 0.3 (much sharper for concentrated portfolios)
        help="Action softmax temperature (Phase 2: sharper = more concentration)"
    )

    # Phase 3: DSR + CVaR 보상 파라미터
    parser.add_argument(
        "--reward-type",
        type=str,
        default="dsr_cvar",
        choices=["basic", "dsr_cvar", "adaptive_risk"],
        help="Reward function type (default: dsr_cvar, Phase-H1: adaptive_risk)",
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

    # Phase E: 위기신호 가중 재균형
    parser.add_argument(
        "--w-r",
        type=float,
        default=0.8,
        help="Market crisis signal weight (T-Cell output) (default: 0.8, Phase E)",
    )
    parser.add_argument(
        "--w-s",
        type=float,
        default=0.15,
        help="Sharpe signal weight (DSR bonus) (default: 0.15, Phase E)",
    )
    parser.add_argument(
        "--w-c",
        type=float,
        default=0.05,
        help="CVaR signal weight (default: 0.05, Phase E)",
    )
    parser.add_argument(
        "--eta-b",
        type=float,
        default=2e-3,
        help="Bias learning rate for crisis neutralization (default: 2e-3, Phase B)",
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

    args = parser.parse_args()

    # Execute
    if args.mode == "train":
        log_dir, model_path = train_irt(args)
        print(f"\nCompleted! Results: {log_dir}")

    elif args.mode == "test":
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
