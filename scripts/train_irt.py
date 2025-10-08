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
import os
from datetime import datetime
import numpy as np
import torch

from finrl.config import (
    INDICATORS,
    SAC_PARAMS,
    TRAIN_START_DATE, TRAIN_END_DATE,
    TEST_START_DATE, TEST_END_DATE
)
from finrl.config_tickers import DOW_30_TICKER
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

# Import IRT Policy
from finrl.agents.irt import IRTPolicy

# ===== Phase 1.9: Tier 0 - 적응형 타겟 엔트로피 =====
from finrl.agents.irt.entropy_estimator import PolicyEntropyEstimator

# ===== Phase 1.9: Tier 2 - 동적 Alpha 스케줄러 =====
from finrl.agents.irt.alpha_scheduler import AlphaScheduler

# ===== 다목표 보상 래퍼 =====
from finrl.meta.env_portfolio_optimization.reward_wrapper import MultiObjectiveRewardWrapper


# ===== Tier 3: 전역 경사 클리핑을 적용한 SAC =====

class SACWithGradClip(SAC):
    """
    수치 안정성을 위한 전역 경사 클리핑을 적용한 SAC.

    References:
    - Neptune.ai (2025): "Gradient clipping is especially beneficial in RL"
    - RLC 2024: "Weight clipping for Deep Continual and Reinforcement Learning"
    - CE-GPPO (2025): Gradient clipping preserves stability in policy optimization
    """

    def __init__(self, *args, max_grad_norm=10.0, **kwargs):
        """
        Args:
            max_grad_norm: 최대 경사 노름 (기본값: 10.0)
        """
        self.max_grad_norm = max_grad_norm
        super().__init__(*args, **kwargs)

    def _setup_model(self):
        """옵티마이저에 경사 클리핑을 추가하도록 오버라이드"""
        super()._setup_model()

        # actor와 critic 옵티마이저를 경사 클리핑으로 래핑
        self._add_grad_clip_to_optimizer(self.policy.actor.optimizer)
        self._add_grad_clip_to_optimizer(self.policy.critic.optimizer)

    def _add_grad_clip_to_optimizer(self, optimizer):
        """파라미터 업데이트 전에 경사를 클리핑하도록 optimizer.step()을 래핑"""
        original_step = optimizer.step
        max_norm = self.max_grad_norm

        def step_with_clip(closure=None):
            # 옵티마이저 스텝 전에 경사 클리핑
            if max_norm is not None:
                params = []
                for param_group in optimizer.param_groups:
                    params.extend(param_group['params'])
                torch.nn.utils.clip_grad_norm_(params, max_norm)

            # 원래 옵티마이저 스텝 호출
            return original_step(closure)

        optimizer.step = step_with_clip


# ===== Phase 1.9 Tier 2: Alpha 업데이트 콜백 =====

from stable_baselines3.common.callbacks import BaseCallback

class AlphaUpdateCallback(BaseCallback):
    """
    학습 중 IRT alpha를 업데이트하는 콜백.

    Alpha 스케줄: 0.3 → 0.7 (Replicator → OT)
    - 초기 학습: Replicator 지배적 (빠른 수렴)
    - 후기 학습: OT 지배적 (구조적 매칭)
    """

    def __init__(self, scheduler: AlphaScheduler, verbose: int = 0):
        super().__init__(verbose)
        self.scheduler = scheduler

    def _on_step(self) -> bool:
        """각 스텝에서 alpha 업데이트."""
        # 현재 alpha 가져오기
        alpha = self.scheduler.get_alpha(self.num_timesteps)

        # IRT 연산자 업데이트
        # 경로: model.policy.actor.irt_actor.irt.alpha
        self.model.policy.actor.irt_actor.irt.alpha = alpha

        # 텐서보드에 기록
        self.logger.record("train/alpha", alpha)

        # 주기적 로깅
        if self.verbose > 0 and self.num_timesteps % 5000 == 0:
            print(f"  [Alpha] Step {self.num_timesteps}: α = {alpha:.3f}")

        return True


# 평가용 메트릭 임포트
from finrl.evaluation.metrics import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_max_drawdown,
    calculate_var,
    calculate_cvar,
    calculate_turnover
)


def create_env(df, stock_dim, tech_indicators):
    """StockTradingEnv 생성"""

    state_space = 1 + (len(tech_indicators) + 2) * stock_dim

    env_kwargs = {
        "df": df,
        "stock_dim": stock_dim,
        "hmax": 100,
        "initial_amount": 1000000,
        "num_stock_shares": [0] * stock_dim,
        "buy_cost_pct": [0.001] * stock_dim,
        "sell_cost_pct": [0.001] * stock_dim,
        "reward_scaling": 1e-4,
        "state_space": state_space,
        "action_space": stock_dim,
        "tech_indicator_list": tech_indicators,
        "print_verbosity": 500
    }

    return StockTradingEnv(**env_kwargs)


def train_irt(args):
    """IRT 모델 학습"""

    print("=" * 70)
    print(f"IRT Training - Dow Jones 30")
    print("=" * 70)

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
    print(f"  Output: {log_dir}")

    # 1. 데이터 다운로드
    print(f"\n[1/5] Downloading data...")
    df = YahooDownloader(
        start_date=args.train_start,
        end_date=args.test_end,
        ticker_list=DOW_30_TICKER
    ).fetch_data()
    print(f"  Downloaded: {df.shape[0]} rows")

    # 2. 특성 엔지니어링
    print(f"\n[2/5] Feature Engineering...")
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_turbulence=False,
        user_defined_feature=False
    )
    df_processed = fe.preprocess_data(df)
    print(f"  Features: {df_processed.shape[1]} columns")

    # 3. 학습/테스트 분할
    print(f"\n[3/5] Splitting data...")
    train_df = data_split(df_processed, args.train_start, args.train_end)
    test_df = data_split(df_processed, args.test_start, args.test_end)
    print(f"  Train: {len(train_df)} rows")
    print(f"  Test: {len(test_df)} rows")

    # 4. 환경 생성
    print(f"\n[4/5] Creating environments...")
    stock_dim = len(train_df.tic.unique())

    # 데이터 누락 경고
    if stock_dim != len(DOW_30_TICKER):
        removed_count = len(DOW_30_TICKER) - stock_dim
        print(f"  ⚠️  주의: {removed_count}개 주식이 데이터 부족으로 제외됨")
        print(f"      (2008년 초 데이터가 없는 종목: Visa (V) 등)")

    print(f"  실제 주식 수: {stock_dim}")
    train_env = create_env(train_df, stock_dim, INDICATORS)
    test_env = create_env(test_df, stock_dim, INDICATORS)

    # ===== Multi-Objective Reward Wrapper 적용 =====
    if args.use_multiobjective:
        print(f"\n  [Multi-Objective Reward] Wrapper 적용 중...")
        train_env = MultiObjectiveRewardWrapper(
            train_env,
            lambda_turnover=args.lambda_turnover,
            lambda_diversity=args.lambda_diversity,
            lambda_drawdown=args.lambda_drawdown,
            tc_rate=args.tc_rate,
            enable_turnover=not args.no_turnover,
            enable_diversity=not args.no_diversity,
            enable_drawdown=not args.no_drawdown
        )
        test_env = MultiObjectiveRewardWrapper(
            test_env,
            lambda_turnover=args.lambda_turnover,
            lambda_diversity=args.lambda_diversity,
            lambda_drawdown=args.lambda_drawdown,
            tc_rate=args.tc_rate,
            enable_turnover=not args.no_turnover,
            enable_diversity=not args.no_diversity,
            enable_drawdown=not args.no_drawdown
        )
        print(f"    - Turnover penalty: {args.lambda_turnover if not args.no_turnover else 'disabled'}")
        print(f"    - Diversity bonus: {args.lambda_diversity if not args.no_diversity else 'disabled'}")
        print(f"    - Drawdown penalty: {args.lambda_drawdown if not args.no_drawdown else 'disabled'}")
        print(f"    - Transaction cost: {args.tc_rate * 100:.2f}%")
    else:
        print(f"\n  [단순 보상] 원본 ln(V_t/V_{{t-1}}) 사용 (Baseline 비교용)")

    print(f"  State space: {train_env.state_space}")
    print(f"  Action space: {train_env.action_space}")

    # 5. IRT 모델 학습
    print(f"\n[5/5] Training SAC + IRT Policy...")

    # 콜백 설정
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=os.path.join(log_dir, "checkpoints"),
        name_prefix="irt_model"
    )

    eval_callback = EvalCallback(
        Monitor(test_env),
        best_model_save_path=os.path.join(log_dir, "best_model"),
        log_path=os.path.join(log_dir, "eval"),
        eval_freq=5000,
        deterministic=True,
        render=False
    )

    # IRT Policy 인자
    policy_kwargs = {
        "emb_dim": args.emb_dim,
        "m_tokens": args.m_tokens,
        "M_proto": args.M_proto,
        "alpha": args.alpha,
        "eps": args.eps,
        "eta_0": args.eta_0,
        "eta_1": args.eta_1,
        "market_feature_dim": args.market_feature_dim,
        "xai_reg_weight": args.xai_reg_weight,
        # Gaussian 정책 파라미터
        "log_std_min": -20,
        "log_std_max": 2,
        # 절제 연구 옵션
        "use_shared_decoder": args.shared_decoder,
    }

    # SAC 파라미터
    sac_params = SAC_PARAMS.copy()
    if 'ent_coef' in sac_params and isinstance(sac_params['ent_coef'], str):
        sac_params['ent_coef'] = 'auto'

    print(f"  SAC params (initial): {sac_params}")
    print(f"  IRT params: {policy_kwargs}")

    # ===== Phase 1.9 Tier 0: 적응형 타겟 엔트로피 =====
    print(f"\n[Tier 0] Estimating empirical policy entropy...")

    # 단계 1: 초기 모델 생성 (엔트로피 추정용)
    model_temp = SACWithGradClip(
        policy=IRTPolicy,
        env=train_env,
        policy_kwargs=policy_kwargs,
        max_grad_norm=10.0,
        **sac_params,
        verbose=0,
        tensorboard_log=None
    )

    # 단계 2: 경험적 엔트로피 추정
    estimator = PolicyEntropyEstimator(n_states=100, n_samples_per_state=20)
    mean_entropy, std_entropy = estimator.estimate(model_temp.policy, train_env)

    print(f"  Empirical entropy: {mean_entropy:.3f} ± {std_entropy:.3f} nats")

    # 단계 3: 타겟 엔트로피 설정 (경험적 값의 70%, Ahmed et al. 2019 따름)
    target_entropy = 0.7 * mean_entropy

    print(f"  Target entropy: {target_entropy:.3f} nats (70% of empirical)")

    # 이전 방법과 비교
    n_stocks = train_env.action_space.shape[0]
    old_target = -0.5 * np.log(n_stocks)
    print(f"  Previous method: -0.5 * log(N) = {old_target:.3f} nats")
    print(f"  Improvement: {target_entropy - old_target:+.3f} nats")

    # 단계 4: SAC 타겟 엔트로피 오버라이드
    sac_params['target_entropy'] = target_entropy

    # 임시 모델 정리
    del model_temp

    # 전체 타임스텝
    total_timesteps = 250 * args.episodes
    print(f"  Total timesteps: {total_timesteps}")

    # ===== Phase 1.9 Tier 2: 동적 Alpha 스케줄러 =====
    print(f"\n[Tier 2] Setting up alpha scheduler...")

    # 콜백 리스트
    callbacks = [checkpoint_callback, eval_callback]

    # 적응형 alpha 활성화 확인
    if args.adaptive_alpha:
        print(f"  Using Adaptive Alpha (Tier 3 - Entropy-based)")
        stock_dim = len(train_df.tic.unique())
        alpha_scheduler = AlphaScheduler(
            schedule_type='adaptive',
            action_dim=stock_dim,
            alpha_min=0.05,
            alpha_max=0.40,
            warmup_steps=5000,
            lr=3e-4
        )
        # Add alpha_scheduler to policy_kwargs for adaptive mode
        policy_kwargs['alpha_scheduler'] = alpha_scheduler
        print(f"  Target entropy: {alpha_scheduler.target_entropy:.1f}")
        print(f"  Alpha range: [{alpha_scheduler.alpha_min}, {alpha_scheduler.alpha_max}]")
        print(f"  Warmup steps: {alpha_scheduler.warmup_steps}")
        print(f"  Note: Alpha updates happen inside IRTPolicy using real log_prob")
    else:
        print(f"  Using Cosine Alpha Scheduler (Tier 2 - Step-based)")
        alpha_scheduler = AlphaScheduler(
            schedule_type='cosine',
            alpha_start=0.3,
            alpha_end=0.7,
            total_steps=total_timesteps
        )
        alpha_callback = AlphaUpdateCallback(alpha_scheduler, verbose=1)
        callbacks.append(alpha_callback)  # Only add callback for step-based scheduler
        print(f"  Alpha schedule: 0.3 → 0.7 (cosine)")
        print(f"  Early training: Replicator dominant (빠른 수렴)")
        print(f"  Late training: OT dominant (구조적 매칭)")

    # Step 5: Create final model with adaptive target entropy
    print(f"\n[5/5] Creating SAC + IRT model with adaptive target entropy...")
    model = SACWithGradClip(
        policy=IRTPolicy,
        env=train_env,
        policy_kwargs=policy_kwargs,
        max_grad_norm=10.0,  # Tier 3: Global gradient clipping
        **sac_params,
        verbose=1,
        tensorboard_log=os.path.join(log_dir, "tensorboard")
    )

    print(f"  Model created with target_entropy = {target_entropy:.3f} nats")

    print(f"\n  Starting training...")

    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True
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

    return log_dir, final_model_path


def test_irt(args, model_path=None):
    """IRT 모델 평가"""

    print("=" * 70)
    print(f"IRT Evaluation")
    print("=" * 70)

    # Model path
    if model_path is None:
        if args.checkpoint is None:
            raise ValueError("--checkpoint required for --mode test")
        model_path = args.checkpoint

    print(f"\n[Config]")
    print(f"  Model: {model_path}")
    print(f"  Test: {args.test_start} ~ {args.test_end}")

    # 1. Prepare data
    print(f"\n[1/4] Downloading test data...")
    df = YahooDownloader(
        start_date=args.test_start,
        end_date=args.test_end,
        ticker_list=DOW_30_TICKER
    ).fetch_data()
    print(f"  Downloaded: {df.shape[0]} rows")

    # 2. Feature Engineering
    print(f"\n[2/4] Feature Engineering...")
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_turbulence=False,
        user_defined_feature=False
    )
    df_processed = fe.preprocess_data(df)
    test_df = data_split(df_processed, args.test_start, args.test_end)
    print(f"  Test rows: {len(test_df)}")

    # 3. Load environment and model
    print(f"\n[3/4] Loading model...")
    stock_dim = len(test_df.tic.unique())
    print(f"  실제 주식 수: {stock_dim}")
    test_env = create_env(test_df, stock_dim, INDICATORS)

    # ===== Multi-Objective Reward Wrapper 적용 (학습과 동일한 설정) =====
    if args.use_multiobjective:
        print(f"  [Multi-Objective Reward] Wrapper 적용 중...")
        test_env = MultiObjectiveRewardWrapper(
            test_env,
            lambda_turnover=args.lambda_turnover,
            lambda_diversity=args.lambda_diversity,
            lambda_drawdown=args.lambda_drawdown,
            tc_rate=args.tc_rate,
            enable_turnover=not args.no_turnover,
            enable_diversity=not args.no_diversity,
            enable_drawdown=not args.no_drawdown
        )

    model = SACWithGradClip.load(model_path, env=test_env)
    print(f"  Model loaded successfully")

    # 4. Run evaluation
    print(f"\n[4/4] Running evaluation...")
    obs, _ = test_env.reset()
    done = False
    portfolio_values = [1000000]
    total_reward = 0

    # IRT 데이터 수집 준비
    irt_data_list = {
        'w': [],
        'w_rep': [],
        'w_ot': [],
        'crisis_levels': [],
        'crisis_types': [],
        'cost_matrices': [],
        'weights': []
    }

    step = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)

        # IRT info 수집
        if hasattr(model.policy, 'get_irt_info'):
            info_dict = model.policy.get_irt_info()
            if info_dict is not None:
                # Batch=1이므로 [0] 인덱스로 추출
                irt_data_list['w'].append(info_dict['w'][0].cpu().numpy())
                irt_data_list['w_rep'].append(info_dict['w_rep'][0].cpu().numpy())
                irt_data_list['w_ot'].append(info_dict['w_ot'][0].cpu().numpy())
                irt_data_list['crisis_levels'].append(info_dict['crisis_level'][0].cpu().numpy())
                irt_data_list['crisis_types'].append(info_dict['crisis_types'][0].cpu().numpy())
                irt_data_list['cost_matrices'].append(info_dict['cost_matrix'][0].cpu().numpy())

                # Action을 weight로 변환 (simplex 정규화)
                weights = action / (action.sum() + 1e-8)
                irt_data_list['weights'].append(weights)

        obs, reward, done, truncated, info = test_env.step(action)
        total_reward += reward
        done = done or truncated

        # Portfolio value
        state = np.array(test_env.state)
        cash = state[0]
        prices = state[1:stock_dim+1]
        holdings = state[stock_dim+1:2*stock_dim+1]
        pv = cash + np.sum(prices * holdings)
        portfolio_values.append(pv)

        step += 1

    print(f"  Evaluation completed: {step} steps")

    # IRT 데이터 NumPy 배열로 변환
    if irt_data_list['w']:
        irt_data = {
            'w_rep': np.array(irt_data_list['w_rep']),  # [T, M]
            'w_ot': np.array(irt_data_list['w_ot']),    # [T, M]
            'weights': np.array(irt_data_list['weights']),  # [T, N]
            'crisis_levels': np.array(irt_data_list['crisis_levels']).squeeze(),  # [T]
            'crisis_types': np.array(irt_data_list['crisis_types']),  # [T, K]
            'prototype_weights': np.array(irt_data_list['w']),  # [T, M]
            'cost_matrices': np.array(irt_data_list['cost_matrices']),  # [T, m, M]
            'symbols': DOW_30_TICKER[:stock_dim],  # 실제 주식 수만큼
            'metrics': {
                'sharpe_ratio': 0,  # calculate_metrics() 필요
                'sortino_ratio': 0,
                'max_drawdown': 0,
                'volatility': 0
            }
        }
    else:
        print("  Warning: No IRT data collected")
        irt_data = None

    # 5. Results
    final_value = portfolio_values[-1]
    total_return = (final_value - 1000000) / 1000000

    print(f"\n" + "=" * 70)
    print(f"Evaluation Results")
    print("=" * 70)
    print(f"\n[Period]")
    print(f"  Start: {args.test_start}")
    print(f"  End: {args.test_end}")
    print(f"  Steps: {step}")

    print(f"\n[Performance]")
    print(f"  Initial value: $1,000,000.00")
    print(f"  Final value: ${final_value:,.2f}")
    print(f"  Total return: {total_return*100:.2f}%")
    print(f"  Total reward: {total_reward:.4f}")

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
            irt_data=irt_data
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
        pv = np.array(portfolio_values)

        # Metrics 실제 계산
        downside_returns = returns[returns < 0]

        metrics = {
            'total_return': float(total_return),
            'sharpe_ratio': float(calculate_sharpe_ratio(returns, risk_free_rate=0.02, periods_per_year=252)),
            'sortino_ratio': float(calculate_sortino_ratio(returns, target_return=0.02, periods_per_year=252)),
            'calmar_ratio': float(calculate_calmar_ratio(returns, periods_per_year=252)),
            'max_drawdown': float(calculate_max_drawdown(pv)),
            'var_5': float(calculate_var(returns, alpha=0.05)),
            'cvar_5': float(calculate_cvar(returns, alpha=0.05)),
            'downside_deviation': float(np.std(downside_returns) * np.sqrt(252)) if len(downside_returns) > 0 else 0.0,
            'avg_turnover': float(calculate_turnover(irt_data['weights'])) if len(irt_data['weights']) > 1 else 0.0
        }

        results = {
            'returns': returns,
            'values': pv,
            'weights': irt_data['weights'],
            'crisis_levels': irt_data['crisis_levels'],
            'crisis_types': irt_data['crisis_types'],
            'prototype_weights': irt_data['prototype_weights'],
            'w_rep': irt_data['w_rep'],
            'w_ot': irt_data['w_ot'],
            'eta': np.zeros(len(returns)),  # eta (learning rate) not currently collected
            'cost_matrices': irt_data['cost_matrices'],
            'symbols': irt_data['symbols'],
            'metrics': metrics
        }

        # Config (alpha 정보 포함)
        config = {
            'irt': {
                'alpha': args.alpha
            }
        }

        print(f"\n[JSON 저장]")
        print(f"  Output: {log_dir}")
        save_evaluation_results(results, Path(log_dir), config)

    return {
        'portfolio_values': portfolio_values,
        'final_value': final_value,
        'total_return': total_return,
        'total_reward': total_reward,
        'steps': step
    }


def main():
    parser = argparse.ArgumentParser(description="IRT Policy Training/Evaluation")

    # Mode
    parser.add_argument("--mode", type=str, default="both",
                        choices=['train', 'test', 'both'],
                        help="Execution mode (default: both)")

    # Data period (defaults from config.py)
    parser.add_argument("--train-start", type=str, default=TRAIN_START_DATE,
                        help=f"Training start date (default: {TRAIN_START_DATE})")
    parser.add_argument("--train-end", type=str, default=TRAIN_END_DATE,
                        help=f"Training end date (default: {TRAIN_END_DATE})")
    parser.add_argument("--test-start", type=str, default=TEST_START_DATE,
                        help=f"Test start date (default: {TEST_START_DATE})")
    parser.add_argument("--test-end", type=str, default=TEST_END_DATE,
                        help=f"Test end date (default: {TEST_END_DATE})")

    # Training settings
    parser.add_argument("--episodes", type=int, default=200,
                        help="Number of episodes (default: 200)")
    parser.add_argument("--output", type=str, default="logs",
                        help="Output directory (default: logs)")

    # IRT parameters
    parser.add_argument("--emb-dim", type=int, default=128,
                        help="IRT embedding dimension (default: 128)")
    parser.add_argument("--m-tokens", type=int, default=6,
                        help="Number of epitope tokens (default: 6)")
    parser.add_argument("--M-proto", type=int, default=8,
                        help="Number of prototypes (default: 8)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Initial OT-Replicator mixing ratio (default: 0.5, overridden by scheduler if used)")
    parser.add_argument("--eps", type=float, default=0.10,
                        help="Sinkhorn entropy (default: 0.10)")
    parser.add_argument("--eta-0", type=float, default=0.05,
                        help="Base learning rate (Replicator) (default: 0.05)")
    parser.add_argument("--eta-1", type=float, default=0.15,
                        help="Crisis increase (Replicator) (default: 0.15)")
    parser.add_argument("--market-feature-dim", type=int, default=12,
                        help="Market feature dimension (default: 12)")

    # Tier 3: Adaptive Alpha
    parser.add_argument("--adaptive-alpha", action="store_true",
                        help="Enable Adaptive Alpha Controller (Tier 3)")

    # Ablation study options
    parser.add_argument("--shared-decoder", action="store_true",
                        help="Use shared decoder with prototype conditioning (ablation study)")
    parser.add_argument("--no-tcell", action="store_true",
                        help="Disable T-Cell crisis detection (ablation study)")
    parser.add_argument("--xai-reg-weight", type=float, default=0.01,
                        help="XAI regularization weight (default: 0.01)")

    # Evaluation only
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint path for evaluation (required for --mode test)")

    # Visualization (기본값: 모두 저장)
    parser.add_argument("--no-plot", action="store_true",
                        help="Disable saving evaluation plots (default: enabled)")
    parser.add_argument("--output-plot", type=str, default=None,
                        help="Plot output directory (default: log_dir/evaluation_plots)")
    parser.add_argument("--no-json", action="store_true",
                        help="Disable saving evaluation results as JSON (default: enabled)")

    # ===== Multi-Objective Reward 옵션 =====
    parser.add_argument("--use-multiobjective", action="store_true", default=True,
                        help="다목적 보상 사용 (default: True)")
    parser.add_argument("--no-multiobjective", dest="use_multiobjective", action="store_false",
                        help="다목적 보상 비활성화 (Baseline 비교용)")

    # 보상 함수 하이퍼파라미터
    parser.add_argument("--lambda-turnover", type=float, default=0.02,
                        help="회전율 패널티 가중치 (default: 0.02, increased from 0.01)")
    parser.add_argument("--lambda-diversity", type=float, default=0.15,
                        help="다양성 보너스 가중치 (default: 0.15, increased from 0.1)")
    parser.add_argument("--lambda-drawdown", type=float, default=0.05,
                        help="낙폭 패널티 가중치 (default: 0.05)")
    parser.add_argument("--tc-rate", type=float, default=0.001,
                        help="거래 비용률 (default: 0.001 = 0.1%%)")

    # Ablation study용 플래그
    parser.add_argument("--no-turnover", action="store_true",
                        help="회전율 패널티 비활성화 (ablation)")
    parser.add_argument("--no-diversity", action="store_true",
                        help="다양성 보너스 비활성화 (ablation)")
    parser.add_argument("--no-drawdown", action="store_true",
                        help="낙폭 패널티 비활성화 (ablation)")

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
