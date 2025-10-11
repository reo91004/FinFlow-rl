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
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

# Import IRT Policy
from finrl.agents.irt import IRTPolicy

# Import evaluation function
import sys

sys.path.insert(0, os.path.dirname(__file__))
from evaluate import evaluate_model


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
        "print_verbosity": 500,
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
    train_env = create_env(train_df, stock_dim, INDICATORS)
    test_env = create_env(test_df, stock_dim, INDICATORS)
    print(f"  State space: {train_env.state_space}")
    print(f"  Action space: {train_env.action_space}")

    # 5. Train IRT model
    print(f"\n[5/5] Training SAC + IRT Policy...")

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

    # IRT Policy kwargs
    policy_kwargs = {
        "emb_dim": args.emb_dim,
        "m_tokens": args.m_tokens,
        "M_proto": args.M_proto,
        "alpha": args.alpha,
        "alpha_min": args.alpha_min,
        "alpha_max": args.alpha_max if args.alpha_max else args.alpha,
        "ema_beta": args.ema_beta,
        "eps": args.eps,
        "max_iters": args.max_iters,
        "replicator_temp": args.replicator_temp,
        "eta_0": args.eta_0,
        "eta_1": args.eta_1,
        "gamma": args.gamma,
        "market_feature_dim": args.market_feature_dim,
    }

    # SAC parameters (Phase 2.5: 안정화)
    sac_params = SAC_PARAMS.copy()

    # if 'ent_coef' in sac_params and isinstance(sac_params['ent_coef'], str):
    #     sac_params['ent_coef'] = 'auto'

    # 온도 폭주 방지: 고정 온도 사용
    sac_params["ent_coef"] = 0.05  # 'auto' 대신 고정 값
    sac_params["learning_starts"] = 5000  # 100 → 5000 (웜업 증가)

    print(f"  SAC params: {sac_params}")
    print(f"  IRT params: {policy_kwargs}")

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
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
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

    # evaluate.py의 evaluate_model() 재사용
    portfolio_values, irt_data, metrics = evaluate_model(
        model_path=model_path,
        model_class=SAC,
        test_start=args.test_start,
        test_end=args.test_end,
        stock_tickers=DOW_30_TICKER,
        tech_indicators=INDICATORS,
        initial_amount=1000000,
        verbose=True,
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
            "weights": irt_data["weights"],
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
        "--episodes", type=int, default=200, help="Number of episodes (default: 200)"
    )
    parser.add_argument(
        "--output", type=str, default="logs", help="Output directory (default: logs)"
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
        default=0.3,
        help="Base OT-Replicator mixing ratio (default: 0.3)",
    )
    parser.add_argument(
        "--alpha-min",
        type=float,
        default=0.06,
        help="Crisis minimum alpha (default: 0.06)",
    )
    parser.add_argument(
        "--alpha-max",
        type=float,
        default=None,
        help="Normal maximum alpha (default: --alpha value)",
    )
    parser.add_argument(
        "--ema-beta",
        type=float,
        default=0.85,
        help="EMA memory coefficient (default: 0.85)",
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
        default=0.7,
        help="Replicator softmax temperature (default: 0.7, Phase 1)",
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
        default=0.18,
        help="Crisis increase (Replicator) (default: 0.18)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.8,
        help="Co-stimulation weight in cost function (default: 0.8)",
    )
    parser.add_argument(
        "--market-feature-dim",
        type=int,
        default=12,
        help="Market feature dimension (default: 12)",
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
