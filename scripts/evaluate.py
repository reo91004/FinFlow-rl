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
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from finrl.config_tickers import DOW_30_TICKER
from finrl.config import INDICATORS, TEST_START_DATE, TEST_END_DATE
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3 import SAC, PPO, A2C, TD3, DDPG


def create_env(df, stock_dim, tech_indicators):
    """환경 생성"""
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


def calculate_metrics(portfolio_values, initial_amount=1000000, weights_history=None):
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
        calculate_turnover
    )

    pv = np.array(portfolio_values)

    # Daily returns
    returns = (pv[1:] - pv[:-1]) / pv[:-1]

    # Total return
    total_return = (pv[-1] - initial_amount) / initial_amount

    # Annualized return (assuming 252 trading days)
    n_days = len(pv) - 1
    annualized_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0

    # Volatility (annualized)
    volatility = np.std(returns) * np.sqrt(252)

    # Sharpe Ratio (using detailed calculation from metrics.py)
    sharpe_ratio = calculate_sharpe_ratio(returns, risk_free_rate=0.02, periods_per_year=252)

    # Maximum Drawdown (using detailed calculation from metrics.py)
    max_drawdown = calculate_max_drawdown(pv)

    # Calmar Ratio (using detailed calculation from metrics.py)
    calmar_ratio = calculate_calmar_ratio(returns, periods_per_year=252)

    # Sortino Ratio (using detailed calculation from metrics.py)
    sortino_ratio = calculate_sortino_ratio(returns, target_return=0.02, periods_per_year=252)

    # VaR and CVaR (5% level)
    var_5 = calculate_var(returns, alpha=0.05)
    cvar_5 = calculate_cvar(returns, alpha=0.05)

    # Downside deviation
    downside_returns = returns[returns < 0]
    downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0

    # Turnover (if weights history provided)
    avg_turnover = 0.0
    if weights_history is not None and len(weights_history) > 1:
        avg_turnover = calculate_turnover(np.array(weights_history))

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
        "avg_turnover": avg_turnover,
        "final_value": pv[-1],
        "n_steps": n_days
    }


def plot_results(portfolio_values, output_dir, model_name="Model", model_path=None, irt_data=None):
    """
    시각화 생성 (finrl.evaluation.visualizer 사용)

    일반 모델: 3개 시각화
    IRT 모델: 14개 시각화

    Args:
        portfolio_values: 포트폴리오 가치 배열
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
        irt_data=irt_data
    )


def detect_model_type(model_path):
    """모델 파일명에서 모델 타입 자동 감지"""

    filename = os.path.basename(model_path).lower()

    if 'sac' in filename:
        return 'sac', SAC
    elif 'ppo' in filename:
        return 'ppo', PPO
    elif 'a2c' in filename:
        return 'a2c', A2C
    elif 'td3' in filename:
        return 'td3', TD3
    elif 'ddpg' in filename:
        return 'ddpg', DDPG
    else:
        # 경로에서 찾기
        path_lower = model_path.lower()
        for name in ['sac', 'ppo', 'a2c', 'td3', 'ddpg']:
            if name in path_lower:
                return name, {'sac': SAC, 'ppo': PPO, 'a2c': A2C, 'td3': TD3, 'ddpg': DDPG}[name]

        raise ValueError(
            f"모델 타입을 자동 감지할 수 없습니다: {model_path}\n"
            f"파일명 또는 경로에 모델명(sac/ppo/a2c/td3/ddpg)을 포함하거나 --model-type 인자를 명시하세요."
        )


def evaluate_direct(args, model_name, model_class):
    """Direct 방식: SB3 모델 직접 사용"""

    print(f"\n[Method: Direct - SB3 predict()]")

    # 1. 데이터 준비
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

    # 3. 환경 및 모델 로드
    print(f"\n[3/4] Loading model...")
    stock_dim = len(test_df.tic.unique())
    print(f"  실제 주식 수: {stock_dim}")
    test_env = create_env(test_df, stock_dim, INDICATORS)

    model = model_class.load(args.model, env=test_env)
    print(f"  Model loaded successfully")

    # 4. 평가 실행
    print(f"\n[4/4] Running evaluation...")
    obs, _ = test_env.reset()
    done = False
    portfolio_values = [1000000]

    # IRT 모델 감지
    is_irt = 'irt' in args.model.lower()

    # IRT 데이터 수집 준비
    if is_irt:
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
        if is_irt and hasattr(model.policy, 'get_irt_info'):
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
        done = done or truncated

        # Portfolio value 계산
        state = np.array(test_env.state)
        cash = state[0]
        prices = state[1:stock_dim+1]
        holdings = state[stock_dim+1:2*stock_dim+1]
        pv = cash + np.sum(prices * holdings)
        portfolio_values.append(pv)

        step += 1

    print(f"  Evaluation completed: {step} steps")

    # IRT 데이터 변환
    irt_data = None
    if is_irt and irt_data_list['w']:
        irt_data = {
            'w_rep': np.array(irt_data_list['w_rep']),  # [T, M]
            'w_ot': np.array(irt_data_list['w_ot']),    # [T, M]
            'weights': np.array(irt_data_list['weights']),  # [T, N]
            'crisis_levels': np.array(irt_data_list['crisis_levels']).squeeze(),  # [T]
            'crisis_types': np.array(irt_data_list['crisis_types']),  # [T, K]
            'prototype_weights': np.array(irt_data_list['w']),  # [T, M]
            'cost_matrices': np.array(irt_data_list['cost_matrices']),  # [T, m, M]
            'symbols': DOW_30_TICKER[:stock_dim],  # 실제 주식 수만큼
            'metrics': None  # main()에서 calculate_metrics()로 계산
        }

    return portfolio_values, irt_data


def evaluate_drlagent(args, model_name, model_class):
    """DRLAgent 방식: DRLAgent.DRL_prediction() 사용"""

    print(f"\n[Method: DRLAgent - DRL_prediction()]")

    # 1. 데이터 준비
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

    # 3. 환경 생성 및 모델 로드
    print(f"\n[3/4] Loading model...")
    stock_dim = len(test_df.tic.unique())
    print(f"  실제 주식 수: {stock_dim}")
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
        "print_verbosity": 500
    }

    e_test_gym = StockTradingEnv(**env_kwargs)

    model = model_class.load(args.model)
    print(f"  Model loaded successfully")

    # 4. DRL_prediction 실행
    print(f"\n[4/4] Running DRL_prediction...")
    account_memory, actions_memory = DRLAgent.DRL_prediction(
        model=model,
        environment=e_test_gym,
        deterministic=True
    )

    print(f"  Evaluation completed: {len(account_memory)} steps")

    # account_memory에서 portfolio_values 추출
    portfolio_values = account_memory['account_value'].tolist()

    # DRLAgent 방식은 IRT 데이터 수집 불가
    return portfolio_values, None


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
        model_class = {
            'sac': SAC, 'ppo': PPO, 'a2c': A2C, 'td3': TD3, 'ddpg': DDPG
        }[model_name]

    print(f"\n[Config]")
    print(f"  Model: {args.model}")
    print(f"  Type: {model_name.upper()}")
    print(f"  Method: {args.method}")
    print(f"  Test: {args.test_start} ~ {args.test_end}")

    # 평가 방식 선택
    if args.method == "direct":
        portfolio_values, irt_data = evaluate_direct(args, model_name, model_class)
    elif args.method == "drlagent":
        portfolio_values, irt_data = evaluate_drlagent(args, model_name, model_class)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    # 지표 계산
    metrics = calculate_metrics(portfolio_values)

    # IRT 데이터에 metrics 추가
    if irt_data is not None:
        irt_data['metrics'] = metrics

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
        plot_results(portfolio_values, output_dir,
                    model_name=model_name.upper(),
                    model_path=args.model,
                    irt_data=irt_data)

    # 8. JSON 저장 (기본 활성화)
    if not args.no_json:
        from finrl.evaluation.visualizer import save_evaluation_results
        from pathlib import Path

        output_dir = args.output_json or os.path.dirname(args.model)

        # IRT 데이터가 있으면 상세 JSON 저장, 없으면 기본 메트릭만 저장
        if irt_data is not None:
            # 상세 evaluation_results.json + evaluation_insights.json 저장
            returns = np.diff(portfolio_values) / portfolio_values[:-1]

            results = {
                'returns': returns,
                'values': np.array(portfolio_values),
                'weights': irt_data['weights'],
                'crisis_levels': irt_data['crisis_levels'],
                'crisis_types': irt_data['crisis_types'],
                'prototype_weights': irt_data['prototype_weights'],
                'w_rep': irt_data['w_rep'],
                'w_ot': irt_data['w_ot'],
                'eta': np.zeros(len(returns)),  # TODO: eta 정보 수집
                'cost_matrices': irt_data['cost_matrices'],
                'symbols': irt_data['symbols'],
                'metrics': metrics
            }

            # Config (IRT Policy의 경우)
            config = {
                'irt': {
                    'alpha': 0.3  # TODO: 모델에서 추출하거나 인자로 받기
                }
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
                    "steps": metrics['n_steps']
                },
                "metrics": {
                    "total_return": float(metrics['total_return']),
                    "annualized_return": float(metrics['annualized_return']),
                    "volatility": float(metrics['volatility']),
                    "sharpe_ratio": float(metrics['sharpe_ratio']),
                    "sortino_ratio": float(metrics['sortino_ratio']),
                    "calmar_ratio": float(metrics['calmar_ratio']),
                    "max_drawdown": float(metrics['max_drawdown']),
                    "final_value": float(metrics['final_value']),
                    "profit_loss": float(metrics['final_value'] - 1000000)
                },
                "timestamp": datetime.now().isoformat()
            }

            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)

            print(f"\n  Results saved to: {output_file}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="모델 상세 평가")

    parser.add_argument("--model", type=str, required=True,
                        help="모델 파일 경로 (.zip)")
    parser.add_argument("--model-type", type=str, default=None,
                        choices=['sac', 'ppo', 'a2c', 'td3', 'ddpg'],
                        help="모델 타입 (자동 감지 실패 시 명시)")

    parser.add_argument("--method", type=str, default="direct",
                        choices=['direct', 'drlagent'],
                        help="평가 방식 (default: direct)")

    parser.add_argument("--test-start", type=str, default=TEST_START_DATE,
                        help=f"Test start date (default: {TEST_START_DATE})")
    parser.add_argument("--test-end", type=str, default=TEST_END_DATE,
                        help=f"Test end date (default: {TEST_END_DATE})")

    parser.add_argument("--no-plot", action="store_true",
                        help="시각화 결과 저장 비활성화 (기본: 활성화)")
    parser.add_argument("--no-json", action="store_true",
                        help="JSON 결과 저장 비활성화 (기본: 활성화)")

    parser.add_argument("--output", type=str, default=None,
                        help="Plot 출력 디렉토리 (기본: 모델 디렉토리/evaluation_plots)")
    parser.add_argument("--output-json", type=str, default=None,
                        help="JSON 출력 파일 (기본: 모델 디렉토리/evaluation_results.json)")

    args = parser.parse_args()

    metrics = main(args)
