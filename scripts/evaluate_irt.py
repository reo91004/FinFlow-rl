# scripts/evaluate_irt.py

"""
IRT 평가 스크립트

사용법:
python scripts/evaluate_irt.py --checkpoint logs/YYYYMMDD_HHMMSS/checkpoints/best_model.pth --config configs/default_irt.yaml
"""

import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict

from src.agents.bcell_irt import BCellIRTActor
from src.algorithms.critics.redq import REDQCritic
from src.environments.portfolio_env import PortfolioEnv
from src.data.market_loader import DataLoader
from src.data.feature_extractor import FeatureExtractor
from src.evaluation.metrics import MetricsCalculator
from src.utils.logger import FinFlowLogger
from src.utils.training_utils import resolve_device

class IRTEvaluator:
    """IRT 모델 평가기"""

    def __init__(self, config: Dict, checkpoint_path: str):
        self.config = config
        self.checkpoint_path = Path(checkpoint_path)
        self.device = resolve_device(config.get('device', 'auto'))

        self.logger = FinFlowLogger("IRTEvaluator")
        self.metrics_calc = MetricsCalculator()

        # 데이터 로드
        self._load_data()

        # 모델 로드
        self._load_model()

    def _load_data(self):
        """테스트 데이터 로드"""
        data_config = self.config['data']

        loader = DataLoader(cache_dir='data/cache')
        self.price_data = loader.download_data(
            symbols=data_config['symbols'],
            start_date=data_config['test_start'],
            end_date=data_config['test_end'],
            use_cache=data_config.get('cache', True)
        )

        self.logger.info(f"테스트 데이터 로드 완료: {len(self.price_data)}일")

    def _load_model(self):
        """체크포인트에서 모델 로드"""
        # 차원 계산
        n_assets = len(self.price_data.columns)
        feature_dim = self.config.get('feature_dim', 12)
        state_dim = feature_dim + n_assets + 1

        # IRT Actor
        irt_config = self.config.get('irt', {})
        self.actor = BCellIRTActor(
            state_dim=state_dim,
            action_dim=n_assets,
            emb_dim=irt_config.get('emb_dim', 128),
            m_tokens=irt_config.get('m_tokens', 6),
            M_proto=irt_config.get('M_proto', 8),
            alpha=irt_config.get('alpha', 0.3)
        ).to(self.device)

        # 체크포인트 로드
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor.eval()

        self.logger.info(f"모델 로드 완료: {self.checkpoint_path}")

    def evaluate(self):
        """평가 실행"""
        # 환경 생성
        env_config = self.config['env']
        objective_config = self.config.get('objectives')

        env = PortfolioEnv(
            price_data=self.price_data,
            feature_extractor=FeatureExtractor(window=20),
            initial_capital=env_config.get('initial_balance', 1000000),
            transaction_cost=env_config.get('transaction_cost', 0.001),
            max_leverage=env_config.get('max_leverage', 1.0),
            objective_config=objective_config,
            use_advanced_reward=(objective_config is not None)
        )

        # 에피소드 실행
        state, _ = env.reset()
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        episode_return = 0
        all_returns = []
        all_weights = []
        crisis_levels = []
        crisis_types = []
        prototype_weights = []
        w_rep_list = []  # Replicator 가중치
        w_ot_list = []   # OT 가중치
        cost_matrices = []  # Cost matrix (샘플링)
        eta_list = []    # Crisis-adaptive learning rate

        done = False
        truncated = False
        step = 0

        while not (done or truncated):
            with torch.no_grad():
                action, info = self.actor(state_tensor, deterministic=True)

            action_np = action.cpu().numpy()[0]

            # 환경 스텝
            next_state, reward, done, truncated, _ = env.step(action_np)

            # 기록
            episode_return += reward
            all_returns.append(reward)
            all_weights.append(action_np)
            crisis_levels.append(info['crisis_level'].item())
            crisis_types.append(info['crisis_types'].cpu().numpy()[0])
            prototype_weights.append(info['w'].cpu().numpy()[0])
            w_rep_list.append(info['w_rep'].cpu().numpy()[0])
            w_ot_list.append(info['w_ot'].cpu().numpy()[0])
            eta_list.append(info['eta'].item())

            # Cost matrix는 크기가 크므로 10 step마다만 샘플링
            if step % 10 == 0:
                cost_matrices.append(info['cost_matrix'].cpu().numpy()[0])

            state = next_state
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            step += 1

        # 메트릭 계산
        returns_array = np.array(all_returns)
        metrics = self.metrics_calc.calculate_all_metrics(returns_array)
        metrics['total_return'] = episode_return
        metrics['avg_crisis_level'] = np.mean(crisis_levels)

        # 결과 출력
        self.logger.info("="*60)
        self.logger.info("평가 결과")
        self.logger.info("="*60)
        for key, value in metrics.items():
            self.logger.info(f"{key}: {value:.4f}")

        # 결과 저장
        results = {
            'metrics': metrics,
            'returns': all_returns,
            'weights': all_weights,
            'crisis_levels': crisis_levels,
            'crisis_types': crisis_types,
            'prototype_weights': prototype_weights,
            # IRT 분해 데이터
            'w_rep': w_rep_list,
            'w_ot': w_ot_list,
            'cost_matrices': cost_matrices,
            'eta': eta_list,
            # 메타데이터
            'symbols': self.config['data']['symbols'],
            'price_data': self.price_data.values.tolist(),  # 벤치마크 계산용
            'dates': self.price_data.index.strftime('%Y-%m-%d').tolist()
        }

        save_dir = self.checkpoint_path.parent.parent / 'evaluation'
        save_dir.mkdir(exist_ok=True)

        # NumPy 배열을 리스트로 변환
        import json
        with open(save_dir / 'evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

        self.logger.info(f"결과 저장 완료: {save_dir}")

        # 해석 정보 생성 및 저장
        insights = self._generate_insights(results)
        with open(save_dir / 'evaluation_insights.json', 'w') as f:
            json.dump(insights, f, indent=2, ensure_ascii=False)

        self.logger.info(f"해석 정보 저장 완료: {save_dir / 'evaluation_insights.json'}")

        # 시각화 생성
        self._create_visualizations(results, save_dir)

        return metrics

    def _create_visualizations(self, results: Dict, save_dir: Path):
        """시각화 생성"""
        viz_dir = save_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)

        self.logger.info("시각화를 생성합니다...")

        # 스타일 설정
        sns.set_style("whitegrid")

        # IRT 핵심 시각화
        self._plot_irt_decomposition(results, viz_dir)

        # 기존 시각화
        self._plot_returns(results, viz_dir)
        self._plot_portfolio_weights(results, viz_dir)
        self._plot_crisis_levels(results, viz_dir)
        self._plot_prototype_weights(results, viz_dir)

        # 고급 시각화
        self._plot_stock_analysis(results, viz_dir)
        self._plot_performance_timeline(results, viz_dir)
        self._plot_benchmark_comparison(results, viz_dir)
        self._plot_risk_dashboard(results, viz_dir)

        # IRT 전문 시각화
        self._plot_tcell_analysis(results, viz_dir)
        self._plot_attribution_analysis(results, viz_dir)
        self._plot_cost_matrix(results, viz_dir)

        self.logger.info(f"시각화 완료: {viz_dir}")

    def _plot_irt_decomposition(self, results: Dict, output_dir: Path):
        """IRT 분해 시각화: w_t = (1-α)·w_rep + α·w_ot"""
        w = np.array(results['prototype_weights'])  # [T, M]
        w_rep = np.array(results['w_rep'])  # [T, M]
        w_ot = np.array(results['w_ot'])    # [T, M]
        eta = np.array(results['eta'])      # [T]
        crisis = np.array(results['crisis_levels'])  # [T]

        # α 값 (config에서 가져와야 하지만, 코드에서 0.3으로 하드코딩)
        alpha = self.config.get('irt', {}).get('alpha', 0.3)

        fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

        # (1) 프로토타입 가중치 비교 (대표 프로토타입 하나만)
        # 가장 평균 가중치가 높은 프로토타입 선택
        mean_w = w.mean(axis=0)
        top_proto = mean_w.argmax()

        axes[0].plot(w[:, top_proto], label=f'Final w (Proto {top_proto+1})', linewidth=2, color='black')
        axes[0].plot(w_rep[:, top_proto], label=f'Replicator (1-α={1-alpha:.1f})',
                    linewidth=1.5, linestyle='--', alpha=0.7, color='blue')
        axes[0].plot(w_ot[:, top_proto], label=f'OT (α={alpha:.1f})',
                    linewidth=1.5, linestyle='--', alpha=0.7, color='red')
        axes[0].set_title(f'IRT Decomposition: w = (1-α)·Replicator + α·OT (Proto {top_proto+1})',
                         fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Weight')
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)

        # (2) 전체 프로토타입 L2 norm 비교
        w_norm = np.linalg.norm(w, axis=1)
        w_rep_norm = np.linalg.norm(w_rep, axis=1)
        w_ot_norm = np.linalg.norm(w_ot, axis=1)

        axes[1].plot(w_norm, label='Final w (L2 norm)', linewidth=2, color='black')
        axes[1].plot(w_rep_norm, label='Replicator (L2 norm)',
                    linewidth=1.5, linestyle='--', alpha=0.7, color='blue')
        axes[1].plot(w_ot_norm, label='OT (L2 norm)',
                    linewidth=1.5, linestyle='--', alpha=0.7, color='red')
        axes[1].set_title('IRT Component Magnitude (L2 Norm)', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('L2 Norm')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)

        # (3) Crisis-adaptive learning rate η(c) = η_0 + η_1·c
        ax3_twin = axes[2].twinx()

        axes[2].plot(eta, label='Learning Rate η(c)', linewidth=2, color='green')
        axes[2].set_ylabel('Learning Rate η', color='green')
        axes[2].tick_params(axis='y', labelcolor='green')
        axes[2].set_title('Crisis-Adaptive Learning Rate', fontsize=14, fontweight='bold')
        axes[2].grid(True, alpha=0.3)

        ax3_twin.plot(crisis, label='Crisis Level', linewidth=1.5,
                     linestyle='--', alpha=0.7, color='red')
        ax3_twin.set_ylabel('Crisis Level', color='red')
        ax3_twin.tick_params(axis='y', labelcolor='red')
        ax3_twin.set_ylim(0, 1)

        axes[2].set_xlabel('Time Step')
        axes[2].legend(loc='upper left')
        ax3_twin.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig(output_dir / 'irt_decomposition.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_returns(self, results: Dict, output_dir: Path):
        """수익률 시각화"""
        returns = np.array(results['returns'])
        cumulative_returns = np.cumprod(1 + returns) - 1

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # 일일 수익률
        axes[0].plot(returns, alpha=0.7, label='Daily Returns')
        axes[0].axhline(0, color='red', linestyle='--', alpha=0.5)
        axes[0].set_title('Daily Returns', fontsize=14)
        axes[0].set_xlabel('Time Step')
        axes[0].set_ylabel('Return')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 누적 수익률
        axes[1].plot(cumulative_returns, linewidth=2, label='Cumulative Returns')
        axes[1].fill_between(range(len(cumulative_returns)), 0, cumulative_returns,
                            alpha=0.3)
        axes[1].set_title('Cumulative Returns', fontsize=14)
        axes[1].set_xlabel('Time Step')
        axes[1].set_ylabel('Cumulative Return')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'returns.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_portfolio_weights(self, results: Dict, output_dir: Path):
        """포트폴리오 가중치 시각화"""
        weights = np.array(results['weights'])
        n_assets = weights.shape[1]

        fig, ax = plt.subplots(figsize=(14, 8))

        # 스택 영역 차트
        bottom = np.zeros(len(weights))
        colors = plt.cm.tab20(np.linspace(0, 1, n_assets))

        for i in range(n_assets):
            ax.fill_between(range(len(weights)), bottom, bottom + weights[:, i],
                          alpha=0.8, color=colors[i], label=f'Asset {i+1}')
            bottom += weights[:, i]

        ax.set_title('Portfolio Weights Over Time', fontsize=14)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Weight')
        ax.set_ylim(0, 1)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'portfolio_weights.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_crisis_levels(self, results: Dict, output_dir: Path):
        """위기 레벨 시각화"""
        crisis_levels = np.array(results['crisis_levels'])

        fig, ax = plt.subplots(figsize=(14, 6))

        ax.plot(crisis_levels, linewidth=2, color='red', label='Crisis Level')
        ax.fill_between(range(len(crisis_levels)), 0, crisis_levels,
                       alpha=0.3, color='red')

        # 임계값 표시
        ax.axhline(0.7, color='darkred', linestyle='--', alpha=0.7,
                  label='High Crisis Threshold')
        ax.axhline(0.3, color='orange', linestyle='--', alpha=0.7,
                  label='Medium Crisis Threshold')

        ax.set_title('Crisis Level Detection', fontsize=14)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Crisis Level')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'crisis_levels.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_prototype_weights(self, results: Dict, output_dir: Path):
        """프로토타입 가중치 시각화"""
        proto_weights = np.array(results['prototype_weights'])
        n_protos = proto_weights.shape[1]

        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # 개별 프로토타입 가중치
        for i in range(n_protos):
            axes[0].plot(proto_weights[:, i], alpha=0.7, label=f'Proto {i+1}')

        axes[0].set_title('Prototype Weights Over Time', fontsize=14)
        axes[0].set_ylabel('Weight')
        axes[0].legend(ncol=4)
        axes[0].grid(True, alpha=0.3)

        # 프로토타입 엔트로피 (다양성 지표)
        entropy = -np.sum(proto_weights * np.log(proto_weights + 1e-8), axis=1)
        axes[1].plot(entropy, linewidth=2, color='purple', label='Entropy')
        axes[1].fill_between(range(len(entropy)), 0, entropy,
                           alpha=0.3, color='purple')

        axes[1].set_title('Prototype Diversity (Entropy)', fontsize=14)
        axes[1].set_xlabel('Time Step')
        axes[1].set_ylabel('Entropy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'prototype_weights.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_stock_analysis(self, results: Dict, output_dir: Path):
        """종목별 분석"""
        weights = np.array(results['weights'])  # [T, N]
        returns = np.array(results['returns'])  # [T]
        symbols = results['symbols']
        n_assets = weights.shape[1]

        fig, axes = plt.subplots(2, 1, figsize=(16, 10))

        # (1) 종목별 평균 가중치 (Top 10)
        mean_weights = weights.mean(axis=0)
        top_10_idx = np.argsort(mean_weights)[-10:][::-1]

        axes[0].barh(range(10), mean_weights[top_10_idx], color='steelblue')
        axes[0].set_yticks(range(10))
        axes[0].set_yticklabels([symbols[i] for i in top_10_idx])
        axes[0].set_xlabel('Average Weight')
        axes[0].set_title('Top 10 Holdings by Average Weight', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='x')

        # (2) 종목별 가중치 변동성 (위기 민감도)
        weight_std = weights.std(axis=0)
        top_10_vol_idx = np.argsort(weight_std)[-10:][::-1]

        axes[1].barh(range(10), weight_std[top_10_vol_idx], color='coral')
        axes[1].set_yticks(range(10))
        axes[1].set_yticklabels([symbols[i] for i in top_10_vol_idx])
        axes[1].set_xlabel('Weight Volatility (Std)')
        axes[1].set_title('Top 10 Most Dynamic Holdings (Crisis Sensitive)', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig(output_dir / 'stock_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_performance_timeline(self, results: Dict, output_dir: Path):
        """성과 메트릭 시계열"""
        returns = np.array(results['returns'])
        T = len(returns)

        # Rolling window 계산
        window = min(60, T // 4)  # 60일 또는 전체의 1/4

        fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

        # (1) Rolling Sharpe Ratio
        rolling_sharpe = []
        for i in range(window, T):
            window_returns = returns[i-window:i]
            sharpe = window_returns.mean() / (window_returns.std() + 1e-8) * np.sqrt(252)
            rolling_sharpe.append(sharpe)

        axes[0].plot(range(window, T), rolling_sharpe, linewidth=2, color='blue')
        axes[0].axhline(1.5, color='green', linestyle='--', alpha=0.5, label='Target (1.5)')
        axes[0].set_ylabel('Sharpe Ratio')
        axes[0].set_title(f'Rolling Sharpe Ratio (window={window})', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # (2) Rolling Drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max

        axes[1].fill_between(range(T), 0, drawdown, color='red', alpha=0.3)
        axes[1].plot(drawdown, linewidth=1.5, color='darkred')
        axes[1].axhline(-0.25, color='orange', linestyle='--', alpha=0.5, label='Target (-25%)')
        axes[1].set_ylabel('Drawdown')
        axes[1].set_title('Drawdown Timeline', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # (3) Turnover
        weights = np.array(results['weights'])
        turnover = np.sum(np.abs(np.diff(weights, axis=0)), axis=1)

        axes[2].plot(turnover, linewidth=1.5, color='purple')
        axes[2].axhline(turnover.mean(), color='black', linestyle='--', alpha=0.5,
                       label=f'Mean ({turnover.mean():.3f})')
        axes[2].set_ylabel('Turnover')
        axes[2].set_xlabel('Time Step')
        axes[2].set_title('Portfolio Turnover', fontsize=14, fontweight='bold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'performance_timeline.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_benchmark_comparison(self, results: Dict, output_dir: Path):
        """벤치마크 비교"""
        returns = np.array(results['returns'])
        cumulative = np.cumprod(1 + returns)

        # Equal-weight 벤치마크
        prices = np.array(results['price_data'])
        price_returns = np.diff(prices, axis=0) / prices[:-1, :]
        equal_weight_returns = price_returns.mean(axis=1)
        equal_weight_cumulative = np.cumprod(1 + equal_weight_returns)

        fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

        # (1) 누적 수익률 비교
        T = len(cumulative)
        axes[0].plot(cumulative, linewidth=2, label='IRT Portfolio', color='blue')
        axes[0].plot(range(T), equal_weight_cumulative[:T], linewidth=2,
                    label='Equal-Weight Benchmark', color='gray', linestyle='--')
        axes[0].set_ylabel('Cumulative Return')
        axes[0].set_title('Portfolio vs Benchmark', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # (2) 초과 수익률 (Outperformance)
        outperformance = cumulative - equal_weight_cumulative[:T]
        axes[1].fill_between(range(T), 0, outperformance, where=(outperformance >= 0),
                            color='green', alpha=0.3, label='Outperformance')
        axes[1].fill_between(range(T), 0, outperformance, where=(outperformance < 0),
                            color='red', alpha=0.3, label='Underperformance')
        axes[1].plot(outperformance, linewidth=1.5, color='black')
        axes[1].axhline(0, color='black', linestyle='-', linewidth=0.8)
        axes[1].set_ylabel('Excess Return')
        axes[1].set_xlabel('Time Step')
        axes[1].set_title('Outperformance vs Equal-Weight', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'benchmark_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_risk_dashboard(self, results: Dict, output_dir: Path):
        """리스크 대시보드"""
        returns = np.array(results['returns'])
        T = len(returns)

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # (1) 수익률 히스토그램 + VaR/CVaR
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(returns, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()

        ax1.axvline(var_95, color='orange', linestyle='--', linewidth=2, label=f'VaR(5%) = {var_95:.4f}')
        ax1.axvline(cvar_95, color='red', linestyle='--', linewidth=2, label=f'CVaR(5%) = {cvar_95:.4f}')
        ax1.set_xlabel('Daily Return')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Return Distribution & Risk Metrics', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # (2) Drawdown Waterfall
        ax2 = fig.add_subplot(gs[0, 1])
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max

        ax2.fill_between(range(T), 0, drawdown, color='red', alpha=0.3)
        ax2.plot(drawdown, linewidth=1.5, color='darkred')
        max_dd = drawdown.min()
        max_dd_idx = drawdown.argmin()
        ax2.scatter([max_dd_idx], [max_dd], color='black', s=100, zorder=5,
                   label=f'Max DD = {max_dd:.2%}')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Drawdown')
        ax2.set_title('Drawdown Waterfall', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # (3) Risk-Return Scatter (Rolling windows)
        ax3 = fig.add_subplot(gs[1, 0])
        window = min(60, T // 4)
        rolling_returns = []
        rolling_vols = []

        for i in range(window, T, 5):  # 5 step마다 샘플링
            window_rets = returns[i-window:i]
            rolling_returns.append(window_rets.mean() * 252)
            rolling_vols.append(window_rets.std() * np.sqrt(252))

        sc = ax3.scatter(rolling_vols, rolling_returns, c=range(len(rolling_vols)),
                        cmap='viridis', alpha=0.6, s=50)
        ax3.set_xlabel('Annualized Volatility')
        ax3.set_ylabel('Annualized Return')
        ax3.set_title(f'Risk-Return Profile (Rolling {window}d)', fontsize=12, fontweight='bold')
        plt.colorbar(sc, ax=ax3, label='Time')
        ax3.grid(True, alpha=0.3)

        # (4) Crisis vs Non-Crisis Performance
        ax4 = fig.add_subplot(gs[1, 1])
        crisis_levels = np.array(results['crisis_levels'])
        crisis_mask = crisis_levels > 0.5
        non_crisis_mask = ~crisis_mask

        crisis_returns = returns[crisis_mask]
        non_crisis_returns = returns[non_crisis_mask]

        data = [non_crisis_returns, crisis_returns]
        labels = ['Non-Crisis', 'Crisis (c>0.5)']
        bp = ax4.boxplot(data, labels=labels, patch_artist=True,
                        boxprops=dict(facecolor='lightblue', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2))

        ax4.axhline(0, color='black', linestyle='--', linewidth=0.8)
        ax4.set_ylabel('Daily Return')
        ax4.set_title('Returns: Crisis vs Non-Crisis', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')

        plt.suptitle('Risk Dashboard', fontsize=16, fontweight='bold', y=0.98)
        plt.savefig(output_dir / 'risk_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_tcell_analysis(self, results: Dict, output_dir: Path):
        """T-Cell 상세 분석: crisis types, danger signals"""
        crisis_types = np.array(results['crisis_types'])  # [T, K]
        crisis_levels = np.array(results['crisis_levels'])  # [T]
        returns = np.array(results['returns'])

        K = crisis_types.shape[1]  # Crisis type 개수
        type_names = [f'Type {i+1}' for i in range(K)]

        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

        # (1) Crisis Type Distribution (평균)
        ax1 = fig.add_subplot(gs[0, 0])
        mean_types = crisis_types.mean(axis=0)
        colors_types = plt.cm.Set3(np.linspace(0, 1, K))

        ax1.bar(range(K), mean_types, color=colors_types, edgecolor='black', linewidth=1.5)
        ax1.set_xticks(range(K))
        ax1.set_xticklabels(type_names, rotation=45)
        ax1.set_ylabel('Average Activation')
        ax1.set_title('Average Crisis Type Distribution', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')

        # (2) Crisis Type Time Series (Top 3)
        ax2 = fig.add_subplot(gs[0, 1])
        top_3_types = mean_types.argsort()[-3:][::-1]

        for idx in top_3_types:
            ax2.plot(crisis_types[:, idx], label=type_names[idx], linewidth=1.5, alpha=0.8)

        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Activation')
        ax2.set_title('Top 3 Crisis Types Over Time', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # (3) Crisis Level vs Returns Scatter
        ax3 = fig.add_subplot(gs[1, 0])
        scatter = ax3.scatter(crisis_levels, returns, c=crisis_levels,
                            cmap='Reds', alpha=0.5, s=30)
        ax3.axhline(0, color='black', linestyle='--', linewidth=0.8)
        ax3.axvline(0.5, color='orange', linestyle='--', linewidth=0.8, label='Crisis Threshold')
        ax3.set_xlabel('Crisis Level')
        ax3.set_ylabel('Daily Return')
        ax3.set_title('Crisis Level vs Returns', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax3, label='Crisis Level')

        # (4) Crisis Type Correlation Heatmap
        ax4 = fig.add_subplot(gs[1, 1])
        corr_matrix = np.corrcoef(crisis_types.T)
        im = ax4.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        ax4.set_xticks(range(K))
        ax4.set_yticks(range(K))
        ax4.set_xticklabels(type_names, rotation=45, ha='right')
        ax4.set_yticklabels(type_names)
        ax4.set_title('Crisis Type Correlation', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax4, label='Correlation')

        # 값 표시
        for i in range(K):
            for j in range(K):
                text = ax4.text(j, i, f'{corr_matrix[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=8)

        # (5) Crisis Regime Analysis (High/Medium/Low)
        ax5 = fig.add_subplot(gs[2, :])
        high_crisis = crisis_levels > 0.7
        medium_crisis = (crisis_levels >= 0.3) & (crisis_levels <= 0.7)
        low_crisis = crisis_levels < 0.3

        regimes = ['Low\n(c<0.3)', 'Medium\n(0.3≤c≤0.7)', 'High\n(c>0.7)']
        regime_counts = [low_crisis.sum(), medium_crisis.sum(), high_crisis.sum()]
        regime_returns = [
            returns[low_crisis].mean() if low_crisis.any() else 0,
            returns[medium_crisis].mean() if medium_crisis.any() else 0,
            returns[high_crisis].mean() if high_crisis.any() else 0
        ]

        x = np.arange(len(regimes))
        width = 0.35

        bars1 = ax5.bar(x - width/2, regime_counts, width, label='Count', color='steelblue', alpha=0.7)
        ax5_twin = ax5.twinx()
        bars2 = ax5_twin.bar(x + width/2, regime_returns, width, label='Avg Return',
                            color='coral', alpha=0.7)

        ax5.set_xlabel('Crisis Regime')
        ax5.set_ylabel('Count', color='steelblue')
        ax5_twin.set_ylabel('Average Return', color='coral')
        ax5.set_title('Performance by Crisis Regime', fontsize=12, fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels(regimes)
        ax5.tick_params(axis='y', labelcolor='steelblue')
        ax5_twin.tick_params(axis='y', labelcolor='coral')
        ax5.legend(loc='upper left')
        ax5_twin.legend(loc='upper right')
        ax5.grid(True, alpha=0.3, axis='y')

        plt.suptitle('T-Cell Crisis Analysis', fontsize=16, fontweight='bold', y=0.99)
        plt.savefig(output_dir / 'tcell_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_attribution_analysis(self, results: Dict, output_dir: Path):
        """Attribution Analysis: 종목별/프로토타입별 수익 기여도"""
        weights = np.array(results['weights'])  # [T, N]
        proto_weights = np.array(results['prototype_weights'])  # [T, M]
        returns = np.array(results['returns'])
        symbols = results['symbols']
        prices = np.array(results['price_data'])

        N = weights.shape[1]
        M = proto_weights.shape[1]
        T = len(returns)

        # 종목별 수익률 계산
        price_returns = np.diff(prices, axis=0) / prices[:-1, :]  # [T-1, N]
        price_returns = price_returns[:T, :]  # 길이 맞추기

        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

        # (1) 종목별 수익 기여도 (Top 10)
        ax1 = fig.add_subplot(gs[0, :])
        # 각 종목의 기여도 = weight * return (누적)
        contributions = weights[:-1, :] * price_returns
        total_contributions = contributions.sum(axis=0)
        top_10_idx = np.abs(total_contributions).argsort()[-10:][::-1]

        colors = ['green' if c > 0 else 'red' for c in total_contributions[top_10_idx]]
        ax1.barh(range(10), total_contributions[top_10_idx], color=colors, edgecolor='black')
        ax1.set_yticks(range(10))
        ax1.set_yticklabels([symbols[i] for i in top_10_idx])
        ax1.axvline(0, color='black', linestyle='-', linewidth=0.8)
        ax1.set_xlabel('Cumulative Return Contribution')
        ax1.set_title('Top 10 Stock Contributions (Cumulative)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')

        # (2) 종목별 기여도 시계열 (Top 3)
        ax2 = fig.add_subplot(gs[1, 0])
        top_3_idx = np.abs(total_contributions).argsort()[-3:][::-1]

        cumulative_contrib = np.cumsum(contributions, axis=0)
        for idx in top_3_idx:
            ax2.plot(cumulative_contrib[:, idx], label=symbols[idx], linewidth=2)

        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Cumulative Contribution')
        ax2.set_title('Top 3 Stock Contributions Over Time', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # (3) 프로토타입별 활용도 (평균 가중치)
        ax3 = fig.add_subplot(gs[1, 1])
        proto_mean = proto_weights.mean(axis=0)
        proto_names = [f'Proto {i+1}' for i in range(M)]
        colors_proto = plt.cm.tab10(np.linspace(0, 1, M))

        ax3.bar(range(M), proto_mean, color=colors_proto, edgecolor='black')
        ax3.set_xticks(range(M))
        ax3.set_xticklabels(proto_names, rotation=45)
        ax3.set_ylabel('Average Weight')
        ax3.set_title('Prototype Utilization', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')

        # (4) 프로토타입 기여도 추정 (간접 계산)
        # 프로토타입별 가중치 변화와 수익률의 상관관계
        ax4 = fig.add_subplot(gs[2, :])
        proto_contrib = np.zeros(M)

        for j in range(M):
            # 프로토타입 j가 활성화될 때의 평균 수익률
            high_activation = proto_weights[:, j] > proto_weights[:, j].mean()
            if high_activation.any():
                proto_contrib[j] = returns[high_activation].mean()

        colors_contrib = ['green' if c > 0 else 'red' for c in proto_contrib]
        ax4.bar(range(M), proto_contrib, color=colors_contrib, edgecolor='black')
        ax4.set_xticks(range(M))
        ax4.set_xticklabels(proto_names, rotation=45)
        ax4.axhline(0, color='black', linestyle='-', linewidth=0.8)
        ax4.set_ylabel('Average Return (High Activation)')
        ax4.set_title('Prototype Performance Attribution', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')

        plt.suptitle('Attribution Analysis', fontsize=16, fontweight='bold', y=0.99)
        plt.savefig(output_dir / 'attribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_cost_matrix(self, results: Dict, output_dir: Path):
        """Cost Matrix Heatmap: Immunological cost structure"""
        cost_matrices = results['cost_matrices']  # List of [m, M] matrices (샘플링됨)

        if len(cost_matrices) == 0:
            self.logger.warning("Cost matrix 데이터가 없어 시각화를 건너뜁니다.")
            return

        # 평균 cost matrix 계산
        avg_cost = np.mean(cost_matrices, axis=0)  # [m, M]
        m, M = avg_cost.shape

        # 시간에 따른 cost 변화 (첫/중간/마지막)
        n_samples = len(cost_matrices)
        samples_to_plot = [0, n_samples//2, n_samples-1] if n_samples > 2 else [0]

        fig = plt.figure(figsize=(16, 10))
        n_plots = len(samples_to_plot) + 1  # +1 for average
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # (1) Average Cost Matrix
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(avg_cost, cmap='RdYlGn_r', aspect='auto')
        ax1.set_xlabel('Prototype Index (M)')
        ax1.set_ylabel('Epitope Index (m)')
        ax1.set_title('Average Immunological Cost Matrix', fontsize=12, fontweight='bold')
        ax1.set_xticks(range(M))
        ax1.set_xticklabels([f'P{i+1}' for i in range(M)])
        ax1.set_yticks(range(m))
        ax1.set_yticklabels([f'E{i+1}' for i in range(m)])
        plt.colorbar(im1, ax=ax1, label='Cost')

        # (2) Cost Distribution
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(avg_cost.flatten(), bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        ax2.axvline(avg_cost.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean = {avg_cost.mean():.3f}')
        ax2.set_xlabel('Cost Value')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Cost Distribution', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # (3) Cost Evolution (Early)
        if n_samples > 0:
            ax3 = fig.add_subplot(gs[1, 0])
            im3 = ax3.imshow(cost_matrices[0], cmap='RdYlGn_r', aspect='auto')
            ax3.set_xlabel('Prototype Index (M)')
            ax3.set_ylabel('Epitope Index (m)')
            ax3.set_title('Early Episode Cost', fontsize=12, fontweight='bold')
            ax3.set_xticks(range(M))
            ax3.set_xticklabels([f'P{i+1}' for i in range(M)])
            ax3.set_yticks(range(m))
            ax3.set_yticklabels([f'E{i+1}' for i in range(m)])
            plt.colorbar(im3, ax=ax3, label='Cost')

        # (4) Cost Evolution (Late)
        if n_samples > 1:
            ax4 = fig.add_subplot(gs[1, 1])
            im4 = ax4.imshow(cost_matrices[-1], cmap='RdYlGn_r', aspect='auto')
            ax4.set_xlabel('Prototype Index (M)')
            ax4.set_ylabel('Epitope Index (m)')
            ax4.set_title('Late Episode Cost', fontsize=12, fontweight='bold')
            ax4.set_xticks(range(M))
            ax4.set_xticklabels([f'P{i+1}' for i in range(M)])
            ax4.set_yticks(range(m))
            ax4.set_yticklabels([f'E{i+1}' for i in range(m)])
            plt.colorbar(im4, ax=ax4, label='Cost')

        plt.suptitle('Immunological Cost Matrix (γ, λ, ρ effects)', fontsize=16, fontweight='bold', y=0.99)
        plt.savefig(output_dir / 'cost_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_insights(self, results: Dict) -> Dict:
        """
        구조화된 해석 정보 생성

        그림 없이도 JSON으로 IRT 의사결정을 해석할 수 있도록
        핵심 인사이트를 추출한다.

        Args:
            results: evaluation_results.json과 동일한 구조

        Returns:
            insights: 구조화된 해석 정보
        """
        returns = np.array(results['returns'])
        weights = np.array(results['weights'])
        crisis_levels = np.array(results['crisis_levels'])
        crisis_types = np.array(results['crisis_types'])
        proto_weights = np.array(results['prototype_weights'])
        w_rep = np.array(results['w_rep'])
        w_ot = np.array(results['w_ot'])
        eta_list = np.array(results['eta'])
        symbols = results['symbols']

        n_steps = len(returns)
        n_assets = weights.shape[1]

        # ===== 1. Summary =====
        metrics = results['metrics']
        summary = {
            'total_return': float(metrics['total_return']),
            'sharpe_ratio': float(metrics['sharpe_ratio']),
            'sortino_ratio': float(metrics['sortino_ratio']),
            'calmar_ratio': float(metrics['calmar_ratio']),
            'max_drawdown': float(metrics['max_drawdown']),
            'avg_crisis_level': float(metrics['avg_crisis_level']),
            'total_steps': int(n_steps)
        }

        # ===== 2. Top Holdings =====
        avg_weights = weights.mean(axis=0)  # [n_assets]
        top_indices = np.argsort(avg_weights)[::-1][:10]  # Top 10

        # 각 자산의 수익 기여도 계산 (weight * portfolio_return)
        portfolio_returns = returns
        weight_contributions = []
        for i in range(n_assets):
            # 근사: avg_weight * total_return
            contrib = avg_weights[i] * metrics['total_return']
            weight_contributions.append(contrib)
        weight_contributions = np.array(weight_contributions)

        top_holdings = []
        for idx in top_indices:
            top_holdings.append({
                'symbol': symbols[idx] if idx < len(symbols) else f'Asset_{idx}',
                'avg_weight': float(avg_weights[idx]),
                'contribution': float(weight_contributions[idx])
            })

        # ===== 3. Crisis vs Normal Analysis =====
        crisis_threshold = 0.5
        crisis_mask = crisis_levels > crisis_threshold
        normal_mask = ~crisis_mask

        crisis_returns = returns[crisis_mask] if crisis_mask.sum() > 0 else np.array([0])
        normal_returns = returns[normal_mask] if normal_mask.sum() > 0 else np.array([0])

        def safe_sharpe(rets):
            if len(rets) == 0 or rets.std() == 0:
                return 0.0
            return float(rets.mean() / rets.std() * np.sqrt(252))

        crisis_vs_normal = {
            'crisis': {
                'sharpe': safe_sharpe(crisis_returns),
                'avg_return': float(crisis_returns.mean()),
                'volatility': float(crisis_returns.std()),
                'steps': int(crisis_mask.sum())
            },
            'normal': {
                'sharpe': safe_sharpe(normal_returns),
                'avg_return': float(normal_returns.mean()),
                'volatility': float(normal_returns.std()),
                'steps': int(normal_mask.sum())
            }
        }

        # ===== 4. IRT Decomposition =====
        # w = (1-α)·w_rep + α·w_ot
        # Contribution: L2 norm 기준
        alpha = self.config['irt']['alpha']
        w_rep_contrib = (1 - alpha) * np.linalg.norm(w_rep, axis=1).mean()
        w_ot_contrib = alpha * np.linalg.norm(w_ot, axis=1).mean()
        total_contrib = w_rep_contrib + w_ot_contrib

        irt_decomposition = {
            'avg_w_rep_contribution': float(w_rep_contrib / total_contrib) if total_contrib > 0 else 0.0,
            'avg_w_ot_contribution': float(w_ot_contrib / total_contrib) if total_contrib > 0 else 0.0,
            'correlation_w_rep_w_ot': float(np.corrcoef(w_rep.flatten(), w_ot.flatten())[0, 1]),
            'avg_eta': float(eta_list.mean()),
            'max_eta': float(eta_list.max()),
            'min_eta': float(eta_list.min())
        }

        # ===== 5. Prototype Analysis =====
        avg_proto_weights = proto_weights.mean(axis=0)  # [M]
        top_proto_indices = np.argsort(avg_proto_weights)[::-1][:3].tolist()

        # Entropy: -Σ p·log(p)
        entropy = -np.sum(proto_weights * np.log(proto_weights + 1e-8), axis=1)

        prototype_analysis = {
            'most_used_prototypes': top_proto_indices,
            'prototype_avg_weights': avg_proto_weights.tolist(),
            'avg_entropy': float(entropy.mean()),
            'max_entropy': float(entropy.max()),
            'min_entropy': float(entropy.min())
        }

        # ===== 6. Risk Metrics =====
        risk_metrics = {
            'VaR_5': float(metrics.get('var_5', 0.0)),
            'CVaR_5': float(metrics.get('cvar_5', 0.0)),
            'downside_deviation': float(metrics.get('downside_deviation', 0.0)),
            'avg_turnover': float(metrics.get('avg_turnover', 0.0))
        }

        # ===== 7. T-Cell Insights =====
        crisis_regime_pct = crisis_mask.sum() / len(crisis_mask)

        # Crisis types: [n_steps, K] → K차원별 평균
        avg_crisis_types = crisis_types.mean(axis=0)  # [K]
        top_crisis_type_indices = np.argsort(avg_crisis_types)[::-1][:3].tolist()

        tcell_insights = {
            'crisis_regime_pct': float(crisis_regime_pct),
            'top_crisis_types': top_crisis_type_indices,
            'avg_crisis_type_distribution': avg_crisis_types.tolist(),
            'avg_danger_level': float(crisis_levels.mean())
        }

        # ===== 종합 =====
        insights = {
            'summary': summary,
            'top_holdings': top_holdings,
            'crisis_vs_normal': crisis_vs_normal,
            'irt_decomposition': irt_decomposition,
            'prototype_analysis': prototype_analysis,
            'risk_metrics': risk_metrics,
            'tcell_insights': tcell_insights
        }

        return insights

def main():
    parser = argparse.ArgumentParser(description='IRT Evaluation Script')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--config', type=str, default='configs/default_irt.yaml',
                       help='Path to config file')
    args = parser.parse_args()

    # 설정 로드
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 평가 실행
    evaluator = IRTEvaluator(config, args.checkpoint)
    metrics = evaluator.evaluate()

if __name__ == '__main__':
    main()