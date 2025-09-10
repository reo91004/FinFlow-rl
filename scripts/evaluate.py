# scripts/evaluate.py

import numpy as np
import pandas as pd
import torch
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.core.env import PortfolioEnv
from src.agents.b_cell import BCell
from src.agents.t_cell import TCell
from src.agents.memory import MemoryCell
from src.agents.gating import GatingNetwork
from src.analysis.explainer import XAIExplainer
from src.analysis.metrics import calculate_sharpe_ratio, calculate_cvar, calculate_max_drawdown
from src.analysis.visualization import plot_portfolio_weights, plot_equity_curve, plot_drawdown
from src.utils.logger import FinFlowLogger, get_session_directory
from src.utils.seed import set_seed

class FinFlowEvaluator:
    """
    FinFlow 시스템 평가기
    
    백테스팅, 벤치마크 비교, 성능 분석
    """
    
    def __init__(self,
                 checkpoint_path: str,
                 data_path: str,
                 config_path: Optional[str] = None,
                 device: str = 'cpu'):
        """
        Args:
            checkpoint_path: 체크포인트 경로
            data_path: 평가 데이터 경로
            config_path: 설정 파일 경로
            device: 디바이스
        """
        self.checkpoint_path = checkpoint_path
        self.data_path = data_path
        self.device = device
        
        self.logger = FinFlowLogger("Evaluator")
        self.logger.info("평가기 초기화")
        
        # Load configuration
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = self._get_default_config()
        
        # Initialize components
        self._load_models()
        self._load_data()
        
        # Results storage
        self.results = {}
        self.session_dir = get_session_directory()
        self.run_dir = self.session_dir
        (self.run_dir / "reports").mkdir(parents=True, exist_ok=True)
        self.viz_dir = self.run_dir / "reports"
        self.viz_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_default_config(self) -> Dict:
        """기본 설정"""
        return {
            'env': {
                'initial_balance': 1000000,
                'transaction_cost': 0.001,
                'max_weight': 0.2,
                'min_weight': 0.0,
                'window_size': 30
            },
            'evaluation': {
                'n_episodes': 10,
                'benchmarks': ['equal_weight', 'market_cap', 'momentum'],
                'metrics': ['sharpe', 'cvar', 'max_drawdown', 'turnover']
            }
        }
    
    def _load_models(self):
        """모델 로드"""
        self.logger.info(f"체크포인트 로드: {self.checkpoint_path}")
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Get dimensions from checkpoint
        state_dim = 43  # Default
        action_dim = 10  # Default
        
        # Initialize components
        self.b_cell = BCell(state_dim, action_dim, device=self.device)
        self.b_cell.load_state_dict(checkpoint['b_cell'])
        self.b_cell.eval()
        
        self.gating_network = GatingNetwork(state_dim).to(self.device)
        self.gating_network.load_state_dict(checkpoint['gating_network'])
        self.gating_network.eval()
        
        self.t_cell = TCell()
        if 't_cell' in checkpoint:
            self.t_cell.load_state(checkpoint['t_cell'])
        
        self.memory_cell = MemoryCell()
        if 'memory_cell' in checkpoint:
            memory_data = checkpoint['memory_cell']
            self.memory_cell.memories = memory_data.get('memories', [])
        
        # Initialize XAI
        feature_names = self._get_feature_names()
        action_names = [f"Asset_{i}" for i in range(action_dim)]
        
        self.explainer = XAIExplainer(
            self.b_cell.actor,
            feature_names,
            action_names,
            self.device
        )
        
        self.logger.info("모델 로드 완료")
    
    def _load_data(self):
        """데이터 로드"""
        self.logger.info(f"데이터 로드: {self.data_path}")
        
        # Load price data
        data_file = Path(self.data_path) / "test_data.npz"
        if data_file.exists():
            data = np.load(data_file)
            self.prices = data['prices']
            self.returns = data['returns']
            self.features = data.get('features', None)
        else:
            # Generate synthetic data for testing
            self.logger.warning("테스트 데이터 없음 - 합성 데이터 생성")
            self.prices = np.random.randn(252, 10).cumsum(axis=0) + 100
            self.prices = np.exp(self.prices / 100)
            self.returns = np.diff(self.prices, axis=0) / self.prices[:-1]
    
    def _get_feature_names(self) -> List[str]:
        """특징 이름 생성"""
        feature_names = []
        
        # Market features
        for i in range(10):
            feature_names.extend([
                f"return_{i}",
                f"volatility_{i}",
                f"volume_{i}"
            ])
        
        # Portfolio features
        for i in range(10):
            feature_names.append(f"weight_{i}")
        
        # Crisis features
        feature_names.extend([
            "crisis_overall",
            "crisis_volatility",
            "crisis_correlation"
        ])
        
        return feature_names[:43]  # Ensure 43 features
    
    def evaluate(self):
        """전체 평가 실행"""
        self.logger.info("=" * 50)
        self.logger.info("평가 시작")
        self.logger.info("=" * 50)
        
        # 1. Backtest FinFlow
        self.logger.info("\n1. FinFlow 백테스팅")
        finflow_results = self._backtest_finflow()
        self.results['finflow'] = finflow_results
        
        # 2. Benchmark comparisons
        self.logger.info("\n2. 벤치마크 비교")
        benchmark_results = self._evaluate_benchmarks()
        self.results['benchmarks'] = benchmark_results
        
        # 3. Stability analysis
        self.logger.info("\n3. 안정성 분석")
        stability_results = self._analyze_stability()
        self.results['stability'] = stability_results
        
        # 4. XAI analysis
        self.logger.info("\n4. XAI 분석")
        xai_results = self._analyze_explainability()
        self.results['xai'] = xai_results
        
        # 5. Generate visualizations
        self.logger.info("\n5. 시각화 생성")
        self._create_visualizations()
        
        # 6. Generate report
        self.logger.info("\n6. 보고서 생성")
        self._generate_report()
        
        self.logger.info("=" * 50)
        self.logger.info("평가 완료")
        self._print_summary()
        self.logger.info("=" * 50)
    
    def _backtest_finflow(self) -> Dict:
        """FinFlow 백테스팅"""
        env = PortfolioEnv(**self.config['env'])
        
        episode_returns = []
        episode_actions = []
        episode_rewards = []
        episode_equity_curves = []
        xai_reports = []
        
        for episode in tqdm(range(self.config['evaluation']['n_episodes']), desc="백테스팅"):
            state, _ = env.reset()
            done = False
            
            returns = []
            actions = []
            rewards = []
            equity_curve = [1.0]
            step_count = 0
            
            while not done:
                # Get action from FinFlow
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                
                # Crisis detection
                crisis_info = self.t_cell.detect_crisis(env.get_market_data())
                
                # Memory guidance
                memory_guidance = self.memory_cell.get_memory_guidance(
                    state, crisis_info['overall_crisis']
                )
                
                # Gating decision
                gating_decision = self.gating_network(
                    state_tensor, memory_guidance, crisis_info['overall_crisis']
                )
                
                # Select action
                action = self.b_cell.select_action(
                    state_tensor,
                    bcell_type=gating_decision.selected_bcell,
                    deterministic=True
                )
                
                # XAI Analysis - 마지막 스텝 또는 10스텝마다
                if step_count % 10 == 0 or done:
                    # XAI 3함수 호출
                    local_attr = self.explainer.local_attribution(state, action)
                    cf_report = self.explainer.counterfactual(state, action, deltas={"volatility": -0.2})
                    reg_report = self.explainer.regime_report(
                        crisis_info, 
                        shap_topk=5, 
                        similar_cases=memory_guidance
                    )
                    
                    # Decision card 생성 및 저장
                    decision_card = {
                        "timestamp": datetime.now().isoformat(),
                        "episode": episode,
                        "step": step_count,
                        "action": list(map(float, action)),
                        "local_attribution": local_attr,
                        "counterfactual": cf_report,
                        "regime_report": reg_report,
                        "crisis_info": crisis_info
                    }
                    
                    # JSON 저장
                    card_path = self.run_dir / "reports" / f"decision_card_ep{episode}_step{step_count}.json"
                    with open(card_path, 'w') as f:
                        json.dump(decision_card, f, indent=2)
                    
                    xai_reports.append(decision_card)
                
                # Step
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                portfolio_return = info.get('portfolio_return', reward)
                returns.append(portfolio_return)
                actions.append(action)
                rewards.append(reward)
                equity_curve.append(equity_curve[-1] * (1 + portfolio_return))
                
                state = next_state
                step_count += 1
            
            episode_returns.append(returns)
            episode_actions.append(actions)
            episode_rewards.append(rewards)
            episode_equity_curves.append(equity_curve)
        
        # Calculate metrics
        all_returns = np.concatenate(episode_returns)
        all_actions = np.vstack(np.concatenate(episode_actions))
        all_equity = np.concatenate(episode_equity_curves)
        
        # 지표 계산
        sharpe = calculate_sharpe_ratio(all_returns, risk_free_rate=0.0)
        cvar_95 = calculate_cvar(all_returns, alpha=0.05)
        mdd = calculate_max_drawdown(all_equity)
        
        # 기본 메트릭과 새 메트릭 통합
        metrics = self._calculate_metrics(all_returns, all_actions)
        metrics.update({
            'sharpe_ratio_new': float(sharpe),
            'cvar_95_new': float(cvar_95),
            'max_drawdown_new': float(mdd)
        })
        
        # 메트릭 리포트 저장
        metrics_report = {
            "sharpe": float(sharpe),
            "cvar_95": float(cvar_95),
            "max_drawdown": float(mdd),
            "total_return": float(metrics['total_return']),
            "annual_return": float(metrics['annual_return']),
            "volatility": float(metrics['volatility'])
        }
        (self.run_dir / "reports" / "metrics.json").write_text(json.dumps(metrics_report, indent=2))
        
        # 시각화 생성 및 저장
        # Equity curve
        plot_equity_curve(all_equity, save_path=self.run_dir / "reports" / "equity_curve.png")
        
        # Drawdown
        plot_drawdown(all_equity, save_path=self.run_dir / "reports" / "drawdown.png")
        
        # Portfolio weights
        asset_names = [f"Asset_{i}" for i in range(all_actions.shape[1])]
        # 최근 가중치 사용 (시계열 대신 마지막 스텝의 스냅샷)
        latest_weights = all_actions[-1] if len(all_actions) > 0 else all_actions[0]
        plot_portfolio_weights(
            latest_weights, 
            asset_names,
            save_path=self.run_dir / "reports" / "weights.png"
        )
        
        self.logger.info(f"XAI 리포트 {len(xai_reports)}개 생성 완료")
        self.logger.info(f"메트릭 및 시각화 저장 완료: {self.run_dir / 'reports'}")
        
        return {
            'metrics': metrics,
            'returns': all_returns,
            'actions': all_actions,
            'rewards': np.concatenate(episode_rewards),
            'equity_curve': all_equity,
            'xai_reports': xai_reports
        }
    
    def _evaluate_benchmarks(self) -> Dict:
        """벤치마크 전략 평가"""
        benchmarks = {}
        
        for strategy in self.config['evaluation']['benchmarks']:
            self.logger.info(f"벤치마크 평가: {strategy}")
            
            if strategy == 'equal_weight':
                returns, actions = self._equal_weight_strategy()
            elif strategy == 'market_cap':
                returns, actions = self._market_cap_strategy()
            elif strategy == 'momentum':
                returns, actions = self._momentum_strategy()
            else:
                continue
            
            metrics = self._calculate_metrics(returns, actions)
            benchmarks[strategy] = {
                'metrics': metrics,
                'returns': returns,
                'actions': actions
            }
        
        return benchmarks
    
    def _equal_weight_strategy(self) -> Tuple[np.ndarray, np.ndarray]:
        """균등 가중 전략"""
        n_assets = self.returns.shape[1]
        weights = np.ones(n_assets) / n_assets
        
        returns = []
        actions = []
        
        for t in range(len(self.returns)):
            portfolio_return = np.dot(weights, self.returns[t])
            returns.append(portfolio_return)
            actions.append(weights.copy())
        
        return np.array(returns), np.array(actions)
    
    def _market_cap_strategy(self) -> Tuple[np.ndarray, np.ndarray]:
        """시가총액 가중 전략 (시뮬레이션)"""
        # Simulate market cap weights
        market_caps = np.random.lognormal(10, 2, self.returns.shape[1])
        weights = market_caps / market_caps.sum()
        
        returns = []
        actions = []
        
        for t in range(len(self.returns)):
            portfolio_return = np.dot(weights, self.returns[t])
            returns.append(portfolio_return)
            actions.append(weights.copy())
            
            # Update weights based on returns
            weights = weights * (1 + self.returns[t])
            weights = weights / weights.sum()
        
        return np.array(returns), np.array(actions)
    
    def _momentum_strategy(self) -> Tuple[np.ndarray, np.ndarray]:
        """모멘텀 전략"""
        lookback = 20
        returns_list = []
        actions_list = []
        
        for t in range(lookback, len(self.returns)):
            # Calculate momentum
            past_returns = self.returns[t-lookback:t].mean(axis=0)
            
            # Rank and select top assets
            ranks = np.argsort(past_returns)[::-1]
            weights = np.zeros(len(ranks))
            weights[ranks[:3]] = 1/3  # Top 3 assets
            
            portfolio_return = np.dot(weights, self.returns[t])
            returns_list.append(portfolio_return)
            actions_list.append(weights)
        
        return np.array(returns_list), np.array(actions_list)
    
    def _calculate_metrics(self, returns: np.ndarray, actions: np.ndarray) -> Dict:
        """성능 메트릭 계산"""
        metrics = {}
        
        # Sharpe Ratio
        if len(returns) > 1:
            metrics['sharpe_ratio'] = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        else:
            metrics['sharpe_ratio'] = 0
        
        # CVaR (5%)
        sorted_returns = np.sort(returns)
        n_tail = max(1, len(sorted_returns) // 20)
        metrics['cvar_5'] = np.mean(sorted_returns[:n_tail])
        
        # Max Drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        metrics['max_drawdown'] = np.min(drawdown)
        
        # Turnover
        if len(actions) > 1:
            turnovers = [np.sum(np.abs(actions[i] - actions[i-1])) / 2 
                        for i in range(1, len(actions))]
            metrics['avg_turnover'] = np.mean(turnovers)
        else:
            metrics['avg_turnover'] = 0
        
        # Additional metrics
        metrics['total_return'] = np.prod(1 + returns) - 1
        metrics['annual_return'] = (1 + metrics['total_return']) ** (252 / len(returns)) - 1
        metrics['volatility'] = np.std(returns) * np.sqrt(252)
        metrics['win_rate'] = np.mean(returns > 0)
        
        return metrics
    
    def _analyze_stability(self) -> Dict:
        """안정성 분석"""
        finflow_actions = self.results['finflow']['actions']
        
        # Weight stability
        weight_changes = np.diff(finflow_actions, axis=0)
        weight_stability = 1 / (1 + np.std(weight_changes))
        
        # Concentration
        concentrations = np.sum(finflow_actions ** 2, axis=1)
        avg_concentration = np.mean(concentrations)
        
        # Effective assets
        effective_assets = np.sum(finflow_actions > 0.01, axis=1)
        avg_effective = np.mean(effective_assets)
        
        return {
            'weight_stability': weight_stability,
            'avg_concentration': avg_concentration,
            'avg_effective_assets': avg_effective,
            'max_single_weight': np.max(finflow_actions),
            'min_single_weight': np.min(finflow_actions[finflow_actions > 0])
        }
    
    def _analyze_explainability(self) -> Dict:
        """XAI 분석"""
        # Sample some decisions for explanation
        sample_states = []
        sample_actions = []
        
        env = PortfolioEnv(**self.config['env'])
        state, _ = env.reset()
        
        for _ in range(5):  # Explain 5 decisions
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Get action
            action = self.b_cell.select_action(state_tensor, deterministic=True)
            
            sample_states.append(state)
            sample_actions.append(action)
            
            # Step (for next state)
            state, _, done, _, _ = env.step(action)
            if done:
                break
        
        # Generate explanations
        explanations = []
        for state, action in zip(sample_states, sample_actions):
            explanation = self.explainer.explain_decision(state, action)
            explanations.append(explanation)
        
        # Aggregate feature importance
        all_importance = {}
        for exp in explanations:
            for feature, importance in exp.feature_importance.items():
                if feature not in all_importance:
                    all_importance[feature] = []
                all_importance[feature].append(importance)
        
        avg_importance = {k: np.mean(v) for k, v in all_importance.items()}
        
        return {
            'avg_feature_importance': avg_importance,
            'n_explanations': len(explanations),
            'avg_confidence': np.mean([exp.confidence for exp in explanations])
        }
    
    def _create_visualizations(self):
        """시각화 생성"""
        # 1. Performance comparison
        self._plot_performance_comparison()
        
        # 2. Returns distribution
        self._plot_returns_distribution()
        
        # 3. Portfolio weights over time
        self._plot_portfolio_weights()
        
        # 4. Drawdown analysis
        self._plot_drawdown()
    
    def _plot_performance_comparison(self):
        """성능 비교 플롯"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Collect metrics
        strategies = ['FinFlow'] + list(self.results['benchmarks'].keys())
        metrics_data = {
            'Sharpe Ratio': [],
            'CVaR (5%)': [],
            'Max Drawdown': [],
            'Turnover': []
        }
        
        for strategy in strategies:
            if strategy == 'FinFlow':
                m = self.results['finflow']['metrics']
            else:
                m = self.results['benchmarks'][strategy]['metrics']
            
            metrics_data['Sharpe Ratio'].append(m['sharpe_ratio'])
            metrics_data['CVaR (5%)'].append(m['cvar_5'])
            metrics_data['Max Drawdown'].append(m['max_drawdown'])
            metrics_data['Turnover'].append(m['avg_turnover'])
        
        # Plot each metric
        for ax, (metric_name, values) in zip(axes.flat, metrics_data.items()):
            ax.bar(strategies, values)
            ax.set_title(metric_name)
            ax.set_ylabel('Value')
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / "performance_comparison.png")
        plt.close()
    
    def _plot_returns_distribution(self):
        """수익률 분포 플롯"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot returns distribution
        finflow_returns = self.results['finflow']['returns']
        
        ax.hist(finflow_returns, bins=50, alpha=0.7, label='FinFlow', density=True)
        
        for name, data in self.results['benchmarks'].items():
            ax.hist(data['returns'], bins=50, alpha=0.5, label=name, density=True)
        
        ax.set_xlabel('Returns')
        ax.set_ylabel('Density')
        ax.set_title('Returns Distribution')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / "returns_distribution.png")
        plt.close()
    
    def _plot_portfolio_weights(self):
        """포트폴리오 가중치 플롯"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        actions = self.results['finflow']['actions']
        
        # Plot stacked area chart
        x = np.arange(len(actions))
        
        # Select top 5 assets by average weight
        avg_weights = actions.mean(axis=0)
        top_assets = np.argsort(avg_weights)[::-1][:5]
        
        bottom = np.zeros(len(actions))
        for i in top_assets:
            ax.fill_between(x, bottom, bottom + actions[:, i], 
                           label=f'Asset {i}', alpha=0.7)
            bottom += actions[:, i]
        
        # Others
        others = actions[:, [i for i in range(actions.shape[1]) if i not in top_assets]].sum(axis=1)
        ax.fill_between(x, bottom, bottom + others, label='Others', alpha=0.7)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Portfolio Weight')
        ax.set_title('Portfolio Weights Over Time')
        ax.legend(loc='upper right')
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / "portfolio_weights.png")
        plt.close()
    
    def _plot_drawdown(self):
        """드로다운 플롯"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Calculate drawdown for FinFlow
        returns = self.results['finflow']['returns']
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        ax.fill_between(np.arange(len(drawdown)), 0, drawdown, 
                        color='red', alpha=0.3, label='FinFlow')
        ax.plot(drawdown, color='red', linewidth=2)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Drawdown')
        ax.set_title('Drawdown Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / "drawdown.png")
        plt.close()
    
    def _generate_report(self):
        """평가 보고서 생성"""
        report_path = get_session_directory() / "reports" / "evaluation_report.json"
        report_path.parent.mkdir(exist_ok=True)
        
        # Save detailed results
        with open(report_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = self._make_serializable(self.results)
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"평가 보고서 저장: {report_path}")
    
    def _make_serializable(self, obj):
        """객체를 JSON 직렬화 가능하게 변환"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        else:
            return obj
    
    def _print_summary(self):
        """요약 출력"""
        print("\n" + "=" * 50)
        print("평가 요약")
        print("=" * 50)
        
        # FinFlow metrics
        finflow_metrics = self.results['finflow']['metrics']
        print(f"\nFinFlow 성능:")
        print(f"  Sharpe Ratio: {finflow_metrics['sharpe_ratio']:.3f}")
        print(f"  CVaR (5%): {finflow_metrics['cvar_5']:.3f}")
        print(f"  Max Drawdown: {finflow_metrics['max_drawdown']:.3f}")
        print(f"  Annual Return: {finflow_metrics['annual_return']*100:.1f}%")
        print(f"  Volatility: {finflow_metrics['volatility']*100:.1f}%")
        
        # Benchmark comparison
        print(f"\n벤치마크 대비:")
        for name, data in self.results['benchmarks'].items():
            metrics = data['metrics']
            sharpe_diff = finflow_metrics['sharpe_ratio'] - metrics['sharpe_ratio']
            print(f"  vs {name}: Sharpe +{sharpe_diff:.3f}" if sharpe_diff > 0 
                  else f"  vs {name}: Sharpe {sharpe_diff:.3f}")
        
        # Stability
        stability = self.results['stability']
        print(f"\n안정성 지표:")
        print(f"  Weight Stability: {stability['weight_stability']:.3f}")
        print(f"  Avg Concentration: {stability['avg_concentration']:.3f}")
        print(f"  Avg Effective Assets: {stability['avg_effective_assets']:.1f}")
        
        # XAI
        xai = self.results['xai']
        print(f"\nXAI 분석:")
        print(f"  Avg Confidence: {xai['avg_confidence']:.3f}")
        print(f"  Top Features:")
        
        top_features = sorted(xai['avg_feature_importance'].items(), 
                             key=lambda x: x[1], reverse=True)[:3]
        for feature, importance in top_features:
            print(f"    - {feature}: {importance*100:.1f}%")


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='FinFlow Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--data', type=str, default='data/test',
                       help='Path to test data')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create evaluator
    evaluator = FinFlowEvaluator(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        config_path=args.config,
        device=args.device
    )
    
    # Run evaluation
    evaluator.evaluate()


if __name__ == "__main__":
    main()