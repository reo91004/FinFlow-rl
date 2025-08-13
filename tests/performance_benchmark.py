# tests/performance_benchmark.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import time
from typing import Dict, List
from core import ImmunePortfolioBacktester
from constant import *


class PerformanceBenchmark:
    """BIPD 시스템 성능 벤치마크"""
    
    def __init__(self):
        self.symbols = STOCK_SYMBOLS
        self.train_start = TRAIN_START_DATE
        self.train_end = TRAIN_END_DATE
        self.test_start = TEST_START_DATE
        self.test_end = TEST_END_DATE
        
    def run_comprehensive_benchmark(self, n_runs=10) -> Dict:
        """포괄적 성능 벤치마크"""
        print(f"=== BIPD 시스템 성능 벤치마크 (실행 횟수: {n_runs}) ===")
        
        results = {
            'sharpe_ratios': [],
            'total_returns': [],
            'max_drawdowns': [],
            'volatilities': [],
            'execution_times': [],
            'convergence_episodes': [],
            'final_rewards': []
        }
        
        for run in range(n_runs):
            print(f"\n실행 {run + 1}/{n_runs}")
            start_time = time.time()
            
            try:
                # 백테스터 초기화
                backtester = ImmunePortfolioBacktester(
                    self.symbols, self.train_start, self.train_end, 
                    self.test_start, self.test_end
                )
                
                # 단일 실행
                portfolio_returns, immune_system = backtester.backtest_single_run(
                    seed=42 + run,
                    return_model=True,
                    use_learning_bcells=True,
                    use_hierarchical=True,
                    use_curriculum=True,
                    logging_level="minimal"  # 빠른 실행을 위해
                )
                
                # 성과 계산
                metrics = backtester.calculate_metrics(portfolio_returns)
                execution_time = time.time() - start_time
                
                # 결과 저장
                results['sharpe_ratios'].append(metrics['Sharpe Ratio'])
                results['total_returns'].append(metrics['Total Return'])
                results['max_drawdowns'].append(metrics['Max Drawdown'])
                results['volatilities'].append(metrics['Volatility'])
                results['execution_times'].append(execution_time)
                
                # RL 특정 메트릭
                if hasattr(immune_system, 'bcells') and immune_system.bcells:
                    avg_final_reward = np.mean([
                        bcell.last_critic_loss if hasattr(bcell, 'last_critic_loss') else 0
                        for bcell in immune_system.bcells
                    ])
                    results['final_rewards'].append(avg_final_reward)
                
                print(f"  샤프 비율: {metrics['Sharpe Ratio']:.3f}")
                print(f"  총 수익률: {metrics['Total Return']:.2%}")
                print(f"  최대 낙폭: {metrics['Max Drawdown']:.2%}")
                print(f"  실행 시간: {execution_time:.1f}초")
                
            except Exception as e:
                print(f"  실행 {run + 1} 실패: {e}")
                continue
        
        return self._analyze_benchmark_results(results)
    
    def _analyze_benchmark_results(self, results: Dict) -> Dict:
        """벤치마크 결과 분석"""
        analysis = {}
        
        for metric, values in results.items():
            if values:  # 빈 리스트가 아닌 경우
                analysis[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'count': len(values)
                }
        
        return analysis
    
    def run_stability_test(self, target_sharpe=0.5, n_runs=20) -> Dict:
        """안정성 테스트 - 목표 성과 달성 확률"""
        print(f"=== 안정성 테스트 (목표 샤프 비율: {target_sharpe}) ===")
        
        success_count = 0
        sharpe_ratios = []
        
        for run in range(n_runs):
            try:
                backtester = ImmunePortfolioBacktester(
                    self.symbols, self.train_start, self.train_end,
                    self.test_start, self.test_end
                )
                
                portfolio_returns, _ = backtester.backtest_single_run(
                    seed=100 + run,
                    return_model=False,
                    use_learning_bcells=True,
                    use_hierarchical=True,
                    use_curriculum=True,
                    logging_level="minimal"
                )
                
                metrics = backtester.calculate_metrics(portfolio_returns)
                sharpe = metrics['Sharpe Ratio']
                sharpe_ratios.append(sharpe)
                
                if sharpe >= target_sharpe:
                    success_count += 1
                
                print(f"실행 {run + 1}: 샤프 비율 {sharpe:.3f}")
                
            except Exception as e:
                print(f"실행 {run + 1} 실패: {e}")
                sharpe_ratios.append(0.0)
        
        success_rate = success_count / n_runs
        
        return {
            'success_rate': success_rate,
            'target_sharpe': target_sharpe,
            'avg_sharpe': np.mean(sharpe_ratios),
            'std_sharpe': np.std(sharpe_ratios),
            'success_count': success_count,
            'total_runs': n_runs
        }
    
    def compare_with_baseline(self) -> Dict:
        """Buy & Hold 기준선과 비교"""
        print("=== Buy & Hold 기준선과 비교 ===")
        
        # BIPD 성과
        backtester = ImmunePortfolioBacktester(
            self.symbols, self.train_start, self.train_end,
            self.test_start, self.test_end
        )
        
        bipd_returns, _ = backtester.backtest_single_run(
            seed=42,
            return_model=False,
            use_learning_bcells=True,
            use_hierarchical=True,
            use_curriculum=True,
            logging_level="minimal"
        )
        bipd_metrics = backtester.calculate_metrics(bipd_returns)
        
        # Buy & Hold 기준선 (균등 가중)
        baseline_returns = backtester.calculate_baseline_performance()
        baseline_metrics = backtester.calculate_metrics(baseline_returns)
        
        comparison = {
            'bipd': bipd_metrics,
            'baseline': baseline_metrics,
            'improvement': {
                'sharpe_ratio': bipd_metrics['Sharpe Ratio'] - baseline_metrics['Sharpe Ratio'],
                'total_return': bipd_metrics['Total Return'] - baseline_metrics['Total Return'],
                'max_drawdown': bipd_metrics['Max Drawdown'] - baseline_metrics['Max Drawdown'],
            }
        }
        
        print(f"BIPD 샤프 비율: {bipd_metrics['Sharpe Ratio']:.3f}")
        print(f"기준선 샤프 비율: {baseline_metrics['Sharpe Ratio']:.3f}")
        print(f"개선도: {comparison['improvement']['sharpe_ratio']:.3f}")
        
        return comparison
    
    def test_convergence_speed(self) -> Dict:
        """학습 수렴 속도 테스트"""
        print("=== 학습 수렴 속도 테스트 ===")
        
        backtester = ImmunePortfolioBacktester(
            self.symbols, self.train_start, self.train_end,
            self.test_start, self.test_end
        )
        
        # 짧은 학습으로 수렴 테스트
        start_time = time.time()
        portfolio_returns, immune_system = backtester.backtest_single_run(
            seed=42,
            return_model=True,
            use_learning_bcells=True,
            use_hierarchical=True,
            use_curriculum=True,
            logging_level="sample"
        )
        training_time = time.time() - start_time
        
        # B-Cell 학습 상태 확인
        learning_stats = {}
        if hasattr(immune_system, 'bcells'):
            learning_stats = {
                'avg_update_counter': np.mean([
                    bcell.update_counter for bcell in immune_system.bcells
                ]),
                'avg_experience_buffer_size': np.mean([
                    len(bcell.experience_buffer) for bcell in immune_system.bcells
                ]),
                'avg_epsilon': np.mean([
                    bcell.epsilon for bcell in immune_system.bcells
                ])
            }
        
        return {
            'training_time_seconds': training_time,
            'learning_stats': learning_stats,
            'final_metrics': backtester.calculate_metrics(portfolio_returns)
        }


def main():
    """벤치마크 실행"""
    benchmark = PerformanceBenchmark()
    
    print("BIPD 시스템 성능 벤치마크 시작...")
    
    # 1. 기본 성능 벤치마크
    basic_results = benchmark.run_comprehensive_benchmark(n_runs=5)
    
    print("\n=== 종합 성능 분석 ===")
    for metric, stats in basic_results.items():
        print(f"{metric}:")
        print(f"  평균: {stats['mean']:.4f}")
        print(f"  표준편차: {stats['std']:.4f}")
        print(f"  범위: [{stats['min']:.4f}, {stats['max']:.4f}]")
    
    # 2. 안정성 테스트
    stability_results = benchmark.run_stability_test(target_sharpe=0.3, n_runs=10)
    
    print(f"\n=== 안정성 테스트 결과 ===")
    print(f"성공률: {stability_results['success_rate']:.1%}")
    print(f"평균 샤프 비율: {stability_results['avg_sharpe']:.3f}")
    
    # 3. 기준선 비교
    comparison_results = benchmark.compare_with_baseline()
    
    print(f"\n=== 기준선 대비 개선도 ===")
    print(f"샤프 비율 개선: {comparison_results['improvement']['sharpe_ratio']:.3f}")
    print(f"수익률 개선: {comparison_results['improvement']['total_return']:.2%}")
    
    # 4. 수렴 속도 테스트
    convergence_results = benchmark.test_convergence_speed()
    
    print(f"\n=== 학습 수렴 속도 ===")
    print(f"훈련 시간: {convergence_results['training_time_seconds']:.1f}초")
    if convergence_results['learning_stats']:
        print(f"평균 업데이트 횟수: {convergence_results['learning_stats']['avg_update_counter']:.0f}")
        print(f"평균 경험 버퍼 크기: {convergence_results['learning_stats']['avg_experience_buffer_size']:.0f}")
    
    print("\n🎯 벤치마크 완료!")


if __name__ == "__main__":
    main()