# tests/test_statistical_significance.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple
from core import ImmunePortfolioBacktester
from constant import *


class StatisticalSignificanceTest:
    """통계적 유의성 검증 테스트"""
    
    def __init__(self):
        self.symbols = STOCK_SYMBOLS[:5]  # 테스트용으로 5개 종목만
        self.alpha = 0.05  # 유의수준 5%
        
    def test_vs_buy_and_hold(self, n_runs=10) -> Dict:
        """Buy & Hold 전략 대비 통계적 유의성 검증"""
        print("=== BIPD vs Buy & Hold 통계적 유의성 테스트 ===")
        
        bipd_sharpes = []
        buyhold_sharpes = []
        bipd_returns = []
        buyhold_returns = []
        
        for run in range(n_runs):
            print(f"  실행 {run + 1}/{n_runs}")
            
            # 백테스터 초기화
            backtester = ImmunePortfolioBacktester(
                self.symbols, TRAIN_START_DATE, TRAIN_END_DATE,
                TEST_START_DATE, TEST_END_DATE
            )
            
            try:
                # BIPD 성과
                bipd_portfolio_returns, _ = backtester.backtest_single_run(
                    seed=42 + run,
                    return_model=False,
                    use_learning_bcells=True,
                    use_hierarchical=True,
                    use_curriculum=False,  # 빠른 테스트
                    logging_level="minimal"
                )
                
                bipd_metrics = backtester.calculate_metrics(bipd_portfolio_returns)
                bipd_sharpes.append(bipd_metrics['Sharpe Ratio'])
                bipd_returns.append(bipd_metrics['Total Return'])
                
                # Buy & Hold 기준선
                baseline_returns = backtester.calculate_baseline_performance()
                baseline_metrics = backtester.calculate_metrics(baseline_returns)
                buyhold_sharpes.append(baseline_metrics['Sharpe Ratio'])
                buyhold_returns.append(baseline_metrics['Total Return'])
                
            except Exception as e:
                print(f"    실행 {run + 1} 실패: {e}")
                continue
        
        return self._analyze_statistical_significance(
            bipd_sharpes, buyhold_sharpes, bipd_returns, buyhold_returns
        )
    
    def _analyze_statistical_significance(
        self, bipd_sharpes: List[float], buyhold_sharpes: List[float],
        bipd_returns: List[float], buyhold_returns: List[float]
    ) -> Dict:
        """통계적 유의성 분석"""
        
        results = {
            'sample_size': min(len(bipd_sharpes), len(buyhold_sharpes)),
            'alpha': self.alpha
        }
        
        if results['sample_size'] < 3:
            results['error'] = "충분한 샘플 수 없음"
            return results
        
        # 샤프 비율 비교
        sharpe_t_stat, sharpe_p_value = stats.ttest_rel(bipd_sharpes, buyhold_sharpes)
        
        results['sharpe_ratio'] = {
            'bipd_mean': np.mean(bipd_sharpes),
            'bipd_std': np.std(bipd_sharpes),
            'buyhold_mean': np.mean(buyhold_sharpes),
            'buyhold_std': np.std(buyhold_sharpes),
            'improvement': np.mean(bipd_sharpes) - np.mean(buyhold_sharpes),
            't_statistic': sharpe_t_stat,
            'p_value': sharpe_p_value,
            'is_significant': sharpe_p_value < self.alpha,
            'effect_size': self._calculate_cohens_d(bipd_sharpes, buyhold_sharpes)
        }
        
        # 총 수익률 비교
        return_t_stat, return_p_value = stats.ttest_rel(bipd_returns, buyhold_returns)
        
        results['total_return'] = {
            'bipd_mean': np.mean(bipd_returns),
            'bipd_std': np.std(bipd_returns),
            'buyhold_mean': np.mean(buyhold_returns),
            'buyhold_std': np.std(buyhold_returns),
            'improvement': np.mean(bipd_returns) - np.mean(buyhold_returns),
            't_statistic': return_t_stat,
            'p_value': return_p_value,
            'is_significant': return_p_value < self.alpha,
            'effect_size': self._calculate_cohens_d(bipd_returns, buyhold_returns)
        }
        
        return results
    
    def _calculate_cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """Cohen's d (효과 크기) 계산"""
        n1, n2 = len(group1), len(group2)
        s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        
        # 합동 표준편차
        pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        
        # Cohen's d
        return (np.mean(group1) - np.mean(group2)) / pooled_std
    
    def test_system_stability(self, n_runs=15) -> Dict:
        """시스템 안정성 테스트"""
        print("=== 시스템 안정성 테스트 ===")
        
        sharpe_ratios = []
        success_count = 0
        failure_count = 0
        
        target_sharpe = 0.5  # 목표 샤프 비율
        
        for run in range(n_runs):
            print(f"  안정성 테스트 {run + 1}/{n_runs}")
            
            backtester = ImmunePortfolioBacktester(
                self.symbols, TRAIN_START_DATE, TRAIN_END_DATE,
                TEST_START_DATE, TEST_END_DATE
            )
            
            try:
                portfolio_returns, _ = backtester.backtest_single_run(
                    seed=100 + run,
                    return_model=False,
                    use_learning_bcells=True,
                    use_hierarchical=True,
                    use_curriculum=False,
                    logging_level="minimal"
                )
                
                metrics = backtester.calculate_metrics(portfolio_returns)
                sharpe = metrics['Sharpe Ratio']
                sharpe_ratios.append(sharpe)
                
                if sharpe >= target_sharpe:
                    success_count += 1
                else:
                    failure_count += 1
                    
            except Exception as e:
                print(f"    실행 {run + 1} 실패: {e}")
                failure_count += 1
                sharpe_ratios.append(0.0)  # 실패한 경우 0으로 처리
        
        # 안정성 메트릭 계산
        success_rate = success_count / n_runs
        sharpe_mean = np.mean(sharpe_ratios)
        sharpe_std = np.std(sharpe_ratios)
        coefficient_of_variation = sharpe_std / sharpe_mean if sharpe_mean != 0 else float('inf')
        
        return {
            'total_runs': n_runs,
            'success_count': success_count,
            'failure_count': failure_count,
            'success_rate': success_rate,
            'target_sharpe': target_sharpe,
            'sharpe_statistics': {
                'mean': sharpe_mean,
                'std': sharpe_std,
                'min': np.min(sharpe_ratios),
                'max': np.max(sharpe_ratios),
                'median': np.median(sharpe_ratios),
                'coefficient_of_variation': coefficient_of_variation
            },
            'is_stable': success_rate >= 0.7 and coefficient_of_variation <= 0.5
        }
    
    def run_comprehensive_validation(self) -> Dict:
        """종합 검증 실행"""
        print("=== BIPD 시스템 종합 검증 ===\n")
        
        # 1. 통계적 유의성 검증
        significance_results = self.test_vs_buy_and_hold(n_runs=8)
        
        # 2. 시스템 안정성 검증
        stability_results = self.test_system_stability(n_runs=10)
        
        # 3. 종합 평가
        overall_score = self._calculate_overall_score(significance_results, stability_results)
        
        return {
            'significance_test': significance_results,
            'stability_test': stability_results,
            'overall_assessment': overall_score
        }
    
    def _calculate_overall_score(self, significance: Dict, stability: Dict) -> Dict:
        """종합 평가 점수 계산"""
        score = 0
        max_score = 100
        
        # 통계적 유의성 점수 (50점 만점)
        if not significance.get('error'):
            if significance['sharpe_ratio']['is_significant']:
                score += 30
            if significance['total_return']['is_significant']:
                score += 20
        
        # 시스템 안정성 점수 (50점 만점)
        if stability['is_stable']:
            score += 30
        if stability['success_rate'] >= 0.6:
            score += 20
        
        grade = 'A' if score >= 80 else 'B' if score >= 60 else 'C' if score >= 40 else 'D'
        
        return {
            'score': score,
            'max_score': max_score,
            'percentage': score / max_score,
            'grade': grade,
            'passed': score >= 60
        }


def main():
    """통계적 유의성 검증 실행"""
    test_suite = StatisticalSignificanceTest()
    
    try:
        results = test_suite.run_comprehensive_validation()
        
        # 결과 출력
        print("\n" + "="*60)
        print("📊 종합 검증 결과")
        print("="*60)
        
        # 통계적 유의성 결과
        sig = results['significance_test']
        if not sig.get('error'):
            print(f"\n🔬 통계적 유의성 (샘플 수: {sig['sample_size']})")
            
            sharpe = sig['sharpe_ratio']
            print(f"  샤프 비율:")
            print(f"    BIPD: {sharpe['bipd_mean']:.3f} ± {sharpe['bipd_std']:.3f}")
            print(f"    Buy&Hold: {sharpe['buyhold_mean']:.3f} ± {sharpe['buyhold_std']:.3f}")
            print(f"    개선도: {sharpe['improvement']:.3f}")
            print(f"    p-value: {sharpe['p_value']:.4f} ({'✅ 유의함' if sharpe['is_significant'] else '❌ 유의하지 않음'})")
            print(f"    효과 크기: {sharpe['effect_size']:.3f}")
            
            ret = sig['total_return']
            print(f"  총 수익률:")
            print(f"    BIPD: {ret['bipd_mean']:.2%} ± {ret['bipd_std']:.2%}")
            print(f"    Buy&Hold: {ret['buyhold_mean']:.2%} ± {ret['buyhold_std']:.2%}")
            print(f"    개선도: {ret['improvement']:.2%}")
            print(f"    p-value: {ret['p_value']:.4f} ({'✅ 유의함' if ret['is_significant'] else '❌ 유의하지 않음'})")
            
        # 안정성 결과
        stab = results['stability_test']
        print(f"\n⚖️ 시스템 안정성")
        print(f"  성공률: {stab['success_rate']:.1%} ({stab['success_count']}/{stab['total_runs']})")
        print(f"  평균 샤프 비율: {stab['sharpe_statistics']['mean']:.3f}")
        print(f"  변동 계수: {stab['sharpe_statistics']['coefficient_of_variation']:.3f}")
        print(f"  안정성: {'✅ 안정함' if stab['is_stable'] else '❌ 불안정함'}")
        
        # 종합 평가
        overall = results['overall_assessment']
        print(f"\n🏆 종합 평가")
        print(f"  점수: {overall['score']}/{overall['max_score']} ({overall['percentage']:.1%})")
        print(f"  등급: {overall['grade']}")
        print(f"  합격: {'✅ PASS' if overall['passed'] else '❌ FAIL'}")
        
        return overall['passed']
        
    except Exception as e:
        print(f"\n❌ 검증 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 BIPD 시스템이 통계적 유의성 검증을 통과했습니다!")
    else:
        print("\n⚠️ BIPD 시스템이 일부 검증 기준을 충족하지 못했습니다.")
        print("   더 많은 데이터나 파라미터 조정이 필요할 수 있습니다.")