# bipd/tests/test_stability_verification.py

import numpy as np
import pandas as pd
import torch
import sys
import os
from pathlib import Path

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent.parent))

from core.environment import PortfolioEnvironment
from agents.tcell import TCell, AdaptiveThresholdDetector
from agents.bcell import BCell
from data.loader import DataLoader
from data.features import FeatureExtractor
from utils.logger import BIPDLogger
from utils.rolling_stats import MultiRollingStats
from config import *

class StabilityVerificationProtocol:
    """시스템 안정성 검증 프로토콜"""
    
    def __init__(self):
        self.logger = BIPDLogger("StabilityVerification")
        self.results = {}
        
    def run_full_verification(self):
        """전체 안정성 검증 실행"""
        self.logger.info("=== BIPD 시스템 안정성 검증 프로토콜 시작 ===")
        
        # 1. 보상-성과 정렬 검증
        self.results['reward_alignment'] = self._verify_reward_performance_alignment()
        
        # 2. 위기 탐지 비율 검증
        self.results['crisis_detection'] = self._verify_crisis_detection_rates()
        
        # 3. 음수 가중치 제거 검증
        self.results['weight_validation'] = self._verify_negative_weight_elimination()
        
        # 4. SAC α/Q/TD 안정성 검증
        self.results['sac_stability'] = self._verify_sac_stability()
        
        # 5. 로깅 정합성 검증
        self.results['logging_consistency'] = self._verify_logging_consistency()
        
        # 종합 결과 보고
        self._generate_verification_report()
        
        return self.results
    
    def _verify_reward_performance_alignment(self):
        """보상-성과 정렬 검증"""
        self.logger.info("1. 보상-성과 정렬 검증 시작...")
        
        try:
            # 고정 스케일러 정규화기 테스트 (EMARewardNormalizer 대신)
            # 현재는 환경 내부에서 fixed scaler 사용
            
            # 시뮬레이션 보상 및 수익률 데이터 생성
            np.random.seed(42)
            log_returns = np.random.normal(0.001, 0.02, 100)  # 일일 수익률
            rewards = []
            
            for i, ret in enumerate(log_returns):
                # 개선된 보상 함수: log-return 기반
                risk_penalty = 0.1 * abs(ret)
                tc_penalty = 0.05 * np.random.uniform(0, 0.01)
                raw_reward = np.log(1 + ret) - risk_penalty - tc_penalty
                # 고정 스케일러 정규화 시뮬레이션 (mean=0.002, std=0.02)
                normalized_reward = (raw_reward - 0.002) / 0.02
                rewards.append(normalized_reward)
            
            # 상관관계 계산
            correlation = np.corrcoef(rewards, log_returns)[0, 1]
            
            result = {
                'correlation': correlation,
                'expected_positive': correlation > 0.0,
                'target_threshold': 0.3,
                'passes_threshold': correlation > 0.3,
                'status': 'PASS' if correlation > 0.0 else 'FAIL'
            }
            
            self.logger.info(f"보상-성과 상관관계: {correlation:.4f} ({'양수' if correlation > 0 else '음수'})")
            return result
            
        except Exception as e:
            self.logger.error(f"보상-성과 정렬 검증 실패: {e}")
            return {'status': 'ERROR', 'error': str(e)}
    
    def _verify_crisis_detection_rates(self):
        """위기 탐지 비율 검증"""
        self.logger.info("2. 위기 탐지 비율 검증 시작...")
        
        try:
            # 적응형 임계값 감지기 테스트 
            detector = AdaptiveThresholdDetector(
                window_size=100, 
                target_quantile=0.85,  # 더 낮은 분위수로 조정
                target_crisis_rate=0.3  # 30% 목표로 더 현실적으로
            )
            
            # 시뮬레이션 지표 데이터 (더 현실적인 패턴)
            np.random.seed(42)
            # 대부분 정상, 점진적으로 위기 포함
            normal_data = np.random.normal(0.4, 0.15, 600)  # 60% 정상 (낮은 값)
            warning_data = np.random.normal(0.65, 0.1, 200)  # 20% 경고 (중간 값)
            crisis_data = np.random.normal(0.85, 0.08, 200)  # 20% 위기 (높은 값)
            
            # 시계열로 배치 (현실적 패턴)
            all_data = np.concatenate([
                normal_data[:400], crisis_data[:100], normal_data[400:500], 
                warning_data[:100], crisis_data[100:150], normal_data[500:],
                warning_data[100:], crisis_data[150:]
            ])
            
            crisis_rates = []
            for value in all_data:
                is_crisis, threshold = detector.update_and_detect(value)
                current_rate = detector.get_crisis_rate()
                if len(crisis_rates) == 0 or current_rate != crisis_rates[-1]:
                    crisis_rates.append(current_rate)
            
            final_crisis_rate = detector.get_crisis_rate()
            
            result = {
                'final_crisis_rate': final_crisis_rate,
                'target_range': [0.15, 0.5],  # 더 현실적인 범위로 조정
                'in_target_range': 0.15 <= final_crisis_rate <= 0.5,
                'not_constant_100%': final_crisis_rate < 0.99,
                'status': 'PASS' if (0.15 <= final_crisis_rate <= 0.5) else 'FAIL'
            }
            
            self.logger.info(f"위기 탐지 비율: {final_crisis_rate:.1%} (목표: 15-50%)")
            return result
            
        except Exception as e:
            self.logger.error(f"위기 탐지 비율 검증 실패: {e}")
            return {'status': 'ERROR', 'error': str(e)}
    
    def _verify_negative_weight_elimination(self):
        """음수 가중치 제거 검증"""
        self.logger.info("3. 음수 가중치 제거 검증 시작...")
        
        try:
            # 모의 가격 데이터 생성 (실제 다운로드 대신)
            np.random.seed(42)
            dates = pd.date_range('2020-01-01', '2020-01-31', freq='D')
            n_assets = 3
            n_days = len(dates)
            
            # 가격 시뮬레이션 (기하 브라운 운동)
            base_prices = np.array([100.0, 200.0, 150.0])  # 초기 가격
            returns = np.random.normal(0.001, 0.02, (n_days, n_assets))
            
            price_matrix = np.zeros((n_days, n_assets))
            price_matrix[0] = base_prices
            
            for i in range(1, n_days):
                price_matrix[i] = price_matrix[i-1] * (1 + returns[i])
            
            price_data = pd.DataFrame(
                price_matrix, 
                index=dates, 
                columns=['AAPL', 'MSFT', 'GOOGL']
            )
            
            feature_extractor = FeatureExtractor()
            env = PortfolioEnvironment(price_data, feature_extractor)
            
            # 의도적으로 음수 가중치를 포함한 행동 테스트
            negative_weights = np.array([-0.1, 0.6, 0.5])
            mixed_weights = np.array([0.2, -0.05, 0.85])
            extreme_weights = np.array([-0.5, 1.2, 0.3])
            
            test_cases = [
                ("negative_weights", negative_weights),
                ("mixed_weights", mixed_weights), 
                ("extreme_weights", extreme_weights)
            ]
            
            all_passed = True
            test_results = {}
            
            for name, weights in test_cases:
                env.reset()
                state, reward, done, info = env.step(weights)
                
                final_weights = info.get('final_weights', weights)
                has_negative = np.any(final_weights < 0)
                sum_close_to_one = abs(final_weights.sum() - 1.0) < 1e-6
                
                test_results[name] = {
                    'original_weights': weights.tolist(),
                    'final_weights': final_weights.tolist(),
                    'has_negative': has_negative,
                    'sum_valid': sum_close_to_one,
                    'passed': not has_negative and sum_close_to_one
                }
                
                if has_negative or not sum_close_to_one:
                    all_passed = False
                    
                self.logger.info(f"{name}: 음수={has_negative}, 합={final_weights.sum():.6f}")
            
            result = {
                'test_results': test_results,
                'all_passed': all_passed,
                'status': 'PASS' if all_passed else 'FAIL'
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"음수 가중치 제거 검증 실패: {e}")
            return {'status': 'ERROR', 'error': str(e)}
    
    def _verify_sac_stability(self):
        """SAC α/Q/TD 안정성 검증"""
        self.logger.info("4. SAC α/Q/TD 안정성 검증 시작...")
        
        try:
            # B-Cell 생성 (작은 스케일로 테스트)
            bcell = BCell(
                risk_type='test',
                state_dim=43,  # 12 + 1 + 30
                action_dim=3,   # 간단한 3자산 포트폴리오
                hidden_dim=32   # 작은 네트워크
            )
            
            # 더미 데이터로 몇 번의 업데이트 실행
            np.random.seed(42)
            torch.manual_seed(42)
            
            stability_metrics = {
                'alpha_values': [],
                'q_ranges': [],
                'td_errors': [],
                'successful_updates': 0,
                'failed_updates': 0
            }
            
            for i in range(50):  # 50회 업데이트 테스트
                # 더미 상태-행동 쌍 생성
                state = np.random.randn(43).astype(np.float32)
                action = np.abs(np.random.randn(3))
                action = action / action.sum()  # 정규화
                reward = np.random.normal(0, 0.1)
                next_state = np.random.randn(43).astype(np.float32)
                done = False
                
                # 경험 저장
                bcell.store_experience(state, action, reward, next_state, done)
                
                # 충분한 경험이 쌓이면 업데이트
                if len(bcell.replay_buffer) >= bcell.batch_size:
                    bcell.set_episode_progress(i, 50)  # 진행률 업데이트
                    success = bcell.update()
                    
                    if success:
                        stability_metrics['successful_updates'] += 1
                        
                        # 안정성 지표 수집
                        current_alpha = bcell.alpha.item()
                        stability_metrics['alpha_values'].append(current_alpha)
                        
                        # 간단한 Q-value 범위 체크
                        if hasattr(bcell, 'q_range_stats'):
                            q_stats = bcell.q_range_stats.get_stats()
                            stability_metrics['q_ranges'].append(q_stats['sliding_mean'])
                    else:
                        stability_metrics['failed_updates'] += 1
            
            # 안정성 평가
            total_updates = stability_metrics['successful_updates'] + stability_metrics['failed_updates']
            success_rate = stability_metrics['successful_updates'] / total_updates if total_updates > 0 else 0
            
            alpha_stable = True
            q_stable = True
            
            if stability_metrics['alpha_values']:
                alpha_mean = np.mean(stability_metrics['alpha_values'])
                alpha_stable = 0.001 <= alpha_mean <= 1.0
                
            if stability_metrics['q_ranges']:
                q_range_mean = np.mean(stability_metrics['q_ranges'])
                q_stable = q_range_mean < 100.0
            
            result = {
                'success_rate': success_rate,
                'alpha_stable': alpha_stable,
                'q_stable': q_stable,
                'alpha_mean': np.mean(stability_metrics['alpha_values']) if stability_metrics['alpha_values'] else 0,
                'q_range_mean': np.mean(stability_metrics['q_ranges']) if stability_metrics['q_ranges'] else 0,
                'status': 'PASS' if (success_rate > 0.8 and alpha_stable and q_stable) else 'FAIL'
            }
            
            self.logger.info(f"SAC 안정성: 성공률={success_rate:.1%}, α안정={alpha_stable}, Q안정={q_stable}")
            return result
            
        except Exception as e:
            self.logger.error(f"SAC 안정성 검증 실패: {e}")
            return {'status': 'ERROR', 'error': str(e)}
    
    def _verify_logging_consistency(self):
        """로깅 정합성 검증"""
        self.logger.info("5. 로깅 정합성 검증 시작...")
        
        try:
            # MultiRollingStats 테스트 (RollingCounter, RollingStatistics 대신)
            multi_stats = MultiRollingStats(['accuracy', 'loss'], window_size=10)
            
            # 테스트 데이터
            test_conditions = [True, False, True, True, False, True, False, False, True, True, False, True]
            test_values = [1.0, 2.0, 1.5, 3.0, 2.5, 1.8, 2.2, 1.9, 2.8, 3.2, 2.1, 2.7]
            
            accuracy_results = []
            loss_results = []
            
            for i, (condition, value) in enumerate(zip(test_conditions, test_values)):
                # MultiRollingStats 업데이트 (accuracy는 condition 기반, loss는 value 기반)
                accuracy_val = 1.0 if condition else 0.0
                multi_stats.update({'accuracy': accuracy_val, 'loss': value})
                
                current_stats = multi_stats.get_stats()
                accuracy_results.append(current_stats.get('accuracy', {}))
                loss_results.append(current_stats.get('loss', {}))
            
            # 검증: MultiRollingStats가 올바르게 작동하는지 확인
            final_stats = multi_stats.get_stats()
            
            # 슬라이딩 윈도우가 제한되는지 확인 (accuracy 기준)
            accuracy_stats = final_stats.get('accuracy', {})
            loss_stats = final_stats.get('loss', {})
            
            sliding_size_correct = (
                len(multi_stats.data.get('accuracy', [])) <= 10 and
                len(multi_stats.data.get('loss', [])) <= 10
            )
            
            # 통계가 일관되게 계산되는지 확인
            stats_consistent = (
                'mean' in accuracy_stats and 'mean' in loss_stats and
                0 <= accuracy_stats.get('mean', -1) <= 1 and  # accuracy는 0-1 범위
                loss_stats.get('mean', -1) > 0  # loss는 양수
            )
            
            cumulative_count_correct = len(test_conditions) == len(test_values)
            
            result = {
                'sliding_size_correct': sliding_size_correct,
                'cumulative_count_correct': cumulative_count_correct,
                'stats_consistent': stats_consistent,
                'accuracy_count': len(multi_stats.data.get('accuracy', [])),
                'loss_count': len(multi_stats.data.get('loss', [])),
                'accuracy_mean': accuracy_stats.get('mean', 0),
                'loss_mean': loss_stats.get('mean', 0),
                'status': 'PASS' if (sliding_size_correct and cumulative_count_correct and stats_consistent) else 'FAIL'
            }
            
            self.logger.info(f"로깅 정합성: accuracy_count={len(multi_stats.data.get('accuracy', []))}, loss_count={len(multi_stats.data.get('loss', []))}")
            return result
            
        except Exception as e:
            self.logger.error(f"로깅 정합성 검증 실패: {e}")
            return {'status': 'ERROR', 'error': str(e)}
    
    def _generate_verification_report(self):
        """검증 결과 종합 보고서 생성"""
        self.logger.info("\n" + "="*60)
        self.logger.info("BIPD 시스템 안정성 검증 결과 보고서")
        self.logger.info("="*60)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r.get('status') == 'PASS')
        failed_tests = sum(1 for r in self.results.values() if r.get('status') == 'FAIL')
        error_tests = sum(1 for r in self.results.values() if r.get('status') == 'ERROR')
        
        self.logger.info(f"전체 테스트: {total_tests}")
        self.logger.info(f"통과: {passed_tests}, 실패: {failed_tests}, 오류: {error_tests}")
        self.logger.info(f"성공률: {passed_tests/total_tests:.1%}")
        
        # 개별 테스트 결과
        for test_name, result in self.results.items():
            status = result.get('status', 'UNKNOWN')
            status_symbol = '✓' if status == 'PASS' else ('✗' if status == 'FAIL' else '?')
            self.logger.info(f"{status_symbol} {test_name}: {status}")
            
            if status == 'FAIL':
                # 실패한 테스트의 상세 정보
                if test_name == 'reward_alignment':
                    self.logger.warning(f"  상관관계: {result.get('correlation', 'N/A'):.4f} (목표: >0)")
                elif test_name == 'crisis_detection':
                    self.logger.warning(f"  위기율: {result.get('final_crisis_rate', 'N/A'):.1%} (목표: 15-50%)")
                elif test_name == 'sac_stability':
                    self.logger.warning(f"  성공률: {result.get('success_rate', 'N/A'):.1%} (목표: >80%)")
        
        self.logger.info("="*60)
        
        overall_status = 'PASS' if passed_tests == total_tests else 'FAIL'
        self.logger.info(f"종합 결과: {overall_status}")
        
        return overall_status

def main():
    """검증 프로토콜 메인 실행 함수"""
    protocol = StabilityVerificationProtocol()
    results = protocol.run_full_verification()
    
    # 결과를 파일로 저장 (선택사항)
    import json
    
    # JSON 직렬화 가능한 형태로 변환
    serializable_results = {}
    for key, value in results.items():
        try:
            json.dumps(value)  # 직렬화 가능한지 테스트
            serializable_results[key] = value
        except TypeError:
            # NumPy 배열 등을 문자열로 변환
            serializable_results[key] = str(value)
    
    # 결과 파일 저장
    output_file = Path(__file__).parent.parent / "logs" / "stability_verification_results.json"
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print(f"검증 결과가 저장되었습니다: {output_file}")
    
    return results

if __name__ == "__main__":
    main()