# bipd/main.py

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from datetime import datetime

from config import *
from data import DataLoader
from core import BIPDTrainer
from utils import BIPDLogger

def main():
    """BIPD 시스템 메인 실행 함수"""
    
    # 시드 설정
    set_seed(GLOBAL_SEED)
    
    # 디렉토리 생성
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    # 로거 초기화
    logger = BIPDLogger("Main")
    
    print("=" * 80)
    print("BIPD (Behavioral Immune Portfolio Defense) 시스템 시작")
    print("=" * 80)
    print(f"설정 요약:")
    print(f"  - 종목: {len(SYMBOLS)}개 (Dow Jones 30)")
    print(f"  - 훈련기간: {TRAIN_START} ~ {TRAIN_END}")
    print(f"  - 테스트기간: {TEST_START} ~ {TEST_END}")
    print(f"  - 에피소드: {N_EPISODES}개")
    print(f"  - 초기자본: {INITIAL_CAPITAL:,.0f}원")
    print()
    
    # 상세 로그는 파일에만
    logger.debug("=" * 80)
    logger.debug("BIPD (Behavioral Immune Portfolio Defense) 시스템 시작")
    logger.debug("=" * 80)
    logger.debug(f"설정 요약:")
    logger.debug(f"  - 종목: {SYMBOLS}")
    logger.debug(f"  - 훈련기간: {TRAIN_START} ~ {TRAIN_END}")
    logger.debug(f"  - 테스트기간: {TEST_START} ~ {TEST_END}")
    logger.debug(f"  - 에피소드: {N_EPISODES}개")
    logger.debug(f"  - 초기자본: {INITIAL_CAPITAL:,.0f}원")
    
    try:
        # 1. 데이터 로드
        print("[1단계] 시장 데이터를 로드합니다...")
        data_loader = DataLoader(cache_dir=os.path.join(DATA_DIR, "cache"))
        
        market_data = data_loader.get_market_data(
            symbols=SYMBOLS,
            train_start=TRAIN_START,
            train_end=TRAIN_END,
            test_start=TEST_START,
            test_end=TEST_END
        )
        
        train_data = market_data['train_data']
        test_data = market_data['test_data']
        
        print(f"데이터 로드 완료:")
        print(f"  - 훈련 데이터: {len(train_data)} 거래일")
        print(f"  - 테스트 데이터: {len(test_data)} 거래일")
        print(f"  - 종목 수: {len(SYMBOLS)}개")
        print()
        
        # 상세 정보는 로그에만
        logger.debug(f"데이터 로드 완료:")
        logger.debug(f"  - 훈련 데이터: {len(train_data)} 거래일")
        logger.debug(f"  - 테스트 데이터: {len(test_data)} 거래일")
        logger.debug(f"  - 종목 수: {len(SYMBOLS)}개")
        
        # 2. 훈련자 초기화
        print("[2단계] BIPD 훈련자를 초기화합니다...")
        trainer = BIPDTrainer(
            train_data=train_data,
            test_data=test_data,
            save_dir=MODELS_DIR
        )
        logger.debug("BIPD 훈련자 초기화 완료")
        
        # 3. 시스템 훈련
        print("[3단계] BIPD 시스템 훈련을 시작합니다...")
        print()
        training_results = trainer.train(
            n_episodes=N_EPISODES,
            save_interval=SAVE_INTERVAL
        )
        
        print("\n훈련 결과 요약:")
        print(f"  - 최종 평균 보상: {training_results['final_avg_reward']:.4f}")
        print(f"  - 최고 포트폴리오 가치: {training_results['best_portfolio_value']:,.0f}원")
        print(f"  - 최고 샤프 비율: {training_results['best_sharpe_ratio']:.3f}")
        print(f"  - 훈련 안정성 (표준편차): {training_results['training_stability']:.4f}")
        print()
        
        # 상세 로그
        logger.debug("훈련 결과 요약:")
        logger.debug(f"  - 최종 평균 보상: {training_results['final_avg_reward']:.4f}")
        logger.debug(f"  - 최고 포트폴리오 가치: {training_results['best_portfolio_value']:,.0f}원")
        logger.debug(f"  - 최고 샤프 비율: {training_results['best_sharpe_ratio']:.3f}")
        logger.debug(f"  - 훈련 안정성 (표준편차): {training_results['training_stability']:.4f}")
        
        # 4. 시스템 평가
        print("[4단계] 테스트 데이터로 시스템을 평가합니다...")
        evaluation_results = trainer.evaluate(n_episodes=10)
        
        print("\n평가 결과 요약:")
        print(f"  - 평균 최종 가치: {evaluation_results['avg_final_value']:,.0f}원")
        print(f"  - 평균 샤프 비율: {evaluation_results['avg_sharpe_ratio']:.3f}")
        print(f"  - 평균 최대 낙폭: {evaluation_results['avg_max_drawdown']:.2%}")
        print(f"  - 성공률: {evaluation_results['success_rate']:.1%}")
        print()
        
        # 5. 벤치마크 비교
        print("[5단계] 벤치마크와 성과를 비교합니다...")
        benchmark_results = trainer.benchmark_comparison('equal_weight')
        
        print("\n벤치마크 비교 결과:")
        print(f"  - 수익률 개선: {benchmark_results['outperformance']['value_improvement']:.2%}")
        print(f"  - 샤프비율 개선: {benchmark_results['outperformance']['sharpe_improvement']:.3f}")
        print(f"  - 최대낙폭 개선: {benchmark_results['outperformance']['drawdown_improvement']:.2%}")
        print()
        
        # 6. 시각화 생성
        print("[6단계] 훈련 결과를 시각화합니다...")
        
        # 시각화 저장 경로
        viz_dir = os.path.join(MODELS_DIR, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(viz_dir, f"training_results_{timestamp}.png")
        
        trainer.plot_training_results(save_path=plot_path)
        
        # 7. XAI 설명 생성 (샘플)
        print("[7단계] 시스템 설명을 생성합니다...")
        
        # 테스트 환경에서 샘플 상태 가져오기
        test_state = trainer.test_env.reset()
        explanation = trainer.immune_system.get_system_explanation(test_state)
        
        print("\nBIPD 시스템 의사결정 설명:")
        print(f"  - T-Cell 위기 감지: {explanation['crisis_detection'].get('crisis_level', 0):.3f}")
        print(f"  - 선택된 전략: {explanation['strategy_selection']['selected_strategy']}")
        print(f"  - 메모리 시스템: {explanation['memory_system']['memory_count']}개 경험 보유")
        
        specialization_scores = explanation['strategy_selection']['all_specialization_scores']
        print("  - B-Cell 전문성 점수:")
        for name, score in specialization_scores.items():
            print(f"    * {name}: {score:.3f}")
        print()
        
        # 8. 최종 요약
        print("=" * 80)
        print("BIPD 시스템 실행 완료")
        print("=" * 80)
        
        success_rate = evaluation_results['success_rate']
        avg_improvement = benchmark_results['outperformance']['value_improvement']
        
        if success_rate > 0.7 and avg_improvement > 0.05:
            print("✅ 성공: BIPD 시스템이 벤치마크를 능가하는 성과를 달성했습니다!")
        elif success_rate > 0.5:
            print("⚠️ 부분 성공: 일부 개선이 있으나 추가 최적화가 필요합니다.")
        else:
            print("❌ 개선 필요: 시스템 성능이 기대에 못 미칩니다. 하이퍼파라미터 조정이 필요합니다.")
        
        print(f"\n저장된 파일:")
        print(f"  - 모델: {MODELS_DIR}/")
        print(f"  - 로그: {LOGS_DIR}/")
        print(f"  - 시각화: {plot_path}")
        
        # 상세 로그는 파일에만
        logger.debug("최종 요약 완료")
        logger.debug(f"성공률: {success_rate:.1%}, 개선도: {avg_improvement:.2%}")
        
        return {
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'benchmark_results': benchmark_results,
            'explanation': explanation,
            'success': success_rate > 0.5 and avg_improvement > 0.0
        }
        
    except KeyboardInterrupt:
        print("\n사용자에 의해 실행이 중단되었습니다.")
        logger.warning("사용자에 의해 실행이 중단되었습니다.")
        return {'interrupted': True}
        
    except Exception as e:
        print(f"\n❌ 실행 중 오류가 발생했습니다: {e}")
        logger.error(f"실행 중 오류가 발생했습니다: {e}")
        import traceback
        logger.error(f"상세 오류:\n{traceback.format_exc()}")
        return {'error': str(e)}

if __name__ == "__main__":
    results = main()
    
    if results.get('success', False):
        print("\n🎉 BIPD 시스템이 성공적으로 실행되었습니다!")
    elif results.get('interrupted', False):
        print("\n⏹️ 실행이 중단되었습니다.")
    elif 'error' in results:
        print(f"\n❌ 실행 실패: {results['error']}")
    else:
        print("\n✅ BIPD 시스템 실행이 완료되었습니다.")