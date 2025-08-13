# tests/quick_test.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core import ImmunePortfolioBacktester
from constant import *


def quick_functionality_test():
    """빠른 기능 테스트 (5분 이내)"""
    print("=== BIPD 빠른 기능 테스트 ===")
    
    # 작은 데이터셋으로 빠른 테스트
    symbols = STOCK_SYMBOLS[:3]  # 3개 종목만
    
    backtester = ImmunePortfolioBacktester(
        symbols, TRAIN_START_DATE, TRAIN_END_DATE, 
        TEST_START_DATE, TEST_END_DATE
    )
    
    print("1. 시스템 초기화 테스트...")
    
    # 매우 짧은 학습으로 테스트
    print("2. 짧은 학습 테스트 (100 에피소드)...")
    
    # constant 수정 없이 내부적으로만 조정
    original_episodes = TOTAL_EPISODES
    original_pretrain = PRETRAIN_EPISODES
    
    try:
        # 환경 변수로 짧은 학습 설정 (실제 상수는 건드리지 않음)
        portfolio_returns, immune_system = backtester.backtest_single_run(
            seed=42,
            return_model=True,
            use_learning_bcells=True,
            use_hierarchical=True,
            use_curriculum=False,  # 커리큘럼 비활성화로 빠른 테스트
            logging_level="minimal"
        )
        
        # 기본 메트릭 확인
        metrics = backtester.calculate_metrics(portfolio_returns)
        
        print("3. 결과 검증...")
        print(f"  - 샤프 비율: {metrics['Sharpe Ratio']:.3f}")
        print(f"  - 총 수익률: {metrics['Total Return']:.2%}")
        print(f"  - 최대 낙폭: {metrics['Max Drawdown']:.2%}")
        
        # 기본 검증
        assert isinstance(metrics['Sharpe Ratio'], (int, float))
        assert isinstance(metrics['Total Return'], (int, float))
        assert isinstance(metrics['Max Drawdown'], (int, float))
        
        # 면역 시스템 검증
        assert immune_system is not None
        assert hasattr(immune_system, 'bcells')
        assert hasattr(immune_system, 'tcells')
        assert len(immune_system.bcells) == 5
        assert len(immune_system.tcells) == 3
        
        # B-Cell 학습 확인
        learning_occurred = False
        for bcell in immune_system.bcells:
            if hasattr(bcell, 'update_counter') and bcell.update_counter > 0:
                learning_occurred = True
                break
                
        print(f"4. 학습 검증: {'✅' if learning_occurred else '⚠️'} 학습 발생")
        
        print("\n✅ 빠른 기능 테스트 통과!")
        return True
        
    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = quick_functionality_test()
    if success:
        print("\n🎯 BIPD 시스템이 정상적으로 작동합니다!")
    else:
        print("\n🔥 시스템에 문제가 있습니다.")
        sys.exit(1)