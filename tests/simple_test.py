# tests/simple_test.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from agents import BCell, TCell, MemoryCell
from core.system import ImmunePortfolioSystem
from core.reward import RewardCalculator
from constant import *


def simple_system_test():
    """매우 간단한 시스템 테스트"""
    print("=== 간단한 시스템 기능 테스트 ===")
    
    try:
        # 1. B-Cell 초기화 테스트
        print("1. B-Cell 초기화 테스트...")
        bcell = BCell("test", "volatility", 23, 10)
        assert bcell is not None
        assert hasattr(bcell, 'target_critic_network')
        print("  ✅ B-Cell 초기화 성공")
        
        # 2. T-Cell 초기화 테스트
        print("2. T-Cell 초기화 테스트...")
        tcell = TCell("T1", 0.1)
        assert tcell is not None
        print("  ✅ T-Cell 초기화 성공")
        
        # 3. 시스템 초기화 테스트
        print("3. 시스템 초기화 테스트...")
        system = ImmunePortfolioSystem(n_assets=10, n_tcells=3, n_bcells=5)
        assert system is not None
        assert len(system.bcells) == 5
        assert len(system.tcells) == 3
        print("  ✅ 시스템 초기화 성공")
        
        # 4. 보상 계산 테스트
        print("4. 보상 계산 테스트...")
        reward_calc = RewardCalculator()
        result = reward_calc.calculate_comprehensive_reward(
            current_return=0.01,
            previous_weights=np.ones(10)/10,
            current_weights=np.random.random(10),
            market_features=np.random.random(12),
            crisis_level=0.5
        )
        assert 'total_reward' in result
        assert isinstance(result['total_reward'], float)
        print(f"  ✅ 보상 계산 성공: {result['total_reward']:.3f}")
        
        # 5. Experience 추가 테스트
        print("5. Experience 추가 테스트...")
        state = np.random.random(23)
        action = np.random.random(10)
        action = action / action.sum()  # 정규화
        reward = 0.1
        next_state = np.random.random(23)
        done = False
        
        bcell.add_experience(state, action, reward, next_state, done)
        assert len(bcell.experience_buffer) == 1
        print("  ✅ Experience 추가 성공")
        
        print("\n🎉 모든 간단한 테스트 통과!")
        return True
        
    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = simple_system_test()
    if success:
        print("\n✅ BIPD 기본 컴포넌트가 정상 작동합니다!")
    else:
        print("\n❌ 기본 컴포넌트에 문제가 있습니다.")
        sys.exit(1)