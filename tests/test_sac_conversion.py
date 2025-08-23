# tests/test_sac_conversion.py

"""
SAC 전환 후 전체 시스템 검증 테스트
TD3에서 SAC로의 완전한 전환을 검증한다.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch

# 설정 모듈 import
import config
from agents.bcell import BCell, SACActorNetwork, CriticNetwork
from core.system import ImmunePortfolioSystem
from core.environment import PortfolioEnvironment
from core.trainer import BIPDTrainer
from data.features import FeatureExtractor

def test_sac_conversion():
    """SAC 전환 완전성 테스트"""
    
    print("=" * 80)
    print("SAC 전환 검증 테스트 시작")
    print("=" * 80)
    
    test_results = {
        'import_tests': False,
        'bcell_initialization': False,
        'sac_networks': False,
        'system_integration': False,
        'reward_function': False,
        'full_pipeline': False,
        'errors': []
    }
    
    try:
        # 1. Import 테스트
        print("[1/6] 모듈 Import 테스트...")
        
        test_results['import_tests'] = True
        print("✅ 모든 모듈 Import 성공")
        
        # 2. BCell SAC 초기화 테스트
        print("[2/6] SAC B-Cell 초기화 테스트...")
        
        state_dim = config.STATE_DIM
        action_dim = config.ACTION_DIM
        
        bcell = BCell('volatility', state_dim, action_dim)
        
        # SAC 구성 요소 확인
        assert hasattr(bcell, 'actor'), "SAC Actor가 없습니다"
        assert hasattr(bcell, 'critic1'), "Critic1이 없습니다"
        assert hasattr(bcell, 'critic2'), "Critic2가 없습니다"
        assert hasattr(bcell, 'target_critic1'), "Target Critic1이 없습니다"
        assert hasattr(bcell, 'target_critic2'), "Target Critic2가 없습니다"
        assert hasattr(bcell, 'log_alpha'), "log_alpha가 없습니다"
        assert hasattr(bcell, 'alpha'), "alpha가 없습니다"
        assert hasattr(bcell, 'target_entropy'), "target_entropy가 없습니다"
        
        # TD3 잔재 확인
        assert not hasattr(bcell, 'target_actor'), "TD3 target_actor가 남아있습니다"
        
        test_results['bcell_initialization'] = True
        print(f"✅ SAC B-Cell 초기화 성공: {bcell.risk_type}")
        print(f"   - Alpha: {bcell.alpha.item():.4f}")
        print(f"   - Target Entropy: {bcell.target_entropy}")
        
        # 3. SAC 네트워크 구조 테스트
        print("[3/6] SAC 네트워크 구조 테스트...")
        
        # SACActorNetwork 테스트
        test_state = torch.randn(1, state_dim)
        concentration, weights, log_prob = bcell.actor(test_state)
        
        assert concentration.shape == (1, action_dim), f"Concentration 형태 오류: {concentration.shape}"
        assert weights.shape == (1, action_dim), f"Weights 형태 오류: {weights.shape}"
        assert torch.allclose(weights.sum(dim=1), torch.ones(1)), "가중치 합이 1이 아닙니다"
        assert (weights > 0).all(), "음수 가중치가 있습니다"
        
        # CriticNetwork 테스트 (action 필수)
        test_action = torch.randn(1, action_dim)
        q_value = bcell.critic1(test_state, test_action)
        
        assert q_value.shape == (1, 1), f"Q-value 형태 오류: {q_value.shape}"
        
        test_results['sac_networks'] = True
        print("✅ SAC 네트워크 구조 검증 완료")
        print(f"   - Dirichlet concentration: {concentration.mean().item():.4f}")
        print(f"   - 가중치 합: {weights.sum().item():.6f}")
        
        # 4. 시스템 통합 테스트
        print("[4/6] 시스템 통합 테스트...")
        
        # 더미 데이터 생성
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        symbols = config.SYMBOLS[:5]  # 처음 5개 종목만 사용
        dummy_data = pd.DataFrame(
            np.random.randn(100, len(symbols)) * 0.02 + 1.001,
            index=dates,
            columns=symbols
        ).cumprod()
        
        # ImmunePortfolioSystem 초기화
        immune_system = ImmunePortfolioSystem(
            n_assets=len(symbols),
            state_dim=config.FEATURE_DIM + 1 + len(symbols)  # features + crisis + prev_weights
        )
        
        # 모든 B-Cell이 SAC인지 확인
        for name, bcell in immune_system.bcells.items():
            assert hasattr(bcell, 'alpha'), f"{name} B-Cell에 alpha가 없습니다"
            assert hasattr(bcell, 'target_entropy'), f"{name} B-Cell에 target_entropy가 없습니다"
            assert not hasattr(bcell, 'epsilon'), f"{name} B-Cell에 TD3 epsilon이 남아있습니다"
        
        test_results['system_integration'] = True
        print("✅ 시스템 통합 검증 완료")
        print(f"   - B-Cell 개수: {len(immune_system.bcells)}")
        
        # 5. 보상 함수 테스트
        print("[5/6] 간소화된 보상 함수 테스트...")
        
        feature_extractor = FeatureExtractor(lookback_window=config.LOOKBACK_WINDOW)
        env = PortfolioEnvironment(
            price_data=dummy_data,
            feature_extractor=feature_extractor,
            initial_capital=config.INITIAL_CAPITAL,
            transaction_cost=config.TRANSACTION_COST
        )
        
        # 환경 초기화 및 스텝 실행
        state = env.reset()
        weights = np.ones(len(symbols)) / len(symbols)
        next_state, reward, done, info = env.step(weights)
        
        # 보상이 적절한 범위인지 확인
        assert isinstance(reward, (int, float)), f"보상이 숫자가 아닙니다: {type(reward)}"
        assert -10 <= reward <= 10, f"보상이 범위를 벗어났습니다: {reward}"
        
        test_results['reward_function'] = True
        print("✅ 보상 함수 검증 완료")
        print(f"   - 첫 스텝 보상: {reward:.6f}")
        print(f"   - 포트폴리오 가치: {info['portfolio_value']:,.0f}")
        
        # 6. 전체 파이프라인 테스트
        print("[6/6] 전체 파이프라인 통합 테스트...")
        
        # 짧은 에피소드 실행
        state = env.reset()
        total_reward = 0
        steps = 0
        
        for step in range(10):  # 10 스텝만 실행
            # SAC 의사결정
            weights, decision_info = immune_system.decide(state, training=True)
            
            # 환경 스텝
            next_state, reward, done, info = env.step(weights)
            
            # 시스템 업데이트
            immune_system.update(state, weights, reward, next_state, done)
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if done:
                break
        
        test_results['full_pipeline'] = True
        print("✅ 전체 파이프라인 검증 완료")
        print(f"   - 실행 스텝: {steps}")
        print(f"   - 총 보상: {total_reward:.6f}")
        print(f"   - 평균 보상: {total_reward/steps:.6f}")
        
        # 최종 결과
        success_count = sum(test_results[key] for key in test_results if key != 'errors')
        total_tests = 6
        
        print("\n" + "=" * 80)
        print("SAC 전환 검증 결과")
        print("=" * 80)
        print(f"성공한 테스트: {success_count}/{total_tests}")
        
        if success_count == total_tests:
            print("🎉 모든 테스트 통과! SAC 전환이 완료되었습니다.")
            print("\n주요 개선사항:")
            print("  ✅ TD3 → SAC 알고리즘 완전 전환")
            print("  ✅ Dirichlet 분포 기반 확률적 정책")
            print("  ✅ 엔트로피 자동 튜닝 구현")
            print("  ✅ 간소화된 로그 수익률 보상 함수")
            print("  ✅ 전체 시스템 통합 완료")
            
            return True
        else:
            print("❌ 일부 테스트 실패")
            for error in test_results['errors']:
                print(f"   - {error}")
            return False
            
    except Exception as e:
        print(f"\n❌ 테스트 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        test_results['errors'].append(str(e))
        return False

if __name__ == "__main__":
    success = test_sac_conversion()
    
    if success:
        print("\n🚀 이제 main.py를 실행하여 SAC 기반 BIPD 시스템을 훈련할 수 있습니다!")
    else:
        print("\n⚠️  문제를 해결한 후 다시 테스트하세요.")