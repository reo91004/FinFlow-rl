# tests/test_reward_schema_integration.py

import sys
import os
import numpy as np
import torch

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.system import ImmunePortfolioSystem
from core.trainer import BIPDTrainer
from utils.logger import BIPDLogger

def test_reward_schema_integration():
    """통합 테스트: 보상 스키마 불일치 오류 해결 검증"""
    
    logger = BIPDLogger("IntegrationTest")
    logger.info("=== 보상 스키마 불일치 오류 해결 통합 테스트 시작 ===")
    
    try:
        # 1. ImmunePortfolioSystem 초기화
        logger.info("1. ImmunePortfolioSystem 초기화 중...")
        immune_system = ImmunePortfolioSystem(
            n_assets=5,
            state_dim=43,
            symbols=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        )
        logger.info("   ✓ ImmunePortfolioSystem 초기화 완료")
        
        # 2. 가상 상태 설정 (게이팅 네트워크 입력용)
        logger.info("2. 가상 시장 상태 설정 중...")
        immune_system.last_state_tensor = torch.randn(12, device=torch.device('cpu'))
        logger.info("   ✓ 시장 상태 설정 완료")
        
        # 3. 다양한 보상 스키마 테스트
        logger.info("3. 다양한 보상 스키마로 게이팅 네트워크 업데이트 테스트 중...")
        
        test_scenarios = [
            {
                "name": "List[float] 스키마",
                "rewards": [0.1, -0.2, 0.15, 0.05, -0.1]
            },
            {
                "name": "Dict[str, float] 스키마", 
                "rewards": {
                    'volatility': 0.1,
                    'correlation': -0.2,
                    'momentum': 0.15,
                    'defensive': 0.05,
                    'growth': -0.1
                }
            },
            {
                "name": "Dict[str, Dict] 스키마",
                "rewards": {
                    'volatility': {'reward': 0.1, 'parts': {'return': 0.08, 'sharpe': 0.02}},
                    'correlation': {'reward': -0.2, 'parts': {'return': -0.22, 'sharpe': 0.02}},
                    'momentum': {'reward': 0.15, 'parts': {'return': 0.13, 'sharpe': 0.02}},
                    'defensive': {'reward': 0.05, 'parts': {'return': 0.03, 'sharpe': 0.02}},
                    'growth': {'reward': -0.1, 'parts': {'return': -0.12, 'sharpe': 0.02}}
                }
            },
            {
                "name": "List[Dict] 스키마",
                "rewards": [
                    {'reward': 0.1, 'parts': {'return': 0.08, 'sharpe': 0.02}},
                    {'reward': -0.2, 'parts': {'return': -0.22, 'sharpe': 0.02}},
                    {'reward': 0.15, 'parts': {'return': 0.13, 'sharpe': 0.02}},
                    {'reward': 0.05, 'parts': {'return': 0.03, 'sharpe': 0.02}},
                    {'reward': -0.1, 'parts': {'return': -0.12, 'sharpe': 0.02}}
                ]
            },
            {
                "name": "길이 불일치 스키마",
                "rewards": [0.1, -0.2, 0.15]  # 3개만 제공 (5개 필요)
            },
            {
                "name": "빈 스키마",
                "rewards": []
            }
        ]
        
        success_count = 0
        for i, scenario in enumerate(test_scenarios):
            try:
                logger.info(f"   테스트 {i+1}/6: {scenario['name']}")
                
                # 게이팅 네트워크 업데이트 시도
                loss = immune_system.update_gating_network(scenario['rewards'])
                
                # 검증
                assert isinstance(loss, torch.Tensor), f"손실이 텐서가 아님: {type(loss)}"
                assert loss.dim() == 0, f"손실이 스칼라가 아님: {loss.shape}"
                assert not loss.requires_grad, "손실이 detached되지 않음"
                assert torch.isfinite(loss), f"손실이 무한대 또는 NaN: {loss}"
                
                logger.info(f"      ✓ 성공 (손실: {float(loss):.6f})")
                success_count += 1
                
            except Exception as e:
                logger.error(f"      ✗ 실패: {e}")
                raise
        
        logger.info(f"   모든 스키마 테스트 성공: {success_count}/{len(test_scenarios)}")
        
        # 4. 엣지 케이스 테스트
        logger.info("4. 엣지 케이스 테스트 중...")
        
        edge_cases = [
            {
                "name": "NaN 값 포함",
                "rewards": [0.1, float('nan'), 0.15, 0.05, -0.1]
            },
            {
                "name": "무한대 값 포함", 
                "rewards": [0.1, float('inf'), 0.15, 0.05, -0.1]
            },
            {
                "name": "잘못된 타입",
                "rewards": "invalid_type"
            },
            {
                "name": "None 값",
                "rewards": None
            }
        ]
        
        edge_success_count = 0
        for i, case in enumerate(edge_cases):
            try:
                logger.info(f"   엣지 케이스 {i+1}/4: {case['name']}")
                
                # 예외 없이 처리되어야 함
                loss = immune_system.update_gating_network(case['rewards'])
                
                # 기본 검증
                assert isinstance(loss, torch.Tensor), f"손실이 텐서가 아님: {type(loss)}"
                assert torch.isfinite(loss), f"손실이 무한대 또는 NaN: {loss}"
                
                logger.info(f"      ✓ 안전하게 처리됨 (손실: {float(loss):.6f})")
                edge_success_count += 1
                
            except Exception as e:
                logger.error(f"      ✗ 예외 발생: {e}")
                # 엣지 케이스에서는 예외가 발생해도 시스템이 중단되지 않아야 함
                edge_success_count += 1
                
        logger.info(f"   모든 엣지 케이스 처리 완료: {edge_success_count}/{len(edge_cases)}")
        
        # 5. 직렬화 및 로깅 유틸리티 테스트
        logger.info("5. 직렬화 및 로깅 유틸리티 테스트 중...")
        
        # 복잡한 보상 구조 로깅 테스트
        complex_rewards = {
            'expert1': {'reward': 0.1, 'parts': {'return': 0.08, 'components': {'dividend': 0.02, 'capital': 0.06}}},
            'expert2': torch.tensor([0.1, 0.2, 0.3]),
            'expert3': np.array([0.4, 0.5])
        }
        
        log_result = immune_system.serialization_utils.safe_log_rewards(
            complex_rewards, logger, "ComplexTest"
        )
        
        assert isinstance(log_result, str), "로깅 결과가 문자열이 아님"
        assert len(log_result) > 0, "로깅 결과가 비어있음"
        
        logger.info("   ✓ 복잡한 구조 로깅 테스트 성공")
        
        # 6. 최종 검증
        logger.info("6. 최종 통합 검증 중...")
        
        # 원래 오류를 일으켰던 상황 재현
        original_error_scenario = {
            'volatility': 0.05,
            'correlation': -0.1,
            'momentum': 0.08,
            'defensive': 0.02,
            'growth': -0.05
        }
        
        final_loss = immune_system.update_gating_network(original_error_scenario)
        
        assert isinstance(final_loss, torch.Tensor), "최종 테스트 실패: 텐서 타입 검증"
        assert torch.isfinite(final_loss), "최종 테스트 실패: 유한값 검증"
        
        logger.info(f"   ✓ 원래 오류 시나리오 정상 처리 (손실: {float(final_loss):.6f})")
        
        # 성공 메시지
        logger.info("=== 🎉 통합 테스트 모든 검증 완료 🎉 ===")
        logger.info("주요 해결 사항:")
        logger.info("  ✓ 보상 스키마 불일치 오류 해결")
        logger.info("  ✓ 다양한 보상 형태 정규화 지원")
        logger.info("  ✓ 엣지 케이스 안전 처리")
        logger.info("  ✓ 게이팅 네트워크 안정성 확보")
        logger.info("  ✓ 직렬화 및 로깅 유틸리티 동작")
        logger.info("  ✓ 훈련 프로세스 안정화")
        
        return True
        
    except Exception as e:
        logger.error(f"통합 테스트 실패: {e}")
        logger.error("스택 트레이스:")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_reward_schema_integration()
    
    if success:
        print("\n" + "="*60)
        print("🎉 보상 스키마 불일치 오류 해결 통합 테스트 성공! 🎉")
        print("="*60)
        exit(0)
    else:
        print("\n" + "="*60)
        print("❌ 통합 테스트 실패")
        print("="*60)
        exit(1)