# tests/test_schema_normalization.py

import unittest
import sys
import os
import numpy as np
import torch

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.system import ImmunePortfolioSystem, SerializationUtils, LoggingEnhancer
from utils.logger import BIPDLogger
from config import *

class TestSchemaNormalization(unittest.TestCase):
    """스키마 정규화 기능 단위 테스트"""
    
    def setUp(self):
        """테스트 환경 설정"""
        self.n_assets = 5
        self.state_dim = 43
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        
        # ImmunePortfolioSystem 인스턴스 생성
        self.immune_system = ImmunePortfolioSystem(
            n_assets=self.n_assets,
            state_dim=self.state_dim,
            symbols=self.symbols
        )
        
        # 테스트용 로거
        self.logger = BIPDLogger("TestLogger")
        self.serialization_utils = SerializationUtils()
    
    def test_normalize_expert_rewards_list_float(self):
        """List[float] 형태 보상 정규화 테스트"""
        expert_rewards = [0.1, -0.2, 0.15, 0.05, -0.1]
        
        rewards_tensor, parts_list = self.immune_system._normalize_expert_rewards(expert_rewards)
        
        # 검증
        self.assertIsInstance(rewards_tensor, torch.Tensor)
        self.assertEqual(rewards_tensor.shape, (5,))
        self.assertEqual(rewards_tensor.device, torch.device(DEVICE))
        self.assertIsNone(parts_list)
        
        # 값 검증
        expected = torch.tensor(expert_rewards, device=DEVICE, dtype=torch.float32)
        torch.testing.assert_close(rewards_tensor, expected)
    
    def test_normalize_expert_rewards_list_dict(self):
        """List[Dict] 형태 보상 정규화 테스트"""
        expert_rewards = [
            {'reward': 0.1, 'parts': {'return': 0.08, 'sharpe': 0.02}},
            {'reward': -0.2, 'parts': {'return': -0.25, 'sharpe': 0.05}},
            {'reward': 0.15, 'parts': {'return': 0.12, 'sharpe': 0.03}},
            {'reward': 0.05, 'parts': {'return': 0.03, 'sharpe': 0.02}},
            {'reward': -0.1, 'parts': {'return': -0.12, 'sharpe': 0.02}}
        ]
        
        rewards_tensor, parts_list = self.immune_system._normalize_expert_rewards(expert_rewards)
        
        # 검증
        self.assertIsInstance(rewards_tensor, torch.Tensor)
        self.assertEqual(rewards_tensor.shape, (5,))
        self.assertIsNotNone(parts_list)
        self.assertEqual(len(parts_list), 5)
        
        # 값 검증
        expected_rewards = [0.1, -0.2, 0.15, 0.05, -0.1]
        expected = torch.tensor(expected_rewards, device=DEVICE, dtype=torch.float32)
        torch.testing.assert_close(rewards_tensor, expected)
        
        # parts 검증
        self.assertEqual(parts_list[0]['return'], 0.08)
        self.assertEqual(parts_list[1]['return'], -0.25)
    
    def test_normalize_expert_rewards_dict_float(self):
        """Dict[str, float] 형태 보상 정규화 테스트"""
        expert_rewards = {
            'volatility': 0.1,
            'correlation': -0.2,
            'momentum': 0.15,
            'defensive': 0.05,
            'growth': -0.1
        }
        
        rewards_tensor, parts_list = self.immune_system._normalize_expert_rewards(expert_rewards)
        
        # 검증
        self.assertIsInstance(rewards_tensor, torch.Tensor)
        self.assertEqual(rewards_tensor.shape, (5,))
        self.assertIsNotNone(parts_list)
        
        # 순서가 bcell_names와 일치하는지 확인
        for i, bcell_name in enumerate(self.immune_system.bcell_names):
            expected_reward = expert_rewards[bcell_name]
            self.assertAlmostEqual(float(rewards_tensor[i]), expected_reward, places=5)
    
    def test_normalize_expert_rewards_dict_dict(self):
        """Dict[str, Dict] 형태 보상 정규화 테스트"""
        expert_rewards = {
            'volatility': {'reward': 0.1, 'parts': {'return': 0.08, 'sharpe': 0.02}},
            'correlation': {'reward': -0.2, 'parts': {'return': -0.25, 'sharpe': 0.05}},
            'momentum': {'reward': 0.15, 'parts': {'return': 0.12, 'sharpe': 0.03}},
            'defensive': {'reward': 0.05, 'parts': {'return': 0.03, 'sharpe': 0.02}},
            'growth': {'reward': -0.1, 'parts': {'return': -0.12, 'sharpe': 0.02}}
        }
        
        rewards_tensor, parts_list = self.immune_system._normalize_expert_rewards(expert_rewards)
        
        # 검증
        self.assertIsInstance(rewards_tensor, torch.Tensor)
        self.assertEqual(rewards_tensor.shape, (5,))
        self.assertIsNotNone(parts_list)
        
        # 순서와 값 검증
        for i, bcell_name in enumerate(self.immune_system.bcell_names):
            expected_reward = expert_rewards[bcell_name]['reward']
            self.assertAlmostEqual(float(rewards_tensor[i]), expected_reward, places=5)
            
            # parts 검증
            expected_parts = expert_rewards[bcell_name]['parts']
            self.assertEqual(parts_list[i], expected_parts)
    
    def test_normalize_expert_rewards_length_mismatch(self):
        """길이 불일치 처리 테스트"""
        # 너무 짧은 리스트
        expert_rewards = [0.1, -0.2, 0.15]  # 3개 (5개 필요)
        
        rewards_tensor, parts_list = self.immune_system._normalize_expert_rewards(expert_rewards)
        
        # 패딩으로 5개가 되어야 함
        self.assertEqual(rewards_tensor.shape, (5,))
        self.assertEqual(float(rewards_tensor[3]), 0.0)  # 패딩된 값
        self.assertEqual(float(rewards_tensor[4]), 0.0)  # 패딩된 값
        
        # 너무 긴 리스트
        expert_rewards = [0.1, -0.2, 0.15, 0.05, -0.1, 0.2, 0.3]  # 7개
        
        rewards_tensor, parts_list = self.immune_system._normalize_expert_rewards(expert_rewards)
        
        # 처음 5개만 사용되어야 함
        self.assertEqual(rewards_tensor.shape, (5,))
        expected = torch.tensor([0.1, -0.2, 0.15, 0.05, -0.1], device=DEVICE, dtype=torch.float32)
        torch.testing.assert_close(rewards_tensor, expected)
    
    def test_normalize_expert_rewards_empty_input(self):
        """빈 입력 처리 테스트"""
        expert_rewards = []
        
        rewards_tensor, parts_list = self.immune_system._normalize_expert_rewards(expert_rewards)
        
        # 모두 0으로 패딩되어야 함
        self.assertEqual(rewards_tensor.shape, (5,))
        expected = torch.zeros(5, device=DEVICE, dtype=torch.float32)
        torch.testing.assert_close(rewards_tensor, expected)
    
    def test_normalize_expert_rewards_invalid_type(self):
        """잘못된 타입 처리 테스트"""
        expert_rewards = "invalid_type"
        
        rewards_tensor, parts_list = self.immune_system._normalize_expert_rewards(expert_rewards)
        
        # 제로 텐서를 반환해야 함
        self.assertEqual(rewards_tensor.shape, (5,))
        expected = torch.zeros(5, device=DEVICE, dtype=torch.float32)
        torch.testing.assert_close(rewards_tensor, expected)
        self.assertIsNone(parts_list)
    
    def test_serialization_utils_safe_tensor_to_python(self):
        """텐서 직렬화 유틸리티 테스트"""
        # PyTorch 텐서
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = self.serialization_utils.safe_tensor_to_python(tensor)
        self.assertEqual(result, [1.0, 2.0, 3.0])
        
        # NumPy 배열
        array = np.array([1.0, 2.0, 3.0])
        result = self.serialization_utils.safe_tensor_to_python(array)
        self.assertEqual(result, [1.0, 2.0, 3.0])
        
        # 복합 구조
        complex_obj = {
            'tensor': torch.tensor([1.0, 2.0]),
            'array': np.array([3.0, 4.0]),
            'list': [torch.tensor(5.0), np.float32(6.0)]
        }
        result = self.serialization_utils.safe_tensor_to_python(complex_obj)
        expected = {
            'tensor': [1.0, 2.0],
            'array': [3.0, 4.0],
            'list': [5.0, 6.0]
        }
        self.assertEqual(result, expected)
    
    def test_serialization_utils_safe_json_serialize(self):
        """JSON 직렬화 유틸리티 테스트"""
        # 정상 케이스
        obj = {'a': 1, 'b': [2, 3]}
        result = self.serialization_utils.safe_json_serialize(obj)
        self.assertIn('"a":1', result)
        self.assertIn('"b":[2,3]', result)
        
        # 긴 문자열 잘림 테스트
        long_obj = {'data': 'x' * 2000}
        result = self.serialization_utils.safe_json_serialize(long_obj, max_length=100)
        self.assertTrue(result.endswith('...'))
        self.assertLess(len(result), 105)  # max_length + "..."
    
    def test_serialization_utils_safe_log_rewards(self):
        """보상 로깅 유틸리티 테스트"""
        # 텐서 로깅
        tensor_rewards = torch.tensor([0.1, -0.2, 0.15])
        result = self.serialization_utils.safe_log_rewards(tensor_rewards, self.logger, "Test")
        self.assertIn("[Test]", result)
        self.assertIn("Tensor", result)
        
        # Dict 로깅
        dict_rewards = {'expert1': 0.1, 'expert2': -0.2}
        result = self.serialization_utils.safe_log_rewards(dict_rewards, self.logger, "Test")
        self.assertIn("[Test]", result)
        self.assertIn("보상", result)
        
        # List 로깅
        list_rewards = [0.1, -0.2, 0.15]
        result = self.serialization_utils.safe_log_rewards(list_rewards, self.logger, "Test")
        self.assertIn("[Test]", result)
        self.assertIn("보상", result)
    
    def test_update_gating_network_integration(self):
        """게이팅 네트워크 업데이트 통합 테스트"""
        # 가상의 상태 설정
        self.immune_system.last_state_tensor = torch.randn(12, device=DEVICE)
        
        # 다양한 스키마로 테스트
        test_cases = [
            # List[float]
            [0.1, -0.2, 0.15, 0.05, -0.1],
            # Dict[str, float]
            {
                'volatility': 0.1,
                'correlation': -0.2,
                'momentum': 0.15,
                'defensive': 0.05,
                'growth': -0.1
            },
            # Dict[str, Dict]
            {
                'volatility': {'reward': 0.1, 'parts': {}},
                'correlation': {'reward': -0.2, 'parts': {}},
                'momentum': {'reward': 0.15, 'parts': {}},
                'defensive': {'reward': 0.05, 'parts': {}},
                'growth': {'reward': -0.1, 'parts': {}}
            }
        ]
        
        for i, expert_rewards in enumerate(test_cases):
            with self.subTest(test_case=i):
                try:
                    loss = self.immune_system.update_gating_network(expert_rewards)
                    
                    # 손실이 텐서이고 스칼라 값이어야 함
                    self.assertIsInstance(loss, torch.Tensor)
                    self.assertEqual(loss.dim(), 0)  # 스칼라
                    self.assertFalse(loss.requires_grad)  # detached
                    
                    # 손실 값이 유한해야 함
                    self.assertTrue(torch.isfinite(loss))
                    
                except Exception as e:
                    self.fail(f"게이팅 네트워크 업데이트 실패 (테스트 케이스 {i}): {e}")

class TestLoggingEnhancer(unittest.TestCase):
    """로깅 강화 유틸리티 테스트"""
    
    def setUp(self):
        self.logger = BIPDLogger("TestLogger")
        self.enhancer = LoggingEnhancer(self.logger, "TestContext")
    
    def test_debug_with_context(self):
        """컨텍스트 디버그 로깅 테스트"""
        # 예외가 발생하지 않아야 함
        self.enhancer.debug_with_context("테스트 메시지")
        self.enhancer.debug_with_context("테스트 메시지", {"key": "value"})
        
        # 호출 카운트가 증가해야 함
        self.assertEqual(self.enhancer.call_count, 2)
    
    def test_log_tensor_stats(self):
        """텐서 통계 로깅 테스트"""
        tensor = torch.randn(10, 5)
        
        # 예외가 발생하지 않아야 함
        self.enhancer.log_tensor_stats(tensor, "테스트_텐서")
        
        # None 텐서 처리
        self.enhancer.log_tensor_stats(None, "빈_텐서")

if __name__ == '__main__':
    unittest.main(verbosity=2)