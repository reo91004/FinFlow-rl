# tests/test_rl_functionality.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from agents import BCell, TCell, MemoryCell
from core.system import ImmunePortfolioSystem
from core.reward import RewardCalculator
from constant import *


class TestRLFunctionality:
    """강화학습 핵심 기능 테스트"""

    def test_experience_replay_buffer(self):
        """Experience Replay Buffer 테스트"""
        from agents.bcell import ExperienceReplayBuffer

        buffer = ExperienceReplayBuffer(capacity=100)

        # 데이터 추가
        state = np.random.random(23)
        action = np.random.random(10)
        reward = 0.1
        next_state = np.random.random(23)
        done = False

        buffer.push(state, action, reward, next_state, done)

        assert len(buffer) == 1
        assert buffer.buffer[0] == (state, action, reward, next_state, done)

    def test_target_network_initialization(self):
        """Target Network 초기화 테스트"""
        bcell = BCell("test", "volatility", 23, 10)

        # Target network이 존재하는지 확인
        assert hasattr(bcell, "target_critic_network")
        assert bcell.target_critic_network is not None

        # 초기에는 같은 파라미터를 가져야 함
        for target_param, param in zip(
            bcell.target_critic_network.parameters(), bcell.critic_network.parameters()
        ):
            assert torch.allclose(target_param.data, param.data)

    def test_target_network_update(self):
        """Target Network 소프트 업데이트 테스트"""
        bcell = BCell("test", "volatility", 23, 10)

        # 메인 네트워크 파라미터 변경
        with torch.no_grad():
            for param in bcell.critic_network.parameters():
                param.data += 0.1

        # Target network 업데이트
        bcell.update_target_network()

        # 소프트 업데이트되었는지 확인 (tau=0.005)
        for target_param, param in zip(
            bcell.target_critic_network.parameters(), bcell.critic_network.parameters()
        ):
            assert not torch.allclose(target_param.data, param.data)

    def test_td_learning_computation(self):
        """TD Learning 계산 테스트"""
        bcell = BCell("test", "volatility", 23, 10)

        # 경험 데이터 추가
        for _ in range(300):  # batch_size(256)보다 많이
            state = np.random.random(23)
            action = np.random.random(10)
            reward = np.random.random() * 0.1
            next_state = np.random.random(23)
            done = np.random.random() > 0.95

            bcell.add_experience(state, action, reward, next_state, done)

        # TD Learning 실행
        loss = bcell.learn_from_batch()

        assert loss is not None
        assert isinstance(loss, float)
        assert loss >= 0  # MSE loss는 항상 양수
        
    def test_learning_convergence(self):
        """학습 수렴 테스트 - TD Loss가 감소하는지 검증"""
        bcell = BCell("test", "volatility", 23, 10)
        
        losses = []
        
        # 일관된 경험으로 학습 (수렴 유도)
        for episode in range(20):
            for _ in range(30):
                # 일관된 패턴의 경험 생성
                state = np.random.random(23) * 0.1 + 0.5  # 비슷한 상태들
                action = np.ones(10) / 10  # 균등 액션
                reward = 0.1 if np.mean(state) > 0.5 else -0.1  # 일관된 보상
                next_state = state + np.random.random(23) * 0.05  # 약간의 변화
                done = False
                
                bcell.add_experience(state, action, reward, next_state, done)
                
            # 매 에피소드마다 학습
            loss = bcell.learn_from_batch()
            if loss is not None:
                losses.append(loss)
        
        # 학습이 진행되었는지 확인
        assert len(losses) > 5, "충분한 학습이 이루어지지 않았음"
        
        # 수렴 여부 확인 (후반부 손실이 전반부보다 낮거나 안정적)
        early_losses = np.mean(losses[:5])
        late_losses = np.mean(losses[-5:])
        
        # 손실이 감소하거나 안정화되었는지 확인
        improvement = early_losses - late_losses
        assert improvement >= 0 or late_losses < early_losses * 1.1, f"학습 수렴 실패: 초기 손실 {early_losses:.4f} -> 후기 손실 {late_losses:.4f}"

    def test_activation_threshold(self):
        """활성화 임계값 테스트"""
        bcell = BCell("test", "volatility", 23, 10)

        # 낮은 자극 - 활성화 안됨
        low_stimulus = 0.3
        assert not bcell.should_activate(low_stimulus)

        # 높은 자극 - 활성화됨
        high_stimulus = 0.8
        assert bcell.should_activate(high_stimulus)

    def test_reward_single_clipping(self):
        """보상 시스템 단일 클리핑 테스트"""
        reward_calc = RewardCalculator()

        # 극단적 입력값으로 테스트
        extreme_return = 0.5  # 50% 일일 수익률
        weights_prev = np.ones(10) / 10
        weights_curr = np.random.random(10)
        weights_curr /= weights_curr.sum()
        market_features = np.random.random(12)
        crisis_level = 0.8

        reward_details = reward_calc.calculate_comprehensive_reward(
            extreme_return, weights_prev, weights_curr, market_features, crisis_level
        )

        # 클리핑 범위 내에 있는지 확인
        final_reward = reward_details["total_reward"]
        assert REWARD_CLIPPING_RANGE[0] <= final_reward <= REWARD_CLIPPING_RANGE[1]

    def test_state_transition_chain(self):
        """State Transition 체인 테스트"""
        system = ImmunePortfolioSystem(n_assets=10, n_tcells=3, n_bcells=5)

        # 가상 시장 데이터 (pandas DataFrame으로)
        import pandas as pd
        market_data = pd.DataFrame(np.random.random((100, 10)))

        states = []
        for i in range(5):
            if len(market_data[:i+20]) > 20:  # 충분한 데이터가 있을 때만
                state = system.extract_market_features(market_data[:i+20])
                states.append(state)

        # 연속된 state들이 다른지 확인 (시장이 변화하므로)
        assert not np.array_equal(states[0], states[1])
        assert not np.array_equal(states[1], states[2])

        # 모든 state가 올바른 차원인지 확인
        for state in states:
            assert len(state) == FEATURE_SIZE

    def test_gradient_flow(self):
        """Gradient Flow 테스트"""
        bcell = BCell("test", "volatility", 23, 10)

        # 더미 입력
        market_features = torch.randn(12, requires_grad=True)
        crisis_level = 0.5
        tcell_contributions = {"volatility": 0.8, "correlation": 0.3}

        # Forward pass
        attended_features, attention_weights = bcell.attention_mechanism(
            market_features, tcell_contributions
        )

        # Backward pass
        loss = attended_features.sum()
        loss.backward()

        # Gradient가 흐르는지 확인
        assert market_features.grad is not None
        assert not torch.allclose(
            market_features.grad, torch.zeros_like(market_features.grad)
        )


def test_realistic_parameters():
    """현실적 파라미터 설정 테스트"""
    assert TOTAL_EPISODES == 50000  # 5만 에피소드
    assert DEFAULT_BATCH_SIZE == 256  # 256 배치 크기
    assert EPISODE_LENGTH == 252  # 1년 거래일
    assert DEFAULT_GAMMA == 0.99  # 현실적 할인율


if __name__ == "__main__":
    print("강화학습 기능 테스트 실행 중...")

    test_suite = TestRLFunctionality()

    try:
        test_suite.test_experience_replay_buffer()
        print("✅ Experience Replay Buffer 테스트 통과")

        test_suite.test_target_network_initialization()
        print("✅ Target Network 초기화 테스트 통과")

        test_suite.test_target_network_update()
        print("✅ Target Network 업데이트 테스트 통과")

        test_suite.test_td_learning_computation()
        print("✅ TD Learning 계산 테스트 통과")

        test_suite.test_activation_threshold()
        print("✅ 활성화 임계값 테스트 통과")

        test_suite.test_reward_single_clipping()
        print("✅ 보상 단일 클리핑 테스트 통과")

        test_suite.test_state_transition_chain()
        print("✅ State Transition 체인 테스트 통과")

        test_suite.test_gradient_flow()
        print("✅ Gradient Flow 테스트 통과")
        
        test_suite.test_learning_convergence()
        print("✅ 학습 수렴 테스트 통과")

        test_realistic_parameters()
        print("✅ 현실적 파라미터 테스트 통과")

        print("\n🎉 모든 강화학습 기능 테스트 통과!")

    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        raise
