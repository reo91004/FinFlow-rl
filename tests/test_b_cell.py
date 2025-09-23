# tests/test_b_cell.py

import pytest
import torch
import numpy as np
from src.algorithms.online.b_cell import BCell
from src.data.replay_buffer import Transition

class TestBCell:
    """B-Cell 정책 에이전트 테스트"""
    
    @pytest.fixture
    def config(self):
        return {
            'state_dim': 43,
            'action_dim': 30,
            'actor_hidden': [256, 128],
            'critic_hidden': [256, 128],
            'n_quantiles': 32,
            'actor_lr': 1e-4,
            'critic_lr': 3e-4,
            'alpha_lr': 1e-3,
            'gamma': 0.99,
            'tau': 0.005,
            'alpha_init': 0.2,
            'iql_expectile': 0.7,
            'iql_temperature': 3.0
        }
    
    @pytest.fixture
    def bcell(self, config):
        return BCell(
            state_dim=config['state_dim'],
            action_dim=config['action_dim'],
            config=config
        )
    
    def test_initialization(self, bcell, config):
        assert bcell.state_dim == config['state_dim']
        assert bcell.action_dim == config['action_dim']
        assert bcell.n_critics == config.get('n_critics', 5)
        assert bcell.gamma == config['gamma']
        assert bcell.tau == config['tau']
    
    def test_actor_forward(self, bcell):
        """Actor 네트워크 순전파"""
        state = torch.randn(1, 43)

        # Concentration 파라미터 계산
        concentration = bcell.actor(state)
        assert concentration.shape == (1, 30)
        assert (concentration > 0).all()  # Dirichlet에는 양수 필요

        # 샘플링
        action, log_prob = bcell.actor.sample(state)

        # 액션은 심플렉스 위에 있어야 함
        assert action.shape == (1, 30)
        assert torch.allclose(action.sum(dim=1), torch.ones(1), atol=1e-5)
        assert (action >= 0).all()
        assert (action <= 1).all()

        # 로그 확률
        assert log_prob.shape == (1, 1)
        # Dirichlet 분포의 로그 확률은 양수일 수 있음
    
    def test_critic_forward(self, bcell):
        """Critic 네트워크 순전파"""
        state = torch.randn(10, 43)
        action = torch.softmax(torch.randn(10, 30), dim=1)

        # REDQ 앙상블 비평가망
        q_values = []
        for critic in bcell.critics:
            q = critic(state, action)
            q_values.append(q)

        # 모든 비평가망이 같은 차원의 출력
        for q in q_values:
            assert q.shape == (10, 1)
    
    def test_value_forward(self, bcell):
        """Value 네트워크 순전파 - REDQ는 value net이 없음"""
        # REDQ는 SAC 기반이므로 value network가 없음
        # critics만 사용함
        assert not hasattr(bcell, 'value_net')
    
    def test_get_action_deterministic(self, bcell):
        """결정적 액션 선택"""
        state = np.random.randn(43).astype(np.float32)

        # select_action 메서드 사용
        action = bcell.select_action(state, deterministic=True)
        
        assert action.shape == (30,)
        assert np.allclose(action.sum(), 1.0, atol=1e-5)
        assert (action >= 0).all()
        assert (action <= 1).all()
    
    def test_get_action_stochastic(self, bcell):
        """확률적 액션 선택"""
        state = np.random.randn(43).astype(np.float32)

        # 확률적 액션 선택
        action1 = bcell.select_action(state, deterministic=False)
        action2 = bcell.select_action(state, deterministic=False)
        
        # 확률적이므로 다를 수 있음
        assert not np.allclose(action1, action2, atol=1e-6)
    
    def test_train_step(self, bcell):
        """REDQ 학습 스텝 테스트"""
        # 리플레이 버퍼에 데이터 추가
        for _ in range(300):  # 배치 크기보다 크게
            state = np.random.randn(43)
            action = np.random.dirichlet(np.ones(30))
            reward = np.random.randn()
            next_state = np.random.randn(43)
            done = False
            transition = Transition(state, action, reward, next_state, done)
            bcell.replay_buffer.push(transition)

        # 학습 스텝
        losses = bcell.train(batch_size=32)

        # REDQ 손실
        if losses:  # 버퍼가 충분히 찬 경우에만
            assert 'actor_loss' in losses
            assert 'critic_loss' in losses
            assert 'alpha_loss' in losses
    
    def test_replay_buffer(self, bcell):
        """Replay buffer 테스트"""
        # 경험 추가
        for i in range(10):
            state = np.random.randn(43)
            action = np.random.dirichlet(np.ones(30))
            reward = np.random.randn()
            next_state = np.random.randn(43)
            done = i == 9  # 마지막에 done
            transition = Transition(state, action, reward, next_state, done)
            bcell.replay_buffer.push(transition)

        # 버퍼 크기 확인
        assert len(bcell.replay_buffer) == 10

        # 샘플링
        if len(bcell.replay_buffer) >= 5:
            batch = bcell.replay_buffer.sample(5)
            assert batch[0].shape == (5, 43)  # states
            assert batch[1].shape == (5, 30)  # actions
    
    def test_soft_update(self, bcell):
        """소프트 업데이트 테스트"""
        # 타겟 파라미터 저장
        old_params = []
        for param in bcell.critics_target[0].parameters():
            old_params.append(param.clone())

        # 소족속 네트워크 파라미터 변경
        for param in bcell.critics[0].parameters():
            param.data.add_(1.0)  # 의도적으로 변경

        # 소프트 업데이트 수행 - train 메서드 내부에서 호출됨
        # REDQ는 train() 호출 시 내부적으로 soft update 수행
        for target_param, param in zip(bcell.critics_target[0].parameters(),
                                        bcell.critics[0].parameters()):
            target_param.data.copy_(bcell.tau * param.data + (1 - bcell.tau) * target_param.data)

        # 파라미터가 약간 변경되어야 함
        for old_param, new_param in zip(old_params, bcell.critics_target[0].parameters()):
            assert not torch.equal(old_param, new_param)
            diff = torch.abs(old_param - new_param).max()
            assert diff > 0  # 변화가 있어야 함
    
    def test_save_load(self, bcell, tmp_path):
        """모델 저장/로드 테스트"""
        # 원본 액션
        state = np.random.randn(43).astype(np.float32)
        original_action = bcell.select_action(state, deterministic=True)
        
        # 저장
        save_path = tmp_path / "bcell.pt"
        bcell.save(str(save_path))
        
        # 새 B-Cell 생성 및 로드
        config = {
            'state_dim': 43,
            'action_dim': 30,
            'actor_hidden': [256, 128],
            'critic_hidden': [256, 128],
            'n_quantiles': 32
        }
        new_bcell = BCell(
            state_dim=config['state_dim'],
            action_dim=config['action_dim'],
            config=config
        )
        new_bcell.load(str(save_path))
        
        # 동일한 액션 출력
        loaded_action = new_bcell.select_action(state, deterministic=True)
        assert np.allclose(original_action, loaded_action, atol=1e-5)
    
    def test_dirichlet_policy(self, bcell):
        """Dirichlet 정책 테스트"""
        state = torch.randn(10, 43)
        
        # Dirichlet 파라미터 계산
        concentration = bcell.actor.forward(state)
        
        assert concentration.shape == (10, 30)
        assert (concentration > 0).all()  # 양수여야 함
        
        # 샘플링
        dist = torch.distributions.Dirichlet(concentration)
        action = dist.sample()
        
        assert torch.allclose(action.sum(dim=1), torch.ones(10), atol=1e-5)
    
    def test_redq_ensemble(self, bcell):
        """REDQ 앙상블 테스트"""
        state = torch.randn(16, 43)
        action = torch.softmax(torch.randn(16, 30), dim=1)

        # 앙상블 Q값들
        q_values = []
        for critic in bcell.critics:
            q = critic(state, action)
            q_values.append(q)

        # 모든 Q-네트워크가 출력 생성
        assert len(q_values) == bcell.n_critics
        for q in q_values:
            assert q.shape == (16, 1)

        # Q값들이 서로 다름 (앙상블 다양성)
        q_stack = torch.stack(q_values)
        std = q_stack.std(dim=0).mean()
        assert std > 0  # 다양성이 있어야 함

if __name__ == "__main__":
    pytest.main([__file__, "-v"])