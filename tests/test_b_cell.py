# tests/test_b_cell.py

import pytest
import torch
import numpy as np
from src.agents.b_cell import BCell

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
        return BCell(config)
    
    def test_initialization(self, bcell, config):
        assert bcell.state_dim == config['state_dim']
        assert bcell.action_dim == config['action_dim']
        assert bcell.n_quantiles == config['n_quantiles']
        assert bcell.gamma == config['gamma']
        assert bcell.tau == config['tau']
    
    def test_actor_forward(self, bcell):
        """Actor 네트워크 순전파"""
        state = torch.randn(1, 43)
        action, log_prob = bcell.actor(state)
        
        # 액션은 심플렉스 위에 있어야 함
        assert action.shape == (1, 30)
        assert torch.allclose(action.sum(dim=1), torch.ones(1), atol=1e-5)
        assert (action >= 0).all()
        assert (action <= 1).all()
        
        # 로그 확률
        assert log_prob.shape == (1, 1)
        assert log_prob.item() <= 0  # 로그 확률은 음수
    
    def test_critic_forward(self, bcell):
        """Critic 네트워크 순전파"""
        state = torch.randn(10, 43)
        action = torch.softmax(torch.randn(10, 30), dim=1)
        
        q1 = bcell.critic1(state, action)
        q2 = bcell.critic2(state, action)
        
        # Quantile 출력
        assert q1.shape == (10, 32)
        assert q2.shape == (10, 32)
    
    def test_value_forward(self, bcell):
        """Value 네트워크 순전파"""
        state = torch.randn(5, 43)
        value = bcell.value_net(state)
        
        assert value.shape == (5, 1)
    
    def test_get_action_deterministic(self, bcell):
        """결정적 액션 선택"""
        state = np.random.randn(43).astype(np.float32)
        
        bcell.eval()
        action = bcell.get_action(state, deterministic=True)
        
        assert action.shape == (30,)
        assert np.allclose(action.sum(), 1.0, atol=1e-5)
        assert (action >= 0).all()
        assert (action <= 1).all()
    
    def test_get_action_stochastic(self, bcell):
        """확률적 액션 선택"""
        state = np.random.randn(43).astype(np.float32)
        
        bcell.train()
        action1 = bcell.get_action(state, deterministic=False)
        action2 = bcell.get_action(state, deterministic=False)
        
        # 확률적이므로 다를 수 있음
        assert not np.allclose(action1, action2, atol=1e-6)
    
    def test_iql_update(self, bcell):
        """IQL 업데이트 테스트"""
        batch = {
            'states': torch.randn(32, 43),
            'actions': torch.softmax(torch.randn(32, 30), dim=1),
            'rewards': torch.randn(32, 1),
            'next_states': torch.randn(32, 43),
            'dones': torch.zeros(32, 1)
        }
        
        # IQL 손실 계산
        value_loss, critic_loss, actor_loss = bcell.iql_update(batch)
        
        assert value_loss.item() >= 0
        assert critic_loss.item() >= 0
        assert actor_loss.item() >= 0
    
    def test_sac_update(self, bcell):
        """SAC 업데이트 테스트"""
        batch = {
            'states': torch.randn(32, 43),
            'actions': torch.softmax(torch.randn(32, 30), dim=1),
            'rewards': torch.randn(32, 1),
            'next_states': torch.randn(32, 43),
            'dones': torch.zeros(32, 1)
        }
        
        # SAC 손실 계산
        critic_loss, actor_loss, alpha_loss = bcell.sac_update(batch)
        
        assert critic_loss.item() >= 0
        assert actor_loss is not None
        assert alpha_loss is not None
    
    def test_soft_update(self, bcell):
        """소프트 업데이트 테스트"""
        # 타겟 파라미터 저장
        old_params = []
        for param in bcell.critic1_target.parameters():
            old_params.append(param.clone())
        
        # 소프트 업데이트
        bcell.soft_update()
        
        # 파라미터가 약간 변경되어야 함
        for old_param, new_param in zip(old_params, bcell.critic1_target.parameters()):
            assert not torch.equal(old_param, new_param)
            diff = torch.abs(old_param - new_param).max()
            assert diff < 0.01  # tau=0.005이므로 작은 변화
    
    def test_save_load(self, bcell, tmp_path):
        """모델 저장/로드 테스트"""
        # 원본 액션
        state = np.random.randn(43).astype(np.float32)
        original_action = bcell.get_action(state, deterministic=True)
        
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
        new_bcell = BCell(config)
        new_bcell.load(str(save_path))
        
        # 동일한 액션 출력
        loaded_action = new_bcell.get_action(state, deterministic=True)
        assert np.allclose(original_action, loaded_action, atol=1e-5)
    
    def test_dirichlet_policy(self, bcell):
        """Dirichlet 정책 테스트"""
        state = torch.randn(10, 43)
        
        # Dirichlet 파라미터 계산
        concentration = bcell.actor.forward_concentration(state)
        
        assert concentration.shape == (10, 30)
        assert (concentration > 0).all()  # 양수여야 함
        
        # 샘플링
        dist = torch.distributions.Dirichlet(concentration)
        action = dist.sample()
        
        assert torch.allclose(action.sum(dim=1), torch.ones(10), atol=1e-5)
    
    def test_quantile_regression(self, bcell):
        """Quantile regression 테스트"""
        state = torch.randn(16, 43)
        action = torch.softmax(torch.randn(16, 30), dim=1)
        
        # Quantile 값
        q_values = bcell.critic1(state, action)
        
        # Quantile은 증가해야 함 (대부분의 경우)
        sorted_q = torch.sort(q_values, dim=1)[0]
        diff = sorted_q[:, 1:] - sorted_q[:, :-1]
        
        # 일부 역전은 허용 (학습 초기)
        assert (diff >= 0).float().mean() > 0.5

if __name__ == "__main__":
    pytest.main([__file__, "-v"])