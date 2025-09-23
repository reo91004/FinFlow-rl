# src/baselines/standard_sac.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional
from tqdm import tqdm

from src.core.env import PortfolioEnv
from src.core.networks import DirichletActor, QNetwork
from src.core.replay import PrioritizedReplayBuffer, Transition
from src.data.loader import DataLoader
from src.data.features import FeatureExtractor
from src.utils.logger import FinFlowLogger
from src.utils.optimizer_utils import polyak_update, clip_gradients


class StandardSAC:
    """
    표준 SAC (Soft Actor-Critic)
    면역학적 메타포 없이 순수 SAC만 구현
    """

    def __init__(self, config: Dict):
        """
        Args:
            config: 설정 딕셔너리
        """
        self.config = config
        self.logger = FinFlowLogger("StandardSAC")
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

        # 데이터 로드
        self._load_data()

        # 차원 설정
        n_assets = len(self.price_data.columns)
        feature_dim = config.get('feature_dim', 12)
        self.state_dim = feature_dim + n_assets  # features + weights (위기 레벨 제외)
        self.action_dim = n_assets

        # 네트워크
        self._build_networks()

        # 리플레이 버퍼
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=config.get('buffer_size', 100000),
            alpha=0.6,
            beta=0.4
        )

        self.logger.info(f"Standard SAC 초기화 완료 (state_dim={self.state_dim}, action_dim={self.action_dim})")

    def _load_data(self):
        """데이터 로드"""
        loader = DataLoader(self.config.get('data', {}))
        self.price_data = loader.load()

        # 학습/테스트 분할
        n = len(self.price_data)
        train_end = int(n * 0.8)

        self.train_data = self.price_data[:train_end]
        self.test_data = self.price_data[train_end:]

        self.feature_extractor = FeatureExtractor()

    def _build_networks(self):
        """네트워크 구축"""
        hidden_dims = self.config.get('hidden_dims', [256, 256])

        # Actor
        self.actor = DirichletActor(
            self.state_dim, self.action_dim,
            hidden_dims=hidden_dims
        ).to(self.device)

        # Critics
        self.critic_1 = QNetwork(
            self.state_dim, self.action_dim,
            hidden_dims=hidden_dims
        ).to(self.device)

        self.critic_2 = QNetwork(
            self.state_dim, self.action_dim,
            hidden_dims=hidden_dims
        ).to(self.device)

        # Target critics
        self.critic_1_target = QNetwork(
            self.state_dim, self.action_dim,
            hidden_dims=hidden_dims
        ).to(self.device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())

        self.critic_2_target = QNetwork(
            self.state_dim, self.action_dim,
            hidden_dims=hidden_dims
        ).to(self.device)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        # Temperature
        self.log_alpha = torch.tensor(np.log(0.2), requires_grad=True, device=self.device)
        self.target_entropy = -self.action_dim

        # Optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=self.config.get('actor_lr', 3e-4)
        )

        self.critic_optimizer = optim.Adam(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            lr=self.config.get('critic_lr', 3e-4)
        )

        self.alpha_optimizer = optim.Adam(
            [self.log_alpha],
            lr=self.config.get('alpha_lr', 3e-4)
        )

        # Hyperparameters
        self.gamma = self.config.get('gamma', 0.99)
        self.tau = self.config.get('tau', 0.005)

    def train_step(self, batch_size: int = 256) -> Dict:
        """
        SAC 학습 스텝

        Args:
            batch_size: 배치 크기

        Returns:
            손실 딕셔너리
        """
        if len(self.replay_buffer) < batch_size:
            return {}

        # 배치 샘플링
        batch, weights, indices = self.replay_buffer.sample(batch_size)
        states = torch.FloatTensor(batch['states']).to(self.device)
        actions = torch.FloatTensor(batch['actions']).to(self.device)
        rewards = torch.FloatTensor(batch['rewards']).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(batch['next_states']).to(self.device)
        dones = torch.FloatTensor(batch['dones']).unsqueeze(1).to(self.device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

        # Critic 업데이트
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            target_q1 = self.critic_1_target(next_states, next_actions)
            target_q2 = self.critic_2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.log_alpha.exp() * next_log_probs
            target_value = rewards + (1 - dones) * self.gamma * target_q

        current_q1 = self.critic_1(states, actions)
        current_q2 = self.critic_2(states, actions)

        critic_1_loss = (weights * F.mse_loss(current_q1, target_value, reduction='none')).mean()
        critic_2_loss = (weights * F.mse_loss(current_q2, target_value, reduction='none')).mean()

        self.critic_optimizer.zero_grad()
        (critic_1_loss + critic_2_loss).backward()
        clip_gradients([self.critic_1, self.critic_2], max_norm=1.0)
        self.critic_optimizer.step()

        # Actor 업데이트
        actions_new, log_probs = self.actor.sample(states)
        q1_new = self.critic_1(states, actions_new)
        q2_new = self.critic_2(states, actions_new)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.log_alpha.exp() * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        clip_gradients(self.actor, max_norm=1.0)
        self.actor_optimizer.step()

        # Temperature 업데이트
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Target 네트워크 업데이트
        polyak_update(self.critic_1_target, self.critic_1, self.tau)
        polyak_update(self.critic_2_target, self.critic_2, self.tau)

        # PER 우선순위 업데이트
        td_error = (current_q1 - target_value).detach()
        new_priorities = td_error.abs().cpu().numpy().flatten() + 1e-6
        self.replay_buffer.update_priorities(indices, new_priorities)

        return {
            'critic_1_loss': critic_1_loss.item(),
            'critic_2_loss': critic_2_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.log_alpha.exp().item(),
            'entropy': -log_probs.mean().item()
        }

    def backtest(self, config: Dict) -> Dict:
        """
        백테스트 실행

        Args:
            config: 설정 딕셔너리

        Returns:
            백테스트 메트릭
        """
        self.logger.info("Standard SAC 학습 시작")

        # 환경 생성
        train_env = PortfolioEnv(
            price_data=self.train_data,
            feature_extractor=self.feature_extractor,
            config=config.get('env', {})
        )

        test_env = PortfolioEnv(
            price_data=self.test_data,
            feature_extractor=self.feature_extractor,
            config=config.get('env', {})
        )

        # 학습
        n_episodes = config.get('n_episodes', 100)
        for episode in tqdm(range(n_episodes), desc="SAC 학습"):
            state, _ = train_env.reset()
            # 위기 레벨 제거 (표준 SAC는 위기 감지 없음)
            if len(state) > self.state_dim:
                state = state[:self.state_dim]

            done = False
            while not done:
                # 행동 선택
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    action, _ = self.actor.sample(state_tensor)
                    action = action.cpu().numpy()[0]

                # 환경 스텝
                next_state, reward, done, truncated, _ = train_env.step(action)
                if len(next_state) > self.state_dim:
                    next_state = next_state[:self.state_dim]

                # 경험 저장
                self.replay_buffer.push(
                    Transition(state, action, reward, next_state, done or truncated)
                )

                # 학습
                if len(self.replay_buffer) > 1000:
                    self.train_step()

                state = next_state
                if done or truncated:
                    break

        # 테스트
        self.logger.info("백테스트 시작")
        returns = []
        portfolio_values = []
        initial_value = test_env.initial_balance

        for episode in range(10):
            state, _ = test_env.reset()
            if len(state) > self.state_dim:
                state = state[:self.state_dim]

            episode_return = 0
            done = False

            while not done:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    action = self.actor.get_action(state_tensor, deterministic=True)
                    action = action.cpu().numpy()[0]

                state, reward, done, truncated, info = test_env.step(action)
                if len(state) > self.state_dim:
                    state = state[:self.state_dim]

                episode_return += reward
                if 'portfolio_value' in info:
                    portfolio_values.append(info['portfolio_value'])

                if done or truncated:
                    break

            returns.append(episode_return)

        # 메트릭 계산
        returns = np.array(returns)
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)

        # MDD 계산
        if portfolio_values:
            portfolio_values = np.array(portfolio_values)
            running_max = np.maximum.accumulate(portfolio_values)
            drawdown = (portfolio_values - running_max) / running_max
            mdd = np.min(drawdown)
        else:
            mdd = 0

        metrics = {
            'sharpe': sharpe,
            'returns': np.mean(returns),
            'std': np.std(returns),
            'mdd': mdd,
            'episodes': n_episodes,
        }

        self.logger.info(f"백테스트 완료: Sharpe={sharpe:.3f}, Return={np.mean(returns):.4f}")

        return metrics