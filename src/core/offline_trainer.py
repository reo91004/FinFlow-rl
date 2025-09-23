# src/core/offline_trainer.py

"""
오프라인 학습 조정자

목적: IQL 또는 TD3BC 오프라인 학습 관리
의존: iql.py, td3bc.py, offline_dataset.py
사용처: FinFlowTrainer._offline_pretrain()
역할: 오프라인 에이전트 학습 및 가중치 준비

구현 내용:
- 메서드별 update 인터페이스 통합 (IQL vs TD3BC)
- Early stopping 및 검증 손실 추적
- 학습 통계 모니터링
- 가중치 저장/로드
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional
from tqdm import tqdm
from src.core.iql import IQLAgent
from src.utils.logger import FinFlowLogger

class OfflineTrainer:
    """
    오프라인 RL 학습기
    IQL 또는 TD3+BC 선택 가능
    """

    def __init__(self,
                 method: str,
                 state_dim: int,
                 action_dim: int,
                 config: Dict,
                 device: torch.device):
        """
        Args:
            method: 'iql' 또는 'td3bc'
            state_dim: 상태 차원
            action_dim: 행동 차원
            config: 오프라인 설정
            device: 디바이스
        """
        self.method = method
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.device = device
        self.logger = FinFlowLogger(f"Offline-{method.upper()}")

        # 에이전트 초기화
        if method == 'iql':
            self.agent = IQLAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=config.get('hidden_dims', [256, 256])[0] if isinstance(config.get('hidden_dims', 256), list) else config.get('hidden_dims', 256),
                expectile=config.get('expectile', 0.7),
                temperature=config.get('temperature', 3.0),
                discount=config.get('gamma', 0.99),
                tau=config.get('tau', 0.005),
                learning_rate=config.get('critic_lr', 3e-4),
                device=device
            )
        elif method == 'td3bc':
            # TD3+BC 구현
            from src.core.td3bc import TD3BCAgent
            self.agent = TD3BCAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                config=config,
                device=device
            )
        else:
            raise ValueError(f"Unknown offline method: {method}")

        self.logger.info(f"{method.upper()} 오프라인 학습기 초기화 완료")

    def train(self, dataset: Dict, validation_data: Optional[Dict] = None) -> nn.Module:
        """
        오프라인 학습

        Args:
            dataset: 오프라인 데이터셋
            validation_data: 검증 데이터셋 (선택)

        Returns:
            학습된 에이전트
        """
        self.logger.info(f"{self.method.upper()} 오프라인 학습 시작")
        self.logger.info(f"데이터셋 크기: {len(dataset['states'])} 샘플")

        # 데이터 준비
        states = torch.FloatTensor(dataset['states']).to(self.device)
        actions = torch.FloatTensor(dataset['actions']).to(self.device)
        rewards = torch.FloatTensor(dataset['rewards']).to(self.device)
        next_states = torch.FloatTensor(dataset['next_states']).to(self.device)
        dones = torch.FloatTensor(dataset['dones']).to(self.device)

        dataset_size = len(states)
        batch_size = self.config.get('batch_size', 256)
        n_epochs = self.config.get('epochs', 100)

        # 학습 통계
        best_loss = float('inf')
        patience = self.config.get('early_stopping_patience', 10)
        patience_counter = 0

        # 학습 루프
        for epoch in range(n_epochs):
            # 배치 샘플링
            indices = np.random.permutation(dataset_size)

            epoch_losses = []
            batch_count = 0

            for i in range(0, dataset_size, batch_size):
                batch_indices = indices[i:min(i + batch_size, dataset_size)]

                batch = {
                    'states': states[batch_indices],
                    'actions': actions[batch_indices],
                    'rewards': rewards[batch_indices],
                    'next_states': next_states[batch_indices],
                    'dones': dones[batch_indices]
                }

                # 에이전트 업데이트 (메서드별 호출 방식)
                if self.method == 'iql':
                    # IQL은 개별 텐서를 받음
                    losses = self.agent.update(
                        batch['states'],
                        batch['actions'],
                        batch['rewards'],
                        batch['next_states'],
                        batch['dones']
                    )
                elif self.method == 'td3bc':
                    # TD3BC는 batch Dict를 받음
                    losses = self.agent.update(batch)
                else:
                    losses = {}
                epoch_losses.append(losses)
                batch_count += 1

            # 에폭 통계
            avg_losses = self._aggregate_losses(epoch_losses)

            # 검증 (선택적)
            if validation_data is not None and epoch % 5 == 0:
                val_loss = self._validate(validation_data)
                avg_losses['val_loss'] = val_loss

                # Early stopping
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break

            # 로깅
            if epoch % 10 == 0:
                log_str = f"Epoch {epoch}/{n_epochs}"
                for key, value in avg_losses.items():
                    log_str += f", {key}={value:.4f}"
                self.logger.info(log_str)

        self.logger.info("오프라인 학습 완료")
        return self.agent

    def _aggregate_losses(self, losses_list: list) -> Dict:
        """손실 집계"""
        if not losses_list:
            return {}

        aggregated = {}
        keys = losses_list[0].keys()

        for key in keys:
            values = [loss[key] for loss in losses_list if key in loss]
            if values:
                aggregated[key] = np.mean(values)

        return aggregated

    def _validate(self, validation_data: Dict) -> float:
        """검증 데이터로 평가"""
        states = torch.FloatTensor(validation_data['states']).to(self.device)
        actions = torch.FloatTensor(validation_data['actions']).to(self.device)
        rewards = torch.FloatTensor(validation_data['rewards']).to(self.device)
        next_states = torch.FloatTensor(validation_data['next_states']).to(self.device)
        dones = torch.FloatTensor(validation_data['dones']).to(self.device)

        batch_size = self.config.get('batch_size', 256)
        total_loss = 0
        n_batches = 0

        with torch.no_grad():
            for i in range(0, len(states), batch_size):
                batch_end = min(i + batch_size, len(states))
                batch = {
                    'states': states[i:batch_end],
                    'actions': actions[i:batch_end],
                    'rewards': rewards[i:batch_end],
                    'next_states': next_states[i:batch_end],
                    'dones': dones[i:batch_end]
                }

                # 검증 손실 계산
                if hasattr(self.agent, 'compute_validation_loss'):
                    loss = self.agent.compute_validation_loss(batch)
                else:
                    # 기본 MSE 손실
                    predicted_actions = self.agent.actor(batch['states'])
                    loss = ((predicted_actions - batch['actions']) ** 2).mean()

                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(n_batches, 1)

    def save(self, path: str):
        """학습된 모델 저장"""
        self.agent.save(path)
        self.logger.info(f"오프라인 모델 저장: {path}")

    def load(self, path: str):
        """모델 로드"""
        self.agent.load(path)
        self.logger.info(f"오프라인 모델 로드: {path}")

    def evaluate_on_env(self, env, n_episodes: int = 10) -> Dict:
        """환경에서 평가"""
        self.logger.info(f"환경 평가 시작: {n_episodes} 에피소드")

        episode_returns = []
        episode_lengths = []

        for episode in range(n_episodes):
            state, _ = env.reset()
            episode_return = 0
            episode_length = 0

            done = False
            while not done:
                # 행동 선택
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    action = self.agent.select_action(state_tensor, deterministic=True)
                    action = action.cpu().numpy()[0]

                # 환경 스텝
                next_state, reward, done, truncated, _ = env.step(action)
                episode_return += reward
                episode_length += 1
                state = next_state

                if done or truncated:
                    break

            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)

        # 통계 계산
        stats = {
            'mean_return': np.mean(episode_returns),
            'std_return': np.std(episode_returns),
            'max_return': np.max(episode_returns),
            'min_return': np.min(episode_returns),
            'mean_length': np.mean(episode_lengths),
        }

        self.logger.info(f"평가 완료: 평균 수익률={stats['mean_return']:.4f} ± {stats['std_return']:.4f}")

        return stats