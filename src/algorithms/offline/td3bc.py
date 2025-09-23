# src/core/td3bc.py

"""
TD3+BC (Twin Delayed DDPG + Behavior Cloning)

목적: 오프라인 RL with 행동 복제 정규화
의존: networks.py, optimizer_utils.py
사용처: OfflineTrainer → BCell 가중치 전이
역할: IQL 대안 오프라인 학습 방법

구현 내용:
- BC weight로 데이터 정책 유지
- Twin critics로 Q 과대추정 방지
- 지연된 정책 업데이트
- Q 정규화로 온라인 미세조정 준비
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
from src.models.networks import DirichletActor, QNetwork
from src.utils.logger import FinFlowLogger
from src.utils.training_utils import polyak_update, clip_gradients

class TD3BCAgent:
    """
    TD3+BC: Twin Delayed DDPG with Behavior Cloning
    오프라인 강화학습을 위한 TD3 + BC 정규화
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 config: Dict,
                 device: torch.device = torch.device("cpu")):
        """
        Args:
            state_dim: 상태 차원
            action_dim: 행동 차원
            config: TD3+BC 설정
            device: 디바이스
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.logger = FinFlowLogger("TD3BC")

        # Hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.005)
        self.policy_delay = config.get('policy_delay', 2)
        self.bc_weight = config.get('bc_weight', 2.5)  # BC 정규화 가중치
        self.normalize_q = config.get('normalize_q', True)  # Q 정규화 여부

        # Actor
        self.actor = DirichletActor(
            state_dim, action_dim,
            hidden_dims=config.get('hidden_dims', [256, 256])
        ).to(device)

        self.actor_target = DirichletActor(
            state_dim, action_dim,
            hidden_dims=config.get('hidden_dims', [256, 256])
        ).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Twin Critics
        self.critic_1 = QNetwork(
            state_dim, action_dim,
            hidden_dims=config.get('hidden_dims', [256, 256])
        ).to(device)

        self.critic_2 = QNetwork(
            state_dim, action_dim,
            hidden_dims=config.get('hidden_dims', [256, 256])
        ).to(device)

        self.critic_1_target = QNetwork(
            state_dim, action_dim,
            hidden_dims=config.get('hidden_dims', [256, 256])
        ).to(device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())

        self.critic_2_target = QNetwork(
            state_dim, action_dim,
            hidden_dims=config.get('hidden_dims', [256, 256])
        ).to(device)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=config.get('actor_lr', 3e-4)
        )

        self.critic_optimizer = optim.Adam(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            lr=config.get('critic_lr', 3e-4)
        )

        # Training statistics
        self.total_iterations = 0

        self.logger.info(f"TD3+BC 초기화 완료 (bc_weight={self.bc_weight})")

    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> np.ndarray:
        """
        행동 선택

        Args:
            state: 상태 텐서
            deterministic: 결정적 행동 여부

        Returns:
            행동 배열
        """
        with torch.no_grad():
            if deterministic:
                action, _ = self.actor.get_action(state, deterministic=True)
            else:
                action, _ = self.actor.sample(state)

            return action.cpu().numpy()

    def update(self, batch: Dict) -> Dict:
        """
        TD3+BC 업데이트

        Args:
            batch: 학습 배치

        Returns:
            손실 딕셔너리
        """
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards'].unsqueeze(1)
        next_states = batch['next_states']
        dones = batch['dones'].unsqueeze(1)

        losses = {}

        # Q 정규화를 위한 통계 (선택적)
        if self.normalize_q and self.total_iterations == 0:
            with torch.no_grad():
                random_actions, _ = self.actor.sample(states)
                q1_random = self.critic_1(states, random_actions)
                q2_random = self.critic_2(states, random_actions)
                q_random = torch.min(q1_random, q2_random)
                self.q_mean = q_random.mean().item()
                self.q_std = q_random.std().item() + 1e-6

        # Critic 업데이트
        with torch.no_grad():
            # 타겟 정책으로 다음 행동 선택
            next_actions, _ = self.actor_target.get_action(next_states, deterministic=True)

            # 타겟 Q값 계산
            target_q1 = self.critic_1_target(next_states, next_actions)
            target_q2 = self.critic_2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.gamma * target_q

        # 현재 Q값
        current_q1 = self.critic_1(states, actions)
        current_q2 = self.critic_2(states, actions)

        # Critic 손실
        critic_1_loss = F.mse_loss(current_q1, target_q)
        critic_2_loss = F.mse_loss(current_q2, target_q)
        critic_loss = critic_1_loss + critic_2_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        clip_gradients(self.critic_1, max_norm=1.0)
        clip_gradients(self.critic_2, max_norm=1.0)
        self.critic_optimizer.step()

        losses['critic_1_loss'] = critic_1_loss.item()
        losses['critic_2_loss'] = critic_2_loss.item()

        # Actor 업데이트 (지연된 정책 업데이트)
        if self.total_iterations % self.policy_delay == 0:
            # 정책 행동
            policy_actions, _ = self.actor.get_action(states, deterministic=True)

            # Q값 계산
            q1 = self.critic_1(states, policy_actions)
            q2 = self.critic_2(states, policy_actions)
            q = torch.min(q1, q2)

            # Q 정규화 (선택적)
            if self.normalize_q:
                q = (q - self.q_mean) / self.q_std

            # BC 손실 (데이터셋 행동과의 차이)
            bc_loss = F.mse_loss(policy_actions, actions)

            # TD3+BC 손실: -Q + λ * BC
            lmbda = self.bc_weight / q.abs().mean().detach()
            actor_loss = -q.mean() + lmbda * bc_loss

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            clip_gradients(self.actor, max_norm=1.0)
            self.actor_optimizer.step()

            # 타겟 네트워크 소프트 업데이트
            polyak_update(self.actor_target, self.actor, self.tau)
            polyak_update(self.critic_1_target, self.critic_1, self.tau)
            polyak_update(self.critic_2_target, self.critic_2, self.tau)

            losses['actor_loss'] = actor_loss.item()
            losses['bc_loss'] = bc_loss.item()
            losses['q_mean'] = q.mean().item()
            losses['lambda'] = lmbda.item()

        self.total_iterations += 1

        return losses

    def compute_validation_loss(self, batch: Dict) -> torch.Tensor:
        """
        검증 손실 계산

        Args:
            batch: 검증 배치

        Returns:
            검증 손실
        """
        states = batch['states']
        actions = batch['actions']

        with torch.no_grad():
            # 정책 행동
            policy_actions, _ = self.actor.get_action(states, deterministic=True)

            # BC 손실
            bc_loss = F.mse_loss(policy_actions, actions)

            # Q값 차이 (선택적)
            q1_policy = self.critic_1(states, policy_actions)
            q1_data = self.critic_1(states, actions)
            q_diff = (q1_policy - q1_data).abs().mean()

            # 종합 검증 손실
            val_loss = bc_loss + 0.1 * q_diff

        return val_loss

    def save(self, path: str):
        """모델 저장"""
        torch.save({
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_1': self.critic_1.state_dict(),
            'critic_2': self.critic_2.state_dict(),
            'critic_1_target': self.critic_1_target.state_dict(),
            'critic_2_target': self.critic_2_target.state_dict(),
            'total_iterations': self.total_iterations,
        }, path)
        self.logger.info(f"TD3+BC 모델 저장: {path}")

    def load(self, path: str):
        """모델 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_1.load_state_dict(checkpoint['critic_1'])
        self.critic_2.load_state_dict(checkpoint['critic_2'])
        self.critic_1_target.load_state_dict(checkpoint['critic_1_target'])
        self.critic_2_target.load_state_dict(checkpoint['critic_2_target'])
        self.total_iterations = checkpoint['total_iterations']
        self.logger.info(f"TD3+BC 모델 로드: {path}")

    def state_dict(self):
        """상태 딕셔너리 반환"""
        return {
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_1': self.critic_1.state_dict(),
            'critic_2': self.critic_2.state_dict(),
            'critic_1_target': self.critic_1_target.state_dict(),
            'critic_2_target': self.critic_2_target.state_dict(),
            'total_iterations': self.total_iterations,
        }