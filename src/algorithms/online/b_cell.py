# src/agents/b_cell.py

"""
B-Cell: 적응형 포트폴리오 전략 에이전트

목적: REDQ (Randomized Ensemble Double Q) 기반 온라인 학습
의존: networks.py, replay.py, optimizer_utils.py
사용처: FinFlowTrainer (메인 에이전트)
역할: T-Cell 신호에 적응하는 포트폴리오 전략 실행

구현 내용:
- 오프라인 가중치 로드 (IQL/TD3BC → BCell)
- 위기 수준에 따른 리스크 회피도 조정
- n_critics=5 앙상블로 안정성 확보
- UTD=20으로 샘플 효율성 극대화
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict

from src.models.networks import DirichletActor, QNetwork
from src.data.replay_buffer import PrioritizedReplayBuffer
from src.utils.logger import FinFlowLogger
from src.utils.training_utils import polyak_update, clip_gradients

class BCell:
    """
    B-Cell: 적응형 포트폴리오 전략 에이전트
    단일 에이전트가 T-Cell 신호에 따라 적응
    REDQ (Randomized Ensemble Double Q) 구현
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 config: Dict,
                 device: torch.device = torch.device("cpu")):
        """
        Args:
            state_dim: 상태 차원
            action_dim: 액션 차원 (포트폴리오 자산 수)
            config: 설정
            device: 연산 디바이스
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config  # config 저장
        self.device = device
        self.logger = FinFlowLogger("BCell")

        # Hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.005)
        self.alpha = config.get('alpha', 0.2)  # 엔트로피 계수
        self.target_entropy = -action_dim  # -dim(A)

        # REDQ settings
        self.n_critics = config.get('n_critics', 5)  # Q 앙상블 크기
        self.m_sample = config.get('m_sample', 2)    # 업데이트시 샘플 수
        self.utd_ratio = config.get('utd_ratio', 20) # Update-to-Data ratio

        # Networks
        self.actor = DirichletActor(
            state_dim, action_dim,
            hidden_dims=config.get('hidden_dims', [256, 256])
        ).to(device)

        # REDQ: Q 앙상블
        self.critics = nn.ModuleList([
            QNetwork(state_dim, action_dim, config.get('hidden_dims', [256, 256]))
            for _ in range(self.n_critics)
        ]).to(device)

        self.critics_target = nn.ModuleList([
            QNetwork(state_dim, action_dim, config.get('hidden_dims', [256, 256]))
            for _ in range(self.n_critics)
        ]).to(device)

        # 타겟 네트워크 초기화
        for critic, critic_target in zip(self.critics, self.critics_target):
            critic_target.load_state_dict(critic.state_dict())

        # Temperature parameter
        self.log_alpha = torch.tensor(np.log(self.alpha),
                                     requires_grad=True, device=device)

        # Optimizers (learning rate를 명시적으로 float 변환)
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=float(config.get('actor_lr', 3e-4))
        )

        self.critics_optimizer = optim.Adam(
            self.critics.parameters(),
            lr=float(config.get('critic_lr', 3e-4))
        )

        self.alpha_optimizer = optim.Adam(
            [self.log_alpha],
            lr=float(config.get('alpha_lr', 3e-4))
        )

        # Replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=config.get('buffer_size', 100000),
            alpha=0.6,
            beta=0.4
        )

        # Risk adaptation parameters
        self.base_risk_aversion = 1.0
        self.current_risk_aversion = 1.0
        self.crisis_threshold = config.get('crisis_threshold', 0.7)  # 위기 수준 임계값

        # Training statistics
        self.training_step = 0

        self.logger.info(f"B-Cell 초기화 완료: REDQ (n_critics={self.n_critics}, m_sample={self.m_sample}, UTD={self.utd_ratio})")

    def adapt_to_crisis(self, crisis_level: float):
        """
        T-Cell 신호에 따른 적응

        Args:
            crisis_level: 0 (정상) ~ 1 (극단 위기)
        """
        # 위기 수준에 따른 리스크 회피도 조정
        if crisis_level > self.crisis_threshold:
            self.current_risk_aversion = 2.0  # 방어적
            self.logger.debug(f"위기 수준 {crisis_level:.2f} - 방어적 모드 활성화")
        elif crisis_level > 0.4:
            self.current_risk_aversion = 1.5  # 중립
            self.logger.debug(f"위기 수준 {crisis_level:.2f} - 중립 모드")
        else:
            self.current_risk_aversion = 1.0  # 공격적
            self.logger.debug(f"위기 수준 {crisis_level:.2f} - 공격적 모드")

        # 엔트로피 조정 (위기시 탐험 감소)
        target_alpha = self.alpha * (2.0 - crisis_level)
        self.log_alpha.data = torch.log(torch.tensor(target_alpha, device=self.device))

    def select_action(self, state: np.ndarray,
                     crisis_level: float = 0.0,
                     deterministic: bool = False) -> np.ndarray:
        """
        행동 선택

        Args:
            state: 현재 상태
            crisis_level: T-Cell 위기 수준
            deterministic: 결정적 행동 여부
        """
        # 위기 적응
        self.adapt_to_crisis(crisis_level)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if deterministic:
                action, _ = self.actor.get_action(state_tensor, deterministic=True)
            else:
                action, _ = self.actor.sample(state_tensor)

            # 리스크 회피도 적용 (포트폴리오 집중도 조정)
            action = action.cpu().numpy()[0]

            if self.current_risk_aversion > 1.0:
                # 균등 가중치 방향으로 조정
                uniform = np.ones_like(action) / len(action)
                blend_ratio = 1.0 / self.current_risk_aversion
                action = blend_ratio * action + (1 - blend_ratio) * uniform
                action = action / action.sum()  # 재정규화

        return action

    def train(self, batch_size: int = None) -> Dict:
        """
        REDQ 학습 스텝
        Args:
            batch_size: 배치 크기 (None이면 config에서 가져옴)
        """
        if batch_size is None:
            batch_size = self.config.get('batch_size', 256)
        if len(self.replay_buffer) < batch_size:
            return {}

        losses = {}

        # High UTD ratio (REDQ 핵심)
        for utd_step in range(self.utd_ratio):
            batch, weights, indices = self.replay_buffer.sample(batch_size)
            states = torch.FloatTensor(batch['states']).to(self.device)
            actions = torch.FloatTensor(batch['actions']).to(self.device)
            rewards = torch.FloatTensor(batch['rewards']).unsqueeze(1).to(self.device)
            next_states = torch.FloatTensor(batch['next_states']).to(self.device)
            dones = torch.FloatTensor(batch['dones']).unsqueeze(1).to(self.device)
            weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

            # Critic 업데이트 (REDQ: 랜덤 서브셋)
            with torch.no_grad():
                next_actions, next_log_probs = self.actor.sample(next_states)

                # M개 랜덤 타겟 크리틱 선택
                target_indices = np.random.choice(
                    self.n_critics, self.m_sample, replace=False
                )

                target_q_values = []
                for idx in target_indices:
                    target_q = self.critics_target[idx](next_states, next_actions)
                    target_q_values.append(target_q)

                # 최소값 사용 (보수적 추정)
                min_target_q = torch.min(torch.stack(target_q_values), dim=0)[0]
                target_q = rewards + self.gamma * (1 - dones) * (
                    min_target_q - self.log_alpha.exp() * next_log_probs
                )

            # 모든 크리틱 업데이트
            critic_loss = 0
            td_errors = []
            for critic in self.critics:
                current_q = critic(states, actions)
                td_error = current_q - target_q
                loss = (weights * 0.5 * td_error.pow(2)).mean()
                critic_loss += loss
                td_errors.append(td_error.detach())

            critic_loss = critic_loss / self.n_critics

            self.critics_optimizer.zero_grad()
            critic_loss.backward()
            clip_gradients(self.critics, max_norm=1.0)
            self.critics_optimizer.step()

            # PER 우선순위 업데이트
            mean_td_error = torch.stack(td_errors).mean(dim=0)
            new_priorities = mean_td_error.abs().cpu().numpy().flatten() + 1e-6
            self.replay_buffer.update_priorities(indices, new_priorities)

            if utd_step == 0:
                losses['critic_loss'] = critic_loss.item()

        # Actor 업데이트 (낮은 빈도)
        if self.training_step % 2 == 0:
            actions_new, log_probs = self.actor.sample(states)

            # 모든 크리틱 평균
            q_values = []
            for critic in self.critics:
                q = critic(states, actions_new)
                q_values.append(q)
            mean_q = torch.stack(q_values).mean(dim=0)

            actor_loss = (self.log_alpha.exp() * log_probs - mean_q).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            clip_gradients(self.actor, max_norm=1.0)
            self.actor_optimizer.step()

            losses['actor_loss'] = actor_loss.item()

            # 온도 업데이트
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            losses['alpha'] = self.log_alpha.exp().item()
            losses['entropy'] = -log_probs.mean().item()

        # 타겟 네트워크 소프트 업데이트
        for critic, critic_target in zip(self.critics, self.critics_target):
            polyak_update(critic_target, critic, self.tau)

        self.training_step += 1

        return losses

    def load_from_offline(self, offline_agent):
        """
        오프라인 에이전트(IQL/TD3+BC)로부터 가중치 로드

        Args:
            offline_agent: 사전학습된 오프라인 에이전트
        """
        if hasattr(offline_agent, 'actor'):
            self.actor.load_state_dict(offline_agent.actor.state_dict())
            self.logger.info("오프라인 정책 로드 완료")

        if hasattr(offline_agent, 'critic') or hasattr(offline_agent, 'q_network'):
            # IQL의 경우 q_network, TD3+BC의 경우 critic
            source_critic = getattr(offline_agent, 'critic', None) or getattr(offline_agent, 'q_network', None)
            if source_critic:
                # 첫 번째 크리틱에만 로드, 나머지는 복사
                self.critics[0].load_state_dict(source_critic.state_dict())
                for i in range(1, self.n_critics):
                    # 약간의 노이즈 추가하여 다양성 확보
                    self.critics[i].load_state_dict(self.critics[0].state_dict())
                    for param in self.critics[i].parameters():
                        param.data += torch.randn_like(param.data) * 0.01

                # 타겟 네트워크도 업데이트
                for critic, critic_target in zip(self.critics, self.critics_target):
                    critic_target.load_state_dict(critic.state_dict())

                self.logger.info("오프라인 가치 함수 로드 완료")

    def save(self, path: str):
        """모델 저장"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critics': [c.state_dict() for c in self.critics],
            'critics_target': [c.state_dict() for c in self.critics_target],
            'log_alpha': self.log_alpha.data,
            'training_step': self.training_step,
        }, path)
        self.logger.info(f"B-Cell 모델 저장: {path}")

    def load(self, path: str):
        """모델 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])

        for i, critic in enumerate(self.critics):
            critic.load_state_dict(checkpoint['critics'][i])
        for i, critic_target in enumerate(self.critics_target):
            critic_target.load_state_dict(checkpoint['critics_target'][i])

        self.log_alpha.data = checkpoint['log_alpha']
        self.training_step = checkpoint['training_step']

        self.logger.info(f"B-Cell 모델 로드: {path}")

    def state_dict(self):
        """상태 딕셔너리 반환"""
        return {
            'actor': self.actor.state_dict(),
            'critics': [c.state_dict() for c in self.critics],
            'critics_target': [c.state_dict() for c in self.critics_target],
            'log_alpha': self.log_alpha.data,
            'training_step': self.training_step,
        }