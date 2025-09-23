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

from src.models.networks import DirichletActor, QNetwork, QuantileNetwork
from src.data.replay_buffer import PrioritizedReplayBuffer
from src.utils.logger import FinFlowLogger
from src.utils.training_utils import polyak_update, clip_gradients

class BCell:
    """
    B-Cell: 적응형 포트폴리오 전략 에이전트
    단일 에이전트가 T-Cell 신호에 따라 적응

    현재 구현:
    - REDQ (Randomized Ensemble Double Q) 알고리즘

    향후 확장 가능:
    - TQC (Truncated Quantile Critics) 알고리즘
    - config에서 'algorithm': 'REDQ' 또는 'TQC' 선택 가능
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

        # Algorithm selection
        self.algorithm_type = config.get('algorithm', 'REDQ')  # 'REDQ' or 'TQC'

        # Hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.005)
        self.alpha = config.get('alpha', 0.2)  # 엔트로피 계수
        self.target_entropy = -action_dim  # -dim(A)

        # Algorithm-specific settings
        if self.algorithm_type == 'TQC':
            # TQC settings - config에서 distributional 설정 가져오기
            dist_config = config.get('distributional', {})
            self.n_critics = config.get('n_critics', 2)  # TQC usually uses fewer critics
            self.n_quantiles = dist_config.get('n_quantiles', config.get('n_quantiles', 25))
            self.top_quantiles_to_drop = config.get('top_quantiles_to_drop_per_net', 2)
            self.quantile_embedding_dim = dist_config.get('quantile_embedding_dim', 64)
            self.risk_measure = dist_config.get('risk_measure', 'cvar')
            self.risk_sensitivity = dist_config.get('risk_sensitivity', 0.05)
            self.utd_ratio = config.get('utd_ratio', 1)  # TQC uses standard update frequency
        else:
            # REDQ settings
            self.n_critics = config.get('n_critics', 5)  # Q 앙상블 크기
            self.m_sample = config.get('m_sample', 2)    # 업데이트시 샘플 수
            self.utd_ratio = config.get('utd_ratio', 20) # Update-to-Data ratio

        # Networks
        # Dirichlet 설정 가져오기
        dirichlet_config = config.get('dirichlet', {})
        self.actor = DirichletActor(
            state_dim, action_dim,
            hidden_dims=config.get('hidden_dims', [256, 256]),
            min_concentration=dirichlet_config.get('min_concentration', 1.0),
            max_concentration=dirichlet_config.get('max_concentration', 50.0),
            base_concentration=dirichlet_config.get('base_concentration', 1.5),
            dynamic_concentration=dirichlet_config.get('dynamic_concentration', True),
            crisis_scaling=dirichlet_config.get('crisis_scaling', 0.5),
            action_smoothing=dirichlet_config.get('action_smoothing', True),
            smoothing_alpha=dirichlet_config.get('smoothing_alpha', 0.95)
        ).to(device)

        # Critic networks based on algorithm
        if self.algorithm_type == 'TQC':
            # TQC: Quantile critics
            self.critics = nn.ModuleList([
                QuantileNetwork(
                    state_dim, action_dim,
                    n_quantiles=self.n_quantiles,
                    hidden_dims=config.get('hidden_dims', [256, 256]),
                    quantile_embedding_dim=self.quantile_embedding_dim
                )
                for _ in range(self.n_critics)
            ]).to(device)

            self.critics_target = nn.ModuleList([
                QuantileNetwork(
                    state_dim, action_dim,
                    n_quantiles=self.n_quantiles,
                    hidden_dims=config.get('hidden_dims', [256, 256]),
                    quantile_embedding_dim=self.quantile_embedding_dim
                )
                for _ in range(self.n_critics)
            ]).to(device)
        else:
            # REDQ: Standard Q networks
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

        # 위기 수준 로깅 통계
        self.crisis_stats = {
            'count': 0,
            'sum': 0.0,
            'min': 1.0,
            'max': 0.0,
            'last_logged_step': 0,
            'log_interval': 100,  # 100스텝마다 통계 로그
            'mode_changes': 0,
            'current_mode': None,  # '공격적', '중립', '방어적'
            'mode_history': []  # (step, mode) 튜플 리스트
        }

        if self.algorithm_type == 'TQC':
            self.logger.info(f"B-Cell 초기화 완료: TQC (n_critics={self.n_critics}, n_quantiles={self.n_quantiles}, top_drop={self.top_quantiles_to_drop})")
        else:
            self.logger.info(f"B-Cell 초기화 완료: REDQ (n_critics={self.n_critics}, m_sample={self.m_sample}, UTD={self.utd_ratio})")

    def _log_crisis_statistics(self):
        """위기 수준 통계 로그 출력"""
        if self.crisis_stats['count'] > 0:
            avg_crisis = self.crisis_stats['sum'] / self.crisis_stats['count']
            self.logger.info(
                f"위기 수준 통계 (최근 {self.crisis_stats['log_interval']}스텝): "
                f"평균={avg_crisis:.3f}, 최소={self.crisis_stats['min']:.3f}, "
                f"최대={self.crisis_stats['max']:.3f}, 현재 모드={self.crisis_stats['current_mode']}, "
                f"모드 변경 횟수={self.crisis_stats['mode_changes']}"
            )
            # 통계 리셋 (모드 정보는 유지)
            self.crisis_stats['min'] = 1.0
            self.crisis_stats['max'] = 0.0

    def get_crisis_summary(self) -> Dict:
        """위기 대응 요약 정보 반환"""
        if self.crisis_stats['count'] == 0:
            return {}

        avg_crisis = self.crisis_stats['sum'] / self.crisis_stats['count']
        mode_distribution = {}

        # 모드 히스토리에서 분포 계산
        if len(self.crisis_stats['mode_history']) > 1:
            for i in range(len(self.crisis_stats['mode_history']) - 1):
                mode = self.crisis_stats['mode_history'][i][1]
                duration = self.crisis_stats['mode_history'][i+1][0] - self.crisis_stats['mode_history'][i][0]
                mode_distribution[mode] = mode_distribution.get(mode, 0) + duration

        return {
            '평균_위기수준': avg_crisis,
            '현재_모드': self.crisis_stats['current_mode'],
            '모드_변경횟수': self.crisis_stats['mode_changes'],
            '모드_분포': mode_distribution
        }

    def adapt_to_crisis(self, crisis_level: float):
        """
        T-Cell 신호에 따른 적응

        Args:
            crisis_level: 0 (정상) ~ 1 (극단 위기)
        """
        # 통계 수집
        self.crisis_stats['count'] += 1
        self.crisis_stats['sum'] += crisis_level
        self.crisis_stats['min'] = min(self.crisis_stats['min'], crisis_level)
        self.crisis_stats['max'] = max(self.crisis_stats['max'], crisis_level)

        # 위기 수준에 따른 리스크 회피도 조정 및 모드 결정
        prev_mode = self.crisis_stats['current_mode']

        if crisis_level > self.crisis_threshold:
            self.current_risk_aversion = 2.0  # 방어적
            new_mode = '방어적'
        elif crisis_level > 0.4:
            self.current_risk_aversion = 1.5  # 중립
            new_mode = '중립'
        else:
            self.current_risk_aversion = 1.0  # 공격적
            new_mode = '공격적'

        # 모드가 변경될 때만 로그
        if prev_mode != new_mode:
            self.crisis_stats['current_mode'] = new_mode
            self.crisis_stats['mode_changes'] += 1
            self.crisis_stats['mode_history'].append((self.training_step, new_mode))
            self.logger.info(f"위기 대응 모드 변경: {prev_mode} → {new_mode} (위기 수준: {crisis_level:.3f})")

        # 주기적으로 통계 로그 출력
        if self.crisis_stats['count'] - self.crisis_stats['last_logged_step'] >= self.crisis_stats['log_interval']:
            self._log_crisis_statistics()
            self.crisis_stats['last_logged_step'] = self.crisis_stats['count']

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
        학습 스텝 (REDQ 또는 TQC)
        Args:
            batch_size: 배치 크기 (None이면 config에서 가져옴)
        """
        if batch_size is None:
            batch_size = self.config.get('batch_size', 256)
        if len(self.replay_buffer) < batch_size:
            return {}

        # Algorithm-specific training
        if self.algorithm_type == 'TQC':
            return self._train_tqc(batch_size)
        else:
            return self._train_redq(batch_size)

    def _train_redq(self, batch_size: int) -> Dict:
        """
        REDQ 학습 스텝
        """

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

    def _train_tqc(self, batch_size: int) -> Dict:
        """
        TQC (Truncated Quantile Critics) 학습 스텝
        """
        losses = {}

        # Standard update frequency for TQC (not high UTD like REDQ)
        for utd_step in range(self.utd_ratio):
            batch, weights, indices = self.replay_buffer.sample(batch_size)
            states = torch.FloatTensor(batch['states']).to(self.device)
            actions = torch.FloatTensor(batch['actions']).to(self.device)
            rewards = torch.FloatTensor(batch['rewards']).unsqueeze(1).to(self.device)
            next_states = torch.FloatTensor(batch['next_states']).to(self.device)
            dones = torch.FloatTensor(batch['dones']).unsqueeze(1).to(self.device)
            weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

            # Critic update with quantile regression
            with torch.no_grad():
                next_actions, next_log_probs = self.actor.sample(next_states)

                # Get quantile values from all target critics
                all_target_quantiles = []
                for critic_target in self.critics_target:
                    # Get truncated quantiles (drop top k)
                    truncated = critic_target.get_truncated_quantiles(
                        next_states, next_actions,
                        top_quantiles_to_drop=self.top_quantiles_to_drop
                    )
                    all_target_quantiles.append(truncated)

                # Concatenate all truncated quantiles
                cat_target_quantiles = torch.cat(all_target_quantiles, dim=-1)

                # Take mean across all quantiles as target
                mean_target_q = cat_target_quantiles.mean(dim=-1, keepdim=True)

                # Compute target with entropy
                target_value = rewards + self.gamma * (1 - dones) * (
                    mean_target_q - self.log_alpha.exp() * next_log_probs
                )

            # Update each critic with quantile loss
            critic_loss = 0
            td_errors = []

            for critic in self.critics:
                # Get current quantile predictions
                current_quantiles = critic(states, actions)  # [batch_size, n_quantiles]

                # Expand target for all quantiles
                target_expanded = target_value.expand(-1, self.n_quantiles)

                # Compute quantile Huber loss
                loss = self.quantile_huber_loss(
                    current_quantiles,
                    target_expanded,
                    weights=weights
                )
                critic_loss += loss

                # TD error for prioritized replay
                with torch.no_grad():
                    td_error = (current_quantiles.mean(dim=-1, keepdim=True) - target_value).abs()
                    td_errors.append(td_error)

            # Average critic loss
            critic_loss = critic_loss / self.n_critics

            # Optimize critics
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            clip_gradients(self.critics, max_norm=10.0)
            self.critic_optimizer.step()

            # Update priorities in replay buffer
            if hasattr(self.replay_buffer, 'update_priorities'):
                mean_td_error = torch.stack(td_errors).mean(dim=0).squeeze().cpu().numpy()
                self.replay_buffer.update_priorities(indices, mean_td_error)

            losses['critic_loss'] = critic_loss.item()
            losses['mean_td_error'] = mean_td_error.mean().item() if 'mean_td_error' in locals() else 0

            # Actor update (same as REDQ)
            if self.training_step % 2 == 0:  # Less frequent actor update
                actions_pred, log_probs = self.actor.sample(states)

                # Use minimum quantile values for actor loss
                min_q_values = []
                for critic in self.critics:
                    q_quantiles = critic(states, actions_pred)
                    min_q = q_quantiles.mean(dim=-1, keepdim=True)  # Mean over quantiles
                    min_q_values.append(min_q)

                min_q = torch.min(torch.stack(min_q_values), dim=0)[0]
                actor_loss = (self.log_alpha.exp() * log_probs - min_q).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                clip_gradients([self.actor], max_norm=10.0)
                self.actor_optimizer.step()

                losses['actor_loss'] = actor_loss.item()

                # Temperature update
                alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

                losses['alpha'] = self.log_alpha.exp().item()
                losses['entropy'] = -log_probs.mean().item()

        # Soft target update
        for critic, critic_target in zip(self.critics, self.critics_target):
            polyak_update(critic_target, critic, self.tau)

        self.training_step += 1

        return losses

    def quantile_huber_loss(self, quantiles: torch.Tensor, targets: torch.Tensor,
                           weights: torch.Tensor = None, kappa: float = 1.0) -> torch.Tensor:
        """
        Quantile Huber loss for TQC

        Args:
            quantiles: Predicted quantiles [batch_size, n_quantiles]
            targets: Target values [batch_size, n_quantiles]
            weights: Importance sampling weights [batch_size, 1]
            kappa: Huber loss threshold

        Returns:
            loss: Scalar loss value
        """
        # Get quantile fractions
        if self.algorithm_type == 'TQC' and hasattr(self.critics[0], 'quantile_fractions'):
            tau = self.critics[0].quantile_fractions.unsqueeze(0)  # [1, n_quantiles]
        else:
            # Fallback to uniform quantiles
            n_quantiles = quantiles.shape[-1]
            tau = torch.linspace(0, 1, n_quantiles + 1, device=self.device)[1:]
            tau = tau[:-1] + tau.diff() / 2
            tau = tau.unsqueeze(0)

        # Compute TD errors
        td_errors = targets - quantiles  # [batch_size, n_quantiles]

        # Huber loss
        huber_loss = torch.where(
            td_errors.abs() <= kappa,
            0.5 * td_errors.pow(2),
            kappa * (td_errors.abs() - 0.5 * kappa)
        )

        # Quantile regression loss
        quantile_loss = torch.abs(tau - (td_errors < 0).float()) * huber_loss

        # Apply importance sampling weights if provided
        if weights is not None:
            quantile_loss = weights * quantile_loss.mean(dim=-1, keepdim=True)
            return quantile_loss.mean()
        else:
            return quantile_loss.mean()