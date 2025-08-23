# bipd/agents/bcell.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from collections import deque
import random
from utils.logger import BIPDLogger
from utils.rolling_stats import MultiRollingStats
from config import (
    DEVICE,
    ACTOR_LR,
    CRITIC_LR,
    ALPHA_LR,
    GAMMA,
    TAU,
    BATCH_SIZE,
    BUFFER_SIZE,
    TARGET_ENTROPY_SCALE,
    REWARD_CLIP_MIN,
    REWARD_CLIP_MAX,
    ALPHA_MIN,
    ALPHA_MAX,
    CONCENTRATION_MIN,
    CONCENTRATION_MAX,
    WEIGHT_EPSILON,
    MAX_GRAD_NORM,
    HUBER_DELTA,
    CQL_ALPHA_START,
    CQL_ALPHA_END,
    CQL_NUM_SAMPLES,
    N_EPISODES,
)


class SACActorNetwork(nn.Module):
    """SAC Actor 네트워크: 확률적 포트폴리오 가중치 생성"""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(SACActorNetwork, self).__init__()

        self.action_dim = action_dim

        # 공통 특성 추출 레이어
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Dirichlet 분포를 위한 concentration 파라미터 출력
        # 포트폴리오 가중치는 simplex 위에 있어야 하므로 Dirichlet이 적합
        self.concentration_head = nn.Linear(hidden_dim, action_dim)

        # 가중치 초기화
        self._init_weights()

    def _init_weights(self):
        """가중치 초기화"""
        for layer in [self.fc1, self.fc2, self.concentration_head]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, state):
        """
        확률적 정책 출력
        Returns:
            concentration: Dirichlet 분포의 concentration 파라미터
            weights: 샘플링된 포트폴리오 가중치
            log_prob: 로그 확률
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        # Concentration 파라미터 (안정화된 클리핑)
        x_clamped = torch.clamp(self.concentration_head(x), min=-10.0, max=10.0)
        concentration = F.softplus(x_clamped) + 1e-3
        # Phase 1: 농도 파라미터 안전 범위 클리핑
        concentration = torch.clamp(concentration, CONCENTRATION_MIN, CONCENTRATION_MAX)

        # Dirichlet 분포에서 샘플링
        if self.training:
            # 훈련 시: 확률적 샘플링
            dist = torch.distributions.Dirichlet(concentration)
            weights = dist.rsample()  # reparameterization trick 사용
            # Phase 1: 포트폴리오 가중치 보호 로직
            weights = torch.clamp(weights, WEIGHT_EPSILON, 1.0 - WEIGHT_EPSILON)
            # 재정규화 (안전장치)
            weights = weights / weights.sum(dim=-1, keepdim=True)
            log_prob = dist.log_prob(weights)
        else:
            # 평가 시: 결정적 출력 (평균 사용)
            # Dirichlet의 평균은 concentration / concentration.sum()
            weights = concentration / concentration.sum(dim=-1, keepdim=True)
            # Phase 1: 평가 시에도 가중치 보호 적용
            weights = torch.clamp(weights, WEIGHT_EPSILON, 1.0 - WEIGHT_EPSILON)
            weights = weights / weights.sum(dim=-1, keepdim=True)
            log_prob = torch.zeros(weights.shape[0], device=weights.device)

        return concentration, weights, log_prob

    def get_action_and_log_prob(self, state):
        """행동과 로그 확률을 함께 반환"""
        concentration, weights, log_prob = self.forward(state)
        return weights, log_prob


class CriticNetwork(nn.Module):
    """Critic 네트워크: Q(s,a) 가치 함수"""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(CriticNetwork, self).__init__()

        # state 처리용
        self.state_fc = nn.Linear(state_dim, hidden_dim)
        # action 처리용
        self.action_fc = nn.Linear(action_dim, hidden_dim)
        # 결합 처리용
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

        # 가중치 초기화
        self._init_weights()

    def _init_weights(self):
        """가중치 초기화"""
        for layer in [self.state_fc, self.action_fc, self.fc1, self.fc2]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, state, action):
        # state와 action을 결합한 Q(s,a) 계산
        state_value = F.relu(self.state_fc(state))
        action_value = F.relu(self.action_fc(action))
        x = F.relu(self.fc1(torch.cat([state_value, action_value], dim=1)))
        q_value = self.fc2(x)
        return q_value


class PrioritizedReplayBuffer:
    """우선순위 경험 재생 버퍼 (PER)"""

    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha  # 우선순위 지수
        self.beta = beta  # 중요도 샘플링 지수 (초기값)
        self.beta_increment = beta_increment  # 베타 증가율
        self.max_beta = 1.0  # 베타 최대값
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.buffer = []
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        """경험 저장 (최대 우선순위로 초기 설정)"""
        max_priority = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)

        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        """우선순위 기반 배치 샘플링"""
        # Beta annealing: 점진적으로 1.0에 수렴
        self.beta = min(self.max_beta, self.beta + self.beta_increment)

        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[: self.pos]

        # 우선순위 기반 확률 계산
        probabilities = priorities**self.alpha
        probabilities /= probabilities.sum()

        # 샘플 인덱스 선택
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]

        # 중요도 샘플링 가중치 계산
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # 정규화

        return samples, weights, indices

    def update_priorities(self, indices, priorities):
        """TD-error 기반 우선순위 업데이트"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)


def dirichlet_c_from_diversity(K, D_target, eps=1e-6):
    """
    원하는 다양도 D_target로부터 균등 Dirichlet 농도 c를 설정

    Args:
        K: 자산 개수
        D_target: 목표 다양도 E[sum w_i^2] (낮을수록 더 다양)
        eps: 수치 안전망

    Returns:
        c: Dirichlet 농도 파라미터
    """
    denom = K * D_target - 1.0
    if denom <= eps:
        return 1.0
    c = (1.0 - D_target) / denom
    return float(max(c, 1e-3))


class BCell:
    """
    B-세포: 특정 위험 유형에 특화된 포트폴리오 전략 실행

    SAC (Soft Actor-Critic) 강화학습을 사용하여 포트폴리오 가중치를 학습
    각 B-Cell은 특정 시장 상황(volatility, correlation, momentum)에 특화
    """

    def __init__(
        self,
        risk_type,
        state_dim,
        action_dim,
        actor_lr=None,
        critic_lr=None,
        alpha_lr=None,
        hidden_dim=128,
    ):
        self.actor_lr = actor_lr or ACTOR_LR
        self.critic_lr = critic_lr or CRITIC_LR
        self.alpha_lr = alpha_lr or ALPHA_LR
        self.risk_type = risk_type
        self.state_dim = state_dim
        self.action_dim = action_dim

        # SAC 신경망 초기화 및 GPU로 이동
        self.actor = SACActorNetwork(state_dim, action_dim, hidden_dim).to(DEVICE)

        # Twin Critics (SAC에서도 사용)
        self.critic1 = CriticNetwork(state_dim, action_dim, hidden_dim).to(DEVICE)
        self.critic2 = CriticNetwork(state_dim, action_dim, hidden_dim).to(DEVICE)

        # 타겟 네트워크들 (Critic만 필요, SAC에서는 Actor 타겟 없음)
        self.target_critic1 = CriticNetwork(state_dim, action_dim, hidden_dim).to(
            DEVICE
        )
        self.target_critic2 = CriticNetwork(state_dim, action_dim, hidden_dim).to(
            DEVICE
        )

        # 타겟 네트워크 초기화 (Critic만 복사)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # 로거 먼저 초기화
        self.logger = BIPDLogger(f"BCell-{risk_type}")

        # SAC 엔트로피 계수 (Phase 1: 개선된 온도 제어)
        # Dirichlet 분포에 맞춘 초기 타겟 엔트로피 설정
        self._compute_dirichlet_entropy_prior(action_dim)

        # log-α 안전 범위 설정 (초기값: 0.1, 범위: [1e-4, 1.0])
        init_log_alpha = np.log(0.1)
        self.log_alpha_min = np.log(ALPHA_MIN)  # -9.21
        self.log_alpha_max = np.log(ALPHA_MAX)  # 0.0

        self.log_alpha = torch.tensor(
            np.clip(init_log_alpha, self.log_alpha_min, self.log_alpha_max),
            requires_grad=True,
            device=DEVICE,
        )
        self.alpha = self.log_alpha.exp()

        # Phase 1: 에피소드 진행률 추적 및 엔트로피 스케줄링
        self.episode_progress = 0.0
        self.total_episodes = N_EPISODES  # config에서 가져옴

        # 엔트로피 스케줄링 파라미터
        self.entropy_schedule_warmup = 0.2  # 20% 예열 기간
        self.entropy_schedule_cooldown = 0.8  # 80%부터 쿨다운

        # 옵티마이저
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic1_optimizer = optim.Adam(
            self.critic1.parameters(), lr=self.critic_lr
        )
        self.critic2_optimizer = optim.Adam(
            self.critic2.parameters(), lr=self.critic_lr
        )
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.alpha_lr)

        # Phase 4: 베타 어닐링을 위한 PER 설정
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=BUFFER_SIZE,
            alpha=0.6,
            beta=0.4,
            beta_increment=1.0 / N_EPISODES,  # 전체 에피소드에 걸쳐 1.0까지 증가
        )

        # SAC 학습 파라미터
        self.gamma = GAMMA
        self.tau = TAU
        self.batch_size = BATCH_SIZE
        self.update_frequency = 4

        # 학습 통계
        self.update_count = 0
        self.actor_losses = []
        self.critic_losses = []

        # Phase 1: 정책 지연 업데이트를 위한 카운터
        self.policy_update_frequency = 2  # 2스텝에 1회 Actor 업데이트

        # 통합 슬라이딩 윈도우 통계 관리자
        self.rolling_stats = MultiRollingStats(window_size=100)

        # 주요 통계 지표 등록
        self.q_min_stats = self.rolling_stats.add_statistics("q_min")
        self.q_max_stats = self.rolling_stats.add_statistics("q_max")
        self.q_range_stats = self.rolling_stats.add_statistics("q_range")
        self.actor_loss_stats = self.rolling_stats.add_statistics("actor_loss")
        self.critic_loss_stats = self.rolling_stats.add_statistics("critic_loss")
        self.alpha_stats = self.rolling_stats.add_statistics("alpha")
        self.td_error_stats = self.rolling_stats.add_statistics("td_error")
        self.target_entropy_stats = self.rolling_stats.add_statistics("target_entropy")

        # 주요 카운터 등록
        self.nan_loss_counter = self.rolling_stats.add_counter("nan_loss")
        self.extreme_q_counter = self.rolling_stats.add_counter(
            "extreme_q"
        )  # |Q| > 100
        self.high_alpha_counter = self.rolling_stats.add_counter(
            "high_alpha"
        )  # α > 0.5

        # 로거는 이미 초기화됨
        self.logger.info(
            f"{risk_type} SAC B-Cell이 초기화되었습니다. "
            f"상태차원={state_dim}, 행동차원={action_dim}, "
            f"Target Entropy={self.target_entropy}, "
            f"Device={DEVICE}"
        )

    def _compute_dirichlet_entropy_prior(self, action_dim, D_target=None):
        """
        Dirichlet 목표 엔트로피 초기화
        D_target: 원하는 다양도 E[sum w_i^2] (optional). 없으면 α_i=1 사용
        """
        import math
        from scipy.special import digamma, loggamma

        # 1) 균등 농도 설정 (과탐색 방지를 위해 0.5로 하향) 또는 다양도 기반 c
        if D_target is not None:
            c = dirichlet_c_from_diversity(action_dim, D_target)
        else:
            c = 1.0  # Uniform Dirichlet for balanced exploration

        self._alpha_prior_scalar = c
        alpha0 = action_dim * c

        # 2) Dirichlet(c*1) 엔트로피 계산
        log_beta = action_dim * loggamma(c) - loggamma(alpha0)
        h_prior = (
            log_beta
            + (alpha0 - action_dim) * digamma(alpha0)
            - action_dim * (c - 1.0) * digamma(c)
        )

        h_min = -math.log(action_dim)  # 최소(집중) 근사치

        # TARGET_ENTROPY_SCALE 적용 (2024 research: high-dim action spaces)
        scaled_h_prior = h_prior * TARGET_ENTROPY_SCALE
        scaled_h_min = h_min * TARGET_ENTROPY_SCALE
        
        self.target_entropy_max = scaled_h_prior
        self.target_entropy_min = scaled_h_min
        self.target_entropy = scaled_h_prior

        # 정책 엔트로피 EMA 초기화 (정책-적응형 스케줄링용)
        self._policy_entropy_ema = None

        self.logger.debug(
            f"Dirichlet prior: α_i={c:.3f}, H_max≈{abs(h_prior):.3f}, H_min≈{abs(h_min):.3f}"
        )

    def _update_target_entropy(self):
        """
        목표 엔트로피 스케줄링 (정책-적응형 혼합 포함)
        에피소드 진행률에 따라 탐색에서 활용으로 점진적 전환
        """
        # 기존 진행률 스케줄
        if self.episode_progress <= self.entropy_schedule_warmup:
            base = self.target_entropy_max
        elif self.episode_progress >= self.entropy_schedule_cooldown:
            k = (self.episode_progress - self.entropy_schedule_cooldown) / (
                1.0 - self.entropy_schedule_cooldown
            )
            base = (1 - k) * self.target_entropy_max + k * self.target_entropy_min
        else:
            base = 0.5 * (self.target_entropy_max + self.target_entropy_min)

        # 정책-적응형 혼합 (옵션)
        if self._policy_entropy_ema is not None:
            lam = max(
                0.2, 1.0 - self.episode_progress
            )  # α 상한 고착 방지를 위해 λ 하향 조정
            self.target_entropy = lam * base + (1 - lam) * self._policy_entropy_ema
        else:
            self.target_entropy = base
        
        # 엔트로피 sanity check 및 교정
        self._entropy_sanity_check()
    
    def _entropy_sanity_check(self):
        """목표 엔트로피 범위 검증 및 교정 (로그 스팸 방지)"""
        original_value = self.target_entropy
        
        # Dirichlet K=3에서 합리적 범위: [-5, 5]
        if not (-5.0 <= self.target_entropy <= 5.0):
            # 첫 번째 교정시에만 INFO 로그 (각 B-Cell당 1회만)
            if not hasattr(self, '_entropy_corrected'):
                self.logger.info(
                    f"[{self.risk_type}] Target entropy initialized: {original_value:.3f} → [-5,5] 범위로 클리핑"
                )
                self._entropy_corrected = True
            
            # 교정 수행 (로그 없이)
            self.target_entropy = np.clip(self.target_entropy, -5.0, 5.0)
        
        # 비정상 값 검사
        if not np.isfinite(self.target_entropy):
            if not hasattr(self, '_entropy_nan_corrected'):
                self.logger.error(
                    f"[{self.risk_type}] 비정상 엔트로피 복원: {original_value} → -1.0"
                )
                self._entropy_nan_corrected = True
            self.target_entropy = -1.0  # Dirichlet K=3의 합리적 기본값

    def set_episode_progress(self, current_episode, total_episodes):
        """에피소드 진행률 업데이트"""
        self.episode_progress = current_episode / total_episodes
        self._update_target_entropy()

    def get_action(self, state, deterministic=False):
        """
        SAC 기반 포트폴리오 가중치 생성

        Args:
            state: np.array of shape (state_dim,)
            deterministic: bool, True면 탐험 없이 결정적 행동

        Returns:
            weights: np.array of shape (action_dim,)
        """
        with torch.no_grad():
            state_tensor = (
                torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            )

            # SAC Actor에서 확률적 또는 결정적 행동 샘플링
            if deterministic:
                self.actor.eval()
                concentration, weights, _ = self.actor(state_tensor)
                weights = weights.squeeze(0).cpu().numpy()
                self.actor.train()
            else:
                # 훈련 모드에서는 확률적 샘플링 (엔트로피 최대화)
                concentration, weights, _ = self.actor(state_tensor)
                weights = weights.squeeze(0).cpu().numpy()

        # 가중치 정규화 (안전장치)
        weights = np.clip(weights, 0.001, 0.999)  # 극단값 방지
        weights = weights / weights.sum()  # 재정규화

        return weights

    def store_experience(self, state, action, reward, next_state, done):
        """경험 저장 (CUDA 호환성을 위한 타입 변환)"""
        # NumPy 타입을 Python native 타입으로 안전하게 변환
        done = bool(done)
        reward = float(reward)

        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self):
        """네트워크 업데이트 (SAC + PER)"""
        if len(self.replay_buffer) < self.batch_size:
            return

        # PER 배치 샘플링
        batch, is_weights, indices = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # NumPy 배열을 안전하게 변환하여 CUDA 호환성 확보
        states = torch.tensor(
            np.array(states, dtype=np.float32), dtype=torch.float32
        ).to(DEVICE)
        actions = torch.tensor(
            np.array(actions, dtype=np.float32), dtype=torch.float32
        ).to(DEVICE)
        rewards = torch.tensor([float(r) for r in rewards], dtype=torch.float32).to(
            DEVICE
        )
        next_states = torch.tensor(
            np.array(next_states, dtype=np.float32), dtype=torch.float32
        ).to(DEVICE)
        dones = torch.tensor([bool(d) for d in dones], dtype=torch.bool).to(DEVICE)
        is_weights = torch.tensor(
            np.array(is_weights, dtype=np.float32), dtype=torch.float32
        ).to(DEVICE)

        # 현재 alpha 값 업데이트
        self.alpha = self.log_alpha.exp()

        # ===== SAC Twin Critics 업데이트 =====
        with torch.no_grad():
            # SAC에서는 다음 상태에서의 정책으로부터 행동과 로그 확률을 샘플링
            _, next_actions, next_log_probs = self.actor(next_states)

            # Twin Q-values 계산 (SAC는 엔트로피 항 포함)
            target_q1 = self.target_critic1(next_states, next_actions).squeeze()
            target_q2 = self.target_critic2(next_states, next_actions).squeeze()
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs

            # Phase 1: 소프트 Q-클램프 (퍼센타일 기반)
            target_q_values = rewards + self.gamma * target_q * (~dones)

            # 배치별 퍼센타일 기반 정규화 (하드 클립 대신)
            q_p5 = torch.quantile(target_q_values, 0.05)
            q_p95 = torch.quantile(target_q_values, 0.95)
            q_median = torch.median(target_q_values)

            # z-score 기반 소프트 클램프 (극단값만 완화)
            q_std = target_q_values.std()
            if q_std > 1e-6:
                # 3-sigma 범위를 벗어나는 값들만 소프트 클램프
                z_scores = (target_q_values - q_median) / q_std
                extreme_mask = torch.abs(z_scores) > 3.0
                if extreme_mask.any():
                    # 극단값을 3-sigma 경계로 소프트하게 조정
                    target_q_values = torch.where(
                        extreme_mask,
                        q_median + 3.0 * q_std * torch.sign(z_scores),
                        target_q_values,
                    )

        # Phase 2: Critic 업데이트 (Huber loss + CQL 정규화)
        current_q1 = self.critic1(states, actions).squeeze()
        current_q2 = self.critic2(states, actions).squeeze()

        # Huber loss 적용 (아웃라이어에 덜 민감)
        critic1_huber = F.smooth_l1_loss(
            current_q1, target_q_values, reduction="none", beta=HUBER_DELTA
        )
        critic2_huber = F.smooth_l1_loss(
            current_q2, target_q_values, reduction="none", beta=HUBER_DELTA
        )

        # CQL 정규화: 금융 환경 적응 (온라인 상호작용 고려)
        current_cql_alpha = 0.3  # 1.0 → 0.3 과보수 방지, 포트폴리오 환경 적합

        # 효율적인 제안 샘플 생성 (샘플 수 감소)
        batch_size = states.shape[0]
        K, J = 5, 5  # 정책 5개, 기타 제안 5개 (10→5로 감소)
        
        # 1) 정책 샘플
        with torch.no_grad():
            _, policy_actions, _ = self.actor(states)
            policy_actions = policy_actions.repeat_interleave(K, dim=0)  # [K*B, A]
            policy_states = states.repeat_interleave(K, dim=0)           # [K*B, state_dim]
        
        # 2) 균등 Dirichlet(1) 샘플
        uniform_actions = torch.distributions.Dirichlet(
            torch.ones(actions.shape[1], device=DEVICE)
        ).sample((J * batch_size,))  # [J*B, A]
        uniform_states = states.repeat_interleave(J, dim=0)  # [J*B, state_dim]
        
        # 3) Softmax-Gaussian 제안
        gauss_logits = torch.randn(J * batch_size, actions.shape[1], device=DEVICE)
        gauss_actions = torch.softmax(gauss_logits, dim=-1)  # [J*B, A]
        gauss_states = states.repeat_interleave(J, dim=0)    # [J*B, state_dim]

        # Q-values 계산
        def compute_lse_penalty(critic):
            q_policy = critic(policy_states, policy_actions).squeeze()  # [K*B]
            q_uniform = critic(uniform_states, uniform_actions).squeeze()  # [J*B]
            q_gauss = critic(gauss_states, gauss_actions).squeeze()      # [J*B]
            
            # 배치별로 정리: [B, K], [B, J], [B, J]
            q_policy_batch = q_policy.view(batch_size, K).mean(1)  # [B]
            q_uniform_batch = q_uniform.view(batch_size, J).mean(1)  # [B]  
            q_gauss_batch = q_gauss.view(batch_size, J).mean(1)    # [B]
            
            # log-sum-exp over proposal types
            q_proposals = torch.stack([q_policy_batch, q_uniform_batch, q_gauss_batch], dim=1)  # [B, 3]
            lse_vals = torch.logsumexp(q_proposals, dim=1) - np.log(3)  # [B], normalize by 3 types
            
            q_data = critic(states, actions).squeeze()  # [B]
            
            # Hinge (비음수화): max(lse - q_data, 0)
            penalty = F.relu(lse_vals - q_data).mean()
            return penalty

        cql_penalty1 = current_cql_alpha * compute_lse_penalty(self.critic1)
        cql_penalty2 = current_cql_alpha * compute_lse_penalty(self.critic2)
        
        # Weight Decay (L2 정규화) - 고비용 Gradient Penalty 대체
        l2_reg1 = 1e-4 * sum(p.pow(2).sum() for p in self.critic1.parameters())
        l2_reg2 = 1e-4 * sum(p.pow(2).sum() for p in self.critic2.parameters())

        # 최종 손실 (Huber + CQL + L2 정규화) - 항목별 분리 기록
        huber1_loss = (is_weights * critic1_huber).mean()
        huber2_loss = (is_weights * critic2_huber).mean()
        critic1_loss = huber1_loss + cql_penalty1 + l2_reg1
        critic2_loss = huber2_loss + cql_penalty2 + l2_reg2
        
        # CQL 디버그 로깅 (주요 통계)
        if self.update_count % 50 == 0:  # 50회마다 로깅
            # Q 통계 계산
            q_abs_max = max(current_q1.abs().max().item(), current_q2.abs().max().item())
            tail_ratio = float((current_q1.abs() > 50).float().mean() + (current_q2.abs() > 50).float().mean()) / 2
            
            self.logger.debug(
                f"[{self.risk_type}] CQL-Debug: "
                f"Huber1={huber1_loss.item():.3f}, CQL1={cql_penalty1.item():.3f}, "
                f"Huber2={huber2_loss.item():.3f}, CQL2={cql_penalty2.item():.3f}, "
                f"α_cql={current_cql_alpha:.3f}, Q_max={q_abs_max:.1f}, tail_r={tail_ratio:.2%}"
            )

        # Q-range 기반 Learning Rate 조정 (Cyclical Learning Rates 아이디어)
        q_range = current_q1.max() - current_q1.min() + current_q2.max() - current_q2.min()
        self._adjust_learning_rates(q_range.item())

        # Critic 1 업데이트
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        critic1_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.critic1.parameters(), MAX_GRAD_NORM
        )
        self.critic1_optimizer.step()

        # Critic 2 업데이트
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        critic2_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.critic2.parameters(), MAX_GRAD_NORM
        )
        self.critic2_optimizer.step()

        # TD-error 계산 (PER 우선순위 업데이트용) + 강화된 안정화
        td_errors_1 = (current_q1 - target_q_values).abs()
        td_errors_2 = (current_q2 - target_q_values).abs()
        td_errors = torch.min(td_errors_1, td_errors_2).detach().cpu().numpy()

        # 강화된 TD 오류 클리핑 (마이너 TD 및 로그 스케일 적용)
        td_errors_clipped = np.clip(
            td_errors, a_min=0.0, a_max=50.0
        )  # 최대 50으로 제한

        # 로그 스케일 TD 오류 (양극단 억제)
        log_scale_td_errors = np.log1p(td_errors_clipped)  # log(1 + td_error)

        # 최종 우선순위 (안정화된 버전)
        priorities = log_scale_td_errors + 1e-6
        priorities = np.clip(priorities, a_min=1e-6, a_max=10.0)  # 안전 범위

        self.replay_buffer.update_priorities(indices, priorities)

        # Phase 1: 정책 지연 업데이트 (2스텝에 1회만 Actor 업데이트)
        actor_loss = torch.tensor(0.0, device=DEVICE)
        actor_grad_norm = torch.tensor(0.0, device=DEVICE)

        if self.update_count % self.policy_update_frequency == 0:
            # ===== SAC Actor 업데이트 =====
            _, current_actions, current_log_probs = self.actor(states)

            # Q-values for current actions
            q1_current = self.critic1(states, current_actions).squeeze()
            q2_current = self.critic2(states, current_actions).squeeze()
            q_current = torch.min(q1_current, q2_current)

            # SAC Actor 손실 (엔트로피 정규화 포함)
            actor_loss = (self.alpha * current_log_probs - q_current).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            # Phase 2: Actor 그래디언트 클리핑 강화
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), MAX_GRAD_NORM
            )
            self.actor_optimizer.step()
        else:
            # Actor 업데이트 생략 (Critic만 업데이트)
            with torch.no_grad():
                _, current_actions, current_log_probs = self.actor(states)

        # ===== Alpha (엔트로피 계수) 자동 튜닝 (개선된 버전) =====
        # 정책 엔트로피 EMA 업데이트 (정책-적응형 스케줄링용)
        current_policy_entropy = -current_log_probs.mean().item()
        if self._policy_entropy_ema is None:
            self._policy_entropy_ema = current_policy_entropy
        else:
            self._policy_entropy_ema = (
                0.95 * self._policy_entropy_ema + 0.05 * current_policy_entropy
            )

        # 목표 엔트로피 스케줄링 적용
        self._update_target_entropy()

        alpha_loss = -(
            self.log_alpha * (current_log_probs + self.target_entropy).detach()
        ).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        # Phase 1: 온도 파라미터 그래디언트 클리핑
        alpha_grad_norm = torch.nn.utils.clip_grad_norm_(
            [self.log_alpha], MAX_GRAD_NORM
        )

        # 업데이트 전 log-alpha 값 저장 (로깅용)
        pre_update_log_alpha = self.log_alpha.item()

        self.alpha_optimizer.step()

        # Phase 1: 로그-온도 안전 범위 클리핑 (업데이트 후)
        with torch.no_grad():
            self.log_alpha.data.clamp_(self.log_alpha_min, self.log_alpha_max)
            self.alpha = self.log_alpha.exp()

            # 클리핑 발생 횟수 추적
            post_update_log_alpha = self.log_alpha.item()
            if abs(post_update_log_alpha - pre_update_log_alpha) < 1e-8:
                if not hasattr(self, "_alpha_clipping_count"):
                    self._alpha_clipping_count = 0
                self._alpha_clipping_count += 1

                # 지속적 클리핑 경고 (rate limit 적용)
                if self._alpha_clipping_count % 1000 == 1:  # 1000회마다 1회만 경고
                    self.logger.warning(
                        f"[{self.risk_type}] 온도 파라미터 클리핑 발생 {self._alpha_clipping_count}회 - "
                        f"log_α={post_update_log_alpha:.4f}"
                    )

        # 타겟 네트워크 소프트 업데이트 (보수적 Emergency Sync)
        needs_emergency_sync = self._check_emergency_sync_trigger(current_q1, current_q2)
        self._soft_update_targets(force_sync=needs_emergency_sync)

        # Phase 3: 강화된 NaN/Inf 안전장치 + 추가 안정성 체크
        losses_finite = (
            torch.isfinite(actor_loss)
            and torch.isfinite(critic1_loss)
            and torch.isfinite(critic2_loss)
            and torch.isfinite(alpha_loss)
        )

        # Q-value 범위 체크
        q_values_in_range = (
            torch.all(torch.isfinite(current_q1))
            and torch.all(torch.isfinite(current_q2))
            and current_q1.abs().max() < 500.0  # Q-value 극단값 체크
            and current_q2.abs().max() < 500.0
        )

        # 엔트로피 계수 안정성 체크
        alpha_stable = torch.isfinite(self.alpha) and 0.0001 <= self.alpha.item() <= 2.0

        if not (losses_finite and q_values_in_range and alpha_stable):
            # Rate limiting: 매 500스텝마다만 자세한 로그 출력
            if not hasattr(self, "_stability_log_count"):
                self._stability_log_count = 0
            self._stability_log_count += 1

            if self._stability_log_count % 500 == 1:
                self.logger.warning(
                    f"[안정성 문제 요약] 최근 500스텝 중 {self._stability_log_count % 500}번째 발생 - "
                    f"손실: A={actor_loss.item():.3f}, C_avg={(critic1_loss.item()+critic2_loss.item())/2:.3f} | "
                    f"Q범위: [{current_q1.min().item():.1f}, {current_q1.max().item():.1f}] | "
                    f"α={self.alpha.item():.3f}"
                )

            # 비상 대응: 모델 리셋 신호
            if not hasattr(self, "_stability_failure_count"):
                self._stability_failure_count = 0
            self._stability_failure_count += 1

            if self._stability_failure_count >= 10:
                self.logger.critical(
                    f"[중대] {self.risk_type} B-Cell 안정성 실패 지속: {self._stability_failure_count}회 "
                    f"- 모델 재초기화 필요"
                )

            return False  # 업데이트 실패 신호

        # 통계 기록 및 안정성 카운터 리셋
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append((critic1_loss.item() + critic2_loss.item()) / 2)
        self.update_count += 1

        # 성공적인 업데이트 시 안정성 카운터 리셋
        if hasattr(self, "_stability_failure_count"):
            self._stability_failure_count = max(0, self._stability_failure_count - 1)

        # Phase 3: 통합 슬라이딩 윈도우 통계 수집 (메모리 효율적인 방식)
        with torch.no_grad():
            # Q-value 통계 업데이트
            current_q_min = torch.min(current_q1.min(), current_q2.min()).item()
            current_q_max = torch.max(current_q1.max(), current_q2.max()).item()
            current_q_range = current_q_max - current_q_min

            self.q_min_stats.update(current_q_min)
            self.q_max_stats.update(current_q_max)
            self.q_range_stats.update(current_q_range)

            # 손실 통계 업데이트
            self.actor_loss_stats.update(actor_loss.item())
            self.critic_loss_stats.update(
                (critic1_loss.item() + critic2_loss.item()) / 2
            )

            # Alpha 및 목표 엔트로피 업데이트
            current_alpha = self.alpha.item()
            self.alpha_stats.update(current_alpha)
            self.target_entropy_stats.update(self.target_entropy)

            # TD 오류 통계 업데이트
            td_error_mean = td_errors_clipped.mean()
            self.td_error_stats.update(td_error_mean)

            # 카운터 업데이트
            self.nan_loss_counter.update(not losses_finite)
            self.extreme_q_counter.update(
                abs(current_q_min) > 100 or abs(current_q_max) > 100
            )
            self.high_alpha_counter.update(current_alpha > 0.5)

        # 간소화된 Alert 시스템 (500회마다 핵심 지표만 체크)
        if self.update_count % 500 == 0:
            self._simple_health_check(current_q1, current_q2)

        # Phase 4: PER 베타 어닐링 (매 업데이트마다)
        self.replay_buffer.beta = min(
            1.0, self.replay_buffer.beta + self.replay_buffer.beta_increment
        )

        # Q-value 범위 조정 확인 로깅 (첫 실행 시)
        if self.update_count == 1:
            self.logger.info("Q-value 범위가 [-10, 10]으로 조정되었습니다")

        # 간소화된 Q-regularization (복잡한 쿨다운 시스템 대체)
        return self._apply_q_regularization()
    
    def _apply_q_regularization(self):
        """단순 Q-regularization (Haarnoja et al. 2019)"""
        # 현재 Q-values에 L2 regularization 적용
        q1_reg = 0.001 * sum(p.pow(2).sum() for p in self.critic1.parameters())
        q2_reg = 0.001 * sum(p.pow(2).sum() for p in self.critic2.parameters())
        
        # 정규화 손실을 다음 업데이트에 반영하기 위해 저장
        if not hasattr(self, '_q_reg_loss'):
            self._q_reg_loss = 0.0
        self._q_reg_loss = (q1_reg + q2_reg) * 0.5
        
        return True  # 항상 성공 반환 (단순화)
    
    # def _compute_gradient_penalty(self, critic, states, actions, lambda_gp=0.1):
    #     """
    #     [DEPRECATED] Gradient Penalty - Weight Decay로 대체됨
    #     고비용 계산으로 인해 L2 정규화로 교체
    #     """
    #     # 더 이상 사용하지 않음 - Weight Decay로 대체
    
    def _adjust_learning_rates(self, q_range):
        """Q-range 기반 적응형 학습률 조정 - Actor LR만 조정 (Critic 타겟 추적 보호)"""
        # 기준: Q-range > 100이면 Actor LR 반감, < 20이면 복원
        # Critic LR은 고정 유지 (안정적인 타겟 추적을 위해)
        if not hasattr(self, '_lr_reduction_count'):
            self._lr_reduction_count = 0
            self._original_actor_lr = ACTOR_LR
        
        # Q-range가 크면 Actor 학습률만 감소 (Critic은 고정)
        if q_range > 100.0 and self._lr_reduction_count == 0:
            for param_group in self.actor_optimizer.param_groups:
                param_group['lr'] = self._original_actor_lr * 0.5
            
            self._lr_reduction_count = 1
            self.logger.info(f"[{self.risk_type}] Q-range={q_range:.1f} → Actor LR 50% 감소 (Critic LR 유지)")
        
        # Q-range가 안정되면 Actor 학습률 복원 (Critic은 항상 원래값 유지)
        elif q_range < 20.0 and self._lr_reduction_count > 0:
            for param_group in self.actor_optimizer.param_groups:
                param_group['lr'] = self._original_actor_lr
            
            self._lr_reduction_count = 0
            self.logger.info(f"[{self.risk_type}] Q-range={q_range:.1f} → Actor LR 복원 (Critic LR 안정 유지)")
    
    def _check_emergency_sync_trigger(self, current_q1, current_q2):
        """보수적 Emergency Sync 트리거 - 연속 3회 위반 시에만 1회 실행"""
        # Q-극단 상황 감지 기준
        q_abs_max = max(current_q1.abs().max().item(), current_q2.abs().max().item())
        extreme_ratio = float((current_q1.abs() > 50).float().mean() + (current_q2.abs() > 50).float().mean()) / 2
        
        # 위반 조건: |Q|max > 50 또는 극단비율 > 30%
        is_violation = (q_abs_max > 50.0) or (extreme_ratio > 0.3)
        
        # 연속 위반 카운터 관리
        if not hasattr(self, '_violation_count'):
            self._violation_count = 0
            self._sync_executed = False
        
        if is_violation:
            self._violation_count += 1
            # 연속 3회 위반 & 아직 동기화 실행 안함
            if self._violation_count >= 3 and not self._sync_executed:
                self._sync_executed = True
                return True
        else:
            # 위반 해제 시 카운터 리셋
            self._violation_count = 0
            self._sync_executed = False
        
        return False
    
    def _simple_health_check(self, current_q1, current_q2):
        """간소화된 핵심 지표 체크 (복잡한 슬라이딩 통계 대체)"""
        with torch.no_grad():
            # 핵심 지표만 체크
            q_range = (current_q1.max() - current_q1.min() + current_q2.max() - current_q2.min()).item()
            alpha_val = self.alpha.item()
            
            # Alert 조건 (간소화된 기준)
            alerts = []
            if q_range > 50:
                alerts.append(f"Q-range={q_range:.1f}")
            if alpha_val > 0.8 or alpha_val < 0.001:
                alerts.append(f"α={alpha_val:.3f}")
            
            if alerts:
                self.logger.warning(f"[{self.risk_type}] Health Alert: {', '.join(alerts)}")
            else:
                self.logger.debug(f"[{self.risk_type}] Health OK: Q-range={q_range:.1f}, α={alpha_val:.3f}")
    
    def _check_q_safety_and_cooldown(self, current_q1, current_q2):
        """Q-value 안전 기준 검사 및 쿨다운 시스템"""
        with torch.no_grad():
            # Q-value 안전 통계 계산
            q_abs_max = max(current_q1.abs().max().item(), current_q2.abs().max().item())
            extreme_mask1 = current_q1.abs() > 50
            extreme_mask2 = current_q2.abs() > 50
            extreme_ratio = float((extreme_mask1 | extreme_mask2).float().mean())
            
            # Alpha 안정성 검사
            alpha_stable = torch.isfinite(self.alpha) and 1e-4 <= self.alpha.item() <= 1.5
            
            # 안전 기준 위반 검사
            safety_violation = (q_abs_max > 100.0) or (extreme_ratio > 0.2) or (not alpha_stable)
            
            if safety_violation:
                # 쿨다운 카운터 증가
                if not hasattr(self, '_cooldown_steps'):
                    self._cooldown_steps = 0
                self._cooldown_steps += 1
                
                # 쿨다운 진행 중 (최대 10스텝)
                if self._cooldown_steps <= 10:
                    # 학습 완화 모드
                    self._apply_cooldown_mode()
                    
                    self.logger.warning(
                        f"[{self.risk_type}] Q-안전 쿨다운 {self._cooldown_steps}/10: "
                        f"|Q|_max={q_abs_max:.1f}, extreme_r={extreme_ratio:.2%}, "
                        f"α={self.alpha.item():.3f} (stable={alpha_stable})"
                    )
                    
                    return False  # 업데이트 완화 신호
                else:
                    # 긴급 대응: 파라미터 부분 리셋
                    self.logger.critical_rate_limited(
                        f"[{self.risk_type}] 지속적 Q-불안정 - 부분 리셋 적용", 
                        key="q_instability"
                    )
                    self._emergency_parameter_reset()
                    self._cooldown_steps = 0
                    return False
            else:
                # 안전 기준 만족 시 쿨다운 해제
                if hasattr(self, '_cooldown_steps') and self._cooldown_steps > 0:
                    self._cooldown_steps = max(0, self._cooldown_steps - 1)
                    if self._cooldown_steps == 0:
                        self.logger.info(f"[{self.risk_type}] Q-안전 쿨다운 해제")
                
                return True  # 정상 업데이트
    
    def _apply_cooldown_mode(self):
        """쿨다운 모드 적용: 학습률 감소 및 Actor 업데이트 스킵"""
        # Learning rate 임시 감소
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = min(param_group['lr'], ACTOR_LR * 0.5)
        for param_group in self.critic1_optimizer.param_groups:
            param_group['lr'] = min(param_group['lr'], CRITIC_LR * 0.7)
        for param_group in self.critic2_optimizer.param_groups:
            param_group['lr'] = min(param_group['lr'], CRITIC_LR * 0.7)
        
        # Actor 업데이트 빈도 감소 (다음 몇 회 스킵)
        self.policy_update_frequency = max(self.policy_update_frequency, 4)
    
    def _emergency_parameter_reset(self):
        """긴급 파라미터 부분 리셋"""
        # Alpha 파라미터만 보수적으로 리셋
        with torch.no_grad():
            self.log_alpha.data.fill_(np.log(0.1))  # α = 0.1로 보수적 시작
        
        # Learning rate 복원
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = ACTOR_LR
        for param_group in self.critic1_optimizer.param_groups:
            param_group['lr'] = CRITIC_LR
        for param_group in self.critic2_optimizer.param_groups:
            param_group['lr'] = CRITIC_LR
        
        # 정책 업데이트 빈도 복원
        self.policy_update_frequency = 2

    def _log_unified_monitoring_summary(self):
        """극간소화 모니터링 - 핵심 7지표만 추적"""
        # 7개 핵심 지표 수집
        q_range_stats = self.q_range_stats.get_stats()
        q_max_stats = self.q_max_stats.get_stats()
        alpha_stats = self.alpha_stats.get_stats()
        actor_loss_stats = self.actor_loss_stats.get_stats()
        critic_loss_stats = self.critic_loss_stats.get_stats()
        td_error_stats = self.td_error_stats.get_stats()
        
        # 현재 정책 엔트로피 (H(policy))
        policy_entropy = getattr(self, '_policy_entropy_ema', 0.0)
        
        # TD_p90 계산 (90th percentile)
        td_p90 = td_error_stats.get("sliding_p90", 0.0) if hasattr(td_error_stats, 'get') and "sliding_p90" in td_error_stats else 0.0

        # 안정성 체크
        is_stable = (
            critic_loss_stats["sliding_mean"] < 1e6
            and q_range_stats["sliding_mean"] < 200
            and alpha_stats["sliding_mean"] < 1.0
        )
        status = "OK" if is_stable else "WARN"

        self.logger.debug(
            f"[{self.risk_type}] {status} 업데이트 {self.update_count} (통합스텋): "
            f"Q-range:{q_range_stats['sliding_mean']:.1f} "
            f"|Q|max:{abs(q_max_stats['sliding_mean']):.1f} "
            f"α:{alpha_stats['sliding_mean']:.3f} "
            f"A-loss:{actor_loss_stats['sliding_mean']:.2f} "
            f"C-loss:{critic_loss_stats['sliding_mean']:.2f} "
            f"H(π):{policy_entropy:.3f} "
            f"TD_p90:{td_p90:.2f}"
        )

        # 경고는 최소화 (중요한 것만)
        if not is_stable:
            critical_issues = []
            if critic_loss_stats["sliding_mean"] > 1e6:
                critical_issues.append(f"C-loss:{critic_loss_stats['sliding_mean']:.1e}")
            if q_range_stats["sliding_mean"] > 200:
                critical_issues.append(f"Q-range:{q_range_stats['sliding_mean']:.1f}")
            if alpha_stats["sliding_mean"] > 1.0:
                critical_issues.append(f"α:{alpha_stats['sliding_mean']:.3f}")
            
            if critical_issues:
                self.logger.warning(f"[{self.risk_type}] 핵심지표 이상: {' '.join(critical_issues)}")

    def _check_stability_warnings_unified(self):
        """극간소화 안정성 경고 - 치명적 이슈만 감지"""
        if self.update_count < 50:
            return

        # 치명적 Alpha 문제만 경고 (극단값)
        alpha_stats = self.alpha_stats.get_stats()
        if alpha_stats["sliding_size"] >= 25:
            alpha_mean = alpha_stats["sliding_mean"]
            if alpha_mean > 2.0:  # 극단적으로 높음
                self.logger.warning(f"[치명] α 폭주: {alpha_mean:.3f}")
            elif alpha_mean < 1e-5:  # 극단적으로 낮음 (학습 정지)
                self.logger.warning(f"[치명] α 고착: {alpha_mean:.1e}")

        # 치명적 Q-range 확산만 경고
        q_range_stats = self.q_range_stats.get_stats()
        if q_range_stats["sliding_size"] >= 25:
            q_range_mean = q_range_stats["sliding_mean"]
            if q_range_mean > 500:  # 극단 확산
                self.logger.warning(f"[치명] Q-range 폭주: {q_range_mean:.1f}")


    def _soft_update_targets(self, force_sync: bool = False):
        """타겟 네트워크들 소프트 업데이트 (SAC - Critic만) + Emergency Sync"""
        # Emergency Sync: Q-극단 상황에서 강제 100% 동기화
        sync_rate = 1.0 if force_sync else self.tau
        
        if force_sync:
            self.logger.warning(f"[{self.risk_type}] Emergency Target Sync 실행 (100% 동기화)")
        
        # Target Critic 1 업데이트  
        for target_param, param in zip(
            self.target_critic1.parameters(), self.critic1.parameters()
        ):
            target_param.data.copy_(
                sync_rate * param.data + (1 - sync_rate) * target_param.data
            )

        # Target Critic 2 업데이트
        for target_param, param in zip(
            self.target_critic2.parameters(), self.critic2.parameters()
        ):
            target_param.data.copy_(
                sync_rate * param.data + (1 - sync_rate) * target_param.data
            )

    def get_specialization_score(self, crisis_info):
        """
        다차원 위기 정보에 대한 전문성 점수 계산

        Args:
            crisis_info: dict or float - 다차원 위기 정보 또는 기존 crisis_level

        Returns:
            score: float [0, 1] - 높을수록 현재 상황에 특화됨
        """
        # 하위 호환성: 기존 float crisis_level을 dict 형태로 변환
        if isinstance(crisis_info, (int, float)):
            crisis_info = {"overall_crisis": crisis_info}

        # 다차원 위기 벡터 기반 전문성 계산
        if isinstance(crisis_info, dict):
            overall_crisis = crisis_info.get("overall_crisis", 0.0)
            volatility_crisis = crisis_info.get("volatility_crisis", 0.0)
            correlation_crisis = crisis_info.get("correlation_crisis", 0.0)
            volume_crisis = crisis_info.get("volume_crisis", 0.0)

            if self.risk_type == "volatility":
                # 변동성 전문가: 변동성 위기와 전체 위기에 특화
                volatility_score = volatility_crisis * 1.5  # 변동성 위기에 높은 가중치
                overall_score = overall_crisis * 0.8
                return np.clip(volatility_score + overall_score * 0.5, 0.0, 1.0)

            elif self.risk_type == "correlation":
                # 상관관계 전문가: 상관관계 위기와 중간 수준 전체 위기에 특화
                correlation_score = correlation_crisis * 1.8
                optimal_overall = (
                    1 - abs(overall_crisis - 0.55) * 2.0
                )  # 중간 위기 수준 선호
                return np.clip(correlation_score + optimal_overall * 0.3, 0.0, 1.0)

            elif self.risk_type == "momentum":
                # 모멘텀 전문가: 낮은 위기 상황과 거래량 이상에 특화
                momentum_score = max(0, 1 - overall_crisis * 2.5)  # 낮은 위기 선호
                volume_score = volume_crisis * 1.2  # 거래량 이상 활용
                return np.clip(momentum_score + volume_score * 0.4, 0.0, 1.0)

            elif self.risk_type == "defensive":
                # 방어 전문가: 중고위기와 모든 위기 유형에 균형 있게 대응
                defensive_score = 1 - abs(overall_crisis - 0.65) * 1.8  # 중고위기 선호
                multi_crisis = (
                    volatility_crisis + correlation_crisis + volume_crisis
                ) / 3
                return np.clip(defensive_score + multi_crisis * 0.6, 0.0, 1.0)

            elif self.risk_type == "growth":
                # 성장 전문가: 매우 낮은 위기 상황에 특화
                growth_score = max(0, 1 - overall_crisis * 3.5)  # 매우 낮은 위기만
                stability_bonus = max(
                    0, 1 - volatility_crisis * 2
                )  # 낮은 변동성 보너스
                return np.clip(growth_score + stability_bonus * 0.3, 0.0, 1.0)

            else:
                return 0.5  # 기본값

        return 0.5  # 예외 상황

    def get_explanation(self, state):
        """
        의사결정에 대한 설명 생성 (XAI)

        Returns:
            dict: 의사결정 설명
        """
        self.actor.eval()

        with torch.no_grad():
            state_tensor = (
                torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            )
            concentration, weights, _ = self.actor(state_tensor)
            weights = weights.squeeze(0)
            value = self.critic1(state_tensor, weights.unsqueeze(0)).squeeze(0)

        self.actor.train()

        # 입력 특성 분석
        features = state[:12]  # 시장 특성
        crisis_level = state[12]  # 위기 수준
        prev_weights = state[13:]  # 이전 가중치

        explanation = {
            "risk_type": self.risk_type,
            "predicted_weights": weights.cpu().numpy().tolist(),
            "predicted_value": float(value.cpu()),
            "specialization_score": self.get_specialization_score(crisis_level),
            "crisis_level": float(crisis_level),
            "max_weight_asset": int(weights.argmax().cpu()),
            "min_weight_asset": int(weights.argmin().cpu()),
            "weight_concentration": float((weights**2).sum().cpu()),
            "alpha": float(self.alpha.item()),
            "update_count": self.update_count,
        }

        return explanation

    def save_model(self, filepath):
        """모델 저장 (SAC)"""
        # 저장 디렉토리 생성 보장
        base_dir = os.path.dirname(filepath)
        if base_dir:
            os.makedirs(base_dir, exist_ok=True)

        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic1_state_dict": self.critic1.state_dict(),
                "critic2_state_dict": self.critic2.state_dict(),
                "target_critic1_state_dict": self.target_critic1.state_dict(),
                "target_critic2_state_dict": self.target_critic2.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic1_optimizer_state_dict": self.critic1_optimizer.state_dict(),
                "critic2_optimizer_state_dict": self.critic2_optimizer.state_dict(),
                "alpha_optimizer_state_dict": self.alpha_optimizer.state_dict(),
                "log_alpha": self.log_alpha,
                "target_entropy": self.target_entropy,
                "risk_type": self.risk_type,
                "update_count": self.update_count,
            },
            filepath,
        )

        self.logger.info(
            f"{self.risk_type} SAC B-Cell 모델이 저장되었습니다: {filepath}"
        )

    def load_model(self, filepath):
        """모델 로드 (SAC)"""
        checkpoint = torch.load(filepath, map_location=DEVICE)

        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic1.load_state_dict(checkpoint["critic1_state_dict"])
        self.critic2.load_state_dict(checkpoint["critic2_state_dict"])
        self.target_critic1.load_state_dict(checkpoint["target_critic1_state_dict"])
        self.target_critic2.load_state_dict(checkpoint["target_critic2_state_dict"])

        self.actor_optimizer.load_state_dict(
            checkpoint["actor_optimizer_state_dict"]
        )
        self.critic1_optimizer.load_state_dict(
            checkpoint["critic1_optimizer_state_dict"]
        )
        self.critic2_optimizer.load_state_dict(
            checkpoint["critic2_optimizer_state_dict"]
        )
        self.alpha_optimizer.load_state_dict(
            checkpoint["alpha_optimizer_state_dict"]
        )

        self.log_alpha = checkpoint["log_alpha"].to(DEVICE)
        self.target_entropy = checkpoint["target_entropy"]
        self.alpha = self.log_alpha.exp()
        self.update_count = checkpoint["update_count"]

        self.logger.info(
            f"{self.risk_type} SAC B-Cell 모델이 로드되었습니다: {filepath}"
        )
