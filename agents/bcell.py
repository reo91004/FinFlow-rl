# bipd/agents/bcell.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import time
from collections import deque
import random
from utils.logger import BIPDLogger
from utils.rolling_stats import MultiRollingStats
from utils.extreme_q_monitor import DualQMonitor
from utils.portfolio_utils import project_to_capped_simplex
from agents.utils.dirichlet_entropy import target_entropy_from_symmetric_alpha
from agents.utils.entropy_schedule import RegimeAdaptiveEntropyScheduler
from agents.utils.entropy_target import DirichletEntropyTracker
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
    ENABLE_CQL,
    ENABLE_KL_REGULARIZATION,
    KL_PENALTY_WEIGHT,
    N_EPISODES,
    ROLLING_STATS_WINDOW,
    # 새로운 안정화 파라미터들
    TARGET_ENTROPY_FROM_DIRICHLET,
    DIRICHLET_ALPHA_STAR,
    LOG_ALPHA_MIN,
    LOG_ALPHA_MAX,
    DIRICHLET_CONCENTRATION_MIN,
    DIRICHLET_CONCENTRATION_MAX,
    PORTFOLIO_WEIGHT_MIN,
    CRITIC_GRAD_NORM,
    ACTOR_GRAD_NORM,
    ALPHA_GRAD_NORM,
    ENHANCED_HUBER_DELTA,
    Q_TARGET_HARD_CLIP_MIN,
    Q_TARGET_HARD_CLIP_MAX,
    Q_VALUE_STABILITY_CHECK,
    Q_MONITOR_WINDOW_SIZE,
    Q_EXTREME_THRESHOLD,
)

# Dirichlet 수치 안정화 파라미터
MIN_CONC = 0.20   # 극소 농도 방지
MAX_CONC = 50.0   # 과도 균등화 방지

def stable_dirichlet(raw_conc: torch.Tensor) -> torch.distributions.Dirichlet:
    """
    수치적으로 안정한 Dirichlet 분포 생성
    
    Args:
        raw_conc: 네트워크 원출력 (softplus 적용 전)
        
    Returns:
        torch.distributions.Dirichlet: 안정화된 Dirichlet 분포
    """
    conc = F.softplus(raw_conc) + 1e-4
    conc = conc.clamp_(MIN_CONC, MAX_CONC)
    return torch.distributions.Dirichlet(conc)


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

        # Dirichlet 농도 파라미터 계산
        x_clamped = torch.clamp(self.concentration_head(x), min=-10.0, max=10.0)
        concentration = F.softplus(x_clamped) + DIRICHLET_CONCENTRATION_MIN
        # 농도 파라미터 범위 제한
        concentration = torch.clamp(concentration, 
                                  min=DIRICHLET_CONCENTRATION_MIN, 
                                  max=DIRICHLET_CONCENTRATION_MAX)

        # 안정화된 Dirichlet 분포에서 샘플링
        if self.training:
            # 훈련 시: 안정화된 확률적 샘플링
            dist = stable_dirichlet(self.concentration_head(x))  # raw 출력 직접 사용
            weights = dist.rsample()  # reparameterization trick 사용
            log_prob = dist.log_prob(weights)
            
            # 정책 엔트로피 계산 (적응형 엔트로피 추적용)
            policy_entropy = dist.entropy()
        else:
            # 평가 시: 결정적 출력 (안정화된 평균 사용)
            conc = F.softplus(self.concentration_head(x)) + 1e-4
            conc = conc.clamp_(MIN_CONC, MAX_CONC)
            weights = conc / conc.sum(dim=-1, keepdim=True)
            log_prob = torch.zeros(weights.shape[0], device=weights.device)
            policy_entropy = torch.zeros(weights.shape[0], device=weights.device)

        # 심플렉스 투영 적용 (제약 조건 엄밀 보장)
        batch_size = weights.shape[0]
        projected_weights = torch.zeros_like(weights)
        for i in range(batch_size):
            weight_np = weights[i].detach().cpu().numpy()
            projected_np = project_to_capped_simplex(
                weight_np, 
                target_sum=1.0,
                w_min=PORTFOLIO_WEIGHT_MIN,
                w_max=1.0 - PORTFOLIO_WEIGHT_MIN * (self.action_dim - 1)
            )
            projected_weights[i] = torch.from_numpy(projected_np).to(weights.device)
        
        weights = projected_weights

        return concentration, weights, log_prob, policy_entropy

    def get_action_and_log_prob(self, state):
        """행동과 로그 확률을 함께 반환"""
        concentration, weights, log_prob, policy_entropy = self.forward(state)
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

        # SAC 엔트로피 계수
        # Dirichlet 분포에 맞춘 초기 타겟 엔트로피 설정
        self._compute_dirichlet_entropy_prior(action_dim)
        
        # DirichletEntropyTracker 초기화 (α 고착 해결용)
        self.use_enhanced_entropy_tracking = True  # 플래그로 토글 가능
        if self.use_enhanced_entropy_tracking:
            self.entropy_tracker = DirichletEntropyTracker(
                k=action_dim, 
                init_alpha=1.5, 
                ema=0.98, 
                margin=0.5,
                floor=-12.0, 
                ceil=-0.1
            )
        else:
            self.entropy_tracker = None
        
        # 적응형 엔트로피 스케줄러 초기화
        self.use_adaptive_entropy = True  # Phase 3 기능 활성화
        if self.use_adaptive_entropy:
            from agents.utils.entropy_schedule import create_entropy_scheduler_for_symbols
            self.entropy_scheduler = create_entropy_scheduler_for_symbols(action_dim)
        else:
            self.entropy_scheduler = None

        # log-α 범위 설정
        init_log_alpha = np.log(0.1)
        self.log_alpha_min = LOG_ALPHA_MIN  # config에서 가져옴
        self.log_alpha_max = LOG_ALPHA_MAX  # config에서 가져옴

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
        self.rolling_stats = MultiRollingStats(window_size=ROLLING_STATS_WINDOW)
        
        # Q-value 극단치 모니터
        self.q_monitor = DualQMonitor(
            window_size=Q_MONITOR_WINDOW_SIZE, 
            q_low=0.05, 
            q_high=0.95,
            extreme_threshold=Q_EXTREME_THRESHOLD,  # config에서 가져옴
            logger_name=f"QMonitor-{risk_type}"
        )

        # 주요 통계 지표 등록 (확장된 모니터링)
        self.q_min_stats = self.rolling_stats.add_statistics("q_min")
        self.q_max_stats = self.rolling_stats.add_statistics("q_max")
        self.q_range_stats = self.rolling_stats.add_statistics("q_range")
        self.actor_loss_stats = self.rolling_stats.add_statistics("actor_loss")
        self.critic_loss_stats = self.rolling_stats.add_statistics("critic_loss")
        self.alpha_stats = self.rolling_stats.add_statistics("alpha")
        self.td_error_stats = self.rolling_stats.add_statistics("td_error")
        self.target_entropy_stats = self.rolling_stats.add_statistics("target_entropy")
        
        # 추가된 강화 모니터링 지표
        self.policy_entropy_stats = self.rolling_stats.add_statistics("policy_entropy")
        self.log_alpha_stats = self.rolling_stats.add_statistics("log_alpha") 
        self.alpha_grad_norm_stats = self.rolling_stats.add_statistics("alpha_grad_norm")
        self.critic_grad_norm_stats = self.rolling_stats.add_statistics("critic_grad_norm")
        self.actor_grad_norm_stats = self.rolling_stats.add_statistics("actor_grad_norm")
        self.q_target_range_stats = self.rolling_stats.add_statistics("q_target_range")

        # 주요 카운터 등록 (확장된 모니터링)
        self.nan_loss_counter = self.rolling_stats.add_counter("nan_loss")
        self.extreme_q_counter = self.rolling_stats.add_counter("extreme_q")  # |Q| > 100
        self.high_alpha_counter = self.rolling_stats.add_counter("high_alpha")  # α > 0.5
        
        # 추가된 카운터
        self.alpha_clipping_counter = self.rolling_stats.add_counter("alpha_clipping")
        self.q_target_clipping_counter = self.rolling_stats.add_counter("q_target_clipping")
        self.high_grad_norm_counter = self.rolling_stats.add_counter("high_grad_norm")

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
        D_target: 원하는 다양도 E[sum w_i^2] (optional). 없으면 균형잡힌 α*=1.5 사용
        """
        import math

        # 균형잡힌 농도 설정
        if D_target is not None:
            alpha_star = dirichlet_c_from_diversity(action_dim, D_target)
        else:
            alpha_star = DIRICHLET_ALPHA_STAR  # config에서 가져옴

        self._alpha_prior_scalar = alpha_star

        # 2) 새로운 유틸리티 함수로 정확한 Dirichlet 엔트로피 계산
        h_prior = target_entropy_from_symmetric_alpha(action_dim, alpha_star)

        # 3) 극단 엔트로피 범위 설정
        h_max = target_entropy_from_symmetric_alpha(action_dim, 1.0)  # 균등 분포
        h_min = -math.log(action_dim)  # 집중 분포 근사치

        self.target_entropy_max = h_max
        self.target_entropy_min = h_min
        self.target_entropy = h_prior

        # 정책 엔트로피 EMA 초기화 (정책-적응형 스케줄링용)
        self._policy_entropy_ema = None

        self.logger.debug(
            f"Dirichlet 설정: α*={alpha_star:.3f}, H_target≈{h_prior:.2f}, "
            f"H_range=[{h_min:.2f}, {h_max:.2f}]"
        )

    def update_adaptive_target_entropy(self, crisis_level: float, market_stability: float = None):
        """
        T-Cell 신호 기반 적응형 엔트로피 업데이트 (Phase 3)
        
        Args:
            crisis_level: T-Cell에서 감지한 위기 레벨 (0-1)
            market_stability: 시장 안정도 (선택적)
        """
        if self.use_adaptive_entropy and self.entropy_scheduler is not None:
            # 스케줄러로부터 새로운 목표 엔트로피 받기
            new_target_entropy, regime_info = self.entropy_scheduler.update_regime_and_get_target_entropy(
                crisis_level, market_stability
            )
            
            # 목표 엔트로피 업데이트
            old_target = self.target_entropy
            self.target_entropy = new_target_entropy
            
            # 실제 정책 엔트로피와의 갭 추적
            if hasattr(self, '_last_policy_entropy') and self._last_policy_entropy is not None:
                self.entropy_scheduler.track_entropy_gap(self._last_policy_entropy)
            
            # 상당한 변화가 있을 때만 로그
            if abs(new_target_entropy - old_target) > 0.1:
                self.logger.debug(
                    f"엔트로피 적응: {old_target:.3f} -> {new_target_entropy:.3f} "
                    f"(레짐: {regime_info['regime']}, 위기: {crisis_level:.2f})"
                )
            
            return regime_info
        else:
            return None

    def _update_target_entropy(self):
        """
        목표 엔트로피 스케줄링 (기존 방식 - 적응형이 비활성화된 경우)
        에피소드 진행률에 따라 탐색에서 활용으로 점진적 전환
        """
        # 적응형 엔트로피가 활성화된 경우 이 메소드는 건너뜀
        if self.use_adaptive_entropy:
            return
            
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
                concentration, weights, _, policy_entropy = self.actor(state_tensor)
                weights = weights.squeeze(0).cpu().numpy()
                self.actor.train()
            else:
                # 훈련 모드에서는 확률적 샘플링 (엔트로피 최대화)
                concentration, weights, _, policy_entropy = self.actor(state_tensor)
                weights = weights.squeeze(0).cpu().numpy()
                
                # 정책 엔트로피 추적 (적응형 엔트로피용)
                if self.use_adaptive_entropy and policy_entropy is not None:
                    self._last_policy_entropy = float(policy_entropy.squeeze(0).cpu().numpy())

        # 가중치 정규화
        max_weight = 1.0 - PORTFOLIO_WEIGHT_MIN * len(weights)
        weights = np.clip(weights, PORTFOLIO_WEIGHT_MIN, max_weight)
        weights = weights / weights.sum()  # 재정규화

        return weights

    def store_experience(self, state, action, reward, next_state, done):
        """경험 저장 (CUDA 호환성을 위한 타입 변환)"""
        # NumPy/Torch 타입을 Python native 타입으로 안전하게 변환
        done = bool(done.item() if hasattr(done, 'item') else done)
        reward = float(reward.item() if hasattr(reward, 'item') else reward)

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
            _, next_actions, next_log_probs, _ = self.actor(next_states)

            # Twin Q-values 계산 (SAC는 엔트로피 항 포함)
            target_q1 = self.target_critic1(next_states, next_actions).squeeze()
            target_q2 = self.target_critic2(next_states, next_actions).squeeze()
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs

            # 대폭 강화된 Q-타깃 클리핑 (극단값 완전 억제)
            target_q_values = rewards + self.gamma * target_q * (~dones)

            # 하드 클리핑
            target_q_values = torch.clamp(target_q_values, 
                                         min=Q_TARGET_HARD_CLIP_MIN, 
                                         max=Q_TARGET_HARD_CLIP_MAX)

            # 2단계: 배치별 IQR 기반 아웃라이어 제거
            if target_q_values.numel() > 4:  # 최소 5개 이상일 때만 적용
                q25 = torch.quantile(target_q_values, 0.25)
                q75 = torch.quantile(target_q_values, 0.75)
                iqr = q75 - q25
                
                # IQR 기반 아웃라이어 경계 (1.5 * IQR)
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
                
                # 아웃라이어를 경계값으로 클리핑
                target_q_values = torch.clamp(target_q_values, lower_bound, upper_bound)

            # 3단계: 추가 안전장치 - NaN/Inf 제거
            target_q_values = torch.where(
                torch.isfinite(target_q_values), 
                target_q_values, 
                torch.zeros_like(target_q_values)
            )

        # Phase 2: Critic 업데이트 (Huber loss + CQL 정규화)
        current_q1 = self.critic1(states, actions).squeeze()
        current_q2 = self.critic2(states, actions).squeeze()
        
        # Q-value 극단치 모니터링
        q_monitor_result = self.q_monitor.update_and_check_both(
            current_q1.detach().cpu().numpy(),
            current_q2.detach().cpu().numpy()
        )

        # Huber loss 적용
        critic1_huber = F.smooth_l1_loss(
            current_q1, target_q_values, reduction="none", beta=ENHANCED_HUBER_DELTA
        )
        critic2_huber = F.smooth_l1_loss(
            current_q2, target_q_values, reduction="none", beta=ENHANCED_HUBER_DELTA
        )

        # CQL 정규화 (조건부 활성화)
        cql_penalty1 = torch.tensor(0.0, device=DEVICE)
        cql_penalty2 = torch.tensor(0.0, device=DEVICE)
        
        if ENABLE_CQL:
            # CQL 정규화: LogSumExp 기반 표준 CQL 구현
            # 점진적 CQL 강도 스케줄링
            progress = self.episode_progress if hasattr(self, "episode_progress") else 0.0
            current_cql_alpha = CQL_ALPHA_START + progress * (
                CQL_ALPHA_END - CQL_ALPHA_START
            )

            # 표준 CQL: 무작위 샘플링된 행동들에 대한 LogSumExp
            batch_size = states.shape[0]
            num_samples = CQL_NUM_SAMPLES

            # 무작위 행동 샘플링 (포트폴리오 제약 유지)
            random_actions = torch.rand(
                batch_size * num_samples, actions.shape[1], device=DEVICE
            )
            random_actions = random_actions / random_actions.sum(dim=1, keepdim=True)

            # 상태 확장 (각 상태에 대해 num_samples개 행동)
            expanded_states = states.repeat_interleave(num_samples, dim=0)

            # Q-values 계산
            q1_random = self.critic1(expanded_states, random_actions).view(
                batch_size, num_samples
            )
            q2_random = self.critic2(expanded_states, random_actions).view(
                batch_size, num_samples
            )

            # LogSumExp 계산 (안정화된 버전)
            logsumexp_q1 = torch.logsumexp(q1_random, dim=1)
            logsumexp_q2 = torch.logsumexp(q2_random, dim=1)

            # 데이터 Q-values (현재 배치)
            q1_data = current_q1
            q2_data = current_q2

            # CQL 손실 계산 (표준 LogSumExp - 데이터 Q)
            cql_penalty1 = current_cql_alpha * (logsumexp_q1.mean() - q1_data.mean())
            cql_penalty2 = current_cql_alpha * (logsumexp_q2.mean() - q2_data.mean())

        # 최종 손실 (Huber + 조건부 CQL)
        critic1_loss = (is_weights * critic1_huber).mean() + cql_penalty1
        critic2_loss = (is_weights * critic2_huber).mean() + cql_penalty2

        # Critic 1 업데이트
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        critic1_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.critic1.parameters(), max_norm=CRITIC_GRAD_NORM
        )
        self.critic1_optimizer.step()

        # Critic 2 업데이트
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        critic2_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.critic2.parameters(), max_norm=CRITIC_GRAD_NORM
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
            _, current_actions, current_log_probs, _ = self.actor(states)

            # Q-values for current actions
            q1_current = self.critic1(states, current_actions).squeeze()
            q2_current = self.critic2(states, current_actions).squeeze()
            q_current = torch.min(q1_current, q2_current)

            # SAC Actor 손실 (엔트로피 정규화 포함)
            actor_loss = (self.alpha * current_log_probs - q_current).mean()
            
            # KL 정규화 (조건부 활성화)
            if ENABLE_KL_REGULARIZATION:
                # 균등 분포와의 KL divergence 계산
                uniform_log_probs = torch.log(torch.ones_like(current_actions) / current_actions.shape[-1])
                kl_penalty = F.kl_div(uniform_log_probs, current_actions, reduction='batchmean')
                actor_loss += KL_PENALTY_WEIGHT * kl_penalty

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            # Actor 그래디언트 클리핑
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), max_norm=ACTOR_GRAD_NORM
            )
            self.actor_optimizer.step()
        else:
            # Actor 업데이트 생략 (Critic만 업데이트)
            with torch.no_grad():
                _, current_actions, current_log_probs, _ = self.actor(states)

        # Alpha (엔트로피 계수) 자동 튜닝
        # 정책 엔트로피 EMA 업데이트 (정책-적응형 스케줄링용)
        current_policy_entropy = -current_log_probs.mean().item()
        if self._policy_entropy_ema is None:
            self._policy_entropy_ema = current_policy_entropy
        else:
            self._policy_entropy_ema = (
                0.95 * self._policy_entropy_ema + 0.05 * current_policy_entropy
            )

        # 강화된 엔트로피 트래킹 사용 여부에 따른 타깃 결정
        if self.use_enhanced_entropy_tracking and self.entropy_tracker is not None:
            # DirichletEntropyTracker로 타깃 엔트로피 계산 (α 고착 방지)
            enhanced_target = self.entropy_tracker.update_and_get_target(current_policy_entropy)
            
            # 기존 적응형 시스템과 혼합 (점진적 전환)
            self._update_target_entropy()  # 기존 시스템 업데이트
            
            # 혼합 비율: 강화된 시스템을 점진적으로 증가
            mix_ratio = 0.7  # 70% 강화 시스템, 30% 기존 시스템
            final_target = mix_ratio * enhanced_target + (1 - mix_ratio) * self.target_entropy
            
            # 로깅 (500회마다)
            if self.update_count % 500 == 0:
                self.logger.debug(
                    f"[{self.risk_type}] 엔트로피 트래킹: 관측={current_policy_entropy:.3f}, "
                    f"기존타깃={self.target_entropy:.3f}, 강화타깃={enhanced_target:.3f}, "
                    f"최종타깃={final_target:.3f}"
                )
        else:
            # 기존 목표 엔트로피 스케줄링만 사용
            self._update_target_entropy()
            final_target = self.target_entropy

        alpha_loss = -(
            self.log_alpha * (current_log_probs + final_target).detach()
        ).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        # 온도 파라미터 그래디언트 클리핑
        alpha_grad_norm = torch.nn.utils.clip_grad_norm_(
            [self.log_alpha], max_norm=ALPHA_GRAD_NORM
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
            clipping_occurred = (
                abs(post_update_log_alpha - self.log_alpha_min) < 1e-6 or 
                abs(post_update_log_alpha - self.log_alpha_max) < 1e-6
            )
            
            if clipping_occurred:
                if not hasattr(self, "_alpha_clipping_count"):
                    self._alpha_clipping_count = 0
                self._alpha_clipping_count += 1

                # 클리핑 경고 (500회마다 1회로 완화)
                if self._alpha_clipping_count % 500 == 1:
                    boundary = "하한" if post_update_log_alpha <= self.log_alpha_min + 1e-6 else "상한"
                    self.logger.warning(
                        f"[{self.risk_type}] α {boundary} 클리핑 {self._alpha_clipping_count}회 - "
                        f"log_α={post_update_log_alpha:.4f} (범위: [{self.log_alpha_min:.1f}, {self.log_alpha_max:.1f}])"
                    )

        # 타겟 네트워크 소프트 업데이트
        self._soft_update_targets()

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
            and current_q1.abs().max() < Q_VALUE_STABILITY_CHECK
            and current_q2.abs().max() < Q_VALUE_STABILITY_CHECK
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
                # 재초기화 경고 스팸 방지: 30초 쿨다운 적용
                if not hasattr(self, "_last_reinit_log_t"):
                    self._last_reinit_log_t = 0
                now = time.time()
                if now - self._last_reinit_log_t > 30:
                    self.logger.critical(
                        f"[중대] {self.risk_type} B-Cell 안정성 실패 지속: {self._stability_failure_count}회 "
                        f"- 모델 재초기화 필요"
                    )
                    self._last_reinit_log_t = now

            return False  # 업데이트 실패 신호

        # 통계 기록 및 안정성 카운터 리셋
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append((critic1_loss.item() + critic2_loss.item()) / 2)
        self.update_count += 1

        # 성공적인 업데이트 시 안정성 카운터 리셋
        if hasattr(self, "_stability_failure_count"):
            self._stability_failure_count = max(0, self._stability_failure_count - 1)

        # 강화된 통합 슬라이딩 윈도우 통계 수집
        with torch.no_grad():
            # Q-value 통계 업데이트
            current_q_min = torch.min(current_q1.min(), current_q2.min()).item()
            current_q_max = torch.max(current_q1.max(), current_q2.max()).item()
            current_q_range = current_q_max - current_q_min

            self.q_min_stats.update(current_q_min)
            self.q_max_stats.update(current_q_max)
            self.q_range_stats.update(current_q_range)

            # Q-target 통계 추가
            target_q_min = target_q_values.min().item()
            target_q_max = target_q_values.max().item()
            target_q_range = target_q_max - target_q_min
            self.q_target_range_stats.update(target_q_range)

            # 손실 통계 업데이트
            self.actor_loss_stats.update(actor_loss.item())
            self.critic_loss_stats.update(
                (critic1_loss.item() + critic2_loss.item()) / 2
            )

            # Alpha 및 목표 엔트로피 확장 업데이트
            current_alpha = self.alpha.item()
            current_log_alpha = self.log_alpha.item()
            self.alpha_stats.update(current_alpha)
            self.log_alpha_stats.update(current_log_alpha)
            self.target_entropy_stats.update(self.target_entropy)

            # 정책 엔트로피 추가 (현재 배치 평균)
            current_policy_entropy = -current_log_probs.mean().item()
            self.policy_entropy_stats.update(current_policy_entropy)

            # 그라디언트 노름 추가
            self.alpha_grad_norm_stats.update(alpha_grad_norm.item() if isinstance(alpha_grad_norm, torch.Tensor) else alpha_grad_norm)
            avg_critic_grad_norm = (critic1_grad_norm + critic2_grad_norm) / 2
            self.critic_grad_norm_stats.update(avg_critic_grad_norm.item() if isinstance(avg_critic_grad_norm, torch.Tensor) else avg_critic_grad_norm)
            if isinstance(actor_grad_norm, torch.Tensor):
                self.actor_grad_norm_stats.update(actor_grad_norm.item())

            # TD 오류 통계 업데이트
            td_error_mean = td_errors_clipped.mean()
            self.td_error_stats.update(td_error_mean)

            # 기존 카운터 업데이트
            self.nan_loss_counter.update(not losses_finite)
            self.extreme_q_counter.update(
                abs(current_q_min) > Q_VALUE_STABILITY_CHECK or abs(current_q_max) > Q_VALUE_STABILITY_CHECK
            )
            self.high_alpha_counter.update(current_alpha > 0.5)
            
            # 새로운 카운터 업데이트
            alpha_clipped = (
                abs(current_log_alpha - LOG_ALPHA_MIN) < 1e-6 or 
                abs(current_log_alpha - LOG_ALPHA_MAX) < 1e-6
            )
            self.alpha_clipping_counter.update(alpha_clipped)
            
            q_target_clipped = (
                abs(target_q_min - Q_TARGET_HARD_CLIP_MIN) < 1e-6 or 
                abs(target_q_max - Q_TARGET_HARD_CLIP_MAX) < 1e-6
            )
            self.q_target_clipping_counter.update(q_target_clipped)
            
            high_grad_norm = (
                alpha_grad_norm > ALPHA_GRAD_NORM * 0.8 or
                avg_critic_grad_norm > CRITIC_GRAD_NORM * 0.8
            )
            self.high_grad_norm_counter.update(high_grad_norm)

        # 주기적 요약 로깅 (500회마다) - 통합 슬라이딩 윈도우 기반
        if self.rolling_stats.should_report(self.update_count, report_interval=500):
            self._log_unified_monitoring_summary(q_monitor_result)

        # Phase 3: 안정성 경고 확인 (200회마다) - 슬라이딩 윈도우 기반
        if self.update_count % 200 == 0:
            self._check_stability_warnings_unified()

        # Phase 4: PER 베타 어닐링 (매 업데이트마다)
        self.replay_buffer.beta = min(
            1.0, self.replay_buffer.beta + self.replay_buffer.beta_increment
        )

        # Q-value 범위 조정 확인 로깅 (첫 실행 시)
        if self.update_count == 1:
            self.logger.info("Q-value 범위가 [-10, 10]으로 조정되었습니다")

        return True  # 성공적인 업데이트

    def _log_unified_monitoring_summary(self, q_monitor_result=None):
        """통합 슬라이딩 윈도우 모니터링 요약"""
        # Q-value 요약
        q_min_stats = self.q_min_stats.get_stats()
        q_max_stats = self.q_max_stats.get_stats()
        q_range_stats = self.q_range_stats.get_stats()

        # 손실 요약
        actor_loss_stats = self.actor_loss_stats.get_stats()
        critic_loss_stats = self.critic_loss_stats.get_stats()

        # Alpha 요약
        alpha_stats = self.alpha_stats.get_stats()
        target_entropy_stats = self.target_entropy_stats.get_stats()

        # 카운터 요약
        nan_counter = self.nan_loss_counter.get_stats()
        extreme_q_counter = self.extreme_q_counter.get_stats()
        high_alpha_counter = self.high_alpha_counter.get_stats()

        # 안정성 체크
        is_stable = (
            critic_loss_stats["sliding_mean"] < 1e6
            and q_range_stats["sliding_mean"] < 200
            and alpha_stats["sliding_mean"] < 1.0
        )
        stability_marker = "✓" if is_stable else "⚠"
        
        # Q-모니터 정보 추가
        q_monitor_info = ""
        if q_monitor_result:
            combined_rate = q_monitor_result.get("combined_extreme_rate", 0.0)
            severe_mismatch = q_monitor_result.get("severe_mismatch", False)
            mismatch_marker = "⚠" if severe_mismatch else ""
            q_monitor_info = f"Qext:{combined_rate:.1%}{mismatch_marker} "

        # 추가 통계 수집
        policy_entropy_stats = self.policy_entropy_stats.get_stats()
        log_alpha_stats = self.log_alpha_stats.get_stats()
        alpha_clipping_stats = self.alpha_clipping_counter.get_stats()
        high_grad_norm_stats = self.high_grad_norm_counter.get_stats()

        self.logger.debug(
            f"[{self.risk_type}] {stability_marker} 업데이트 {self.update_count} (강화모니터링): "
            f"손실(A:{actor_loss_stats['sliding_mean']:.2f}/C:{critic_loss_stats['sliding_mean']:.2f}) "
            f"Q범위[μ{q_min_stats['sliding_mean']:.1f},μ{q_max_stats['sliding_mean']:.1f}] "
            f"Q렉μ{q_range_stats['sliding_mean']:.1f} {q_monitor_info}"
            f"αμ{alpha_stats['sliding_mean']:.3f}(logα:{log_alpha_stats['sliding_mean']:.2f}) "
            f"H정책μ{policy_entropy_stats['sliding_mean']:.2f}/H목표μ{target_entropy_stats['sliding_mean']:.2f} "
            f"EP={getattr(self, 'episode_progress', 0.0):.1%} | "
            f"클립(α:{alpha_clipping_stats['sliding_rate']:.1%}, 그래드:{high_grad_norm_stats['sliding_rate']:.1%}) "
            f"이상(NaN:{nan_counter['sliding_rate']:.1%}, Q>{extreme_q_counter['sliding_rate']:.1%}, α>{high_alpha_counter['sliding_rate']:.1%})"
        )

        # 경고 체크
        if not is_stable:
            issues = []
            if critic_loss_stats["sliding_mean"] > 1e6:
                issues.append(f"C손실 {critic_loss_stats['sliding_mean']:.1e}")
            if q_range_stats["sliding_mean"] > 200:
                issues.append(f"Q범위 {q_range_stats['sliding_mean']:.1f}")
            if alpha_stats["sliding_mean"] > 1.0:
                issues.append(f"α {alpha_stats['sliding_mean']:.3f}")

            self.logger.warning(f"[{self.risk_type}] 안정성 문제: {', '.join(issues)}")

    def _check_stability_warnings_unified(self):
        """강화된 통합 슬라이딩 윈도우 기반 안정성 경고"""
        if self.update_count < 50:
            return

        # Alpha 관련 경고 (확장됨)
        alpha_stats = self.alpha_stats.get_stats()
        log_alpha_stats = self.log_alpha_stats.get_stats()
        alpha_clipping_stats = self.alpha_clipping_counter.get_stats()
        
        if alpha_stats["sliding_size"] >= 25:
            alpha_mean = alpha_stats["sliding_mean"]
            alpha_std = alpha_stats["sliding_std"]
            log_alpha_mean = log_alpha_stats["sliding_mean"]

            if alpha_mean > 1.0:
                self.logger.warning(
                    f"[통합] 높은 온도 계수: α={alpha_mean:.4f}±{alpha_std:.4f} (log_α={log_alpha_mean:.2f})"
                )
            elif alpha_mean < 1e-3:
                self.logger.warning(
                    f"[통합] 낮은 온도 계수: α={alpha_mean:.4f}±{alpha_std:.4f} (log_α={log_alpha_mean:.2f})"
                )

        # 알파 클리핑 경고
        if alpha_clipping_stats["sliding_rate"] > 0.2:  # 20% 초과시
            self.logger.warning(
                f"[통합] 높은 α 클리핑 비율: {alpha_clipping_stats['sliding_rate']:.1%} "
                f"({alpha_clipping_stats['sliding_count']}/{alpha_clipping_stats['sliding_size']})"
            )

        # 엔트로피 불일치 경고
        policy_entropy_stats = self.policy_entropy_stats.get_stats()
        target_entropy_stats = self.target_entropy_stats.get_stats()
        
        if (policy_entropy_stats["sliding_size"] >= 25 and 
            target_entropy_stats["sliding_size"] >= 25):
            policy_H = policy_entropy_stats["sliding_mean"]
            target_H = target_entropy_stats["sliding_mean"] 
            entropy_gap = abs(policy_H - target_H)
            
            if entropy_gap > 10.0:  # 큰 엔트로피 차이
                self.logger.warning(
                    f"[통합] 큰 엔트로피 차이: 정책H={policy_H:.2f}, 목표H={target_H:.2f} (차이:{entropy_gap:.2f})"
                )

        # Q-value 범위 경고 (확장됨)
        q_range_stats = self.q_range_stats.get_stats()
        q_target_range_stats = self.q_target_range_stats.get_stats()
        
        if q_range_stats["sliding_size"] >= 25:
            q_range_mean = q_range_stats["sliding_mean"]
            if q_range_mean > 150:  # 임계값 상향
                q_target_mean = q_target_range_stats["sliding_mean"]
                self.logger.warning(
                    f"[통합] Q-value 범위 확대: Q범위={q_range_mean:.2f}, Q타깃범위={q_target_mean:.2f}"
                )

        # 그라디언트 노름 경고
        high_grad_norm_stats = self.high_grad_norm_counter.get_stats()
        if high_grad_norm_stats["sliding_rate"] > 0.3:  # 30% 초과시
            self.logger.warning(
                f"[통합] 높은 그라디언트 노름 비율: {high_grad_norm_stats['sliding_rate']:.1%} "
                f"({high_grad_norm_stats['sliding_count']}/{high_grad_norm_stats['sliding_size']})"
            )

        # 이상치 비율 경고 (기존 + 개선)
        nan_stats = self.nan_loss_counter.get_stats()
        extreme_q_stats = self.extreme_q_counter.get_stats()

        if nan_stats["sliding_rate"] > 0.03:  # 3% 초과 (더 엄격)
            self.logger.warning(
                f"[통합] 높은 NaN 비율: {nan_stats['sliding_rate']:.1%} "
                f"({nan_stats['sliding_count']}/{nan_stats['sliding_size']})"
            )

        if extreme_q_stats["sliding_rate"] > 0.15:  # 15% 초과 (완화)
            self.logger.warning(
                f"[통합] 높은 극단 Q-value 비율: {extreme_q_stats['sliding_rate']:.1%} "
                f"({extreme_q_stats['sliding_count']}/{extreme_q_stats['sliding_size']})"
            )

    def _soft_update_targets(self):
        """타겟 네트워크들 소프트 업데이트 (SAC - Critic만)"""
        # Target Critic 1 업데이트
        for target_param, param in zip(
            self.target_critic1.parameters(), self.critic1.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        # Target Critic 2 업데이트
        for target_param, param in zip(
            self.target_critic2.parameters(), self.critic2.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
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
            concentration, weights, _, _ = self.actor(state_tensor)
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
