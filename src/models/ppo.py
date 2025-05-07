"""
Proximal Policy Optimization(PPO) 알고리즘 모듈

주식 포트폴리오 관리를 위한 PPO 알고리즘을 구현합니다.
Actor-Critic 모델을 사용하여 학습하며, EMA(Exponential Moving Average) 모델,
Early stopping, 학습률 스케줄러 등 다양한 안정화 기법을 적용합니다.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import traceback
import gc
from src.models.actor_critic import ActorCritic
from src.constants import (
    DEFAULT_HIDDEN_DIM,
    DEFAULT_LR,
    DEFAULT_GAMMA,
    DEFAULT_K_EPOCHS,
    DEFAULT_EPS_CLIP,
    MODEL_SAVE_PATH,
    DEVICE,
    EARLY_STOPPING_PATIENCE,
    VALIDATION_INTERVAL,
    VALIDATION_EPISODES,
    LR_SCHEDULER_T_MAX,
    LR_SCHEDULER_ETA_MIN,
    LAMBDA_GAE,
    RMS_EPSILON,
    CLIP_OBS,
    DEFAULT_ENTROPY_COEF,
    DEFAULT_CRITIC_COEF,
    DEFAULT_GAE_LAMBDA,
    DEFAULT_MAX_GRAD_NORM,
    PPO_BATCH_SIZE,
)

class PPO:
    """
    PPO(Proximal Policy Optimization) 알고리즘 구현.
    
    참고: https://arxiv.org/abs/1707.06347
    
    주요 기능:
    - 정책 네트워크와 가치 네트워크 공유 (actor-critic architecture)
    - 다중 에폭 업데이트
    - GAE(Generalized Advantage Estimation) 사용
    - 입력값 스케일링 및 정규화
    - EMA(Exponential Moving Average) 모델 사용 가능
    """

    def __init__(
        self,
        n_assets,
        n_features,
        hidden_dim=DEFAULT_HIDDEN_DIM,
        lr=DEFAULT_LR,
        gamma=DEFAULT_GAMMA,
        k_epochs=DEFAULT_K_EPOCHS,
        eps_clip=DEFAULT_EPS_CLIP,
        entropy_coef=DEFAULT_ENTROPY_COEF,
        critic_coef=DEFAULT_CRITIC_COEF,
        gae_lambda=DEFAULT_GAE_LAMBDA,
        max_grad_norm=DEFAULT_MAX_GRAD_NORM,
        batch_size=PPO_BATCH_SIZE,
        use_ema=True,
        ema_decay=0.995,
        use_scheduler=True,
        checkpoint_dir=None,
    ):
        """
        초기화 함수.
        
        Args:
            n_assets: 자산 개수
            n_features: 입력 특성 개수
            hidden_dim: 은닉층의 뉴런 수
            lr: 학습률
            gamma: 할인율
            k_epochs: 각 업데이트마다 에폭 수
            eps_clip: PPO 클리핑 파라미터
            use_ema: EMA 모델 사용 여부
            ema_decay: EMA 모델의 감쇠율
            use_scheduler: 학습률 스케줄러 사용 여부
            checkpoint_dir: 체크포인트 저장 디렉토리
        """
        self.n_assets = n_assets
        self.n_features = n_features
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.entropy_coef = entropy_coef
        self.critic_coef = critic_coef
        self.gae_lambda = gae_lambda
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.use_scheduler = use_scheduler
        self.checkpoint_dir = checkpoint_dir
        self.best_reward = -float("inf")
        self.model_path = checkpoint_dir or os.path.join(os.getcwd(), "models")
        
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        # 정책과 가치 네트워크
        self.policy = ActorCritic(n_assets, n_features, hidden_dim=hidden_dim).to(device)
        
        # EMA 모델 (사용 시)
        if self.use_ema:
            self.policy_ema = ActorCritic(n_assets, n_features, hidden_dim=hidden_dim).to(device)
            # 가중치 복사
            for ema_param, param in zip(self.policy_ema.parameters(), self.policy.parameters()):
                ema_param.data.copy_(param.data)
            # 기울기 계산 비활성화
            for param in self.policy_ema.parameters():
                param.requires_grad = False
        
        # 옵티마이저
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # 학습률 스케줄러 (사용 시)
        if self.use_scheduler:
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.95)

        self.best_validation_reward = -float("inf")
        self.no_improvement_episodes = 0
        self.should_stop_early = False
        self.obs_rms = None

        # GPU 설정 (성능 향상 최적화 옵션)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # 행렬 곱셈 연산 정밀도 설정 (A100/H100 등 TensorFloat32 지원 시 유리)
            # torch.set_float32_matmul_precision('high') # 또는 'medium'

    def update_lr_scheduler(self):
        """학습률 스케줄러를 업데이트합니다."""
        if self.use_scheduler and self.scheduler:
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            return current_lr
        return None

    def validate(self, env, n_episodes=VALIDATION_EPISODES):
        """
        현재 정책을 검증하여 Early Stopping에 사용할 보상을 계산합니다.

        Args:
            env: 검증에 사용할 환경 (StockPortfolioEnv)
            n_episodes: 실행할 검증 에피소드 수

        Returns:
            float: 평균 검증 보상
        """
        # 평가 모드로 설정
        self.policy_old.eval()
        if self.use_ema:
            self.policy_ema.eval()

        total_reward = 0

        for _ in range(n_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False

            while not done:
                # EMA 모델 사용 (있는 경우)
                if self.use_ema:
                    with torch.no_grad():
                        state_tensor = (
                            torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
                        )
                        action_probs, _ = self.policy_ema(state_tensor)
                        dist = torch.distributions.Categorical(action_probs)
                        action_idx = dist.sample()

                        # 원-핫 인코딩으로 변환
                        action = torch.zeros_like(action_probs)
                        action.scatter_(1, action_idx.unsqueeze(-1), 1.0)
                        action = action.squeeze(0).cpu().numpy()
                else:
                    action, _, _ = self.policy_old.act(state)

                next_state, reward, terminated, truncated, info = env.step(action)
                episode_reward += info.get("raw_reward", reward)

                if terminated or truncated:
                    done = True
                else:
                    state = next_state

            total_reward += episode_reward

        # 학습 모드로 복원
        self.policy_old.train()
        if self.use_ema:
            self.policy_ema.train()

        # 평균 검증 보상 반환
        return total_reward / n_episodes

    def check_early_stopping(self, validation_reward):
        """
        검증 보상에 기반하여 Early Stopping 여부를 확인합니다.

        Args:
            validation_reward: 현재 검증 보상

        Returns:
            bool: True면 학습 중단, False면 계속 진행
        """
        if not self.use_scheduler:
            return False

        if validation_reward > self.best_validation_reward:
            # 성능 향상이 있으면 최고 기록 갱신 및 인내심 카운터 리셋
            self.best_validation_reward = validation_reward
            self.no_improvement_episodes = 0
            return False
        else:
            # 성능 향상이 없으면 인내심 카운터 증가
            self.no_improvement_episodes += 1

            # 로깅
            self.logger.info(
                f"최고 검증 보상 {self.best_validation_reward:.4f} 대비 향상 없음. "
                f"인내심 카운터: {self.no_improvement_episodes}/{EARLY_STOPPING_PATIENCE}"
            )

            # 인내심 카운터가 임계값을 넘으면 학습 중단
            if self.no_improvement_episodes >= EARLY_STOPPING_PATIENCE:
                self.logger.warning(
                    f"Early Stopping 조건 충족! {EARLY_STOPPING_PATIENCE} 에피소드 동안 "
                    f"성능 향상 없음. 최고 검증 보상: {self.best_validation_reward:.4f}"
                )
                self.should_stop_early = True
                return True

        return False

    def update_ema_model(self):
        """
        EMA(Exponential Moving Average) 모델의 가중치를 업데이트합니다.
        ema_weight = decay * ema_weight + (1 - decay) * current_weight
        """
        if not self.use_ema:
            return

        with torch.no_grad():
            for ema_param, current_param in zip(
                self.policy_ema.parameters(), self.policy.parameters()
            ):
                ema_param.data.mul_(self.ema_decay).add_(
                    current_param.data, alpha=1.0 - self.ema_decay
                )

    def save_model(self, episode, reward):
        """최고 성능 모델의 가중치와 옵티마이저 상태, obs_rms 통계를 저장합니다."""
        if reward > self.best_reward:
            self.best_reward = reward
            save_file = os.path.join(self.model_path, "best_model.pth")
            try:
                # 저장할 데이터 구성
                checkpoint = {
                    "episode": episode,
                    "model_state_dict": self.policy.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_reward": self.best_reward,
                }

                # EMA 모델이 있으면 EMA 상태도 저장
                if self.use_ema:
                    checkpoint["ema_model_state_dict"] = self.policy_ema.state_dict()
                    checkpoint["ema_decay"] = self.ema_decay

                # obs_rms가 있으면 통계량 추가
                if self.obs_rms is not None:
                    checkpoint.update(
                        {
                            "obs_rms_mean": self.obs_rms.mean,
                            "obs_rms_var": self.obs_rms.var,
                            "obs_rms_count": self.obs_rms.count,
                        }
                    )

                torch.save(checkpoint, save_file)
                self.logger.info(
                    f"새로운 최고 성능 모델 저장! 에피소드: {episode}, 보상: {reward:.4f} -> {save_file}"
                )
            except Exception as e:
                self.logger.error(f"모델 저장 중 오류 발생: {e}")

    def load_model(self, model_file=None):
        """저장된 모델 가중치와 옵티마이저 상태, 상태 정규화 통계를 불러옵니다."""
        if model_file is None:
            model_file = os.path.join(self.model_path, "best_model.pth")

        if not os.path.exists(model_file):
            self.logger.warning(f"저장된 모델 파일 없음: {model_file}")
            return False

        try:
            checkpoint = torch.load(model_file, map_location=DEVICE, weights_only=False)

            self.policy.load_state_dict(checkpoint["model_state_dict"])
            self.policy_old.load_state_dict(checkpoint["model_state_dict"])

            # EMA 모델 로드 (있는 경우)
            if self.use_ema and "ema_model_state_dict" in checkpoint:
                self.policy_ema.load_state_dict(checkpoint["ema_model_state_dict"])
                self.ema_decay = checkpoint.get("ema_decay", self.ema_decay)
                self.logger.info(f"EMA 모델 로드 완료 (decay: {self.ema_decay})")
            elif self.use_ema:
                # EMA 모델이 저장되지 않았으면 일반 모델로 초기화
                self.policy_ema.load_state_dict(checkpoint["model_state_dict"])
                self.logger.info("EMA 모델이 저장되지 않아 일반 모델로 초기화됨")

            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            self.best_reward = checkpoint.get("best_reward", -float("inf"))

            if "obs_rms_mean" in checkpoint and checkpoint["obs_rms_mean"] is not None:
                from src.models.running_mean_std import RunningMeanStd
                if self.obs_rms is None:
                    self.obs_rms = RunningMeanStd(
                        shape=(self.n_assets, self.n_features)
                    )
                self.obs_rms.mean = checkpoint["obs_rms_mean"]
                self.obs_rms.var = checkpoint["obs_rms_var"]
                self.obs_rms.count = checkpoint["obs_rms_count"]
                self.logger.info("저장된 상태 정규화(obs_rms) 통계 로드 완료.")
            else:
                self.obs_rms = None

            self.logger.info(
                f"모델 로드 성공! ({model_file}), 최고 보상: {self.best_reward:.4f}"
            )
            return True

        except (KeyError, TypeError) as load_err:
            self.logger.warning(
                f"모델 파일 로드 중 오류 ({model_file}): {load_err}. 가중치만 로드 시도합니다."
            )
            try:
                weights = torch.load(model_file, map_location=DEVICE, weights_only=True)
                self.policy.load_state_dict(weights)
                self.policy_old.load_state_dict(weights)
                if self.use_ema:
                    self.policy_ema.load_state_dict(weights)
                self.logger.info(
                    f"모델 가중치 로드 성공 (weights_only=True)! ({model_file})"
                )
                self.best_reward = -float("inf")
                self.obs_rms = None
                return True
            except Exception as e_inner:
                self.logger.error(
                    f"weights_only=True 로도 모델 로드 실패 ({model_file}): {e_inner}"
                )
                return False
        except Exception as e:
            self.logger.error(f"모델 로드 중 예상치 못한 오류 발생 ({model_file}): {e}")
            return False

    def select_action(self, state, use_ema=True):
        """
        추론 시 액션 선택 (EMA 모델 사용 옵션 추가)
        use_ema=True면 EMA 모델 사용, False면 일반 모델 사용
        """
        if self.use_ema and use_ema:
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
                concentration, _ = self.policy_ema(state_tensor)
                dist = torch.distributions.Dirichlet(concentration)
                action = dist.mean  # 평균값 사용 (샘플링 없이 결정론적)
                return action.squeeze(0).cpu().numpy()
        else:
            action, _, _ = self.policy_old.act(state)
            return action

    def compute_returns_and_advantages(self, rewards, is_terminals, values):
        """
        Generalized Advantage Estimation (GAE)를 사용하여 Advantage와 Return을 계산합니다.

        Args:
            rewards (list): 에피소드/배치에서 얻은 보상 리스트.
            is_terminals (list): 각 스텝의 종료 여부 리스트.
            values (np.ndarray): 각 상태에 대한 크리틱의 가치 예측값 배열.

        Returns:
            tuple: (returns_tensor, advantages_tensor)
                   - returns_tensor (torch.Tensor): 계산된 Return (Target Value).
                   - advantages_tensor (torch.Tensor): 계산된 Advantage.
                   오류 발생 시 빈 텐서 반환.
        """
        if not rewards or values.size == 0:
            self.logger.warning("GAE 계산 시 rewards 또는 values 배열이 비어있습니다.")
            return torch.tensor([], device=DEVICE), torch.tensor([], device=DEVICE)

        returns = np.zeros_like(rewards, dtype=np.float32)
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae_lam = 0.0

        next_value = values[-1] * (1.0 - float(is_terminals[-1]))

        for i in reversed(range(len(rewards))):
            mask = 1.0 - float(is_terminals[i])
            delta = rewards[i] + self.gamma * next_value * mask - values[i]
            last_gae_lam = delta + self.gamma * LAMBDA_GAE * mask * last_gae_lam
            advantages[i] = last_gae_lam
            returns[i] = last_gae_lam + values[i]
            next_value = values[i]

        try:
            returns_tensor = torch.from_numpy(returns).float().to(DEVICE)
            advantages_tensor = torch.from_numpy(advantages).float().to(DEVICE)
        except Exception as e:
            self.logger.error(f"Return/Advantage 텐서 변환 중 오류: {e}")
            return torch.tensor([], device=DEVICE), torch.tensor([], device=DEVICE)

        if torch.isnan(returns_tensor).any() or torch.isinf(returns_tensor).any():
            returns_tensor = torch.nan_to_num(returns_tensor, nan=0.0)
        if torch.isnan(advantages_tensor).any() or torch.isinf(advantages_tensor).any():
            advantages_tensor = torch.nan_to_num(advantages_tensor, nan=0.0)

        return returns_tensor, advantages_tensor

    def update(self, memory):
        """
        PPO 정책을 메모리에 있는 트랜지션을 사용하여 업데이트합니다.
        
        Args:
            memory: 학습에 사용할 경험 저장 객체
            
        Returns:
            평균 정책 손실
        """
        # 메모리에서 트랜지션 추출
        old_states = memory.states.to(device).detach()
        old_actions = memory.actions.to(device).detach()
        old_logprobs = memory.logprobs.to(device).detach()
        old_rewards = memory.rewards.to(device).detach()
        old_dones = memory.dones.to(device).detach()

        # 메모리가 충분히 채워지지 않았을 경우
        if len(old_states) < 10:  # 최소 배치 크기
            return 0.0

        # GAE(Generalized Advantage Estimation) 계산
        advantages = []
        gae = 0
        with torch.no_grad():
            values = self.policy.critic(old_states).squeeze()
            next_value = values[-1].item()  # 마지막 상태의 가치

            for i in reversed(range(len(old_rewards))):
                if old_dones[i]:
                    next_value = 0  # 에피소드 종료 시 다음 가치는 0
                
                # 델타 = 보상 + 감마 * 다음 가치 - 현재 가치
                delta = old_rewards[i] + self.gamma * next_value * (1 - old_dones[i]) - values[i]
                
                # GAE = 델타 + 감마 * 감쇠 계수 * 다음 GAE (에피소드 종료 시 0)
                gae = delta + self.gamma * self.gae_lambda * (1 - old_dones[i]) * gae
                advantages.insert(0, gae)
                
                next_value = values[i].item()
                
        advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
        
        # Returns = Advantages + Values
        returns = advantages + values
        
        # 정규화 (평균 0, 분산 1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 미니배치 업데이트
        loss_epochs = 0
        num_samples = len(old_states)
        num_batches = max(num_samples // self.batch_size, 1)
        
        for _ in range(self.k_epochs):
            # 데이터 셔플
            perm = torch.randperm(num_samples)
            
            for i in range(num_batches):
                # 미니배치 인덱스
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, num_samples)
                idx = perm[start_idx:end_idx]
                
                # 미니배치 데이터
                batch_states = old_states[idx]
                batch_actions = old_actions[idx]
                batch_logprobs = old_logprobs[idx]
                batch_advantages = advantages[idx]
                batch_returns = returns[idx]
                
                # 현재 정책에서의 로그 확률 및 엔트로피 계산
                mu, sigma = self.policy.actor(batch_states)
                dist = torch.distributions.Normal(mu, sigma)
                curr_logprobs = dist.log_prob(batch_actions).sum(1)
                entropy = dist.entropy().sum(1).mean()
                
                # 현재 가치 함수값 계산
                curr_values = self.policy.critic(batch_states).squeeze()
                
                # PPO 비율 계산
                ratios = torch.exp(curr_logprobs - batch_logprobs)
                
                # 서로게이트 손실 계산
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # 가치 함수 손실 계산
                critic_loss = F.mse_loss(curr_values, batch_returns)
                
                # 전체 손실 계산 (액터, 크리틱, 엔트로피)
                loss = actor_loss + self.critic_coef * critic_loss - self.entropy_coef * entropy
                
                # 그래디언트 계산 및 업데이트
                self.optimizer.zero_grad()
                loss.backward()
                
                # 그래디언트 클리핑
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    
                self.optimizer.step()
                
                loss_epochs += loss.item()
                
        # EMA 모델 업데이트
        if self.use_ema:
            self.update_ema_model()
        
        # 학습률 스케줄러 업데이트
        if self.use_scheduler:
            self.scheduler.step()
            
        # 평균 손실 반환
        avg_loss = loss_epochs / (self.k_epochs * num_batches)
        return avg_loss 