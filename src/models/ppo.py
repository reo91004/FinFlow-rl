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
    CLIP_OBS
)

class PPO:
    """
    Proximal Policy Optimization (PPO) 알고리즘 클래스입니다.
    Actor-Critic 모델을 사용하여 포트폴리오 관리 문제를 학습합니다.
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
        model_path=MODEL_SAVE_PATH,
        logger=None,
        use_ema=True,
        ema_decay=0.99,
        use_lr_scheduler=True,
        use_early_stopping=True,
    ):

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.model_path = model_path
        self.logger = logger or logging.getLogger("PortfolioRL")  # 로거 없으면 기본 설정 사용
        self.n_assets = n_assets
        self.n_features = n_features  # 추가

        # EMA 가중치 옵션
        self.use_ema = use_ema
        self.ema_decay = ema_decay

        # 학습률 스케줄러 및 Early Stopping 설정
        self.use_lr_scheduler = use_lr_scheduler
        self.use_early_stopping = use_early_stopping
        self.early_stopping_patience = EARLY_STOPPING_PATIENCE
        self.best_validation_reward = -float("inf")
        self.no_improvement_episodes = 0
        self.should_stop_early = False

        os.makedirs(model_path, exist_ok=True)

        # 정책 네트워크 (현재 정책, 이전 정책)
        self.policy = ActorCritic(n_assets, n_features, hidden_dim).to(DEVICE)
        self.policy_old = ActorCritic(n_assets, n_features, hidden_dim).to(DEVICE)
        self.policy_old.load_state_dict(self.policy.state_dict())  # 가중치 복사

        # EMA 모델 (학습 안정성을 위한 Exponential Moving Average)
        if self.use_ema:
            self.policy_ema = ActorCritic(n_assets, n_features, hidden_dim).to(DEVICE)
            self.policy_ema.load_state_dict(self.policy.state_dict())
            # EMA 모델의 파라미터는 업데이트되지 않도록 설정
            for param in self.policy_ema.parameters():
                param.requires_grad = False

        # 옵티마이저 및 손실 함수
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.MseLoss = nn.MSELoss()  # 크리틱 손실용

        # 학습률 스케줄러 (Cosine Annealing)
        if self.use_lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=LR_SCHEDULER_T_MAX, eta_min=LR_SCHEDULER_ETA_MIN
            )
            self.logger.info(
                f"Cosine Annealing LR 스케줄러 설정: T_max={LR_SCHEDULER_T_MAX}, eta_min={LR_SCHEDULER_ETA_MIN}"
            )

        self.best_reward = -float("inf")  # 최고 성능 모델 저장을 위한 변수
        self.obs_rms = None  # 학습된 상태 정규화 통계 저장용

        # GPU 설정 (성능 향상 최적화 옵션)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # 행렬 곱셈 연산 정밀도 설정 (A100/H100 등 TensorFloat32 지원 시 유리)
            # torch.set_float32_matmul_precision('high') # 또는 'medium'

    def update_lr_scheduler(self):
        """학습률 스케줄러를 업데이트합니다."""
        if self.use_lr_scheduler and self.scheduler:
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
        if not self.use_early_stopping:
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
                f"인내심 카운터: {self.no_improvement_episodes}/{self.early_stopping_patience}"
            )

            # 인내심 카운터가 임계값을 넘으면 학습 중단
            if self.no_improvement_episodes >= self.early_stopping_patience:
                self.logger.warning(
                    f"Early Stopping 조건 충족! {self.early_stopping_patience} 에피소드 동안 "
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
        """훈련된 모델 가중치와 옵티마이저 상태, obs_rms 통계를 로드합니다."""
        if model_file is None:
            model_file = os.path.join(self.model_path, "best_model.pth")

        if not os.path.exists(model_file):
            self.logger.warning(f"모델 파일이 존재하지 않음: {model_file}")
            return False

        try:
            # GPU/CPU 환경에 맞춰 모델 로드
            if torch.cuda.is_available():
                checkpoint = torch.load(model_file)
            else:
                checkpoint = torch.load(model_file, map_location=torch.device("cpu"))

            self.policy.load_state_dict(checkpoint["model_state_dict"])
            self.policy_old.load_state_dict(checkpoint["model_state_dict"])

            if self.use_ema and "ema_state_dict" in checkpoint:
                # EMA 모델 가중치가 저장되어 있으면 로드
                self.policy_ema.load_state_dict(checkpoint["ema_state_dict"])
                self.logger.info("EMA 모델 가중치 로드 완료")

            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # 상태 정규화 통계 로드 (존재하는 경우)
            if "obs_rms_mean" in checkpoint and "obs_rms_var" in checkpoint and "obs_rms_count" in checkpoint:
                from src.models.running_mean_std import RunningMeanStd
                
                # 모델이 저장된 시점의 obs_rms shape 확인
                saved_shape = checkpoint["obs_rms_mean"].shape
                expected_shape = (self.n_assets, self.n_features)
                
                if saved_shape != expected_shape:
                    self.logger.error(
                        f"저장된 obs_rms shape({saved_shape})이 현재 모델의 expected shape({expected_shape})과 일치하지 않음. "
                        f"이로 인해 정규화가 부정확할 수 있음."
                    )
                    return False
                
                # 상태 정규화 통계 복원
                self.obs_rms = RunningMeanStd(shape=expected_shape)
                self.obs_rms.mean = checkpoint["obs_rms_mean"]
                self.obs_rms.var = checkpoint["obs_rms_var"]
                self.obs_rms.count = checkpoint["obs_rms_count"]
                self.logger.info("상태 정규화 통계(obs_rms) 로드 완료")

            self.logger.info(f"모델 로드 완료: {model_file}")

            # 로드된 best_reward 값이 있으면 업데이트
            if "best_reward" in checkpoint:
                self.best_reward = checkpoint["best_reward"]
                self.logger.info(f"최고 보상값 로드됨: {self.best_reward:.4f}")

            return True

        except Exception as e:
            self.logger.error(f"모델 로드 중 오류 발생: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
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
        """메모리에 저장된 경험을 사용하여 정책(policy)을 업데이트합니다."""
        if not memory.states:
            self.logger.warning("업데이트 시도: 메모리가 비어있습니다.")
            return 0.0

        total_loss_val = 0.0

        try:
            old_states = torch.stack(
                [torch.from_numpy(s).float() for s in memory.states]
            ).to(DEVICE)
            old_actions = torch.stack(
                [torch.from_numpy(a).float() for a in memory.actions]
            ).to(DEVICE)
            old_logprobs = torch.tensor(memory.logprobs, dtype=torch.float32).to(DEVICE)
            old_values = torch.tensor(memory.values, dtype=torch.float32).to(DEVICE)

            old_values_np = old_values.cpu().numpy()
            returns, advantages = self.compute_returns_and_advantages(
                memory.rewards, memory.is_terminals, old_values_np
            )

            if returns.numel() == 0 or advantages.numel() == 0:
                self.logger.error("GAE 계산 실패로 PPO 업데이트 중단.")
                return 0.0

            adv_mean = advantages.mean()
            adv_std = advantages.std()
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)

            if torch.isnan(advantages).any() or torch.isinf(advantages).any():
                self.logger.warning("Advantage 정규화 후 NaN/Inf 발견. 0으로 대체.")
                advantages = torch.nan_to_num(advantages, nan=0.0)

            for _ in range(self.k_epochs):
                logprobs, entropy, state_values = self.policy.evaluate(
                    old_states, old_actions
                )
                ratios = torch.exp(logprobs - old_logprobs.detach())
                surr1 = ratios * advantages
                surr2 = (
                    torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
                    * advantages
                )
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = self.MseLoss(state_values, returns)
                entropy_loss = entropy.mean()
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_loss

                if torch.isnan(loss) or torch.isinf(loss):
                    self.logger.error(
                        f"손실 계산 중 NaN/Inf 발생! Actor: {actor_loss.item()}, Critic: {critic_loss.item()}, Entropy: {entropy_loss.item()}. 해당 배치 업데이트 건너뛰었습니다."
                    )
                    total_loss_val = 0.0
                    break

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()
                total_loss_val += loss.item()

                # 온도 파라미터 업데이트
                self.policy.update_temperature()

                # EMA 모델 가중치 업데이트
                if self.use_ema:
                    self.update_ema_model()

            if total_loss_val != 0.0 or self.k_epochs == 0:
                self.policy_old.load_state_dict(self.policy.state_dict())
                return total_loss_val / self.k_epochs if self.k_epochs > 0 else 0.0
            else:
                return 0.0

        except Exception as e:
            self.logger.error(f"PPO 업데이트 중 예상치 못한 오류 발생: {e}")
            self.logger.error(traceback.format_exc())
            return 0.0 