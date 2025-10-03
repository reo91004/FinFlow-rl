# src/algorithms/offline/bc_agent.py

"""
Behavioral Cloning Agent for IRT Warm-start

목적: IQL을 대체하는 단순한 모방 학습
의존: torch, BCellIRTActor
사용처: TrainerIRT (Phase 1)
역할: Offline 데이터로 "reasonable starting point" 제공

핵심 차이점 (vs IQL):
- AWR 없음 → 전이 bias 없음
- Expectile 없음 → Q-value 스케일 불일치 없음
- 단순 MLE → 빠른 학습 (30 epochs, 2-3분)

수식:
θ* = argmax E[(s,a)~D] [log π_θ(a|s)]

Dirichlet Mixture:
π(a|s) = 1/M Σ_j Dir(a; α_j(s))
"""

import torch
import torch.nn as nn
from torch.distributions import Dirichlet
from typing import Dict
from src.utils.logger import FinFlowLogger


class BCAgent:
    """
    Simple Behavioral Cloning without AWR or Expectile

    BC는 offline 데이터의 행동을 단순 모방하여 정책 초기화.
    전이 학습 bias가 없어 IQL보다 안정적.
    """

    def __init__(self,
                 actor: nn.Module,
                 lr: float = 3e-4,
                 device: torch.device = torch.device('cpu'),
                 dirichlet_min: float = 1.0,
                 dirichlet_max: float = 100.0):
        """
        Args:
            actor: BCellIRTActor instance
            lr: Learning rate
            device: PyTorch device
            dirichlet_min: Dirichlet concentration minimum
            dirichlet_max: Dirichlet concentration maximum
        """
        self.actor = actor
        self.device = device
        self.dirichlet_min = dirichlet_min
        self.dirichlet_max = dirichlet_max
        self.logger = FinFlowLogger("BCAgent")

        # Optimizer
        self.optimizer = torch.optim.Adam(
            actor.parameters(),
            lr=lr,
            betas=(0.9, 0.999),  # 표준 Adam 설정
            weight_decay=1e-5    # 약한 L2 regularization
        )

        self.logger.info(f"BC Agent 초기화 - lr={lr}, device={device}, "
                        f"dirichlet_range=[{dirichlet_min}, {dirichlet_max}]")

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        BC 업데이트: Offline 행동의 log-likelihood 최대화

        Args:
            batch: Dict with 'states', 'actions'
                states: [B, state_dim]
                actions: [B, action_dim] (portfolio weights)

        Returns:
            Dict with loss, mean alpha, entropy
        """
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)

        # ===== Forward: M개 프로토타입의 Dirichlet parameters =====
        # BCellIRTActor.forward_prototypes() 사용
        # Returns: [B, M, N] where M=prototypes, N=assets
        alphas = self.actor.forward_prototypes(states)
        B, M, N = alphas.shape

        # ===== Mixture of Dirichlet: p(a|s) = 1/M Σ_j Dir(a; α_j) =====
        # Log-sum-exp trick for numerical stability
        log_probs = []

        for j in range(M):
            alpha_j = alphas[:, j, :]  # [B, N]

            # Dirichlet concentration 범위 제한 (수치 안정성)
            alpha_j_clamped = torch.clamp(alpha_j, min=self.dirichlet_min, max=self.dirichlet_max)

            # Dirichlet distribution
            dist_j = Dirichlet(alpha_j_clamped)

            # Log probability of observed actions
            log_prob_j = dist_j.log_prob(actions)  # [B]

            log_probs.append(log_prob_j)

        # Log-sum-exp for mixture: log(1/M Σ exp(log p_j))
        log_probs_stacked = torch.stack(log_probs, dim=0)  # [M, B]
        log_prob_mixture = torch.logsumexp(log_probs_stacked, dim=0) - torch.log(
            torch.tensor(M, dtype=torch.float32, device=self.device)
        )

        # ===== Negative Log-Likelihood Loss =====
        nll_loss = -log_prob_mixture.mean()

        # ===== Optimize =====
        self.optimizer.zero_grad()
        nll_loss.backward()

        # Gradient clipping (IQL보다 관대하게 설정)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)

        self.optimizer.step()

        # ===== Metrics (for monitoring) =====
        with torch.no_grad():
            mean_alpha = alphas.mean().item()

            # Entropy of mixture policy (approximation via sampling)
            # H(π) ≈ -E[log π(a)] where a ~ π
            sample_actions = []
            for _ in range(5):  # 5번 샘플링
                sample_log_probs = []
                for j in range(M):
                    alpha_j = alphas[:, j, :]
                    alpha_j_clamped = torch.clamp(alpha_j, min=self.dirichlet_min, max=self.dirichlet_max)
                    dist_j = Dirichlet(alpha_j_clamped)
                    action_sample = dist_j.sample()  # [B, N]

                    # Log prob of this sample under mixture
                    log_prob_sample = dist_j.log_prob(action_sample)
                    sample_log_probs.append(log_prob_sample)

                sample_log_probs_stacked = torch.stack(sample_log_probs, dim=0)
                sample_log_prob_mix = torch.logsumexp(sample_log_probs_stacked, dim=0) - torch.log(
                    torch.tensor(M, dtype=torch.float32, device=self.device)
                )
                sample_actions.append(-sample_log_prob_mix)

            approx_entropy = torch.stack(sample_actions).mean().item()

        return {
            'loss': nll_loss.item(),
            'mean_alpha': mean_alpha,
            'entropy': approx_entropy,
            'grad_norm': grad_norm.item()
        }

    def save(self, path: str):
        """BC 모델 저장 (actor만)"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
        self.logger.info(f"BC 모델 저장: {path}")

    def load(self, path: str):
        """BC 모델 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.logger.info(f"BC 모델 로드: {path}")
