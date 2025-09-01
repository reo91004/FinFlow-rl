# core/advantage.py

import torch
import numpy as np
from typing import List, Tuple, Union


def compute_gae(
    rewards: Union[List, np.ndarray, torch.Tensor],
    values: Union[List, np.ndarray, torch.Tensor], 
    gamma: float = 0.99,
    lambda_gae: float = 0.95,
    normalize: bool = True
) -> torch.Tensor:
    """
    Generalized Advantage Estimation (GAE) 계산
    
    논문: "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
    목적: 정책 그래디언트의 분산을 줄이면서 바이어스를 제어
    
    Args:
        rewards: 보상 시퀀스 [T,]
        values: 상태 가치 시퀀스 [T+1,] (마지막은 다음 상태 가치)
        gamma: 할인 인수
        lambda_gae: GAE λ 파라미터 (바이어스-분산 절충)
        normalize: 어드밴티지 정규화 여부
        
    Returns:
        advantages: GAE 어드밴티지 [T,]
    """
    # 텐서 변환 및 장치 처리
    if isinstance(rewards, (list, np.ndarray)):
        rewards = torch.tensor(rewards, dtype=torch.float32)
    if isinstance(values, (list, np.ndarray)):
        values = torch.tensor(values, dtype=torch.float32)
    
    device = rewards.device
    T = len(rewards)
    
    if len(values) != T + 1:
        raise ValueError(f"values 길이({len(values)})는 rewards 길이({T}) + 1이어야 합니다")
    
    # TD 오차 계산
    # δ_t = r_t + γV(s_{t+1}) - V(s_t)
    deltas = rewards + gamma * values[1:] - values[:-1]
    
    # GAE 계산 (역방향)
    advantages = torch.zeros(T, device=device)
    advantage = 0.0
    
    for t in reversed(range(T)):
        advantage = deltas[t] + gamma * lambda_gae * advantage
        advantages[t] = advantage
    
    # 어드밴티지 정규화 (0 평균, 단위 분산)
    if normalize and T > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return advantages


def compute_gae_with_dones(
    rewards: Union[List, np.ndarray, torch.Tensor],
    values: Union[List, np.ndarray, torch.Tensor],
    dones: Union[List, np.ndarray, torch.Tensor],
    gamma: float = 0.99,
    lambda_gae: float = 0.95,
    normalize: bool = True
) -> torch.Tensor:
    """
    에피소드 종료를 고려한 GAE 계산
    
    Args:
        rewards: 보상 시퀀스 [T,]
        values: 상태 가치 시퀀스 [T+1,]
        dones: 에피소드 종료 플래그 [T,]
        gamma: 할인 인수
        lambda_gae: GAE λ 파라미터
        normalize: 어드밴티지 정규화 여부
        
    Returns:
        advantages: GAE 어드밴티지 [T,]
    """
    # 텐서 변환
    if isinstance(rewards, (list, np.ndarray)):
        rewards = torch.tensor(rewards, dtype=torch.float32)
    if isinstance(values, (list, np.ndarray)):
        values = torch.tensor(values, dtype=torch.float32)
    if isinstance(dones, (list, np.ndarray)):
        dones = torch.tensor(dones, dtype=torch.bool)
    
    device = rewards.device
    T = len(rewards)
    
    # TD 오차 계산 (에피소드 종료 시 다음 상태 가치를 0으로)
    next_values = values[1:] * (~dones)  # done=True면 다음 가치는 0
    deltas = rewards + gamma * next_values - values[:-1]
    
    # GAE 계산 (역방향, 에피소드 경계 고려)
    advantages = torch.zeros(T, device=device)
    advantage = 0.0
    
    for t in reversed(range(T)):
        advantage = deltas[t] + gamma * lambda_gae * advantage * (~dones[t])
        advantages[t] = advantage
    
    # 어드밴티지 정규화
    if normalize and T > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return advantages


def compute_returns(
    rewards: Union[List, np.ndarray, torch.Tensor],
    gamma: float = 0.99,
    normalize: bool = False
) -> torch.Tensor:
    """
    할인된 누적 보상(returns) 계산
    
    Args:
        rewards: 보상 시퀀스 [T,]
        gamma: 할인 인수
        normalize: 반환값 정규화 여부
        
    Returns:
        returns: 할인된 누적 보상 [T,]
    """
    if isinstance(rewards, (list, np.ndarray)):
        rewards = torch.tensor(rewards, dtype=torch.float32)
    
    device = rewards.device
    T = len(rewards)
    returns = torch.zeros(T, device=device)
    
    # 역방향 계산
    running_return = 0.0
    for t in reversed(range(T)):
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return
    
    # 정규화 (선택적)
    if normalize and T > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    return returns


def compute_lambda_returns(
    rewards: Union[List, np.ndarray, torch.Tensor],
    values: Union[List, np.ndarray, torch.Tensor],
    gamma: float = 0.99,
    lambda_ret: float = 0.95
) -> torch.Tensor:
    """
    λ-반환(λ-returns) 계산
    
    Args:
        rewards: 보상 시퀀스 [T,]
        values: 상태 가치 시퀀스 [T+1,]
        gamma: 할인 인수
        lambda_ret: λ 파라미터
        
    Returns:
        lambda_returns: λ-반환 [T,]
    """
    # 먼저 GAE 계산
    advantages = compute_gae(rewards, values, gamma, lambda_ret, normalize=False)
    
    # λ-반환 = 어드밴티지 + 가치
    values_tensor = values[:-1] if isinstance(values, torch.Tensor) else torch.tensor(values[:-1], dtype=torch.float32)
    lambda_returns = advantages + values_tensor
    
    return lambda_returns


class AdvantageEstimator:
    """
    어드밴티지 추정 클래스 (배치 처리 및 통계 추적)
    """
    
    def __init__(self, gamma: float = 0.99, lambda_gae: float = 0.95):
        """
        Args:
            gamma: 할인 인수
            lambda_gae: GAE λ 파라미터
        """
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        
        # 통계 추적
        self.advantage_history = []
        self.return_history = []
        
    def compute_batch_advantages(
        self,
        batch_rewards: List[List],
        batch_values: List[List],
        batch_dones: List[List] = None,
        normalize_per_episode: bool = False
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        배치 단위로 어드밴티지와 반환값 계산
        
        Args:
            batch_rewards: 에피소드별 보상 리스트
            batch_values: 에피소드별 가치 리스트
            batch_dones: 에피소드별 종료 플래그 리스트
            normalize_per_episode: 에피소드별 정규화 여부
            
        Returns:
            batch_advantages: 에피소드별 어드밴티지 리스트
            batch_returns: 에피소드별 λ-반환 리스트
        """
        batch_advantages = []
        batch_returns = []
        
        for i, (rewards, values) in enumerate(zip(batch_rewards, batch_values)):
            if batch_dones is not None and i < len(batch_dones):
                dones = batch_dones[i]
                advantages = compute_gae_with_dones(
                    rewards, values, dones, 
                    self.gamma, self.lambda_gae, 
                    normalize=normalize_per_episode
                )
            else:
                advantages = compute_gae(
                    rewards, values, 
                    self.gamma, self.lambda_gae, 
                    normalize=normalize_per_episode
                )
            
            # λ-반환 계산
            lambda_returns = compute_lambda_returns(
                rewards, values, self.gamma, self.lambda_gae
            )
            
            batch_advantages.append(advantages)
            batch_returns.append(lambda_returns)
            
            # 통계 저장
            self.advantage_history.extend(advantages.tolist())
            self.return_history.extend(lambda_returns.tolist())
        
        return batch_advantages, batch_returns
    
    def get_statistics(self) -> dict:
        """어드밴티지 추정 통계 반환"""
        if not self.advantage_history:
            return {}
        
        advantages = np.array(self.advantage_history[-1000:])  # 최근 1000개만
        returns = np.array(self.return_history[-1000:])
        
        return {
            'advantage_mean': advantages.mean(),
            'advantage_std': advantages.std(),
            'advantage_min': advantages.min(),
            'advantage_max': advantages.max(),
            'return_mean': returns.mean(),
            'return_std': returns.std(),
            'return_min': returns.min(),
            'return_max': returns.max(),
            'samples': len(advantages)
        }
    
    def reset_history(self):
        """통계 히스토리 초기화"""
        self.advantage_history = []
        self.return_history = []