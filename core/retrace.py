# core/retrace.py

import torch
import numpy as np
from typing import Tuple, Optional


def retrace_targets(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    behavior_probs: torch.Tensor,
    target_probs: torch.Tensor,
    gamma: float = 0.99,
    lambda_param: float = 0.95,
    c_clip: float = 1.0
) -> torch.Tensor:
    """
    Retrace(λ) 타깃 계산
    
    논문: "Safe and efficient off-policy reinforcement learning"
    목적: 오프폴리시 학습에서 중요도 샘플링의 고분산 문제 해결
    
    Args:
        rewards: 보상 [T,]
        values: 현재 상태의 값 [T,]
        next_values: 다음 상태의 값 [T,]
        dones: 종료 플래그 [T,]
        behavior_probs: 행동 정책 확률 [T,]
        target_probs: 타깃 정책 확률 [T,]
        gamma: 할인 인수
        lambda_param: Retrace λ 파라미터
        c_clip: 중요도 가중치 클리핑 상한
    
    Returns:
        retrace_targets: Retrace 타깃 값들 [T,]
    """
    device = rewards.device
    T = len(rewards)
    
    if T == 0:
        return torch.empty(0, device=device)
    
    # 중요도 가중치 계산 및 클리핑
    importance_weights = target_probs / (behavior_probs + 1e-8)
    c_values = lambda_param * torch.clamp(importance_weights, max=c_clip)
    
    # Retrace 타깃 계산 (역방향)
    targets = torch.zeros_like(values)
    
    # 마지막 스텝
    targets[-1] = rewards[-1] + gamma * next_values[-1] * (~dones[-1])
    
    # 역방향 계산
    for t in range(T - 2, -1, -1):
        if dones[t]:
            # 에피소드 종료 시
            targets[t] = rewards[t]
        else:
            # Retrace 재귀식
            td_error = rewards[t] + gamma * next_values[t] - values[t]
            targets[t] = values[t] + td_error + gamma * c_values[t + 1] * (targets[t + 1] - next_values[t])
    
    return targets


def retrace_multistep_targets(
    trajectories: list,  # [(state, action, reward, next_state, done, behavior_prob, target_prob)]
    value_function: callable,
    gamma: float = 0.99,
    lambda_param: float = 0.95,
    c_clip: float = 1.0,
    n_steps: int = 3
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    다중 스텝 Retrace 타깃 계산
    
    Args:
        trajectories: 궤적 데이터 리스트
        value_function: 상태 가치 함수 (state -> value)
        gamma: 할인 인수
        lambda_param: Retrace λ 파라미터
        c_clip: 중요도 가중치 클리핑 상한
        n_steps: 다중 스텝 수
    
    Returns:
        states: 시작 상태들
        targets: 해당하는 Retrace 타깃들
    """
    if len(trajectories) < n_steps:
        # 충분한 데이터가 없으면 단일 스텝으로 fallback
        states = torch.stack([torch.tensor(traj[0]) for traj in trajectories])
        rewards = torch.stack([torch.tensor(traj[2]) for traj in trajectories])
        next_states = torch.stack([torch.tensor(traj[3]) for traj in trajectories])
        dones = torch.stack([torch.tensor(traj[4]) for traj in trajectories])
        
        with torch.no_grad():
            values = value_function(states)
            next_values = value_function(next_states)
        
        behavior_probs = torch.stack([torch.tensor(traj[5]) for traj in trajectories])
        target_probs = torch.stack([torch.tensor(traj[6]) for traj in trajectories])
        
        targets = retrace_targets(
            rewards, values.squeeze(), next_values.squeeze(), dones,
            behavior_probs, target_probs, gamma, lambda_param, c_clip
        )
        
        return states, targets
    
    # 다중 스텝 처리
    states = []
    targets = []
    
    for i in range(len(trajectories) - n_steps + 1):
        segment = trajectories[i:i + n_steps]
        
        # 세그먼트에서 데이터 추출
        segment_states = torch.stack([torch.tensor(traj[0]) for traj in segment])
        segment_rewards = torch.stack([torch.tensor(traj[2]) for traj in segment])
        segment_next_states = torch.stack([torch.tensor(traj[3]) for traj in segment])
        segment_dones = torch.stack([torch.tensor(traj[4]) for traj in segment])
        segment_behavior_probs = torch.stack([torch.tensor(traj[5]) for traj in segment])
        segment_target_probs = torch.stack([torch.tensor(traj[6]) for traj in segment])
        
        # 상태 가치 계산
        with torch.no_grad():
            segment_values = value_function(segment_states).squeeze()
            segment_next_values = value_function(segment_next_states).squeeze()
        
        # Retrace 타깃 계산
        segment_targets = retrace_targets(
            segment_rewards, segment_values, segment_next_values,
            segment_dones, segment_behavior_probs, segment_target_probs,
            gamma, lambda_param, c_clip
        )
        
        # 시작 상태와 해당 타깃만 저장
        states.append(segment_states[0])
        targets.append(segment_targets[0])
    
    return torch.stack(states), torch.stack(targets)


def safe_importance_sampling(
    behavior_probs: torch.Tensor,
    target_probs: torch.Tensor,
    max_ratio: float = 10.0,
    min_prob: float = 1e-8
) -> torch.Tensor:
    """
    안전한 중요도 샘플링 가중치 계산
    
    Args:
        behavior_probs: 행동 정책 확률
        target_probs: 타깃 정책 확률
        max_ratio: 최대 중요도 비율
        min_prob: 최소 확률 (0으로 나누기 방지)
    
    Returns:
        safe_weights: 안전한 중요도 가중치
    """
    # 확률 하한선 적용
    safe_behavior_probs = torch.clamp(behavior_probs, min=min_prob)
    safe_target_probs = torch.clamp(target_probs, min=min_prob)
    
    # 중요도 가중치 계산 및 클리핑
    importance_weights = safe_target_probs / safe_behavior_probs
    safe_weights = torch.clamp(importance_weights, max=max_ratio)
    
    return safe_weights