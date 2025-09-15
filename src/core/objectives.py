# src/core/objectives.py

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class ObjectiveMetrics:
    """목적함수 계산 결과"""
    total_objective: float
    sharpe_ratio: float
    cvar_loss: float
    turnover_penalty: float
    drawdown_penalty: float
    components: Dict[str, float]

class DifferentialSharpe(nn.Module):
    """
    미분 가능한 샤프 비율 구현
    
    EMA 기반 온라인 추정으로 그래디언트 계산 가능
    """
    
    def __init__(self, alpha: float = 0.99, epsilon: float = 1e-8):
        super().__init__()
        self.alpha = alpha
        self.epsilon = epsilon
        
        # EMA 추적 변수
        self.register_buffer('mean_ema', torch.tensor(0.0))
        self.register_buffer('var_ema', torch.tensor(0.0))
        self.register_buffer('count', torch.tensor(0))
    
    def forward(self, returns: torch.Tensor) -> torch.Tensor:
        """
        배치 수익률에 대한 미분가능 샤프 비율 계산
        
        Args:
            returns: 수익률 텐서 [batch_size] or [batch_size, time]
            
        Returns:
            sharpe: 미분가능 샤프 비율
        """
        if returns.dim() == 2:
            # Flatten time dimension
            returns = returns.view(-1)
        
        batch_mean = returns.mean()
        batch_var = returns.var(unbiased=False)
        
        # EMA 업데이트
        if self.training:
            if self.count == 0:
                self.mean_ema = batch_mean
                self.var_ema = batch_var
            else:
                self.mean_ema = self.alpha * self.mean_ema + (1 - self.alpha) * batch_mean
                self.var_ema = self.alpha * self.var_ema + (1 - self.alpha) * batch_var
            self.count += 1
        
        # 샤프 비율 계산 (연율화)
        std = torch.sqrt(self.var_ema + self.epsilon)
        sharpe = self.mean_ema / std * np.sqrt(252)
        
        return sharpe
    
    def reset(self):
        """EMA 상태 초기화"""
        self.mean_ema.zero_()
        self.var_ema.zero_()
        self.count.zero_()

class CVaRConstraint(nn.Module):
    """
    Conditional Value at Risk (CVaR) 제약
    
    하위 α% 수익률의 평균을 계산하여 꼬리 위험 측정
    """
    
    def __init__(self, alpha: float = 0.05, target: float = -0.02):
        super().__init__()
        self.alpha = alpha
        self.target = target  # CVaR 목표 (예: -2%)
    
    def forward(self, returns: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        CVaR 계산 및 제약 위반 손실

        Args:
            returns: 수익률 텐서 [batch_size]

        Returns:
            cvar: CVaR 값
            violation: 제약 위반 손실 (0 if satisfied)
        """
        # 하위 α% 분위수 계산
        k = max(1, int(len(returns) * self.alpha))
        bottom_k = torch.topk(-returns, k, largest=True)[0]  # 가장 작은 k개

        # CVaR = 하위 α% 평균
        cvar = -bottom_k.mean()

        # 강한 CVaR 페널티 (제약 위반 시)
        violation = torch.clamp(self.target - cvar, min=0.0) * 10.0  # 강한 페널티

        return cvar, violation

class PortfolioObjective(nn.Module):
    """
    통합 포트폴리오 목적함수
    
    Sharpe 최대화 + CVaR 제약 + 턴오버/낙폭 페널티
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # 컴포넌트 초기화
        self.diff_sharpe = DifferentialSharpe(
            alpha=config.get('sharpe_ema_alpha', 0.99),
            epsilon=config.get('sharpe_epsilon', 1e-8)
        )
        
        self.cvar_constraint = CVaRConstraint(
            alpha=config.get('cvar_alpha', 0.05),
            target=config.get('cvar_target', -0.02)
        )
        
        # 가중치
        self.sharpe_beta = config.get('sharpe_beta', 1.0)
        self.lambda_cvar = config.get('lambda_cvar', 1.0)
        self.lambda_turn = config.get('lambda_turn', 0.1)
        self.lambda_dd = config.get('lambda_dd', 0.0)
        
        # 보상 클리핑
        self.r_clip = config.get('r_clip', 5.0)
        
        # Lagrange 승수 (CVaR 제약용)
        self.register_buffer('lagrange_lambda', torch.tensor(1.0))
        self.lagrange_lr = 0.01
    
    def forward(self, 
                returns: torch.Tensor,
                weights: torch.Tensor,
                prev_weights: torch.Tensor) -> Tuple[torch.Tensor, ObjectiveMetrics]:
        """
        목적함수 계산
        
        Args:
            returns: 포트폴리오 수익률 [batch_size]
            weights: 현재 가중치 [batch_size, n_assets]
            prev_weights: 이전 가중치 [batch_size, n_assets]
            
        Returns:
            objective: 최대화할 목적함수 값
            metrics: 상세 메트릭
        """
        # 수익률 클리핑
        returns = torch.clamp(returns, -self.r_clip, self.r_clip)
        
        # 1. Sharpe ratio
        sharpe = self.diff_sharpe(returns)
        
        # 2. CVaR constraint
        cvar, cvar_violation = self.cvar_constraint(returns)
        
        # 3. Turnover penalty
        turnover = torch.abs(weights - prev_weights).sum(dim=-1).mean()
        turnover_penalty = self.lambda_turn * turnover
        
        # 4. Drawdown penalty (optional)
        if self.lambda_dd > 0:
            cum_returns = torch.cumprod(1 + returns, dim=0)
            running_max = torch.cummax(cum_returns, dim=0)[0]
            drawdown = (cum_returns - running_max) / (running_max + 1e-8)
            max_drawdown = -drawdown.min()
            drawdown_penalty = self.lambda_dd * max_drawdown
        else:
            drawdown_penalty = torch.tensor(0.0)
        
        # 5. 통합 목적함수 (CVaR 페널티 강화)
        objective = (
            returns.mean() +  # 기본 수익률
            self.sharpe_beta * sharpe -  # 샤프 보너스
            self.lambda_cvar * cvar_violation -  # CVaR 페널티 (강화)
            turnover_penalty -  # 턴오버 페널티
            drawdown_penalty  # 낙폭 페널티
        )
        
        # Lagrange 승수 업데이트 (CVaR 제약 만족을 위해)
        if self.training:
            with torch.no_grad():
                self.lagrange_lambda += self.lagrange_lr * cvar_violation
                self.lagrange_lambda = torch.clamp(self.lagrange_lambda, 0.0, 10.0)
        
        # 메트릭 생성
        metrics = ObjectiveMetrics(
            total_objective=objective.item(),
            sharpe_ratio=sharpe.item(),
            cvar_loss=cvar.item(),
            turnover_penalty=turnover_penalty.item(),
            drawdown_penalty=drawdown_penalty.item(),
            components={
                'mean_return': returns.mean().item(),
                'cvar_violation': cvar_violation.item(),
                'lagrange_lambda': self.lagrange_lambda.item(),
                'turnover': turnover.item()
            }
        )
        
        return objective, metrics

class RewardNormalizer:
    """
    보상 정규화 (EMA 기반)
    
    안정적인 학습을 위한 보상 스케일링
    """
    
    def __init__(self, alpha: float = 0.99):
        self.alpha = alpha
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
    
    def normalize(self, reward: float) -> float:
        """보상 정규화"""
        if self.count == 0:
            self.mean = reward
            self.var = 0.0
            self.count = 1
            return reward
        
        # EMA 업데이트
        self.mean = self.alpha * self.mean + (1 - self.alpha) * reward
        delta = reward - self.mean
        self.var = self.alpha * self.var + (1 - self.alpha) * delta**2
        self.count += 1
        
        # 정규화
        std = np.sqrt(self.var + 1e-8)
        normalized = (reward - self.mean) / std
        
        # 클리핑
        return np.clip(normalized, -3, 3)
    
    def reset(self):
        """상태 초기화"""
        self.mean = 0.0
        self.var = 1.0
        self.count = 0


def portfolio_objective(returns_t: torch.Tensor, weights_t: torch.Tensor, cfg: Dict) -> torch.Tensor:
    """
    포트폴리오 목적함수 편의 함수 (CVaR 통합)

    Args:
        returns_t: 수익률 텐서 [batch_size] 또는 [batch_size, n_assets]
        weights_t: 포트폴리오 가중치 [batch_size, n_assets]
        cfg: 설정 딕셔너리

    Returns:
        objective: 통합 목적함수 값
    """
    # 포트폴리오 수익률 계산
    if returns_t.dim() == 2:
        port_ret = (returns_t * weights_t).sum(dim=-1)
    else:
        port_ret = returns_t

    # 기본 샤프 보상
    sharpe_reward = port_ret.mean() / (port_ret.std() + 1e-8) * np.sqrt(252)

    # CVaR 페널티 계산 (alpha=0.95, target=-0.02)
    alpha = 0.95  # cvar_alpha=0.05와 동치
    threshold = -0.02  # -2% 목표
    var = torch.quantile(port_ret, q=1.0 - alpha)
    cvar = port_ret[port_ret <= var].mean() if (port_ret <= var).sum() > 0 else var
    cvar_penalty = torch.clamp(threshold - cvar, min=0.0) * 10.0  # 강한 페널티

    # 턴오버 페널티 (선택적)
    turnover_penalty = 0.0
    if 'prev_weights' in cfg:
        turnover = torch.abs(weights_t - cfg['prev_weights']).sum(dim=-1).mean()
        turnover_penalty = turnover * cfg.get('lambda_turn', 0.1)

    # 최종 목적함수
    lambda_cvar = cfg.get('lambda_cvar', 1.0)
    total = sharpe_reward - lambda_cvar * cvar_penalty - turnover_penalty

    return total