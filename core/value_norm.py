# core/value_norm.py

import torch
import numpy as np
from typing import Optional


class PopArt:
    """
    PopArt (Pop-art) 적응형 정규화
    
    논문: "Pop-art: accelerating online reinforcement learning with adaptive preprocessing"
    목적: 비정상성 보상/값 분포에서 신경망 타깃을 안정화
    """
    
    def __init__(self, eps: float = 1e-5, momentum: float = 0.99, device: Optional[torch.device] = None):
        """
        Args:
            eps: 수치적 안정성을 위한 작은 값
            momentum: EMA 업데이트 모멘텀 (높을수록 느린 적응)
            device: 텐서 디바이스
        """
        self.eps = eps
        self.momentum = momentum
        self.device = device or torch.device('cpu')
        
        # 통계 추적
        self.mu = 0.0  # 평균
        self.sigma2 = 1.0  # 분산
        self.count = 0
        
        # 정규화기 활성화 플래그
        self.enabled = True
    
    def update(self, values: torch.Tensor) -> None:
        """
        새로운 값들로 평균/분산 통계 업데이트
        
        Args:
            values: 새로운 값 텐서 [batch_size,] 또는 스칼라
        """
        if not self.enabled:
            return
            
        # 텐서를 CPU로 이동하고 numpy로 변환
        if isinstance(values, torch.Tensor):
            values_np = values.detach().cpu().numpy()
        else:
            values_np = np.array(values)
        
        # 스칼라/1D 텐서 처리
        if values_np.ndim == 0:
            values_np = np.array([values_np])
        elif values_np.ndim > 1:
            values_np = values_np.flatten()
        
        # 현재 배치 통계
        batch_mean = values_np.mean()
        batch_var = values_np.var()
        
        if self.count == 0:
            # 초기화
            self.mu = batch_mean
            self.sigma2 = max(batch_var, self.eps)
        else:
            # EMA 업데이트
            self.mu = self.momentum * self.mu + (1 - self.momentum) * batch_mean
            self.sigma2 = self.momentum * self.sigma2 + (1 - self.momentum) * batch_var
        
        self.count += 1
        
        # 분산 하한선 보장
        self.sigma2 = max(self.sigma2, self.eps)
    
    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        """
        값들을 현재 통계로 정규화
        
        Args:
            values: 정규화할 값 텐서
            
        Returns:
            normalized_values: 정규화된 값 텐서
        """
        if not self.enabled:
            return values
        
        sigma = np.sqrt(self.sigma2 + self.eps)
        normalized = (values - self.mu) / sigma
        
        return normalized
    
    def denormalize(self, normalized_values: torch.Tensor) -> torch.Tensor:
        """
        정규화된 값들을 원래 스케일로 복원
        
        Args:
            normalized_values: 정규화된 값 텐서
            
        Returns:
            denormalized_values: 복원된 값 텐서
        """
        if not self.enabled:
            return normalized_values
        
        sigma = np.sqrt(self.sigma2 + self.eps)
        denormalized = normalized_values * sigma + self.mu
        
        return denormalized
    
    def get_stats(self) -> dict:
        """현재 통계 반환"""
        return {
            'mu': self.mu,
            'sigma2': self.sigma2,
            'sigma': np.sqrt(self.sigma2 + self.eps),
            'count': self.count,
            'enabled': self.enabled
        }
    
    def reset(self) -> None:
        """통계 초기화"""
        self.mu = 0.0
        self.sigma2 = 1.0
        self.count = 0
    
    def enable(self) -> None:
        """정규화 활성화"""
        self.enabled = True
    
    def disable(self) -> None:
        """정규화 비활성화 (bypass 모드)"""
        self.enabled = False


class PopArtLinear(torch.nn.Module):
    """
    PopArt 통합 선형 레이어
    
    출력 레이어에 PopArt를 통합하여 가중치 자동 보정 수행
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: Optional[torch.device] = None):
        """
        Args:
            in_features: 입력 특성 수
            out_features: 출력 특성 수  
            bias: 편향 사용 여부
            device: 텐서 디바이스
        """
        super().__init__()
        
        self.device = device or torch.device('cpu')
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias).to(self.device)
        self.popart = PopArt(device=self.device)
        
        # 이전 통계 저장 (가중치 보정용)
        self.prev_mu = 0.0
        self.prev_sigma = 1.0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파 (정규화된 출력 반환)
        
        Args:
            x: 입력 텐서
            
        Returns:
            output: 정규화된 출력
        """
        # 선형 레이어 통과
        raw_output = self.linear(x)
        
        # PopArt 정규화
        normalized_output = self.popart.normalize(raw_output)
        
        return normalized_output
    
    def update_and_correct(self, targets: torch.Tensor) -> None:
        """
        타깃으로 PopArt 업데이트 및 가중치 보정
        
        Args:
            targets: 업데이트할 타깃 값들
        """
        # 이전 통계 저장
        stats = self.popart.get_stats()
        self.prev_mu = stats['mu']
        self.prev_sigma = stats['sigma']
        
        # PopArt 업데이트
        self.popart.update(targets)
        
        # 가중치 보정 (출력 불변성 유지)
        self._correct_weights()
    
    def _correct_weights(self) -> None:
        """
        PopArt 업데이트 후 가중치 보정 수행
        """
        if not self.popart.enabled:
            return
        
        new_stats = self.popart.get_stats()
        new_mu = new_stats['mu']
        new_sigma = new_stats['sigma']
        
        # 스케일 비율 계산
        scale_ratio = self.prev_sigma / new_sigma if new_sigma > self.popart.eps else 1.0
        
        with torch.no_grad():
            # 가중치 보정
            self.linear.weight.data *= scale_ratio
            
            # 편향 보정 (있는 경우)
            if self.linear.bias is not None:
                bias_correction = (self.prev_mu - new_mu * scale_ratio)
                self.linear.bias.data = self.linear.bias.data * scale_ratio + bias_correction
    
    def get_denormalized_output(self, normalized_output: torch.Tensor) -> torch.Tensor:
        """
        정규화된 출력을 원래 스케일로 복원
        
        Args:
            normalized_output: 정규화된 출력
            
        Returns:
            denormalized_output: 복원된 출력
        """
        return self.popart.denormalize(normalized_output)
    
    def get_stats(self) -> dict:
        """PopArt 통계 반환"""
        return self.popart.get_stats()
    
    def enable_popart(self) -> None:
        """PopArt 활성화"""
        self.popart.enable()
    
    def disable_popart(self) -> None:
        """PopArt 비활성화"""
        self.popart.disable()