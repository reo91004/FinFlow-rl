# agents/utils/dirichlet_entropy.py

import torch
import numpy as np
from torch import Tensor
from scipy.special import digamma, loggamma


def dirichlet_entropy(alpha: Tensor) -> Tensor:
    """
    Dirichlet(α) 분포의 미분 엔트로피 계산
    
    H(X) = log B(α) + (α₀-K)ψ(α₀) - Σᵢ(αᵢ-1)ψ(αᵢ)
    
    Args:
        alpha: (..., K) shape tensor with α_i > 0
        
    Returns:
        entropy: (...) shape tensor containing differential entropy values
    """
    # α₀ = Σα_i 계산
    alpha0 = alpha.sum(dim=-1)
    
    # log B(α) = Σlog Γ(α_i) - log Γ(α₀) 계산
    logB = torch.lgamma(alpha).sum(dim=-1) - torch.lgamma(alpha0)
    
    # 엔트로피 계산
    H = (logB + 
         (alpha0 - alpha.size(-1)) * torch.digamma(alpha0) -
         ((alpha - 1.0) * torch.digamma(alpha)).sum(dim=-1))
    
    return H


def target_entropy_from_symmetric_alpha(K: int, alpha_star: float) -> float:
    """
    대칭 Dirichlet(α*) 분포로부터 타깃 엔트로피 계산
    
    Args:
        K: 차원 수 (자산 개수)
        alpha_star: 대칭 농도 파라미터 (α_i = α* for all i)
        
    Returns:
        target_entropy: 계산된 타깃 엔트로피 값
    """
    # 모든 α_i = α*인 대칭 벡터 생성
    alpha_symmetric = torch.full((K,), float(alpha_star))
    
    # 엔트로피 계산
    entropy = dirichlet_entropy(alpha_symmetric)
    
    return float(entropy.item())


def scipy_dirichlet_entropy(K: int, alpha_star: float) -> float:
    """
    SciPy를 사용한 대칭 Dirichlet 엔트로피 계산 (참조용)
    
    이 함수는 PyTorch 버전의 정확성을 검증하기 위한 참조 구현입니다.
    """
    # α₀ = K * α*
    alpha0 = K * alpha_star
    
    # log B(α) 계산
    logB = K * loggamma(alpha_star) - loggamma(alpha0)
    
    # 엔트로피 계산
    H = (logB + 
         (alpha0 - K) * digamma(alpha0) -
         K * (alpha_star - 1.0) * digamma(alpha_star))
    
    return float(H)


def compute_dirichlet_target_entropies():
    """
    다양한 α* 값에 대한 타깃 엔트로피 계산 및 출력
    
    포트폴리오 다양성에 따른 적절한 타깃 엔트로피 값을 찾기 위한 참조표 생성
    """
    K = 30  # Dow Jones 30 stocks
    
    print(f"=== Dirichlet Target Entropy Reference (K={K}) ===")
    print("α*\tTarget Entropy\tDiversity Level")
    print("-" * 50)
    
    alpha_values = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
    
    for alpha_star in alpha_values:
        target_entropy = target_entropy_from_symmetric_alpha(K, alpha_star)
        
        # 다양성 수준 해석
        if alpha_star < 1.0:
            diversity = "Very High (Corner Preference)"
        elif alpha_star == 1.0:
            diversity = "High (Uniform)"  
        elif alpha_star < 2.0:
            diversity = "Moderate"
        elif alpha_star < 5.0:
            diversity = "Low (Central Concentration)"
        else:
            diversity = "Very Low (Strong Concentration)"
            
        print(f"{alpha_star:.1f}\t{target_entropy:.2f}\t\t{diversity}")
    
    return alpha_values


def validate_entropy_calculation():
    """
    PyTorch와 SciPy 구현 간 일치성 검증
    """
    K = 30
    test_alphas = [0.5, 1.0, 1.5, 2.0]
    
    print("\n=== Entropy Calculation Validation ===")
    print("α*\tPyTorch\t\tSciPy\t\tDifference")
    print("-" * 50)
    
    for alpha_star in test_alphas:
        torch_entropy = target_entropy_from_symmetric_alpha(K, alpha_star)
        scipy_entropy = scipy_dirichlet_entropy(K, alpha_star)
        diff = abs(torch_entropy - scipy_entropy)
        
        print(f"{alpha_star:.1f}\t{torch_entropy:.4f}\t\t{scipy_entropy:.4f}\t\t{diff:.6f}")
        
        if diff > 1e-4:
            print(f"WARNING: Large difference detected for α*={alpha_star}")


if __name__ == "__main__":
    # 참조표 생성 및 검증
    compute_dirichlet_target_entropies()
    validate_entropy_calculation()
    
    # 권장 설정 출력
    print("\n=== Recommended Settings ===")
    print("For balanced portfolio diversity: α* = 1.5, Target Entropy ≈ -72.56")
    print("For high diversity: α* = 1.0, Target Entropy ≈ -71.26") 
    print("For conservative concentration: α* = 2.0, Target Entropy ≈ -74.64")