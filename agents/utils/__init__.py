# agents/utils/__init__.py

from .dirichlet_entropy import (
    dirichlet_entropy,
    target_entropy_from_symmetric_alpha,
    scipy_dirichlet_entropy,
    compute_dirichlet_target_entropies,
    validate_entropy_calculation
)

__all__ = [
    'dirichlet_entropy',
    'target_entropy_from_symmetric_alpha', 
    'scipy_dirichlet_entropy',
    'compute_dirichlet_target_entropies',
    'validate_entropy_calculation'
]