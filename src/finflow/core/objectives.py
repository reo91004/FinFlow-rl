import torch, torch.nn.functional as F

def quantile_huber_loss(target, pred, taus, kappa=1.0):
    # target/pred: (B,Q), taus: (Q,)
    u = target - pred
    huber = torch.where(u.abs() <= kappa, 0.5*u.pow(2), kappa*(u.abs()-0.5*kappa))
    weight = torch.abs(taus - (u.detach() < 0).float())
    return (weight * huber / kappa).mean()

def cvar_from_quantiles(quantiles, alpha: float):
    # quantiles: (B,Q) sorted by τ implicitly by network head index ordering
    Q = quantiles.shape[1]
    k = max(1, int(alpha * Q))
    lower = quantiles[:, :k]
    return lower.mean(dim=1, keepdim=True)

def sharpe_surrogate(quantiles, eps: float = 1e-6):
    mu = quantiles.mean(dim=1, keepdim=True)
    sd = quantiles.std(dim=1, keepdim=True)
    return mu / (sd + eps)

def turnover_l1(a, a_prev):
    return (a - a_prev).abs().sum(dim=1, keepdim=True)
