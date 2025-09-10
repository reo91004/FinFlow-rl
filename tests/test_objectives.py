import torch
from finflow.core.objectives import quantile_huber_loss, cvar_from_quantiles, sharpe_surrogate

def test_objectives():
    B,Q = 32, 25
    pred = torch.randn(B,Q)
    tgt = torch.randn(B,Q)
    taus = (torch.arange(Q)+0.5)/Q
    loss = quantile_huber_loss(tgt, pred, taus)
    cvar = cvar_from_quantiles(pred, 0.05)
    s = sharpe_surrogate(pred)
    assert torch.isfinite(loss)
    assert torch.isfinite(cvar).all()
    assert torch.isfinite(s).all()
