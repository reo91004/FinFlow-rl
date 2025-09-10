import numpy as np, pandas as pd

def rolling_sharpe(returns: np.ndarray, window: int = 252, eps: float = 1e-6):
    r = pd.Series(returns)
    mu = r.rolling(window).mean()
    sd = r.rolling(window).std()
    s = (mu / (sd + eps)).fillna(0.0)
    return s.values

def drawdown_curve(equity: np.ndarray):
    cummax = np.maximum.accumulate(equity)
    return (equity - cummax) / (cummax + 1e-12)

def cvar(returns: np.ndarray, alpha: float = 0.05):
    if len(returns) == 0: return 0.0
    q = np.quantile(returns, alpha)
    tail = returns[returns <= q]
    if len(tail) == 0: return q
    return tail.mean()
