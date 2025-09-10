import numpy as np, pandas as pd
from finflow.core.env import PortfolioEnv
from finflow.data.features import compute_features

def test_env_step():
    idx = pd.date_range("2020-01-01", periods=100, freq="D")
    p = pd.DataFrame({"A": np.linspace(100,110,100), "B": np.linspace(50,55,100)}, index=idx)
    X = compute_features(p, 10)
    env = PortfolioEnv(p, X, lookback=10, cost_bps=5)
    obs = env.reset()
    w = np.array([0.6,0.4], dtype=np.float32)
    o2, r, d, info = env.step(w)
    assert isinstance(r, float)
