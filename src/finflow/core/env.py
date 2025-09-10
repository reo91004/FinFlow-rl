import numpy as np

class PortfolioEnv:
    def __init__(self, prices, features, lookback=20, cost_bps=5, no_trade_band=0.0, max_leverage=1.0):
        self.prices = prices
        self.features = features.loc[prices.index.intersection(features.index)]
        self.lookback = lookback
        self.cost_bps = cost_bps
        self.no_trade_band = no_trade_band
        self.max_leverage = max_leverage
        self.assets = list(prices.columns)
        self.n_assets = len(self.assets)
        self.rets = self.prices.pct_change().fillna(0.0).values
        self.t0 = lookback
        self.obs_dim = self.features.shape[1]
        self.act_dim = self.n_assets
        self.reset()

    def reset(self):
        self.t = self.t0
        self.done = False
        self.w = np.ones(self.n_assets, dtype=np.float32)/self.n_assets
        self.equity = 1.0
        return self._obs()

    def step(self, w_new):
        w_new = np.clip(w_new, 1e-8, None); w_new = w_new / w_new.sum()
        delta = w_new - self.w
        mask = np.abs(delta) < self.no_trade_band
        w_exec = self.w.copy(); w_exec[~mask] = w_new[~mask]
        turn = float(np.sum(np.abs(w_exec - self.w)))
        cost = (self.cost_bps * 1e-4) * turn
        r = float(np.dot(w_exec, self.rets[self.t]))
        r_net = r - cost
        self.equity *= (1.0 + r_net)
        prev = self.w.copy()
        self.w = w_exec
        self.t += 1
        if self.t >= len(self.rets)-1: self.done = True
        return self._obs(), r_net, self.done, {"turnover": turn, "cost": cost, "equity": self.equity, "prev_w": prev, "exec_w": w_exec}

    def _obs(self):
        t_idx = self.prices.index[self.t]
        return self.features.loc[t_idx].values.astype(np.float32)
