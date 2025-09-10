import pandas as pd, numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD

def compute_features(prices: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    rets = prices.pct_change().fillna(0.0)
    vol = rets.rolling(lookback).std().fillna(0.0)
    feats = []
    for c in prices.columns:
        rsi = RSIIndicator(prices[c], window=14).rsi().fillna(0.0)
        macd = MACD(prices[c]).macd().fillna(0.0)
        tmp = pd.DataFrame({(f"ret_{i}", c): rets[c].shift(i).fillna(0.0) for i in range(1,6)})
        tmp[("vol", c)] = vol[c]
        tmp[("rsi", c)] = rsi
        tmp[("macd", c)] = macd
        feats.append(tmp)
    X = pd.concat(feats, axis=1).dropna()
    return X
