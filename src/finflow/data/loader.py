import pandas as pd, numpy as np, yfinance as yf

def load_prices(symbols, start, end, interval="1d"):
    frames = []
    for s in symbols:
        df = yf.download(s, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
        if df.empty: continue
        frames.append(df["Close"].rename(s))
    if not frames:
        raise RuntimeError("No data downloaded. Check network/symbols/dates.")
    prices = pd.concat(frames, axis=1).dropna(how="any")
    return prices

def to_returns(prices: pd.DataFrame):
    return prices.pct_change().dropna(how="any")
