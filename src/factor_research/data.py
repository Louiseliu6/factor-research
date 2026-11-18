"""Download and cache daily adjusted close prices."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path(__file__).resolve().parents[2] / "data"


def load_prices(tickers: list[str], start: str, end: str, refresh: bool = False) -> pd.DataFrame:
    """Return a (date x ticker) DataFrame of adjusted close prices, cached as CSV."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / f"prices_{start}_{end}_{len(tickers)}.csv"
    if cache.exists() and not refresh:
        return pd.read_csv(cache, index_col=0, parse_dates=True)

    import yfinance as yf

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"].dropna(how="all", axis=1).sort_index()
    prices.to_csv(cache)
    return prices


def sp500_tickers() -> list[str]:
    """Current S&P 500 members from Wikipedia.

    Warning: using today's members for a historical backtest creates survivorship
    bias (firms that went bankrupt or were dropped are missing). Discuss this in
    the README, or use point-in-time membership if you can get it.
    """
    # Wikipedia returns 403 to the default Python user agent, so send a descriptive one.
    table = pd.read_html(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        storage_options={"User-Agent": "Mozilla/5.0 (factor-research backtest)"},
    )[0]
    return table["Symbol"].str.replace(".", "-", regex=False).tolist()


def synthetic_prices(n_assets: int = 100, n_days: int = 2520, seed: int = 0) -> pd.DataFrame:
    """Random-walk prices with a small built-in momentum effect, for tests and offline work."""
    rng = np.random.default_rng(seed)
    drift = rng.normal(0.0002, 0.0004, n_assets)
    rets = np.zeros((n_days, n_assets))
    for t in range(n_days):
        persist = 0.02 * rets[max(t - 60, 0):t].mean(axis=0) if t > 0 else 0
        rets[t] = drift + persist + rng.normal(0, 0.015, n_assets)
    dates = pd.bdate_range("2014-01-01", periods=n_days)
    cols = [f"A{i:03d}" for i in range(n_assets)]
    return pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=dates, columns=cols)
