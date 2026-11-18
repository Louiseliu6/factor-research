"""Factor definitions. Each returns a (date x ticker) score; higher = more attractive.

A score at date t uses only prices up to and including t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def momentum_12_1(prices: pd.DataFrame) -> pd.DataFrame:
    """Return from t-252 to t-21: classic 12-month momentum, skipping the last month."""
    return prices.shift(21) / prices.shift(252) - 1


def short_term_reversal(prices: pd.DataFrame) -> pd.DataFrame:
    """Minus the last 21-day return: last month's losers tend to bounce."""
    return -(prices / prices.shift(21) - 1)


def low_volatility(prices: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """Minus the annualised volatility of daily returns over the window."""
    return -prices.pct_change(fill_method=None).rolling(window).std() * np.sqrt(252)


FACTORS = {
    "momentum_12_1": momentum_12_1,
    "reversal_1m": short_term_reversal,
    "low_vol": low_volatility,
}
