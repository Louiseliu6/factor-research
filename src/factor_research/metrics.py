"""Performance and signal-quality metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def annual_return(r: pd.Series) -> float:
    return float((1 + r).prod() ** (252 / len(r)) - 1)


def sharpe(r: pd.Series) -> float:
    return float(np.sqrt(252) * r.mean() / r.std()) if r.std() > 0 else float("nan")


def max_drawdown(r: pd.Series) -> float:
    equity = (1 + r).cumprod()
    return float((equity / equity.cummax() - 1).min())


def information_coefficient(scores: pd.DataFrame, prices: pd.DataFrame, horizon: int = 21) -> pd.Series:
    """Daily Spearman correlation between today's score and the next `horizon`-day return."""
    fwd = prices.shift(-horizon) / prices - 1
    s, f = scores.rank(axis=1), fwd.rank(axis=1)
    return s.corrwith(f, axis=1).dropna()


def summary(r: pd.Series) -> dict[str, float]:
    return {
        "ann_return": annual_return(r),
        "ann_vol": float(r.std() * np.sqrt(252)),
        "sharpe": sharpe(r),
        "max_drawdown": max_drawdown(r),
    }
