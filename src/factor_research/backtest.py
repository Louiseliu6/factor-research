"""Vectorised long-short quantile backtest with transaction costs."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class BacktestResult:
    weights: pd.DataFrame      # weights held during each day
    gross: pd.Series           # daily return before costs
    costs: pd.Series           # daily cost
    net: pd.Series             # daily return after costs
    turnover: pd.Series        # sum |w_t - w_{t-1}| on rebalance days


def quantile_weights(scores: pd.DataFrame, n_quantiles: int = 5) -> pd.DataFrame:
    """Equal-weight long the top quantile, short the bottom; each leg sums to 1 in absolute value."""
    ranks = scores.rank(axis=1, pct=True)
    long = (ranks > 1 - 1 / n_quantiles).astype(float)
    short = (ranks <= 1 / n_quantiles).astype(float)
    long = long.div(long.sum(axis=1).replace(0, float("nan")), axis=0)
    short = short.div(short.sum(axis=1).replace(0, float("nan")), axis=0)
    return (long - short).fillna(0.0)


def run_backtest(
    prices: pd.DataFrame,
    scores: pd.DataFrame,
    n_quantiles: int = 5,
    rebalance: str = "ME",
    cost_bps: float = 10.0,
) -> BacktestResult:
    """Rebalance at each period end using that day's scores; trade at that close.

    The weights decided on day t earn the return from t to t+1, so the
    `.shift(1)` below is what prevents look-ahead bias.
    """
    returns = prices.pct_change(fill_method=None).fillna(0.0)
    rebal_days = scores.resample(rebalance).last().index
    pos = scores.index.searchsorted(rebal_days, side="right") - 1
    rebal_days = scores.index[pos[pos >= 0]].unique()  # last trading day of each period
    target = quantile_weights(scores.loc[rebal_days], n_quantiles)

    held = target.reindex(prices.index).ffill().fillna(0.0)
    gross = (held.shift(1) * returns).sum(axis=1)

    turnover = target.diff().abs().sum(axis=1)
    turnover.iloc[0] = target.iloc[0].abs().sum()
    costs = (turnover * cost_bps / 1e4).reindex(prices.index).fillna(0.0).shift(1).fillna(0.0)
    return BacktestResult(held, gross, costs, gross - costs, turnover)
