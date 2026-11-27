"""Market-beta attribution: how much of a factor's PnL is just market exposure?"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .backtest import BacktestResult


def market_return(prices: pd.DataFrame) -> pd.Series:
    """Equal-weight average daily return of the universe, used as the market proxy."""
    return prices.pct_change(fill_method=None).mean(axis=1).fillna(0.0)


def rolling_beta(prices: pd.DataFrame, market: pd.Series, window: int = 252, min_periods: int = 126) -> pd.DataFrame:
    """Beta of each stock to the market over a trailing window; the value at t uses returns up to t only."""
    r = prices.pct_change(fill_method=None)
    # market return on the same days each stock has data, so both moments use the same sample
    m = pd.DataFrame(np.broadcast_to(market.to_numpy()[:, None], r.shape), index=r.index, columns=r.columns).where(r.notna())
    mean = lambda x: x.rolling(window, min_periods=min_periods).mean()
    cov = mean(r * m) - mean(r) * mean(m)
    var = mean(m * m) - mean(m) ** 2
    return cov / var


def market_regression(r: pd.Series, market: pd.Series, hac_lags: int = 5) -> dict[str, float]:
    """OLS of daily returns on the market: r = alpha + beta * m + e, with Newey-West (HAC) t-stats."""
    df = r.rename("r").to_frame().join(market.rename("m"), how="inner").dropna()
    fit = sm.OLS(df["r"], sm.add_constant(df["m"])).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    return {
        "beta": float(fit.params["m"]),
        "alpha_ann": float(fit.params["const"] * 252),
        "alpha_tstat": float(fit.tvalues["const"]),
        "r2": float(fit.rsquared),
    }


def beta_hedged(res: BacktestResult, betas: pd.DataFrame, market: pd.Series) -> pd.Series:
    """Net returns after shorting the market by the portfolio's ex-ante beta.

    Stock betas are measured on each rebalance day and held until the next one,
    and the hedge set at the close of t earns the market return from t to t+1,
    the same timing as the portfolio itself. Hedge costs are ignored (index futures are cheap).
    """
    rebal_days = res.turnover.index
    held_betas = betas.reindex(rebal_days).reindex(res.weights.index).ffill()
    port_beta = (res.weights * held_betas.fillna(0.0)).sum(axis=1)
    return res.net - port_beta.shift(1).fillna(0.0) * market


def portfolio_beta_path(res: BacktestResult, betas: pd.DataFrame) -> pd.Series:
    """Ex-ante portfolio beta on each rebalance day (for diagnostics)."""
    rebal_days = res.turnover.index
    w = res.weights.reindex(rebal_days)
    return (w * betas.reindex(rebal_days).fillna(0.0)).sum(axis=1).replace(0.0, np.nan).dropna()
