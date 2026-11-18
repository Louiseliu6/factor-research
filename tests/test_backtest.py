import numpy as np
import pandas as pd

from factor_research import data, factors
from factor_research.backtest import quantile_weights, run_backtest


def test_weights_are_dollar_neutral():
    prices = data.synthetic_prices(n_assets=50, n_days=400)
    w = quantile_weights(factors.momentum_12_1(prices).dropna(how="all"))
    assert np.allclose(w.sum(axis=1), 0)
    assert np.allclose(w.abs().sum(axis=1), 2)


def test_no_lookahead_changing_future_prices_does_not_change_past():
    prices = data.synthetic_prices(n_assets=50, n_days=600)
    cut = prices.index[450]
    shocked = prices.copy()
    shocked.loc[shocked.index > cut] *= np.random.default_rng(1).uniform(0.5, 2, shocked.loc[shocked.index > cut].shape)
    for fn in factors.FACTORS.values():
        a = run_backtest(prices, fn(prices)).net.loc[:cut]
        b = run_backtest(shocked, fn(shocked)).net.loc[:cut]
        pd.testing.assert_series_equal(a, b)


def test_engine_passes_signal_to_next_day_return():
    """Sanity check: a score that peeks at tomorrow must look amazing, the same score lagged must not."""
    prices = data.synthetic_prices(n_assets=50, n_days=600)
    cheat = prices.pct_change(fill_method=None).shift(-1)  # uses the future on purpose
    res = run_backtest(prices, cheat.fillna(0), rebalance="D", cost_bps=0)
    honest = run_backtest(prices, cheat.shift(1).fillna(0), rebalance="D", cost_bps=0)
    assert res.gross.mean() > 10 * abs(honest.gross.mean())
