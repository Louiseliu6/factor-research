import numpy as np
import pandas as pd

from factor_research import attribution, data, factors
from factor_research.backtest import run_backtest


def _market_model_prices(n_assets=40, n_days=800, seed=0):
    """Prices driven by one market factor with known betas."""
    rng = np.random.default_rng(seed)
    betas = rng.uniform(0.5, 1.5, n_assets)
    m = rng.normal(0.0004, 0.01, n_days)
    rets = m[:, None] * betas + rng.normal(0, 0.01, (n_days, n_assets))
    idx = pd.bdate_range("2015-01-01", periods=n_days)
    prices = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=[f"A{i:02d}" for i in range(n_assets)])
    return prices, pd.Series(betas, index=prices.columns), pd.Series(m, index=idx)


def test_rolling_beta_recovers_true_betas():
    prices, true_b, m = _market_model_prices()
    est = attribution.rolling_beta(prices, m).iloc[-1]
    assert np.corrcoef(est, true_b)[0, 1] > 0.9
    assert np.allclose(est, true_b, atol=0.25)


def test_rolling_beta_has_no_lookahead():
    prices, _, m = _market_model_prices()
    cut = prices.index[500]
    shocked, m2 = prices.copy(), m.copy()
    after = shocked.index > cut
    shocked.loc[after] *= np.random.default_rng(1).uniform(0.5, 2, shocked.loc[after].shape)
    m2.loc[after] = 0.05
    a = attribution.rolling_beta(prices, m).loc[:cut]
    b = attribution.rolling_beta(shocked, m2).loc[:cut]
    pd.testing.assert_frame_equal(a, b)


def test_hedge_removes_market_exposure():
    prices, true_b, m = _market_model_prices(n_assets=60, n_days=1500)
    scores = pd.DataFrame(np.tile(true_b.to_numpy(), (len(prices), 1)), index=prices.index, columns=prices.columns)
    res = run_backtest(prices, scores, cost_bps=0)  # long high-beta, short low-beta: strongly long the market
    betas = attribution.rolling_beta(prices, m)
    raw = attribution.market_regression(res.net, m)["beta"]
    hedged = attribution.market_regression(attribution.beta_hedged(res, betas, m).loc[prices.index[300]:], m)["beta"]
    assert raw > 0.5
    assert abs(hedged) < 0.1


def test_runs_on_synthetic_data():
    prices = data.synthetic_prices(n_assets=50, n_days=600)
    m = attribution.market_return(prices)
    res = run_backtest(prices, factors.low_volatility(prices))
    out = attribution.beta_hedged(res, attribution.rolling_beta(prices, m), m)
    assert out.index.equals(prices.index) and out.notna().all()
