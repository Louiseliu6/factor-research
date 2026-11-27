"""How much of each factor's PnL is market beta? Regress on the market, then hedge it out.

    python run_attribution.py              # cached S&P 500 data (run run_backtest.py first)
    python run_attribution.py --synthetic  # offline synthetic data
"""
import argparse

import matplotlib.pyplot as plt
import pandas as pd

pd.set_option("display.width", 200)

from factor_research import attribution, data, factors, metrics
from factor_research.backtest import run_backtest

parser = argparse.ArgumentParser()
parser.add_argument("--synthetic", action="store_true")
parser.add_argument("--start", default="2010-01-01")
parser.add_argument("--end", default="2025-12-31")
parser.add_argument("--cost-bps", type=float, default=10.0)
args = parser.parse_args()

prices = data.synthetic_prices() if args.synthetic else data.load_prices(data.sp500_tickers(), args.start, args.end)
market = attribution.market_return(prices)
betas = attribution.rolling_beta(prices, market)

rows, curves = {}, {}
for name, fn in factors.FACTORS.items():
    res = run_backtest(prices, fn(prices), cost_bps=args.cost_bps)
    hedged = attribution.beta_hedged(res, betas, market)
    reg = attribution.market_regression(res.net, market)
    rows[name] = {
        "raw_ann_mean": res.net.mean() * 252,
        "beta": reg["beta"],
        "alpha_ann": reg["alpha_ann"],
        "alpha_tstat": reg["alpha_tstat"],
        "r2": reg["r2"],
        "avg_exante_beta": attribution.portfolio_beta_path(res, betas).mean(),
        "hedged_ann_return": metrics.annual_return(hedged),
        "hedged_sharpe": metrics.sharpe(hedged),
        "hedged_max_dd": metrics.max_drawdown(hedged),
    }
    curves[f"{name} (raw)"] = (1 + res.net).cumprod()
    curves[f"{name} (beta-hedged)"] = (1 + hedged).cumprod()

print(f"Market proxy: equal-weight universe return, {metrics.annual_return(market):.1%} a year")
print(pd.DataFrame(rows).T.round(3).to_string())

suffix = "_synthetic" if args.synthetic else ""
fig, ax = plt.subplots(figsize=(9, 5))
for color, name in zip(["tab:blue", "tab:green", "tab:purple"], factors.FACTORS):
    curves[f"{name} (raw)"].plot(ax=ax, color=color, logy=True, label=f"{name} (raw)")
    curves[f"{name} (beta-hedged)"].plot(ax=ax, color=color, linestyle="--", logy=True, label=f"{name} (beta-hedged)")
ax.set_title(f"Raw (solid) vs beta-hedged (dashed) long-short portfolios, net of {args.cost_bps:.0f} bps")
ax.set_ylabel("Growth of 1 (log scale)")
ax.legend()
plt.tight_layout()
plt.savefig(f"reports/figures/beta_hedged{suffix}.png", dpi=150)
print(f"Saved reports/figures/beta_hedged{suffix}.png")
