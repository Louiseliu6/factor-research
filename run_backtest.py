"""Run every factor and print a summary table.

    python run_backtest.py              # S&P 500 via yfinance (needs internet)
    python run_backtest.py --synthetic  # offline synthetic data
"""
import argparse

import matplotlib.pyplot as plt
import pandas as pd

pd.set_option("display.width", 200)

from factor_research import data, factors, metrics
from factor_research.backtest import run_backtest

parser = argparse.ArgumentParser()
parser.add_argument("--synthetic", action="store_true")
parser.add_argument("--start", default="2010-01-01")
parser.add_argument("--end", default="2025-12-31")
parser.add_argument("--cost-bps", type=float, default=10.0)
args = parser.parse_args()

prices = data.synthetic_prices() if args.synthetic else data.load_prices(data.sp500_tickers(), args.start, args.end)

rows, curves = {}, {}
for name, fn in factors.FACTORS.items():
    scores = fn(prices)
    res = run_backtest(prices, scores, cost_bps=args.cost_bps)
    ic = metrics.information_coefficient(scores, prices)
    # IC t-stat divided by sqrt(21) because daily 21-day forward returns overlap (rough Newey-West stand-in)
    rows[name] = {**metrics.summary(res.net), "mean_IC": ic.mean(), "IC_tstat": ic.mean() / ic.std() * len(ic) ** 0.5 / 21 ** 0.5,
                  "turnover_per_rebal": res.turnover.mean()}
    curves[name] = (1 + res.net).cumprod()

print(pd.DataFrame(rows).T.round(3).to_string())
pd.DataFrame(curves).plot(figsize=(9, 5), title=f"Long-short quintile portfolios, net of {args.cost_bps:.0f} bps")
plt.ylabel("Growth of 1")
plt.tight_layout()
suffix = "_synthetic" if args.synthetic else ""
plt.savefig(f"reports/figures/equity_curves{suffix}.png", dpi=150)
print(f"Saved reports/figures/equity_curves{suffix}.png")
