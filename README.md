# Factor research: momentum, reversal and low volatility on US equities

> **Status: work in progress.** Replace this block with your one-sentence result once you have run it on real data, e.g.
> "12-1 momentum earned a net Sharpe of X on S&P 500 stocks 2010–2025, but most of it disappears at 20 bps costs."

![equity curves](reports/figures/equity_curves.png)

## Question
Do the classic cross-sectional factors still work in large-cap US stocks after realistic trading costs?

## Method
- Universe: S&P 500 constituents, daily adjusted prices (yfinance), 2010–2025.
- Factors: 12-1 momentum, 1-month reversal, 63-day low volatility (`src/factor_research/factors.py`).
- Portfolio: monthly rebalanced, dollar-neutral long top quintile / short bottom quintile, equal weight.
- Costs: 10 bps per unit of turnover by default (`--cost-bps`).
- Signal quality: daily rank IC against 21-day forward returns.
- Look-ahead protection: weights decided at close t earn the t → t+1 return; `tests/` checks that changing future prices never changes past PnL.

## Results
| Factor | Ann. return | Sharpe | Max DD | Mean IC | Turnover |
|---|---|---|---|---|---|
| TODO | | | | | |

## Limitations
- Survivorship bias: the universe is today's S&P 500 members.
- No borrow costs for the short leg; flat cost model.
- TODO: anything else you find.

## Run it
```bash
pip install -e ".[dev]"
pytest
python run_backtest.py --synthetic   # offline check
python run_backtest.py               # real data
```
