# Algo Trading Backtester

A modular backtesting engine for systematic trading strategies, tested on real market data via `yfinance`.

The engine is strategy-agnostic — adding a new strategy means writing one function that returns a signal series. Everything else (simulation, metrics, plotting) stays the same.

---

## Strategies

| Strategy | Type | Key idea |
|---|---|---|
| MA Crossover | Trend-following | Long when 50d MA > 200d MA |
| Bollinger Bands | Mean reversion | Trade price deviations from a rolling mean |
| RSI | Mean reversion | Trade extreme momentum readings |
| Pairs Trading | Market-neutral | Trade the spread between two cointegrated assets |

---

## Structure

```
├── src/
│   ├── data.py             — load price data from Yahoo Finance
│   ├── backtester.py       — simulation engine (signals → equity curve)
│   ├── metrics.py          — Sharpe, drawdown, CAGR, win rate
│   └── strategies/
│       ├── ma_crossover.py
│       ├── mean_reversion.py
│       ├── rsi.py
│       └── pairs_trading.py
└── notebooks/
    ├── 01_ma_crossover.ipynb
    ├── 02_mean_reversion.ipynb
    ├── 03_rsi.ipynb
    └── 04_pairs_trading.ipynb
```

---

## Notebooks

**01 — MA Crossover:** signal generation, equity curve vs. buy-and-hold, Sharpe heatmap across window combinations.

**02 — Mean Reversion:** Bollinger Band visualization with trade markers, parameter sweep over window and band width.

**03 — RSI:** four-panel chart (price, RSI, equity, drawdown), threshold sensitivity analysis.

**04 — Pairs Trading:** cointegration test, hedge ratio estimation via OLS, z-score signals, scan across candidate pairs.

---

## Backtester design

Signals are executed with a one-day lag to avoid look-ahead bias. Transaction costs are applied as a flat fraction of trade value on every position change.

```python
from src.data import load_prices
from src.backtester import run
from src.metrics import summary
from src.strategies.ma_crossover import signals

prices = load_prices('AAPL', '2015-01-01', '2024-01-01')
sig    = signals(prices, short_window=50, long_window=200)
result = run(prices, sig, initial_capital=10_000, transaction_cost=0.001)
summary(result['equity'], result['strategy_ret'])
```

---

## Setup

```bash
pip install -r requirements.txt
```

```
numpy
pandas
scipy
matplotlib
yfinance
statsmodels
jupyter
```

---

## A note on results

Past performance on historical data is not predictive. Any strategy that looks good in a backtest may be overfit to the specific period tested — especially when parameters are tuned. The parameter sweep plots in each notebook are there to show whether results are robust across configurations or dependent on one lucky setting.
