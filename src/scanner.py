import numpy as np
import pandas as pd
import yfinance as yf
import requests
import threading
import io

from backtester import run
from metrics import sharpe_ratio, max_drawdown, annualized_return
from strategies.ma_crossover import signals as ma_signals
from strategies.mean_reversion import signals as mr_signals
from strategies.rsi import signals as rsi_signals

# parameter grids per strategy
PARAM_GRIDS = {
    "MA Crossover": [
        {"short_window": sw, "long_window": lw}
        for sw in [10, 20, 30, 50]
        for lw in [50, 100, 150, 200]
        if sw < lw
    ],
    "Mean Reversion": [
        {"window": w, "n_std": s}
        for w in [10, 20, 30, 40]
        for s in [1.0, 1.5, 2.0, 2.5]
    ],
    "RSI": [
        {"window": w, "oversold": os, "overbought": ob}
        for w in [7, 14, 21]
        for os, ob in [(25, 75), (30, 70), (35, 65)]
    ],
}


def _generate_signals(strategy, prices, params):
    """Route to the right signal function. Returns signal Series."""
    if strategy == "MA Crossover":
        return ma_signals(prices, **params)
    elif strategy == "Mean Reversion":
        return mr_signals(prices, **params)
    elif strategy == "RSI":
        sig, _ = rsi_signals(prices, **params)
        return sig


def calmar_ratio(equity_curve, periods_per_year=252):
    """Annualized return divided by absolute max drawdown."""
    cagr = annualized_return(equity_curve, periods_per_year)
    mdd = abs(max_drawdown(equity_curve))
    return cagr / mdd if mdd > 0 else np.nan


def leveraged_metrics(result, leverage=2.0, cost_per_day=0.0):
    """
    Apply leverage to a backtest result.
    Daily return is multiplied by leverage factor.
    cost_per_day accounts for financing cost (e.g. 0.0001 = ~2.5% annualized for 2x).

    Returns adjusted equity curve and returns.
    """
    lev_ret = result["strategy_ret"] * leverage - cost_per_day
    # clip at -1 to avoid negative equity (margin call floor)
    lev_ret = lev_ret.clip(lower=-1)
    lev_equity = 10_000 * (1 + lev_ret).cumprod()
    return lev_equity, lev_ret


def fetch_sp500(progress_cb=None):
    """
    Fetch S&P 500 tickers from Wikipedia and download all price data in one call.
    """

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    tables = pd.read_html(io.StringIO(response.text))
    sp500 = next(t for t in tables if "Symbol" in t.columns)
    tickers = sp500["Symbol"].str.replace(".", "-", regex=False).tolist()

    if progress_cb:
        progress_cb("Downloading price data for S&P 500...")

    prices_df = yf.download(
        tickers,
        start="2015-01-01",
        end="2024-01-01",
        auto_adjust=True,
        progress=False,
    )["Close"]

    prices_df = prices_df.dropna(thresh=int(len(prices_df) * 0.95), axis=1)

    return prices_df


def run_scan(
    strategy, ranking, prices_df, progress_cb=None, leverage=2.0, financing_cost=0.0001
):
    """
    Run a full parameter sweep across all tickers in prices_df.

    Params:
        strategy      : one of PARAM_GRIDS keys
        ranking       : 'Total Return' | 'Sharpe Ratio' | 'Calmar Ratio' | 'Leveraged'
        prices_df     : DataFrame of closing prices
        progress_cb   : callback(value, max_value) for progress updates
        leverage      : leverage factor for 'Leveraged' ranking
        financing_cost: daily financing cost applied when leveraged

    Returns: pd.DataFrame of top results, sorted by ranking metric.
    """
    grid = PARAM_GRIDS[strategy]
    tickers = prices_df.columns.tolist()
    total = len(tickers) * len(grid)
    results = []
    count = 0

    for ticker in tickers:
        prices = prices_df[ticker].dropna()
        if len(prices) < 300:  # skip tickers with too little history
            continue

        for params in grid:
            count += 1
            if progress_cb:
                progress_cb(count, total)

            try:
                sig = _generate_signals(strategy, prices, params)
                result = run(prices, sig)

                if ranking == "Leveraged":
                    eq, ret = leveraged_metrics(result, leverage, financing_cost)
                else:
                    eq = result["equity"]
                    ret = result["strategy_ret"]

                cagr = annualized_return(eq)
                sharpe = sharpe_ratio(ret)
                mdd = max_drawdown(eq)
                calmar = calmar_ratio(eq)
                bh_cagr = annualized_return(result["bh_equity"])

                row = {
                    "Ticker": ticker,
                    "Params": params,
                    "CAGR": cagr,
                    "Sharpe": sharpe,
                    "Max Drawdown": mdd,
                    "Calmar": calmar,
                    "BH CAGR": bh_cagr,
                    "Beats B&H": cagr > bh_cagr,
                    "_result": result,  # kept for plotting, not shown in table
                }
                results.append(row)

            except Exception:
                continue

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    sort_col = {
        "Total Return": "CAGR",
        "Sharpe Ratio": "Sharpe",
        "Calmar Ratio": "Calmar",
        "Leveraged": "CAGR",
    }[ranking]

    return df.sort_values(sort_col, ascending=False).reset_index(drop=True)
