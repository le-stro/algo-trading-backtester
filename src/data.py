import yfinance as yf
import pandas as pd


def load_prices(ticker, start, end, column="Close"):
    """
    Download daily price data from Yahoo Finance.

    Params:
        ticker : e.g. 'AAPL'
        start  : 'YYYY-MM-DD'
        end    : 'YYYY-MM-DD'
        column : price column to return (default: adjusted close)

    Returns: pd.Series with DatetimeIndex
    """
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")
    return df[column].squeeze()


def load_multi(tickers, start, end, column="Close"):
    """
    Download prices for multiple tickers.
    Returns: pd.DataFrame, one column per ticker.
    """
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    return df[column]


def compute_returns(prices):
    """Daily log returns. Log preferred over simple returns for multi-period compounding."""
    import numpy as np
    return np.log(prices / prices.shift(1)).dropna()
