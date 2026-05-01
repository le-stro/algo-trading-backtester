import pandas as pd


def signals(prices, short_window=50, long_window=200):
    """
    Moving average crossover strategy.

    Logic:
        short MA > long MA  → long  ( 1)
        short MA < long MA  → out   ( 0)

    The classic '50/200 Golden Cross'. Simple trend-following:
    buy when short-term momentum is above long-term trend.

    Params:
        short_window : lookback for the fast moving average
        long_window  : lookback for the slow moving average

    Returns: pd.Series of signals (1 or 0), aligned to prices index.
    """
    short_ma = prices.rolling(short_window).mean()
    long_ma  = prices.rolling(long_window).mean()

    signal = (short_ma > long_ma).astype(int)
    return signal
