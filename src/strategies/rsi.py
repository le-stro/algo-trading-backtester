import pandas as pd
import numpy as np


def compute_rsi(prices, window=14):
    """
    Relative Strength Index (RSI).

    RSI measures the speed and magnitude of recent price changes.
    Ranges from 0 to 100:
        > 70 → overbought (momentum may reverse down)
        < 30 → oversold   (momentum may reverse up)

    Uses Wilder's smoothing (exponential, not simple moving average).
    """
    delta = prices.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, min_periods=window, adjust=False).mean()

    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def signals(prices, window=14, oversold=30, overbought=70):
    """
    RSI-based mean reversion signals.

    Logic:
        RSI < oversold   → long  ( 1)
        RSI > overbought → short (-1)
        otherwise        → flat  ( 0)

    Params:
        window     : RSI lookback period (standard: 14)
        oversold   : RSI threshold to go long
        overbought : RSI threshold to go short

    Returns: pd.Series of signals (1, 0, -1), aligned to prices index.
    """
    rsi = compute_rsi(prices, window)

    signal = pd.Series(0, index=prices.index)
    signal[rsi < oversold]   =  1
    signal[rsi > overbought] = -1

    return signal, rsi  # return rsi too for plotting in notebook
