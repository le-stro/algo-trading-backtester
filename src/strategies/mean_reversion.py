import pandas as pd


def signals(prices, window=20, n_std=2.0):
    """
    Mean reversion strategy using Bollinger Bands.

    Bollinger Bands define a channel around a rolling mean:
        upper band = mean + n_std * std
        lower band = mean - n_std * std

    Logic:
        price < lower band  → long  ( 1)  price is "too low", expect reversion up
        price > upper band  → short (-1)  price is "too high", expect reversion down
        price inside bands  → flat  ( 0)

    Params:
        window : lookback window for mean and std
        n_std  : number of standard deviations for band width

    Returns: pd.Series of signals (1, 0, -1), aligned to prices index.
    """
    rolling_mean = prices.rolling(window).mean()
    rolling_std  = prices.rolling(window).std()

    upper = rolling_mean + n_std * rolling_std
    lower = rolling_mean - n_std * rolling_std

    signal = pd.Series(0, index=prices.index)
    signal[prices < lower] =  1
    signal[prices > upper] = -1

    return signal
