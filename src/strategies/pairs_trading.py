import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import coint


def test_cointegration(series_a, series_b, significance=0.05):
    """
    Engle-Granger cointegration test.
    Returns (is_cointegrated, p_value).

    Two series are cointegrated if their linear combination is stationary —
    i.e. they drift together and deviations are temporary (mean-reverting).
    """
    _, p_value, _ = coint(series_a, series_b)
    return p_value < significance, p_value


def compute_spread(series_a, series_b):
    """
    OLS regression of series_a on series_b to find the hedge ratio β:
        spread = series_a - β * series_b

    β tells you how many units of B to short per unit of A held long.
    """
    model = OLS(series_a, add_constant(series_b)).fit()
    beta  = model.params.iloc[1]
    spread = series_a - beta * series_b
    return spread, beta


def signals(series_a, series_b, window=60, entry_z=2.0, exit_z=0.5):
    """
    Pairs trading strategy based on z-score of the spread.

    When the spread is unusually wide (high z-score):
        → long A, short B  (expect spread to narrow)
    When unusually narrow (low z-score):
        → short A, long B

    Logic (signal applies to series_a; series_b gets opposite position):
        z >  entry_z → short A  (-1)
        z < -entry_z → long A   ( 1)
        |z| < exit_z → flat     ( 0)

    Params:
        window  : rolling window for z-score normalization
        entry_z : z-score threshold to open a position
        exit_z  : z-score threshold to close a position

    Returns: (signal, spread, z_score)
    """
    spread, beta = compute_spread(series_a, series_b)

    rolling_mean = spread.rolling(window).mean()
    rolling_std  = spread.rolling(window).std()
    z_score = (spread - rolling_mean) / rolling_std

    signal = pd.Series(0, index=series_a.index)

    # entry
    signal[z_score >  entry_z] = -1
    signal[z_score < -entry_z] =  1

    # exit: override entry signals when z is close to 0
    signal[z_score.abs() < exit_z] = 0

    return signal, spread, z_score, beta
