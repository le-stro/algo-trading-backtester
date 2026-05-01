import numpy as np
import pandas as pd


def run(prices, signals, initial_capital=10_000, transaction_cost=0.001):
    """
    Core backtesting engine. Takes a price series and a signal series,
    simulates trades, and returns an equity curve + daily returns.

    Params:
        prices           : pd.Series of daily prices
        signals          : pd.Series of integer signals aligned to prices
                           1 = long, 0 = out of market, -1 = short
        initial_capital  : starting portfolio value in currency units
        transaction_cost : fraction of trade value charged per trade
                           (e.g. 0.001 = 0.1% per side)

    Returns: pd.DataFrame with columns:
        signal       : the input signal
        price        : asset price
        position     : actual position held (after 1-day execution lag)
        trade        : 1 where position changed, else 0
        strategy_ret : daily return of the strategy
        bh_ret       : daily return of buy-and-hold
        equity       : strategy portfolio value
        bh_equity    : buy-and-hold portfolio value
    """
    df = pd.DataFrame({"signal": signals, "price": prices}).dropna()

    # execute signals with 1-day lag — avoids look-ahead bias
    df["position"] = df["signal"].shift(1).fillna(0)

    # detect position changes to apply transaction costs
    df["trade"] = (df["position"] != df["position"].shift(1)).astype(int)

    # daily price return
    df["price_ret"] = df["price"].pct_change().fillna(0)

    # strategy return = position * price return - cost on trade days
    df["strategy_ret"] = (
        df["position"] * df["price_ret"]
        - df["trade"] * transaction_cost
    )

    # buy-and-hold benchmark
    df["bh_ret"] = df["price_ret"]

    # compound equity curves
    df["equity"]    = initial_capital * (1 + df["strategy_ret"]).cumprod()
    df["bh_equity"] = initial_capital * (1 + df["bh_ret"]).cumprod()

    return df
