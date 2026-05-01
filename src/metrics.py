import numpy as np
import pandas as pd


def sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    """
    Annualized Sharpe ratio.
    """
    excess = returns - risk_free_rate / periods_per_year
    if excess.std() == 0:
        return np.nan
    return np.sqrt(periods_per_year) * excess.mean() / excess.std()


def max_drawdown(equity_curve):
    """
    Largest peak-to-trough decline in portfolio value.
    Returns a negative number (e.g. -0.23 means -23%).
    """
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    return drawdown.min()


def annualized_return(equity_curve, periods_per_year=252):
    """
    Compound annual growth rate (CAGR) from an equity curve.
    """
    n_periods = len(equity_curve)
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    return total_return ** (periods_per_year / n_periods) - 1


def calmar_ratio(equity_curve, periods_per_year=252):
    """
    Ratio of annualized return to max drawdown (absolute).
    High Calmar ratio indicates better risk-adjusted returns.
    """
    ann_return = annualized_return(equity_curve, periods_per_year)
    mdd = max_drawdown(equity_curve)

    # Vermeidung von Division durch Null, falls kein Drawdown vorhanden ist
    if mdd == 0:
        return np.nan

    return ann_return / abs(mdd)


def win_rate(returns):
    """Fraction of trades with positive return."""
    trades = returns[returns != 0]
    if len(trades) == 0:
        return np.nan
    return (trades > 0).sum() / len(trades)


def summary(equity_curve, returns, risk_free_rate=0.0):
    """
    Print a performance summary including Calmar Ratio.
    """
    metrics = {
        "Annualized return": f"{annualized_return(equity_curve):.2%}",
        "Sharpe ratio": f"{sharpe_ratio(returns, risk_free_rate):.2f}",
        "Max drawdown": f"{max_drawdown(equity_curve):.2%}",
        "Calmar ratio": f"{calmar_ratio(equity_curve):.2f}",
        "Win rate": f"{win_rate(returns):.2%}",
        "Total trades": int((returns != 0).sum()),
    }
    for k, v in metrics.items():
        print(f"  {k:<22}: {v}")
    return metrics
