"""Performance metrics for alpha signal evaluation."""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, annualization: int = 252) -> float:
    """
    Calculate annualized Sharpe ratio.

    Args:
        returns: Daily returns series
        annualization: Trading days per year (default 252)

    Returns:
        Annualized Sharpe ratio
    """
    if returns.std() == 0:
        return 0.0
    return returns.mean() / returns.std() * np.sqrt(annualization)


def annualized_return(returns: pd.Series, annualization: int = 252) -> float:
    """
    Calculate annualized return.

    Args:
        returns: Daily returns series
        annualization: Trading days per year (default 252)

    Returns:
        Annualized return as decimal (0.10 = 10%)
    """
    return returns.mean() * annualization


def annualized_volatility(returns: pd.Series, annualization: int = 252) -> float:
    """
    Calculate annualized volatility.

    Args:
        returns: Daily returns series
        annualization: Trading days per year (default 252)

    Returns:
        Annualized volatility as decimal
    """
    return returns.std() * np.sqrt(annualization)


def return_on_gmv(
    pnl: pd.Series,
    weights: pd.DataFrame,
) -> float:
    """
    Calculate return on Gross Market Value.

    GMV = sum of absolute weights (gross exposure)
    Return on GMV = total PnL / average GMV

    Args:
        pnl: Daily PnL series
        weights: Daily weights DataFrame (stocks x dates)

    Returns:
        Total return on GMV as decimal
    """
    gmv = weights.abs().sum(axis=1)
    avg_gmv = gmv.mean()
    if avg_gmv == 0:
        return 0.0
    total_pnl = pnl.sum()
    return total_pnl / avg_gmv


def max_drawdown(cumulative_returns: pd.Series) -> float:
    """
    Calculate maximum drawdown.

    Args:
        cumulative_returns: Cumulative returns series (1 + cumsum of returns)

    Returns:
        Maximum drawdown as positive decimal (0.20 = 20% drawdown)
    """
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    return -drawdown.min()


@dataclass
class Drawdown:
    """A single drawdown period."""

    start: pd.Timestamp
    trough: pd.Timestamp
    end: pd.Timestamp  # Recovery date (or last date if not recovered)
    depth: float  # Maximum depth as positive decimal
    duration: int  # Days from start to trough
    recovery: int  # Days from trough to recovery (or -1 if not recovered)

    def __repr__(self) -> str:
        recovery_str = f"{self.recovery}d" if self.recovery >= 0 else "ongoing"
        return (
            f"Drawdown({self.depth:.1%} from {self.start.date()} to {self.trough.date()}, "
            f"duration={self.duration}d, recovery={recovery_str})"
        )


def top_drawdowns(
    cumulative_returns: pd.Series,
    n: int = 5,
) -> List[Drawdown]:
    """
    Find the top N drawdowns by depth.

    Args:
        cumulative_returns: Cumulative returns series
        n: Number of top drawdowns to return

    Returns:
        List of Drawdown objects sorted by depth (largest first)
    """
    running_max = cumulative_returns.cummax()
    drawdown_pct = (cumulative_returns - running_max) / running_max

    drawdowns = []
    in_drawdown = False
    start_idx = None
    trough_idx = None
    trough_val = 0.0

    for i, (idx, dd) in enumerate(drawdown_pct.items()):
        if dd < 0 and not in_drawdown:
            # Starting a new drawdown
            in_drawdown = True
            start_idx = drawdown_pct.index[max(0, i - 1)]  # Peak before drawdown
            trough_idx = idx
            trough_val = dd
        elif dd < 0 and in_drawdown:
            # Continuing drawdown
            if dd < trough_val:
                trough_idx = idx
                trough_val = dd
        elif dd >= 0 and in_drawdown:
            # Recovered from drawdown
            in_drawdown = False
            drawdowns.append(Drawdown(
                start=start_idx,
                trough=trough_idx,
                end=idx,
                depth=-trough_val,
                duration=(trough_idx - start_idx).days,
                recovery=(idx - trough_idx).days,
            ))

    # Handle ongoing drawdown at end of series
    if in_drawdown:
        drawdowns.append(Drawdown(
            start=start_idx,
            trough=trough_idx,
            end=drawdown_pct.index[-1],
            depth=-trough_val,
            duration=(trough_idx - start_idx).days,
            recovery=-1,  # Not recovered
        ))

    # Sort by depth and return top N
    drawdowns.sort(key=lambda x: x.depth, reverse=True)
    return drawdowns[:n]


def calmar_ratio(
    returns: pd.Series,
    annualization: int = 252,
) -> float:
    """
    Calculate Calmar ratio (annualized return / max drawdown).

    Args:
        returns: Daily returns series
        annualization: Trading days per year

    Returns:
        Calmar ratio
    """
    cumulative = (1 + returns).cumprod()
    mdd = max_drawdown(cumulative)
    if mdd == 0:
        return np.inf if returns.mean() > 0 else 0.0
    ann_ret = annualized_return(returns, annualization)
    return ann_ret / mdd


def sortino_ratio(
    returns: pd.Series,
    annualization: int = 252,
    target: float = 0.0,
) -> float:
    """
    Calculate Sortino ratio (return / downside deviation).

    Args:
        returns: Daily returns series
        annualization: Trading days per year
        target: Target return (default 0)

    Returns:
        Annualized Sortino ratio
    """
    downside = returns[returns < target]
    if len(downside) == 0 or downside.std() == 0:
        return np.inf if returns.mean() > target else 0.0
    downside_std = np.sqrt((downside ** 2).mean())
    return (returns.mean() - target) / downside_std * np.sqrt(annualization)
