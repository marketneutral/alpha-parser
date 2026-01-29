"""Backtest engine for alpha signals."""

from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

from alpha_parser.signal import Signal
from alpha_parser.context import compute_context
from .metrics import (
    sharpe_ratio,
    max_drawdown,
    top_drawdowns,
    return_on_gmv,
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    sortino_ratio,
    Drawdown,
)


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    # Time series
    pnl: pd.Series  # Daily PnL
    cumulative_pnl: pd.Series  # Cumulative PnL
    weights: pd.DataFrame  # Daily weights
    returns: pd.Series  # Daily returns (PnL / GMV)

    # Summary metrics
    total_return: float  # Total cumulative return
    sharpe: float  # Annualized Sharpe ratio
    annual_return: float  # Annualized return
    annual_volatility: float  # Annualized volatility
    max_drawdown: float  # Maximum drawdown
    calmar: float  # Calmar ratio
    sortino: float  # Sortino ratio
    return_gmv: float  # Return on GMV

    # Drawdowns
    drawdowns: list  # Top drawdowns

    # Position stats
    avg_long_count: float  # Average number of long positions
    avg_short_count: float  # Average number of short positions
    avg_turnover: float  # Average daily turnover (sum of weight changes)

    def summary(self) -> str:
        """Return a formatted summary string."""
        lines = [
            "=" * 50,
            "BACKTEST RESULTS",
            "=" * 50,
            "",
            "Performance Metrics:",
            f"  Total Return:      {self.total_return:>10.2%}",
            f"  Annual Return:     {self.annual_return:>10.2%}",
            f"  Annual Volatility: {self.annual_volatility:>10.2%}",
            f"  Sharpe Ratio:      {self.sharpe:>10.2f}",
            f"  Sortino Ratio:     {self.sortino:>10.2f}",
            f"  Calmar Ratio:      {self.calmar:>10.2f}",
            f"  Max Drawdown:      {self.max_drawdown:>10.2%}",
            f"  Return on GMV:     {self.return_gmv:>10.2%}",
            "",
            "Position Statistics:",
            f"  Avg Long Count:    {self.avg_long_count:>10.1f}",
            f"  Avg Short Count:   {self.avg_short_count:>10.1f}",
            f"  Avg Daily Turnover:{self.avg_turnover:>10.2%}",
            "",
            "Top Drawdowns:",
        ]

        for i, dd in enumerate(self.drawdowns[:5], 1):
            lines.append(f"  {i}. {dd}")

        lines.append("=" * 50)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"BacktestResult(sharpe={self.sharpe:.2f}, "
            f"return={self.total_return:.2%}, "
            f"max_dd={self.max_drawdown:.2%})"
        )


class Backtest:
    """
    WorldQuant-style backtest engine for alpha signals.

    Runs a daily-rebalanced long-short backtest with:
    - Dollar-neutral portfolio (equal long/short exposure)
    - Weights derived from signal.to_weights()
    - PnL = sum(weights * forward_returns)

    Example:
        >>> signal = alpha("rank(returns(20)) - 0.5")
        >>> bt = Backtest(signal)
        >>> result = bt.run(data)
        >>> print(result.summary())
    """

    def __init__(
        self,
        signal: Signal,
        holding_period: int = 1,
        transaction_cost: float = 0.0,
    ):
        """
        Initialize backtest.

        Args:
            signal: Alpha signal to backtest
            holding_period: Days to hold positions before rebalancing (default 1)
            transaction_cost: One-way transaction cost as decimal (default 0)
        """
        self.signal = signal
        self.holding_period = holding_period
        self.transaction_cost = transaction_cost

    def run(
        self,
        data: Dict[str, pd.DataFrame],
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> BacktestResult:
        """
        Run the backtest.

        Args:
            data: Dictionary with 'close', 'volume', etc.
            start_date: Optional start date for backtest
            end_date: Optional end date for backtest

        Returns:
            BacktestResult with all metrics and time series
        """
        # Evaluate signal
        with compute_context():
            weights = self.signal.to_weights(data)

        # Calculate forward returns
        close = data["close"]
        forward_returns = close.pct_change(self.holding_period).shift(-self.holding_period)

        # Align dates
        common_idx = weights.index.intersection(forward_returns.index)
        if start_date:
            common_idx = common_idx[common_idx >= start_date]
        if end_date:
            common_idx = common_idx[common_idx <= end_date]

        weights = weights.loc[common_idx]
        forward_returns = forward_returns.loc[common_idx]

        # Drop rows where we don't have valid data
        valid_mask = weights.notna().any(axis=1) & forward_returns.notna().any(axis=1)
        weights = weights.loc[valid_mask]
        forward_returns = forward_returns.loc[valid_mask]

        # Fill NaN weights with 0
        weights = weights.fillna(0)
        forward_returns = forward_returns.fillna(0)

        # Calculate PnL (before costs)
        daily_pnl = (weights * forward_returns).sum(axis=1)

        # Calculate turnover and transaction costs
        weight_changes = weights.diff().abs().sum(axis=1)
        weight_changes.iloc[0] = weights.iloc[0].abs().sum()  # Initial positions
        turnover = weight_changes

        # Subtract transaction costs (both entry and exit)
        if self.transaction_cost > 0:
            daily_pnl = daily_pnl - turnover * self.transaction_cost

        # Cumulative PnL
        cumulative_pnl = daily_pnl.cumsum()

        # Calculate returns (relative to GMV)
        gmv = weights.abs().sum(axis=1)
        returns = daily_pnl / gmv.replace(0, np.nan)
        returns = returns.fillna(0)

        # Cumulative returns for drawdown calculation
        cumulative_returns = (1 + returns).cumprod()

        # Position stats
        long_counts = (weights > 0).sum(axis=1)
        short_counts = (weights < 0).sum(axis=1)

        # Calculate metrics
        result = BacktestResult(
            # Time series
            pnl=daily_pnl,
            cumulative_pnl=cumulative_pnl,
            weights=weights,
            returns=returns,
            # Summary metrics
            total_return=cumulative_returns.iloc[-1] - 1 if len(cumulative_returns) > 0 else 0.0,
            sharpe=sharpe_ratio(returns),
            annual_return=annualized_return(returns),
            annual_volatility=annualized_volatility(returns),
            max_drawdown=max_drawdown(cumulative_returns),
            calmar=calmar_ratio(returns),
            sortino=sortino_ratio(returns),
            return_gmv=return_on_gmv(daily_pnl, weights),
            # Drawdowns
            drawdowns=top_drawdowns(cumulative_returns, n=5),
            # Position stats
            avg_long_count=long_counts.mean(),
            avg_short_count=short_counts.mean(),
            avg_turnover=turnover.mean(),
        )

        return result

    def run_walk_forward(
        self,
        data: Dict[str, pd.DataFrame],
        train_period: int = 252,
        test_period: int = 63,
    ) -> pd.DataFrame:
        """
        Run walk-forward analysis.

        Splits data into rolling train/test periods and reports
        out-of-sample performance for each period.

        Args:
            data: Dictionary with price data
            train_period: Days in training period
            test_period: Days in test period

        Returns:
            DataFrame with metrics for each test period
        """
        close = data["close"]
        dates = close.index

        results = []
        i = train_period

        while i + test_period <= len(dates):
            test_start = dates[i]
            test_end = dates[min(i + test_period - 1, len(dates) - 1)]

            result = self.run(data, start_date=test_start, end_date=test_end)

            results.append({
                "start": test_start,
                "end": test_end,
                "sharpe": result.sharpe,
                "return": result.total_return,
                "volatility": result.annual_volatility,
                "max_drawdown": result.max_drawdown,
            })

            i += test_period

        return pd.DataFrame(results)
