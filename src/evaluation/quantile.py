"""Quantile analysis for alpha signals."""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from alpha_parser.signal import Signal
from alpha_parser.context import compute_context
from .metrics import sharpe_ratio, annualized_return


@dataclass
class QuantileResult:
    """Results from quantile analysis."""

    # Per-quantile statistics
    quantile_returns: pd.DataFrame  # Daily returns per quantile
    quantile_cumulative: pd.DataFrame  # Cumulative returns per quantile
    mean_returns: pd.Series  # Mean return by quantile
    sharpe_by_quantile: pd.Series  # Sharpe ratio by quantile
    hit_rate: pd.Series  # % of days with positive return by quantile

    # Long-short spread
    spread_returns: pd.Series  # Daily long-short returns (Q5 - Q1)
    spread_cumulative: pd.Series  # Cumulative spread returns
    spread_sharpe: float  # Sharpe of long-short spread
    spread_mean: float  # Mean daily return of spread

    # Monotonicity
    is_monotonic: bool  # Whether returns increase/decrease with quantile
    rank_ic: float  # Rank information coefficient

    # Metadata
    n_quantiles: int
    n_days: int
    n_stocks_per_day: float  # Average stocks per day

    def summary(self) -> str:
        """Return a formatted summary string."""
        lines = [
            "=" * 50,
            "QUANTILE ANALYSIS",
            "=" * 50,
            "",
            f"Quantiles: {self.n_quantiles}",
            f"Days: {self.n_days}",
            f"Avg Stocks/Day: {self.n_stocks_per_day:.0f}",
            "",
            "Returns by Quantile (annualized):",
        ]

        for q in self.mean_returns.index:
            ann_ret = self.mean_returns[q] * 252
            sharpe = self.sharpe_by_quantile[q]
            hit = self.hit_rate[q]
            lines.append(f"  Q{q}: {ann_ret:>7.2%}  Sharpe: {sharpe:>5.2f}  Hit: {hit:>5.1%}")

        lines.extend([
            "",
            "Long-Short Spread (Q{} - Q1):".format(self.n_quantiles),
            f"  Annual Return: {self.spread_mean * 252:>7.2%}",
            f"  Sharpe Ratio:  {self.spread_sharpe:>7.2f}",
            "",
            f"Rank IC: {self.rank_ic:.3f}",
            f"Monotonic: {'Yes' if self.is_monotonic else 'No'}",
            "=" * 50,
        ])

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"QuantileResult(n_quantiles={self.n_quantiles}, "
            f"spread_sharpe={self.spread_sharpe:.2f}, "
            f"ic={self.rank_ic:.3f})"
        )


class QuantileAnalysis:
    """
    Quantile analysis for alpha signals.

    Sorts stocks into quantiles by signal value each day,
    then tracks the forward returns of each quantile.

    Example:
        >>> signal = alpha("rank(returns(20)) - 0.5")
        >>> qa = QuantileAnalysis(signal, n_quantiles=5)
        >>> result = qa.run(data)
        >>> print(result.summary())
    """

    def __init__(
        self,
        signal: Signal,
        n_quantiles: int = 5,
        holding_period: int = 1,
    ):
        """
        Initialize quantile analysis.

        Args:
            signal: Alpha signal to analyze
            n_quantiles: Number of quantiles (default 5 for quintiles)
            holding_period: Forward return period in days (default 1)
        """
        self.signal = signal
        self.n_quantiles = n_quantiles
        self.holding_period = holding_period

    def run(
        self,
        data: Dict[str, pd.DataFrame],
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> QuantileResult:
        """
        Run quantile analysis.

        Args:
            data: Dictionary with 'close', 'volume', etc.
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            QuantileResult with all statistics
        """
        # Evaluate signal
        with compute_context():
            signal_values = self.signal.evaluate(data)

        # Calculate forward returns
        close = data["close"]
        forward_returns = close.pct_change(self.holding_period).shift(-self.holding_period)

        # Align dates
        common_idx = signal_values.index.intersection(forward_returns.index)
        if start_date:
            common_idx = common_idx[common_idx >= start_date]
        if end_date:
            common_idx = common_idx[common_idx <= end_date]

        signal_values = signal_values.loc[common_idx]
        forward_returns = forward_returns.loc[common_idx]

        # Drop rows where we don't have valid data
        valid_mask = signal_values.notna().any(axis=1) & forward_returns.notna().any(axis=1)
        signal_values = signal_values.loc[valid_mask]
        forward_returns = forward_returns.loc[valid_mask]

        # Assign quantiles for each day
        quantiles = signal_values.apply(
            lambda row: pd.qcut(
                row.dropna(),
                q=self.n_quantiles,
                labels=range(1, self.n_quantiles + 1),
                duplicates="drop",
            ).reindex(row.index),
            axis=1,
        )

        # Calculate returns by quantile
        quantile_returns = {}
        for q in range(1, self.n_quantiles + 1):
            # Mask for stocks in this quantile
            q_mask = (quantiles == q)
            # Equal weight within quantile
            q_weights = q_mask.astype(float)
            q_weights = q_weights.div(q_weights.sum(axis=1), axis=0).fillna(0)
            # Portfolio return
            q_returns = (q_weights * forward_returns).sum(axis=1)
            quantile_returns[q] = q_returns

        quantile_returns_df = pd.DataFrame(quantile_returns)

        # Cumulative returns
        quantile_cumulative = (1 + quantile_returns_df).cumprod()

        # Mean returns by quantile
        mean_returns = quantile_returns_df.mean()

        # Sharpe by quantile
        sharpe_by_quantile = quantile_returns_df.apply(sharpe_ratio)

        # Hit rate (% positive days)
        hit_rate = (quantile_returns_df > 0).mean()

        # Long-short spread
        spread_returns = quantile_returns_df[self.n_quantiles] - quantile_returns_df[1]
        spread_cumulative = (1 + spread_returns).cumprod()
        spread_sharpe = sharpe_ratio(spread_returns)
        spread_mean = spread_returns.mean()

        # Check monotonicity
        is_monotonic = (
            mean_returns.is_monotonic_increasing or
            mean_returns.is_monotonic_decreasing
        )

        # Rank IC (correlation between signal rank and forward return)
        rank_ics = []
        for idx in signal_values.index:
            sig_row = signal_values.loc[idx].dropna()
            ret_row = forward_returns.loc[idx].reindex(sig_row.index).dropna()
            common = sig_row.index.intersection(ret_row.index)
            if len(common) >= 5:
                ic = sig_row[common].corr(ret_row[common], method="spearman")
                if not np.isnan(ic):
                    rank_ics.append(ic)
        rank_ic = np.mean(rank_ics) if rank_ics else 0.0

        # Count stats
        n_days = len(signal_values)
        n_stocks_per_day = signal_values.notna().sum(axis=1).mean()

        return QuantileResult(
            quantile_returns=quantile_returns_df,
            quantile_cumulative=quantile_cumulative,
            mean_returns=mean_returns,
            sharpe_by_quantile=sharpe_by_quantile,
            hit_rate=hit_rate,
            spread_returns=spread_returns,
            spread_cumulative=spread_cumulative,
            spread_sharpe=spread_sharpe,
            spread_mean=spread_mean,
            is_monotonic=is_monotonic,
            rank_ic=rank_ic,
            n_quantiles=self.n_quantiles,
            n_days=n_days,
            n_stocks_per_day=n_stocks_per_day,
        )

    def ic_summary(
        self,
        data: Dict[str, pd.DataFrame],
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        Calculate detailed IC (Information Coefficient) statistics.

        Args:
            data: Dictionary with price data
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            DataFrame with IC statistics
        """
        # Evaluate signal
        with compute_context():
            signal_values = self.signal.evaluate(data)

        # Calculate forward returns
        close = data["close"]
        forward_returns = close.pct_change(self.holding_period).shift(-self.holding_period)

        # Align dates
        common_idx = signal_values.index.intersection(forward_returns.index)
        if start_date:
            common_idx = common_idx[common_idx >= start_date]
        if end_date:
            common_idx = common_idx[common_idx <= end_date]

        signal_values = signal_values.loc[common_idx]
        forward_returns = forward_returns.loc[common_idx]

        # Calculate daily IC
        ics = []
        for idx in signal_values.index:
            sig_row = signal_values.loc[idx].dropna()
            ret_row = forward_returns.loc[idx].reindex(sig_row.index).dropna()
            common = sig_row.index.intersection(ret_row.index)
            if len(common) >= 5:
                # Rank IC
                rank_ic = sig_row[common].corr(ret_row[common], method="spearman")
                # Pearson IC
                pearson_ic = sig_row[common].corr(ret_row[common], method="pearson")
                ics.append({
                    "date": idx,
                    "rank_ic": rank_ic,
                    "pearson_ic": pearson_ic,
                    "n_stocks": len(common),
                })

        ic_df = pd.DataFrame(ics).set_index("date")

        # Summary statistics
        summary = {
            "Mean Rank IC": ic_df["rank_ic"].mean(),
            "Std Rank IC": ic_df["rank_ic"].std(),
            "IC IR (Rank)": ic_df["rank_ic"].mean() / ic_df["rank_ic"].std() if ic_df["rank_ic"].std() > 0 else 0,
            "Mean Pearson IC": ic_df["pearson_ic"].mean(),
            "Std Pearson IC": ic_df["pearson_ic"].std(),
            "IC IR (Pearson)": ic_df["pearson_ic"].mean() / ic_df["pearson_ic"].std() if ic_df["pearson_ic"].std() > 0 else 0,
            "% Positive IC": (ic_df["rank_ic"] > 0).mean(),
            "Avg Stocks": ic_df["n_stocks"].mean(),
        }

        return pd.Series(summary)
