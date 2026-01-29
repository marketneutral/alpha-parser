"""Evaluation module for alpha signal backtesting and analysis."""

from .backtest import Backtest, BacktestResult
from .metrics import (
    sharpe_ratio,
    max_drawdown,
    top_drawdowns,
    return_on_gmv,
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    sortino_ratio,
)
from .quantile import QuantileAnalysis, QuantileResult

__all__ = [
    # Backtest
    "Backtest",
    "BacktestResult",
    # Metrics
    "sharpe_ratio",
    "max_drawdown",
    "top_drawdowns",
    "return_on_gmv",
    "annualized_return",
    "annualized_volatility",
    "calmar_ratio",
    "sortino_ratio",
    # Quantile
    "QuantileAnalysis",
    "QuantileResult",
]
