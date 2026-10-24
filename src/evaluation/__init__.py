"""Evaluation module for alpha signal backtesting and analysis."""

from .backtest import Backtest, BacktestResult
from .book import BookSettings, build_book
from .metrics import (
    sharpe_ratio,
    max_drawdown,
    top_drawdowns,
    return_on_gmv,
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    sortino_ratio,
    turnover,
    fitness,
    margin_bps,
)
from .quantile import QuantileAnalysis, QuantileResult

__all__ = [
    # Backtest
    "Backtest",
    "BacktestResult",
    # Book construction
    "BookSettings",
    "build_book",
    # Metrics
    "sharpe_ratio",
    "max_drawdown",
    "top_drawdowns",
    "return_on_gmv",
    "annualized_return",
    "annualized_volatility",
    "calmar_ratio",
    "sortino_ratio",
    "turnover",
    "fitness",
    "margin_bps",
    # Quantile
    "QuantileAnalysis",
    "QuantileResult",
]
