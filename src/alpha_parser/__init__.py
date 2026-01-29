"""
Alpha Parser - A DSL for defining quantitative trading signals.

Example usage:
    from alpha_parser import alpha, compute_weights, compute_context

    # Parse a signal expression
    signal = alpha("rank(-returns(20) / volatility(60))")

    # Evaluate with data
    result = signal.evaluate(data)

    # Convert to portfolio weights
    weights = signal.to_weights(data, normalize=True, long_only=False)

    # Use compute context for caching
    with compute_context():
        signal1 = alpha("-returns(20) / volatility(60)")
        signal2 = alpha("rank(returns(252))")
        result1 = signal1.evaluate(data)
        result2 = signal2.evaluate(data)
"""

# Context management
from .context import compute_context, get_context, ComputeContext

# Base class
from .signal import Signal

# Operators
from .operators import (
    Constant, BinaryOp,
    Add, Sub, Mul, Div, Neg,
    Greater, Less, GreaterEqual, LessEqual, Equal, NotEqual,
    And, Or, Not,
    IsValid, is_valid,
    _ensure_signal,
)

# Data access
from .data import Field, close, open, high, low, field, LazyData

# Primitives
from .primitives import Returns, Volatility, Volume, returns, volatility, volume

# Time-series operations
from .timeseries import (
    TsMean, TsStd, TsSum, TsMax, TsMin, Delay, Delta, TsRank, FillForward,
    ts_mean, ts_std, ts_sum, ts_max, ts_min, delay, delta, ts_rank, fill_forward,
)

# Cross-sectional operations
from .crosssection import Rank, ZScore, Demean, Quantile, Winsorize, rank, zscore, demean, quantile, winsorize

# Conditional operations
from .conditional import Where, where

# Group operations
from .groups import (
    GroupRank, GroupDemean, GroupNeutralize, GroupCountValid,
    group_rank, group_demean, group_neutralize, group_count_valid,
)

# Parser
from .parser import AlphaParser, alpha, compute_weights


__all__ = [
    # Context
    'compute_context',
    'get_context',
    'ComputeContext',

    # Base
    'Signal',

    # Operators
    'Constant',
    'BinaryOp',
    'Add',
    'Sub',
    'Mul',
    'Div',
    'Neg',
    'Greater',
    'Less',
    'GreaterEqual',
    'LessEqual',
    'Equal',
    'NotEqual',
    'And',
    'Or',
    'Not',
    'IsValid',
    'is_valid',
    '_ensure_signal',

    # Data access
    'Field',
    'LazyData',
    'close',
    'open',
    'high',
    'low',
    'field',

    # Primitives
    'Returns',
    'Volatility',
    'Volume',
    'returns',
    'volatility',
    'volume',

    # Time-series
    'TsMean',
    'TsStd',
    'TsSum',
    'TsMax',
    'TsMin',
    'Delay',
    'Delta',
    'TsRank',
    'FillForward',
    'ts_mean',
    'ts_std',
    'ts_sum',
    'ts_max',
    'ts_min',
    'delay',
    'delta',
    'ts_rank',
    'fill_forward',

    # Cross-sectional
    'Rank',
    'ZScore',
    'Demean',
    'Quantile',
    'Winsorize',
    'rank',
    'zscore',
    'demean',
    'quantile',
    'winsorize',

    # Conditional
    'Where',
    'where',

    # Group
    'GroupRank',
    'GroupDemean',
    'GroupNeutralize',
    'GroupCountValid',
    'group_rank',
    'group_demean',
    'group_neutralize',
    'group_count_valid',

    # Parser
    'AlphaParser',
    'alpha',
    'compute_weights',
]
