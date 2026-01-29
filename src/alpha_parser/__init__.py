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
    Log, Abs, Sign, Sqrt, Power, Max, Min,
    log, abs_, sign, sqrt, power, max_, min_,
    _ensure_signal,
)

# Data access
from .data import Field, close, open, high, low, field, LazyData

# Primitives
from .primitives import Returns, Volatility, Volume, Adv, returns, volatility, volume, adv

# Time-series operations
from .timeseries import (
    TsMean, TsStd, TsSum, TsMax, TsMin, Delay, Delta, TsRank, FillForward,
    TsCorr, TsCov, Ewma, TsArgmax, TsArgmin, TsSkew, TsKurt, DecayLinear,
    TsMeanEvents, TsStdEvents, TsSumEvents, TsCountEvents,
    ts_mean, ts_std, ts_sum, ts_max, ts_min, delay, delta, ts_rank, fill_forward,
    ts_corr, ts_cov, ewma, ts_argmax, ts_argmin, ts_skew, ts_kurt, decay_linear,
    ts_mean_events, ts_std_events, ts_sum_events, ts_count_events,
)

# Cross-sectional operations
from .crosssection import (
    Rank, ZScore, Demean, Quantile, Winsorize, Scale, Truncate,
    rank, zscore, demean, quantile, winsorize, scale, truncate,
)

# Conditional operations
from .conditional import Where, where

# Group operations
from .groups import (
    GroupRank, GroupDemean, GroupCountValid,
    group_rank, group_demean, group_count_valid,
)

# Parser
from .parser import AlphaParser, alpha, compute_weights

# Risk model (optional - requires statsmodels)
try:
    from .risk import FactorRiskModel, FactorDefinition, RiskModelResults, DEFAULT_STYLE_FACTORS, PRICE_ONLY_FACTORS
    _HAS_RISK = True
except ImportError:
    _HAS_RISK = False

# Evaluation (sibling package)
from evaluation import (
    Backtest, BacktestResult,
    QuantileAnalysis, QuantileResult,
    sharpe_ratio, max_drawdown, top_drawdowns, return_on_gmv,
    annualized_return, annualized_volatility, calmar_ratio, sortino_ratio,
)


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
    'Log',
    'Abs',
    'Sign',
    'Sqrt',
    'Power',
    'Max',
    'Min',
    'log',
    'abs_',
    'sign',
    'sqrt',
    'power',
    'max_',
    'min_',
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
    'Adv',
    'returns',
    'volatility',
    'volume',
    'adv',

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
    'TsCorr',
    'TsCov',
    'Ewma',
    'TsArgmax',
    'TsArgmin',
    'TsSkew',
    'TsKurt',
    'DecayLinear',
    'ts_mean',
    'ts_std',
    'ts_sum',
    'ts_max',
    'ts_min',
    'delay',
    'delta',
    'ts_rank',
    'fill_forward',
    'ts_corr',
    'ts_cov',
    'ewma',
    'ts_argmax',
    'ts_argmin',
    'ts_skew',
    'ts_kurt',
    'decay_linear',
    'TsMeanEvents',
    'TsStdEvents',
    'TsSumEvents',
    'TsCountEvents',
    'ts_mean_events',
    'ts_std_events',
    'ts_sum_events',
    'ts_count_events',

    # Cross-sectional
    'Rank',
    'ZScore',
    'Demean',
    'Quantile',
    'Winsorize',
    'Scale',
    'Truncate',
    'rank',
    'zscore',
    'demean',
    'quantile',
    'winsorize',
    'scale',
    'truncate',

    # Conditional
    'Where',
    'where',

    # Group
    'GroupRank',
    'GroupDemean',
    'GroupCountValid',
    'group_rank',
    'group_demean',
    'group_count_valid',

    # Parser
    'AlphaParser',
    'alpha',
    'compute_weights',

    # Risk model (optional)
    *(['FactorRiskModel', 'FactorDefinition', 'RiskModelResults', 'DEFAULT_STYLE_FACTORS', 'PRICE_ONLY_FACTORS'] if _HAS_RISK else []),

    # Evaluation
    'Backtest',
    'BacktestResult',
    'QuantileAnalysis',
    'QuantileResult',
    'sharpe_ratio',
    'max_drawdown',
    'top_drawdowns',
    'return_on_gmv',
    'annualized_return',
    'annualized_volatility',
    'calmar_ratio',
    'sortino_ratio',
]
