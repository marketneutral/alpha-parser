"""
Qex (Quant Expression) - A DSL for defining quantitative trading signals.

Example usage:
    from qex import qex, compute_weights, compute_context

    # Parse a signal expression
    signal = qex("rank(-returns(20) / volatility(60))")

    # Evaluate with data
    result = signal.evaluate(data)

    # Convert to portfolio weights
    weights = signal.to_weights(data, normalize=True, long_only=False)

    # Use compute context for caching
    with compute_context():
        signal1 = qex("-returns(20) / volatility(60)")
        signal2 = qex("rank(returns(252))")
        result1 = signal1.evaluate(data)
        result2 = signal2.evaluate(data)

Backwards compatibility:
    `alpha()` and `AlphaParser` are still available as aliases.
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
from .data import (
    Field, close, open, high, low, field, LazyData,
    DayOfWeek, DayOfMonth, MonthOfYear,
    day_of_week, day_of_month, month_of_year,
)

# Primitives
from .primitives import Returns, Volatility, Volume, Adv, returns, volatility, volume, adv

# Time-series operations
from .timeseries import (
    TsMean, TsStd, TsSum, TsMax, TsMin, TsVar, Delay, Delta, TsRank, FillForward,
    TsCorr, TsCov, Ewma, EwmaVar, EwmaCov, TsBeta, TsBetaEwma,
    TsArgmax, TsArgmin, TsSkew, TsKurt, DecayLinear,
    TsMeanEvents, TsStdEvents, TsSumEvents, TsCountEvents,
    ts_mean, ts_std, ts_sum, ts_max, ts_min, ts_var, delay, delta, ts_rank, fill_forward,
    ts_corr, ts_cov, ewma, ewma_var, ewma_cov, ts_beta, ts_beta_ewma,
    ts_argmax, ts_argmin, ts_skew, ts_kurt, decay_linear,
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
    GroupRank, GroupDemean, GroupCountValid, GroupStd, GroupSum,
    group_rank, group_demean, group_count_valid, group_std, group_sum,
)

# Parser
from .parser import QexParser, qex, AlphaParser, alpha, compute_weights

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

    # Calendar
    'DayOfWeek',
    'DayOfMonth',
    'MonthOfYear',
    'day_of_week',
    'day_of_month',
    'month_of_year',

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
    'TsVar',
    'Delay',
    'Delta',
    'TsRank',
    'FillForward',
    'TsCorr',
    'TsCov',
    'Ewma',
    'EwmaVar',
    'EwmaCov',
    'TsBeta',
    'TsBetaEwma',
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
    'ts_var',
    'delay',
    'delta',
    'ts_rank',
    'fill_forward',
    'ts_corr',
    'ts_cov',
    'ewma',
    'ewma_var',
    'ewma_cov',
    'ts_beta',
    'ts_beta_ewma',
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
    'GroupStd',
    'GroupSum',
    'group_rank',
    'group_demean',
    'group_count_valid',
    'group_std',
    'group_sum',

    # Parser
    'QexParser',
    'qex',
    'AlphaParser',  # backwards compatibility
    'alpha',  # backwards compatibility
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
