import numpy as np
import pandas as pd
import hashlib
from contextlib import contextmanager
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import ast
import operator


# ============================================================================
# Context Management for Caching
# ============================================================================

class ComputeContext:
    """Global computation context with shared cache"""

    def __init__(self):
        self.cache = {}
        self.data_hash = None

    def get_data_hash(self, data: Dict[str, pd.DataFrame]) -> str:
        """Create stable hash of data for cache validation"""
        parts = []
        for key in sorted(data.keys()):
            df = data[key]
            if isinstance(df, pd.DataFrame):
                parts.append(f"{key}:{df.shape}:{df.index[0]}:{df.index[-1]}")
            else:
                parts.append(f"{key}:{id(df)}")
        return hashlib.md5("".join(parts).encode()).hexdigest()

    def clear(self):
        """Clear the cache"""
        self.cache = {}
        self.data_hash = None


_context: Optional[ComputeContext] = None


@contextmanager
def compute_context():
    """Context manager for alpha computations with shared cache"""
    global _context
    old_context = _context
    _context = ComputeContext()
    try:
        yield _context
    finally:
        _context = old_context


def get_context() -> Optional[ComputeContext]:
    """Get current compute context"""
    return _context


# ============================================================================
# Core Signal Classes
# ============================================================================

class Signal(ABC):
    """Base class for all signal expressions"""

    @abstractmethod
    def _compute(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Actual computation logic - implemented by subclasses"""
        pass

    @abstractmethod
    def _cache_key(self) -> tuple:
        """Return hashable key for this computation"""
        pass

    def evaluate(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Evaluate signal with optional global caching"""
        ctx = get_context()

        if ctx is None:
            return self._compute(data)

        # Validate data hasn't changed
        data_hash = ctx.get_data_hash(data)
        if ctx.data_hash is None:
            ctx.data_hash = data_hash
        elif ctx.data_hash != data_hash:
            ctx.cache = {}
            ctx.data_hash = data_hash

        # Check cache
        cache_key = self._cache_key()
        if cache_key in ctx.cache:
            return ctx.cache[cache_key]

        # Compute and cache
        result = self._compute(data)
        ctx.cache[cache_key] = result
        return result

    def to_weights(self,
                   data: Dict[str, pd.DataFrame],
                   normalize: bool = True,
                   long_only: bool = False) -> pd.DataFrame:
        """
        Convert signal to portfolio weights for backtesting integration.

        Args:
            data: Price and volume data
            normalize: If True, normalize weights to sum to 1 (or 0 for long-short)
            long_only: If True, clip negative weights to 0

        Returns:
            DataFrame of portfolio weights
        """
        weights = self.evaluate(data)

        if long_only:
            weights = weights.clip(lower=0)

        if normalize:
            if long_only:
                row_sums = weights.sum(axis=1)
                weights = weights.div(row_sums, axis=0)
            else:
                row_abs_sums = weights.abs().sum(axis=1)
                weights = weights.div(row_abs_sums, axis=0)

        return weights.fillna(0)

    # Arithmetic operators
    def __neg__(self):
        return Neg(self)

    def __add__(self, other):
        return Add(self, _ensure_signal(other))

    def __sub__(self, other):
        return Sub(self, _ensure_signal(other))

    def __mul__(self, other):
        return Mul(self, _ensure_signal(other))

    def __truediv__(self, other):
        return Div(self, _ensure_signal(other))

    def __radd__(self, other):
        return Add(_ensure_signal(other), self)

    def __rsub__(self, other):
        return Sub(_ensure_signal(other), self)

    def __rmul__(self, other):
        return Mul(_ensure_signal(other), self)

    def __rtruediv__(self, other):
        return Div(_ensure_signal(other), self)

    # Comparison operators
    def __gt__(self, other):
        return Greater(self, _ensure_signal(other))

    def __lt__(self, other):
        return Less(self, _ensure_signal(other))

    def __ge__(self, other):
        return GreaterEqual(self, _ensure_signal(other))

    def __le__(self, other):
        return LessEqual(self, _ensure_signal(other))

    def __eq__(self, other):
        return Equal(self, _ensure_signal(other))

    def __ne__(self, other):
        return NotEqual(self, _ensure_signal(other))

    # Logical operators
    def __and__(self, other):
        return And(self, _ensure_signal(other))

    def __or__(self, other):
        return Or(self, _ensure_signal(other))

    def __invert__(self):
        return Not(self)


def _ensure_signal(x):
    """Convert constants to Constant signals"""
    if isinstance(x, Signal):
        return x
    return Constant(x)


# ============================================================================
# Basic Operations
# ============================================================================

class Constant(Signal):
    def __init__(self, value):
        self.value = value

    def _compute(self, data):
        return self.value

    def _cache_key(self):
        return ('Constant', self.value)


class BinaryOp(Signal):
    def __init__(self, left: Signal, right: Signal):
        self.left = left
        self.right = right

    def _cache_key(self):
        return (self.__class__.__name__, self.left._cache_key(), self.right._cache_key())


class Add(BinaryOp):
    def _compute(self, data):
        return self.left.evaluate(data) + self.right.evaluate(data)


class Sub(BinaryOp):
    def _compute(self, data):
        return self.left.evaluate(data) - self.right.evaluate(data)


class Mul(BinaryOp):
    def _compute(self, data):
        return self.left.evaluate(data) * self.right.evaluate(data)


class Div(BinaryOp):
    def _compute(self, data):
        return self.left.evaluate(data) / self.right.evaluate(data)


class Neg(Signal):
    def __init__(self, signal: Signal):
        self.signal = signal

    def _compute(self, data):
        return -self.signal.evaluate(data)

    def _cache_key(self):
        return ('Neg', self.signal._cache_key())


# ============================================================================
# Comparison Operations
# ============================================================================

class Greater(BinaryOp):
    def _compute(self, data):
        return (self.left.evaluate(data) > self.right.evaluate(data)).astype(float)


class Less(BinaryOp):
    def _compute(self, data):
        return (self.left.evaluate(data) < self.right.evaluate(data)).astype(float)


class GreaterEqual(BinaryOp):
    def _compute(self, data):
        return (self.left.evaluate(data) >= self.right.evaluate(data)).astype(float)


class LessEqual(BinaryOp):
    def _compute(self, data):
        return (self.left.evaluate(data) <= self.right.evaluate(data)).astype(float)


class Equal(BinaryOp):
    def _compute(self, data):
        return (self.left.evaluate(data) == self.right.evaluate(data)).astype(float)


class NotEqual(BinaryOp):
    def _compute(self, data):
        return (self.left.evaluate(data) != self.right.evaluate(data)).astype(float)


# ============================================================================
# Logical Operations
# ============================================================================

class And(BinaryOp):
    def _compute(self, data):
        left = self.left.evaluate(data).astype(bool)
        right = self.right.evaluate(data).astype(bool)
        return (left & right).astype(float)


class Or(BinaryOp):
    def _compute(self, data):
        left = self.left.evaluate(data).astype(bool)
        right = self.right.evaluate(data).astype(bool)
        return (left | right).astype(float)


class Not(Signal):
    def __init__(self, signal: Signal):
        self.signal = signal

    def _compute(self, data):
        return (~self.signal.evaluate(data).astype(bool)).astype(float)

    def _cache_key(self):
        return ('Not', self.signal._cache_key())


# ============================================================================
# Data Access
# ============================================================================

class Field(Signal):
    """Access a raw data field by name"""
    def __init__(self, name: str):
        self.name = name

    def _compute(self, data):
        if self.name not in data:
            raise ValueError(f"Field '{self.name}' not found in data. "
                           f"Available fields: {list(data.keys())}")
        return data[self.name]

    def _cache_key(self):
        return ('Field', self.name)


def close() -> Field:
    return Field('close')


def open() -> Field:
    return Field('open')


def high() -> Field:
    return Field('high')


def low() -> Field:
    return Field('low')


def field(name: str) -> Field:
    """Access any field by name"""
    return Field(name)


# ============================================================================
# Primitive Operations
# ============================================================================

class Returns(Signal):
    def __init__(self, period: int, price_field: str = 'close'):
        self.period = period
        self.price_field = price_field

    def _compute(self, data):
        prices = data[self.price_field]
        return prices.pct_change(self.period)

    def _cache_key(self):
        return ('Returns', self.period, self.price_field)


class Volatility(Signal):
    def __init__(self, period: int, price_field: str = 'close'):
        self.period = period
        self.price_field = price_field

    def _compute(self, data):
        prices = data[self.price_field]
        returns = prices.pct_change()
        return returns.rolling(self.period).std() * np.sqrt(252)

    def _cache_key(self):
        return ('Volatility', self.period, self.price_field)


class Volume(Signal):
    def __init__(self, period: int):
        self.period = period

    def _compute(self, data):
        vol = data['volume']
        return vol.rolling(self.period).mean()

    def _cache_key(self):
        return ('Volume', self.period)


def returns(period: int) -> Returns:
    return Returns(period)


def volatility(period: int) -> Volatility:
    return Volatility(period)


def volume(period: int) -> Volume:
    return Volume(period)


# ============================================================================
# Time-Series Operations
# ============================================================================

class TsMean(Signal):
    """Rolling mean of any signal"""
    def __init__(self, signal: Signal, period: int):
        self.signal = signal
        self.period = period

    def _compute(self, data):
        values = self.signal.evaluate(data)
        return values.rolling(self.period).mean()

    def _cache_key(self):
        return ('TsMean', self.signal._cache_key(), self.period)


class TsStd(Signal):
    """Rolling std dev of any signal"""
    def __init__(self, signal: Signal, period: int):
        self.signal = signal
        self.period = period

    def _compute(self, data):
        values = self.signal.evaluate(data)
        return values.rolling(self.period).std()

    def _cache_key(self):
        return ('TsStd', self.signal._cache_key(), self.period)


class TsSum(Signal):
    """Rolling sum of any signal"""
    def __init__(self, signal: Signal, period: int):
        self.signal = signal
        self.period = period

    def _compute(self, data):
        values = self.signal.evaluate(data)
        return values.rolling(self.period).sum()

    def _cache_key(self):
        return ('TsSum', self.signal._cache_key(), self.period)


class TsMax(Signal):
    """Rolling max of any signal"""
    def __init__(self, signal: Signal, period: int):
        self.signal = signal
        self.period = period

    def _compute(self, data):
        values = self.signal.evaluate(data)
        return values.rolling(self.period).max()

    def _cache_key(self):
        return ('TsMax', self.signal._cache_key(), self.period)


class TsMin(Signal):
    """Rolling min of any signal"""
    def __init__(self, signal: Signal, period: int):
        self.signal = signal
        self.period = period

    def _compute(self, data):
        values = self.signal.evaluate(data)
        return values.rolling(self.period).min()

    def _cache_key(self):
        return ('TsMin', self.signal._cache_key(), self.period)


class Delay(Signal):
    """Lag/shift any signal"""
    def __init__(self, signal: Signal, period: int):
        self.signal = signal
        self.period = period

    def _compute(self, data):
        values = self.signal.evaluate(data)
        return values.shift(self.period)

    def _cache_key(self):
        return ('Delay', self.signal._cache_key(), self.period)


class Delta(Signal):
    """Difference from N periods ago"""
    def __init__(self, signal: Signal, period: int):
        self.signal = signal
        self.period = period

    def _compute(self, data):
        values = self.signal.evaluate(data)
        return values - values.shift(self.period)

    def _cache_key(self):
        return ('Delta', self.signal._cache_key(), self.period)


class TsRank(Signal):
    """Time-series rank (percentile within rolling window)"""
    def __init__(self, signal: Signal, period: int):
        self.signal = signal
        self.period = period

    def _compute(self, data):
        values = self.signal.evaluate(data)
        return values.rolling(self.period).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan
        )

    def _cache_key(self):
        return ('TsRank', self.signal._cache_key(), self.period)


def ts_mean(signal: Signal, period: int) -> TsMean:
    return TsMean(signal, period)


def ts_std(signal: Signal, period: int) -> TsStd:
    return TsStd(signal, period)


def ts_sum(signal: Signal, period: int) -> TsSum:
    return TsSum(signal, period)


def ts_max(signal: Signal, period: int) -> TsMax:
    return TsMax(signal, period)


def ts_min(signal: Signal, period: int) -> TsMin:
    return TsMin(signal, period)


def delay(signal: Signal, period: int) -> Delay:
    return Delay(signal, period)


def delta(signal: Signal, period: int) -> Delta:
    return Delta(signal, period)


def ts_rank(signal: Signal, period: int) -> TsRank:
    return TsRank(signal, period)


# ============================================================================
# Cross-Sectional Operations
# ============================================================================

class Rank(Signal):
    def __init__(self, signal: Signal):
        self.signal = signal

    def _compute(self, data):
        values = self.signal.evaluate(data)
        return values.rank(axis=1, pct=True)

    def _cache_key(self):
        return ('Rank', self.signal._cache_key())


class ZScore(Signal):
    def __init__(self, signal: Signal):
        self.signal = signal

    def _compute(self, data):
        values = self.signal.evaluate(data)
        mean = values.mean(axis=1)
        std = values.std(axis=1)
        return values.sub(mean, axis=0).div(std, axis=0)

    def _cache_key(self):
        return ('ZScore', self.signal._cache_key())


class Demean(Signal):
    """Subtract cross-sectional mean"""
    def __init__(self, signal: Signal):
        self.signal = signal

    def _compute(self, data):
        values = self.signal.evaluate(data)
        mean = values.mean(axis=1)
        return values.sub(mean, axis=0)

    def _cache_key(self):
        return ('Demean', self.signal._cache_key())


def rank(signal: Signal) -> Rank:
    return Rank(signal)


def zscore(signal: Signal) -> ZScore:
    return ZScore(signal)


def demean(signal: Signal) -> Demean:
    return Demean(signal)


# ============================================================================
# Conditional Operations
# ============================================================================

class Where(Signal):
    """Ternary operator: where(condition, if_true, if_false)"""
    def __init__(self, condition: Signal, if_true: Signal, if_false: Signal):
        self.condition = condition
        self.if_true = if_true
        self.if_false = if_false

    def _compute(self, data):
        cond = self.condition.evaluate(data).astype(bool)
        true_vals = self.if_true.evaluate(data)
        false_vals = self.if_false.evaluate(data)
        return pd.DataFrame(
            np.where(cond, true_vals, false_vals),
            index=true_vals.index,
            columns=true_vals.columns
        )

    def _cache_key(self):
        return ('Where',
                self.condition._cache_key(),
                self.if_true._cache_key(),
                self.if_false._cache_key())


def where(condition: Signal, if_true: Signal, if_false: Signal) -> Where:
    return Where(condition, if_true, if_false)


# ============================================================================
# Group Operations
# ============================================================================

class GroupRank(Signal):
    """Rank within groups (e.g., industry-neutral rank)"""
    def __init__(self, signal: Signal, groups: str):
        self.signal = signal
        self.groups = groups

    def _compute(self, data):
        values = self.signal.evaluate(data)

        if 'groups' not in data or self.groups not in data['groups'].columns:
            raise ValueError(f"Group column '{self.groups}' not found in data['groups']")

        group_df = data['groups'][self.groups]

        result = pd.DataFrame(
            np.nan,
            index=values.index,
            columns=values.columns
        )

        for date in values.index:
            date_values = values.loc[date]
            date_groups = group_df.loc[date] if date in group_df.index else group_df.iloc[-1]

            for group_name in date_groups.unique():
                if pd.isna(group_name):
                    continue
                mask = date_groups == group_name
                group_values = date_values[mask]
                ranked = group_values.rank(pct=True)
                result.loc[date, mask] = ranked

        return result

    def _cache_key(self):
        return ('GroupRank', self.signal._cache_key(), self.groups)


class GroupDemean(Signal):
    """Demean within groups (subtract group mean)"""
    def __init__(self, signal: Signal, groups: str):
        self.signal = signal
        self.groups = groups

    def _compute(self, data):
        values = self.signal.evaluate(data)

        if 'groups' not in data or self.groups not in data['groups'].columns:
            raise ValueError(f"Group column '{self.groups}' not found in data['groups']")

        group_df = data['groups'][self.groups]

        result = pd.DataFrame(
            np.nan,
            index=values.index,
            columns=values.columns
        )

        for date in values.index:
            date_values = values.loc[date]
            date_groups = group_df.loc[date] if date in group_df.index else group_df.iloc[-1]

            for group_name in date_groups.unique():
                if pd.isna(group_name):
                    continue
                mask = date_groups == group_name
                group_values = date_values[mask]
                demeaned = group_values - group_values.mean()
                result.loc[date, mask] = demeaned

        return result

    def _cache_key(self):
        return ('GroupDemean', self.signal._cache_key(), self.groups)


class GroupNeutralize(Signal):
    """Make signal neutral to groups (zero weight per group)"""
    def __init__(self, signal: Signal, groups: str):
        self.signal = signal
        self.groups = groups

    def _compute(self, data):
        values = self.signal.evaluate(data)

        if 'groups' not in data or self.groups not in data['groups'].columns:
            raise ValueError(f"Group column '{self.groups}' not found in data['groups']")

        group_df = data['groups'][self.groups]
        result = values.copy()

        for date in values.index:
            date_values = values.loc[date]
            date_groups = group_df.loc[date] if date in group_df.index else group_df.iloc[-1]

            for group_name in date_groups.unique():
                if pd.isna(group_name):
                    continue
                mask = date_groups == group_name
                group_values = date_values[mask]
                result.loc[date, mask] = group_values - group_values.mean()

        return result

    def _cache_key(self):
        return ('GroupNeutralize', self.signal._cache_key(), self.groups)


def group_rank(signal: Signal, groups: str) -> GroupRank:
    return GroupRank(signal, groups)


def group_demean(signal: Signal, groups: str) -> GroupDemean:
    return GroupDemean(signal, groups)


def group_neutralize(signal: Signal, groups: str) -> GroupNeutralize:
    return GroupNeutralize(signal, groups)


# ============================================================================
# Parser
# ============================================================================

class AlphaParser:
    """Parse string expressions into Signal objects"""

    def __init__(self):
        self.functions = {
            # Data access
            'close': close,
            'open': open,
            'high': high,
            'low': low,
            'field': field,

            # Primitives
            'returns': returns,
            'volatility': volatility,
            'volume': volume,

            # Time-series ops
            'ts_mean': ts_mean,
            'ts_std': ts_std,
            'ts_sum': ts_sum,
            'ts_max': ts_max,
            'ts_min': ts_min,
            'delay': delay,
            'delta': delta,
            'ts_rank': ts_rank,

            # Cross-sectional ops
            'rank': rank,
            'zscore': zscore,
            'demean': demean,

            # Conditional
            'where': where,

            # Group ops
            'group_rank': group_rank,
            'group_demean': group_demean,
            'group_neutralize': group_neutralize,
        }

        self.binops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
            ast.Gt: operator.gt,
            ast.Lt: operator.lt,
            ast.GtE: operator.ge,
            ast.LtE: operator.le,
            ast.Eq: operator.eq,
            ast.NotEq: operator.ne,
            ast.BitAnd: operator.and_,
            ast.BitOr: operator.or_,
        }

        self.unaryops = {
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
            ast.Invert: operator.invert,
        }

    def parse(self, expression: str) -> Signal:
        """Parse string expression into Signal tree"""
        tree = ast.parse(expression, mode='eval')
        return self._visit(tree.body)

    def _visit(self, node: ast.AST) -> Signal:
        """Recursively visit AST nodes and build Signal tree"""

        if isinstance(node, ast.BinOp):
            left = self._visit(node.left)
            right = self._visit(node.right)
            op = self.binops[type(node.op)]
            return op(left, right)

        elif isinstance(node, ast.UnaryOp):
            operand = self._visit(node.operand)
            op = self.unaryops[type(node.op)]
            return op(operand)

        elif isinstance(node, ast.Compare):
            # Handle comparison operators (e.g., a < b, a > b)
            left = self._visit(node.left)
            # Only support single comparisons for now
            if len(node.ops) != 1:
                raise ValueError("Chained comparisons not supported")
            op = self.binops[type(node.ops[0])]
            right = self._visit(node.comparators[0])
            return op(left, right)

        elif isinstance(node, ast.Call):
            func_name = node.func.id
            if func_name not in self.functions:
                raise ValueError(f"Unknown function: {func_name}")

            args = [self._visit_arg(arg) for arg in node.args]

            return self.functions[func_name](*args)

        elif isinstance(node, (ast.Constant, ast.Num)):
            value = node.value if isinstance(node, ast.Constant) else node.n
            return Constant(value)

        elif isinstance(node, ast.Str):
            return node.s

        else:
            raise ValueError(f"Unsupported node type: {type(node)}")

    def _visit_arg(self, node: ast.AST) -> Any:
        """Visit argument node - could be Signal or literal"""
        if isinstance(node, ast.Call):
            return self._visit(node)
        elif isinstance(node, (ast.Constant, ast.Num)):
            return node.value if isinstance(node, ast.Constant) else node.n
        elif isinstance(node, ast.Str):
            return node.s
        else:
            return self._visit(node)


def alpha(expression: str) -> Signal:
    """Parse and return signal from string expression"""
    parser = AlphaParser()
    return parser.parse(expression)


def compute_weights(expression: str,
                    data: Dict[str, pd.DataFrame],
                    **kwargs) -> pd.DataFrame:
    """
    Parse expression and return portfolio weights.

    This is the main entry point for backtesting integration.
    """
    signal = alpha(expression)
    return signal.to_weights(data, **kwargs)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Generate synthetic data
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'BAC', 'GS']

    np.random.seed(42)
    prices = pd.DataFrame(
        np.exp(np.random.randn(len(dates), len(tickers)).cumsum(axis=0) * 0.01 + 5),
        index=dates,
        columns=tickers
    )
    volumes = pd.DataFrame(
        np.random.lognormal(15, 1, (len(dates), len(tickers))),
        index=dates,
        columns=tickers
    )

    # Add custom fields
    market_caps = pd.DataFrame(
        np.random.lognormal(10, 0.5, (len(dates), len(tickers))),
        index=dates,
        columns=tickers
    )

    # Create sector groups
    sector_data = {}
    for ticker in tickers:
        sector_data[ticker] = 'Tech' if ticker in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'] else 'Finance'

    groups_df = pd.DataFrame(index=dates, columns=tickers)
    for ticker in tickers:
        groups_df[ticker] = sector_data[ticker]

    # Simple container for group data that mimics DataFrame column access
    class GroupsContainer:
        def __init__(self, **kwargs):
            self._groups = kwargs
            self.columns = list(kwargs.keys())

        def __getitem__(self, key):
            return self._groups[key]

        def __contains__(self, key):
            return key in self._groups

    data = {
        'close': prices,
        'volume': volumes,
        'market_cap': market_caps,
        'groups': GroupsContainer(sector=groups_df)
    }

    print("=" * 80)
    print("Alpha DSL Examples")
    print("=" * 80)

    # Example 1: Simple reversal
    print("\n1. Simple reversal normalized by volatility:")
    print("   Expression: -returns(20) / volatility(60)")
    with compute_context() as ctx:
        signal1 = alpha("-returns(20) / volatility(60)")
        result1 = signal1.evaluate(data)
        print(f"   Result shape: {result1.shape}")
        print(f"   Cache size: {len(ctx.cache)}")
        print(result1.tail(3))

    # Example 2: Momentum with filter
    print("\n2. Momentum filtered by low volatility:")
    print("   Expression: rank(returns(252)) * (volatility(60) < 0.3)")
    signal2 = alpha("rank(returns(252)) * (volatility(60) < 0.3)")
    result2 = signal2.evaluate(data)
    print(f"   Result shape: {result2.shape}")
    print(result2.tail(3))

    # Example 3: Conditional logic
    print("\n3. Conditional momentum:")
    print("   Expression: where(returns(5) > 0, rank(returns(20)), -rank(returns(20)))")
    signal3 = alpha("where(returns(5) > 0, rank(returns(20)), -rank(returns(20)))")
    result3 = signal3.evaluate(data)
    print(f"   Result shape: {result3.shape}")
    print(result3.tail(3))

    # Example 4: Group operations
    print("\n4. Sector-neutral momentum:")
    print("   Expression: group_rank(returns(20), 'sector')")
    signal4 = alpha("group_rank(returns(20), 'sector')")
    result4 = signal4.evaluate(data)
    print(f"   Result shape: {result4.shape}")
    print(result4.tail(3))

    # Example 5: Multiple alphas with shared cache
    print("\n5. Multiple alphas with shared computation:")
    alphas = {
        'Reversal/Vol': "-returns(20) / volatility(60)",
        'Momentum': "rank(returns(252))",
        'Volume': "rank(volume(20))",
        'Combined': "rank(-returns(20)/volatility(60)) + rank(returns(252))"
    }

    with compute_context() as ctx:
        results = {}
        for name, expr in alphas.items():
            sig = alpha(expr)
            results[name] = sig.evaluate(data)
            print(f"   {name}: cache size = {len(ctx.cache)}")

    # Example 6: Convert to weights for backtesting
    print("\n6. Convert signal to portfolio weights:")
    print("   Expression: rank(-returns(20)/volatility(60))")
    signal6 = alpha("rank(-returns(20)/volatility(60))")
    weights = signal6.to_weights(data, normalize=True, long_only=False)
    print(f"   Weights shape: {weights.shape}")
    print(f"   Weights sum per day (should be ~0 for long-short):")
    print(weights.sum(axis=1).tail(3))

    # Example 7: Custom field access
    print("\n7. Custom field (market cap momentum):")
    print("   Expression: delta(field('market_cap'), 20) / ts_std(field('market_cap'), 60)")
    signal7 = alpha("delta(field('market_cap'), 20) / ts_std(field('market_cap'), 60)")
    result7 = signal7.evaluate(data)
    print(f"   Result shape: {result7.shape}")
    print(result7.tail(3))

    print("\n" + "=" * 80)
    print("DSL is ready! Use alpha('your_expression') to parse signals.")
    print("=" * 80)
