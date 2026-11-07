"""Arithmetic, comparison, and logical operators for signals.

Comparisons and logic are three-valued: a comparison with a missing side is missing (not 0, which would be a
confident "no"), and & | ~ follow Kleene's rules, as pandas' NA does: false & missing is false, true | missing is
true, and every other combination with a missing side is missing.
"""

import operator as _op

import numpy as np
import pandas as pd

from .signal import Signal


def _missing(x):
    """Where x has no value: NaN, or None, cell by cell for a frame; one bool for a scalar."""
    if isinstance(x, (pd.DataFrame, pd.Series)):
        return x.isna()
    return x is None or (isinstance(x, float) and np.isnan(x))


def _compare(op, left, right):
    """op(left, right) as 1.0 or 0.0, and NaN where either side is missing."""
    if not isinstance(left, (pd.DataFrame, pd.Series)) and not isinstance(right, (pd.DataFrame, pd.Series)):
        return np.nan if _missing(left) or _missing(right) else float(op(left, right))
    result = op(left, right).astype(float)
    missing = _missing(left) | _missing(right)
    return result.where(~missing)


def _truth(x):
    """(known true, known false) for a value read as a condition: nonzero is true, 0 is false, missing is neither."""
    if isinstance(x, (pd.DataFrame, pd.Series)):
        known = x.notna()
        return (x != 0) & known, (x == 0) & known
    if _missing(x):
        return False, False
    return bool(x != 0), bool(x == 0)


def _logic(true, false):
    """1.0 where known true, 0.0 where known false, NaN where unknown."""
    if isinstance(true, (pd.DataFrame, pd.Series)) or isinstance(false, (pd.DataFrame, pd.Series)):
        frame = true if isinstance(true, (pd.DataFrame, pd.Series)) else false
        true = true if isinstance(true, (pd.DataFrame, pd.Series)) else pd.DataFrame(true, index=frame.index, columns=frame.columns) if isinstance(frame, pd.DataFrame) else pd.Series(true, index=frame.index)
        return true.astype(float).where(true | false)
    return 1.0 if true else 0.0 if false else np.nan


def _ensure_signal(x):
    """Convert constants to Constant signals, and None to the None literal."""
    if isinstance(x, Signal):
        return x
    if x is None:
        return NoneValue()
    return Constant(x)


class Constant(Signal):
    """Constant value signal."""

    def __init__(self, value):
        self.value = value

    def _compute(self, data):
        return self.value

    def _cache_key(self):
        return ('Constant', self.value)


class NoneValue(Signal):
    """The None literal: no value here, on purpose, as in where(cond, x, None).

    It is missing like any NaN (cross-sections and the book leave it out, and arithmetic with it
    stays missing), but its origin is known: explain_missing reports a cell that is missing
    because of it as excluded on purpose, not as a calculation that failed.
    """

    def _compute(self, data):
        return float('nan')

    def _cache_key(self):
        return ('None',)


class BinaryOp(Signal):
    """Base class for binary operations."""

    def __init__(self, left: Signal, right: Signal):
        self.left = left
        self.right = right

    def _cache_key(self):
        return (self.__class__.__name__, self.left._cache_key(), self.right._cache_key())


# Arithmetic Operations

class Add(BinaryOp):
    """Addition of two signals."""

    def _compute(self, data):
        return self.left.evaluate(data) + self.right.evaluate(data)


class Sub(BinaryOp):
    """Subtraction of two signals."""

    def _compute(self, data):
        return self.left.evaluate(data) - self.right.evaluate(data)


class Mul(BinaryOp):
    """Multiplication of two signals."""

    def _compute(self, data):
        return self.left.evaluate(data) * self.right.evaluate(data)


class Div(BinaryOp):
    """Division of two signals."""

    def _compute(self, data):
        return self.left.evaluate(data) / self.right.evaluate(data)


class Neg(Signal):
    """Negation of a signal."""

    def __init__(self, signal: Signal):
        self.signal = signal

    def _compute(self, data):
        return -self.signal.evaluate(data)

    def _cache_key(self):
        return ('Neg', self.signal._cache_key())


# Comparison Operations

class Greater(BinaryOp):
    """Greater than comparison."""

    def _compute(self, data):
        return _compare(_op.gt, self.left.evaluate(data), self.right.evaluate(data))


class Less(BinaryOp):
    """Less than comparison."""

    def _compute(self, data):
        return _compare(_op.lt, self.left.evaluate(data), self.right.evaluate(data))


class GreaterEqual(BinaryOp):
    """Greater than or equal comparison."""

    def _compute(self, data):
        return _compare(_op.ge, self.left.evaluate(data), self.right.evaluate(data))


class LessEqual(BinaryOp):
    """Less than or equal comparison."""

    def _compute(self, data):
        return _compare(_op.le, self.left.evaluate(data), self.right.evaluate(data))


class Equal(BinaryOp):
    """Equality comparison."""

    def _compute(self, data):
        return _compare(_op.eq, self.left.evaluate(data), self.right.evaluate(data))


class NotEqual(BinaryOp):
    """Not equal comparison."""

    def _compute(self, data):
        return _compare(_op.ne, self.left.evaluate(data), self.right.evaluate(data))


# Logical Operations

class And(BinaryOp):
    """Logical AND of two signals."""

    def _compute(self, data):
        (lt, lf), (rt, rf) = _truth(self.left.evaluate(data)), _truth(self.right.evaluate(data))
        return _logic(lt & rt, lf | rf)                     # false if either is false; true if both are true


class Or(BinaryOp):
    """Logical OR of two signals."""

    def _compute(self, data):
        (lt, lf), (rt, rf) = _truth(self.left.evaluate(data)), _truth(self.right.evaluate(data))
        return _logic(lt | rt, lf & rf)                     # true if either is true; false if both are false


class Not(Signal):
    """Logical NOT of a signal."""

    def __init__(self, signal: Signal):
        self.signal = signal

    def _compute(self, data):
        true, false = _truth(self.signal.evaluate(data))
        return _logic(false, true)                          # missing stays missing

    def _cache_key(self):
        return ('Not', self.signal._cache_key())


class IsValid(Signal):
    """Returns 1 where signal is not NaN, 0 otherwise."""

    def __init__(self, signal: Signal):
        self.signal = signal

    def _compute(self, data):
        values = self.signal.evaluate(data)
        return values.notna().astype(float)

    def _cache_key(self):
        return ('IsValid', self.signal._cache_key())


def is_valid(signal: Signal) -> IsValid:
    """Create an IsValid signal."""
    return IsValid(signal)


class IsFinite(Signal):
    """1 where the signal is a finite number, 0 where it is missing or infinite.

    is_valid counts an infinity as valid (it is a value); is_finite does not. Use it to keep the
    result of a division by zero out of a signal: where(is_finite(x), x, None).
    """

    def __init__(self, signal: Signal):
        self.signal = signal

    def _compute(self, data):
        import numpy as np
        import pandas as pd
        values = self.signal.evaluate(data)
        if isinstance(values, pd.DataFrame):
            return pd.DataFrame(np.isfinite(values.to_numpy(dtype=float)).astype(float),
                                index=values.index, columns=values.columns)
        return float(values is not None and np.isfinite(values))

    def _cache_key(self):
        return ('IsFinite', self.signal._cache_key())


def is_finite(signal) -> IsFinite:
    """Create an IsFinite signal: 1 where finite, 0 where missing or infinite."""
    return IsFinite(_ensure_signal(signal))


# Math Operations

class Log(Signal):
    """Natural logarithm of a signal."""

    def __init__(self, signal: Signal):
        self.signal = signal

    def _compute(self, data):
        import numpy as np
        values = self.signal.evaluate(data)
        return np.log(values)

    def _cache_key(self):
        return ('Log', self.signal._cache_key())


class Exp(Signal):
    """e raised to a signal: the inverse of log.

    exp(ts_sum(log(1 + returns(1)), n)) - 1 compounds daily returns, 1 / (1 + exp(-x)) is a sigmoid,
    and scale(exp(x / T)) is a cross-sectional softmax with temperature T. An argument too large to
    represent gives inf (no warning).
    """

    def __init__(self, signal: Signal):
        self.signal = signal

    def _compute(self, data):
        import numpy as np
        values = self.signal.evaluate(data)
        with np.errstate(over='ignore'):
            return np.exp(values)

    def _cache_key(self):
        return ('Exp', self.signal._cache_key())


class Abs(Signal):
    """Absolute value of a signal."""

    def __init__(self, signal: Signal):
        self.signal = signal

    def _compute(self, data):
        import numpy as np
        values = self.signal.evaluate(data)
        return np.abs(values)

    def _cache_key(self):
        return ('Abs', self.signal._cache_key())


class Sign(Signal):
    """Sign of a signal (-1, 0, or 1)."""

    def __init__(self, signal: Signal):
        self.signal = signal

    def _compute(self, data):
        import numpy as np
        values = self.signal.evaluate(data)
        return np.sign(values)

    def _cache_key(self):
        return ('Sign', self.signal._cache_key())


class Sqrt(Signal):
    """Square root of a signal."""

    def __init__(self, signal: Signal):
        self.signal = signal

    def _compute(self, data):
        import numpy as np
        values = self.signal.evaluate(data)
        return np.sqrt(values)

    def _cache_key(self):
        return ('Sqrt', self.signal._cache_key())


class Power(Signal):
    """Raise signal to a power."""

    def __init__(self, signal: Signal, exponent):
        self.signal = signal
        self.exponent = exponent

    def _compute(self, data):
        values = self.signal.evaluate(data)
        exponent = self.exponent.evaluate(data) if isinstance(self.exponent, Signal) else self.exponent
        return values ** exponent

    def _cache_key(self):
        exponent = self.exponent._cache_key() if isinstance(self.exponent, Signal) else self.exponent
        return ('Power', self.signal._cache_key(), exponent)


class Max(BinaryOp):
    """Element-wise maximum of two signals."""

    def _compute(self, data):
        import numpy as np
        import pandas as pd
        left = self.left.evaluate(data)
        right = self.right.evaluate(data)
        frame = left if isinstance(left, pd.DataFrame) else right
        if not isinstance(frame, pd.DataFrame):
            return np.maximum(left, right)                 # both constants
        return pd.DataFrame(np.maximum(left, right), index=frame.index, columns=frame.columns)


class Min(BinaryOp):
    """Element-wise minimum of two signals."""

    def _compute(self, data):
        import numpy as np
        import pandas as pd
        left = self.left.evaluate(data)
        right = self.right.evaluate(data)
        frame = left if isinstance(left, pd.DataFrame) else right
        if not isinstance(frame, pd.DataFrame):
            return np.minimum(left, right)                 # both constants
        return pd.DataFrame(np.minimum(left, right), index=frame.index, columns=frame.columns)


# Factory functions for math operations

def log(signal) -> Log:
    """Create a Log signal (natural logarithm)."""
    return Log(_ensure_signal(signal))


def exp(signal) -> Exp:
    """Create an Exp signal (e to the signal; the inverse of log)."""
    return Exp(_ensure_signal(signal))


def abs_(signal) -> Abs:
    """Create an Abs signal (absolute value)."""
    return Abs(_ensure_signal(signal))


def sign(signal) -> Sign:
    """Create a Sign signal."""
    return Sign(_ensure_signal(signal))


def sqrt(signal) -> Sqrt:
    """Create a Sqrt signal."""
    return Sqrt(_ensure_signal(signal))


def power(signal, exponent) -> Power:
    """Create a Power signal. Either argument may be a number or a signal."""
    return Power(_ensure_signal(signal), exponent)


def max_(left: Signal, right: Signal) -> Max:
    """Create a Max signal (element-wise maximum)."""
    return Max(_ensure_signal(left), _ensure_signal(right))


def min_(left: Signal, right: Signal) -> Min:
    """Create a Min signal (element-wise minimum)."""
    return Min(_ensure_signal(left), _ensure_signal(right))
