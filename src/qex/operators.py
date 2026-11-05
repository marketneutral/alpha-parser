"""Arithmetic, comparison, and logical operators for signals."""

from .signal import Signal


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
        return (self.left.evaluate(data) > self.right.evaluate(data)).astype(float)


class Less(BinaryOp):
    """Less than comparison."""

    def _compute(self, data):
        return (self.left.evaluate(data) < self.right.evaluate(data)).astype(float)


class GreaterEqual(BinaryOp):
    """Greater than or equal comparison."""

    def _compute(self, data):
        return (self.left.evaluate(data) >= self.right.evaluate(data)).astype(float)


class LessEqual(BinaryOp):
    """Less than or equal comparison."""

    def _compute(self, data):
        return (self.left.evaluate(data) <= self.right.evaluate(data)).astype(float)


class Equal(BinaryOp):
    """Equality comparison."""

    def _compute(self, data):
        return (self.left.evaluate(data) == self.right.evaluate(data)).astype(float)


class NotEqual(BinaryOp):
    """Not equal comparison."""

    def _compute(self, data):
        return (self.left.evaluate(data) != self.right.evaluate(data)).astype(float)


# Logical Operations

class And(BinaryOp):
    """Logical AND of two signals."""

    def _compute(self, data):
        left = self.left.evaluate(data).astype(bool)
        right = self.right.evaluate(data).astype(bool)
        return (left & right).astype(float)


class Or(BinaryOp):
    """Logical OR of two signals."""

    def _compute(self, data):
        left = self.left.evaluate(data).astype(bool)
        right = self.right.evaluate(data).astype(bool)
        return (left | right).astype(float)


class Not(Signal):
    """Logical NOT of a signal."""

    def __init__(self, signal: Signal):
        self.signal = signal

    def _compute(self, data):
        return (~self.signal.evaluate(data).astype(bool)).astype(float)

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
