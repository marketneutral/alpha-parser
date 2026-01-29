"""Arithmetic, comparison, and logical operators for signals."""

from .signal import Signal


def _ensure_signal(x):
    """Convert constants to Constant signals."""
    if isinstance(x, Signal):
        return x
    return Constant(x)


class Constant(Signal):
    """Constant value signal."""

    def __init__(self, value):
        self.value = value

    def _compute(self, data):
        return self.value

    def _cache_key(self):
        return ('Constant', self.value)


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
        values = self.signal.evaluate(data)
        return values.abs()

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
        return values ** self.exponent

    def _cache_key(self):
        return ('Power', self.signal._cache_key(), self.exponent)


class Max(BinaryOp):
    """Element-wise maximum of two signals."""

    def _compute(self, data):
        import numpy as np
        import pandas as pd
        left = self.left.evaluate(data)
        right = self.right.evaluate(data)
        return pd.DataFrame(
            np.maximum(left, right),
            index=left.index,
            columns=left.columns
        )


class Min(BinaryOp):
    """Element-wise minimum of two signals."""

    def _compute(self, data):
        import numpy as np
        import pandas as pd
        left = self.left.evaluate(data)
        right = self.right.evaluate(data)
        return pd.DataFrame(
            np.minimum(left, right),
            index=left.index,
            columns=left.columns
        )


# Factory functions for math operations

def log(signal: Signal) -> Log:
    """Create a Log signal (natural logarithm)."""
    return Log(signal)


def abs_(signal: Signal) -> Abs:
    """Create an Abs signal (absolute value)."""
    return Abs(signal)


def sign(signal: Signal) -> Sign:
    """Create a Sign signal."""
    return Sign(signal)


def sqrt(signal: Signal) -> Sqrt:
    """Create a Sqrt signal."""
    return Sqrt(signal)


def power(signal: Signal, exponent) -> Power:
    """Create a Power signal."""
    return Power(signal, exponent)


def max_(left: Signal, right: Signal) -> Max:
    """Create a Max signal (element-wise maximum)."""
    return Max(_ensure_signal(left), _ensure_signal(right))


def min_(left: Signal, right: Signal) -> Min:
    """Create a Min signal (element-wise minimum)."""
    return Min(_ensure_signal(left), _ensure_signal(right))
