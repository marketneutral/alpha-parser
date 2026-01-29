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
