"""Base Signal class for all signal expressions."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict

import pandas as pd

from .context import get_context

if TYPE_CHECKING:
    from .operators import (
        Add, Sub, Mul, Div, Neg,
        Greater, Less, GreaterEqual, LessEqual, Equal, NotEqual,
        And, Or, Not, Constant
    )


class Signal(ABC):
    """Base class for all signal expressions."""

    @abstractmethod
    def _compute(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Actual computation logic - implemented by subclasses."""
        pass

    @abstractmethod
    def _cache_key(self) -> tuple:
        """Return hashable key for this computation."""
        pass

    def evaluate(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Evaluate signal with optional global caching."""
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
        from .operators import Neg
        return Neg(self)

    def __add__(self, other):
        from .operators import Add, _ensure_signal
        return Add(self, _ensure_signal(other))

    def __sub__(self, other):
        from .operators import Sub, _ensure_signal
        return Sub(self, _ensure_signal(other))

    def __mul__(self, other):
        from .operators import Mul, _ensure_signal
        return Mul(self, _ensure_signal(other))

    def __truediv__(self, other):
        from .operators import Div, _ensure_signal
        return Div(self, _ensure_signal(other))

    def __radd__(self, other):
        from .operators import Add, _ensure_signal
        return Add(_ensure_signal(other), self)

    def __rsub__(self, other):
        from .operators import Sub, _ensure_signal
        return Sub(_ensure_signal(other), self)

    def __rmul__(self, other):
        from .operators import Mul, _ensure_signal
        return Mul(_ensure_signal(other), self)

    def __rtruediv__(self, other):
        from .operators import Div, _ensure_signal
        return Div(_ensure_signal(other), self)

    # Comparison operators
    def __gt__(self, other):
        from .operators import Greater, _ensure_signal
        return Greater(self, _ensure_signal(other))

    def __lt__(self, other):
        from .operators import Less, _ensure_signal
        return Less(self, _ensure_signal(other))

    def __ge__(self, other):
        from .operators import GreaterEqual, _ensure_signal
        return GreaterEqual(self, _ensure_signal(other))

    def __le__(self, other):
        from .operators import LessEqual, _ensure_signal
        return LessEqual(self, _ensure_signal(other))

    def __eq__(self, other):
        from .operators import Equal, _ensure_signal
        return Equal(self, _ensure_signal(other))

    def __ne__(self, other):
        from .operators import NotEqual, _ensure_signal
        return NotEqual(self, _ensure_signal(other))

    # Logical operators
    def __and__(self, other):
        from .operators import And, _ensure_signal
        return And(self, _ensure_signal(other))

    def __or__(self, other):
        from .operators import Or, _ensure_signal
        return Or(self, _ensure_signal(other))

    def __invert__(self):
        from .operators import Not
        return Not(self)
