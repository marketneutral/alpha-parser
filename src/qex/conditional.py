"""Conditional operations for signals."""

import numpy as np
import pandas as pd

from .signal import Signal
from .operators import Constant, _ensure_signal


class Where(Signal):
    """Ternary operator: where(condition, if_true, if_false). A nonzero condition takes if_true, a zero one if_false,
    and a missing condition gives a missing value: not knowing the condition is not knowing the answer."""

    def __init__(self, condition: Signal, if_true: Signal, if_false: Signal):
        self.condition = _ensure_signal(condition)
        self.if_true = _ensure_signal(if_true)
        self.if_false = _ensure_signal(if_false)

    def _compute(self, data):
        cond = self.condition.evaluate(data)
        true_vals = self.if_true.evaluate(data)
        false_vals = self.if_false.evaluate(data)

        # Get reference index/columns from condition (always a DataFrame)
        index = cond.index
        columns = cond.columns

        # Handle scalar values (from Constant signals)
        if not isinstance(true_vals, pd.DataFrame):
            true_vals = pd.DataFrame(true_vals, index=index, columns=columns)
        if not isinstance(false_vals, pd.DataFrame):
            false_vals = pd.DataFrame(false_vals, index=index, columns=columns)

        c = cond.to_numpy(dtype=float)
        chosen = np.where(c != 0, true_vals.reindex_like(cond).to_numpy(), false_vals.reindex_like(cond).to_numpy())
        return pd.DataFrame(np.where(np.isnan(c), np.nan, chosen), index=index, columns=columns)

    def _cache_key(self):
        return ('Where',
                self.condition._cache_key(),
                self.if_true._cache_key(),
                self.if_false._cache_key())


def where(condition: Signal, if_true: Signal, if_false: Signal) -> Where:
    """Create a Where signal."""
    return Where(condition, if_true, if_false)


class Coalesce(Signal):
    """x where it has a value, y where it does not: coalesce(close() > 8, 0) is the comparison where it is known and
    0 where it is not. y is a number or another signal. It says in the expression what a missing value should mean."""

    def __init__(self, x: Signal, y: Signal):
        self.x = _ensure_signal(x)
        self.y = _ensure_signal(y)

    def _compute(self, data):
        x, y = self.x.evaluate(data), self.y.evaluate(data)
        if not isinstance(x, (pd.DataFrame, pd.Series)):
            return y if x is None or (isinstance(x, float) and np.isnan(x)) else x
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.reindex_like(x)
        return x.where(x.notna(), y)

    def _cache_key(self):
        return ('Coalesce', self.x._cache_key(), self.y._cache_key())


def coalesce(x: Signal, y: Signal) -> Coalesce:
    """x where it has a value, else y (a number or a signal)."""
    return Coalesce(x, y)
