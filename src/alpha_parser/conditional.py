"""Conditional operations for signals."""

import numpy as np
import pandas as pd

from .signal import Signal
from .operators import Constant, _ensure_signal


class Where(Signal):
    """Ternary operator: where(condition, if_true, if_false)."""

    def __init__(self, condition: Signal, if_true: Signal, if_false: Signal):
        self.condition = _ensure_signal(condition)
        self.if_true = _ensure_signal(if_true)
        self.if_false = _ensure_signal(if_false)

    def _compute(self, data):
        cond = self.condition.evaluate(data).astype(bool)
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

        return pd.DataFrame(
            np.where(cond, true_vals, false_vals),
            index=index,
            columns=columns
        )

    def _cache_key(self):
        return ('Where',
                self.condition._cache_key(),
                self.if_true._cache_key(),
                self.if_false._cache_key())


def where(condition: Signal, if_true: Signal, if_false: Signal) -> Where:
    """Create a Where signal."""
    return Where(condition, if_true, if_false)
