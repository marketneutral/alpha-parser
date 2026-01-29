"""Conditional operations for signals."""

import numpy as np
import pandas as pd

from .signal import Signal


class Where(Signal):
    """Ternary operator: where(condition, if_true, if_false)."""

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
    """Create a Where signal."""
    return Where(condition, if_true, if_false)
