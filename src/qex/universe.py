"""The cross-sectional universe: which names cross-sectional operators compare on each day.

Time-series operators always see a name's whole history, including days before it entered
the universe, so a new member has its past returns, volatility and beta. Cross-sectional
operators (rank, zscore, demean, quantile, winsorize, scale, regression_neut, the group
operators, and the cross-sectional prior of ts_beta_shrunk) are different: their result for
a name depends on which other names they are compared with. Inside cross_section_universe(mask)
they compare only the names in the mask that day, and a name outside it gets NaN.

A consequence: a time-series operator applied to a cross-sectional result, such as
ts_mean(rank(x), 20), has no history for a name from before the day it joined.

Example:
    >>> with cross_section_universe(data['member'].notna()):
    ...     r = qex("rank(returns(5))").evaluate(data)     # ranked among members only

build_book does this itself with BookSettings(universe=..., mask_cross_section=True).
"""

import itertools
from contextlib import contextmanager
from typing import Optional

import pandas as pd

_mask: Optional[pd.DataFrame] = None
_token: Optional[int] = None
_tokens = itertools.count()


@contextmanager
def cross_section_universe(mask: Optional[pd.DataFrame]):
    """Restrict cross-sectional operators to the names where mask is True (or finite), day by day.

    mask is a date x ticker DataFrame of booleans, or a universe field (1 in, NaN out). Dates or
    tickers it does not cover are outside the universe. None leaves operators unrestricted.
    """
    global _mask, _token
    old = _mask, _token
    if mask is not None and mask.dtypes.ne(bool).any():
        mask = mask.notna() & mask.ne(0)
    _mask, _token = mask, (None if mask is None else next(_tokens))
    try:
        yield
    finally:
        _mask, _token = old


def in_universe(values: pd.DataFrame) -> pd.DataFrame:
    """values with every name outside the current cross-sectional universe set to NaN."""
    if _mask is None:
        return values
    return values.where(_mask.reindex(index=values.index, columns=values.columns, fill_value=False))


def universe_token() -> Optional[int]:
    """An id for the active universe (None when unrestricted), so cached results never cross universes."""
    return _token
