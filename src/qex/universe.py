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

A universe is a mask, the name of a data field (1 in, NaN out), or a universe expression: qex that gives each stock
a number each day, nonzero for a member. An expression is evaluated over every stock with no universe active (it
defines the cross-section, so it cannot be computed inside one), must give numbers, and is computed once per
compute_context:

    >>> STANDARD = "latch((close() >= 2) & (field('mcap') >= 300e6), field('mcap') < 240e6)"
    >>> with cross_section_universe(STANDARD, data):
    ...     r = qex("rank(returns(5))").evaluate(data)
"""

import itertools
from contextlib import contextmanager
from typing import Optional

import numpy as np
import pandas as pd

_mask: Optional[pd.DataFrame] = None
_token: Optional[int] = None
_tokens = itertools.count()


def universe_mask(universe, data=None) -> Optional[pd.DataFrame]:
    """The members, date x ticker True/False, of a universe given as None, a mask, a data field's name, or a universe
    expression (qex text or a Signal) evaluated on data. Text that names a field of data is that field."""
    if universe is None:
        return None
    if isinstance(universe, pd.DataFrame):
        return universe if not universe.dtypes.ne(bool).any() else (universe.notna() & universe.ne(0))
    from .signal import Signal
    if isinstance(universe, str):
        if data is not None and universe in data:
            return universe_mask(_field_frame(data, universe))
        from .parser import qex
        universe = qex(universe)
    if not isinstance(universe, Signal):
        raise TypeError(f"a universe is a mask, a field's name or a qex expression, not {type(universe).__name__}")
    if data is None:
        raise TypeError("a universe expression needs the data to evaluate it on: cross_section_universe(expr, data)")
    from .kinds import kind, LABELS
    if kind(universe) == LABELS:
        raise ValueError("a universe gives numbers (nonzero for a member), not labels: compare the labels "
                         "(group('category') == 'Domestic Common Stock')")
    from .context import get_context, compute_context
    ctx = get_context()
    key = ("universe mask", universe._cache_key())
    if ctx is not None and key in ctx.cache:
        ctx.check(data)
        if key in ctx.cache:
            return ctx.cache[key]
    global _mask, _token
    old = _mask, _token
    _mask, _token = None, None                            # a universe is computed over every stock
    try:
        if ctx is None:
            with compute_context():
                values = universe.evaluate(data)
        else:
            values = universe.evaluate(data)
    finally:
        _mask, _token = old
    if not isinstance(values, pd.DataFrame):
        raise ValueError("a universe gives each stock a value on each day, not one value for all of them")
    v = values.to_numpy(dtype=float)
    mask = pd.DataFrame(np.isfinite(v) & (v != 0), index=values.index, columns=values.columns)
    if ctx is not None:
        ctx.cache[key] = mask
    return mask


def _field_frame(data, name):
    from .data import resolve_data
    return resolve_data(data)[name]


@contextmanager
def cross_section_universe(mask, data=None):
    """Restrict cross-sectional operators to the members of a universe, day by day.

    mask is a date x ticker DataFrame of booleans, a universe field (1 in, NaN out), or, with data, a field's name or a
    universe expression (see the module docstring). Dates or tickers it does not cover are outside the universe. None
    leaves operators unrestricted.
    """
    global _mask, _token
    mask = universe_mask(mask, data)
    old = _mask, _token
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


def finite_in_universe(values: pd.DataFrame) -> pd.DataFrame:
    """in_universe(values) with infinities also set to NaN.

    For the operators that average, scale or regress across a cross-section (zscore, demean,
    winsorize, scale, regression_neut, group_demean, group_std, group_sum): one infinite value would otherwise make the day's mean or sum infinite and
    empty the whole cross-section. The infinite name itself gets NaN. Rank-based operators
    (rank, quantile, group_rank) don't need this: an infinity simply ranks first or last.
    """
    values = in_universe(values)
    return values.where(~values.isin([np.inf, -np.inf]))


def universe_token() -> Optional[int]:
    """An id for the active universe (None when unrestricted), so cached results never cross universes."""
    return _token
