"""When an operator with a fit (ts_match's twins, pca_factor's components) re-fits: on a calendar, not by counting rows.

Counting rows from the first row of the data made the schedule depend on where the data began: the same expression,
evaluated on data loaded from 2020-01-01 and from 2020-01-20, chose its twins on different days, and on 2021-12-31
only 23% of stocks had the same twin in both. Nothing looked ahead, but one expression was two books, and a backtest's
start date changed every number after it.

So the refit days are anchored to a fixed calendar: the weekdays counted from Monday 1970-01-05, cut into blocks of
`every` weekdays, and a refit on the first row of each block (a holiday at a block's start moves it to the next
trading day). The same day has the same refit, and so the same fit, whatever row the data begins on; given `window`
rows of history before it, the fit itself is the same too. Weekdays, not rows: a block of 21 weekdays holds about 20
trading days in a year with holidays, and it is the price of a schedule nobody's data can move.

An index that is not dates (a toy frame indexed by integers) keeps the old rule, every `every` rows from the first fit.
"""
import numpy as np
import pandas as pd

EPOCH = np.datetime64("1970-01-05", "D")          # a Monday: weekday blocks are counted from it


def refit_rows(index, first: int, every: int) -> np.ndarray:
    """The row positions, at or after `first` (the first row with enough history), on which a fit is made: the first
    row of each block of `every` weekdays counted from EPOCH. Increasing, and the same set of dates for any index that
    contains them, whatever its first row."""
    T = len(index)
    if every < 1:
        raise ValueError("a refit schedule needs every >= 1")
    if first >= T:
        return np.array([], dtype=int)
    if not isinstance(index, pd.DatetimeIndex):
        return np.arange(first, T, every)
    days = index.values.astype("datetime64[D]")
    block = np.busday_count(EPOCH, days) // every   # the weekday block each row falls in
    starts = np.ones(T, bool)
    starts[1:] = block[1:] != block[:-1]            # the first row of its block
    # row 0 of the data may be mid-block; it starts a block only if it is the block's first weekday
    starts[0] = np.busday_count(EPOCH, days[0]) % every == 0
    rows = np.flatnonzero(starts)
    return rows[rows >= first]
