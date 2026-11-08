"""ts_match: each stock's statistical twin.

ts_match(value, key, lookback, metric='corr', within=None, matching='nearest', rebalance=21, min_overlap=None) finds,
for each stock, the stock whose history of `key` over the last `lookback` days is most similar, and returns that
twin's `value`: an ordinary dates x tickers matrix. Every other cross-sectional operator compares a stock with a crowd
(the market, its sector, all its peers at once); this one compares it with one particular stock.

Semantics:
- Point in time: the matches used on day t are chosen from `key` up to day t - 1.
- No stock is its own twin.
- Matches are chosen on rebalance days (every `rebalance` rows, the first once `lookback` rows exist) and held in
  between. A match ends early when either stock leaves the cross-sectional universe.
- A stock with no eligible twin has no value; so does one whose twin's value is missing (that does not pick another).
- `within` (a group's name or any labels expression): the twin must share the stock's group on the rebalance day.
- `matching='exclusive'`: reciprocal pairs that don't overlap, chosen greedily from the closest pair down.

The selection is an internal node, MatchSelection, cached on everything except `value`: every ts_match on the same
key and options shares it, so reading a twin's return and its volume costs one selection.
"""

from typing import Optional

import numpy as np
import pandas as pd

from .signal import Signal
from .universe import universe_members

METRICS = ("corr", "spearman", "cosine", "euclidean", "manhattan")
MATCHINGS = ("nearest", "exclusive")


def similarity(window: np.ndarray, metric: str, min_overlap: int) -> np.ndarray:
    """N x N similarities (higher is closer) between the columns of a T x N window, each pair over the days both have.

    With M the mask of valid days and X the window with gaps at zero, the pairwise counts, sums and cross-products are
    M'M, X'M, (X*X)'M and X'X, so every pair uses exactly its shared days. Distances are divided by the shared days, so
    pairs with gaps compare fairly. NaN where a pair shares fewer than min_overlap days, and on the diagonal.
    """
    if metric == "spearman":                                    # a correlation of ranks
        window = pd.DataFrame(window).rank().to_numpy()
        metric = "corr"
    valid = np.isfinite(window)
    x = np.where(valid, window, 0.0)
    m = valid.astype(float)
    n = m.T @ m
    sx = x.T @ m
    sxx = (x * x).T @ m
    sxy = x.T @ x
    with np.errstate(invalid="ignore", divide="ignore"):
        if metric == "corr":
            cov = n * sxy - sx * sx.T
            var = n * sxx - sx * sx
            s = cov / np.sqrt(var * var.T)
        elif metric == "cosine":
            s = sxy / np.sqrt(sxx * sxx.T)
        elif metric == "euclidean":                             # root mean squared distance
            s = -np.sqrt(np.maximum(sxx + sxx.T - 2 * sxy, 0) / n)
        elif metric == "manhattan":                             # mean absolute distance, one stock against all
            s = np.empty_like(n)
            for i in range(x.shape[1]):
                s[i] = -(np.abs(x[:, [i]] - x) * (m[:, [i]] * m)).sum(axis=0) / n[i]
        else:
            raise ValueError(f"metric must be one of {METRICS}")
    s[~np.isfinite(s)] = np.nan
    s[n < min_overlap] = np.nan
    np.fill_diagonal(s, np.nan)
    return s


def _nearest(s: np.ndarray) -> np.ndarray:
    """Each stock's most similar eligible stock, or -1; ties go to the lower column."""
    best = np.argmax(np.where(np.isnan(s), -np.inf, s), axis=1)
    return np.where(np.isfinite(s).any(axis=1), best, -1)


def _exclusive(s: np.ndarray) -> np.ndarray:
    """Reciprocal pairs, greedily: the closest pair first, then the closest of those still free."""
    i, j = np.triu_indices_from(s, 1)
    keep = np.isfinite(s[i, j])
    i, j = i[keep], j[keep]
    peer = np.full(len(s), -1)
    for k in np.lexsort((j, i, -s[i, j])):                      # by similarity, ties by position
        if peer[i[k]] < 0 and peer[j[k]] < 0:
            peer[i[k]], peer[j[k]] = j[k], i[k]
    return peer


class MatchResult:
    """A MatchSelection's value: each stock's twin's column on each date (-1: none), and the dates and names."""

    __slots__ = ("peers", "index", "columns")

    def __init__(self, peers: np.ndarray, index, columns):
        self.peers, self.index, self.columns = peers, index, columns


class MatchSelection(Signal):
    """The twins behind ts_match: an internal node, shared by every ts_match on the same key and options."""

    value_is_frame = False

    def __init__(self, key: Signal, lookback: int, metric: str, within, matching: str, rebalance: int,
                 min_overlap: int):
        self.key = key
        self.lookback = lookback
        self.metric = metric
        self.groups = within                                    # named groups, so kinds and the DAG treat it as one
        self.matching = matching
        self.rebalance = rebalance
        self.min_overlap = min_overlap

    def _compute(self, data):
        key = self.key.evaluate(data)
        if not isinstance(key, pd.DataFrame):
            raise ValueError("ts_match's key must give each stock a value on each day")
        values = key.to_numpy(dtype=float)
        values = np.where(np.isfinite(values), values, np.nan)
        T, N = values.shape
        members = universe_members(key.index, key.columns)
        eligible = np.ones((T, N), bool) if members is None else np.asarray(members, bool)
        if self.groups is None:
            codes = np.zeros((T, N), int)
        else:
            from .groups import _get_group_data, _group_codes
            codes, _ = _group_codes(key, _get_group_data(data, self.groups))
        choose = _nearest if self.matching == "nearest" else _exclusive
        peers = np.full((T, N), -1)
        current = np.full(N, -1)
        for t in range(self.lookback, T):
            if (t - self.lookback) % self.rebalance == 0:
                window = values[t - self.lookback:t]                  # up to t - 1: point in time
                current = np.full(N, -1)
                for g in np.unique(codes[t][eligible[t] & (codes[t] >= 0)]):
                    columns = np.flatnonzero(eligible[t] & (codes[t] == g))
                    if len(columns) < 2:
                        continue
                    chosen = choose(similarity(window[:, columns], self.metric, self.min_overlap))
                    found = chosen >= 0
                    current[columns[found]] = columns[chosen[found]]
            held = (current >= 0) & eligible[t] & eligible[t, np.maximum(current, 0)]     # leaving ends a match
            peers[t] = np.where(held, current, -1)
        return MatchResult(peers, key.index, key.columns)

    def _cache_key(self):
        from .groups import _groups_key
        return ('MatchSelection', self.key._cache_key(), self.lookback, self.metric,
                None if self.groups is None else _groups_key(self.groups), self.matching, self.rebalance,
                self.min_overlap)


class TsMatch(Signal):
    """Each stock's twin's value: `value` read from the stock chosen by its MatchSelection (see the module)."""

    def __init__(self, value: Signal, selection: MatchSelection):
        self.value = value
        self.selection = selection

    def _compute(self, data):
        sel = self.selection.evaluate(data)
        value = self.value.evaluate(data)
        if isinstance(value, pd.Series):
            value = pd.DataFrame({c: value for c in sel.columns}, index=value.index)
        elif not isinstance(value, pd.DataFrame):
            value = pd.DataFrame(float(value), index=sel.index, columns=sel.columns)
        v = value.reindex(index=sel.index, columns=sel.columns).to_numpy(dtype=float)
        rows = np.arange(len(sel.index))[:, None]
        read = np.where(sel.peers >= 0, v[rows, np.maximum(sel.peers, 0)], np.nan)
        return pd.DataFrame(read, index=sel.index, columns=sel.columns)

    def _cache_key(self):
        return ('TsMatch', self.value._cache_key(), self.selection._cache_key())


def _whole(name: str, what: str, v, low: int) -> int:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not float(v).is_integer() or v < low:
        raise ValueError(f"ts_match()'s {name} must be a whole number, at least {low} (got {v!r}): {what}")
    return int(v)


def ts_match(value: Signal, key: Signal, lookback: int, metric: str = 'corr', within=None,
             matching: str = 'nearest', rebalance: int = 21, min_overlap: Optional[int] = None) -> TsMatch:
    """Create a TsMatch signal: each stock's statistical twin's value.

    ts_match(value, key, lookback, metric='corr', within=None, matching='nearest', rebalance=21, min_overlap=None)
    """
    from .operators import NoneValue
    from .groups import _as_groups
    for name, s in (("value", value), ("key", key)):
        if not isinstance(s, Signal):
            raise ValueError(f"ts_match()'s {name} is an expression, such as returns(5), not {s!r}")
    lookback = _whole("lookback", "the days of key to compare", lookback, 2)
    if metric not in METRICS:
        raise ValueError(f"ts_match()'s metric is one of {', '.join(repr(m) for m in METRICS)}, not {metric!r}")
    if matching not in MATCHINGS:
        raise ValueError(f"ts_match()'s matching is 'nearest' (each stock's closest) or 'exclusive' (reciprocal "
                         f"pairs), not {matching!r}")
    rebalance = _whole("rebalance", "how many days the matches are held", rebalance, 1)
    if isinstance(min_overlap, NoneValue):
        min_overlap = None
    min_overlap = lookback if min_overlap is None else _whole(
        "min_overlap", "the fewest days a pair must share", min_overlap, 2)
    if min_overlap > lookback:
        raise ValueError(f"ts_match()'s min_overlap ({min_overlap}) can't be more than its lookback ({lookback})")
    if isinstance(within, NoneValue):
        within = None
    if within is not None:
        within = _as_groups(within)
    return TsMatch(value, MatchSelection(key, lookback, metric, within, matching, rebalance, min_overlap))
