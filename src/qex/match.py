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

METRICS = ("corr", "euclidean", "path", "coint")
MATCHINGS = ("nearest", "exclusive")
DISTANCES = ("euclidean", "path")                               # scores in the metric's units: smaller is closer
COINT_STATS = ("t", "hedge", "half_life", "spread_vol")         # what ts_match_score can report for a 'coint' pair
RETURN_KEYED = ("path", "coint")                                 # metrics that compound the key: it is daily returns
_RETURN_SIZE = 0.25                                             # a daily return's typical size is about 0.01


def _check_daily_returns(name: str, metric: str, values: np.ndarray):
    """'path' and 'coint' compound their key as daily returns: refuse what can't be one. Below -1 is a loss of more
    than everything; a typical size (median absolute value) above 0.25 is a rank, a z-score, a price or a volume,
    not a day's return, whose typical size is about 0.01."""
    finite = values[np.isfinite(values)]
    if not finite.size:
        return
    usage = (f"its key is daily returns: returns(1), or residual daily returns such as "
             f"ts_resid(returns(1), returns(1, 'market'), 252)")
    if finite.min() < -1:
        raise ValueError(f"{name}(metric={metric!r}) compounds its key into price paths, so {usage}; the key has "
                         f"values below -1, a loss of more than everything, which no return can be")
    typical = float(np.median(np.abs(finite)))
    if typical > _RETURN_SIZE:
        raise ValueError(f"{name}(metric={metric!r}) compounds its key into price paths, so {usage}; the key's typical "
                         f"size is {typical:.3g}, where a day's return is about 0.01: it looks like a rank, a z-score, a "
                         f"price or a volume")


def _log_paths(window: np.ndarray):
    """Daily returns compounded into log price paths starting at 0 (a missing day is a flat day), and the mask."""
    valid = np.isfinite(window)
    return np.cumsum(np.log1p(np.where(valid, window, 0.0)), axis=0), valid


def _coint(window: np.ndarray):
    """Engle-Granger for every pair at once. The key is daily returns, compounded into log paths L. For stock i on
    stock j: the hedge ratio b = cov(L_i, L_j) / var(L_j), the spread e = L_i - b L_j (centered), and the
    Dickey-Fuller regression of its daily change on its previous level, de_t = g e_{t-1} + u_t. Every sum is a
    quadratic form in b of cross-products of the centered paths, their lags and their changes, so all pairs come from
    a few m x m matrices. Returns (t, b, g, spread_sd), each m x m, row i the regression of i on column j."""
    L, _ = _log_paths(window)
    L = L - L.mean(axis=0)
    lag, d = L[:-1], np.diff(L, axis=0)
    C = L.T @ L                                                   # sums of products of the paths
    Q = lag.T @ lag                                               # of the lagged paths
    M = d.T @ lag                                                 # change of one against the other's previous level
    D = d.T @ d                                                   # of the changes
    di = lambda X: np.diagonal(X)
    with np.errstate(invalid="ignore", divide="ignore"):
        b = C / di(C)[None, :]                                    # i regressed on j
        see = di(Q)[:, None] - 2 * b * Q + b ** 2 * di(Q)[None, :]           # sum of e_{t-1}^2
        sde = di(M)[:, None] - b * M - b * M.T + b ** 2 * di(M)[None, :]     # sum of de_t * e_{t-1}
        sdd = di(D)[:, None] - 2 * b * D + b ** 2 * di(D)[None, :]           # sum of de_t^2
        g = sde / see
        ssr = np.maximum(sdd - g * sde, 0)
        n = len(d)
        t = g / np.sqrt(ssr / (n - 1) / see)
        spread_sd = np.sqrt(np.maximum(di(C)[:, None] - 2 * b * C + b ** 2 * di(C)[None, :], 0) / len(L))
    return t, b, g, spread_sd


def similarity(window: np.ndarray, metric: str, min_overlap: int, stats: bool = False):
    """N x N similarities (higher is closer) between the columns of a T x N window, NaN where a pair shares fewer
    than min_overlap days and on the diagonal. With stats=True for 'coint', also {stat: N x N} for each pair (row i:
    stock i on stock j).

    'corr' and 'euclidean' compare each pair over the days both have: with M the mask of valid days and X the window
    with gaps at zero, the counts, sums and cross-products are M'M, X'M, (X*X)'M and X'X. 'path' (Gatev, Goetzmann and
    Rouwenhorst) compounds the key (daily returns) into price paths from 100 at the window's start and takes the root
    mean squared distance between them. 'coint' (Engle-Granger) scores a pair by the Dickey-Fuller t of its spread's
    reversion, the mean of the two directions, negated so that higher is closer; a pair whose hedge ratio is not
    positive (either way) is not a long-short pair and is not compared.
    """
    valid = np.isfinite(window)
    m = valid.astype(float)
    n = m.T @ m
    extra = None
    with np.errstate(invalid="ignore", divide="ignore"):
        if metric == "coint":
            t, b, g, sd = _coint(window)
            s = -(t + t.T) / 2
            s[(b <= 0) | (b.T <= 0)] = np.nan                     # a pair is long one, short the other: hedge > 0
            if stats:
                extra = {"t": (t + t.T) / 2, "hedge": b, "half_life": np.log(0.5) / np.log1p(g), "spread_vol": sd}
        else:
            if metric == "path":
                # a missing day is a flat day inside the path; distances over the days both have a return
                window = np.where(valid, 100 * np.cumprod(1 + np.where(valid, window, 0.0), axis=0), np.nan)
                metric = "euclidean"
            x = np.where(valid, window, 0.0)
            sx = x.T @ m
            sxx = (x * x).T @ m
            sxy = x.T @ x
            if metric == "corr":
                cov = n * sxy - sx * sx.T
                var = n * sxx - sx * sx
                s = cov / np.sqrt(var * var.T)
            elif metric == "euclidean":                         # root mean squared distance
                s = -np.sqrt(np.maximum(sxx + sxx.T - 2 * sxy, 0) / n)
            else:
                raise ValueError(f"metric must be one of {METRICS}")
    s[~np.isfinite(s)] = np.nan
    s[n < min_overlap] = np.nan
    np.fill_diagonal(s, np.nan)
    return (s, extra) if stats else s


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
    """A MatchSelection's value: each stock's twin's column on each date (-1: none), the similarity of the stock and
    its twin when chosen (higher is closer; NaN without a twin), and the dates and names."""

    __slots__ = ("peers", "scores", "stats", "index", "columns")

    def __init__(self, peers: np.ndarray, scores: np.ndarray, index, columns, stats=None):
        self.peers, self.scores, self.index, self.columns = peers, scores, index, columns
        self.stats = stats or {}                                  # 'coint': each chosen pair's t, hedge, half-life, width


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
        if self.metric in RETURN_KEYED:
            _check_daily_returns("ts_match", self.metric, values)
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
        scores = np.full((T, N), np.nan)
        current = np.full(N, -1)
        current_score = np.full(N, np.nan)
        coint = self.metric == "coint"
        stats = {k: np.full((T, N), np.nan) for k in COINT_STATS} if coint else {}
        current_stats = {k: np.full(N, np.nan) for k in stats}
        for t in range(self.lookback, T):
            if (t - self.lookback) % self.rebalance == 0:
                window = values[t - self.lookback:t]                  # up to t - 1: point in time
                current = np.full(N, -1)
                current_score = np.full(N, np.nan)
                current_stats = {k: np.full(N, np.nan) for k in stats}
                for g in np.unique(codes[t][eligible[t] & (codes[t] >= 0)]):
                    columns = np.flatnonzero(eligible[t] & (codes[t] == g))
                    if len(columns) < 2:
                        continue
                    s, extra = similarity(window[:, columns], self.metric, self.min_overlap, stats=True)
                    chosen = choose(s)
                    found = np.flatnonzero(chosen >= 0)
                    current[columns[found]] = columns[chosen[found]]
                    current_score[columns[found]] = s[found, chosen[found]]
                    for k in stats:
                        current_stats[k][columns[found]] = extra[k][found, chosen[found]]
            held = (current >= 0) & eligible[t] & eligible[t, np.maximum(current, 0)]     # leaving ends a match
            peers[t] = np.where(held, current, -1)
            scores[t] = np.where(held, current_score, np.nan)
            for k in stats:
                stats[k][t] = np.where(held, current_stats[k], np.nan)
        return MatchResult(peers, scores, key.index, key.columns, stats)

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
        value = value.reindex(index=sel.index, columns=sel.columns)
        labels = not all(pd.api.types.is_numeric_dtype(t) for t in value.dtypes)
        v = value.to_numpy(dtype=object if labels else float)              # labels gather as labels
        rows = np.arange(len(sel.index))[:, None]
        read = np.where(sel.peers >= 0, v[rows, np.maximum(sel.peers, 0)], np.nan)
        out = pd.DataFrame(read, index=sel.index, columns=sel.columns)
        return out if labels else out.astype(float)

    def _cache_key(self):
        return ('TsMatch', self.value._cache_key(), self.selection._cache_key())


class TsMatchScore(Signal):
    """How alike each stock and its twin were when the twin was chosen, from ts_match's selection, in the metric's own
    units: a correlation for 'corr'; the distance (positive, smaller is closer) for 'euclidean' and 'path', the latter
    in index points of paths from 100; for 'coint' the Dickey-Fuller t of the pair's spread (more negative, stronger
    reversion), or with stat= the stock's hedge ratio on its twin, the spread's half-life in days, or its standard
    deviation in log points. Held until the next rebalance; no value without a twin. Reads the same internal
    MatchSelection node as every ts_match on the same key and options."""

    def __init__(self, selection: MatchSelection, stat: Optional[str] = None):
        self.selection = selection
        self.stat = stat

    def _compute(self, data):
        sel = self.selection.evaluate(data)
        if self.stat is not None:
            values = sel.stats[self.stat]
        elif self.selection.metric == "coint":
            values = -sel.scores                                  # the t itself
        elif self.selection.metric in DISTANCES:
            values = -sel.scores
        else:
            values = sel.scores
        return pd.DataFrame(values, index=sel.index, columns=sel.columns)

    def _cache_key(self):
        key = ('TsMatchScore', self.selection._cache_key())
        return key if self.stat in (None, "t") else key + (self.stat,)


def _whole(name: str, what: str, v, low: int) -> int:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not float(v).is_integer() or v < low:
        raise ValueError(f"ts_match()'s {name} must be a whole number, at least {low} (got {v!r}): {what}")
    return int(v)


def _selection(name: str, key, lookback, metric, within, matching, rebalance, min_overlap) -> MatchSelection:
    """A MatchSelection from ts_match's (or ts_match_score's) arguments, each checked."""
    from .operators import NoneValue
    from .groups import _as_groups
    if not isinstance(key, Signal):
        raise ValueError(f"{name}()'s key is an expression, such as returns(1), not {key!r}")
    lookback = _whole("lookback", "the days of key to compare", lookback, 2)
    if metric not in METRICS:
        raise ValueError(f"{name}()'s metric is one of {', '.join(repr(m) for m in METRICS)}, not {metric!r}")
    from .primitives import Returns
    if metric in RETURN_KEYED and isinstance(key, Returns) and key.period != 1:
        raise ValueError(f"{name}(metric={metric!r}) compounds its key day by day, so its key is daily returns, "
                         f"returns(1), not returns({key.period}): overlapping {key.period}-day returns compounded as "
                         f"daily ones give the wrong paths")
    if matching not in MATCHINGS:
        raise ValueError(f"{name}()'s matching is 'nearest' (each stock's closest) or 'exclusive' (reciprocal "
                         f"pairs), not {matching!r}")
    rebalance = _whole("rebalance", "how many days the matches are held", rebalance, 1)
    if isinstance(min_overlap, NoneValue):
        min_overlap = None
    min_overlap = lookback if min_overlap is None else _whole(
        "min_overlap", "the fewest days a pair must share", min_overlap, 2)
    if min_overlap > lookback:
        raise ValueError(f"{name}()'s min_overlap ({min_overlap}) can't be more than its lookback ({lookback})")
    if isinstance(within, NoneValue):
        within = None
    if within is not None:
        within = _as_groups(within)
    return MatchSelection(key, lookback, metric, within, matching, rebalance, min_overlap)


def ts_match(value: Signal, key: Signal, lookback: int, metric: str = 'corr', within=None,
             matching: str = 'nearest', rebalance: int = 21, min_overlap: Optional[int] = None) -> TsMatch:
    """Create a TsMatch signal: each stock's statistical twin's value.

    ts_match(value, key, lookback, metric='corr', within=None, matching='nearest', rebalance=21, min_overlap=None)
    """
    if not isinstance(value, Signal):
        raise ValueError(f"ts_match()'s value is an expression, such as returns(5), not {value!r}")
    return TsMatch(value, _selection("ts_match", key, lookback, metric, within, matching, rebalance, min_overlap))


def ts_match_score(key: Signal, lookback: int, metric: str = 'corr', within=None, matching: str = 'nearest',
                   rebalance: int = 21, min_overlap: Optional[int] = None, stat: Optional[str] = None) -> TsMatchScore:
    """Create a TsMatchScore signal: how alike each stock and its twin were when chosen, ts_match's arguments without
    the value; with metric='coint', stat= is 't' (the default), 'hedge', 'half_life' or 'spread_vol'."""
    from .operators import NoneValue
    if isinstance(stat, NoneValue):
        stat = None
    sel = _selection("ts_match_score", key, lookback, metric, within, matching, rebalance, min_overlap)
    if stat is not None:
        if metric != "coint":
            raise ValueError(f"ts_match_score()'s stat= is for metric='coint' (the pair's {', '.join(COINT_STATS)}); "
                             f"with metric={metric!r} the score is its similarity")
        if stat not in COINT_STATS:
            raise ValueError(f"ts_match_score()'s stat is one of {', '.join(repr(k) for k in COINT_STATS)}, "
                             f"not {stat!r}")
        if stat == "t":
            stat = None                                           # the default: one expression, one cache key
    return TsMatchScore(sel, stat)
