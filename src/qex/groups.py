"""Group operations for signals (e.g., sector-neutral operations).

A group operator's groups are a group's name ('sector', looked up in the data) or any expression that gives each
stock a label on each day: quantile(field('mcap'), 5), coalesce(group('sector'), 'other'), or group_cross(...) of
several groupings. A stock with no label is in no group, and the operator gives it no value.
"""

from typing import Union

import numpy as np
import pandas as pd

from .signal import Signal
from .data import resolve_data
from .universe import in_universe, finite_in_universe, universe_members


GroupsArg = Union[str, Signal]


def _as_groups(groups):
    """A group operator's groups as given: a group's name, or an expression giving labels."""
    if isinstance(groups, (str, Signal)):
        return groups
    raise TypeError(f"groups are a group's name ('sector') or an expression that gives each stock a label "
                    f"(quantile(x, 5)), not {groups!r}")


def _groups_key(groups):
    return groups._cache_key() if isinstance(groups, Signal) else groups


def _get_group_data(data, groups) -> pd.DataFrame:
    """The labels, dates x names: an expression's value, or by name from data['sector'] or data['groups']['sector']."""
    if isinstance(groups, Signal):
        labels = groups.evaluate(data)
        if not isinstance(labels, pd.DataFrame):
            raise ValueError(f"groups must give each stock a label on each day, not one value for all of them "
                             f"({labels!r}); write a group's name, or an expression such as quantile(x, 5)")
        return labels
    data = resolve_data(data)

    # Try direct access first: data['sector']
    if groups in data:
        return data[groups]

    # Try nested format: data['groups']['sector']
    if 'groups' in data:
        groups_container = data['groups']
        # Support both pd.DataFrame and custom GroupsContainer-like objects
        if groups in groups_container:
            return groups_container[groups]

    raise ValueError(
        f"Group '{groups}' not found. "
        f"Expected data['{groups}'] or data['groups']['{groups}']"
    )


def _align_labels(group_df: pd.DataFrame, index, columns) -> pd.DataFrame:
    """Labels on these dates and names, as of each date: a date missing from group_df takes the latest row on or
    before it (labels given monthly hold until the next row), a date before group_df's first row has no group, and a
    name missing from it has no group (NaN). Never a later row: that would group stocks by labels not yet known."""
    groups = group_df.reindex(columns=columns)
    if not groups.index.is_monotonic_increasing:
        groups = groups.sort_index()
    if groups.index.equals(index):
        return groups.astype(object)
    return groups.astype(object).reindex(index, method='ffill')


def _group_codes(values: pd.DataFrame, group_df: pd.DataFrame):
    """Integer group codes aligned to values' dates and columns, -1 where the group is NaN.

    A date missing from group_df takes the latest row on or before it (none before its first row), and
    a column missing from it has no group (NaN).
    """
    rows = _align_labels(group_df, values.index, values.columns)
    codes, uniques = pd.factorize(rows.to_numpy(dtype=object).ravel())
    return codes.reshape(rows.shape), len(uniques)


def _group_totals(v: np.ndarray, codes: np.ndarray, n_groups: int):
    """Per (date, group): the key of every cell, and the sum and count of its non-NaN values."""
    key = np.arange(v.shape[0])[:, None] * n_groups + codes
    ok = (codes >= 0) & ~np.isnan(v)
    size = v.shape[0] * n_groups
    sums = np.bincount(key[ok], weights=v[ok], minlength=size)
    counts = np.bincount(key[ok], minlength=size)
    return key, sums, counts


def _in_group(codes: np.ndarray, values: pd.DataFrame) -> np.ndarray:
    """The cells that get their group's value: names with a group that day, inside the cross-sectional universe."""
    has = codes >= 0
    members = universe_members(values.index, values.columns)
    return has if members is None else has & members


class Group(Signal):
    """Access group/categorical data as a Signal.

    Enables filtering by group membership in expressions like:
        where(group('sector')=='Financials', 0, returns(60))

    Supports both data formats:
        - data['sector'] (top-level)
        - data['groups']['sector'] (nested)
    """

    def __init__(self, name: str):
        self.name = name

    def _compute(self, data):
        return _get_group_data(data, self.name)

    def _cache_key(self):
        return ('Group', self.name)


def group(name: str) -> Group:
    """Access group/categorical data for use in expressions.

    Example:
        # Filter out Financials sector
        qex("where(group('sector')=='Financials', 0, returns(60))")

        # Only trade Technology stocks
        qex("where(group('sector')=='Technology', returns(20), 0)")
    """
    return Group(name)


class GroupRank(Signal):
    """Rank within groups (e.g., industry-neutral rank)."""

    def __init__(self, signal: Signal, groups: GroupsArg):
        self.signal = signal
        self.groups = _as_groups(groups)

    def _compute(self, data):
        values = in_universe(self.signal.evaluate(data))
        group_df = _get_group_data(data, self.groups)

        codes, _ = _group_codes(values, group_df)
        T, N = codes.shape
        has = (codes >= 0).ravel()
        long = pd.DataFrame({
            'date': np.repeat(np.arange(T), N)[has],
            'group': codes.ravel()[has],
            'value': values.to_numpy(dtype=float).ravel()[has],
        })
        ranked = long.groupby(['date', 'group'])['value'].rank(pct=True).to_numpy()
        out = np.full(T * N, np.nan)
        out[has] = ranked
        return pd.DataFrame(out.reshape(T, N), index=values.index, columns=values.columns)

    def _cache_key(self):
        return ('GroupRank', self.signal._cache_key(), _groups_key(self.groups))


class GroupDemean(Signal):
    """Demean within groups (subtract group mean)."""

    def __init__(self, signal: Signal, groups: GroupsArg):
        self.signal = signal
        self.groups = _as_groups(groups)

    def _compute(self, data):
        values = finite_in_universe(self.signal.evaluate(data))
        group_df = _get_group_data(data, self.groups)

        codes, n_groups = _group_codes(values, group_df)
        v = values.to_numpy(dtype=float)
        key, sums, counts = _group_totals(v, codes, n_groups)
        with np.errstate(invalid='ignore', divide='ignore'):
            means = sums / counts
        has = codes >= 0
        out = np.full(v.shape, np.nan)
        out[has] = v[has] - means[key[has]]
        return pd.DataFrame(out, index=values.index, columns=values.columns)

    def _cache_key(self):
        return ('GroupDemean', self.signal._cache_key(), _groups_key(self.groups))


def group_rank(signal: Signal, groups: GroupsArg) -> GroupRank:
    """Create a GroupRank signal."""
    return GroupRank(signal, groups)


def group_demean(signal: Signal, groups: GroupsArg) -> GroupDemean:
    """Create a GroupDemean signal."""
    return GroupDemean(signal, groups)


class GroupQuantile(Signal):
    """Quantile buckets within groups, 1 (lowest) to n, as quantile() within each group: a dependent double sort is
    group_quantile(value, quantile(size, 3), 3)."""

    def __init__(self, signal: Signal, groups: GroupsArg, buckets: int):
        if isinstance(buckets, bool) or not isinstance(buckets, int) or buckets < 1:
            raise ValueError(f"group_quantile takes a whole number of buckets, at least 1 (got {buckets!r})")
        self.signal = signal
        self.groups = _as_groups(groups)
        self.buckets = buckets

    def _compute(self, data):
        ranks = GroupRank(self.signal, self.groups).evaluate(data)          # percentile in (0, 1] within the group
        return np.ceil(ranks * self.buckets).clip(1, self.buckets)

    def _cache_key(self):
        return ('GroupQuantile', self.signal._cache_key(), _groups_key(self.groups), self.buckets)


def group_quantile(signal: Signal, groups: GroupsArg, buckets: int) -> GroupQuantile:
    """Quantile buckets (1 to n, ascending) within each group."""
    return GroupQuantile(signal, groups, buckets)


def _label_text(v):
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


class GroupCross(Signal):
    """One grouping from several: each stock's label is its labels joined, 'Technology|3', and it has none where any
    part has none. group_demean(x, group_cross('sector', quantile(field('mcap'), 3))) demeans within sector and size."""

    def __init__(self, *groups):
        if len(groups) < 2:
            raise ValueError("group_cross takes two or more groupings")
        self.groups = tuple(Group(g) if isinstance(g, str) else _as_groups(g) for g in groups)

    def _compute(self, data):
        parts = [_get_group_data(data, g) for g in self.groups]
        index = parts[0].index
        for p in parts[1:]:
            index = index.union(p.index)
        columns = parts[0].columns
        for p in parts[1:]:
            columns = columns.union(p.columns, sort=False)
        rows = [_align_labels(p, index, columns).to_numpy(dtype=object) for p in parts]
        missing = np.zeros(rows[0].shape, bool)
        for r in rows:
            missing |= pd.isna(r)
        text = np.frompyfunc(_label_text, 1, 1)
        parts_text = [text(r[~missing]) for r in rows]
        for g, t in zip(self.groups, parts_text):
            bad = [v for v in t if "|" in v]
            if bad:                                       # 'a|b' + 'c' and 'a' + 'b|c' would be one group
                raise ValueError(f"group_cross joins labels with '|', so a label cannot contain one: {bad[0]!r}")
        joined = rows[0].copy()
        joined[~missing] = parts_text[0]
        for t in parts_text[1:]:
            joined[~missing] = joined[~missing] + "|" + t
        joined[missing] = np.nan
        return pd.DataFrame(joined, index=index, columns=columns)

    def _cache_key(self):
        return ('GroupCross',) + tuple(g._cache_key() for g in self.groups)


def group_cross(*groups) -> GroupCross:
    """Combine groupings (names or label expressions) into one: group_cross('sector', quantile(field('mcap'), 3))."""
    return GroupCross(*groups)


class GroupCountValid(Signal):
    """How many values each group has over the last ``window`` days, broadcast to every member.

    On each day, the members are the stocks with that label that day, and the count is every value those stocks have
    over the window (a stock counts once per day it has a value; an infinity counts, as in is_valid). With window 1 it
    is the number of the group's members with a value that day. The first days use the shorter window there is. A
    group whose members have no values in the window gets 0; names with no group, or outside the universe, get NaN.
    """

    def __init__(self, signal: Signal, groups: GroupsArg, window: int):
        if isinstance(window, bool) or not isinstance(window, int) or window < 1:
            raise ValueError(f"group_count_valid takes a window of at least 1 day (got {window!r})")
        self.signal = signal
        self.groups = _as_groups(groups)
        self.window = window

    def _compute(self, data):
        values = in_universe(self.signal.evaluate(data))
        group_df = _get_group_data(data, self.groups)

        codes, n_groups = _group_codes(values, group_df)
        per_name = values.notna().astype(float).rolling(self.window, min_periods=1).sum().to_numpy()
        T = codes.shape[0]
        key = np.arange(T)[:, None] * n_groups + codes
        labelled = codes >= 0
        counts = np.bincount(key[labelled], weights=per_name[labelled], minlength=T * max(n_groups, 1))
        has = _in_group(codes, values)
        out = np.full(codes.shape, np.nan)
        out[has] = counts[key[has]]
        return pd.DataFrame(out, index=values.index, columns=values.columns)

    def _cache_key(self):
        return ('GroupCountValid', self.signal._cache_key(), _groups_key(self.groups), self.window)


def group_count_valid(signal: Signal, groups: GroupsArg, window: int) -> GroupCountValid:
    """Create a GroupCountValid signal."""
    return GroupCountValid(signal, groups, window)


class GroupStd(Signal):
    """Standard deviation of a signal across each group's members on each day, broadcast to every member.

    Like group_demean and group_sum, it works on one day's cross-section: the spread of x among the
    stocks that share a label that day (sample standard deviation, ddof 1). A member whose own value
    is missing still gets its group's value; a group with fewer than two values that day gets NaN.

        group_demean(x, 'sector') / group_std(x, 'sector')       a z-score within each sector

    Before qex 0.12.0 it took a window, group_std(x, g, n), and pooled n days of every member's values
    into one sample. That form is refused: for a pair's spread over time use
    let s = group_demean(x, 'pair') in s / ts_std(s, n).
    """

    def __init__(self, signal: Signal, groups: GroupsArg):
        self.signal = signal
        self.groups = _as_groups(groups)

    def _compute(self, data):
        values = finite_in_universe(self.signal.evaluate(data))
        group_df = _get_group_data(data, self.groups)

        codes, n_groups = _group_codes(values, group_df)
        v = values.to_numpy(dtype=float)
        key, sums, counts = _group_totals(v, codes, n_groups)
        has = codes >= 0
        ok = has & ~np.isnan(v)
        with np.errstate(invalid='ignore', divide='ignore'):
            means = sums / counts
            dev = np.zeros(v.shape)
            dev[ok] = (v[ok] - means[key[ok]]) ** 2                  # two passes: deviations from the group mean
            ss = np.bincount(key[ok], weights=dev[ok], minlength=sums.size)
            std = np.sqrt(ss / np.where(counts >= 2, counts - 1, np.nan))
        has = _in_group(codes, values)
        out = np.full(v.shape, np.nan)
        out[has] = std[key[has]]
        return pd.DataFrame(out, index=values.index, columns=values.columns)

    def _cache_key(self):
        return ('GroupStd', self.signal._cache_key(), _groups_key(self.groups))


def group_std(signal: Signal, groups: GroupsArg, *window) -> GroupStd:
    """Create a GroupStd signal: the standard deviation across each group's members that day.

    Example:
        # Within-sector z-score
        group_demean(returns(5), 'sector') / group_std(returns(5), 'sector')
    """
    if window:
        raise ValueError(
            f"group_std() takes a signal and its groups, group_std(x, groups), and gives the spread across the "
            f"group's members on each day, like the other group operators. Before qex 0.12.0 it took a window "
            f"(got {window[0]!r}) and pooled that many days; that form is gone. For a pair's spread over time, "
            f"use let s = group_demean(x, groups) in s / ts_std(s, n).")
    return GroupStd(signal, groups)


class GroupSum(Signal):
    """Sum of values within each group, broadcast to all members.

    Useful for computing group-level aggregates like total sector volume
    or total pair exposure.
    """

    def __init__(self, signal: Signal, groups: GroupsArg):
        self.signal = signal
        self.groups = _as_groups(groups)

    def _compute(self, data):
        values = finite_in_universe(self.signal.evaluate(data))
        group_df = _get_group_data(data, self.groups)

        codes, n_groups = _group_codes(values, group_df)
        v = values.to_numpy(dtype=float)
        key, sums, counts = _group_totals(v, codes, n_groups)
        sums = np.where(counts > 0, sums, np.nan)               # a group with no values that day has no sum, not 0
        # Every member gets its group's sum, including members whose own value is NaN; names outside the universe don't
        has = _in_group(codes, values)
        out = np.full(v.shape, np.nan)
        out[has] = sums[key[has]]
        return pd.DataFrame(out, index=values.index, columns=values.columns)

    def _cache_key(self):
        return ('GroupSum', self.signal._cache_key(), _groups_key(self.groups))


def group_sum(signal: Signal, groups: GroupsArg) -> GroupSum:
    """Create a GroupSum signal.

    Sums values within each group and broadcasts to all members.

    Example:
        # Total sector volume
        sector_volume = group_sum(volume(1), 'sector')

        # Stock's share of sector volume
        volume_share = volume(1) / group_sum(volume(1), 'sector')
    """
    return GroupSum(signal, groups)
