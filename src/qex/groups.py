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
from .universe import in_universe, finite_in_universe


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
    """Labels on these dates and names: a date missing from group_df takes its last row, a name missing from it has
    no group (NaN)."""
    groups = group_df.reindex(columns=columns)
    rows = groups.reindex(index).astype(object)
    missing = ~index.isin(groups.index)
    if missing.any():
        rows.loc[missing] = groups.iloc[-1].values
    return rows


def _group_codes(values: pd.DataFrame, group_df: pd.DataFrame):
    """Integer group codes aligned to values' dates and columns, -1 where the group is NaN.

    A date missing from group_df takes group_df's last row, and a column missing from it has
    no group (NaN), as in the per-date loops these replace.
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
    """Count non-NaN values within each group over a rolling window, broadcast to all members."""

    def __init__(self, signal: Signal, groups: GroupsArg, window: int):
        self.signal = signal
        self.groups = _as_groups(groups)
        self.window = window

    def _compute(self, data):
        values = in_universe(self.signal.evaluate(data))
        group_df = _get_group_data(data, self.groups)

        result = pd.DataFrame(
            np.nan,
            index=values.index,
            columns=values.columns
        )

        # For each date, count valid values in each group over the rolling window
        for i, date in enumerate(values.index):
            # Get the window of data
            start_idx = max(0, i - self.window + 1)
            window_values = values.iloc[start_idx:i + 1]

            date_groups = group_df.loc[date] if date in group_df.index else group_df.iloc[-1]

            for group_name in date_groups.unique():
                if pd.isna(group_name):
                    continue
                mask = date_groups == group_name
                group_tickers = mask[mask].index

                # Count non-NaN values for this group's tickers in the window
                group_window_values = window_values[group_tickers]
                count = group_window_values.notna().sum().sum()

                # Broadcast count to all members of the group
                result.loc[date, mask] = count

        return result

    def _cache_key(self):
        return ('GroupCountValid', self.signal._cache_key(), _groups_key(self.groups), self.window)


def group_count_valid(signal: Signal, groups: GroupsArg, window: int) -> GroupCountValid:
    """Create a GroupCountValid signal."""
    return GroupCountValid(signal, groups, window)


class GroupStd(Signal):
    """Rolling standard deviation within groups.

    Computes the std of each group member's values over a rolling window,
    then broadcasts the group std to all members. Useful for normalizing
    pair spreads.
    """

    def __init__(self, signal: Signal, groups: GroupsArg, window: int):
        self.signal = signal
        self.groups = _as_groups(groups)
        self.window = window

    def _compute(self, data):
        values = finite_in_universe(self.signal.evaluate(data))
        group_df = _get_group_data(data, self.groups)

        result = pd.DataFrame(
            np.nan,
            index=values.index,
            columns=values.columns
        )

        # For each date, compute rolling std within each group
        for i, date in enumerate(values.index):
            if i < self.window - 1:
                continue

            # Get the window of data
            start_idx = i - self.window + 1
            window_values = values.iloc[start_idx:i + 1]

            date_groups = group_df.loc[date] if date in group_df.index else group_df.iloc[-1]

            for group_name in date_groups.unique():
                if pd.isna(group_name):
                    continue
                mask = date_groups == group_name
                group_tickers = mask[mask].index

                # Get all values for this group in the window and compute std
                group_window_values = window_values[group_tickers].values.flatten()
                group_window_values = group_window_values[~np.isnan(group_window_values)]

                if len(group_window_values) >= 2:
                    std = np.std(group_window_values, ddof=1)
                    result.loc[date, mask] = std

        return result

    def _cache_key(self):
        return ('GroupStd', self.signal._cache_key(), _groups_key(self.groups), self.window)


def group_std(signal: Signal, groups: GroupsArg, window: int) -> GroupStd:
    """Create a GroupStd signal.

    Computes rolling standard deviation within each group. Useful for
    normalizing pair spreads to get z-scores.

    Example:
        # Z-score of pair spread
        spread = group_demean(returns(5), 'pair')
        spread_vol = group_std(returns(5), 'pair', 60)
        zscore = spread / spread_vol
    """
    return GroupStd(signal, groups, window)


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
        key, sums, _ = _group_totals(v, codes, n_groups)
        # Every member gets its group's sum, including members whose own value is NaN
        has = codes >= 0
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
