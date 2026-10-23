"""Group operations for signals (e.g., sector-neutral operations)."""

import numpy as np
import pandas as pd

from .signal import Signal
from .data import resolve_data


def _get_group_data(data, groups: str) -> pd.DataFrame:
    """Get group DataFrame, supporting both data['sector'] and data['groups']['sector'] formats."""
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


def _group_codes(values: pd.DataFrame, group_df: pd.DataFrame):
    """Integer group codes aligned to values' dates and columns, -1 where the group is NaN.

    A date missing from group_df takes group_df's last row, and a column missing from it has
    no group (NaN), as in the per-date loops these replace.
    """
    groups = group_df.reindex(columns=values.columns)
    rows = groups.reindex(values.index)
    missing = ~values.index.isin(groups.index)
    if missing.any():
        rows.loc[missing] = groups.iloc[-1].values
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

    def __init__(self, signal: Signal, groups: str):
        self.signal = signal
        self.groups = groups

    def _compute(self, data):
        values = self.signal.evaluate(data)
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
        return ('GroupRank', self.signal._cache_key(), self.groups)


class GroupDemean(Signal):
    """Demean within groups (subtract group mean)."""

    def __init__(self, signal: Signal, groups: str):
        self.signal = signal
        self.groups = groups

    def _compute(self, data):
        values = self.signal.evaluate(data)
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
        return ('GroupDemean', self.signal._cache_key(), self.groups)


def group_rank(signal: Signal, groups: str) -> GroupRank:
    """Create a GroupRank signal."""
    return GroupRank(signal, groups)


def group_demean(signal: Signal, groups: str) -> GroupDemean:
    """Create a GroupDemean signal."""
    return GroupDemean(signal, groups)


class GroupCountValid(Signal):
    """Count non-NaN values within each group over a rolling window, broadcast to all members."""

    def __init__(self, signal: Signal, groups: str, window: int):
        self.signal = signal
        self.groups = groups
        self.window = window

    def _compute(self, data):
        values = self.signal.evaluate(data)
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
        return ('GroupCountValid', self.signal._cache_key(), self.groups, self.window)


def group_count_valid(signal: Signal, groups: str, window: int) -> GroupCountValid:
    """Create a GroupCountValid signal."""
    return GroupCountValid(signal, groups, window)


class GroupStd(Signal):
    """Rolling standard deviation within groups.

    Computes the std of each group member's values over a rolling window,
    then broadcasts the group std to all members. Useful for normalizing
    pair spreads.
    """

    def __init__(self, signal: Signal, groups: str, window: int):
        self.signal = signal
        self.groups = groups
        self.window = window

    def _compute(self, data):
        values = self.signal.evaluate(data)
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
        return ('GroupStd', self.signal._cache_key(), self.groups, self.window)


def group_std(signal: Signal, groups: str, window: int) -> GroupStd:
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

    def __init__(self, signal: Signal, groups: str):
        self.signal = signal
        self.groups = groups

    def _compute(self, data):
        values = self.signal.evaluate(data)
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
        return ('GroupSum', self.signal._cache_key(), self.groups)


def group_sum(signal: Signal, groups: str) -> GroupSum:
    """Create a GroupSum signal.

    Sums values within each group and broadcasts to all members.

    Example:
        # Total sector volume
        sector_volume = group_sum(volume(1), 'sector')

        # Stock's share of sector volume
        volume_share = volume(1) / group_sum(volume(1), 'sector')
    """
    return GroupSum(signal, groups)
