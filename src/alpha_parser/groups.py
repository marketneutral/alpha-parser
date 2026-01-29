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


class GroupRank(Signal):
    """Rank within groups (e.g., industry-neutral rank)."""

    def __init__(self, signal: Signal, groups: str):
        self.signal = signal
        self.groups = groups

    def _compute(self, data):
        values = self.signal.evaluate(data)
        group_df = _get_group_data(data, self.groups)

        result = pd.DataFrame(
            np.nan,
            index=values.index,
            columns=values.columns
        )

        for date in values.index:
            date_values = values.loc[date]
            date_groups = group_df.loc[date] if date in group_df.index else group_df.iloc[-1]

            for group_name in date_groups.unique():
                if pd.isna(group_name):
                    continue
                mask = date_groups == group_name
                group_values = date_values[mask]
                ranked = group_values.rank(pct=True)
                result.loc[date, mask] = ranked

        return result

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

        result = pd.DataFrame(
            np.nan,
            index=values.index,
            columns=values.columns
        )

        for date in values.index:
            date_values = values.loc[date]
            date_groups = group_df.loc[date] if date in group_df.index else group_df.iloc[-1]

            for group_name in date_groups.unique():
                if pd.isna(group_name):
                    continue
                mask = date_groups == group_name
                group_values = date_values[mask]
                demeaned = group_values - group_values.mean()
                result.loc[date, mask] = demeaned

        return result

    def _cache_key(self):
        return ('GroupDemean', self.signal._cache_key(), self.groups)


class GroupNeutralize(Signal):
    """Make signal neutral to groups (zero weight per group)."""

    def __init__(self, signal: Signal, groups: str):
        self.signal = signal
        self.groups = groups

    def _compute(self, data):
        values = self.signal.evaluate(data)
        group_df = _get_group_data(data, self.groups)
        result = values.copy()

        for date in values.index:
            date_values = values.loc[date]
            date_groups = group_df.loc[date] if date in group_df.index else group_df.iloc[-1]

            for group_name in date_groups.unique():
                if pd.isna(group_name):
                    continue
                mask = date_groups == group_name
                group_values = date_values[mask]
                result.loc[date, mask] = group_values - group_values.mean()

        return result

    def _cache_key(self):
        return ('GroupNeutralize', self.signal._cache_key(), self.groups)


def group_rank(signal: Signal, groups: str) -> GroupRank:
    """Create a GroupRank signal."""
    return GroupRank(signal, groups)


def group_demean(signal: Signal, groups: str) -> GroupDemean:
    """Create a GroupDemean signal."""
    return GroupDemean(signal, groups)


def group_neutralize(signal: Signal, groups: str) -> GroupNeutralize:
    """Create a GroupNeutralize signal."""
    return GroupNeutralize(signal, groups)


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
