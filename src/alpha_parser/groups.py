"""Group operations for signals (e.g., sector-neutral operations)."""

import numpy as np
import pandas as pd

from .signal import Signal


class GroupRank(Signal):
    """Rank within groups (e.g., industry-neutral rank)."""

    def __init__(self, signal: Signal, groups: str):
        self.signal = signal
        self.groups = groups

    def _compute(self, data):
        values = self.signal.evaluate(data)

        if 'groups' not in data or self.groups not in data['groups'].columns:
            raise ValueError(f"Group column '{self.groups}' not found in data['groups']")

        group_df = data['groups'][self.groups]

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

        if 'groups' not in data or self.groups not in data['groups'].columns:
            raise ValueError(f"Group column '{self.groups}' not found in data['groups']")

        group_df = data['groups'][self.groups]

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

        if 'groups' not in data or self.groups not in data['groups'].columns:
            raise ValueError(f"Group column '{self.groups}' not found in data['groups']")

        group_df = data['groups'][self.groups]
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
