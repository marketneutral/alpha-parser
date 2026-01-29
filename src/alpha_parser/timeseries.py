"""Time-series operations for signals."""

import numpy as np
import pandas as pd

from .signal import Signal


class TsMean(Signal):
    """Rolling mean of any signal."""

    def __init__(self, signal: Signal, period: int):
        self.signal = signal
        self.period = period

    def _compute(self, data):
        values = self.signal.evaluate(data)
        return values.rolling(self.period).mean()

    def _cache_key(self):
        return ('TsMean', self.signal._cache_key(), self.period)


class TsStd(Signal):
    """Rolling standard deviation of any signal."""

    def __init__(self, signal: Signal, period: int):
        self.signal = signal
        self.period = period

    def _compute(self, data):
        values = self.signal.evaluate(data)
        return values.rolling(self.period).std()

    def _cache_key(self):
        return ('TsStd', self.signal._cache_key(), self.period)


class TsSum(Signal):
    """Rolling sum of any signal."""

    def __init__(self, signal: Signal, period: int):
        self.signal = signal
        self.period = period

    def _compute(self, data):
        values = self.signal.evaluate(data)
        return values.rolling(self.period).sum()

    def _cache_key(self):
        return ('TsSum', self.signal._cache_key(), self.period)


class TsMax(Signal):
    """Rolling max of any signal."""

    def __init__(self, signal: Signal, period: int):
        self.signal = signal
        self.period = period

    def _compute(self, data):
        values = self.signal.evaluate(data)
        return values.rolling(self.period).max()

    def _cache_key(self):
        return ('TsMax', self.signal._cache_key(), self.period)


class TsMin(Signal):
    """Rolling min of any signal."""

    def __init__(self, signal: Signal, period: int):
        self.signal = signal
        self.period = period

    def _compute(self, data):
        values = self.signal.evaluate(data)
        return values.rolling(self.period).min()

    def _cache_key(self):
        return ('TsMin', self.signal._cache_key(), self.period)


class Delay(Signal):
    """Lag/shift any signal."""

    def __init__(self, signal: Signal, period: int):
        self.signal = signal
        self.period = period

    def _compute(self, data):
        values = self.signal.evaluate(data)
        return values.shift(self.period)

    def _cache_key(self):
        return ('Delay', self.signal._cache_key(), self.period)


class Delta(Signal):
    """Difference from N periods ago."""

    def __init__(self, signal: Signal, period: int):
        self.signal = signal
        self.period = period

    def _compute(self, data):
        values = self.signal.evaluate(data)
        return values - values.shift(self.period)

    def _cache_key(self):
        return ('Delta', self.signal._cache_key(), self.period)


class TsRank(Signal):
    """Time-series rank (percentile within rolling window)."""

    def __init__(self, signal: Signal, period: int):
        self.signal = signal
        self.period = period

    def _compute(self, data):
        values = self.signal.evaluate(data)
        return values.rolling(self.period).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan
        )

    def _cache_key(self):
        return ('TsRank', self.signal._cache_key(), self.period)


def ts_mean(signal: Signal, period: int) -> TsMean:
    """Create a TsMean signal."""
    return TsMean(signal, period)


def ts_std(signal: Signal, period: int) -> TsStd:
    """Create a TsStd signal."""
    return TsStd(signal, period)


def ts_sum(signal: Signal, period: int) -> TsSum:
    """Create a TsSum signal."""
    return TsSum(signal, period)


def ts_max(signal: Signal, period: int) -> TsMax:
    """Create a TsMax signal."""
    return TsMax(signal, period)


def ts_min(signal: Signal, period: int) -> TsMin:
    """Create a TsMin signal."""
    return TsMin(signal, period)


def delay(signal: Signal, period: int) -> Delay:
    """Create a Delay signal."""
    return Delay(signal, period)


def delta(signal: Signal, period: int) -> Delta:
    """Create a Delta signal."""
    return Delta(signal, period)


def ts_rank(signal: Signal, period: int) -> TsRank:
    """Create a TsRank signal."""
    return TsRank(signal, period)
