"""Time-series operations for signals."""

import numpy as np
import pandas as pd
from scipy import stats

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
    """Time-series rank (percentile within rolling window).

    Optimized using scipy.stats.rankdata for better performance.
    """

    def __init__(self, signal: Signal, period: int):
        self.signal = signal
        self.period = period

    def _compute(self, data):
        values = self.signal.evaluate(data)

        def _rank_pct(arr):
            """Compute percentile rank of last element in array."""
            valid = ~np.isnan(arr)
            if valid.sum() < 2:
                return np.nan
            # rankdata returns 1-based ranks, convert to percentile
            ranks = stats.rankdata(arr[valid], method='average')
            # Find position of the last element in the valid array
            last_valid_idx = np.where(valid)[0][-1] if valid[-1] else -1
            if last_valid_idx == -1 or not valid[-1]:
                return np.nan
            # Get the rank of the last valid element
            last_rank = ranks[-1]  # Last element in valid subset
            return last_rank / len(ranks)

        return values.rolling(self.period).apply(_rank_pct, raw=True)

    def _cache_key(self):
        return ('TsRank', self.signal._cache_key(), self.period)


class FillForward(Signal):
    """Forward fill missing values for up to N periods."""

    def __init__(self, signal: Signal, limit: int):
        self.signal = signal
        self.limit = limit

    def _compute(self, data):
        values = self.signal.evaluate(data)
        return values.ffill(limit=self.limit)

    def _cache_key(self):
        return ('FillForward', self.signal._cache_key(), self.limit)


class TsCorr(Signal):
    """Rolling correlation between two signals."""

    def __init__(self, signal1: Signal, signal2: Signal, period: int):
        self.signal1 = signal1
        self.signal2 = signal2
        self.period = period

    def _compute(self, data):
        values1 = self.signal1.evaluate(data)
        values2 = self.signal2.evaluate(data)
        return values1.rolling(self.period).corr(values2)

    def _cache_key(self):
        return ('TsCorr', self.signal1._cache_key(), self.signal2._cache_key(), self.period)


class TsCov(Signal):
    """Rolling covariance between two signals."""

    def __init__(self, signal1: Signal, signal2: Signal, period: int):
        self.signal1 = signal1
        self.signal2 = signal2
        self.period = period

    def _compute(self, data):
        values1 = self.signal1.evaluate(data)
        values2 = self.signal2.evaluate(data)
        return values1.rolling(self.period).cov(values2)

    def _cache_key(self):
        return ('TsCov', self.signal1._cache_key(), self.signal2._cache_key(), self.period)


class Ewma(Signal):
    """Exponentially weighted moving average."""

    def __init__(self, signal: Signal, halflife: int):
        self.signal = signal
        self.halflife = halflife

    def _compute(self, data):
        values = self.signal.evaluate(data)
        return values.ewm(halflife=self.halflife).mean()

    def _cache_key(self):
        return ('Ewma', self.signal._cache_key(), self.halflife)


class TsArgmax(Signal):
    """Number of periods since rolling maximum."""

    def __init__(self, signal: Signal, period: int):
        self.signal = signal
        self.period = period

    def _compute(self, data):
        values = self.signal.evaluate(data)

        def _argmax(arr):
            if np.all(np.isnan(arr)):
                return np.nan
            return self.period - 1 - np.nanargmax(arr)

        return values.rolling(self.period).apply(_argmax, raw=True)

    def _cache_key(self):
        return ('TsArgmax', self.signal._cache_key(), self.period)


class TsArgmin(Signal):
    """Number of periods since rolling minimum."""

    def __init__(self, signal: Signal, period: int):
        self.signal = signal
        self.period = period

    def _compute(self, data):
        values = self.signal.evaluate(data)

        def _argmin(arr):
            if np.all(np.isnan(arr)):
                return np.nan
            return self.period - 1 - np.nanargmin(arr)

        return values.rolling(self.period).apply(_argmin, raw=True)

    def _cache_key(self):
        return ('TsArgmin', self.signal._cache_key(), self.period)


class TsSkew(Signal):
    """Rolling skewness."""

    def __init__(self, signal: Signal, period: int):
        self.signal = signal
        self.period = period

    def _compute(self, data):
        values = self.signal.evaluate(data)
        return values.rolling(self.period).skew()

    def _cache_key(self):
        return ('TsSkew', self.signal._cache_key(), self.period)


class TsKurt(Signal):
    """Rolling kurtosis."""

    def __init__(self, signal: Signal, period: int):
        self.signal = signal
        self.period = period

    def _compute(self, data):
        values = self.signal.evaluate(data)
        return values.rolling(self.period).kurt()

    def _cache_key(self):
        return ('TsKurt', self.signal._cache_key(), self.period)


class DecayLinear(Signal):
    """Linearly decaying weighted average (more recent values weighted higher)."""

    def __init__(self, signal: Signal, period: int):
        self.signal = signal
        self.period = period

    def _compute(self, data):
        values = self.signal.evaluate(data)
        # Weights: [1, 2, 3, ..., period] normalized
        weights = np.arange(1, self.period + 1, dtype=float)
        weights = weights / weights.sum()

        def _weighted_mean(arr):
            valid = ~np.isnan(arr)
            if valid.sum() == 0:
                return np.nan
            # Use only valid weights
            w = weights[valid]
            return np.sum(arr[valid] * w) / w.sum()

        return values.rolling(self.period).apply(_weighted_mean, raw=True)

    def _cache_key(self):
        return ('DecayLinear', self.signal._cache_key(), self.period)


# Factory functions

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


def fill_forward(signal: Signal, limit: int) -> FillForward:
    """Create a FillForward signal."""
    return FillForward(signal, limit)


def ts_corr(signal1: Signal, signal2: Signal, period: int) -> TsCorr:
    """Create a TsCorr signal (rolling correlation)."""
    return TsCorr(signal1, signal2, period)


def ts_cov(signal1: Signal, signal2: Signal, period: int) -> TsCov:
    """Create a TsCov signal (rolling covariance)."""
    return TsCov(signal1, signal2, period)


def ewma(signal: Signal, halflife: int) -> Ewma:
    """Create an Ewma signal (exponentially weighted moving average)."""
    return Ewma(signal, halflife)


def ts_argmax(signal: Signal, period: int) -> TsArgmax:
    """Create a TsArgmax signal (periods since max)."""
    return TsArgmax(signal, period)


def ts_argmin(signal: Signal, period: int) -> TsArgmin:
    """Create a TsArgmin signal (periods since min)."""
    return TsArgmin(signal, period)


def ts_skew(signal: Signal, period: int) -> TsSkew:
    """Create a TsSkew signal (rolling skewness)."""
    return TsSkew(signal, period)


def ts_kurt(signal: Signal, period: int) -> TsKurt:
    """Create a TsKurt signal (rolling kurtosis)."""
    return TsKurt(signal, period)


def decay_linear(signal: Signal, period: int) -> DecayLinear:
    """Create a DecayLinear signal (linearly decaying weighted average)."""
    return DecayLinear(signal, period)
