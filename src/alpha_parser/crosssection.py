"""Cross-sectional operations for signals."""

from .signal import Signal


class Rank(Signal):
    """Cross-sectional percentile rank."""

    def __init__(self, signal: Signal):
        self.signal = signal

    def _compute(self, data):
        values = self.signal.evaluate(data)
        return values.rank(axis=1, pct=True)

    def _cache_key(self):
        return ('Rank', self.signal._cache_key())


class ZScore(Signal):
    """Cross-sectional z-score (standardization)."""

    def __init__(self, signal: Signal):
        self.signal = signal

    def _compute(self, data):
        values = self.signal.evaluate(data)
        mean = values.mean(axis=1)
        std = values.std(axis=1)
        return values.sub(mean, axis=0).div(std, axis=0)

    def _cache_key(self):
        return ('ZScore', self.signal._cache_key())


class Demean(Signal):
    """Subtract cross-sectional mean."""

    def __init__(self, signal: Signal):
        self.signal = signal

    def _compute(self, data):
        values = self.signal.evaluate(data)
        mean = values.mean(axis=1)
        return values.sub(mean, axis=0)

    def _cache_key(self):
        return ('Demean', self.signal._cache_key())


def rank(signal: Signal) -> Rank:
    """Create a Rank signal."""
    return Rank(signal)


def zscore(signal: Signal) -> ZScore:
    """Create a ZScore signal."""
    return ZScore(signal)


def demean(signal: Signal) -> Demean:
    """Create a Demean signal."""
    return Demean(signal)
