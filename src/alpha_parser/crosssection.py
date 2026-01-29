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


class Quantile(Signal):
    """Assign values to quantile buckets (e.g., quintiles, deciles)."""

    def __init__(self, signal: Signal, buckets: int):
        self.signal = signal
        self.buckets = buckets

    def _compute(self, data):
        import numpy as np
        values = self.signal.evaluate(data)
        # Get percentile ranks (0-1), then map to buckets (1 to n)
        ranks = values.rank(axis=1, pct=True)
        # Map (0,1] to [1, buckets] - bucket 1 is lowest, bucket n is highest
        # Use ceiling so rank=0.01 -> bucket 1, rank=1.0 -> bucket n
        quantiles = np.ceil(ranks * self.buckets).clip(1, self.buckets)
        return quantiles

    def _cache_key(self):
        return ('Quantile', self.signal._cache_key(), self.buckets)


def quantile(signal: Signal, buckets: int) -> Quantile:
    """Create a Quantile signal. Buckets are numbered 1 to n (1=lowest, n=highest)."""
    return Quantile(signal, buckets)


class Winsorize(Signal):
    """Cap extreme values at specified percentiles cross-sectionally."""

    def __init__(self, signal: Signal, limit: float):
        self.signal = signal
        self.limit = limit  # e.g., 0.05 caps at 5th and 95th percentiles

    def _compute(self, data):
        values = self.signal.evaluate(data)
        # Compute percentiles for each row (cross-section)
        lower = values.quantile(self.limit, axis=1)
        upper = values.quantile(1 - self.limit, axis=1)
        # Clip values to the percentile bounds
        result = values.clip(lower=lower, upper=upper, axis=0)
        return result

    def _cache_key(self):
        return ('Winsorize', self.signal._cache_key(), self.limit)


def winsorize(signal: Signal, limit: float) -> Winsorize:
    """Create a Winsorize signal. Limit is the percentile to cap at (e.g., 0.05 caps at 5th/95th)."""
    return Winsorize(signal, limit)
