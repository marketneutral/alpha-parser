"""Cross-sectional operations for signals."""

from .signal import Signal
from .universe import in_universe


class Rank(Signal):
    """Cross-sectional percentile rank."""

    def __init__(self, signal: Signal):
        self.signal = signal

    def _compute(self, data):
        values = in_universe(self.signal.evaluate(data))
        return values.rank(axis=1, pct=True)

    def _cache_key(self):
        return ('Rank', self.signal._cache_key())


class ZScore(Signal):
    """Cross-sectional z-score (standardization)."""

    def __init__(self, signal: Signal):
        self.signal = signal

    def _compute(self, data):
        values = in_universe(self.signal.evaluate(data))
        mean = values.mean(axis=1)
        std = values.std(axis=1)
        # Handle division by zero - replace 0 std with NaN
        std = std.replace(0, float('nan'))
        return values.sub(mean, axis=0).div(std, axis=0)

    def _cache_key(self):
        return ('ZScore', self.signal._cache_key())


class Demean(Signal):
    """Subtract cross-sectional mean."""

    def __init__(self, signal: Signal):
        self.signal = signal

    def _compute(self, data):
        values = in_universe(self.signal.evaluate(data))
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
        values = in_universe(self.signal.evaluate(data))
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
        values = in_universe(self.signal.evaluate(data))
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


class Scale(Signal):
    """Scale signal so that absolute values sum to 1 cross-sectionally."""

    def __init__(self, signal: Signal):
        self.signal = signal

    def _compute(self, data):
        values = in_universe(self.signal.evaluate(data))
        abs_sum = values.abs().sum(axis=1)
        abs_sum = abs_sum.replace(0, float('nan'))
        return values.div(abs_sum, axis=0)

    def _cache_key(self):
        return ('Scale', self.signal._cache_key())


def scale(signal: Signal) -> Scale:
    """Create a Scale signal. Scales so abs values sum to 1."""
    return Scale(signal)


class Truncate(Signal):
    """Truncate signal to max absolute weight cross-sectionally."""

    def __init__(self, signal: Signal, max_weight: float):
        self.signal = signal
        self.max_weight = max_weight

    def _compute(self, data):
        values = self.signal.evaluate(data)
        return values.clip(lower=-self.max_weight, upper=self.max_weight)

    def _cache_key(self):
        return ('Truncate', self.signal._cache_key(), self.max_weight)


def truncate(signal: Signal, max_weight: float) -> Truncate:
    """Create a Truncate signal. Clips values to [-max_weight, max_weight]."""
    return Truncate(signal, max_weight)


class RegressionNeut(Signal):
    """Cross-sectional residual of y on x, with an intercept, separately each day.

    On each date, over the names where both y and x are valid:
        residual = (y - mean(y)) - b * (x - mean(x)),   b = cov(y, x) / var(x)
    so the result has zero mean and zero cross-sectional covariance with x. Names where either is
    NaN are NaN. A day on which x does not vary (or only one name is valid) gets b = 0.

    The neutrality holds for the expression itself. A book built from it afterwards (truncation,
    clipping, rescaling) is only approximately neutral to x, and less so the more the cap binds.
    """

    def __init__(self, y: Signal, x: Signal):
        self.y = y
        self.x = x

    def _compute(self, data):
        y = in_universe(self.y.evaluate(data))
        x = in_universe(self.x.evaluate(data).reindex_like(y))
        valid = y.notna() & x.notna()
        y, x = y.where(valid), x.where(valid)
        yc = y.sub(y.mean(axis=1), axis=0)
        xc = x.sub(x.mean(axis=1), axis=0)
        var = (xc * xc).sum(axis=1)
        b = ((yc * xc).sum(axis=1) / var.where(var > 0)).fillna(0.0)
        return yc - xc.mul(b, axis=0)

    def _cache_key(self):
        return ('RegressionNeut', self.y._cache_key(), self.x._cache_key())


def regression_neut(y: Signal, x: Signal) -> RegressionNeut:
    """Create a RegressionNeut signal: y with its daily cross-sectional fit on x removed."""
    return RegressionNeut(y, x)
