"""Time-series operations for signals."""

import numpy as np
import pandas as pd
from scipy import stats

from .signal import Signal
from .universe import in_universe

# Default minimum periods for rolling operations (~1 month)
DEFAULT_MIN_PERIODS = 20


def _get_min_periods(period: int, min_periods: int = None) -> int:
    """Get effective min_periods, capped at window size."""
    if min_periods is not None:
        return min(min_periods, period)
    return min(DEFAULT_MIN_PERIODS, period)


class TsMean(Signal):
    """Rolling mean of any signal."""

    def __init__(self, signal: Signal, period: int, min_periods: int = None):
        self.signal = signal
        self.period = period
        self.min_periods = _get_min_periods(period, min_periods)

    def _compute(self, data):
        values = self.signal.evaluate(data)
        return values.rolling(self.period, min_periods=self.min_periods).mean()

    def _cache_key(self):
        return ('TsMean', self.signal._cache_key(), self.period, self.min_periods)


class TsStd(Signal):
    """Rolling standard deviation of any signal."""

    def __init__(self, signal: Signal, period: int, min_periods: int = None):
        self.signal = signal
        self.period = period
        self.min_periods = _get_min_periods(period, min_periods)

    def _compute(self, data):
        values = self.signal.evaluate(data)
        return values.rolling(self.period, min_periods=self.min_periods).std()

    def _cache_key(self):
        return ('TsStd', self.signal._cache_key(), self.period, self.min_periods)


class TsSum(Signal):
    """Rolling sum of any signal."""

    def __init__(self, signal: Signal, period: int, min_periods: int = None):
        self.signal = signal
        self.period = period
        self.min_periods = _get_min_periods(period, min_periods)

    def _compute(self, data):
        values = self.signal.evaluate(data)
        return values.rolling(self.period, min_periods=self.min_periods).sum()

    def _cache_key(self):
        return ('TsSum', self.signal._cache_key(), self.period, self.min_periods)


class TsMax(Signal):
    """Rolling max of any signal."""

    def __init__(self, signal: Signal, period: int, min_periods: int = None):
        self.signal = signal
        self.period = period
        self.min_periods = _get_min_periods(period, min_periods)

    def _compute(self, data):
        values = self.signal.evaluate(data)
        return values.rolling(self.period, min_periods=self.min_periods).max()

    def _cache_key(self):
        return ('TsMax', self.signal._cache_key(), self.period, self.min_periods)


class TsMin(Signal):
    """Rolling min of any signal."""

    def __init__(self, signal: Signal, period: int, min_periods: int = None):
        self.signal = signal
        self.period = period
        self.min_periods = _get_min_periods(period, min_periods)

    def _compute(self, data):
        values = self.signal.evaluate(data)
        return values.rolling(self.period, min_periods=self.min_periods).min()

    def _cache_key(self):
        return ('TsMin', self.signal._cache_key(), self.period, self.min_periods)


class Delay(Signal):
    """Lag/shift any signal.

    Only positive periods are allowed to prevent lookahead bias.
    delay(x, 1) gives yesterday's value of x.
    """

    def __init__(self, signal: Signal, period: int):
        if period < 0:
            raise ValueError(
                f"delay() period must be >= 0 (got {period}). "
                "Negative delays cause lookahead bias."
            )
        self.signal = signal
        self.period = period

    def _compute(self, data):
        values = self.signal.evaluate(data)
        return values.shift(self.period)

    def _cache_key(self):
        return ('Delay', self.signal._cache_key(), self.period)


class Delta(Signal):
    """Difference from N periods ago.

    Only positive periods are allowed to prevent lookahead bias.
    delta(x, 1) = x - delay(x, 1) = today's value minus yesterday's.
    """

    def __init__(self, signal: Signal, period: int):
        if period < 0:
            raise ValueError(
                f"delta() period must be >= 0 (got {period}). "
                "Negative deltas cause lookahead bias."
            )
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

    def __init__(self, signal1: Signal, signal2: Signal, period: int, min_periods: int = None):
        self.signal1 = signal1
        self.signal2 = signal2
        self.period = period
        self.min_periods = _get_min_periods(period, min_periods)

    def _compute(self, data):
        values1 = self.signal1.evaluate(data)
        values2 = self.signal2.evaluate(data)
        return values1.rolling(self.period, min_periods=self.min_periods).corr(values2)

    def _cache_key(self):
        return ('TsCorr', self.signal1._cache_key(), self.signal2._cache_key(), self.period, self.min_periods)


class TsCov(Signal):
    """Rolling covariance between two signals."""

    def __init__(self, signal1: Signal, signal2: Signal, period: int, min_periods: int = None):
        self.signal1 = signal1
        self.signal2 = signal2
        self.period = period
        self.min_periods = _get_min_periods(period, min_periods)

    def _compute(self, data):
        values1 = self.signal1.evaluate(data)
        values2 = self.signal2.evaluate(data)
        return values1.rolling(self.period, min_periods=self.min_periods).cov(values2)

    def _cache_key(self):
        return ('TsCov', self.signal1._cache_key(), self.signal2._cache_key(), self.period, self.min_periods)


class TsVar(Signal):
    """Rolling variance of a signal."""

    def __init__(self, signal: Signal, period: int, min_periods: int = None):
        self.signal = signal
        self.period = period
        self.min_periods = _get_min_periods(period, min_periods)

    def _compute(self, data):
        values = self.signal.evaluate(data)
        return values.rolling(self.period, min_periods=self.min_periods).var()

    def _cache_key(self):
        return ('TsVar', self.signal._cache_key(), self.period, self.min_periods)


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


class EwmaVar(Signal):
    """Exponentially weighted moving variance."""

    def __init__(self, signal: Signal, halflife: int):
        self.signal = signal
        self.halflife = halflife

    def _compute(self, data):
        values = self.signal.evaluate(data)
        return values.ewm(halflife=self.halflife).var()

    def _cache_key(self):
        return ('EwmaVar', self.signal._cache_key(), self.halflife)


class EwmaCov(Signal):
    """Exponentially weighted moving covariance between two signals."""

    def __init__(self, signal1: Signal, signal2: Signal, halflife: int):
        self.signal1 = signal1
        self.signal2 = signal2
        self.halflife = halflife

    def _compute(self, data):
        values1 = self.signal1.evaluate(data)
        values2 = self.signal2.evaluate(data)
        return values1.ewm(halflife=self.halflife).cov(values2)

    def _cache_key(self):
        return ('EwmaCov', self.signal1._cache_key(), self.signal2._cache_key(), self.halflife)


def _paired(y: pd.DataFrame, x: pd.DataFrame):
    """y and x masked to the days where both have a value."""
    both = y.notna() & x.notna()
    return y.where(both), x.where(both)


class TsBeta(Signal):
    """Rolling beta (regression coefficient) of signal1 on signal2.

    Computes: cov(y, x) / var(x) where y=signal1, x=signal2, the OLS slope with an intercept.
    Both moments use only the days where y and x both have a value, so a gap in y (a late
    listing, a halt) doesn't leave var(x) spanning days the covariance never saw.
    Useful for hedge ratios, market beta, factor exposures.
    """

    def __init__(self, signal1: Signal, signal2: Signal, period: int, min_periods: int = None):
        self.signal1 = signal1
        self.signal2 = signal2
        self.period = period
        self.min_periods = _get_min_periods(period, min_periods)

    def _compute(self, data):
        y, x = _paired(self.signal1.evaluate(data), self.signal2.evaluate(data))
        return self._beta(y, x)

    def _beta(self, y, x):
        cov = y.rolling(self.period, min_periods=self.min_periods).cov(x)
        var = x.rolling(self.period, min_periods=self.min_periods).var()
        return cov / var

    def _cache_key(self):
        return ('TsBeta', self.signal1._cache_key(), self.signal2._cache_key(), self.period, self.min_periods)


class TsIntercept(TsBeta):
    """Rolling intercept of signal1 regressed on signal2: mean(y) - ts_beta(y, x) * mean(x).

    The constant in the same window fit as ts_beta, over the same paired days. For stock
    returns on the market this is the daily alpha: the mean return beta doesn't explain.
    The slope is a ts_beta node, so with `let b = ts_beta(...), a = ts_intercept(...)` in a
    compute_context the fit runs once.
    """

    def __init__(self, signal1: Signal, signal2: Signal, period: int, min_periods: int = None):
        super().__init__(signal1, signal2, period, min_periods)
        self.beta = TsBeta(signal1, signal2, period, min_periods)

    def _compute(self, data):
        y, x = _paired(self.signal1.evaluate(data), self.signal2.evaluate(data))
        roll = lambda v: v.rolling(self.period, min_periods=self.min_periods).mean()
        return roll(y) - self.beta.evaluate(data) * roll(x)

    def _cache_key(self):
        return ('TsIntercept', self.signal1._cache_key(), self.signal2._cache_key(), self.period, self.min_periods)


class TsBetaShrunk(Signal):
    """Rolling beta of signal1 on signal2, shrunk toward the cross-sectional mean (Vasicek 1973).

    Each name's OLS beta b over the window is pulled toward the day's cross-sectional mean
    beta in proportion to how noisy it is:

        b_shrunk = w * b + (1 - w) * mean(b),   w = var_cs(b) / (var_cs(b) + se(b)^2)

    where var_cs is the day's cross-sectional variance of the betas and se(b)^2 the name's
    sampling variance, residual variance / ((n - 1) * var(x)), from its n paired observations.
    A name with a short history or a noisy fit moves most of the way to the mean; a long,
    clean history keeps nearly its own beta. Unlike a fixed blend (Blume's 0.67 b + 0.33),
    the weight differs by name, so the ordering of betas changes, not just their scale.

    The mean and variance are over the names with a beta that day, within the cross-sectional
    universe when one is set (see qex.universe). Every name keeps its own shrunk beta.
    """

    def __init__(self, signal1: Signal, signal2: Signal, period: int, min_periods: int = None):
        self.signal1 = signal1
        self.signal2 = signal2
        self.period = period
        self.min_periods = max(_get_min_periods(period, min_periods), 3)

    def _compute(self, data):
        y, x = _paired(self.signal1.evaluate(data), self.signal2.evaluate(data))
        return self._beta(y, x)

    def _beta(self, y, x):
        roll = lambda v: v.rolling(self.period, min_periods=self.min_periods)
        n = roll(x).count()
        vx, vy, cov = roll(x).var(), roll(y).var(), roll(y).cov(x)
        vx = vx.where(vx > 0)
        b = cov / vx
        resid = (vy - cov * b).clip(lower=0) * (n - 1) / (n - 2)
        se2 = resid / ((n - 1) * vx)
        cs = in_universe(b)                                      # the prior comes from the universe
        tau2 = cs.var(axis=1)
        w = (1 / (1 + se2.div(tau2.where(tau2 > 0), axis=0))).where(b.notna())
        return w * b + (1 - w).mul(cs.mean(axis=1), axis=0)

    def _cache_key(self):
        return ('TsBetaShrunk', self.signal1._cache_key(), self.signal2._cache_key(), self.period, self.min_periods)


class TsInterceptShrunk(TsBetaShrunk):
    """Rolling intercept that goes with ts_beta_shrunk: mean(y) - ts_beta_shrunk(y, x) * mean(x).

    Given the shrunk slope, this is the intercept that best fits the window (the least-squares
    line through the means with that slope), so y - a - b * x uses one consistent fit. The
    intercept is not shrunk itself: it moves only by (b - b_shrunk) * mean(x). The slope is a
    ts_beta_shrunk node, shared with any ts_beta_shrunk of the same arguments (see TsIntercept).
    """

    def __init__(self, signal1: Signal, signal2: Signal, period: int, min_periods: int = None):
        super().__init__(signal1, signal2, period, min_periods)
        self.beta = TsBetaShrunk(signal1, signal2, period, min_periods)

    def _compute(self, data):
        y, x = _paired(self.signal1.evaluate(data), self.signal2.evaluate(data))
        roll = lambda v: v.rolling(self.period, min_periods=self.min_periods).mean()
        return roll(y) - self.beta.evaluate(data) * roll(x)

    def _cache_key(self):
        return ('TsInterceptShrunk', self.signal1._cache_key(), self.signal2._cache_key(), self.period, self.min_periods)


class TsBetaEwma(Signal):
    """EWMA-weighted beta (regression coefficient) of signal1 on signal2.

    Computes: ewma_cov(y, x) / ewma_var(x) where y=signal1, x=signal2, on the days where
    both have a value (see TsBeta).
    More robust than equal-weighted rolling beta, gives more weight to recent observations.
    """

    def __init__(self, signal1: Signal, signal2: Signal, halflife: int):
        self.signal1 = signal1
        self.signal2 = signal2
        self.halflife = halflife

    def _compute(self, data):
        y, x = _paired(self.signal1.evaluate(data), self.signal2.evaluate(data))
        cov = y.ewm(halflife=self.halflife).cov(x)
        var = x.ewm(halflife=self.halflife).var()
        return cov / var

    def _cache_key(self):
        return ('TsBetaEwma', self.signal1._cache_key(), self.signal2._cache_key(), self.halflife)


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

        # A window needs all `period` values (rolling's default min_periods), so any NaN in it gives
        # NaN; the full windows are one dot product each with the weights, oldest first.
        v = values.to_numpy(dtype=float)
        out = np.full(v.shape, np.nan)
        if len(v) >= self.period:
            windows = np.lib.stride_tricks.sliding_window_view(v, self.period, axis=0)
            out[self.period - 1:] = windows @ weights
        if isinstance(values, pd.Series):
            return pd.Series(out, index=values.index, name=values.name)
        return pd.DataFrame(out, index=values.index, columns=values.columns)

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


def ts_corr(signal1: Signal, signal2: Signal, period: int, min_periods: int = None) -> TsCorr:
    """Create a TsCorr signal (rolling correlation). min_periods defaults to min(20, period)."""
    return TsCorr(signal1, signal2, period, min_periods)


def ts_cov(signal1: Signal, signal2: Signal, period: int, min_periods: int = None) -> TsCov:
    """Create a TsCov signal (rolling covariance). min_periods defaults to min(20, period)."""
    return TsCov(signal1, signal2, period, min_periods)


def ts_var(signal: Signal, period: int) -> TsVar:
    """Create a TsVar signal (rolling variance)."""
    return TsVar(signal, period)


def ewma(signal: Signal, halflife: int) -> Ewma:
    """Create an Ewma signal (exponentially weighted moving average)."""
    return Ewma(signal, halflife)


def ewma_var(signal: Signal, halflife: int) -> EwmaVar:
    """Create an EwmaVar signal (exponentially weighted moving variance)."""
    return EwmaVar(signal, halflife)


def ewma_cov(signal1: Signal, signal2: Signal, halflife: int) -> EwmaCov:
    """Create an EwmaCov signal (exponentially weighted moving covariance)."""
    return EwmaCov(signal1, signal2, halflife)


def ts_beta(signal1: Signal, signal2: Signal, period: int, min_periods: int = None) -> TsBeta:
    """Create a TsBeta signal (rolling beta: cov(y, x) / var(x)). min_periods defaults to min(20, period)."""
    return TsBeta(signal1, signal2, period, min_periods)


def ts_beta_shrunk(signal1: Signal, signal2: Signal, period: int, min_periods: int = None) -> TsBetaShrunk:
    """Create a TsBetaShrunk signal (rolling beta, Vasicek-shrunk toward the day's cross-sectional mean)."""
    return TsBetaShrunk(signal1, signal2, period, min_periods)


def ts_intercept(signal1: Signal, signal2: Signal, period: int, min_periods: int = None) -> TsIntercept:
    """Create a TsIntercept signal (rolling intercept: mean(y) - ts_beta(y, x) * mean(x))."""
    return TsIntercept(signal1, signal2, period, min_periods)


def ts_intercept_shrunk(signal1: Signal, signal2: Signal, period: int, min_periods: int = None) -> TsInterceptShrunk:
    """Create a TsInterceptShrunk signal (rolling intercept that goes with ts_beta_shrunk)."""
    return TsInterceptShrunk(signal1, signal2, period, min_periods)


def ts_beta_ewma(signal1: Signal, signal2: Signal, halflife: int) -> TsBetaEwma:
    """Create a TsBetaEwma signal (EWMA-weighted beta: ewma_cov / ewma_var)."""
    return TsBetaEwma(signal1, signal2, halflife)


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


# Event-based rolling operations (roll over N non-NaN values, not N rows)

def _apply_event_rolling(values: pd.DataFrame, n_events: int, func) -> pd.DataFrame:
    """Apply a function over the past N non-NaN values for each column."""
    result = pd.DataFrame(index=values.index, columns=values.columns, dtype=float)

    for col in values.columns:
        series = values[col].values
        col_result = np.full(len(series), np.nan)
        valid_values = []

        for i, val in enumerate(series):
            if not np.isnan(val):
                valid_values.append(val)

            if len(valid_values) >= n_events:
                col_result[i] = func(valid_values[-n_events:])

        result[col] = col_result

    return result


class TsMeanEvents(Signal):
    """Rolling mean over past N non-NaN values (events).

    Useful for sparse data like quarterly earnings where you want
    the mean of the past N announcements, not the past N days.
    """

    def __init__(self, signal: Signal, n_events: int):
        self.signal = signal
        self.n_events = n_events

    def _compute(self, data):
        values = self.signal.evaluate(data)
        return _apply_event_rolling(values, self.n_events, np.mean)

    def _cache_key(self):
        return ('TsMeanEvents', self.signal._cache_key(), self.n_events)


class TsStdEvents(Signal):
    """Rolling standard deviation over past N non-NaN values (events).

    Useful for computing volatility of sparse data like earnings surprises
    where you want the std of the past N announcements, not the past N days.
    """

    def __init__(self, signal: Signal, n_events: int):
        self.signal = signal
        self.n_events = n_events

    def _compute(self, data):
        values = self.signal.evaluate(data)
        return _apply_event_rolling(
            values, self.n_events,
            lambda x: np.std(x, ddof=1) if len(x) > 1 else np.nan
        )

    def _cache_key(self):
        return ('TsStdEvents', self.signal._cache_key(), self.n_events)


class TsSumEvents(Signal):
    """Rolling sum over past N non-NaN values (events)."""

    def __init__(self, signal: Signal, n_events: int):
        self.signal = signal
        self.n_events = n_events

    def _compute(self, data):
        values = self.signal.evaluate(data)
        return _apply_event_rolling(values, self.n_events, np.sum)

    def _cache_key(self):
        return ('TsSumEvents', self.signal._cache_key(), self.n_events)


class TsCountEvents(Signal):
    """Count of non-NaN values in a rolling window of rows.

    Different from the other event operations - this counts how many
    events occurred in the past N rows (days), not past N events.
    Useful for measuring event density/frequency.
    """

    def __init__(self, signal: Signal, period: int):
        self.signal = signal
        self.period = period

    def _compute(self, data):
        values = self.signal.evaluate(data)
        # Count non-NaN values in rolling window
        return values.notna().rolling(self.period).sum()

    def _cache_key(self):
        return ('TsCountEvents', self.signal._cache_key(), self.period)


# Factory functions for event-based operations

def ts_mean_events(signal: Signal, n_events: int) -> TsMeanEvents:
    """Create a TsMeanEvents signal (mean over past N non-NaN values)."""
    return TsMeanEvents(signal, n_events)


def ts_std_events(signal: Signal, n_events: int) -> TsStdEvents:
    """Create a TsStdEvents signal (std over past N non-NaN values)."""
    return TsStdEvents(signal, n_events)


def ts_sum_events(signal: Signal, n_events: int) -> TsSumEvents:
    """Create a TsSumEvents signal (sum over past N non-NaN values)."""
    return TsSumEvents(signal, n_events)


def ts_count_events(signal: Signal, period: int) -> TsCountEvents:
    """Create a TsCountEvents signal (count of non-NaN in rolling window)."""
    return TsCountEvents(signal, period)
