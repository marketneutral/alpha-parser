"""Time-series operations for signals."""

from typing import Optional

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
    """Time-series rank: the percentile of today's value within the last ``period`` values.

    ``(average rank of today's value among the window) / period``, with ties given their average
    rank. The window must hold ``period`` valid values, and at least two; otherwise the result
    is NaN (so ``period`` 1 is always NaN).
    """

    def __init__(self, signal: Signal, period: int):
        self.signal = signal
        self.period = period

    def _compute(self, data):
        values = self.signal.evaluate(data)
        if self.period < 2:
            return values.astype(float) * np.nan
        return values.rolling(self.period).rank(method='average', pct=True)

    def _cache_key(self):
        return ('TsRank', self.signal._cache_key(), self.period)


class FillForward(Signal):
    """Forward fill missing values for up to N periods (with no limit, until the next value)."""

    def __init__(self, signal: Signal, limit: Optional[int] = None):
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


def _paired(y, x):
    """y and x with every observation where either is missing set to NaN in both.

    A beta's covariance and variance must come from the same days. Estimating the variance of x
    on days y is missing (before a listing, during a halt) biases the beta.
    """
    if isinstance(x, pd.Series) and isinstance(y, pd.DataFrame):
        x = pd.DataFrame({c: x for c in y.columns}, index=y.index)
    elif isinstance(y, pd.Series) and isinstance(x, pd.DataFrame):
        y = pd.DataFrame({c: y for c in x.columns}, index=x.index)
    both = y.notna() & x.notna()
    return y.where(both), x.where(both)


class TsBeta(Signal):
    """Rolling beta (regression coefficient) of signal1 on signal2.

    Computes: cov(y, x) / var(x) where y=signal1, x=signal2, both over the days in the window
    where y and x are both valid (paired observations, as least squares does).
    Useful for hedge ratios, market beta, factor exposures.
    """

    def __init__(self, signal1: Signal, signal2: Signal, period: int, min_periods: int = None):
        self.signal1 = signal1
        self.signal2 = signal2
        self.period = period
        self.min_periods = _get_min_periods(period, min_periods)

    def _compute(self, data):
        y, x = _paired(self.signal1.evaluate(data), self.signal2.evaluate(data))
        cov = y.rolling(self.period, min_periods=self.min_periods).cov(x)
        var = x.rolling(self.period, min_periods=self.min_periods).var()
        return cov / var

    def _cache_key(self):
        return ('TsBeta', self.signal1._cache_key(), self.signal2._cache_key(), self.period, self.min_periods)


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

    With groups (a group's name or any labels expression), the prior is each group's: a stock's beta
    is pulled toward the mean beta of its own group that day, by that group's spread of betas. A
    stock with no group has no value; a group with fewer than two betas that day has no spread to
    shrink by, so its stocks keep their own betas.
    """

    def __init__(self, signal1: Signal, signal2: Signal, period: int, min_periods: int = None, groups=None):
        self.signal1 = signal1
        self.signal2 = signal2
        self.period = period
        self.min_periods = max(_get_min_periods(period, min_periods), 3)
        self.groups = groups

    def _compute(self, data):
        y = self.signal1.evaluate(data)
        x = self.signal2.evaluate(data)
        both = y.notna() & x.notna()
        y, x = y.where(both), x.where(both)                      # paired observations only
        roll = lambda v: v.rolling(self.period, min_periods=self.min_periods)
        n = roll(both.astype(float).where(both)).count()
        vx, vy, cov = roll(x).var(), roll(y).var(), roll(y).cov(x)
        vx = vx.where(vx > 0)
        b = cov / vx
        resid = (vy - cov * b).clip(lower=0) * (n - 1) / (n - 2)
        se2 = resid / ((n - 1) * vx)
        if self.groups is not None:
            return _shrink_within_groups(b, se2, self.groups, data)
        cs = in_universe(b)                                    # the prior comes from the universe
        tau2 = cs.var(axis=1)
        w = (1 / (1 + se2.div(tau2.where(tau2 > 0), axis=0))).where(b.notna())
        return w * b + (1 - w).mul(cs.mean(axis=1), axis=0)

    def _cache_key(self):
        key = ('TsBetaShrunk', self.signal1._cache_key(), self.signal2._cache_key(), self.period, self.min_periods)
        if self.groups is None:
            return key                                          # unchanged since 0.5: expressions are hashed by it
        from .groups import _groups_key
        return key + (_groups_key(self.groups),)


def _shrink_within_groups(b: pd.DataFrame, se2: pd.DataFrame, groups, data) -> pd.DataFrame:
    """Vasicek shrinkage with each group's prior: the mean and variance of the day's betas within the stock's group
    (members of the universe only). No group: no value. A group with fewer than two betas: the stock's own beta."""
    from .groups import _get_group_data, _group_codes, _group_totals
    cs = in_universe(b)
    codes, n_groups = _group_codes(cs, _get_group_data(data, groups))
    v = cs.to_numpy(dtype=float)
    key, sums, counts = _group_totals(v, codes, n_groups)
    ok = (codes >= 0) & ~np.isnan(v)
    with np.errstate(invalid='ignore', divide='ignore'):
        means = sums / counts
        dev = np.zeros(v.shape)
        dev[ok] = (v[ok] - means[key[ok]]) ** 2
        tau2 = np.bincount(key[ok], weights=dev[ok], minlength=sums.size) / np.where(counts >= 2, counts - 1, np.nan)
    has = codes >= 0
    prior_mean, prior_var = np.full(v.shape, np.nan), np.full(v.shape, np.nan)
    prior_mean[has], prior_var[has] = means[key[has]], tau2[key[has]]
    beta, s2 = b.to_numpy(dtype=float), se2.to_numpy(dtype=float)
    with np.errstate(invalid='ignore', divide='ignore'):
        w = 1 / (1 + s2 / np.where(prior_var > 0, prior_var, np.nan))
        out = w * beta + (1 - w) * prior_mean
    out = np.where(has & ~(prior_var > 0), beta, out)             # no spread in the group: nothing to shrink by
    out[~has] = np.nan
    return pd.DataFrame(out, index=b.index, columns=b.columns)


def _paired_means(y, x, period: int, min_periods: int):
    """y and x on their paired days, and each one's rolling mean over those days."""
    y, x = _paired(y, x)
    roll = lambda v: v.rolling(period, min_periods=min_periods).mean()
    return y, x, roll(y), roll(x)


def _multi_fit(y: pd.DataFrame, xs: list, period: int, min_periods: int, chunk: int = 256):
    """Rolling least squares of y on several regressors and an intercept, per name, over each window's days where y
    and every x have a value: (y, the x's, alpha, [beta_j]) on those days, NaN where a window has fewer than
    max(min_periods, k + 2) such days. Regressors that duplicate each other get the minimum-norm fit."""
    xs = [x if isinstance(x, pd.DataFrame) else pd.DataFrame({c: x for c in y.columns}, index=y.index) for x in xs]
    xs = [x.reindex_like(y) for x in xs]
    k = len(xs)
    ok = y.notna().to_numpy().copy()
    for x in xs:
        ok &= x.notna().to_numpy()
    Y = np.where(ok, y.to_numpy(dtype=float), 0.0)
    X = np.stack([np.where(ok, x.to_numpy(dtype=float), 0.0) for x in xs], axis=2)          # dates x names x k
    roll = lambda a: pd.DataFrame(a.reshape(a.shape[0], -1)).rolling(period, min_periods=1).sum().to_numpy() \
        .reshape(a.shape)
    n = roll(ok.astype(float))
    sy, sx = roll(Y), roll(X)
    sxx = roll(X[:, :, :, None] * X[:, :, None, :])
    sxy = roll(X * Y[:, :, None])
    T, N = Y.shape
    alpha, betas = np.full((T, N), np.nan), np.full((T, N, k), np.nan)
    enough = n >= max(min_periods, k + 2)
    for lo in range(0, T, chunk):                                       # bounded memory: a few hundred dates at a time
        hi = min(lo + chunk, T)
        e = enough[lo:hi]
        if not e.any():
            continue
        nn = n[lo:hi][e]
        mx, my = sx[lo:hi][e] / nn[:, None], sy[lo:hi][e] / nn
        cxx = sxx[lo:hi][e] - nn[:, None, None] * mx[:, :, None] * mx[:, None, :]
        cxy = sxy[lo:hi][e] - nn[:, None] * mx * my[:, None]
        diag = np.abs(np.diagonal(cxx, axis1=1, axis2=2)).prod(axis=1)
        good = np.abs(np.linalg.det(cxx)) > 1e-10 * np.where(diag > 0, diag, np.inf)
        b = np.empty_like(cxy)
        if good.any():
            b[good] = np.linalg.solve(cxx[good], cxy[good][:, :, None])[:, :, 0]
        if (~good).any():                                               # duplicated regressors: the minimum-norm fit
            b[~good] = (np.linalg.pinv(cxx[~good]) @ cxy[~good][:, :, None])[:, :, 0]
        a = np.full((hi - lo, N), np.nan)
        a[e] = my - (b * mx).sum(axis=1)
        alpha[lo:hi] = a
        bb = np.full((hi - lo, N, k), np.nan)
        bb[e] = b
        betas[lo:hi] = bb
    yv = y.where(ok)
    xv = [x.where(ok) for x in xs]
    return yv, xv, pd.DataFrame(alpha, index=y.index, columns=y.columns), \
        [pd.DataFrame(betas[:, :, j], index=y.index, columns=y.columns) for j in range(k)]


class TsAlpha(Signal):
    """Rolling intercept of signal1 on signal2: the alpha of y = alpha + beta * x over the window.

    Fitted on the days both have a value, like ts_beta: alpha = mean(y) - beta * mean(x), both means over
    those paired days. Part of the window, with ts_beta's minimum. In y's units per day: for daily returns
    on the market, the daily alpha (times 252 for a year).

    With more regressors, ts_alpha(y, x1, x2, x3, period), it is one joint least-squares fit of
    y = alpha + b1 x1 + b2 x2 + b3 x3 over each window's days where y and every x have a value (at least
    k + 2 of them): a multi-factor alpha, net of all the regressors at once.
    """

    def __init__(self, signal1: Signal, signal2: Signal, period: int, min_periods: int = None, *more: Signal):
        self.signal1 = signal1
        self.signal2 = signal2
        self.period = period
        self.min_periods = _get_min_periods(period, min_periods)
        self.more = tuple(more) if more else None               # regressors after the first; None keeps the one-x form

    def beta(self) -> Signal:
        """The slope this fit uses; evaluated in the same compute context, so a ts_beta written alongside is shared."""
        return TsBeta(self.signal1, self.signal2, self.period, self.min_periods)

    def _fit(self, data):
        """(y, [x], [beta], alpha) on the fit's days."""
        if getattr(self, "more", None):
            y, xs, a, bs = _multi_fit(self.signal1.evaluate(data),
                                      [s.evaluate(data) for s in (self.signal2,) + self.more],
                                      self.period, self.min_periods)
            return y, xs, bs, a
        b = self.beta().evaluate(data)
        y, x, my, mx = _paired_means(self.signal1.evaluate(data), self.signal2.evaluate(data),
                                     self.period, self.min_periods)
        return y, [x], [b], my - b * mx

    def _compute(self, data):
        return self._fit(data)[3]

    def _cache_key(self):
        key = (type(self).__name__, self.signal1._cache_key(), self.signal2._cache_key(), self.period,
               self.min_periods)
        more = getattr(self, "more", None)
        return key + tuple(s._cache_key() for s in more) if more else key


class TsAlphaShrunk(TsAlpha):
    """The intercept that goes with ts_beta_shrunk's slope: alpha = mean(y) - beta_shrunk * mean(x), over the paired
    days. For a fixed slope this is the least-squares intercept. It differs from ts_alpha by
    (beta - beta_shrunk) * mean(x): what the shrinkage takes off the beta, the alpha picks up.

    With groups, the beta is shrunk toward each group's mean, as in ts_beta_shrunk(y, x, period, groups).
    """

    def __init__(self, signal1: Signal, signal2: Signal, period: int, min_periods: int = None, groups=None):
        self.signal1 = signal1
        self.signal2 = signal2
        self.period = period
        self.min_periods = max(_get_min_periods(period, min_periods), 3)
        self.groups = groups

    def beta(self) -> Signal:
        return TsBetaShrunk(self.signal1, self.signal2, self.period, self.min_periods, self.groups)

    def _cache_key(self):
        key = super()._cache_key()
        if self.groups is None:
            return key
        from .groups import _groups_key
        return key + (_groups_key(self.groups),)


class TsResid(TsAlpha):
    """Today's residual of y on x from the window's fit: y - alpha - beta * x, with alpha and beta from ts_alpha and
    ts_beta over the window ending today. The part of today's y that x doesn't explain: for a stock's return on the
    market's, its idiosyncratic return. No value where y or x is missing today.
    """

    def _compute(self, data):
        y, xs, bs, a = self._fit(data)
        out = y - a
        for x, b in zip(xs, bs):
            out = out - b * x
        return out


class TsResidShrunk(TsAlphaShrunk):
    """Today's residual from ts_beta_shrunk's fit: y - alpha_shrunk - beta_shrunk * x. With groups, the beta is
    shrunk toward each group's mean."""

    _compute = TsResid._compute


class TsBetaEwma(Signal):
    """EWMA-weighted beta (regression coefficient) of signal1 on signal2.

    Computes: ewma_cov(y, x) / ewma_var(x) where y=signal1, x=signal2, both over the days
    where y and x are both valid (paired observations).
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


def fill_forward(signal: Signal, limit: Optional[int] = None) -> FillForward:
    """Carry each value forward over missing days: for up to limit days, or with no limit until the next value."""
    return FillForward(signal, limit)


class Latch(Signal):
    """1 from a day on is true until a day off is true, then 0 until on is true again; 0 before on is first true.

    The state changes only on a day with an answer: a day where on or off is missing changes nothing. On a day both
    are true, off wins. Hysteresis is a latch, enter at one line and leave at a lower one:
    latch(mcap >= 300e6, mcap < 240e6).
    """

    def __init__(self, on: Signal, off: Signal):
        from .conditional import _ensure_signal
        self.on = _ensure_signal(on)
        self.off = _ensure_signal(off)

    def _compute(self, data):
        import numpy as np
        on, off = self.on.evaluate(data), self.off.evaluate(data)
        frames = [x for x in (on, off) if isinstance(x, (pd.DataFrame, pd.Series))]
        if not frames:
            raise ValueError("latch needs a value per day: on and off cannot both be constants")
        like = frames[0]
        on = on if isinstance(on, (pd.DataFrame, pd.Series)) else pd.DataFrame(on, index=like.index, columns=like.columns) if isinstance(like, pd.DataFrame) else pd.Series(on, index=like.index)
        off = off if isinstance(off, (pd.DataFrame, pd.Series)) else on * 0 + off
        off = off.reindex_like(on)
        state = pd.DataFrame(np.nan, index=on.index, columns=on.columns) if isinstance(on, pd.DataFrame) else pd.Series(np.nan, index=on.index)
        state = state.mask(on.notna() & (on != 0), 1.0)
        state = state.mask(off.notna() & (off != 0), 0.0)          # off wins
        return state.ffill().fillna(0.0)

    def _cache_key(self):
        return ('Latch', self.on._cache_key(), self.off._cache_key())


def latch(on: Signal, off: Signal) -> Latch:
    """1 from a day on is true until a day off is true (off wins a tie), 0 before; missing days change nothing."""
    return Latch(on, off)


class TsQuantile(Signal):
    """Rolling q-quantile of a signal over its last ``period`` values.

    Linear interpolation between the two nearest values (numpy's default): q = 0.5 is the median,
    q = 0 and q = 1 are ts_min and ts_max. Missing values are skipped; the window needs at least
    ``min_periods`` of them (default min(20, period)), as the other rolling operators do.

        ts_quantile(returns(1), 252, 0.05)                              the 5% value-at-risk, as a return
        ts_quantile(x, 60, 0.75) - ts_quantile(x, 60, 0.25)              an interquartile range
    """

    def __init__(self, signal: Signal, period: int, q: float, min_periods: int = None):
        if not 0 <= q <= 1:
            raise ValueError(f"ts_quantile() q must be between 0 and 1 (got {q}).")
        if period < 1:
            raise ValueError(f"ts_quantile() period must be >= 1 (got {period}).")
        self.signal = signal
        self.period = period
        self.q = float(q)
        self.min_periods = _get_min_periods(period, min_periods)

    def _compute(self, data):
        values = self.signal.evaluate(data)
        return values.rolling(self.period, min_periods=self.min_periods).quantile(self.q)

    def _cache_key(self):
        return ('TsQuantile', self.signal._cache_key(), self.period, self.q, self.min_periods)


def ts_quantile(signal: Signal, period: int, q: float, min_periods: int = None) -> TsQuantile:
    """Create a TsQuantile signal: the rolling q-quantile (0.5 is the median). min_periods defaults to min(20, period)."""
    return TsQuantile(signal, period, q, min_periods)


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


def _only_constants(s) -> bool:
    """Whether an expression is arithmetic on written numbers only, such as 252 / 2."""
    from .operators import Constant
    from .missing import children
    if isinstance(s, Constant):
        return not isinstance(s.value, str)
    kids = children(s)
    return bool(kids) and all(_only_constants(k) for k in kids)


def _shrunk_args(name: str, rest: tuple):
    """min_periods and groups from the arguments after the period: (y, x, period[, min_periods][, groups]).

    A written number is min_periods; anything else is the groups. So that the two can't be confused, min_periods
    must be written as a number (126, not 252 / 2), and groups come after it."""
    from .groups import _as_groups
    usage = f"{name}(y, x, period[, min_periods][, groups])"
    rest = list(rest)
    min_periods = groups = None
    if rest and isinstance(rest[0], (int, float)) and not isinstance(rest[0], bool):
        min_periods = rest.pop(0)
    if rest:
        groups = rest.pop(0)
        if isinstance(groups, Signal) and _only_constants(groups):
            raise ValueError(f"{name}()'s min_periods must be written as a whole number, such as 126, not as "
                             f"arithmetic: a number is min_periods and anything else is the groups ({usage})")
        if isinstance(groups, (int, float)):
            raise ValueError(f"{name}()'s last argument is the groups, a group's name ('sector') or an expression "
                             f"giving labels, not {groups!r}: {usage}")
        groups = _as_groups(groups)
    if rest:
        if min_periods is None and isinstance(rest[0], (int, float)):
            raise ValueError(f"{name}()'s min_periods comes before its groups: write {name}(y, x, period, "
                             f"{rest[0]!r}, {groups!r}) ({usage})" if isinstance(groups, str) else
                             f"{name}()'s min_periods comes before its groups ({usage})")
        raise ValueError(f"{name}() takes y, x, a period, and optionally min_periods and groups ({usage})")
    return min_periods, groups


def ts_beta_shrunk(signal1: Signal, signal2: Signal, period: int, *rest) -> TsBetaShrunk:
    """Create a TsBetaShrunk signal: the rolling beta, Vasicek-shrunk toward the day's mean beta, or with groups toward
    each group's: ts_beta_shrunk(y, x, period[, min_periods][, groups])."""
    return TsBetaShrunk(signal1, signal2, period, *_shrunk_args("ts_beta_shrunk", rest))


def _fit_args(name: str, y, args: tuple):
    """(regressors, period, min_periods) from (y, x1, ..., xk, period[, min_periods]), each checked."""
    regs = []
    args = list(args)
    while args and isinstance(args[0], Signal):
        regs.append(args.pop(0))
    usage = f"{name}(y, x1, ..., xk, period[, min_periods])"
    if not regs:
        raise ValueError(f"{name}() needs at least one regressor after y: {usage}")
    if not args or len(args) > 2:
        raise ValueError(f"{name}() takes y, one or more regressors, a period and optionally min_periods: {usage}")
    for what, v in zip(("period", "min_periods"), args):
        if isinstance(v, str):
            raise ValueError(f"{name}() takes no groups ({v!r}): a group prior shrinks a beta, so it belongs to "
                             f"{name}_shrunk(y, x, period[, min_periods][, groups])")
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not float(v).is_integer() or v < 1:
            raise ValueError(f"{name}()'s {what} must be a whole number, at least 1 (got {v!r}); "
                             f"the regressors come first: {usage}")
    return regs, args[0], (args[1] if len(args) > 1 else None)


def ts_alpha(signal1: Signal, *args) -> TsAlpha:
    """Create a TsAlpha signal: the rolling intercept of y on one or more regressors,
    ts_alpha(y, x1, ..., xk, period[, min_periods]). With one, mean(y) - beta * mean(x) over the paired days."""
    regs, period, min_periods = _fit_args("ts_alpha", signal1, args)
    return TsAlpha(signal1, regs[0], period, min_periods, *regs[1:])


def ts_alpha_shrunk(signal1: Signal, signal2: Signal, period: int, *rest) -> TsAlphaShrunk:
    """Create a TsAlphaShrunk signal: the intercept that goes with ts_beta_shrunk's slope,
    ts_alpha_shrunk(y, x, period[, min_periods][, groups])."""
    return TsAlphaShrunk(signal1, signal2, period, *_shrunk_args("ts_alpha_shrunk", rest))


def ts_resid(signal1: Signal, *args) -> TsResid:
    """Create a TsResid signal: today's y - alpha - sum(beta_j * x_j), from the window's fit,
    ts_resid(y, x1, ..., xk, period[, min_periods])."""
    regs, period, min_periods = _fit_args("ts_resid", signal1, args)
    return TsResid(signal1, regs[0], period, min_periods, *regs[1:])


def ts_resid_shrunk(signal1: Signal, signal2: Signal, period: int, *rest) -> TsResidShrunk:
    """Create a TsResidShrunk signal: today's residual from ts_beta_shrunk's fit,
    ts_resid_shrunk(y, x, period[, min_periods][, groups])."""
    return TsResidShrunk(signal1, signal2, period, *_shrunk_args("ts_resid_shrunk", rest))


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

def _event_minimum(n_events, min_events):
    """The fewest events a window needs: all n_events unless min_events says fewer."""
    if isinstance(n_events, bool) or not isinstance(n_events, int) or n_events < 1:
        raise ValueError(f"the number of events is a whole number, at least 1 (got {n_events!r})")
    if min_events is None:
        return n_events
    if isinstance(min_events, bool) or not isinstance(min_events, int) or not 1 <= min_events <= n_events:
        raise ValueError(f"the minimum number of events is a whole number from 1 to {n_events} (got {min_events!r})")
    return min_events


def _apply_event_rolling(values: pd.DataFrame, n_events: int, how: str, min_events: Optional[int] = None) -> pd.DataFrame:
    """how ('mean', 'std' or 'sum') over each column's past n_events non-NaN values, once it has min_events of them
    (default n_events). A day without an event carries the window as of the last event."""
    m = _event_minimum(n_events, min_events)
    out = {}
    for col in values.columns:
        s = values[col].dropna()
        r = getattr(s.rolling(n_events, min_periods=m), how)()
        out[col] = r.reindex(values.index).ffill()
    return pd.DataFrame(out, index=values.index, columns=values.columns, dtype=float)


class _EventWindow(Signal):
    """A statistic over the past n_events non-NaN values (events), once there are min_events (default all)."""
    _how = None

    def __init__(self, signal: Signal, n_events: int, min_events: Optional[int] = None):
        _event_minimum(n_events, min_events)
        self.signal = signal
        self.n_events = n_events
        self.min_events = min_events

    def _compute(self, data):
        return _apply_event_rolling(self.signal.evaluate(data), self.n_events, self._how, self.min_events)

    def _cache_key(self):
        key = (type(self).__name__, self.signal._cache_key(), self.n_events)
        return key if self.min_events is None else key + (self.min_events,)


class TsMeanEvents(_EventWindow):
    """Rolling mean over past N non-NaN values (events).

    Useful for sparse data like quarterly earnings where you want
    the mean of the past N announcements, not the past N days.
    """
    _how = "mean"


class TsStdEvents(_EventWindow):
    """Rolling standard deviation (ddof 1) over past N non-NaN values (events).

    Useful for computing volatility of sparse data like earnings surprises
    where you want the std of the past N announcements, not the past N days.
    """
    _how = "std"


class TsSumEvents(_EventWindow):
    """Rolling sum over past N non-NaN values (events)."""
    _how = "sum"


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

def ts_mean_events(signal: Signal, n_events: int, min_events: Optional[int] = None) -> TsMeanEvents:
    """The mean of the past n_events non-NaN values, once there are min_events of them (default n_events)."""
    return TsMeanEvents(signal, n_events, min_events)


def ts_std_events(signal: Signal, n_events: int, min_events: Optional[int] = None) -> TsStdEvents:
    """The std of the past n_events non-NaN values, once there are min_events of them (default n_events)."""
    return TsStdEvents(signal, n_events, min_events)


def ts_sum_events(signal: Signal, n_events: int, min_events: Optional[int] = None) -> TsSumEvents:
    """The sum of the past n_events non-NaN values, once there are min_events of them (default n_events)."""
    return TsSumEvents(signal, n_events, min_events)


def ts_count_events(signal: Signal, period: int) -> TsCountEvents:
    """Create a TsCountEvents signal (count of non-NaN in rolling window)."""
    return TsCountEvents(signal, period)
