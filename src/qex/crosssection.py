"""Cross-sectional operations for signals."""

import numpy as np
import pandas as pd

from .signal import Signal
from .estimate import EstimateOutput, estimate
from .universe import in_universe, finite_in_universe, universe_members


class Rank(Signal):
    """Cross-sectional percentile rank."""

    def __init__(self, signal: Signal):
        self.signal = signal

    def _compute(self, data):
        values = in_universe(self.signal.evaluate(data))
        return values.rank(axis=1, pct=True)

    def _cache_key(self):
        return ('Rank', self.signal._cache_key())


def _normal_scores(frame: pd.DataFrame, codes=None) -> pd.DataFrame:
    """Each name's normal score among the names with a value on its day (and, with codes, in its group that day):
    Phi^-1((r - 0.5) / n) for its rank r from 1 to n. Tied names share the average of the scores they occupy, so each
    day's (or group's) scores still sum to zero."""
    from scipy.special import ndtri
    T, N = frame.shape
    if codes is None:
        # the whole day: vectorized ranks, and the ties (if any) averaged on the days that have them
        first = frame.rank(axis=1, method="first")
        out = ndtri((first - 0.5).div(frame.count(axis=1), axis=0))
        tied = (first != frame.rank(axis=1)) & frame.notna()
        days = np.flatnonzero(tied.any(axis=1).to_numpy())
        if len(days):
            out.iloc[days] = _normal_scores_grouped(frame.iloc[days], None)
        return out
    return _normal_scores_grouped(frame, codes)


def _normal_scores_grouped(frame: pd.DataFrame, codes) -> pd.DataFrame:
    """_normal_scores by grouping a long table on (day, group): exact with ties, slower."""
    from scipy.special import ndtri
    T, N = frame.shape
    v = frame.to_numpy(dtype=float).ravel()
    keep = ~np.isnan(v)
    if codes is not None:
        keep &= (codes >= 0).ravel()
    long = pd.DataFrame({"date": np.repeat(np.arange(T), N)[keep],
                         "group": (np.zeros(T * N, int) if codes is None else codes.ravel())[keep],
                         "value": v[keep]})
    by = long.groupby(["date", "group"])["value"]
    long["score"] = ndtri((by.rank(method="first").to_numpy() - 0.5) / by.transform("count").to_numpy())
    scores = long.groupby(["date", "group", "value"])["score"].transform("mean").to_numpy()
    out = np.full(T * N, np.nan)
    out[keep] = scores
    return pd.DataFrame(out.reshape(T, N), index=frame.index, columns=frame.columns)


class GaussRank(Signal):
    """Cross-sectional rank mapped through the inverse normal: Phi^-1((r - 0.5) / n).

    Each day, over the names in the universe with a value, r is the rank from 1 to n (ties at their
    average rank) and the result the normal quantile at (r - 0.5) / n. A z-score's shape without its
    outliers: the values are those of a normal sample whatever x's distribution, and one extreme name can't
    take the book. Tied names share the average of the scores they occupy, so every day sums to zero. An
    infinity ranks first or last, like rank().
    """

    def __init__(self, signal: Signal):
        self.signal = signal

    def _compute(self, data):
        return _normal_scores(in_universe(self.signal.evaluate(data)))

    def _cache_key(self):
        return ('GaussRank', self.signal._cache_key())


class ZScore(Signal):
    """Cross-sectional z-score (standardization)."""

    def __init__(self, signal: Signal):
        self.signal = signal

    def _compute(self, data):
        values = finite_in_universe(self.signal.evaluate(data))
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
        values = finite_in_universe(self.signal.evaluate(data))
        mean = values.mean(axis=1)
        return values.sub(mean, axis=0)

    def _cache_key(self):
        return ('Demean', self.signal._cache_key())


def rank(signal: Signal) -> Rank:
    """Create a Rank signal."""
    return Rank(signal)


def gaussrank(signal: Signal) -> GaussRank:
    """Create a GaussRank signal: the day's rank mapped through the inverse normal."""
    return GaussRank(signal)


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
        values = finite_in_universe(self.signal.evaluate(data))
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
        values = finite_in_universe(self.signal.evaluate(data))
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


class _XsFitView(EstimateOutput):
    """A named output of the day's cross-sectional fit, estimate('xs_ols', ...): the node is the estimate's, its maths
    the estimator's (qex/fits.py). The class only names it, for its cache key and spelling."""

    y = property(lambda self: self.fit.inputs[0])
    x = property(lambda self: self.fit.inputs[1])
    regressors = property(lambda self: self.fit.inputs[1:])
    more = property(lambda self: self.fit.inputs[2:] or None)


class RegressionNeut(_XsFitView):
    """Cross-sectional residual of y on one or more regressors, with an intercept, separately each day:
    regression_neut(y, x1, ..., xk), which is estimate('xs_ols', 'resid', y, x1, ..., xk).

    One regressor, on each date over the names where both y and x are valid:
        residual = (y - mean(y)) - b * (x - mean(x)),   b = cov(y, x) / var(x)
    so the result has zero mean and zero cross-sectional covariance with x. Names where either is
    NaN are NaN. A day on which x does not vary gets b = 0.

    A day needs at least k + 2 valid names for k regressors; with fewer, the fit's k + 1 numbers
    (intercept and slopes) pass through every point, the residual is rounding noise, and the whole
    day is NaN instead.

    Several regressors, regression_neut(y, x1, x2, ...): one least-squares fit of y on all of them
    at once, over the names where y and every x are valid, so the residual has zero mean and zero
    covariance with each x. This is not the same as nesting, regression_neut(regression_neut(y, x1),
    x2): when x1 and x2 are correlated the second step brings back some of the x1 exposure the first
    removed. Collinear regressors give the least-squares fit of smallest norm, not an error.

    The neutrality holds for the expression itself. A book built from it afterwards (truncation,
    clipping, rescaling) is only approximately neutral to x, and less so the more the cap binds.
    """


class RegressionAlpha(_XsFitView):
    """The intercept of the day's cross-sectional fit (regression_neut's): one number per day, given to every stock;
    regression_alpha(y, x1, ..., xk), which is estimate('xs_ols', 'alpha', ...).

    A property of the day, not of a stock: it is given to every name, in the universe or not, so its history is the
    same for all (NaN for every name on a day with no fit). With today's returns on characteristics known at
    yesterday's close, it is the return of a stock with none of them: on betas, the zero-beta portfolio's return
    (Fama-MacBeth). Its fit is shared with regression_beta on the same y and regressors.
    """


class RegressionBeta(RegressionAlpha):
    """The j-th slope of the day's cross-sectional fit: one number per day, given to every stock;
    regression_beta(y, x1, ..., xk, j), which is estimate('xs_ols', 'betaj', ...).

    With today's returns on characteristics known at yesterday's close, it is the day's return to the j-th
    characteristic, net of the others: a factor return, whose average over time is its premium (Fama-MacBeth).
    """

    j = property(lambda self: int(self.output[4:]))


def regression_alpha(y: Signal, x: Signal, *more: Signal) -> RegressionAlpha:
    """Create a RegressionAlpha signal: the day's intercept of y on the regressors across stocks, for every stock."""
    for s in (x,) + more:
        if not isinstance(s, Signal):
            raise ValueError(f"regression_alpha() takes y and one or more regressors, each an expression, not {s!r}: "
                             f"regression_alpha(y, x1, ..., xk)")
    return estimate('xs_ols', 'alpha', y, x, *more)


def regression_beta(y: Signal, *args) -> RegressionBeta:
    """Create a RegressionBeta signal: the day's slope of y on the j-th regressor across stocks, for every stock,
    regression_beta(y, x1, ..., xk, j); with one regressor j may be left out."""
    usage = "regression_beta(y, x1, ..., xk, j)"
    args = list(args)
    j = None
    if args and not isinstance(args[-1], Signal):
        j = args.pop()
    regs = args
    if not regs or not all(isinstance(r, Signal) for r in regs):
        raise ValueError(f"regression_beta() takes y, one or more regressors, and which slope to give: {usage}")
    if j is None:
        if len(regs) > 1:
            raise ValueError(f"regression_beta() with {len(regs)} regressors needs to know which slope to give, "
                             f"1 to {len(regs)}, as its last argument: {usage}")
        j = 1
    if isinstance(j, bool) or not isinstance(j, (int, float)) or not float(j).is_integer() or not 1 <= j <= len(regs):
        raise ValueError(f"regression_beta()'s last argument says which regressor's slope to give: a whole number "
                         f"from 1 to {len(regs)}, not {j!r} ({usage})")
    return estimate('xs_ols', f'beta{int(j)}', y, *regs)


def regression_neut(y: Signal, x: Signal, *more: Signal) -> RegressionNeut:
    """y with its daily cross-sectional fit on x (and any further regressors) removed:
    estimate('xs_ols', 'resid', y, x, ...)."""
    return estimate('xs_ols', 'resid', y, x, *more)


# --- statistical factors -------------------------------------------------------------------------

class PcaFactor(EstimateOutput):
    """Each stock's beta to the k-th statistical factor of x: a principal component of x across stocks;
    pca_factor(x, window, k, refit), which is estimate('pca', 'betak', x, window=, refit=). The maths is the pca
    estimator's (qex/fits.py); this class only names the output.

    At each refit date (on the calendar, the first once `window` rows exist), qex runs a principal component analysis
    of the trailing `window` rows of x, across the names with values on at least 90% of those days (and, inside a
    universe, in it that day). It is a correlation PCA: each name is demeaned and scaled by its own volatility over the
    window, so a few volatile stocks don't make up the first components. The value is each name's beta to the k-th
    eigenportfolio's return, sigma_i * v_i in x's units, with v the k-th eigenvector of the correlation matrix: the
    exposure a position in that stock carries to the factor. It is held until the next refit.

    Components keep their identity across refits: each refit's components are matched to the previous ones by overlap,
    and signed to agree with them, so k is the k-th largest at the first fit, followed after; the first component is
    signed long the market. A name not in a fit has no value until it enters one.

        regression_neut(r, pca_factor(r, 252, 1), pca_factor(r, 252, 2), pca_factor(r, 252, 3))
                                           r with three statistical factors removed, r = returns(1)

    Every pca_factor on the same x, window and refit reads one fit (one decomposition per refit), in a
    compute_context, in SignalGraph and on Dask. Cost: one SVD of a window x names matrix per refit, about 0.1 s for
    252 x 3,000.
    """

    signal = property(lambda self: self.fit.inputs[0])
    window = property(lambda self: self.fit.window)
    refit = property(lambda self: self.fit.refit)
    k = property(lambda self: int(self.output[4:]))
    basis = property(lambda self: self.fit)


def pca_factor(signal: Signal, window: int, k: int, refit: int = 21) -> PcaFactor:
    """Each stock's beta to the k-th principal component of signal over a trailing window, refit every `refit` days
    (default 21) and held in between: estimate('pca', 'betak', signal, window=window, refit=refit)."""
    from .fits import PCA_MAX_COMPONENTS
    for name, val, low in (("window", window, 3), ("k", k, 1), ("refit", refit, 1)):
        if isinstance(val, bool) or not isinstance(val, int) or val < low:
            raise ValueError(f"pca_factor() {name} must be a whole number of at least {low} (got {val!r}).")
    if k > PCA_MAX_COMPONENTS:
        raise ValueError(f"pca_factor() keeps the first {PCA_MAX_COMPONENTS} components; k = {k} is beyond them.")
    return estimate('pca', f'beta{k}', signal, window=window, refit=refit)
