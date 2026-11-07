"""Cross-sectional operations for signals."""

from .signal import Signal
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


class RegressionNeut(Signal):
    """Cross-sectional residual of y on one or more regressors, with an intercept, separately each day.

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

    def __init__(self, y: Signal, x: Signal, *more: Signal):
        self.y = y
        self.x = x
        self.more = tuple(more) if more else None        # regressors after the first; None keeps the one-x form

    def _compute(self, data):
        if self.more:
            return self._compute_multiple(data)
        y = finite_in_universe(self.y.evaluate(data))
        x = finite_in_universe(self.x.evaluate(data).reindex_like(y))
        valid = y.notna() & x.notna()
        valid = valid & (valid.sum(axis=1) >= 3).to_numpy()[:, None]          # k + 2 names, k = 1
        y, x = y.where(valid), x.where(valid)
        yc = y.sub(y.mean(axis=1), axis=0)
        xc = x.sub(x.mean(axis=1), axis=0)
        var = (xc * xc).sum(axis=1)
        b = ((yc * xc).sum(axis=1) / var.where(var > 0)).fillna(0.0)
        return yc - xc.mul(b, axis=0)

    def _compute_multiple(self, data):
        import numpy as np
        import pandas as pd
        y = finite_in_universe(self.y.evaluate(data))
        xs = [finite_in_universe(s.evaluate(data).reindex_like(y)) for s in (self.x,) + self.more]
        Y = y.to_numpy(dtype=float)
        X = np.stack([x.to_numpy(dtype=float) for x in xs], axis=2)          # dates x names x regressors
        valid = np.isfinite(Y) & np.isfinite(X).all(axis=2)
        out = np.full(Y.shape, np.nan)
        for t in range(Y.shape[0]):
            ok = valid[t]
            if ok.sum() < X.shape[2] + 2:                # an exact fit: the residual would be rounding noise
                continue
            yc = Y[t, ok] - Y[t, ok].mean()
            A = X[t, ok, :] - X[t, ok, :].mean(axis=0)
            beta = np.linalg.lstsq(A, yc, rcond=None)[0]
            out[t, ok] = yc - A @ beta
        return pd.DataFrame(out, index=y.index, columns=y.columns)

    def _cache_key(self):
        more = tuple(s._cache_key() for s in self.more) if self.more else ()
        return ('RegressionNeut', self.y._cache_key(), self.x._cache_key()) + more


def regression_neut(y: Signal, x: Signal, *more: Signal) -> RegressionNeut:
    """Create a RegressionNeut signal: y with its daily cross-sectional fit on x (and any further regressors) removed."""
    return RegressionNeut(y, x, *more)


# --- statistical factors -------------------------------------------------------------------------

PCA_MAX_COMPONENTS = 20          # components kept at each refit; pca_factor's k is at most this
PCA_MIN_COVERAGE = 0.9           # a name enters a fit with values on at least this share of the window's days


def _pca_fits(values, window: int, refit: int, members):
    """The principal components of values (dates x names) refit every `refit` rows on the trailing `window` rows.

    Returns [(position, names, betas)], one per refit position with a fit, in date order; betas is an m x n array,
    row j the names' betas to the j-th eigenportfolio. Correlation PCA: each name is demeaned and divided by its
    volatility over the window, missing days count as its mean. Components are tracked across refits (matched to the
    previous fit's by overlap on the names both share) and signed to agree with it; the first fit is signed so that
    its betas sum to a positive number (the first component is the market, long).
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    v = values.to_numpy(dtype=float)
    v = np.where(np.isfinite(v), v, np.nan)
    T, N = v.shape
    fits, prev = [], None                              # prev: (names, unit eigenvectors) of the last fit
    for p in range(window - 1, T, refit):
        w = v[p - window + 1:p + 1]
        ok = np.isfinite(w).sum(axis=0) >= np.ceil(PCA_MIN_COVERAGE * window)
        if members is not None:
            ok &= members[p]
        names = np.flatnonzero(ok)
        if names.size < 3:
            continue
        z = w[:, names]
        mean = np.nanmean(z, axis=0)
        sd = np.nanstd(z, axis=0, ddof=1)
        keep = np.isfinite(sd) & (sd > 0)
        names, z, mean, sd = names[keep], z[:, keep], mean[keep], sd[keep]
        if names.size < 3:
            continue
        z = np.nan_to_num((z - mean) / sd)
        _, _, vt = np.linalg.svd(z, full_matrices=False)
        comps = vt[:min(PCA_MAX_COMPONENTS, vt.shape[0])]
        if prev is not None:
            common, ip, inew = np.intersect1d(prev[0], names, return_indices=True)
            if common.size >= 2:
                overlap = prev[1][:, ip] @ comps[:, inew].T                       # previous x new
                rows, cols = linear_sum_assignment(-np.abs(overlap))             # rows: previous slots, sorted
                order = list(cols) + [j for j in range(comps.shape[0]) if j not in set(cols)]   # unmatched: by variance
                signs = np.ones(comps.shape[0])
                signs[:len(rows)] = np.where(overlap[rows, cols] < 0, -1.0, 1.0)
                comps = comps[order] * signs[:, None]
            else:
                prev = None
        if prev is None:
            s = np.sign(comps.sum(axis=1))
            comps = comps * np.where(s == 0, 1, s)[:, None]
        prev = (names, comps)
        fits.append((p, names, comps * sd))                                       # beta of x_i: sigma_i * v_i
    return fits


class PcaFactor(Signal):
    """Each stock's beta to the k-th statistical factor of x: a principal component of x across stocks.

    At each refit date (every `refit` rows, the first once `window` rows exist), qex runs a principal component analysis
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

    Every pca_factor on the same x, window and refit shares one decomposition per refit in a compute_context.
    Cost: one SVD of a window x names matrix per refit, about 0.1 s for 252 x 3,000.
    """

    def __init__(self, signal: Signal, window: int, k: int, refit: int = 21):
        for name, val, low in (("window", window, 3), ("k", k, 1), ("refit", refit, 1)):
            if isinstance(val, bool) or not isinstance(val, int) or val < low:
                raise ValueError(f"pca_factor() {name} must be a whole number of at least {low} (got {val!r}).")
        if k > PCA_MAX_COMPONENTS:
            raise ValueError(f"pca_factor() keeps the first {PCA_MAX_COMPONENTS} components; k = {k} is beyond them.")
        self.signal = signal
        self.window = window
        self.k = k
        self.refit = refit

    def _basis(self, data):
        from .context import get_context
        from .universe import universe_token
        values = self.signal.evaluate(data)
        members = universe_members(values.index, values.columns)
        ctx = get_context()
        key = ('PcaBasis', self.signal._cache_key(), self.window, self.refit, universe_token())
        if ctx is not None:
            ctx.check(data)
            if key not in ctx.cache:
                ctx.cache[key] = _pca_fits(values, self.window, self.refit, members)
            return values, ctx.cache[key]
        return values, _pca_fits(values, self.window, self.refit, members)

    def _compute(self, data):
        import numpy as np
        import pandas as pd
        values, fits = self._basis(data)
        T, N = values.shape
        out = np.full((T, N), np.nan)
        for i, (p, names, betas) in enumerate(fits):
            end = fits[i + 1][0] if i + 1 < len(fits) else T
            if self.k <= betas.shape[0]:
                out[p:end, names] = betas[self.k - 1]
        frame = pd.DataFrame(out, index=values.index, columns=values.columns)
        return in_universe(frame)

    def _cache_key(self):
        return ('PcaFactor', self.signal._cache_key(), self.window, self.k, self.refit)


def pca_factor(signal: Signal, window: int, k: int, refit: int = 21) -> PcaFactor:
    """Create a PcaFactor signal: each stock's beta to the k-th principal component of signal over a trailing window,
    refit every `refit` days (default 21) and held in between."""
    return PcaFactor(signal, window, k, refit)
