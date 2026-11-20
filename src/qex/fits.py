"""The built-in estimators: the maths behind ts_beta, ts_alpha, ts_resid (ols), their shrunk versions (ols_shrunk),
regression_neut, regression_alpha, regression_beta (xs_ols) and pca_factor (pca). Learned estimators are written
outside qex, to the same protocol, and registered by importing their package.

Each named operator is a spelling of estimate(...) (see estimate.py): it calls estimate, and its node is an output of
one of these estimators. The maths here is the maths those operators always had, moved, not rewritten: the same
numbers, and each output keeps its operator's cache key (output_key) and spelling (spell), so an expression's
identity is unchanged.
"""
import numpy as np
import pandas as pd

from .estimate import Estimator, register_estimator
from .schedule import refit_rows
from .signal import Signal
from .timeseries import _get_min_periods, _paired
from .universe import finite_in_universe, in_universe, universe_members


def _groups_key(groups):
    from .groups import _groups_key as key
    return key(groups)


def _whole(v, lo=1) -> bool:
    return not isinstance(v, bool) and isinstance(v, (int, float)) and float(v).is_integer() and v >= lo


def _beta_outputs(n_inputs: int, prefix: tuple) -> tuple:
    return prefix + tuple(f"beta{j}" for j in range(1, n_inputs))


def _keys(signals) -> tuple:
    return tuple(s._cache_key() for s in signals)


# --- rolling least squares, per stock: ols and ols_shrunk ----------------------------------------------------------------

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



def _beta_shrunk(y, x, period: int, min_periods: int, groups, data) -> pd.DataFrame:
    """Rolling beta of y on x, shrunk toward the cross-sectional mean (Vasicek 1973), or each group's (ts_beta_shrunk)."""
    both = y.notna() & x.notna()
    y, x = y.where(both), x.where(both)                      # paired observations only
    roll = lambda v: v.rolling(period, min_periods=min_periods)
    n = roll(both.astype(float).where(both)).count()
    vx, vy, cov = roll(x).var(), roll(y).var(), roll(y).cov(x)
    vx = vx.where(vx > 0)
    b = cov / vx
    resid = (vy - cov * b).clip(lower=0) * (n - 1) / (n - 2)
    se2 = resid / ((n - 1) * vx)
    if groups is not None:
        return _shrink_within_groups(b, se2, groups, data)
    cs = in_universe(b)                                    # the prior comes from the universe
    tau2 = cs.var(axis=1)
    w = (1 / (1 + se2.div(tau2.where(tau2 > 0), axis=0))).where(b.notna())
    return w * b + (1 - w).mul(cs.mean(axis=1), axis=0)


def _one_regressor_outputs(y, x, period, min_periods, beta) -> dict:
    """alpha and resid from a slope (a function of get), on the days y and x are paired: alpha = mean(y) -
    beta * mean(x), resid = y - alpha - beta * x."""
    def means(get):
        return _paired_means(y, x, period, min_periods)

    def alpha(get):
        _, _, my, mx = get("_means")
        return my - get("beta1") * mx

    def resid(get):
        yy, xx, _, _ = get("_means")
        return yy - get("alpha") - get("beta1") * xx

    return {"beta1": beta, "_means": means, "alpha": alpha, "resid": resid}


def _check_min_periods(name, hyper):
    mp = hyper.get("min_periods")
    if mp is not None and not _whole(mp):
        raise ValueError(f"estimate({name!r})'s min_periods must be a whole number, at least 1 (got {mp!r})")


@register_estimator
class Ols(Estimator):
    """Rolling least squares of y on one or more regressors and an intercept, per stock, over the days in each window
    where y and every regressor have a value: ts_beta, ts_alpha and ts_resid. One regressor: beta = cov(y, x) / var(x),
    alpha = mean(y) - beta * mean(x). Several: one joint fit (at least k + 2 days)."""
    name, axis, n_inputs = "ols", "stock", (2, None)
    window, refit, refit_default = "required", 1, 1
    hyper = {"min_periods": None}
    dof = 0                                          # descriptive: nothing searched for
    about = "each stock's rolling regression of y on the regressors: the slopes, the intercept, today's residual"

    def outputs_for(self, n_inputs):
        return _beta_outputs(n_inputs, ("alpha", "resid"))

    def check(self, inputs, window, refit, hyper):
        _check_min_periods(self.name, hyper)

    def normalize(self, inputs, window, refit, hyper):
        return {"min_periods": _get_min_periods(window, hyper["min_periods"])}

    def compute(self, values, window, refit, hyper, data):
        y, xs, mp = values[0], values[1:], hyper["min_periods"]
        if len(xs) == 1:
            def beta(get):
                yy, xx = _paired(y, xs[0])
                cov = yy.rolling(window, min_periods=mp).cov(xx)
                var = xx.rolling(window, min_periods=mp).var()
                return cov / var
            return _one_regressor_outputs(y, xs[0], window, mp, beta)
        outs = {"_fit": lambda get: _multi_fit(y, list(xs), window, mp),
                "alpha": lambda get: get("_fit")[2]}

        def resid(get):
            yv, xv, alpha, betas = get("_fit")
            out = yv - alpha
            for x, b in zip(xv, betas):
                out = out - b * x
            return out
        outs["resid"] = resid
        for j in range(1, len(xs) + 1):
            outs[f"beta{j}"] = (lambda j: lambda get: get("_fit")[3][j - 1])(j)
        return outs

    # the operators' identities and spellings
    def fit_key(self, fit):
        return ('TsFit', fit.inputs[0]._cache_key(), _keys(fit.inputs[1:]), fit.window, fit.hyper["min_periods"],
                False, None)

    def output_key(self, output, fit):
        y, xs, w, mp = fit.inputs[0], fit.inputs[1:], fit.window, fit.hyper["min_periods"]
        if output.startswith("beta"):
            if len(xs) == 1:
                return ('TsBeta', y._cache_key(), xs[0]._cache_key(), w, mp)
            return ('TsFitBeta', fit._cache_key(), int(output[4:]))
        return ({"alpha": "TsAlpha", "resid": "TsResid"}[output], y._cache_key(), xs[0]._cache_key(), w, mp) \
            + _keys(xs[1:])

    def view(self, output, n_inputs):
        from . import timeseries as ts
        if output.startswith("beta"):
            return ts.TsBeta if n_inputs == 2 else None
        return {"alpha": ts.TsAlpha, "resid": ts.TsResid}[output]

    def spell(self, output, fit, label):
        args = [label(x) for x in fit.inputs]
        if output.startswith("beta"):
            if len(fit.inputs) > 2:                          # a joint fit's slope: no named operator
                return f"estimate('ols', {output!r}, {', '.join(args)}, window={fit.window})"
            return f"ts_beta({', '.join(args)}, {fit.window})"
        return f"ts_{output}({', '.join(args)}, {fit.window})"

    def spell_fit(self, fit, label):
        return f"ts_fit({', '.join(label(x) for x in fit.inputs)}, {fit.window})"


@register_estimator
class OlsShrunk(Estimator):
    """Rolling least squares on one regressor with each stock's slope shrunk toward the day's mean slope (or its
    group's), by how noisy it is (Vasicek 1973): ts_beta_shrunk, ts_alpha_shrunk and ts_resid_shrunk."""
    name, axis, n_inputs = "ols_shrunk", "stock", (2, 2)
    window, refit, refit_default = "required", 1, 1
    hyper = {"min_periods": None, "groups": None}
    label_options = ("groups",)
    outputs = ("alpha", "resid", "beta1")
    dof = 0                                          # descriptive: nothing searched for
    about = "each stock's rolling slope, Vasicek-shrunk toward the day's (or its group's) mean; its intercept, residual"

    def check(self, inputs, window, refit, hyper):
        _check_min_periods(self.name, hyper)

    def normalize(self, inputs, window, refit, hyper):
        from .groups import _as_groups
        g = hyper["groups"]
        return {"min_periods": max(_get_min_periods(window, hyper["min_periods"]), 3),
                "groups": None if g is None else _as_groups(g)}

    def compute(self, values, window, refit, hyper, data):
        y, x, mp, g = values[0], values[1], hyper["min_periods"], hyper["groups"]
        return _one_regressor_outputs(y, x, window, mp, lambda get: _beta_shrunk(y, x, window, mp, g, data))

    def fit_key(self, fit):
        g = fit.hyper["groups"]
        return ('TsFit', fit.inputs[0]._cache_key(), _keys(fit.inputs[1:]), fit.window, fit.hyper["min_periods"],
                True, None if g is None else _groups_key(g))

    def output_key(self, output, fit):
        g = fit.hyper["groups"]
        name = {"beta1": "TsBetaShrunk", "alpha": "TsAlphaShrunk", "resid": "TsResidShrunk"}[output]
        key = (name, fit.inputs[0]._cache_key(), fit.inputs[1]._cache_key(), fit.window, fit.hyper["min_periods"])
        return key if g is None else key + (_groups_key(g),)

    def view(self, output, n_inputs):
        from . import timeseries as ts
        return {"beta1": ts.TsBetaShrunk, "alpha": ts.TsAlphaShrunk, "resid": ts.TsResidShrunk}[output]

    @staticmethod
    def _args(fit, label):
        g = fit.hyper["groups"]
        args = [label(x) for x in fit.inputs] + [str(fit.window)]
        if g is not None:
            args.append(label(g) if isinstance(g, Signal) else repr(g))
        return ", ".join(args)

    def spell(self, output, fit, label):
        op = {"beta1": "ts_beta_shrunk", "alpha": "ts_alpha_shrunk", "resid": "ts_resid_shrunk"}[output]
        return f"{op}({self._args(fit, label)})"

    def spell_fit(self, fit, label):
        return f"ts_fit({self._args(fit, label)})"


# --- each day's least squares across stocks: xs_ols --------------------------------------------------------------------

def _xs_resid(y, xs) -> pd.DataFrame:
    """regression_neut: y less its day's least-squares fit on the regressors, across the names in the universe where
    y and every regressor are valid (at least k + 2 of them; fewer: the whole day NaN)."""
    y = finite_in_universe(y)
    xs = [finite_in_universe(x.reindex_like(y)) for x in xs]
    if len(xs) == 1:
        x = xs[0]
        valid = y.notna() & x.notna()
        valid = valid & (valid.sum(axis=1) >= 3).to_numpy()[:, None]          # k + 2 names, k = 1
        y, x = y.where(valid), x.where(valid)
        yc = y.sub(y.mean(axis=1), axis=0)
        xc = x.sub(x.mean(axis=1), axis=0)
        var = (xc * xc).sum(axis=1)
        b = ((yc * xc).sum(axis=1) / var.where(var > 0)).fillna(0.0)
        return yc - xc.mul(b, axis=0)
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


def _xs_fit(y, xs):
    """The day's intercept and slopes across stocks (regression_alpha, regression_beta): (alpha, betas), per day."""
    y = finite_in_universe(y)
    xs = [finite_in_universe(x.reindex_like(y)) for x in xs]
    Y = y.to_numpy(dtype=float)
    X = np.stack([x.to_numpy(dtype=float) for x in xs], axis=2)          # dates x names x regressors
    valid = np.isfinite(Y) & np.isfinite(X).all(axis=2)
    k = X.shape[2]
    alpha, betas = np.full(Y.shape[0], np.nan), np.full((Y.shape[0], k), np.nan)
    for t in range(Y.shape[0]):
        ok = valid[t]
        if ok.sum() < k + 2:                                             # regression_neut's minimum
            continue
        ym, xm = Y[t, ok].mean(), X[t, ok, :].mean(axis=0)
        b = np.linalg.lstsq(X[t, ok, :] - xm, Y[t, ok] - ym, rcond=None)[0]
        alpha[t], betas[t] = ym - b @ xm, b
    return alpha, betas


def _broadcast(values, like) -> pd.DataFrame:
    """One number per day, given to every name."""
    return pd.DataFrame(np.repeat(values[:, None], len(like.columns), axis=1), index=like.index, columns=like.columns)


@register_estimator
class XsOls(Estimator):
    """Each day's least squares of y on the regressors and an intercept, across the stocks in the universe:
    regression_neut (the residual), regression_alpha and regression_beta (the intercept and slopes, one number a day,
    given to every stock)."""
    name, axis, n_inputs = "xs_ols", "cross_section", (2, None)
    window, refit, refit_default = 1, 1, 1
    dof = 0                                          # descriptive: nothing searched for
    about = "each day's regression across stocks: the residual, and the intercept and slopes (factor returns)"

    def outputs_for(self, n_inputs):
        return _beta_outputs(n_inputs, ("resid", "alpha"))

    def compute(self, values, window, refit, hyper, data):
        y, xs = values[0], values[1:]
        outs = {"resid": lambda get: _xs_resid(y, xs),
                "_fit": lambda get: _xs_fit(y, xs),
                "alpha": lambda get: _broadcast(get("_fit")[0], y)}
        for j in range(1, len(xs) + 1):
            outs[f"beta{j}"] = (lambda j: lambda get: _broadcast(get("_fit")[1][:, j - 1], y))(j)
        return outs

    def fit_key(self, fit):
        return ('RegressionFit', fit.inputs[0]._cache_key(), _keys(fit.inputs[1:]))

    def output_key(self, output, fit):
        base = _keys(fit.inputs)
        if output == "resid":
            return ('RegressionNeut',) + base
        if output == "alpha":
            return ('RegressionAlpha',) + base
        return ('RegressionBeta',) + base + (int(output[4:]),)

    def view(self, output, n_inputs):
        from . import crosssection as cs
        return {"resid": cs.RegressionNeut, "alpha": cs.RegressionAlpha}.get(output, cs.RegressionBeta)

    def spell(self, output, fit, label):
        args = [label(x) for x in fit.inputs]
        if output == "resid":
            return f"regression_neut({', '.join(args)})"
        if output == "alpha":
            return f"regression_alpha({', '.join(args)})"
        return f"regression_beta({', '.join(args + [output[4:]])})"

    def spell_fit(self, fit, label):
        return f"regression_fit({', '.join(label(x) for x in fit.inputs)})"


# --- statistical factors: pca ----------------------------------------------------------------------------------------

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
    for p in refit_rows(values.index, window - 1, refit):          # on the calendar, not from the first row (schedule.py)
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



@register_estimator
class Pca(Estimator):
    """Principal components of x across stocks on the trailing window, refit on the calendar: each stock's beta to
    the k-th statistical factor (pca_factor). Correlation PCA; components tracked and signed across refits."""
    name, axis, n_inputs = "pca", "panel", (1, 1)
    window, refit, refit_default = "required", "optional", 21
    dof = 0                                          # descriptive: nothing searched for
    about = "each stock's beta to the k-th statistical factor (correlation PCA across stocks)"

    def outputs_for(self, n_inputs):
        return tuple(f"beta{k}" for k in range(1, PCA_MAX_COMPONENTS + 1))

    def check(self, inputs, window, refit, hyper):
        if window < 3:
            raise ValueError("estimate('pca')'s window must be at least 3 days")

    def compute(self, values, window, refit, hyper, data):
        members = universe_members(values[0].index, values[0].columns)
        outs = {"_fits": lambda get: _pca_fits(values[0], window, refit, members)}

        def factor(k):
            def out(get):
                fits = get("_fits")
                T, N = values[0].shape
                o = np.full((T, N), np.nan)
                for i, (p, names, betas) in enumerate(fits):
                    end = fits[i + 1][0] if i + 1 < len(fits) else T
                    if k <= betas.shape[0]:
                        o[p:end, names] = betas[k - 1]
                return in_universe(pd.DataFrame(o, index=values[0].index, columns=values[0].columns))
            return out
        for k in range(1, PCA_MAX_COMPONENTS + 1):
            outs[f"beta{k}"] = factor(k)
        return outs

    def fit_key(self, fit):
        return ('PcaBasis', fit.inputs[0]._cache_key(), fit.window, fit.refit)

    def output_key(self, output, fit):
        return ('PcaFactor', fit.inputs[0]._cache_key(), fit.window, int(output[4:]), fit.refit)

    def view(self, output, n_inputs):
        from .crosssection import PcaFactor
        return PcaFactor

    def spell(self, output, fit, label):
        refit = [str(fit.refit)] if fit.refit != 21 else []
        return f"pca_factor({', '.join([label(fit.inputs[0]), str(fit.window), output[4:]] + refit)})"

    def spell_fit(self, fit, label):
        refit = [str(fit.refit)] if fit.refit != 21 else []
        return f"pca_basis({', '.join([label(fit.inputs[0]), str(fit.window)] + refit)})"
