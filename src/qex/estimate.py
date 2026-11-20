"""estimate(name, output, inputs..., window=, refit=, ...): every fitted quantity in qex, through one operator.

qex had built the same thing four times: rolling least squares (ts_beta, ts_alpha, ts_resid), its shrunk form, the
daily cross-sectional fit (regression_neut, regression_alpha, regression_beta) and the principal components
(pca_factor). Each fits something on a window, refits on a schedule and is read through small operators. An
*estimator* is that pattern with a name: registered once, called by name, read output by output.

    estimate('ols', 'beta1', returns(1), returns(1, 'market'), window=252)        # is ts_beta(..., 252)
    estimate('pca', 'beta2', returns(1), window=252, refit=21)                  # is pca_factor(returns(1), 252, 2)
    estimate('stump', 'w1', vol, mom_ret, rev_ret, window=1260, refit=63)       # a new one

**The operators stay.** ts_beta, regression_neut, pca_factor and the rest are unchanged, with their cache keys: a
built-in estimator *builds the same node*, so `estimate('ols', 'beta1', y, x, window=252)` and `ts_beta(y, x, 252)` are
one node, one cached computation, the same numbers to the last bit. **For readability, prefer the named operator
wherever one exists**; write `estimate` for what has no name (an estimator such as the stump, or an output such as one
slope of a joint fit).

**Python in a registry, never in the string.** An expression names an estimator; Python registers it
(`register_estimator`). An estimator is a class with a name, a version, its outputs, the axis it fits along, and either

- `build(output, inputs, window, refit, hyper)`: the node it is (the built-ins: compositions of existing nodes), or
- `fit(window, hyper, rng, members)` and `step(state, row, hyper, members)`: **the protocol.** The engine
  (EstimateFit) owns the schedule (refits on the calendar, qex.schedule), and hands `fit` only the rows of the window
  ending on the refit day and `step` only one day's row, with the parameters fixed until the next refit. An estimator
  written this way cannot see a later day even by mistake. Randomness comes from a generator seeded by the call and
  the refit's date (and the stock's name, per stock), so one expression always gives one number, wherever the data
  starts.

What `fit` and `step` see follows the estimator's axis:

    axis            fit(window, ...)                      step(state, row, ...)            each output
    series          rows x inputs (market-level)          inputs                           a number, for every stock
    stock           rows x inputs, one stock, per stock   that stock's inputs              that stock's number
    cross_section   rows x stocks x inputs, and members   stocks x inputs, and members     a number per stock
    (and panel)     (rows x stocks: in the universe)      (in the universe that day)       (none outside the universe)

A per-stock estimator may also give `fit_many` and `step_many`, the same over every stock at once (stocks first), to
avoid a loop in Python; by default they call `fit` and `step` stock by stock.
"""
import hashlib
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .signal import Signal
from .schedule import refit_rows


# --- the registry ---------------------------------------------------------------------------------------------------

_REGISTRY: Dict[str, Dict[str, type]] = {}              # name -> {version: class}


class Estimator:
    """An estimator: subclass, set the attributes, implement build, or fit and step, and register it.

        name        how expressions call it: estimate('name', ...), or 'name@version' for one version
        version     a string; a fix ships as a new version, registered beside the old one, never as an edit
        outputs     the outputs it has, or a method outputs_for(n_inputs) when they depend on the inputs ('beta1'...)
        axis        'stock' (each stock's own series), 'cross_section' (each day's stocks), 'panel', or 'series'
        n_inputs    (fewest, most) inputs; most None for any number
        window      'required', 'optional' or a fixed whole number; refit likewise, with a default
        hyper       {name: default} of the options it takes by name beyond window and refit
        about       one sentence: what it measures
    """

    name: str = ""
    version: str = "1"
    outputs: tuple = ()
    axis: str = "series"
    n_inputs: tuple = (1, 1)
    window: object = "required"
    refit: object = "optional"
    refit_default: Optional[int] = 21
    hyper: dict = {}
    about: str = ""

    def outputs_for(self, n_inputs: int) -> tuple:
        return tuple(self.outputs)

    def check(self, inputs: tuple, window, refit, hyper: dict) -> None:
        """Further checks of a call's arguments; raise ValueError with a message that says what to write."""

    # either this
    def build(self, output: str, inputs: tuple, window, refit, hyper: dict) -> Signal:
        raise NotImplementedError

    # or these (the protocol)
    def fit(self, window: np.ndarray, hyper: dict, rng: np.random.Generator, members=None):
        """The parameters, from the window: the days up to and including the refit day, read-only. Its shape follows
        the axis: rows x inputs (series; stock, one stock's), rows x stocks x inputs (cross_section, panel, with
        members, rows x stocks, true where a stock was in the universe). None when it cannot be fitted (too few days):
        the outputs have no value until the next refit."""
        raise NotImplementedError

    def step(self, state, row: np.ndarray, hyper: dict, members=None) -> dict:
        """One day's outputs from the fit in force and that day's inputs only: {output: number} (series; stock, for
        the stock), or {output: one number per stock} (cross_section, panel; members: the day's universe)."""
        raise NotImplementedError

    def fit_many(self, window: np.ndarray, hyper: dict, rngs: list):
        """Per-stock estimators: every stock's fit at once, from window (stocks x rows x inputs); a list of states,
        None for a stock with no fit. By default, fit stock by stock."""
        return [self.fit(window[i], hyper, rngs[i]) for i in range(window.shape[0])]

    def step_many(self, states: list, row: np.ndarray, hyper: dict) -> dict:
        """Per-stock estimators: every stock's outputs on one day at once, from row (stocks x inputs);
        {output: one number per stock}, NaN where a stock has no fit. By default, step stock by stock."""
        outs = {o: np.full(row.shape[0], np.nan) for o in self.outputs_for(row.shape[1])}
        for i, st in enumerate(states):
            if st is not None:
                for o, v in self.step(st, row[i], hyper).items():
                    outs[o][i] = v
        return outs


def register_estimator(cls: type) -> type:
    """Register an estimator class (usable as a decorator). The same name and version twice is an error."""
    if not (isinstance(cls, type) and issubclass(cls, Estimator)):
        raise TypeError("register_estimator() takes a subclass of qex.Estimator")
    if not cls.name or "@" in cls.name or not cls.name.replace("_", "").isalnum():
        raise ValueError(f"an estimator's name is letters, digits and _ (got {cls.name!r})")
    versions = _REGISTRY.setdefault(cls.name, {})
    if cls.version in versions:
        raise ValueError(f"estimator {cls.name!r} version {cls.version!r} is already registered "
                         f"({versions[cls.version].__module__}.{versions[cls.version].__qualname__}); "
                         f"a change ships as a new version")
    builds = cls.build is not Estimator.build
    steps = cls.fit is not Estimator.fit and cls.step is not Estimator.step
    if builds == steps:
        raise TypeError(f"estimator {cls.name!r} implements either build, or fit and step")
    if cls.axis not in ("series", "stock", "cross_section", "panel"):
        raise ValueError(f"estimator {cls.name!r}: its axis is 'series', 'stock', 'cross_section' or 'panel' "
                         f"(got {cls.axis!r})")
    versions[cls.version] = cls
    return cls


def estimators() -> pd.DataFrame:
    """Every registered estimator: name, version, axis, outputs and what it measures."""
    rows = []
    for name, versions in sorted(_REGISTRY.items()):
        for v, cls in versions.items():
            e = cls()
            outs = e.outputs_for(e.n_inputs[1] or e.n_inputs[0] + 1) if not cls.outputs else cls.outputs
            rows.append(dict(name=name, version=v, axis=cls.axis, outputs=", ".join(outs), about=cls.about))
    return pd.DataFrame(rows)


def _version_key(v: str):
    return tuple(int(p) if p.isdigit() else p for p in v.split("."))


def get_estimator(name: str) -> Estimator:
    """The estimator 'name' (its latest version) or 'name@version'."""
    base, _, version = name.partition("@")
    versions = _REGISTRY.get(base)
    if not versions:
        known = ", ".join(sorted(_REGISTRY)) or "none"
        raise ValueError(f"estimator {base!r} is not registered (registered: {known})")
    if version:
        if version not in versions:
            raise ValueError(f"estimator {base!r} has no version {version!r} (registered: {', '.join(versions)})")
        return versions[version]()
    return versions[max(versions, key=_version_key)]()


# --- the operator ---------------------------------------------------------------------------------------------------

_MISSING = object()


def estimate(name, output, *inputs, window=None, refit=None, **hyper) -> Signal:
    """estimate(name, output, inputs..., window=, refit=, options...): the `output` of estimator `name` fitted on the
    inputs, a dates x stocks frame like any other value."""
    usage = "estimate('name', 'output', inputs..., window=, refit=, ...)"
    if not isinstance(name, str) or not isinstance(output, str):
        raise ValueError(f"estimate() takes the estimator's name and the output, each quoted, first: {usage}")
    est = get_estimator(name)
    label = f"estimate({name!r}"
    lo, hi = est.n_inputs
    if not (lo <= len(inputs) and (hi is None or len(inputs) <= hi)):
        want = f"{lo}" if hi == lo else f"{lo} or more" if hi is None else f"{lo} to {hi}"
        raise ValueError(f"{label})'s estimator takes {want} inputs, got {len(inputs)}")
    for i, x in enumerate(inputs):
        if not isinstance(x, Signal):
            raise ValueError(f"{label})'s inputs are expressions; input {i + 1} is {x!r} (options go by name)")
    outs = est.outputs_for(len(inputs))
    if output not in outs:
        raise ValueError(f"{label}) has no output {output!r}; its outputs: {', '.join(outs)}")
    window = _schedule_arg(label, "window", est.window, window, None)
    refit = _schedule_arg(label, "refit", est.refit, refit, est.refit_default)
    unknown = sorted(set(hyper) - set(est.hyper))
    if unknown:
        allowed = ", ".join(["window", "refit"] + list(est.hyper))
        raise ValueError(f"{label}) has no option {unknown[0]!r}; by name it takes {allowed}")
    full = {k: hyper.get(k, d) for k, d in est.hyper.items()}
    est.check(inputs, window, refit, full)
    if type(est).build is not Estimator.build:
        return est.build(output, tuple(inputs), window, refit, full)
    return EstimateOutput(EstimateFit(est, tuple(inputs), window, refit, full), output)


def _schedule_arg(label, what, rule, value, default):
    """window= or refit=, checked against the estimator's rule: 'required', 'optional' (with a default), or a fixed
    whole number."""
    if value is None:
        if rule == "required":
            raise ValueError(f"{label}) needs {what}= (a whole number of days)")
        value = rule if isinstance(rule, int) else default
    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                              or not float(value).is_integer() or value < 1):
        raise ValueError(f"{label})'s {what} must be a whole number of days, at least 1 (got {value!r})")
    if isinstance(rule, int) and value != rule:
        raise ValueError(f"{label})'s {what} is {rule} for this estimator (got {value!r})")
    return None if value is None else int(value)


# --- the protocol's engine: one fit node, read by output --------------------------------------------------------------

class EstimateResult:
    """An EstimateFit's value: each output as a dates array (series) or a dates x stocks array, with the dates and the
    stocks."""

    def __init__(self, index, columns, outputs: Dict[str, np.ndarray]):
        self.index, self.columns, self.outputs = index, columns, outputs


def _stock_seed(name) -> int:
    return int(hashlib.sha256(str(name).encode()).hexdigest()[:8], 16)


class EstimateFit(Signal):
    """The fit behind every estimate(...) of one estimator on the same inputs, window, refit and options: an internal
    node, shared by its outputs as a PcaBasis is by every pca_factor. Its value is an EstimateResult, not a frame."""

    value_is_frame = False

    def __init__(self, estimator: Estimator, inputs: tuple, window: int, refit: int, hyper: dict):
        self.estimator = estimator
        self.inputs = tuple(inputs)
        self.window = window
        self.refit = refit
        self.hyper = dict(hyper)

    def _series(self, values) -> np.ndarray:
        """A market-level input as one series: each day's value, the same for every stock that has one."""
        if isinstance(values, pd.Series):
            return values.to_numpy(dtype=float)
        v = values.astype(float).where(np.isfinite(values.astype(float)))
        lo, hi = v.min(axis=1), v.max(axis=1)                     # skipping the stocks with no value that day
        spread = (hi - lo).fillna(0).to_numpy()
        scale = float(v.abs().max().max()) if v.notna().any().any() else 1.0
        if (spread > 1e-9 * max(scale, 1e-12)).any():
            raise ValueError(f"estimate({self.estimator.name!r}) fits one series: each input must be the same for "
                             f"every stock that day (a market-level value, like returns(1, 'market') or "
                             f"regression_beta(...)), not each stock's own")
        return lo.to_numpy(dtype=float)

    def _compute(self, data):
        values = [x.evaluate(data) for x in self.inputs]
        ref = next((v for v in values if isinstance(v, pd.DataFrame)), None)
        index = ref.index if ref is not None else values[0].index
        columns = ref.columns if ref is not None else pd.Index([])
        est, hyper, T, N = self.estimator, self.hyper, len(index), len(columns)
        names = est.outputs_for(len(self.inputs))
        rows = refit_rows(index, self.window - 1, self.refit)
        seed = int(hashlib.sha256(repr(self._cache_key()).encode()).hexdigest()[:16], 16)
        dated = isinstance(index, pd.DatetimeIndex)
        day_of = lambda r: int(np.datetime64(index[r], "D").astype(int)) if dated else int(r)
        ends = list(rows[1:]) + [T]
        if est.axis == "series":
            X = np.column_stack([self._series(v.reindex(index) if isinstance(v, (pd.Series, pd.DataFrame)) else v)
                                 for v in values])
            X.flags.writeable = False
            outs = {o: np.full(T, np.nan) for o in names}
            for r, end in zip(rows, ends):
                state = est.fit(X[r - self.window + 1:r + 1], hyper, np.random.default_rng([seed, day_of(r)]))
                if state is None:
                    continue
                for t in range(r, end):
                    for o, val in est.step(state, X[t], hyper).items():
                        outs[o][t] = val
            return EstimateResult(index, columns, outs)
        # stock, cross_section, panel: every input as dates x stocks, stacked once (windows are views of it)
        frames = [v.reindex(index=index, columns=columns) if isinstance(v, pd.DataFrame)
                  else pd.DataFrame(np.repeat(v.reindex(index).to_numpy(float)[:, None], N, axis=1), index, columns)
                  for v in values]
        X = np.stack([f.to_numpy(dtype=float) for f in frames], axis=2)          # dates x stocks x inputs
        X.flags.writeable = False
        outs = {o: np.full((T, N), np.nan) for o in names}
        if est.axis == "stock":
            stock_seeds = [_stock_seed(c) for c in columns]
            for r, end in zip(rows, ends):
                day = day_of(r)
                window = X[r - self.window + 1:r + 1].transpose(1, 0, 2)          # stocks x rows x inputs
                states = est.fit_many(window, hyper, [np.random.default_rng([seed, day, k]) for k in stock_seeds])
                for t in range(r, end):
                    for o, val in est.step_many(states, X[t], hyper).items():
                        outs[o][t] = val
            return EstimateResult(index, columns, outs)
        from .universe import universe_members                                   # cross_section, panel
        members = universe_members(index, columns)
        if members is None:
            members = np.ones((T, N), bool)
        members = members & np.isfinite(X).all(axis=2)                           # in the universe, with every input
        members.flags.writeable = False
        for r, end in zip(rows, ends):
            lo = r - self.window + 1
            state = est.fit(X[lo:r + 1], hyper, np.random.default_rng([seed, day_of(r)]), members=members[lo:r + 1])
            if state is None:
                continue
            for t in range(r, end):
                for o, val in est.step(state, X[t], hyper, members=members[t]).items():
                    outs[o][t] = np.where(members[t], val, np.nan)               # none outside the universe
        return EstimateResult(index, columns, outs)

    def _cache_key(self):
        e = self.estimator
        return ('EstimateFit', e.name, e.version, tuple(x._cache_key() for x in self.inputs), self.window,
                self.refit, tuple(sorted((k, _key(v)) for k, v in self.hyper.items())))


def _key(v):
    return v._cache_key() if isinstance(v, Signal) else v


class EstimateOutput(Signal):
    """One output of an estimator's fit, given to every stock: estimate(name, output, ...)."""

    def __init__(self, fit: EstimateFit, output: str):
        self.fit = fit
        self.output = output

    def _compute(self, data):
        res = self.fit.evaluate(data)
        v = res.outputs[self.output]
        if v.ndim == 1:                                                           # a series: every stock's
            v = np.repeat(v[:, None], len(res.columns), axis=1)
        return pd.DataFrame(v, index=res.index, columns=res.columns)

    def _cache_key(self):
        return ('EstimateOutput', self.fit._cache_key(), self.output)


# --- the built-ins: today's operators, as estimators ---------------------------------------------------------------

def _min_periods(hyper):
    mp = hyper.get("min_periods")
    if mp is not None and (isinstance(mp, bool) or not isinstance(mp, (int, float)) or not float(mp).is_integer()
                           or mp < 1):
        raise ValueError(f"min_periods must be a whole number, at least 1 (got {mp!r})")
    return None if mp is None else int(mp)


def _beta_outputs(n_inputs: int, prefix: tuple) -> tuple:
    return prefix + tuple(f"beta{j}" for j in range(1, n_inputs))


@register_estimator
class Ols(Estimator):
    """Rolling least squares of y on one or more regressors, per stock: ts_beta, ts_alpha and ts_resid."""
    name, axis, n_inputs = "ols", "stock", (2, None)
    window, refit, refit_default = "required", 1, 1
    hyper = {"min_periods": None}
    about = "each stock's rolling regression of y on the regressors: the slopes, the intercept, today's residual"

    def outputs_for(self, n_inputs):
        return _beta_outputs(n_inputs, ("alpha", "resid"))

    def build(self, output, inputs, window, refit, hyper):
        from .timeseries import ts_alpha, ts_beta, ts_resid, TsFit, _get_min_periods
        y, xs = inputs[0], inputs[1:]
        mp = _min_periods(hyper)
        tail = (window,) if mp is None else (window, mp)
        if output == "alpha":
            return ts_alpha(y, *xs, *tail)
        if output == "resid":
            return ts_resid(y, *xs, *tail)
        j = int(output[4:])
        if len(xs) == 1:
            return ts_beta(y, xs[0], window, mp)                       # the ts_beta node itself
        return TsFitBeta(TsFit(y, xs, window, _get_min_periods(window, mp)), j)


@register_estimator
class OlsShrunk(Estimator):
    """Rolling least squares on one regressor with the slope shrunk toward the day's mean (or the group's): the
    shrunk fits."""
    name, axis, n_inputs = "ols_shrunk", "stock", (2, 2)
    window, refit, refit_default = "required", 1, 1
    hyper = {"min_periods": None, "groups": None}
    outputs = ("alpha", "resid", "beta1")
    about = "each stock's rolling slope, Vasicek-shrunk toward the day's (or its group's) mean; its intercept, residual"

    def build(self, output, inputs, window, refit, hyper):
        from .timeseries import ts_alpha_shrunk, ts_beta_shrunk, ts_resid_shrunk
        mp, groups = _min_periods(hyper), hyper.get("groups")
        rest = tuple(v for v in (mp, groups) if v is not None)
        f = {"alpha": ts_alpha_shrunk, "resid": ts_resid_shrunk, "beta1": ts_beta_shrunk}[output]
        return f(inputs[0], inputs[1], window, *rest)


@register_estimator
class XsOls(Estimator):
    """Each day's least squares of y on the regressors across stocks: regression_neut, regression_alpha and
    regression_beta."""
    name, axis, n_inputs = "xs_ols", "cross_section", (2, None)
    window, refit, refit_default = 1, 1, 1
    about = "each day's regression across stocks: the residual, and the intercept and slopes (factor returns)"

    def outputs_for(self, n_inputs):
        return _beta_outputs(n_inputs, ("resid", "alpha"))

    def build(self, output, inputs, window, refit, hyper):
        from .crosssection import regression_alpha, regression_beta, regression_neut
        if output == "resid":
            return regression_neut(*inputs)
        if output == "alpha":
            return regression_alpha(*inputs)
        j = int(output[4:])
        return regression_beta(*inputs, j) if len(inputs) > 2 else regression_beta(*inputs)


@register_estimator
class Pca(Estimator):
    """Principal components across stocks, refit on the calendar: pca_factor."""
    name, axis, n_inputs = "pca", "panel", (1, 1)
    window, refit, refit_default = "required", "optional", 21
    about = "each stock's beta to the k-th statistical factor (correlation PCA across stocks)"

    def outputs_for(self, n_inputs):
        from .crosssection import PCA_MAX_COMPONENTS
        return tuple(f"beta{k}" for k in range(1, PCA_MAX_COMPONENTS + 1))

    def build(self, output, inputs, window, refit, hyper):
        from .crosssection import pca_factor
        if window < 3:
            raise ValueError("estimate('pca')'s window must be at least 3 days")
        return pca_factor(inputs[0], window, int(output[4:]), refit)


class TsFitBeta(Signal):
    """The j-th slope of a joint rolling fit (several regressors): estimate('ols', 'betaj', y, x1, ..., xk, window=).
    With one regressor the slope is ts_beta's node; with several there was no named reader until estimate."""

    def __init__(self, fit, j: int):
        self.fit = fit
        self.j = j

    def _compute(self, data):
        return self.fit.evaluate(data).betas[self.j - 1]

    def _cache_key(self):
        return ('TsFitBeta', self.fit._cache_key(), self.j)


# --- a new estimator: the stump ---------------------------------------------------------------------------------------

@register_estimator
class Stump(Estimator):
    """A learned switch between two signals: threshold regression (Hansen 2000) on a market state, a decision stump.

        estimate('stump', 'w1', state, r1, r2, window=1260, refit=63)

    `state` is a market-level series (volatility, a trend, dispersion), `r1` and `r2` two signals' daily returns
    (regression_beta(returns(1), delay(signal, 1)) is a signal's return). On each refit day it pairs the state on day
    u - 1 with the two returns on day u over the window, and finds the threshold on the state that best separates the
    days on which r1 beat r2 from those it did not: the split that most reduces the squared error of r1 - r2 against
    its mean on each side (least squares with one break). Each side keeps the signal that earned more there. Outputs:

        w1          1 if today's state is on the side where r1 earned more, else 0: the weight on the first signal
        threshold   the threshold in force, in the state's units (to watch drift)
        gap         the edge of r1 over r2 above the threshold less the edge below it, a day's return
        p           a permutation test inside the window: how often a split on a shuffled state does as well

    Options: `min_leaf` (the fewest paired days on each side; default a fifth of the window), `quantiles` (the
    thresholds tried: this many quantiles of the state; default 20), `permutations` (default 200; 0 for none). The
    shuffles come from a generator seeded by the call, so one expression always gives one p.
    """
    name, axis, n_inputs = "stump", "series", (3, 3)
    window, refit, refit_default = "required", "optional", 21
    outputs = ("w1", "threshold", "gap", "p")
    hyper = {"min_leaf": None, "quantiles": 20, "permutations": 200}
    about = "a learned threshold on a market state that picks, on each side, the better of two signals (Hansen 2000)"

    def check(self, inputs, window, refit, hyper):
        ml, q, b = hyper["min_leaf"], hyper["quantiles"], hyper["permutations"]
        whole = lambda v, lo: not isinstance(v, bool) and isinstance(v, (int, float)) and float(v).is_integer() and v >= lo
        if ml is not None and not whole(ml, 2):
            raise ValueError(f"estimate('stump')'s min_leaf must be a whole number of days, at least 2 (got {ml!r})")
        if ml is not None and 2 * ml > window - 1:
            raise ValueError(f"estimate('stump')'s min_leaf ({ml}) leaves no split: two leaves need at most half of "
                             f"the window's {window - 1} paired days each")
        if not whole(q, 2):
            raise ValueError(f"estimate('stump')'s quantiles must be a whole number, at least 2 (got {q!r})")
        if not whole(b, 0):
            raise ValueError(f"estimate('stump')'s permutations must be a whole number, 0 or more (got {b!r})")
        if window < 10:
            raise ValueError("estimate('stump')'s window must be at least 10 days")

    @staticmethod
    def _best(s_sorted, d_sorted, cuts):
        """The between-sides sum of squares of each cut (the leaves' sizes fixed by `cuts`, indices into the sorted
        state): its largest value and where."""
        n = len(d_sorted)
        cs = np.cumsum(d_sorted)
        total = cs[-1]
        n_lo = cuts.astype(float)
        s_lo = cs[cuts - 1]
        s_hi = total - s_lo
        n_hi = n - n_lo
        gain = s_lo ** 2 / n_lo + s_hi ** 2 / n_hi - total ** 2 / n
        k = int(np.argmax(gain))
        return gain[k], k

    def fit(self, window, hyper, rng, members=None):
        s, r1, r2 = window[:-1, 0], window[1:, 1], window[1:, 2]      # the state at u - 1, the returns on day u
        d = r1 - r2
        ok = np.isfinite(s) & np.isfinite(d)
        s, d = s[ok], d[ok]
        n = len(s)
        min_leaf = hyper["min_leaf"] or max(2, (len(window) - 1) // 5)
        if n < 2 * min_leaf:
            return None
        order = np.argsort(s, kind="mergesort")
        s_sorted, d_sorted = s[order], d[order]
        levels = np.linspace(0, 1, hyper["quantiles"] + 1)[1:-1]
        thresholds = np.unique(np.quantile(s_sorted, levels))
        cuts = np.searchsorted(s_sorted, thresholds, side="right")    # days at or below each threshold
        keep = (cuts >= min_leaf) & (n - cuts >= min_leaf)
        thresholds, cuts = thresholds[keep], cuts[keep]
        if not len(cuts):
            return None
        gain, k = self._best(s_sorted, d_sorted, cuts)
        c, m = thresholds[k], cuts[k]
        lo, hi = d_sorted[:m].mean(), d_sorted[m:].mean()
        b = int(hyper["permutations"])
        if b:
            beaten = sum(self._best(s_sorted, rng.permutation(d_sorted), cuts)[0] >= gain - 1e-15 for _ in range(b))
            p = (1 + beaten) / (b + 1)
        else:
            p = np.nan
        return dict(threshold=float(c), w_lo=1.0 if lo > 0 else 0.0, w_hi=1.0 if hi > 0 else 0.0,
                    gap=float(hi - lo), p=float(p))

    def step(self, state, row, hyper, members=None):
        s = row[0]
        w1 = np.nan if not np.isfinite(s) else (state["w_lo"] if s <= state["threshold"] else state["w_hi"])
        return dict(w1=w1, threshold=state["threshold"], gap=state["gap"], p=state["p"])
