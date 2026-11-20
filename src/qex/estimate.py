"""estimate(name, output, inputs..., window=, refit=, ...): every fitted quantity in qex, through one operator.

qex had built the same thing four times: rolling least squares (ts_beta, ts_alpha, ts_resid), its shrunk form, the
daily cross-sectional fit (regression_neut, regression_alpha, regression_beta) and the principal components
(pca_factor). Each fits something on a window, refits on a schedule and is read through small operators. An
*estimator* is that pattern with a name: registered once, called by name, read output by output.

    estimate('ols', 'beta1', returns(1), returns(1, 'market'), window=252)        # ts_beta(..., 252) is this
    estimate('pca', 'beta2', returns(1), window=252, refit=21)                  # pca_factor(returns(1), 252, 2) is this
    estimate('stump', 'w1', vol, mom_ret, rev_ret, window=1260, refit=63)       # a new one

**The named operators are sugar.** ts_beta, regression_neut, pca_factor and the rest are spellings of estimate: each
calls it, and the maths lives in the estimator (qex/fits.py). They keep their names, their classes (each a view of
one output, with no maths of its own) and their cache keys, so an expression's identity and numbers are what they
were. **For readability, prefer the named operator wherever one exists**; write `estimate` for what has no name (an
estimator such as the stump, or an output such as one slope of a joint fit).

**Python in a registry, never in the string.** An expression names an estimator; Python registers it
(`register_estimator`). An estimator is a class with a name, a version, its outputs, the axis it fits along, and either

- `compute(values, window, refit, hyper, data)`: every output at once, vectorized over the panel (the built-ins:
  rolling sums, a day's least squares, an SVD per refit). Each output may be a frame or a function of the others,
  computed when first read, so a fit read only for its slope never computes its residual; or
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
import threading
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .signal import Signal
from .schedule import refit_rows


# --- the registry ---------------------------------------------------------------------------------------------------

_REGISTRY: Dict[str, Dict[str, type]] = {}              # name -> {version: class}


class Estimator:
    """An estimator: subclass, set the attributes, implement compute, or fit and step, and register it.

        name        how expressions call it: estimate('name', ...), or 'name@version' for one version
        version     a string; a fix ships as a new version, registered beside the old one, never as an edit
        outputs     the outputs it has, or a method outputs_for(n_inputs) when they depend on the inputs ('beta1'...)
        axis        'stock' (each stock's own series), 'cross_section' (each day's stocks), 'panel', or 'series'
        n_inputs    (fewest, most) inputs; most None for any number
        window      'required', 'optional' or a fixed whole number; refit likewise, with a default
        hyper       {name: default} of the options it takes by name beyond window and refit
        about       one sentence: what it measures
        label_options   the options that take labels (a group's name or a labels expression), not numbers
        dof         required: how many parameters each fit learns by searching the data for what works (a
                    threshold, a choice of side, a weight chosen for its returns), which a score may charge for
                    (qex.complexity). Descriptive statistics (a beta, a mean, a component) are not searched: 0.

    An estimator is called by an expression and never sees one: its inputs arrive as values.
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
    label_options: tuple = ()
    dof: Optional[int] = None

    def dof_of(self, fit) -> int:
        """The dof one fit learns (it may depend on the fit's options); by default the class's dof."""
        return self.dof

    def outputs_for(self, n_inputs: int) -> tuple:
        return tuple(self.outputs)

    def check(self, inputs: tuple, window, refit, hyper: dict) -> None:
        """Further checks of a call's arguments; raise ValueError with a message that says what to write."""

    def normalize(self, inputs: tuple, window, refit, hyper: dict) -> dict:
        """The options as the fit keeps them (defaults resolved): two calls that mean the same fit share one."""
        return hyper

    # either this
    def compute(self, values: list, window, refit, hyper: dict, data) -> dict:
        """Every output, {output: frame (dates x stocks), or a function get -> frame, computed when first read, where
        get(name) gives another output}. values: the inputs, evaluated. data: for an option read from the data (a
        group's labels)."""
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

    # identity and spelling: how the named operators keep theirs (None: the estimate's own)
    def fit_key(self, fit) -> Optional[tuple]:
        return None

    def output_key(self, output: str, fit) -> Optional[tuple]:
        return None

    def view(self, output: str, n_inputs: int) -> Optional[type]:
        """The class of the node an output is (an EstimateOutput subclass named for its operator, such as TsBeta)."""
        return None

    def spell(self, output: str, fit, label) -> Optional[str]:
        """The output as its named operator writes it (label: a node's qex), or None to write it as estimate(...)."""
        return None

    def spell_fit(self, fit, label) -> Optional[str]:
        return None


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
    computes = cls.compute is not Estimator.compute
    steps = cls.fit is not Estimator.fit and cls.step is not Estimator.step
    if computes == steps:
        raise TypeError(f"estimator {cls.name!r} implements either compute, or fit and step")
    if cls.dof is None or isinstance(cls.dof, bool) or not isinstance(cls.dof, int) or cls.dof < 0:
        raise TypeError(f"estimator {cls.name!r} must say how many parameters it learns by searching the data: "
                        f"dof = 0 for a descriptive statistic (a beta, a mean), else the count (got {cls.dof!r})")
    if cls.axis not in ("series", "stock", "cross_section", "panel"):
        raise ValueError(f"estimator {cls.name!r}: its axis is 'series', 'stock', 'cross_section' or 'panel' "
                         f"(got {cls.axis!r})")
    versions[cls.version] = cls
    return cls


def estimators() -> pd.DataFrame:
    """Every registered estimator: name, version, axis, outputs, the dof it learns and what it measures."""
    from . import fits  # noqa: F401  (the built-ins, registered on first use)
    rows = []
    for name, versions in sorted(_REGISTRY.items()):
        for v, cls in versions.items():
            e = cls()
            outs = e.outputs_for(e.n_inputs[1] or e.n_inputs[0] + 1) if not cls.outputs else cls.outputs
            rows.append(dict(name=name, version=v, axis=cls.axis, outputs=", ".join(outs), dof=cls.dof, about=cls.about))
    return pd.DataFrame(rows)


def _version_key(v: str):
    return tuple(int(p) if p.isdigit() else p for p in v.split("."))


def get_estimator(name: str) -> Estimator:
    """The estimator 'name' (its latest version) or 'name@version'."""
    from . import fits  # noqa: F401  (registers the built-ins on first use)
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
    full = est.normalize(tuple(inputs), window, refit, full)
    return _output(EstimateFit(est, tuple(inputs), window, refit, full), output)


def _output(fit: "EstimateFit", output: str) -> "EstimateOutput":
    """The node of one output of a fit: of its named operator's class when it has one."""
    cls = fit.estimator.view(output, len(fit.inputs)) or EstimateOutput
    node = object.__new__(cls)
    EstimateOutput.__init__(node, fit, output)
    return node


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
    return value if value is None or isinstance(value, int) else int(value)


# --- the engine: one fit node, read by output --------------------------------------------------------------------

class EstimateResult:
    """An EstimateFit's value: its outputs, each a dates x stocks frame, computed when first read. Outputs read at
    once on several threads (Dask tasks) wait for one another, so a shared part is computed once."""

    def __init__(self, outputs: dict):
        self._make, self._done = dict(outputs), {}
        self._lock = threading.RLock()

    def get(self, name: str) -> pd.DataFrame:
        with self._lock:
            if name not in self._done:
                v = self._make[name]
                self._done[name] = v(self.get) if callable(v) else v
            return self._done[name]

    def __getstate__(self):                       # sent to another process: what is computed so far, and the rest
        state = dict(self.__dict__)
        del state["_lock"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.RLock()


def _stock_seed(name) -> int:
    return int(hashlib.sha256(str(name).encode()).hexdigest()[:8], 16)


class EstimateFit(Signal):
    """The fit behind every output of one estimator on the same inputs, window, refit and options: an internal node,
    shared by its outputs (ts_beta, ts_alpha and ts_resid on the same y, x and window read one). Its value is an
    EstimateResult, not a frame."""

    value_is_frame = False

    def __init__(self, estimator: Estimator, inputs: tuple, window: int, refit: int, hyper: dict):
        self.estimator = estimator
        self.inputs = tuple(inputs)
        self.window = window
        self.refit = refit
        self.hyper = dict(hyper)
        self.options = tuple(v for v in self.hyper.values() if isinstance(v, Signal))   # inputs too, for the graph

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
        est = self.estimator
        if type(est).compute is not Estimator.compute:
            return EstimateResult(est.compute(values, self.window, self.refit, self.hyper, data))
        ref = next((v for v in values if isinstance(v, pd.DataFrame)), None)
        index = ref.index if ref is not None else values[0].index
        columns = ref.columns if ref is not None else pd.Index([])
        hyper, T, N = self.hyper, len(index), len(columns)
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
            return EstimateResult({o: pd.DataFrame(np.repeat(v[:, None], N, axis=1), index, columns)
                                   for o, v in outs.items()})
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
            return EstimateResult({o: pd.DataFrame(v, index, columns) for o, v in outs.items()})
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
        return EstimateResult({o: pd.DataFrame(v, index, columns) for o, v in outs.items()})

    def _cache_key(self):
        e = self.estimator
        key = e.fit_key(self)
        if key is not None:
            return key
        return ('EstimateFit', e.name, e.version, tuple(x._cache_key() for x in self.inputs), self.window,
                self.refit, tuple(sorted((k, _key(v)) for k, v in self.hyper.items())))


def _key(v):
    return v._cache_key() if isinstance(v, Signal) else v


class EstimateOutput(Signal):
    """One output of an estimator's fit: estimate(name, output, ...). A named operator's class (TsBeta, RegressionNeut,
    PcaFactor) is a subclass that only names it: the same node, with the operator's cache key and spelling."""

    def __init__(self, fit: EstimateFit, output: str):
        self.fit = fit
        self.output = output

    def _compute(self, data):
        return self.fit.evaluate(data).get(self.output)

    def _cache_key(self):
        key = self.fit.estimator.output_key(self.output, self.fit)
        return key if key is not None else ('EstimateOutput', self.fit._cache_key(), self.output)
