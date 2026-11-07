"""Why a value is missing: explain_missing traces every missing cell of a signal to where it began.

qex has one missing value, NaN: the book and the cross-sectional operators leave it out, whatever caused it. But
for research and for trading the cause matters. "Outside the universe" is expected, "excluded on purpose" (the None
literal) is a choice, "short history" is a warm-up, and a "calculation error" (0 / 0, the log of a negative) is a
bug to fix. explain_missing walks the signal's graph from the data up and gives each missing cell a reason and the
node where it began:

    no data               a field has no value that day: before a listing, during a halt, after a delisting
    outside the universe  a cross-sectional operator leaves out a name that is not in the universe that day
    short history         a window has too few values yet; when earlier days were missing, it says why
    excluded on purpose   the None literal, chosen by a where(...)
    no group              a group operator, for a name with no group that day
    calculation error     NaN from finite inputs: 0 / 0, log or sqrt of a negative, a z-score with no spread
    infinite value        a division by zero
    too few names         a day with too few names for a cross-sectional fit: regression_neut with k regressors
                          needs k + 2

Where a missing input explains a missing output, the reason is the input's: a cell takes the reason of the first
input missing there, and a node's own reason applies only where every input had a value. Nothing is computed twice
and nothing changes: it reads the values the signal's own evaluation produced.

    >>> with cross_section_universe(data['member'].notna()):
    ...     report = explain_missing(qex("ts_mean(rank(returns(1)), 10)"), data)
    >>> report.summary(within=data['member'].notna())      # each reason's share of the member-days with no value
    >>> report.on('2003-06-02')                             # every missing name that day, why, and where
"""
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import conditional as _cond
from . import crosssection as _cs
from . import data as _dt
from . import groups as _gr
from . import primitives as _pr
from . import timeseries as _ts
from . import universe as _uni
from .context import compute_context, get_context
from .operators import Constant, NoneValue
from .signal import Signal

PRESENT, NO_DATA, OUTSIDE, HISTORY, EXCLUDED, NO_GROUP, CALC, INFINITE, TOO_FEW = range(9)
REASONS = {
    NO_DATA: "no data",
    OUTSIDE: "outside the universe",
    HISTORY: "short history",
    EXCLUDED: "excluded on purpose",
    NO_GROUP: "no group",
    CALC: "calculation error",
    INFINITE: "infinite value",
    TOO_FEW: "too few names",
}
_CROSS = tuple(c for c in vars(_cs).values() if isinstance(c, type) and issubclass(c, Signal) and c.__module__ == _cs.__name__)
_GROUP_OPS = tuple(c for c in vars(_gr).values() if isinstance(c, type) and issubclass(c, Signal)
                   and c.__module__ == _gr.__name__ and c not in (_gr.Group, _gr.GroupCross))
_TIME = tuple(c for c in vars(_ts).values() if isinstance(c, type) and issubclass(c, Signal) and c.__module__ == _ts.__name__)
_PRIMITIVE_FIELDS = {_pr.Returns: lambda s: [s.price_field], _pr.Volatility: lambda s: [s.price_field],
                     _pr.Volume: lambda s: ["volume"], _pr.Adv: lambda s: [s.price_field, "volume"]}


def children(signal: Signal) -> List[Signal]:
    """A node's inputs: every attribute that is a signal, in the order the node keeps them."""
    out = []
    for v in vars(signal).values():
        if isinstance(v, Signal):
            out.append(v)
        elif isinstance(v, (list, tuple)):
            out.extend(x for x in v if isinstance(x, Signal))
    return out


def _snake(name: str) -> str:
    return {"ZScore": "zscore", "Ewma": "ewma", "EwmaVar": "ewma_var", "EwmaCov": "ewma_cov"}.get(
        name, re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower())


_SYMBOLS = {"Add": "+", "Sub": "-", "Mul": "*", "Div": "/", "Greater": ">", "Less": "<", "GreaterEqual": ">=",
            "LessEqual": "<=", "Equal": "==", "NotEqual": "!=", "And": "&", "Or": "|"}


_DEFAULTS = {"price_field": "close", "refit": 21}                     # arguments not written when they take their default


def label(signal: Signal, nested: bool = False) -> str:
    """A node as it would be written in qex, for saying where a missing value began."""
    if isinstance(signal, NoneValue):
        return "None"
    if isinstance(signal, Constant):
        return repr(signal.value) if isinstance(signal.value, str) else f"{signal.value:g}" if isinstance(signal.value, (int, float)) else str(signal.value)
    if isinstance(signal, _dt.Field):
        return f"{signal.name}()" if signal.name in ("close", "open", "high", "low") else f"field('{signal.name}')"
    name = type(signal).__name__
    if name in ("PcaFactor", "PcaBasis"):               # written pca_factor(x, window, k[, refit]); the basis is internal
        basis = signal.basis if name == "PcaFactor" else signal
        args = [label(basis.signal), str(basis.window)] + ([str(signal.k)] if name == "PcaFactor" else [])
        args += [str(basis.refit)] if basis.refit != 21 else []
        return f"{'pca_factor' if name == 'PcaFactor' else 'pca_basis'}({', '.join(args)})"
    kids = children(signal)
    if name in _SYMBOLS and len(kids) == 2:
        text = f"{label(kids[0], True)} {_SYMBOLS[name]} {label(kids[1], True)}"
        return f"({text})" if nested else text
    if name == "Neg":
        return f"-{label(kids[0], True)}"
    args = [label(k) for k in kids]
    for k, v in vars(signal).items():
        if isinstance(v, Signal) or (isinstance(v, (list, tuple)) and any(isinstance(x, Signal) for x in v)):
            continue
        if v is None or k in ("min_periods",) or _DEFAULTS.get(k) == v:
            continue
        args.append(repr(v) if isinstance(v, str) else str(v))
    return f"{_snake(name)}({', '.join(args)})"


class MissingReport:
    """Why each missing cell of a signal is missing. codes holds a reason (REASONS) per cell, 0 where there is a
    value; causes, for a short history, what made the earlier days missing (a reason, or 0 for the start of the
    data); origins, the node where it began (nodes[i] is its qex)."""

    def __init__(self, codes: pd.DataFrame, causes: pd.DataFrame, origins: pd.DataFrame, nodes: List[str]):
        self.codes, self.causes, self.origins, self.nodes = codes, causes, origins, nodes

    @staticmethod
    def text(code: int, cause: int = 0) -> str:
        """A reason in words: a short history says what made the earlier days missing."""
        if code == HISTORY and cause == HISTORY:
            return f"{REASONS[HISTORY]}: its input was still warming up"
        if code == HISTORY and cause:
            return f"{REASONS[HISTORY]}: {REASONS[cause]} on earlier days"
        return REASONS.get(code, "")

    def summary(self, within: Optional[pd.DataFrame] = None) -> pd.Series:
        """Each reason's share of the cells with no value (only the cells where within is true, if given), most
        common first. A short history is split by what made the earlier days missing."""
        codes, causes = self.codes.to_numpy(), self.causes.to_numpy()
        keep = codes != PRESENT
        if within is not None:
            keep &= within.reindex(index=self.codes.index, columns=self.codes.columns, fill_value=False).to_numpy(bool)
        if not keep.any():
            return pd.Series(dtype=float, name="share of missing")
        pairs = pd.Series(list(zip(codes[keep], np.where(codes[keep] == HISTORY, causes[keep], 0))))
        shares = pairs.value_counts(normalize=True)
        shares.index = [self.text(c, k) for c, k in shares.index]
        return shares.groupby(level=0).sum().sort_values(ascending=False).rename("share of missing")

    def on(self, date, names=None) -> pd.DataFrame:
        """Every missing name on a date (or only names), with its reason and the node where it began."""
        day = self.codes.index.get_indexer([pd.Timestamp(date)], method="nearest")[0]
        codes, causes, origins = self.codes.iloc[day], self.causes.iloc[day], self.origins.iloc[day]
        missing = codes[codes != PRESENT].index if names is None else pd.Index(names)
        return pd.DataFrame({"reason": [self.text(codes[n], causes[n]) for n in missing],
                             "at": [self.nodes[origins[n]] if codes[n] else "" for n in missing]}, index=missing)

    def reason(self, date, name) -> str:
        """Why one cell is missing, and where it began; "" if it has a value."""
        r = self.on(date, [name]).iloc[0]
        return f"{r['reason']}, at {r['at']}" if r["reason"] else ""


def explain_missing(signal: Signal, data: Dict[str, pd.DataFrame], universe: Optional[pd.DataFrame] = None) -> MissingReport:
    """The reason for every missing cell of signal's value on data (see the module's docstring). Evaluate it in the
    cross-sectional universe the signal is used in: pass universe, or call it inside cross_section_universe."""
    if universe is not None:
        with _uni.cross_section_universe(universe, data):        # a mask, a field's name or a universe expression
            return explain_missing(signal, data)
    if get_context() is None:
        with compute_context():
            return explain_missing(signal, data)

    root = signal.evaluate(data)
    if not isinstance(root, pd.DataFrame):
        raise ValueError("explain_missing needs a signal with a value per date and name, not a constant")
    index, columns = root.index, root.columns
    shape = root.shape
    umask = _uni._mask
    in_uni = None if umask is None else umask.reindex(index=index, columns=columns, fill_value=False).to_numpy(bool)

    def frame(v):
        """A node's value as an array of the root's shape: a constant broadcast, a series (the market) per column."""
        if isinstance(v, pd.DataFrame):
            v = v.reindex(index=index, columns=columns)
            if not all(pd.api.types.is_numeric_dtype(t) for t in v.dtypes):       # labels: present (0) or missing
                return np.where(v.isna().to_numpy(), np.nan, 0.0)
            return v.to_numpy(dtype=float, na_value=np.nan)
        if isinstance(v, pd.Series):
            a = v.reindex(index).to_numpy(dtype=float, na_value=np.nan)
            return np.repeat(a[:, None], shape[1], axis=1)
        return np.full(shape, np.nan if v is None else float(v) if np.isscalar(v) and not isinstance(v, str) else np.nan)

    nodes, results, labels = {}, {}, []                   # by cache key: the node, its (code, cause, origin)
    order: List[tuple] = []

    def visit(s):
        k = s._cache_key()
        if k in nodes:
            return
        nodes[k] = s
        for c in children(s):
            visit(c)
        order.append(k)
    visit(signal)
    users = {k: 0 for k in order}                         # how many nodes still need each result
    for k in order:
        for c in children(nodes[k]):
            users[c._cache_key()] += 1

    def rolling_any(mask, window):
        return pd.DataFrame(mask).rolling(window, min_periods=1).sum().to_numpy() > 0

    for k in order:
        s = nodes[k]
        node_id = len(labels)
        labels.append(label(s))
        if isinstance(s, _cs.PcaBasis):                 # a decomposition, not a frame: its input's reasons pass through
            code, cause, origin = (a.copy() for a in results[s.signal._cache_key()])
            results[k] = (code, cause, origin)
            for c in children(s):
                ck = c._cache_key()
                users[ck] -= 1
                if users[ck] == 0 and ck != order[-1]:
                    results.pop(ck, None)
            continue
        v = frame(s.evaluate(data))
        missing = ~np.isfinite(v)
        code = np.zeros(shape, np.int8)
        cause = np.zeros(shape, np.int8)
        origin = np.zeros(shape, np.int32)
        kids = children(s)
        got = [results[c._cache_key()] for c in kids]

        def inherit(mask, res):
            """Cells in mask still without a reason take the input's reason, cause and origin."""
            take = mask & (code == PRESENT) & (res[0] != PRESENT)
            code[take], cause[take], origin[take] = res[0][take], res[1][take], res[2][take]

        def own(mask, reason, why=None):
            take = mask & (code == PRESENT)
            code[take], origin[take] = reason, node_id
            if why is not None:
                cause[take] = why[take]

        if isinstance(s, NoneValue):
            own(missing, EXCLUDED)
        elif isinstance(s, _dt.Field) or isinstance(s, (_dt.DayOfWeek, _dt.DayOfMonth, _dt.MonthOfYear)):
            own(missing, NO_DATA)
        elif isinstance(s, _gr.Group):
            own(missing, NO_GROUP)
        elif isinstance(s, _gr.GroupCross):
            for res in got:
                inherit(missing, res)                                        # the part with no label, and why
            own(missing, NO_GROUP)
        elif isinstance(s, tuple(_PRIMITIVE_FIELDS)):
            base = np.zeros(shape, bool)
            for f in _PRIMITIVE_FIELDS[type(s)](s):
                base |= ~np.isfinite(frame(_dt.panel(data, f)))
            own(missing & base, NO_DATA)
            earlier = rolling_any(base, s.period + 1)
            own(missing, HISTORY, np.where(earlier, NO_DATA, PRESENT).astype(np.int8))
        elif isinstance(s, _cond.Where):
            c = frame(s.condition.evaluate(data))
            inherit(missing & np.isnan(c), got[0])                            # a missing condition: its reason
            chosen = c != 0
            for branch, res, pick in ((s.if_true, got[1], chosen & ~np.isnan(c)), (s.if_false, got[2], c == 0)):
                mask = missing & pick
                if isinstance(branch, NoneValue):
                    own(mask, EXCLUDED)                                     # the where that chose None
                else:
                    inherit(mask, res)
            own(missing, CALC)
        elif isinstance(s, _CROSS + _GROUP_OPS):
            if in_uni is not None:
                own(missing & ~in_uni, OUTSIDE)
            for res in got:
                inherit(missing, res)
            if isinstance(s, _GROUP_OPS):
                g = _gr._get_group_data(data, s.groups)
                codes_, _ = _gr._group_codes(root, g)
                own(missing & (codes_ < 0), NO_GROUP)
                if hasattr(s, "window"):
                    own(missing, HISTORY)
            if isinstance(s, _cs.PcaFactor):
                own(missing, HISTORY)                                       # before its first fit, or not yet in one
            if isinstance(s, _cs.RegressionNeut):
                usable = np.ones(shape, bool)                               # names that could enter the day's fit
                for res in got:
                    usable &= res[0] == PRESENT
                if in_uni is not None:
                    usable &= in_uni
                n_regressors = 1 + len(s.more or ())
                own(missing & (usable.sum(axis=1) < n_regressors + 2)[:, None], TOO_FEW)
            own(missing & np.isinf(v), INFINITE)
            own(missing, CALC)
        elif isinstance(s, _TIME):
            period = getattr(s, "period", None)
            if isinstance(s, _ts.Delay) or isinstance(s, _ts.Delta):
                if isinstance(s, _ts.Delta):
                    for res in got:
                        inherit(missing, res)                               # today's input is missing
                # a value from period days ago that was missing is a short history today, with that day's reason
                for res in got:
                    eff = np.where((res[0] == HISTORY) & (res[1] != PRESENT), res[1], res[0]).astype(np.int8)
                    then = np.concatenate([np.zeros((period,) + shape[1:], np.int8), eff[:-period]]) if period else eff
                    own(missing & (then != PRESENT), HISTORY, then)
                own(missing, HISTORY)
            else:
                for res in got:
                    inherit(missing, res)                                   # today's input is missing
                window = period or getattr(s, "n_events", None)
                if window and not hasattr(s, "n_events"):
                    full = np.ones(shape, bool)
                    why = np.zeros(shape, np.int8)
                    for res in got:
                        present = pd.DataFrame(res[0] == PRESENT).rolling(window, min_periods=1).sum().to_numpy()
                        full &= present >= window
                        # an input short of history itself passes on why; with no why, it was warming up
                        eff = np.where((res[0] == HISTORY) & (res[1] != PRESENT), res[1], res[0])
                        counts = {c: pd.DataFrame(eff == c).rolling(window, min_periods=1).sum().to_numpy()
                                  for c in np.unique(eff) if c != PRESENT}
                        if counts:
                            reasons = np.array(list(counts))
                            best = reasons[np.argmax(np.stack(list(counts.values())), axis=0)]
                            seen = np.max(np.stack(list(counts.values())), axis=0) > 0
                            why = np.where((why == PRESENT) & seen, best, why).astype(np.int8)
                    own(missing & full & np.isinf(v), INFINITE)
                    own(missing & full, CALC)                               # a whole window, and still no value
                    own(missing, HISTORY, why)
                else:
                    own(missing, HISTORY)
        else:
            for res in got:
                inherit(missing, res)
            own(missing & np.isinf(v), INFINITE)
            own(missing, CALC)

        results[k] = (code, cause, origin)
        for c in kids:                                                     # free what no one else needs
            ck = c._cache_key()
            users[ck] -= 1
            if users[ck] == 0 and ck != order[-1]:
                results.pop(ck, None)

    code, cause, origin = results[order[-1]]
    return MissingReport(pd.DataFrame(code, index, columns), pd.DataFrame(cause, index, columns),
                         pd.DataFrame(origin, index, columns), labels)
