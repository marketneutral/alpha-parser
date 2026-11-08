"""Adjusted data and look-ahead: which expressions use an adjusted level in a way that sees the future.

An adjusted price is the price as traded times a factor A(t) that folds in every split and dividend after day t, up
to the end of the data. Its returns are point in time (between two days, the part of A that differs is known by the
later one), but its level is not: on any past day a stock's adjusted level is low partly because the stock later
split, and splits follow rises. So rank(close()) ranks stocks partly on their futures, and close() > 5 compares a
number that knows the future with a fixed dollar amount.

Each field declares an adjustment exponent (the data's manifest says it): +1 for an adjusted price, -1 for adjusted
volume (as traded / A), 0 for anything as traded or as filed (price, shares, volume_traded). The check follows the
exponent through an expression:

    multiplying adds exponents, dividing subtracts them, sqrt halves, power multiplies
    delay and the time-series operators keep it (ts_mean(close()) is still an adjusted level)
    ts_rank, ts_corr, ts_argmax and other scale-free time-series statistics give 0
    log turns exponent e into an additive shift; a difference of two shifts over time cancels (log(close()) - log(delay(close(), 5)))
    exp undoes log (exp(log(close())) is an adjusted level); exp of a plain number is a plain number
    adding, subtracting, max, min and comparing need the same exponent on both sides
    comparing with a constant needs exponent 0
    every cross-sectional and group operator needs exponent 0 in what it compares

so close() / ts_max(close(), 252) is fine (0), close() * volume is dollars traded (0), and rank(close()) is not.

It also checks units where the data declares them: adding a price to a share count is refused (a plain number, such
as 0 * returns(1), adds to anything).

    check(signal, fields, mode="refuse")   raise on the first problem ('refuse'), return them ('warn'), or skip ('off')

`fields` maps a field name to its declaration ({"adjustment": 1, "unit": "usd_per_share"}); fields it does not
declare are not checked, and neither is anything computed from them.
"""
from collections import namedtuple
from typing import Dict, List, Optional

from .signal import Signal

V = namedtuple("V", "e log unit")        # e: exponent (None: unknown); log: an additive shift; unit: tuple or None
UNKNOWN = V(None, False, None)
FREE = V(0, False, ())                   # a dimensionless number with nothing adjusted in it
WILD = V(0, False, None)                 # a constant: exponent 0, takes the other side's unit

UNITS = {"usd_per_share": (("shares", -1), ("usd", 1)), "usd": (("usd", 1),), "shares": (("shares", 1),),
         "count": (), "ratio": (), "percent": (), "number": ()}


class LookAhead(ValueError):
    """An expression that uses adjusted data in a way that sees the future."""


def _unit_mul(a, b, sign=1):
    if a is None or b is None:
        return None
    d = dict(a)
    for k, n in b:
        d[k] = d.get(k, 0) + sign * n
    return tuple(sorted((k, n) for k, n in d.items() if n))


def _unit_text(u):
    if u is None:
        return "?"
    if not u:
        return "a plain number"
    names = {(("shares", -1), ("usd", 1)): "dollars per share", (("usd", 1),): "dollars", (("shares", 1),): "shares"}
    return names.get(u, " ".join(f"{k}^{n}" for k, n in u))


def _classes():
    from . import operators as op, conditional as cond, crosssection as cs, groups as gr, timeseries as ts
    from . import primitives as pr, data as dt
    return op, cond, cs, gr, ts, pr, dt


_PRESERVE = ("TsMean", "TsSum", "TsMax", "TsMin", "Delay", "Ewma", "DecayLinear", "FillForward", "TsMeanEvents",
             "TsSumEvents", "TsStd", "TsStdEvents", "TsQuantile")
_FREE_TS = ("TsRank", "TsArgmax", "TsArgmin", "TsCorr", "TsSkew", "TsKurt", "TsCountEvents", "Latch")
_SQUARE = ("TsVar", "EwmaVar")
_PAIR_SUM = ("TsCov", "EwmaCov")
_PAIR_DIFF = ("TsBeta", "TsBetaShrunk", "TsBetaEwma")
_PAIR_FIRST = ("TsAlpha", "TsAlphaShrunk", "TsResid", "TsResidShrunk")    # in y's units: y - alpha - beta * x
_CALENDAR = ("DayOfWeek", "DayOfMonth", "MonthOfYear")


def check(signal: Signal, fields: Dict[str, dict], mode: str = "refuse", alpha: bool = True) -> List[dict]:
    """The look-ahead and unit problems in an expression: [{kind, part, message}]. Raises LookAhead on the first
    one when mode is 'refuse'.

    alpha: the value itself is compared across stocks (an alpha's book ranks it), so it must not be an adjusted level."""
    if mode == "off" or not fields:
        return []
    found: List[dict] = []
    v = _walk(signal, fields, {}, found, mode)
    if alpha and v.e is not None and (v.e != 0 or v.log):
        from .missing import label
        msg = (f"{label(signal)} is an adjusted level, and an alpha's value is compared across stocks: each stock's "
               f"adjusted level folds in its own later splits and dividends. Use a ratio over time, returns, or a "
               f"level as traded (field('price'), field('mcap')).")
        if mode == "refuse":
            raise LookAhead(msg)
        found.append(dict(kind="alpha is an adjusted level", part=label(signal), message=msg))
    return found


def _walk(s, fields, seen, found, mode):
    key = id(s)
    if key in seen:
        return seen[key]
    op, cond, cs, gr, ts, pr, dt = _classes()
    from .missing import children, label
    name = type(s).__name__
    kids = children(s)

    def w(x):
        return _walk(x, fields, seen, found, mode)

    def problem(kind, part, message):
        f = dict(kind=kind, part=label(part), message=message)
        if mode == "refuse":
            raise LookAhead(message)
        found.append(f)

    def level(v):
        return v.e is not None and (v.e != 0)

    def need_free(v, part, where):
        if v.e is None:
            return
        if level(v):
            what = "an adjusted level" if not v.log else "the log of an adjusted level"
            problem("cross-section of an adjusted level", part,
                    f"{where} compares {label(part)} across stocks, but it is {what}: each stock's adjusted level folds "
                    f"in its own later splits and dividends, so the comparison sees the future. Use a ratio over time "
                    f"(close() / ts_max(close(), 252)), returns, or a level as traded (field('price'), field('mcap')).")

    def same(a, b, part, what):
        if a.e is None or b.e is None:
            return UNKNOWN
        if a == WILD:
            if level(b):
                problem("adjusted level against a number", part,
                        f"{label(part)} {what} an adjusted level with a fixed number; the adjusted level folds in later "
                        f"splits and dividends, so the result depends on the future. Use field('price') for dollar levels.")
            return b
        if b == WILD:
            return same(b, a, part, what)
        if a.e != b.e or a.log != b.log:
            problem("mixed adjustments", part,
                    f"{label(part)} {what} values adjusted differently (exponents {a.e:g} and {b.e:g}): one side folds in "
                    f"later splits and dividends and the other does not. Put both on the same footing, for example "
                    f"close() with close(), or field('price') with field('price').")
            return UNKNOWN
        if a.unit and b.unit and a.unit != b.unit:       # a plain number mixes with anything; two units must agree
            problem("mixed units", part, f"{label(part)} {what} {_unit_text(a.unit)} with {_unit_text(b.unit)}.")
            return V(a.e, a.log, None)
        return V(a.e, a.log, a.unit if a.unit is not None else b.unit)

    if isinstance(s, op.NoneValue) or isinstance(s, op.Constant):
        out = WILD if not isinstance(getattr(s, "value", None), str) else UNKNOWN
    elif isinstance(s, dt.Field):
        d = fields.get(s.name)
        if not d or d.get("adjustment") is None:
            out = UNKNOWN
        else:
            out = V(d["adjustment"], False, UNITS.get(d.get("unit")))
    elif name in _CALENDAR:
        out = FREE
    elif isinstance(s, (gr.Group, gr.GroupCross, gr.Identifier)):
        out = UNKNOWN
    elif isinstance(s, pr.Returns) or isinstance(s, pr.Volatility):
        out = FREE
    elif isinstance(s, pr.Volume):
        d = fields.get("volume") or {}
        out = V(d["adjustment"], False, UNITS.get(d.get("unit"))) if d.get("adjustment") is not None else UNKNOWN
    elif isinstance(s, pr.Adv):
        p, v = fields.get(s.price_field) or {}, fields.get("volume") or {}
        if p.get("adjustment") is None or v.get("adjustment") is None:
            out = UNKNOWN
        else:
            if p["adjustment"] + v["adjustment"] != 0:
                problem("mixed adjustments", s, f"{label(s)} multiplies {s.price_field} by volume, and their "
                        f"adjustments do not cancel, so it is not dollars traded.")
            out = V(0, False, (("usd", 1),))
    elif isinstance(s, (op.Mul, op.Div)):
        a, b = w(s.left), w(s.right)
        if a.e is None or b.e is None:
            out = UNKNOWN
        elif a.log or b.log:
            if (a.log and level(a)) or (b.log and level(b)):
                problem("mixed adjustments", s, f"{label(s)} scales the log of an adjusted level; take a difference "
                        f"over time first (log(close()) - log(delay(close(), 5))).")
            out = UNKNOWN
        else:
            sign = 1 if isinstance(s, op.Mul) else -1
            out = V(a.e + sign * b.e, False, _unit_mul(a.unit if a != WILD else (), b.unit if b != WILD else (), sign))
    elif isinstance(s, (op.Add, op.Sub, op.Max, op.Min)):
        a, b = w(s.left), w(s.right)
        v = same(a, b, s, {"Add": "adds", "Sub": "subtracts", "Max": "takes the larger of", "Min": "takes the smaller of"}[name])
        if isinstance(s, op.Sub) and v.log and a.e is not None and a != WILD and b != WILD:
            v = V(0, False, ())                          # a difference of log levels: the shift cancels
        out = v
    elif isinstance(s, (op.Greater, op.Less, op.GreaterEqual, op.LessEqual, op.Equal, op.NotEqual)):
        same(w(s.left), w(s.right), s, "compares")
        out = FREE
    elif isinstance(s, (op.And, op.Or, op.Not, op.IsValid, op.IsFinite, op.Sign)):
        for k in kids:
            w(k)
        out = FREE
    elif isinstance(s, (op.Neg, op.Abs)):
        out = w(kids[0])
    elif isinstance(s, op.Log):
        v = w(kids[0])
        out = UNKNOWN if v.e is None else (V(v.e, True, ()) if v.e != 0 else FREE)
    elif isinstance(s, op.Exp):
        v = w(kids[0])                                   # exp(log(close())) is the adjusted level again
        out = UNKNOWN if v.e is None else (V(v.e, False, None) if v.log else FREE if v.e == 0 else UNKNOWN)
    elif isinstance(s, op.Sqrt):
        v = w(kids[0])
        out = UNKNOWN if v.e is None or v.log else V(v.e / 2, False, None)
    elif isinstance(s, op.Power):
        v = w(s.signal)
        p = getattr(s.exponent, "value", s.exponent)
        if isinstance(s.exponent, Signal) and not isinstance(s.exponent, op.Constant):
            w(s.exponent)
            out = UNKNOWN if v.e is None or v.e != 0 else FREE
        else:
            out = UNKNOWN if v.e is None or v.log else V(v.e * float(p), False, None)
    elif name == "Delta":
        v = w(s.signal)
        out = V(0, False, ()) if v.log else v
    elif name in _PRESERVE:
        out = w(kids[0])
        for k in kids[1:]:
            w(k)
    elif name in _FREE_TS:
        for k in kids:
            w(k)
        out = FREE
    elif name in _SQUARE:
        v = w(kids[0])
        out = UNKNOWN if v.e is None or v.log else V(2 * v.e, False, None)
    elif name == "MatchSelection":
        # similarity by correlation ignores each stock's scale, so a key may be an adjusted
        # level; a distance compares levels across stocks, so then the key must not be one
        k = w(s.key)
        if s.metric in ("path", "coint"):                         # they compound the key: it is daily returns
            need_free(k, s.key, f"ts_match(metric={s.metric!r}), whose key is daily returns, not a price,")
        elif s.metric == "euclidean":
            need_free(k, s.key, "ts_match()'s distance")
        if isinstance(s.groups, Signal):
            w(s.groups)
        out = UNKNOWN
    elif name == "TsMatchScore":                                # a similarity: a pure number
        w(s.selection)
        out = FREE
    elif name == "TsMatch":                                     # another stock's value, read into this one's column
        w(s.selection)
        need_free(w(s.value), s.value, "ts_match()")
        out = FREE
    elif name in ("TsFit", "RegressionFit"):                    # an internal fit: its inputs are walked
        for k in kids:
            w(k)
        out = UNKNOWN
    elif name in _PAIR_FIRST:
        w(s.fit)
        a, xs = w(s.fit.y), [w(x) for x in s.fit.xs]             # every regressor is walked, however many
        out = UNKNOWN if a.e is None or a.log or any(x.e is None or x.log for x in xs) else V(a.e, False, None)
    elif name in ("RegressionAlpha", "RegressionBeta"):         # like regression_neut: compared across stocks
        w(s.fit)
        for k in (s.fit.y,) + s.fit.xs:
            need_free(w(k), k, f"{label(s).split('(')[0]}()")
        out = FREE
    elif name in _PAIR_SUM or name in _PAIR_DIFF:
        a, b = w(s.signal1), w(s.signal2)
        if a.e is None or b.e is None or a.log or b.log:
            out = UNKNOWN
        else:
            out = V(a.e + b.e if name in _PAIR_SUM else a.e - b.e, False, None)
    elif isinstance(s, cond.Where):
        w(s.condition)
        out = same(w(s.if_true), w(s.if_false), s, "chooses between")
    elif isinstance(s, cond.Coalesce):
        out = same(w(s.x), w(s.y), s, "fills")
    elif name == "PcaBasis":
        out = w(kids[0])                                 # checked where pca_factor compares it across stocks
    elif s.__class__.__module__ == cs.__name__:
        for k in kids:
            part = k.signal if type(k).__name__ == "PcaBasis" else k
            need_free(w(k), part, f"{label(s).split('(')[0]}()")
        out = FREE
    elif s.__class__.__module__ == gr.__name__:
        groups = getattr(s, "groups", None)
        for k in kids:
            if k is groups:
                continue
            need_free(w(k), k, f"{label(s).split('(')[0]}()")
        out = FREE
    else:
        for k in kids:
            w(k)
        out = UNKNOWN
    seen[key] = out
    return out


def exponent(signal: Signal, fields: Dict[str, dict]) -> Optional[float]:
    """The adjustment exponent of an expression's value (None if it reads a field without a declaration)."""
    v = _walk(signal, fields, {}, [], "warn")
    return None if v.e is None else (v.e if not v.log else None)
