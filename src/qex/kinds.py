"""What an expression gives: numbers or labels, checked when it is parsed.

Every qex expression gives a value per date and stock (or one value, for a constant). Most give numbers. A few give
labels: group('sector'), group_cross(...), a string literal, and coalesce or where whose branches are labels. Labels
can be compared (== and !=), used to group (the groups of a group operator), and passed through coalesce and where.
Anywhere else a number is needed, and an expression that would hand an operator labels is refused before anything
runs, with a message that names the part giving labels and says what to write instead.

A field read with field('name') is taken to be numbers: read a field of labels with group('name').
"""
from typing import Optional

from .signal import Signal

NUMBERS, LABELS, EITHER = "numbers", "labels", "either"      # EITHER: None, which fits wherever its branch goes


def _classes():
    from . import operators as op, conditional as cond, groups as gr
    return op, cond, gr


def kind(signal: Signal) -> str:
    """'numbers' or 'labels' (or 'either', for None): what the expression gives. Raises ValueError, naming the part,
    where it hands labels to an operator that needs numbers."""
    return _check(signal, {})


def _label(signal) -> str:
    from .missing import label
    try:
        return label(signal)
    except Exception:                                     # a node label() does not know: its class name will do
        return type(signal).__name__


def _refuse(part, where: str):
    text = _label(part)
    raise ValueError(
        f"{text} gives labels, not numbers, and {where} needs numbers. Compare labels "
        f"({text} == 'Technology') or group by them (group_demean(x, {text}))."
    )


def _join(a: str, b: str, node) -> str:
    """The kind of a value that is a or b (coalesce, the branches of where)."""
    if a == EITHER:
        return b
    if b == EITHER or a == b:
        return a
    raise ValueError(f"{_label(node)} mixes labels and numbers: both sides must give the same kind")


def _check(s: Signal, seen: dict) -> str:
    key = id(s)
    if key in seen:
        return seen[key]
    op, cond, gr = _classes()
    from .missing import children

    def need_numbers(part, where):
        if _check(part, seen) == LABELS:
            _refuse(part, where)

    name = f"{_label_name(s)}"
    if isinstance(s, op.NoneValue):
        out = EITHER
    elif isinstance(s, op.Constant):
        out = LABELS if isinstance(s.value, str) else NUMBERS
    elif isinstance(s, (gr.Group, gr.GroupCross)):
        for part in children(s):
            _check(part, seen)                            # a crossed grouping's parts may be labels or numbers
        out = LABELS
    elif isinstance(s, cond.Coalesce):
        out = _join(_check(s.x, seen), _check(s.y, seen), s)
    elif isinstance(s, cond.Where):
        need_numbers(s.condition, "the condition of where")
        out = _join(_check(s.if_true, seen), _check(s.if_false, seen), s)
    elif isinstance(s, (op.Equal, op.NotEqual)):
        _check(s.left, seen)                              # labels compare with labels, numbers with numbers
        _check(s.right, seen)
        out = NUMBERS
    elif isinstance(s, tuple(c for c in vars(gr).values() if isinstance(c, type) and issubclass(c, Signal)
                             and c.__module__ == gr.__name__)):
        groups = getattr(s, "groups", None)
        for part in children(s):
            if part is groups:
                _check(part, seen)                        # groups are labels, or numbers used as labels
            else:
                need_numbers(part, name)
        out = NUMBERS
    else:
        groups = getattr(s, "groups", None)                       # e.g. ts_beta_shrunk's prior, by group
        for part in children(s):
            if part is groups:
                _check(part, seen)
            else:
                need_numbers(part, name)
        out = NUMBERS
    seen[key] = out
    return out


def _label_name(s: Signal) -> str:
    """How an operator is written, for the message: rank, +, ts_mean."""
    from .missing import _snake, _SYMBOLS
    n = type(s).__name__
    return f"'{_SYMBOLS[n]}'" if n in _SYMBOLS else "'-'" if n == "Neg" else "'~'" if n == "Not" else f"{_snake(n)}()"


def check(signal: Signal) -> Signal:
    """The signal, after checking it hands no operator labels where it needs numbers."""
    kind(signal)
    return signal


def numbers_only(signal: Signal, what: str = "an alpha") -> Optional[str]:
    """None if the expression gives numbers, else a sentence saying why it cannot be `what`."""
    k = kind(signal)
    if k == LABELS:
        text = _label(signal)
        return (f"{text} gives labels, not numbers, so it cannot be {what}: compare them ({text} == 'Technology') "
                f"or group by them (group_rank(x, {text})).")
    return None
