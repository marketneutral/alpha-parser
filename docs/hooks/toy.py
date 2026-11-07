"""Toy examples for the operator reference: a ```toy block lists qex expressions, one per line; each is evaluated on
the small dataset below and shown as a dates x stocks panel, the last one as the result. The panels are computed when
the site is built, so they are always what the code does."""

import html
import re

import numpy as np
import pandas as pd

DATES = pd.bdate_range("2024-01-02", periods=6)
STOCKS = ["A", "B", "C", "D"]
nan = np.nan


def _frame(rows):
    return pd.DataFrame(rows, index=DATES, columns=STOCKS, dtype=float)


def toy_data():
    """Four stocks over six days. C has no price on the third day; earnings are reported on three days."""
    close = _frame([[10, 20, 30, 40],
                    [11, 19, 30.5, 42],
                    [12, 18, nan, 41],
                    [11, 18.5, 31, 43],
                    [13, 17, 32, 44],
                    [14, 16, 31.5, 46]])
    volume = _frame([[100, 300, 200, 400],
                     [120, 280, 210, 380],
                     [150, 260, nan, 500],
                     [110, 300, 260, 420],
                     [200, 250, 240, 440],
                     [180, 240, 230, 600]])
    surprise = _frame([[nan, nan, nan, nan],
                       [0.05, nan, nan, nan],
                       [nan, nan, nan, nan],
                       [nan, nan, -0.02, nan],
                       [nan, -0.01, nan, 0.03],
                       [nan, nan, nan, nan]])
    sector = pd.DataFrame([["Tech", "Tech", "Energy", "Energy"]] * len(DATES), index=DATES, columns=STOCKS)
    market = pd.Series([100, 101, 100.5, 102, 103, 102.5], index=DATES, dtype=float)
    return {"close": close, "open": close * 0.99, "high": close * 1.01, "low": close * 0.98, "volume": volume,
            "surprise": surprise, "sector": sector, "market": market}


def _fmt_panel(frame):
    """Cell strings: labels as they are; numbers with the fewest decimals that show every value, at most 3 below 1 and 2 above."""
    vals = frame.to_numpy()
    if not all(pd.api.types.is_numeric_dtype(d) for d in frame.dtypes):
        return [[("" if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v)) for v in row]
                for row in vals]
    num = vals.astype(float)
    finite = num[np.isfinite(num)]
    dec, most = 0, (3 if finite.size and np.abs(finite).max() < 1 else 2)
    for d in range(most + 1):
        dec = d
        if np.allclose(finite, np.round(finite, d), atol=1e-9):
            break
    def one(v):
        if np.isnan(v):
            return ""
        if np.isinf(v):
            return "∞" if v > 0 else "−∞"
        s = f"{v:.{dec}f}"
        if float(s) == 0:
            s = s.lstrip("-")                              # no −0.00
        return s.replace("-", "−")
    return [[one(v) for v in row] for row in num]


def panel(expr, data, result=False):
    from qex import qex, compute_context
    with compute_context():
        out = qex(expr).evaluate(data)
    if isinstance(out, pd.Series):
        out = pd.DataFrame({s: out for s in STOCKS})
    out = out.reindex(index=DATES, columns=STOCKS)
    cells = _fmt_panel(out)
    head = "".join(f"<th>{s}</th>" for s in STOCKS)
    body = "".join(
        f"<tr><th>{d:%b %d}</th>" + "".join(f'<td class="{"na" if not c else ""}">{c or "·"}</td>' for c in row)
        + "</tr>" for d, row in zip(DATES, cells))
    cls = "toy-panel toy-result" if result else "toy-panel"
    return (f'<figure class="{cls}"><figcaption><code>{html.escape(expr)}</code></figcaption>'
            f'<table class="toy-table"><thead><tr><th></th>{head}</tr></thead><tbody>{body}</tbody></table></figure>')


def render(block, inputs=False):
    """Panels for each expression in the block; the last is the result, unless inputs (```toy inputs) says all are."""
    exprs = [l.strip() for l in block.strip().splitlines() if l.strip()]
    data = toy_data()
    parts = [panel(e, data, result=(not inputs and i == len(exprs) - 1)) for i, e in enumerate(exprs)]
    return '<div class="toy">' + "".join(parts) + "</div>"


_BLOCK = re.compile(r"^```toy( inputs)?\n(.*?)^```\s*$", re.S | re.M)


def blocks(markdown):
    """[(block, inputs)] for each ```toy block."""
    return [(m.group(2), bool(m.group(1))) for m in _BLOCK.finditer(markdown)]


def on_page_markdown(markdown, **kwargs):
    return _BLOCK.sub(lambda m: render(m.group(2), inputs=bool(m.group(1))), markdown)
