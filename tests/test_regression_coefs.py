"""regression_alpha and regression_beta: the intercept and slopes of regression_neut's daily cross-sectional fit."""

import numpy as np
import pandas as pd
import pytest

from qex import qex, cross_section_universe
from qex.missing import explain_missing, label

E = "returns(1), delay(field('val'), 1), delay(field('mom'), 1)"
PREMIA = {"alpha": 0.0005, "val": 0.002, "mom": -0.001}


@pytest.fixture
def panel():
    rng = np.random.default_rng(5)
    T, N = 250, 60
    idx = pd.bdate_range("2024-01-01", periods=T)
    names = [f"S{i:02d}" for i in range(N)]
    val = pd.DataFrame(rng.normal(0, 1, (T, N)), idx, names)
    mom = pd.DataFrame(rng.normal(0, 1, (T, N)), idx, names)
    r = PREMIA["alpha"] + PREMIA["val"] * val.shift(1) + PREMIA["mom"] * mom.shift(1) + rng.normal(0, 0.01, (T, N))
    close = 50 * (1 + r.fillna(0)).cumprod()
    close.iloc[10:13, 3] = np.nan
    return {"close": close, "val": val, "mom": mom}


def _hand_fits(d):
    """Per day, numpy's least squares of y on [1, x1, x2] over the names with all three."""
    y, x1, x2 = (qex(e).evaluate(d) for e in ("returns(1)", "delay(field('val'), 1)", "delay(field('mom'), 1)"))
    out = []
    for t in range(len(y)):
        w = pd.DataFrame({"y": y.iloc[t], "a": x1.iloc[t], "b": x2.iloc[t]}).dropna()
        if len(w) < 4:
            out.append(None)
            continue
        Z = np.column_stack([np.ones(len(w)), w.a, w.b])
        out.append(np.linalg.lstsq(Z, w.y.to_numpy(), rcond=None)[0])
    return out


def test_coefficients_are_the_days_least_squares_fit(panel):
    a = qex(f"regression_alpha({E})").evaluate(panel)
    b1, b2 = (qex(f"regression_beta({E}, {j})").evaluate(panel) for j in (1, 2))
    for t, coef in enumerate(_hand_fits(panel)):
        if coef is None:
            assert a.iloc[t].isna().all() and b1.iloc[t].isna().all()
            continue
        for frame, c in ((a, coef[0]), (b1, coef[1]), (b2, coef[2])):
            assert frame.iloc[t].nunique() == 1                       # one number for the day, every stock
            assert frame.iloc[t, 0] == pytest.approx(c, abs=1e-14)


def test_fama_macbeth_recovers_planted_premia(panel):
    a = qex(f"regression_alpha({E})").evaluate(panel).iloc[:, 0]
    b1 = qex(f"regression_beta({E}, 1)").evaluate(panel).iloc[:, 0]
    b2 = qex(f"regression_beta({E}, 2)").evaluate(panel).iloc[:, 0]
    for series, true in ((a, PREMIA["alpha"]), (b1, PREMIA["val"]), (b2, PREMIA["mom"])):
        mean, se = series.mean(), series.std() / np.sqrt(series.count())
        assert abs(mean - true) < 4 * se                               # the time-series average is the premium


def test_regression_neut_is_y_minus_the_fit(panel):
    y = qex("returns(1)").evaluate(panel)
    x = qex("delay(field('val'), 1)").evaluate(panel)
    neut = qex("regression_neut(returns(1), delay(field('val'), 1))").evaluate(panel)
    a = qex("regression_alpha(returns(1), delay(field('val'), 1))").evaluate(panel)
    b = qex("regression_beta(returns(1), delay(field('val'), 1))").evaluate(panel)
    diff = (neut - (y - a - b * x)).where(neut.notna())
    assert np.nanmax(np.abs(diff.to_numpy())) < 1e-15


def test_a_day_needs_k_plus_2_names(panel):
    d = dict(panel)
    d["val"] = panel["val"].copy()
    d["val"].iloc[100, 3:] = np.nan                                   # three names left on that day (and the next)
    b = qex(f"regression_beta({E}, 1)").evaluate(d)
    assert b.iloc[101].isna().all()                                   # 3 names < 2 regressors + 2
    report = explain_missing(qex(f"regression_beta({E}, 1)"), d)
    assert report.text(int(report.codes.iloc[101, 0])) == "too few names"


def test_the_fit_uses_the_universe_and_every_stock_gets_the_day(panel):
    members = pd.DataFrame(True, index=panel["close"].index, columns=panel["close"].columns)
    members.iloc[:, :20] = False
    with cross_section_universe(members):
        inside = qex(f"regression_beta({E}, 1)").evaluate(panel)
    alone = {k: (v.iloc[:, 20:] if isinstance(v, pd.DataFrame) else v) for k, v in panel.items()}
    expected = qex(f"regression_beta({E}, 1)").evaluate(alone)
    np.testing.assert_allclose(inside.iloc[:, 0].to_numpy(), expected.iloc[:, 0].to_numpy(), atol=1e-14,
                               equal_nan=True)
    assert inside.iloc[:, :20].notna().any().all()                    # outside the universe, still the day's number


@pytest.mark.parametrize("expr, message", [
    (f"regression_beta({E})", "needs to know which slope"),
    (f"regression_beta({E}, 3)", "a whole number from 1 to 2"),
    (f"regression_beta({E}, 1.5)", "a whole number from 1 to 2"),
    ("regression_beta(returns(1), group('sector'), 1)", "gives labels, not numbers"),
])
def test_bad_arguments_are_refused_clearly(expr, message):
    with pytest.raises(ValueError, match=message):
        qex(expr)


def test_labels_read_as_written():
    assert label(qex(f"regression_beta({E}, 2)")) == f"regression_beta({E}, 2)"
    assert label(qex(f"regression_alpha({E})")) == f"regression_alpha({E})"


def test_adjusted_levels_are_refused_as_for_regression_neut():
    from qex.adjust import check
    fields = {"close": dict(adjustment=1, unit="usd_per_share")}
    for expr in ("regression_beta(returns(1), close(), 1)", "regression_alpha(close(), returns(5))"):
        assert [f["kind"] for f in check(qex(expr), fields, mode="warn")] == ["cross-section of an adjusted level"]
