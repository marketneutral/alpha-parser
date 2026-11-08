"""ts_alpha, ts_resid (one regressor or several) and their shrunk versions, and ts_beta_shrunk's group prior (0.13.0)."""

import numpy as np
import pandas as pd
import pytest

from qex import qex, compute_context
from qex.distributed import SignalGraph
from qex.missing import explain_missing

P = 60
Y, X = "returns(1)", "returns(1, 'market')"


@pytest.fixture
def fit_data():
    rng = np.random.default_rng(2)
    n = 300
    idx = pd.bdate_range("2023-01-02", periods=n)
    m = rng.normal(0.0004, 0.01, n)
    cols = {}
    for i, (b, a) in enumerate([(0.6, 0), (0.9, 0), (1.2, 0), (1.6, 0), (1.0, 0.0003), (0.8, 0)]):
        cols[f"S{i}"] = 50 * np.cumprod(1 + a + b * m + rng.normal(0, 0.015, n))
    close = pd.DataFrame(cols, index=idx)
    close.iloc[100:105, 0] = np.nan                                      # a halt inside some windows
    sector = pd.DataFrame([["T", "T", "T", "E", "E", "U"]] * n, index=idx, columns=close.columns)
    return {"close": close, "market": pd.Series(100 * np.cumprod(1 + m), index=idx), "sector": sector}


def ev(expr, data):
    with compute_context():
        return qex(expr).evaluate(data)


def _window(data, stock, t):
    y, x = ev(Y, data)[stock], ev(X, data)[stock]
    w = pd.DataFrame({"y": y.iloc[t - P + 1:t + 1], "x": x.iloc[t - P + 1:t + 1]}).dropna()
    return w, y.iloc[t], x.iloc[t]


DAYS = [59, 60, 110, 150, 299]


def test_ts_alpha_and_ts_resid_are_the_window_regression(fit_data):
    a, r = ev(f"ts_alpha({Y}, {X}, {P})", fit_data), ev(f"ts_resid({Y}, {X}, {P})", fit_data)
    for t in DAYS:
        for s in fit_data["close"].columns:
            w, yt, xt = _window(fit_data, s, t)
            if len(w) < 20:
                assert np.isnan(a[s].iloc[t])
                continue
            slope, icpt = np.polyfit(w.x, w.y, 1)
            assert a[s].iloc[t] == pytest.approx(icpt, abs=1e-12)
            if np.isnan(yt):
                assert np.isnan(r[s].iloc[t])
            else:
                assert r[s].iloc[t] == pytest.approx(yt - icpt - slope * xt, abs=1e-12)


def test_shrunk_alpha_is_the_intercept_for_the_shrunk_slope(fit_data):
    b = ev(f"ts_beta_shrunk({Y}, {X}, {P})", fit_data)
    a, r = ev(f"ts_alpha_shrunk({Y}, {X}, {P})", fit_data), ev(f"ts_resid_shrunk({Y}, {X}, {P})", fit_data)
    for t in DAYS[2:]:
        for s in fit_data["close"].columns:
            w, yt, xt = _window(fit_data, s, t)
            icpt = (w.y - b[s].iloc[t] * w.x).mean()                     # least squares with the slope held fixed
            assert a[s].iloc[t] == pytest.approx(icpt, abs=1e-12)
            if not np.isnan(yt):
                assert r[s].iloc[t] == pytest.approx(yt - icpt - b[s].iloc[t] * xt, abs=1e-12)


def test_shrinking_moves_alpha_by_the_beta_it_takes_off(fit_data):
    """alpha_shrunk - alpha = (beta - beta_shrunk) * mean(x) over the paired days."""
    d = fit_data
    lhs = ev(f"ts_alpha_shrunk({Y}, {X}, {P})", d) - ev(f"ts_alpha({Y}, {X}, {P})", d)
    mx = ev(f"let x = {X}, xp = where(is_valid({Y}), x, None) in ts_mean(xp, {P})", d)
    rhs = (ev(f"ts_beta({Y}, {X}, {P})", d) - ev(f"ts_beta_shrunk({Y}, {X}, {P})", d)) * mx
    np.testing.assert_allclose(lhs.to_numpy(), rhs.to_numpy(), atol=1e-14, equal_nan=True)


def test_group_prior_is_each_groups_mean_and_spread(fit_data):
    d = fit_data
    b = ev(f"ts_beta({Y}, {X}, {P})", d)
    with compute_context():
        n_s2 = qex(f"ts_beta_shrunk({Y}, {X}, {P}, 'sector')")
        g = n_s2.evaluate(d)
    t = 200
    # recompute one day's shrinkage by hand from the betas and their standard errors
    w_paired, *_ = _window(d, "S0", t)
    for sector, members in (("T", ["S0", "S1", "S2"]), ("E", ["S3", "S4"])):
        prior = b.iloc[t][members]
        mu, tau2 = prior.mean(), prior.var()
        for s in members:
            w, *_ = _window(d, s, t)
            n = len(w)
            vx, vy, cov = w.x.var(), w.y.var(), w.x.cov(w.y)
            beta = cov / vx
            se2 = (vy - cov * beta) * (n - 1) / (n - 2) / ((n - 1) * vx)
            wt = tau2 / (tau2 + se2)
            assert g[s].iloc[t] == pytest.approx(wt * beta + (1 - wt) * mu, rel=1e-10)


def test_a_stock_alone_in_its_group_keeps_its_own_beta(fit_data):
    g = ev(f"ts_beta_shrunk({Y}, {X}, {P}, 'sector')", fit_data)
    b = ev(f"ts_beta({Y}, {X}, {P})", fit_data)
    np.testing.assert_allclose(g["S5"].to_numpy(), b["S5"].to_numpy(), equal_nan=True)


def test_a_stock_with_no_group_has_no_value(fit_data):
    d = dict(fit_data)
    d["sector"] = d["sector"].copy()
    d["sector"]["S1"] = np.nan
    g = ev(f"ts_beta_shrunk({Y}, {X}, {P}, 'sector')", d)
    assert g["S1"].isna().all() and g["S0"].notna().any()


def test_min_periods_and_groups_together(fit_data):
    a = ev(f"ts_beta_shrunk({Y}, {X}, {P}, 40, 'sector')", fit_data)
    b = ev(f"ts_beta_shrunk({Y}, {X}, {P}, 'sector')", fit_data)
    assert a.iloc[:39].isna().all().all() and b.iloc[20:39].notna().any().any()
    pd.testing.assert_frame_equal(a.iloc[60:], b.iloc[60:])


def test_groups_can_be_any_labels_expression(fit_data):
    d = dict(fit_data, mcap=fit_data["close"] * 1e6)
    out = ev(f"ts_resid_shrunk({Y}, {X}, {P}, quantile(field('mcap'), 2))", d)
    assert out.notna().any().any()


def test_without_groups_the_cache_key_is_unchanged():
    """Expressions are hashed by their cache key: ts_beta_shrunk's must not change for existing expressions."""
    assert qex(f"ts_beta_shrunk({Y}, {X}, {P})")._cache_key() == (
        'TsBetaShrunk', ('Returns', 1, 'close'), ('Returns', 1, 'market'), 60, 20)
    assert qex(f"ts_beta_shrunk({Y}, {X}, 252, 126)")._cache_key()[-1] == 126
    assert qex(f"ts_beta_shrunk({Y}, {X}, {P}, 'sector')")._cache_key()[-1] == 'sector'


@pytest.mark.parametrize("expr, message", [
    (f"ts_beta_shrunk({Y}, {X}, {P}, 30, 7)", "fifth argument is its groups"),
    (f"ts_alpha({Y}, {X}, {P}, 'sector')", "takes no groups.*ts_alpha_shrunk"),
    (f"ts_resid_shrunk({Y}, {X}, {P}, 0)", "a whole number, at least 1"),
    (f"ts_alpha_shrunk(group('sector'), {X}, {P})", "gives labels, not numbers"),
    (f"ts_beta_shrunk({Y}, {X}, {P}, 30, 'sector', 'industry')", "optionally min_periods and groups"),
])
def test_bad_arguments_are_refused_clearly(expr, message):
    with pytest.raises(ValueError, match=message):
        qex(expr)


def test_graph_and_missing_reports_work(fit_data):
    g = SignalGraph()
    g.add_signal(qex(f"ts_resid_shrunk({Y}, {X}, {P}, 'sector')"), name="r")
    out = g.compute(fit_data)["r"]
    pd.testing.assert_frame_equal(out, ev(f"ts_resid_shrunk({Y}, {X}, {P}, 'sector')", fit_data))
    report = explain_missing(qex(f"ts_alpha({Y}, {X}, {P})"), fit_data)
    assert report.codes.iloc[:19].ne(0).all().all()


# --- several regressors: a multi-factor alpha and residual ---------------------------------------------------------

FACTORS = ("mkt", "smb", "hml")
FF3 = "field('mkt'), field('smb'), field('hml')"


@pytest.fixture
def factor_data():
    rng = np.random.default_rng(3)
    n = 300
    idx = pd.bdate_range("2023-01-02", periods=n)
    f = rng.normal(0, 0.01, (n, 3))
    cols = {}
    for i in range(5):
        b = rng.normal([1, 0.3, -0.2], 0.3)
        cols[f"S{i}"] = 50 * np.cumprod(1 + 0.0002 * i + f @ b + rng.normal(0, 0.01, n))
    close = pd.DataFrame(cols, index=idx)
    close.iloc[100:105, 1] = np.nan                                       # a halt inside some windows
    out = {"close": close}
    out.update({k: pd.Series(f[:, j], index=idx) for j, k in enumerate(FACTORS)})
    return out


def test_several_regressors_are_one_joint_fit(factor_data):
    d = factor_data
    a, r = ev(f"ts_alpha({Y}, {FF3}, {P})", d), ev(f"ts_resid({Y}, {FF3}, {P})", d)
    y = ev(Y, d)
    f = {k: ev(f"field('{k}')", d) for k in FACTORS}
    for t in [59, 60, 110, 150, 299]:
        for s in d["close"].columns:
            w = pd.DataFrame({"y": y[s].iloc[t - P + 1:t + 1], **{k: f[k][s].iloc[t - P + 1:t + 1] for k in FACTORS}})
            w = w.dropna()
            if len(w) < 20:
                assert np.isnan(a[s].iloc[t])
                continue
            Z = np.column_stack([np.ones(len(w)), w[list(FACTORS)].to_numpy()])
            coef = np.linalg.lstsq(Z, w.y.to_numpy(), rcond=None)[0]
            assert a[s].iloc[t] == pytest.approx(coef[0], abs=1e-12)
            if not np.isnan(y[s].iloc[t]):
                fitted = coef[0] + sum(coef[j + 1] * f[k][s].iloc[t] for j, k in enumerate(FACTORS))
                assert r[s].iloc[t] == pytest.approx(y[s].iloc[t] - fitted, abs=1e-12)


def test_a_duplicated_regressor_gives_the_one_regressor_fit(factor_data):
    dup = ev(f"ts_alpha({Y}, field('mkt'), field('mkt'), {P})", factor_data)
    one = ev(f"ts_alpha({Y}, field('mkt'), {P})", factor_data)
    np.testing.assert_allclose(dup.to_numpy(), one.to_numpy(), atol=1e-14, equal_nan=True)


def test_a_fit_needs_k_plus_2_days(factor_data):
    a = ev(f"ts_alpha({Y}, {FF3}, 10, 3)", factor_data)               # 3 regressors: at least 5 days, not 3
    first = a.notna().any(axis=1).to_numpy().argmax()
    assert first == 5                                                 # returns start on row 1; 5 of them by row 5


def test_one_regressor_keeps_its_cache_key():
    assert qex(f"ts_alpha({Y}, field('mkt'), {P})")._cache_key() == (
        'TsAlpha', ('Returns', 1, 'close'), ('Field', 'mkt'), 60, 20)
    assert qex(f"ts_alpha({Y}, {FF3}, {P})")._cache_key() != qex(f"ts_alpha({Y}, field('mkt'), {P})")._cache_key()


@pytest.mark.parametrize("expr, message", [
    (f"ts_alpha({Y}, {P})", "at least one regressor"),
    (f"ts_resid({Y}, {FF3})", "a period"),
    (f"ts_alpha({Y}, {FF3}, {P}, 20, 5)", "a period and optionally min_periods"),
    (f"ts_alpha({Y}, {FF3}, 0)", "period must be a whole number"),
    (f"ts_resid({Y}, field('mkt'), group('sector'), {P})", "gives labels, not numbers"),
])
def test_bad_fit_arguments_are_refused_clearly(expr, message):
    with pytest.raises(ValueError, match=message):
        qex(expr)


def test_several_regressors_in_a_graph_and_missing_report(factor_data):
    expr = f"ts_resid({Y}, {FF3}, {P})"
    g = SignalGraph()
    g.add_signal(qex(expr), name="r")
    pd.testing.assert_frame_equal(g.compute(factor_data)["r"], ev(expr, factor_data))
    report = explain_missing(qex(expr), factor_data)
    assert any("field('hml')" in node for node in report.nodes)
