"""estimate(name, output, inputs..., window=, refit=, ...): the built-in estimators are today's operators (the same
node, the same numbers), the registry, the protocol on every axis, and complexity. Learned estimators (the stump, the
tree) live in qex-estimators, with their own tests; here a small learned switch stands in for one."""
import numpy as np
import pandas as pd
import pytest

import qex as Q
from qex import compute_context, explain_missing, qex
from qex.estimate import Estimator, EstimateFit, register_estimator, get_estimator, _REGISTRY


@pytest.fixture(scope="module")
def panel():
    rng = np.random.default_rng(3)
    T, N = 600, 24
    idx = pd.bdate_range("2016-01-01", periods=T)
    names = [f"S{i:02d}" for i in range(N)]
    market = rng.normal(0, 0.01, T)
    r = market[:, None] * rng.uniform(0.5, 1.5, N) + rng.normal(0, 0.01, (T, N))
    close = pd.DataFrame(50 * np.exp(np.cumsum(r, axis=0)), idx, names)
    sector = pd.DataFrame([[("A", "B", "C")[i % 3] for i in range(N)]] * T, index=idx, columns=names)
    return {"close": close, "market": pd.Series(100 * np.exp(np.cumsum(market)), idx), "sector": sector}


def _ev(expr, d):
    with compute_context():
        return qex(expr).evaluate(d)


class _Switch(Estimator):
    """A stand-in learned estimator: hold r1 if it beat r2 on average over the window, else r2. It learns one side
    (dof 1, declared 3 to stand for a stump), and measures what it used: |2w - 1|, w the share of days r1 won."""
    name, version, axis = "test_switch", "1", "series"
    n_inputs, outputs = (3, 3), ("w1", "p")
    dof = 3

    def fit(self, window, hyper, rng):
        d = window[1:, 1] - window[1:, 2]
        d = d[np.isfinite(d)]
        return None if len(d) < 10 else {"w": float(d.mean() > 0), "share": float((d > 0).mean())}

    def step(self, state, row, hyper):
        return {"w1": state["w"], "p": 0.5}

    def realized_dof(self, state, fit):
        return abs(2 * state["share"] - 1)


if "test_switch" not in _REGISTRY:
    register_estimator(_Switch)

SW = "estimate('test_switch', '{}', returns(1, 'market'), returns(1, 'market'), returns(5, 'market'), window=60)"


# --- the built-ins are the operators ----------------------------------------------------------------------------------

SAME = [
    ("ts_beta(returns(1), returns(1, 'market'), 63)",
     "estimate('ols', 'beta1', returns(1), returns(1, 'market'), window=63)"),
    ("ts_beta(returns(1), returns(1, 'market'), 63, 40)",
     "estimate('ols', 'beta1', returns(1), returns(1, 'market'), window=63, min_periods=40)"),
    ("ts_alpha(returns(1), returns(1, 'market'), 63)",
     "estimate('ols', 'alpha', returns(1), returns(1, 'market'), window=63)"),
    ("ts_alpha(returns(1), returns(1, 'market'), returns(5), 63)",
     "estimate('ols', 'alpha', returns(1), returns(1, 'market'), returns(5), window=63)"),
    ("ts_resid(returns(1), returns(1, 'market'), returns(5), 63, 30)",
     "estimate('ols', 'resid', returns(1), returns(1, 'market'), returns(5), window=63, min_periods=30)"),
    ("ts_beta_shrunk(returns(1), returns(1, 'market'), 63)",
     "estimate('ols_shrunk', 'beta1', returns(1), returns(1, 'market'), window=63)"),
    ("ts_beta_shrunk(returns(1), returns(1, 'market'), 63, 'sector')",
     "estimate('ols_shrunk', 'beta1', returns(1), returns(1, 'market'), window=63, groups='sector')"),
    ("ts_alpha_shrunk(returns(1), returns(1, 'market'), 63, 30, 'sector')",
     "estimate('ols_shrunk', 'alpha', returns(1), returns(1, 'market'), window=63, min_periods=30, groups='sector')"),
    ("ts_resid_shrunk(returns(1), returns(1, 'market'), 63)",
     "estimate('ols_shrunk', 'resid', returns(1), returns(1, 'market'), window=63)"),
    ("regression_neut(returns(5), returns(20))", "estimate('xs_ols', 'resid', returns(5), returns(20))"),
    ("regression_alpha(returns(1), returns(5), returns(20))",
     "estimate('xs_ols', 'alpha', returns(1), returns(5), returns(20))"),
    ("regression_beta(returns(1), returns(5))", "estimate('xs_ols', 'beta1', returns(1), returns(5))"),
    ("regression_beta(returns(1), returns(5), returns(20), 2)",
     "estimate('xs_ols', 'beta2', returns(1), returns(5), returns(20))"),
    ("pca_factor(returns(1), 120, 1)", "estimate('pca', 'beta1', returns(1), window=120)"),
    ("pca_factor(returns(1), 120, 2, 5)", "estimate('pca', 'beta2', returns(1), window=120, refit=5)"),
]


@pytest.mark.parametrize("named, est", SAME)
def test_a_built_in_estimate_is_the_named_operator(panel, named, est):
    a, b = qex(named), qex(est)
    assert a._cache_key() == b._cache_key()                               # one node: one cached computation
    va, vb = _ev(named, panel), _ev(est, panel)
    assert va.notna().to_numpy().sum() > 0
    assert np.array_equal(va.to_numpy(float), vb.to_numpy(float), equal_nan=True)


def test_a_joint_fit_s_slopes_are_its_regression(panel):
    """beta_j of a joint fit, which no named operator reads: the slope of a direct least squares on one window."""
    b2 = _ev("estimate('ols', 'beta2', returns(1), returns(1, 'market'), returns(5), window=63)", panel)
    y = _ev("returns(1)", panel)
    x1, x2 = _ev("returns(1, 'market')", panel), _ev("returns(5)", panel)
    t, c = 300, "S04"
    rows = slice(t - 62, t + 1)
    Y, X = y[c].iloc[rows], np.column_stack([x1[c].iloc[rows], x2[c].iloc[rows]])
    ok = np.isfinite(Y.to_numpy()) & np.isfinite(X).all(axis=1)
    A = np.column_stack([np.ones(ok.sum()), X[ok]])
    coef = np.linalg.lstsq(A, Y.to_numpy()[ok], rcond=None)[0]
    assert b2[c].iloc[t] == pytest.approx(coef[2], rel=1e-8)


def test_outputs_of_one_fit_share_it(panel):
    a, b = qex(SW.format("w1")), qex(SW.format("p"))
    assert a.fit._cache_key() == b.fit._cache_key() and a._cache_key() != b._cache_key()


# --- refusals ----------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("expr, says", [
    ("estimate('nope', 'x', returns(1))", "not registered"),
    ("estimate(returns(1), 'x')", "in quotes"),
    ("estimate('ols', 'beta3', returns(1), returns(5), returns(20), window=63)", "no output 'beta3'"),
    ("estimate('ols', 'beta1', returns(1), window=63)", "takes 2 or more inputs"),
    ("estimate('ols', 'beta1', returns(1), returns(5))", "needs window="),
    ("estimate('ols', 'beta1', returns(1), returns(5), window=63, refit=5)", "refit is 1"),
    ("estimate('ols', 'beta1', returns(1), returns(5), window=63.5)", "whole number"),
    ("estimate('ols', 'beta1', returns(1), returns(5), window=63, lambda_=3)", "no option 'lambda_'"),
    ("estimate('xs_ols', 'resid', returns(1), returns(5), window=5)", "window is 1"),
    ("estimate('ols@2', 'beta1', returns(1), returns(5), window=63)", "no version '2'"),
    ("estimate('ols', 'beta1', returns(1), 5, window=63)", "inputs are expressions"),
    ("estimate('test_switch', 'w1', group('sector'), returns(5), returns(20), window=200)", "labels"),
    ("ts_beta(returns(1), returns(5), period=63)", "by position"),
])
def test_refused_with_a_message(expr, says):
    with pytest.raises(ValueError, match=says):
        qex(expr)


def test_a_version_by_name():
    assert qex("estimate('ols@1', 'beta1', returns(1), returns(5), window=63)")._cache_key() == \
        qex("ts_beta(returns(1), returns(5), 63)")._cache_key()


# --- the registry ------------------------------------------------------------------------------------------------------

class _MeanGap(Estimator):
    name, version, axis = "test_mean_gap", "1", "series"
    dof = 0
    n_inputs, outputs = (1, 1), ("z",)

    def fit(self, window, hyper, rng):
        x = window[:, 0][np.isfinite(window[:, 0])]
        return None if len(x) < 20 else (x.mean(), x.std(ddof=1))

    def step(self, state, row, hyper):
        return {"z": (row[0] - state[0]) / state[1]}


def test_register_a_protocol_estimator_and_use_it(panel):
    if "test_mean_gap" not in _REGISTRY:
        register_estimator(_MeanGap)
    z = _ev("estimate('test_mean_gap', 'z', returns(1, 'market'), window=60, refit=10)", panel)
    m = _ev("returns(1, 'market')", panel).iloc[:, 0]
    assert (z.nunique(axis=1, dropna=True) <= 1).all()                     # the same for every stock
    assert z.notna().any().any()
    t = z.index.get_loc(z.dropna().index[0])                              # the first refit: its fit is the window's
    w = m.iloc[t - 59:t + 1]
    assert z.iloc[t, 0] == pytest.approx((m.iloc[t] - w.mean()) / w.std(ddof=1))
    with pytest.raises(ValueError, match="already registered"):
        register_estimator(_MeanGap)


def test_an_estimator_builds_or_fits_not_both():
    class Neither(Estimator):
        name = "test_neither"
    with pytest.raises(TypeError, match="either compute, or fit and step"):
        register_estimator(Neither)

    class NoAxis(_MeanGap):
        name, axis = "test_no_axis", "sideways"
    with pytest.raises(ValueError, match="its axis is"):
        register_estimator(NoAxis)


def test_the_latest_version_by_default():
    class V2(_MeanGap):
        name, version = "test_versions", "2"

    class V10(_MeanGap):
        name, version = "test_versions", "10"
    for cls in (V2, V10):
        if cls.version not in _REGISTRY.get("test_versions", {}):
            register_estimator(cls)
    assert get_estimator("test_versions").version == "10"                # 10 after 2, not before it
    assert get_estimator("test_versions@2").version == "2"


def test_estimators_lists_them():
    t = Q.estimators()
    assert {"ols", "ols_shrunk", "xs_ols", "pca"} <= set(t["name"])


# --- the protocol, per stock -------------------------------------------------------------------------------------------

class _StockZ(Estimator):
    """Each stock's value today against its own window: (x - mean) / sd, fitted on the window's rows."""
    dof = 0                                        # descriptive: nothing searched for
    name, version, axis = "test_stock_z", "1", "stock"
    n_inputs, outputs = (1, 1), ("z",)

    def fit(self, window, hyper, rng):
        x = window[:, 0][np.isfinite(window[:, 0])]
        return None if len(x) < 10 else (x.mean(), x.std(ddof=1))

    def step(self, state, row, hyper):
        return {"z": (row[0] - state[0]) / state[1]}


class _StockZFast(_StockZ):
    """The same, every stock at once: what fit_many and step_many are for."""
    name = "test_stock_z_fast"

    def fit_many(self, window, hyper, rngs):
        x = window[:, :, 0]                                               # stocks x rows
        n = np.isfinite(x).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean, sd = np.nanmean(x, axis=1), np.nanstd(x, axis=1, ddof=1)
        return [None if k < 10 else (m, v) for k, m, v in zip(n, mean, sd)]


def _reg(cls):
    if cls.name not in _REGISTRY:
        register_estimator(cls)


def test_a_per_stock_estimator_fits_each_stock_s_own_window(panel):
    _reg(_StockZ)
    expr = "estimate('test_stock_z', 'z', returns(5), window=40, refit=10)"
    z, x = _ev(expr, panel), _ev("returns(5)", panel)
    assert z.notna().to_numpy().sum() > 1000
    assert (z.nunique(axis=1) > 1).any()                                  # each stock its own
    t = z.index.get_loc(z["S03"].dropna().index[0])                       # the first refit with a fit
    for c in ("S03", "S17"):
        w = x[c].iloc[t - 39:t + 1]
        assert z[c].iloc[t] == pytest.approx((x[c].iloc[t] - w.mean()) / w.std(ddof=1))
        assert z[c].iloc[t + 3] == pytest.approx((x[c].iloc[t + 3] - w.mean()) / w.std(ddof=1))   # held to the next refit


def test_fit_many_is_the_same_as_fit_stock_by_stock(panel):
    _reg(_StockZ), _reg(_StockZFast)
    a = _ev("estimate('test_stock_z', 'z', returns(5), window=40, refit=10)", panel)
    b = _ev("estimate('test_stock_z_fast', 'z', returns(5), window=40, refit=10)", panel)
    assert np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True)


def test_a_per_stock_estimator_is_point_in_time_and_start_independent(panel):
    _reg(_StockZ)
    expr = "estimate('test_stock_z', 'z', returns(1), window=40, refit=10)"
    base = _ev(expr, panel)
    later = dict(panel, close=panel["close"].copy())
    later["close"].iloc[401:, :] = later["close"].iloc[401:, :] * np.exp(np.random.default_rng(4).normal(0, .05, (len(base) - 401, base.shape[1])))
    moved = _ev(expr, later)
    assert np.allclose(base.iloc[:401].to_numpy(), moved.iloc[:401].to_numpy(), equal_nan=True)
    assert not np.allclose(base.iloc[402:].to_numpy(), moved.iloc[402:].to_numpy(), equal_nan=True)
    short = _ev(expr, {k: (v.iloc[100:] if isinstance(v, (pd.DataFrame, pd.Series)) else v) for k, v in panel.items()})
    common = short.dropna(how="all").index
    assert len(common) > 300
    assert np.allclose(base.loc[common].to_numpy(), short.loc[common].to_numpy(), equal_nan=True)


# --- the protocol, across stocks ---------------------------------------------------------------------------------------

class _XsGap(Estimator):
    """Each stock's value against the median of the universe's values over the window (pooled): a cross-sectional
    (window 1) or panel (longer) estimator."""
    dof = 0                                        # descriptive: nothing searched for
    name, version, axis = "test_xs_gap", "1", "cross_section"
    n_inputs, outputs = (1, 1), ("gap",)

    def fit(self, window, hyper, rng, members=None):
        v = window[:, :, 0][members]
        return None if v.size < 5 else float(np.median(v))

    def step(self, state, row, hyper, members=None):
        return {"gap": row[:, 0] - state}


def test_a_cross_sectional_estimator_sees_the_day_s_universe(panel):
    from qex import cross_section_universe
    _reg(_XsGap)
    x = _ev("returns(5)", panel)
    uni = pd.DataFrame(True, x.index, x.columns)
    uni.iloc[:, :6] = False                                               # six stocks never in the universe
    with compute_context(), cross_section_universe(uni):
        g = qex("estimate('test_xs_gap', 'gap', returns(5), window=1, refit=1)").evaluate(panel)
    assert g.iloc[:, :6].isna().all().all()                               # none outside the universe
    t = 200
    inside = x.iloc[t, 6:]
    assert np.allclose(g.iloc[t, 6:], inside - inside.median())           # the day's members' median


def test_a_panel_estimator_pools_the_window_and_is_point_in_time(panel):
    _reg(_XsGap)
    expr = "estimate('test_xs_gap', 'gap', returns(5), window=20, refit=5)"
    g, x = _ev(expr, panel), _ev("returns(5)", panel)
    r = [i for i in range(len(g)) if g.iloc[i].notna().any()][0]          # the first refit
    assert np.allclose(g.iloc[r], x.iloc[r] - np.nanmedian(x.iloc[r - 19:r + 1].to_numpy()))
    later = dict(panel, close=panel["close"].copy())
    later["close"].iloc[301:, :] = later["close"].iloc[301:, :] * 1.5 * np.exp(np.random.default_rng(6).normal(0, .1, (len(g) - 301, g.shape[1])))
    m = _ev(expr, later)
    assert np.allclose(g.iloc[:301].to_numpy(), m.iloc[:301].to_numpy(), equal_nan=True)
    assert not np.allclose(g.iloc[302:].to_numpy(), m.iloc[302:].to_numpy(), equal_nan=True)


# --- the named operators are sugar -------------------------------------------------------------------------------------

R, M = ("Returns", 1, "close"), ("Returns", 1, "market")

PINNED = [                                                    # the keys these operators have had: alphas are known by them
    ("ts_beta(returns(1), returns(1, 'market'), 63)", ('TsBeta', R, M, 63, 20)),
    ("ts_beta(returns(1), returns(1, 'market'), 10)", ('TsBeta', R, M, 10, 10)),
    ("ts_alpha(returns(1), returns(1, 'market'), returns(5), 63, 30)",
     ('TsAlpha', R, M, 63, 30, ("Returns", 5, "close"))),
    ("ts_resid(returns(1), returns(1, 'market'), 63)", ('TsResid', R, M, 63, 20)),
    ("ts_beta_shrunk(returns(1), returns(1, 'market'), 63)", ('TsBetaShrunk', R, M, 63, 20)),
    ("ts_resid_shrunk(returns(1), returns(1, 'market'), 63, 'sector')", ('TsResidShrunk', R, M, 63, 20, 'sector')),
    ("regression_neut(returns(1), returns(5))", ('RegressionNeut', R, ("Returns", 5, "close"))),
    ("regression_beta(returns(1), returns(5), returns(20), 2)",
     ('RegressionBeta', R, ("Returns", 5, "close"), ("Returns", 20, "close"), 2)),
    ("pca_factor(returns(1), 252, 2)", ('PcaFactor', R, 252, 2, 21)),
]


@pytest.mark.parametrize("expr, key", PINNED)
def test_the_operators_keep_their_cache_keys(expr, key):
    assert qex(expr)._cache_key() == key


def test_a_named_operator_is_a_view_of_an_estimate():
    from qex.estimate import EstimateOutput
    from qex.timeseries import TsBeta
    from qex.crosssection import PcaFactor, RegressionNeut
    for named, est, cls in [("ts_beta(returns(1), returns(5), 63)", "estimate('ols', 'beta1', returns(1), returns(5), window=63)", TsBeta),
                            ("regression_neut(returns(1), returns(5))", "estimate('xs_ols', 'resid', returns(1), returns(5))", RegressionNeut),
                            ("pca_factor(returns(1), 63, 1)", "estimate('pca', 'beta1', returns(1), window=63)", PcaFactor)]:
        a, b = qex(named), qex(est)
        assert type(a) is type(b) is cls and isinstance(a, EstimateOutput)
        assert "_compute" not in vars(cls)                          # no maths of its own
        assert type(a.fit).__name__ == "EstimateFit"


def test_every_output_of_a_fit_reads_one_fit():
    fits = {qex(e).fit._cache_key() for e in ("ts_beta(returns(1), returns(5), 63)", "ts_alpha(returns(1), returns(5), 63)",
                                              "ts_resid(returns(1), returns(5), 63)")}
    assert len(fits) == 1
    fits = {qex(e).fit._cache_key() for e in ("regression_neut(returns(1), returns(5))",
                                              "regression_alpha(returns(1), returns(5))",
                                              "regression_beta(returns(1), returns(5))")}
    assert len(fits) == 1


def test_a_fit_computes_only_the_outputs_read(panel):
    node = qex("ts_beta(returns(1), returns(1, 'market'), 63)")
    with compute_context():
        node.evaluate(panel)
        res = node.fit.evaluate(panel)
    assert set(res._done) == {"beta1"}                               # not the means, the intercept or the residual
    with compute_context():
        qex("ts_resid(returns(1), returns(1, 'market'), 63)").evaluate(panel)


class _Demeaned(Estimator):
    """A vectorized estimator: each stock's value less its rolling mean, and that mean."""
    dof = 0                                        # descriptive: nothing searched for
    name, version, axis = "test_demeaned", "1", "stock"
    n_inputs, outputs = (1, 1), ("gap", "mean")
    refit, refit_default = 1, 1

    def compute(self, values, window, refit, hyper, data):
        return {"mean": lambda get: values[0].rolling(window, min_periods=window).mean(),
                "gap": lambda get: values[0] - get("mean")}


def test_a_vectorized_estimator(panel):
    _reg(_Demeaned)
    a = _ev("estimate('test_demeaned', 'gap', returns(5), window=20)", panel)
    b = _ev("returns(5) - ts_mean(returns(5), 20)", panel)            # min_periods 20 by default
    assert np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True)


# --- complexity: what a score charges for ------------------------------------------------------------------------------

@pytest.mark.parametrize("expr, operators, dof", [
    ("ts_beta(returns(1), returns(1, 'market'), 252)", 3, 0),          # returns(1), the market's, ts_beta
    ("ts_resid(returns(1), returns(1, 'market'), 252)", 3, 0),         # one, however the fit is built
    ("estimate('ols', 'beta1', returns(1), returns(1, 'market'), window=252)", 3, 0),
    ("ts_beta(returns(1), returns(1, 'market'), 252) + ts_resid(returns(1), returns(1, 'market'), 252)", 5, 0),
    ("pca_factor(returns(1), 252, 1) - pca_factor(returns(1), 252, 2)", 4, 0),
    ("ts_match(returns(5), returns(1), 120)", 3, 0),
    ("estimate('test_switch', 'w1', returns(1, 'market'), returns(1, 'market'), returns(5, 'market'), window=200)", 3, 3),
    ("let s = returns(1, 'market'), a = returns(5, 'market') "
     "in estimate('test_switch', 'w1', s, s, a, window=200) * estimate('test_switch', 'p', s, s, a, window=200)", 5, 3),
])
def test_complexity_counts_what_is_written_and_what_is_learned(expr, operators, dof):
    c = Q.complexity(qex(expr))
    assert (c.operators, c.dof) == (operators, dof)
    g = Q.SignalGraph()
    g.add_signal(qex(expr), name="a")
    assert (g.stats()["operators"], g.stats()["dof"]) == (operators, dof)
    assert c.charge(0.003, 0.015) == pytest.approx(0.003 * operators + 0.015 * dof)


def test_an_estimator_must_say_what_it_learns():
    class Silent(Estimator):
        name, axis, n_inputs, outputs = "test_silent", "series", (1, 1), ("x",)

        def fit(self, window, hyper, rng):
            return 0

        def step(self, state, row, hyper):
            return {"x": 0.0}
    with pytest.raises(TypeError, match="dof"):
        register_estimator(Silent)


def test_estimators_lists_the_built_ins_in_a_fresh_process():
    import subprocess, sys
    out = subprocess.run([sys.executable, "-c", "import qex; print(sorted(qex.estimators()['name']))"],
                         capture_output=True, text=True, check=True).stdout
    for name in ("ols", "ols_shrunk", "xs_ols", "pca"):
        assert repr(name) in out


def test_complexity_with_data_charges_what_a_fit_realized(panel):
    e = qex(SW.format("w1"))
    with compute_context():
        nominal, realized = Q.complexity(e).dof, Q.complexity(e, panel).dof
    assert nominal == 3 and 0 <= realized < 1                         # the share of days won, near a half
    with compute_context():
        assert Q.complexity(qex("ts_beta(returns(1), returns(1, 'market'), 63)"), panel).dof == 0
