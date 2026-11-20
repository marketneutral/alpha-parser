"""estimate(name, output, inputs..., window=, refit=, ...): the built-in estimators are today's operators (the same
node, the same numbers), the registry, and the stump."""
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
    a = qex("estimate('stump', 'w1', returns(1, 'market'), returns(1, 'market'), returns(5, 'market'), window=60)")
    b = qex("estimate('stump', 'p', returns(1, 'market'), returns(1, 'market'), returns(5, 'market'), window=60)")
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
    ("estimate('stump', 'w1', returns(1), returns(5), returns(20), window=200, min_leaf=150)", "leaves no split"),
    ("estimate('stump', 'w1', group('sector'), returns(5), returns(20), window=200)", "labels"),
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
    assert {"ols", "ols_shrunk", "xs_ols", "pca", "stump"} <= set(t.name)


# --- the stump ---------------------------------------------------------------------------------------------------------

def _regime(seed=1, T=2000, effect=0.004, threshold=6.0):
    """A wandering state (0 to 10) and two return series; r1 beats r2 the day after the state is above `threshold`."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=T)
    state = pd.Series(np.abs(np.cumsum(rng.normal(0, 1, T))) % 10, idx)
    high = state.shift(1) > threshold
    r1 = pd.Series(rng.normal(0, 0.01, T), idx) + np.where(high, effect, -effect)
    r2 = pd.Series(rng.normal(0, 0.01, T), idx)
    frame = lambda s: pd.DataFrame({c: s for c in ("A", "B", "C")})
    return {"close": frame(pd.Series(100.0, idx)), "s": frame(state), "a": frame(r1), "b": frame(r2)}, state


STUMP = "estimate('stump', '{}', field('s'), field('a'), field('b'), window=504, refit=63)"


def test_the_stump_finds_a_planted_threshold():
    d, state = _regime()
    out = {o: _ev(STUMP.format(o), d)["A"] for o in ("w1", "threshold", "gap", "p")}
    th = out["threshold"].dropna()
    assert ((th > 5.0) & (th < 7.0)).all()                                # planted at 6; the grid is 20 quantiles
    assert (out["p"].dropna() < 0.01).all()
    assert (out["gap"].dropna() > 0).all()                                # r1's edge is larger above it
    w = out["w1"].dropna()
    assert (w == (state[w.index] > out["threshold"][w.index])).all()     # w1 follows today's side of the threshold
    assert (w == (state[w.index] > 6)).mean() > 0.95


def test_with_no_regime_p_is_mostly_large():
    d, _ = _regime(seed=5, effect=0.0)
    p = _ev(STUMP.format("p"), d)["A"].dropna()
    assert p.drop_duplicates().median() > 0.2


def test_the_stump_is_point_in_time():
    """Change every input after day 1200: no output changes through day 1200."""
    d, _ = _regime()
    rng = np.random.default_rng(9)
    later = {}
    for k, v in d.items():
        v = v.copy()
        if k != "close":
            v.iloc[1201:, :] = rng.normal(0, 1, len(v) - 1201)[:, None]
        later[k] = v
    for o in ("w1", "threshold", "gap", "p"):
        a, b = _ev(STUMP.format(o), d)["A"], _ev(STUMP.format(o), later)["A"]
        assert a.iloc[:1201].equals(b.iloc[:1201])
        assert not a.iloc[1201:].equals(b.iloc[1201:])


def test_the_stump_does_not_depend_on_where_the_data_starts_and_repeats():
    d, _ = _regime()
    short = {k: v.iloc[300:] for k, v in d.items()}
    for o in ("w1", "p"):
        a, b = _ev(STUMP.format(o), d)["A"], _ev(STUMP.format(o), short)["A"]
        common = b.dropna().index
        assert len(common) > 1000 and np.allclose(a[common], b[common])
    assert _ev(STUMP.format("p"), d).equals(_ev(STUMP.format("p"), d))


def test_the_stump_fits_one_series():
    d, _ = _regime()
    d = dict(d, own=pd.DataFrame(np.random.default_rng(2).normal(size=d["s"].shape), d["s"].index, d["s"].columns))
    with pytest.raises(ValueError, match="fits one series"):
        _ev("estimate('stump', 'w1', field('own'), field('a'), field('b'), window=504)", d)


def test_before_the_first_fit_is_a_short_history():
    d, _ = _regime()
    r = explain_missing(qex(STUMP.format("w1")), d)
    assert list(r.summary().index) == ["short history"]
    assert r.nodes[-1].startswith("estimate('stump', 'w1', field('s')")


def test_the_stump_s_options(panel):
    d, _ = _regime()
    none = _ev("estimate('stump', 'p', field('s'), field('a'), field('b'), window=504, refit=63, permutations=0)", d)
    assert none.isna().all().all()                                        # no shuffles, no p
    fine = _ev("estimate('stump', 'threshold', field('s'), field('a'), field('b'), window=504, refit=63, quantiles=50)", d)
    assert fine.notna().any().any()


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
    ("estimate('stump', 'w1', returns(1, 'market'), returns(1, 'market'), returns(5, 'market'), window=200)", 3, 3),     # the two series and the stump
    ("let s = returns(1, 'market'), a = returns(5, 'market') "
     "in estimate('stump', 'w1', s, s, a, window=200) * estimate('stump', 'p', s, s, a, window=200)", 5, 3),     # one fit, read twice: 3 dof
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


# --- the stump across stocks -------------------------------------------------------------------------------------------

def _planted_xs(seed=1, T=700, N=200, effect=0.004, cut=0.9):
    """Returns follow y for stocks in the top (1 - cut) of x that day, and z for the rest."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2012-01-02", periods=T)
    cols = [f"S{i:03d}" for i in range(N)]
    x = pd.DataFrame(np.cumsum(rng.normal(0, .3, (T, N)), 0), idx, cols)
    y, z = (pd.DataFrame(rng.normal(size=(T, N)), idx, cols) for _ in range(2))
    sig = pd.DataFrame(np.where(x.rank(axis=1, pct=True) > cut, y, z), idx, cols)
    r = effect * sig.shift(1) + pd.DataFrame(rng.normal(0, .02, (T, N)), idx, cols)
    return {"close": 50 * np.exp(r.fillna(0).cumsum()), "x": x, "y": y, "z": z}


XS = "estimate('stump_xs', '{}', rank(field('x')), field('y'), field('z'), returns(1), window=252, refit=63)"


def test_the_cross_sectional_stump_finds_a_planted_split():
    d = _planted_xs()
    out = {o: _ev(XS.format(o), d) for o in ("w1", "threshold", "p")}
    assert set(out["threshold"].stack().dropna().round(3)) == {0.9}                # the top 10%, in rank units
    assert (out["p"].stack().dropna() < 1e-6).all()
    rk = _ev("rank(field('x'))", d)
    w = out["w1"]
    assert ((w == 1) == (rk > 0.9))[w.notna()].stack().all()             # y above, z below, stock by stock
    assert (w.nunique(axis=1) == 2).any()


def test_with_no_split_its_p_is_not_small():
    p = _ev(XS.format("p"), _planted_xs(seed=3, effect=0.0)).iloc[:, 0].dropna().drop_duplicates()
    assert p.median() > 0.1


def test_the_cross_sectional_stump_is_point_in_time_and_start_independent():
    d = _planted_xs()
    later = {k: v.copy() for k, v in d.items()}
    rng = np.random.default_rng(5)
    for k in ("x", "y", "z"):
        later[k].iloc[451:] = rng.normal(size=later[k].iloc[451:].shape)
    for o in ("w1", "threshold", "p"):
        a, b = _ev(XS.format(o), d), _ev(XS.format(o), later)
        assert np.array_equal(a.iloc[:451].to_numpy(), b.iloc[:451].to_numpy(), equal_nan=True)
    short = {k: v.iloc[100:] for k, v in d.items()}
    a, b = _ev(XS.format("w1"), d), _ev(XS.format("w1"), short)
    common = b.dropna(how="all").index
    assert len(common) > 200 and np.array_equal(a.loc[common].to_numpy(), b.loc[common].to_numpy(), equal_nan=True)


def test_the_cross_sectional_stump_s_options_and_cost():
    with pytest.raises(ValueError, match="min_share"):
        qex(XS.format("w1").replace("refit=63", "refit=63, min_share=0.6"))
    with pytest.raises(ValueError, match="no option 'permutations'"):
        qex(XS.format("w1").replace("refit=63", "refit=63, permutations=10"))
    assert Q.complexity(qex(XS.format("w1"))).dof == 3


def test_estimators_lists_the_built_ins_in_a_fresh_process():
    import subprocess, sys
    out = subprocess.run([sys.executable, "-c", "import qex; print(sorted(qex.estimators()['name']))"],
                         capture_output=True, text=True, check=True).stdout
    for name in ("ols", "ols_shrunk", "xs_ols", "pca", "stump", "stump_xs"):
        assert repr(name) in out


# --- the tree --------------------------------------------------------------------------------------------------------

def _two_states(seed=2, T=2500, effect=0.004):
    """r1 beats r2 only where a > 6 and b < 0 (the day before): a region a stump cannot draw and a depth-2 tree can."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2008-01-01", periods=T)
    a = pd.Series(np.abs(np.cumsum(rng.normal(0, 1, T))) % 10, idx)
    b = pd.Series(np.cumsum(rng.normal(0, .3, T)) % 6 - 3, idx)
    region = (a.shift(1) > 6) & (b.shift(1) < 0)
    r1 = pd.Series(rng.normal(0, .01, T), idx) + np.where(region, effect, -effect / 2)
    r2 = pd.Series(rng.normal(0, .01, T), idx)
    frame = lambda s: pd.DataFrame({c: s for c in "ABC"})
    return {"close": frame(pd.Series(100.0, idx)), "a": frame(a), "b": frame(b), "r1": frame(r1), "r2": frame(r2)}, a, b


TREE = "estimate('tree', '{}', field('a'), field('b'), field('r1'), field('r2'), window=1260, refit=63{})"


def test_the_tree_finds_a_two_state_region():
    d, a, b = _two_states()
    w, leaf = (_ev(TREE.format(o, ""), d)["A"] for o in ("w1", "leaf"))
    m = w.dropna().index
    assert (w[m] == ((a > 6) & (b < 0))[m]).mean() > 0.9
    assert set(leaf.dropna().unique()) <= {0.0, 1.0, 2.0, 3.0} and leaf.dropna().nunique() >= 3
    stump = _ev("estimate('stump', 'w1', field('a'), field('r1'), field('r2'), window=1260, refit=63)", d)["A"]
    assert (stump[m] == ((a > 6) & (b < 0))[m]).mean() < (w[m] == ((a > 6) & (b < 0))[m]).mean()


def test_a_tree_of_depth_one_on_one_state_is_the_stump():
    d, _, _ = _two_states()
    t = _ev("estimate('tree', 'w1', field('a'), field('r1'), field('r2'), window=1260, refit=63, depth=1)", d)
    s = _ev("estimate('stump', 'w1', field('a'), field('r1'), field('r2'), window=1260, refit=63)", d)
    assert t.equals(s)


def test_the_tree_is_point_in_time_start_independent_and_repeatable():
    d, _, _ = _two_states()
    later = {k: v.copy() for k, v in d.items()}
    rng = np.random.default_rng(4)
    for k in ("a", "b", "r1", "r2"):
        later[k].iloc[1801:, :] = rng.normal(0, 1, len(later[k]) - 1801)[:, None]
    for o in ("w1", "leaf", "p"):
        x, y = _ev(TREE.format(o, ""), d)["A"], _ev(TREE.format(o, ""), later)["A"]
        assert x.iloc[:1801].equals(y.iloc[:1801])
    short = {k: v.iloc[200:] for k, v in d.items()}
    x, y = _ev(TREE.format("w1", ""), d)["A"], _ev(TREE.format("w1", ""), short)["A"]
    common = y.dropna().index
    assert len(common) > 500 and x[common].equals(y[common])
    assert _ev(TREE.format("p", ""), d).equals(_ev(TREE.format("p", ""), d))


def test_the_tree_s_cost_and_options():
    assert Q.complexity(qex(TREE.format("w1", ""))).dof == 10                       # 3 splits x (cut, state) + 4 leaves
    assert Q.complexity(qex(TREE.format("w1", ", depth=3"))).dof == 7 * 2 + 8
    assert Q.complexity(qex("estimate('tree', 'w1', field('a'), field('r1'), field('r2'), window=252)")).dof == 7
    with pytest.raises(ValueError, match="depth is 1, 2 or 3"):
        qex(TREE.format("w1", ", depth=4"))
    with pytest.raises(ValueError, match="min_leaf"):
        qex(TREE.format("w1", ", min_leaf=1"))
    d, _, _ = _two_states()
    big = _ev(TREE.format("leaf", ", min_leaf=600"), d)["A"].dropna()                # leaves of 600 days: fewer splits
    assert big.nunique() <= _ev(TREE.format("leaf", ""), d)["A"].dropna().nunique()


# --- the tree's regularizers -------------------------------------------------------------------------------------------

def _churn(expr, d):
    w = _ev(expr, d)["A"].dropna()
    return w, w.diff().abs().mean()


def test_alpha_keeps_the_tree_from_splitting_on_noise():
    d, _, _ = _two_states(seed=7, effect=0.0)
    _, plain = _churn(TREE.format("w1", ""), d)
    w, tested = _churn(TREE.format("w1", ", alpha=0.05"), d)
    assert tested < plain / 10                                      # no splits: the book stops switching
    assert _ev(TREE.format("leaf", ", alpha=0.05"), d)["A"].dropna().nunique() == 1


def test_alpha_keeps_a_real_split():
    d, a, b = _two_states()
    w = _ev(TREE.format("w1", ", alpha=0.05"), d)["A"].dropna()
    inside = ((a > 6) & (b < 0))[w.index]
    assert (w[inside] > 0.5).mean() > 0.5 and (w[~inside] <= 0.5).mean() > 0.85


def test_soft_and_honest_shrink_toward_the_blend_on_noise():
    d, _, _ = _two_states(seed=7, effect=0.0)
    w, _ = _churn(TREE.format("w1", ", soft=1, honest=1"), d)
    assert ((w > 0.02) & (w < 0.98)).mean() > 0.9                    # fractional weights, not 0 or 1


def test_honest_still_finds_the_planted_region():
    d, a, b = _two_states()
    w = _ev(TREE.format("w1", ", honest=1"), d)["A"].dropna()
    inside = ((a > 6) & (b < 0))[w.index]
    assert (w[inside] > 0.5).mean() > 0.7 and (w[~inside] <= 0.5).mean() > 0.8


def test_the_regularizers_options():
    with pytest.raises(ValueError, match="soft is on or off"):
        qex(TREE.format("w1", ", soft=2"))
    with pytest.raises(ValueError, match="alpha is a significance level"):
        qex(TREE.format("w1", ", alpha=1.5"))
    with pytest.raises(ValueError, match="needs permutations above 0"):
        qex(TREE.format("w1", ", alpha=0.05, permutations=0"))
    assert Q.complexity(qex(TREE.format("w1", ", alpha=0.05, soft=1, honest=1"))).dof == 10   # searched all the same


def test_complexity_with_data_charges_the_dof_a_tree_used():
    d, _, _ = _two_states(seed=7, effect=0.0)
    with compute_context():
        nominal = Q.complexity(qex(TREE.format("w1", ", alpha=0.05"))).dof
        plain = Q.complexity(qex(TREE.format("w1", "")), d).dof
        tested = Q.complexity(qex(TREE.format("w1", ", alpha=0.05")), d).dof
        shrunk = Q.complexity(qex(TREE.format("w1", ", soft=1, honest=1")), d).dof
        honest = Q.complexity(qex(TREE.format("w1", ", honest=1")), d).dof
    assert nominal == plain == honest == 10                         # without data, and a hard tree: the full size
    assert tested <= 1.5                                            # on noise alpha leaves one leaf
    assert shrunk < plain                                           # leaves that hold back commit less
    stump = "estimate('stump', 'w1', field('a'), field('r1'), field('r2'), window=1260, refit=63)"
    with compute_context():
        assert Q.complexity(qex(stump), d).dof == 3                 # an estimator that does not measure it: nominal
