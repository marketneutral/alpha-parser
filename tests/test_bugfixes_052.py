"""Regression tests for the 0.5.2 bug fixes (reported in qexarena's IDEAS.md).

Each bug test is built from the bug's minimal repro and fails on 0.5.1. The tests named
"unchanged" or "reference", and the math cases that already worked, check that behavior away
from the bug is exactly as before.
"""

import time

import numpy as np
import pandas as pd
import pytest

from qex import qex, compute_context
from qex.parser import QexParser

NAN = np.nan


def _dates(n, start="2020-01-01"):
    return pd.bdate_range(start, periods=n)


def _ev(expr, data):
    with compute_context():
        return qex(expr).evaluate(data)


# --- 1. ts_beta and ts_beta_ewma pair their observations -------------------------------------

@pytest.fixture
def beta_data():
    rng = np.random.default_rng(0)
    n, k = 300, 6
    m = rng.normal(0, 0.01, n)
    r = pd.DataFrame(m[:, None] * rng.uniform(0.5, 1.5, k) + rng.normal(0, 0.01, (n, k)),
                     index=_dates(n), columns=[f"S{i}" for i in range(k)])
    close = 100 * (1 + r).cumprod()
    close.iloc[:240, 0] = NAN                                   # S0 listed 60 days ago
    close.iloc[100:130, 1] = NAN                                # S1 halted for 30 days
    market = pd.Series(100 * np.cumprod(1 + m), index=close.index)
    return {"close": close, "market": market}


def _paired_ols_slope(y, x):
    ok = y.notna() & x.notna()
    y, x = y[ok], x[ok]
    return ((y - y.mean()) * (x - x.mean())).sum() / ((x - x.mean()) ** 2).sum()


def test_ts_beta_is_paired_least_squares(beta_data):
    beta = _ev("ts_beta(returns(1), returns(1, 'market'), 252)", beta_data)
    y = _ev("returns(1)", beta_data)
    x = _ev("returns(1, 'market')", beta_data)
    end = beta.index[-1]
    window = y.index[-252:]
    for name in ["S0", "S1", "S2"]:
        expected = _paired_ols_slope(y.loc[window, name], x.loc[window, name])
        assert beta.loc[end, name] == pytest.approx(expected, rel=1e-10)


def test_ts_beta_matches_the_old_workaround(beta_data):
    fixed = _ev("ts_beta(returns(1), returns(1, 'market'), 120)", beta_data)
    workaround = _ev("ts_beta(returns(1), returns(1, 'market') + 0 * returns(1), 120)", beta_data)
    pd.testing.assert_frame_equal(fixed, workaround)


def test_ts_beta_unchanged_without_holes(beta_data):
    full = dict(beta_data, close=beta_data["close"].iloc[:, 2:])  # names with no missing prices
    y = _ev("returns(1)", full)
    x = _ev("returns(1, 'market')", full)
    old = y.rolling(60, min_periods=20).cov(x) / x.rolling(60, min_periods=20).var()
    pd.testing.assert_frame_equal(_ev("ts_beta(returns(1), returns(1, 'market'), 60)", full), old)


def test_ts_beta_ewma_is_paired(beta_data):
    beta = _ev("ts_beta_ewma(returns(1), returns(1, 'market'), 30)", beta_data)
    y = _ev("returns(1)", beta_data)
    x = _ev("returns(1, 'market')", beta_data)
    both = y.notna() & x.notna()
    yp, xp = y.where(both), x.where(both)
    expected = yp.ewm(halflife=30).cov(xp) / xp.ewm(halflife=30).var()
    pd.testing.assert_frame_equal(beta, expected)
    unpaired = y.ewm(halflife=30).cov(x) / x.ewm(halflife=30).var()
    assert not np.isclose(beta["S0"].iloc[-1], unpaired["S0"].iloc[-1])


# --- 2. One infinity no longer empties a cross-section ---------------------------------------

@pytest.fixture
def inf_data():
    rng = np.random.default_rng(1)
    n, k = 40, 8
    idx = _dates(n)
    cols = [f"S{i}" for i in range(k)]
    close = pd.DataFrame(100 + rng.normal(0, 1, (n, k)).cumsum(axis=0), index=idx, columns=cols)
    volume = pd.DataFrame(rng.integers(100_000, 1_000_000, (n, k)).astype(float), index=idx, columns=cols)
    volume.iloc[-1, 3] = 0                                      # one zero-volume day
    sector = pd.DataFrame([["A", "A", "A", "A", "B", "B", "B", "B"]] * n, index=idx, columns=cols)
    return {"close": close, "volume": volume, "sector": sector}


SIGNAL = "volume(20) / volume(1)"


@pytest.mark.parametrize("op", [
    "zscore({s})", "demean({s})", "scale({s})", "winsorize({s}, 0.1)",
    "group_demean({s}, 'sector')",
])
def test_infinity_is_left_out_of_the_cross_section(inf_data, op):
    out = _ev(op.format(s=SIGNAL), inf_data).iloc[-1]
    assert np.isinf(_ev(SIGNAL, inf_data).iloc[-1, 3])         # the repro really has an infinity
    assert np.isnan(out["S3"])                                   # the infinite name is missing
    others = out.drop("S3")
    assert np.isfinite(others).all()
    # the others get exactly what they get when the infinite value is NaN
    finite = _ev(SIGNAL, inf_data).replace([np.inf, -np.inf], NAN)
    expected = _ev(op.format(s="field('f')"), dict(inf_data, f=finite)).iloc[-1].drop("S3")
    pd.testing.assert_series_equal(others, expected, check_names=False)


def test_group_sum_ignores_infinity(inf_data):
    out = _ev(f"group_sum({SIGNAL}, 'sector')", inf_data).iloc[-1]
    v = _ev(SIGNAL, inf_data).iloc[-1]
    # every member, the infinite name included, gets the sum of its group's finite values
    assert np.isfinite(out).all()
    assert out["S0"] == pytest.approx(v[["S0", "S1", "S2"]].sum())
    assert out["S5"] == pytest.approx(v[["S4", "S5", "S6", "S7"]].sum())


def test_group_std_ignores_infinity(inf_data):
    out = _ev(f"group_std({SIGNAL}, 'sector', 5)", inf_data).iloc[-1]
    assert np.isfinite(out.drop("S3")).all()


def test_regression_neut_ignores_infinity(inf_data):
    out = _ev(f"regression_neut({SIGNAL}, returns(5))", inf_data).iloc[-1]
    assert np.isnan(out["S3"])
    assert np.isfinite(out.drop("S3")).all()
    assert out.drop("S3").sum() == pytest.approx(0, abs=1e-9)


def test_rank_still_ranks_infinity_first(inf_data):
    out = _ev(f"rank({SIGNAL})", inf_data).iloc[-1]
    assert out["S3"] == 1.0
    assert out.notna().all()


def test_cross_sections_without_infinity_are_unchanged(inf_data):
    clean = dict(inf_data, volume=inf_data["volume"].replace(0, 1.0))
    v = _ev(SIGNAL, clean)
    for op, expected in [
        ("zscore", v.sub(v.mean(axis=1), axis=0).div(v.std(axis=1), axis=0)),
        ("demean", v.sub(v.mean(axis=1), axis=0)),
        ("scale", v.div(v.abs().sum(axis=1), axis=0)),
    ]:
        pd.testing.assert_frame_equal(_ev(f"{op}({SIGNAL})", clean), expected)


# --- 3-5. The factor risk model --------------------------------------------------------------

@pytest.fixture(scope="module")
def risk_setup():
    pytest.importorskip("statsmodels")
    from qex import FactorRiskModel, FactorDefinition

    rng = np.random.default_rng(3)
    T, N = 700, 150
    idx = _dates(T, "2018-01-01")
    cols = [f"S{i}" for i in range(N)]
    mkt = rng.normal(0.0003, 0.012, T)
    beta = rng.normal(1, 0.3, N)
    spec = rng.uniform(0.01, 0.03, N)
    rets = pd.DataFrame(mkt[:, None] * beta + rng.normal(0, 1, (T, N)) * spec, index=idx, columns=cols)
    data = {"close": 100 * (1 + rets).cumprod()}
    factors = [FactorDefinition("short_term_reversal", "zscore(-returns(1))", "1-day reversal")]

    def fit(half_life=10 ** 9, lookback=500):
        model = FactorRiskModel(style_factors=factors, include_industries=False, half_life=half_life)
        return model.fit(data, lookback=lookback)

    return {"fit": fit, "rets": rets, "data": data, "spec": spec, "N": N}


def test_risk_model_uses_yesterdays_exposures(risk_setup, capsys):
    import statsmodels.api as sm

    model = risk_setup["fit"]()
    capsys.readouterr()
    fr = model.factor_returns["short_term_reversal"]
    # Same-day exposures regress r_t on -r_t: a factor "return" with a Sharpe in the hundreds.
    sharpe = fr.mean() / fr.std() * np.sqrt(252)
    assert abs(sharpe) < 3
    # Check one day by hand: day t's returns on day t-1's exposures
    with compute_context():
        exposure = qex("zscore(-returns(1))").evaluate(risk_setup["data"])
    day = fr.index[100]
    prev = exposure.index[exposure.index.get_loc(day) - 1]
    y = risk_setup["rets"].loc[day]
    X = sm.add_constant(exposure.loc[prev].rename("short_term_reversal"))
    assert fr.loc[day] == pytest.approx(sm.OLS(y, X).fit().params["short_term_reversal"], rel=1e-8)


def test_factor_covariance_is_the_covariance_of_factor_returns(risk_setup, capsys):
    model = risk_setup["fit"]()                                   # equal weights (huge half-life)
    capsys.readouterr()
    fr = model.factor_returns[model.factor_covariance.columns]
    pd.testing.assert_frame_equal(model.factor_covariance, fr.cov() * 252, rtol=1e-6)


def test_weighted_covariance_matches_numpy():
    from qex.risk import _weighted_covariance, _weighted_variance

    rng = np.random.default_rng(4)
    frame = pd.DataFrame(rng.normal(size=(200, 3)), columns=list("abc"))
    w = np.exp(np.linspace(-3, 0, 200))
    w /= w.sum()
    expected = np.cov(frame.to_numpy().T, aweights=w)
    np.testing.assert_allclose(_weighted_covariance(frame, w).to_numpy(), expected, rtol=1e-10)
    np.testing.assert_allclose(_weighted_variance(frame, w).to_numpy(), np.diag(expected), rtol=1e-10)
    # missing values: renormalize over the rows present
    holes = frame.copy()
    holes.iloc[:50, 0] = NAN
    ok = holes["a"].notna().to_numpy()
    assert _weighted_variance(holes, w)["a"] == pytest.approx(np.cov(frame["a"][ok], aweights=w[ok]))


def test_exposures_include_the_market_factor(risk_setup, capsys):
    model = risk_setup["fit"]()
    capsys.readouterr()
    assert (model.results_.factor_exposures["market"] == 1.0).all()
    assert "market" in model.factor_attribution(pd.Series(1 / risk_setup["N"], index=risk_setup["rets"].columns))
    exp = model.get_exposures(risk_setup["data"])
    assert (exp["market"] == 1.0).all()


def test_portfolio_risk_forecast_matches_realized(risk_setup, capsys):
    model = risk_setup["fit"]()
    capsys.readouterr()
    rets = risk_setup["rets"]
    w = pd.Series(1 / risk_setup["N"], index=rets.columns)
    total, factor, specific = model.portfolio_risk(w)
    realized = (rets.iloc[-500:] @ w).std() * np.sqrt(252)
    assert total == pytest.approx(realized, rel=0.1)
    assert factor > specific                                    # a diversified long book is market risk


def test_specific_risk_matches_simulated(risk_setup, capsys):
    model = risk_setup["fit"]()
    capsys.readouterr()
    true = pd.Series(risk_setup["spec"] * np.sqrt(252), index=risk_setup["rets"].columns)
    ratio = model.specific_risk / true
    assert ratio.median() == pytest.approx(1.0, abs=0.05)


# --- section 3 #1: ts_rank is vectorized ------------------------------------------------------

def _ts_rank_reference(values, period):
    from scipy import stats

    def rank_last(arr):
        valid = ~np.isnan(arr)
        if valid.sum() < 2 or not valid[-1]:
            return np.nan
        ranks = stats.rankdata(arr[valid], method="average")
        return ranks[-1] / len(ranks)

    return values.rolling(period).apply(rank_last, raw=True)


@pytest.mark.parametrize("period", [1, 2, 5, 20])
def test_ts_rank_matches_reference(period):
    rng = np.random.default_rng(5)
    v = pd.DataFrame(rng.integers(0, 6, (120, 5)).astype(float), index=_dates(120))   # many ties
    v = v.mask(rng.random(v.shape) < 0.05)
    out = _ev(f"ts_rank(field('v'), {period})", {"close": v, "v": v})
    pd.testing.assert_frame_equal(out, _ts_rank_reference(v, period))


def test_ts_rank_is_fast():
    rng = np.random.default_rng(6)
    v = pd.DataFrame(rng.normal(size=(1000, 300)), index=_dates(1000))
    start = time.perf_counter()
    _ev("ts_rank(field('v'), 252)", {"close": v, "v": v})
    assert time.perf_counter() - start < 10                     # was about a minute


# --- section 3 #3: math functions of a number -------------------------------------------------

@pytest.fixture
def small_data():
    idx = _dates(10)
    return {"close": pd.DataFrame({"A": np.linspace(100, 110, 10), "B": np.linspace(50, 45, 10)}, index=idx)}


@pytest.mark.parametrize("expr,fn", [
    ("sqrt(5) * returns(1)", lambda r: np.sqrt(5) * r),
    ("power(2.0, returns(1))", lambda r: 2.0 ** r),
    ("power(returns(1), 2)", lambda r: r ** 2),
    ("power(returns(1), -1)", lambda r: r ** -1),
    ("abs(-3) * returns(1)", lambda r: 3 * r),
    ("log(2) * returns(1)", lambda r: np.log(2) * r),
    ("sign(-2) * returns(1)", lambda r: -r),
    ("max(returns(1), 0)", lambda r: np.maximum(r, 0)),
    ("min(0, returns(1))", lambda r: np.minimum(0, r)),
])
def test_math_functions_accept_numbers(small_data, expr, fn):
    r = _ev("returns(1)", small_data)
    pd.testing.assert_frame_equal(_ev(expr, small_data), fn(r), check_names=False)


def test_max_of_two_numbers(small_data):
    assert _ev("max(2, 3)", small_data) == 3


# --- section 3 #5: signed numbers are numbers -------------------------------------------------

def test_negative_delay_error_shows_the_number():
    with pytest.raises(ValueError, match=r"got -1\)"):
        qex("delay(close(), -1)")
    with pytest.raises(ValueError, match=r"got -3\)"):
        qex("delta(close(), -3)")


def test_signed_literals_parse_as_numbers(small_data):
    parser = QexParser()
    assert parser._visit_arg(__import__("ast").parse("-2.5", mode="eval").body) == -2.5
    assert parser._visit_arg(__import__("ast").parse("+3", mode="eval").body) == 3
    pd.testing.assert_frame_equal(_ev("delay(close(), +1)", small_data), small_data["close"].shift(1))
    pd.testing.assert_frame_equal(_ev("close() * -1", small_data), -small_data["close"])
