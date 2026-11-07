"""qex 0.12.0: exp, is_finite, ts_quantile, regression_neut with several regressors, and group_std as a
one-day cross-sectional operator."""

import numpy as np
import pandas as pd
import pytest

from qex import qex, compute_context, SignalGraph
from qex.adjust import check, exponent, LookAhead
from qex.missing import label
from qex.universe import cross_section_universe

NAN = np.nan


def _ev(expr, data):
    with compute_context():
        return qex(expr).evaluate(data)


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    n, k = 120, 12
    idx = pd.bdate_range("2021-01-01", periods=n)
    cols = [f"S{i}" for i in range(k)]
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.02, (n, k)), axis=0)), index=idx, columns=cols)
    a = pd.DataFrame(rng.normal(size=(n, k)), index=idx, columns=cols)
    b = 0.7 * a + 0.3 * pd.DataFrame(rng.normal(size=(n, k)), index=idx, columns=cols)   # correlated with a
    y = 2 * a - b + pd.DataFrame(rng.normal(size=(n, k)), index=idx, columns=cols)
    sector = pd.DataFrame([["A"] * 4 + ["B"] * 4 + ["C"] * 4] * n, index=idx, columns=cols)
    return {"close": close, "a": a, "b": b, "y": y, "sector": sector}


# --- exp ----------------------------------------------------------------------------------------

def test_exp_inverts_log(data):
    pd.testing.assert_frame_equal(_ev("exp(log(close()))", data), data["close"], rtol=1e-12)


def test_exp_compounds_returns(data):
    out = _ev("exp(ts_sum(log(1 + returns(1)), 5)) - 1", data)
    expected = _ev("returns(5)", data)
    pd.testing.assert_frame_equal(out.iloc[6:], expected.iloc[6:], rtol=1e-10)


def test_exp_softmax_is_scale_of_exp(data):
    out = _ev("scale(exp(field('a') / 0.5))", data)
    e = np.exp(data["a"] / 0.5)
    pd.testing.assert_frame_equal(out, e.div(e.sum(axis=1), axis=0))


def test_exp_of_a_number_and_overflow(data):
    pd.testing.assert_frame_equal(_ev("exp(1) * field('a')", data), np.e * data["a"])
    assert np.isinf(_ev("exp(field('a') * 1000)", data)).any().any()   # inf, no warning raised


# --- is_finite ----------------------------------------------------------------------------------

def test_is_finite(data):
    x = pd.DataFrame({"A": [1.0, np.inf, -np.inf, NAN, 0.0]}, index=pd.bdate_range("2024-01-01", periods=5))
    d = {"close": x.fillna(1).replace([np.inf, -np.inf], 1), "x": x}
    np.testing.assert_array_equal(_ev("is_finite(field('x'))", d)["A"], [1, 0, 0, 0, 1])
    np.testing.assert_array_equal(_ev("is_valid(field('x'))", d)["A"], [1, 1, 1, 0, 1])   # is_valid counts inf
    out = _ev("where(is_finite(field('x')), field('x'), None)", d)["A"]
    assert out.iloc[0] == 1.0 and out.iloc[1:4].isna().all()


# --- ts_quantile --------------------------------------------------------------------------------

@pytest.mark.parametrize("q", [0.0, 0.05, 0.5, 0.75, 1.0])
def test_ts_quantile_matches_pandas(data, q):
    x = data["a"].mask(np.random.default_rng(1).random(data["a"].shape) < 0.1)
    out = _ev(f"ts_quantile(field('x'), 30, {q})", dict(data, x=x))
    pd.testing.assert_frame_equal(out, x.rolling(30, min_periods=20).quantile(q))


def test_ts_quantile_ends_are_min_and_max(data):
    pd.testing.assert_frame_equal(_ev("ts_quantile(field('a'), 10, 0)", data), _ev("ts_min(field('a'), 10)", data))
    pd.testing.assert_frame_equal(_ev("ts_quantile(field('a'), 10, 1)", data), _ev("ts_max(field('a'), 10)", data))


def test_ts_quantile_median_by_hand(data):
    out = _ev("ts_quantile(field('a'), 21, 0.5)", data)
    t = 50
    assert out.iloc[t, 3] == pytest.approx(np.median(data["a"].iloc[t - 20:t + 1, 3]))


def test_ts_quantile_min_periods(data):
    x = data["a"].copy()
    x.iloc[:, 0] = NAN
    x.iloc[40:48, 0] = 1.0                                         # 8 values
    d = dict(data, x=x)
    assert _ev("ts_quantile(field('x'), 20, 0.5)", d)["S0"].isna().all()           # default needs 20
    assert _ev("ts_quantile(field('x'), 20, 0.5, 5)", d)["S0"].iloc[44:48].eq(1).all()


@pytest.mark.parametrize("bad", ["ts_quantile(field('a'), 20, 1.5)", "ts_quantile(field('a'), 20, -0.1)",
                                 "ts_quantile(field('a'), 0, 0.5)"])
def test_ts_quantile_refuses_bad_arguments(bad):
    with pytest.raises(ValueError):
        qex(bad)


# --- regression_neut with several regressors ----------------------------------------------------

def _ols_residual(y, xs):
    out = pd.DataFrame(NAN, index=y.index, columns=y.columns)
    for t in y.index:
        ok = y.loc[t].notna()
        for x in xs:
            ok &= x.loc[t].notna()
        if not ok.any():
            continue
        A = np.column_stack([np.ones(ok.sum())] + [x.loc[t, ok].to_numpy() for x in xs])
        beta = np.linalg.lstsq(A, y.loc[t, ok].to_numpy(), rcond=None)[0]
        out.loc[t, ok] = y.loc[t, ok].to_numpy() - A @ beta
    return out


def test_regression_neut_several_regressors_is_multiple_ols(data):
    out = _ev("regression_neut(field('y'), field('a'), field('b'))", data)
    pd.testing.assert_frame_equal(out, _ols_residual(data["y"], [data["a"], data["b"]]), atol=1e-10)
    # zero mean and zero covariance with every regressor, each day
    for x in ("a", "b"):
        cov = (out * (data[x].sub(data[x].mean(axis=1), axis=0))).sum(axis=1)
        assert np.allclose(cov, 0, atol=1e-9)
    assert np.allclose(out.sum(axis=1), 0, atol=1e-9)


def test_nesting_is_not_the_multiple_regression(data):
    nested = _ev("regression_neut(regression_neut(field('y'), field('a')), field('b'))", data)
    a_c = data["a"].sub(data["a"].mean(axis=1), axis=0)
    # nesting leaves exposure to a when a and b are correlated; the joint fit does not
    assert not np.allclose((nested * a_c).sum(axis=1), 0, atol=1e-6)
    joint = _ev("regression_neut(field('y'), field('a'), field('b'))", data)
    assert np.allclose((joint * a_c).sum(axis=1), 0, atol=1e-9)


def test_regression_neut_one_regressor_unchanged(data):
    one = qex("regression_neut(field('y'), field('a'))")
    assert one._cache_key() == ("RegressionNeut", ("Field", "y"), ("Field", "a"))  # sealed-alpha hashes keep
    assert one.more is None
    out = _ev("regression_neut(field('y'), field('a'))", data)
    pd.testing.assert_frame_equal(out, _ols_residual(data["y"], [data["a"]]), atol=1e-10)


def test_regression_neut_missing_and_collinear(data):
    b = data["b"].copy()
    b.iloc[5, 2] = NAN
    out = _ev("regression_neut(field('y'), field('a'), field('b'), field('a') * 2)", dict(data, b=b))
    assert np.isnan(out.iloc[5, 2]) and out.iloc[5].drop("S2").notna().all()    # collinear: no error
    expected = _ev("regression_neut(field('y'), field('a'), field('b'))", dict(data, b=b))
    pd.testing.assert_frame_equal(out, expected, atol=1e-9)


def test_regression_neut_several_regressors_in_universe(data):
    member = data["a"].notna().astype(float)
    member.iloc[:, :3] = NAN
    with compute_context(), cross_section_universe(member.notna()):
        out = qex("regression_neut(field('y'), field('a'), field('b'))").evaluate(data)
    assert out.iloc[:, :3].isna().all().all()
    inside = [c for c in data["y"].columns[3:]]
    expected = _ols_residual(data["y"][inside], [data["a"][inside], data["b"][inside]])
    pd.testing.assert_frame_equal(out[inside], expected, atol=1e-10)


def test_regression_neut_tools_see_every_regressor():
    e = qex("regression_neut(returns(5), returns(1), volatility(20), field('b'))")
    assert label(e) == "regression_neut(returns(5), returns(1), volatility(20), field('b'))"
    g = SignalGraph()
    g.add_signal(e, name="a")
    assert g.stats()["total_nodes"] == 5
    assert e._cache_key() != qex("regression_neut(returns(5), returns(1), field('b'), volatility(20))")._cache_key()


# --- group_std: one day's cross-section ---------------------------------------------------------

def _group_std_reference(x, g):
    out = pd.DataFrame(NAN, index=x.index, columns=x.columns)
    for t in x.index:
        for label_ in g.loc[t].dropna().unique():
            members = g.columns[g.loc[t] == label_]
            vals = x.loc[t, members].dropna()
            if len(vals) >= 2:
                out.loc[t, members] = np.std(vals, ddof=1)
    return out


def test_group_std_is_the_days_spread_within_each_group(data):
    x = data["a"].mask(np.random.default_rng(2).random(data["a"].shape) < 0.15)
    sector = data["sector"].copy()
    sector.iloc[10:, 0] = "B"                                       # a stock changes group
    sector.iloc[:, 11] = NAN                                         # a stock with no group
    d = dict(data, x=x, sector=sector)
    pd.testing.assert_frame_equal(_ev("group_std(field('x'), 'sector')", d), _group_std_reference(x, sector),
                                  atol=1e-12)


def test_group_std_member_without_a_value_gets_its_groups_spread(data):
    x = data["a"].copy()
    x.iloc[3, 0] = NAN
    out = _ev("group_std(field('x'), 'sector')", dict(data, x=x))
    assert out.iloc[3, 0] == pytest.approx(np.std(x.iloc[3, 1:4], ddof=1))


def test_group_std_needs_two_values():
    idx = pd.bdate_range("2024-01-01", periods=2)
    x = pd.DataFrame({"A": [1.0, 2.0], "B": [NAN, 4.0]}, index=idx)
    g = pd.DataFrame("s", index=idx, columns=["A", "B"])
    out = _ev("group_std(field('x'), 'g')", {"close": x.fillna(1), "x": x, "g": g})
    assert out.iloc[0].isna().all()
    assert out.iloc[1, 0] == pytest.approx(np.std([2.0, 4.0], ddof=1))


def test_group_zscore_composes(data):
    z = _ev("group_demean(field('a'), 'sector') / group_std(field('a'), 'sector')", data)
    for s in ("A", "B", "C"):
        members = data["sector"].columns[data["sector"].iloc[0] == s]
        assert np.allclose(z[members].mean(axis=1), 0, atol=1e-12)
        assert np.allclose(z[members].std(axis=1, ddof=1), 1)


def test_group_std_window_form_is_refused_with_the_replacement():
    with pytest.raises(ValueError, match=r"group_std\(x, groups\).*ts_std\(s, n\)"):
        qex("group_std(returns(5), 'pair', 60)")


# --- the look-ahead checker -------------------------------------------------------------------

def test_new_operators_in_the_adjustment_check():
    f = {"close": {"adjustment": 1, "unit": "usd_per_share"}}
    assert exponent(qex("exp(log(close()))"), f) == 1
    assert exponent(qex("exp(returns(1))"), f) == 0
    assert exponent(qex("ts_quantile(close(), 20, 0.5)"), f) == 1
    assert exponent(qex("is_finite(close())"), f) == 0
    with pytest.raises(LookAhead):
        check(qex("rank(ts_quantile(close(), 20, 0.5))"), f)
    check(qex("rank(close() / ts_quantile(close(), 60, 0.5))"), f)           # a ratio over time is fine


# --- group operators: labels as of each date, empty groups, the universe, group_count_valid ----

def _moving_labels():
    """Labels given on days 1 and 4 only; A moves from s1 to s2 on day 4."""
    idx = pd.bdate_range("2024-01-01", periods=6)
    # A ranks above B but below C, so its rank tells which group it is in
    x = pd.DataFrame({"A": [5.0, 6, 7, 8, 9, 10], "B": [1.0, 2, 3, 4, 5, 6], "C": [100.0] * 6}, index=idx)
    lab = pd.DataFrame({"A": ["s1", "s2"], "B": ["s1", "s1"], "C": ["s2", "s2"]}, index=[idx[0], idx[3]])
    return {"close": x, "x": x, "g": lab}, idx


@pytest.mark.parametrize("op", ["group_demean(field('x'), 'g')", "group_rank(field('x'), 'g')",
                                "group_sum(field('x'), 'g')", "group_std(field('x'), 'g')",
                                "group_count_valid(field('x'), 'g', 1)", "group_quantile(field('x'), 'g', 2)"])
def test_labels_are_taken_as_of_each_date_never_from_later(op):
    d, idx = _moving_labels()
    out = _ev(op, d)
    daily = d["g"].reindex(idx).ffill()                            # what the labels were known to be each day
    expected = _ev(op, dict(d, g=daily))
    pd.testing.assert_frame_equal(out, expected)
    # on days 2 and 3, A is still with B in s1: it is not grouped with C by its day-4 label
    if op.startswith("group_sum"):
        assert out.loc[idx[1], "A"] == 6 + 2 and out.loc[idx[2], "A"] == 7 + 3
    if op.startswith("group_rank"):
        assert out.loc[idx[1], "A"] == 1.0                         # top of s1 (with B), not bottom of s2 (with C)


def test_labels_before_the_first_row_mean_no_group():
    d, idx = _moving_labels()
    late = d["g"].iloc[1:]                                         # labels start on day 4
    out = _ev("group_demean(field('x'), 'g')", dict(d, g=late))
    assert out.iloc[:3].isna().all().all() and out.iloc[3:].notna().all().all()


def test_group_cross_and_book_use_labels_as_of():
    d, idx = _moving_labels()
    daily = d["g"].reindex(idx).ffill()
    size = pd.DataFrame({"A": [1] * 6, "B": [1] * 6, "C": [1] * 6}, index=idx)
    a = _ev("group_demean(field('x'), group_cross('g', field('size')))", dict(d, size=size))
    b = _ev("group_demean(field('x'), group_cross('g', field('size')))", dict(d, g=daily, size=size))
    pd.testing.assert_frame_equal(a, b)
    from evaluation import BookSettings, build_book
    s = BookSettings(delay=0, neutralize="g", truncation=None)
    pd.testing.assert_frame_equal(build_book(qex("field('x')"), d, s), build_book(qex("field('x')"), dict(d, g=daily), s))


def test_group_sum_of_a_group_with_no_values_is_missing_not_zero():
    idx = pd.bdate_range("2024-01-01", periods=2)
    x = pd.DataFrame({"A": [1.0, NAN], "B": [2.0, NAN], "C": [5.0, 6.0], "D": [NAN, NAN]}, index=idx)
    g = pd.DataFrame({"A": "s1", "B": "s1", "C": "s2", "D": "s3"}, index=idx)
    out = _ev("group_sum(field('x'), 'g')", {"close": x.fillna(1), "x": x, "g": g})
    assert out.loc[idx[0], "A"] == 3 and out.loc[idx[0], "B"] == 3
    assert out.loc[idx[1], ["A", "B"]].isna().all()                # s1 has no values on day 2
    assert out["D"].isna().all()                                   # s3 never has one
    x2 = x.copy()
    x2.loc[idx[0], "B"] = NAN
    out2 = _ev("group_sum(field('x'), 'g')", {"close": x.fillna(1), "x": x2, "g": g})
    assert out2.loc[idx[0], "B"] == 1                              # a member with no value still gets its group's sum


@pytest.mark.parametrize("op", ["group_sum(field('x'), 'g')", "group_std(field('x'), 'g')",
                                "group_count_valid(field('x'), 'g', 1)", "group_demean(field('x'), 'g')",
                                "group_rank(field('x'), 'g')"])
def test_names_outside_the_universe_get_no_group_value(op):
    idx = pd.bdate_range("2024-01-01", periods=3)
    x = pd.DataFrame({"A": [1.0, 2, 3], "B": [2.0, 4, 6], "C": [4.0, NAN, 5], "D": [7.0, 8, 9]}, index=idx)
    g = pd.DataFrame({"A": "s", "B": "s", "C": "s", "D": "t"}, index=idx)
    member = pd.DataFrame(True, index=idx, columns=x.columns)
    member["B"] = False
    d = {"close": x.fillna(1), "x": x, "g": g}
    with compute_context(), cross_section_universe(member):
        out = qex(op).evaluate(d)
    assert out["B"].isna().all()                                   # outside the universe
    if op.startswith(("group_sum", "group_count_valid")):
        assert out.loc[idx[1], "C"] == (2.0 if op.startswith("group_sum") else 1)   # a member without a value gets it
    with compute_context(), cross_section_universe(member):        # and B's value is left out of its group's statistics
        a = qex(op).evaluate(d)
    with compute_context(), cross_section_universe(member):
        b = qex(op).evaluate(dict(d, x=x.assign(B=100.0)))
    pd.testing.assert_frame_equal(a, b)


def _count_valid_reference(values, group_df, window):
    """The pre-0.12.0 loop, for labels given every day."""
    result = pd.DataFrame(NAN, index=values.index, columns=values.columns)
    for i, date in enumerate(values.index):
        window_values = values.iloc[max(0, i - window + 1):i + 1]
        date_groups = group_df.loc[date]
        for name in date_groups.unique():
            if pd.isna(name):
                continue
            mask = date_groups == name
            result.loc[date, mask] = window_values[mask[mask].index].notna().sum().sum()
    return result


@pytest.mark.parametrize("window", [1, 3, 10])
def test_group_count_valid_matches_the_old_loop(window):
    rng = np.random.default_rng(9)
    idx = pd.bdate_range("2023-01-01", periods=40)
    cols = [f"S{i}" for i in range(9)]
    x = pd.DataFrame(rng.normal(size=(40, 9)), index=idx, columns=cols).mask(rng.random((40, 9)) < 0.3)
    x.iloc[5, 2] = np.inf                                           # an infinity counts, as in is_valid
    g = pd.DataFrame(rng.choice(["a", "b", "c"], size=(40, 9)), index=idx, columns=cols).astype(object)
    g.iloc[:, 8] = NAN
    g.iloc[10:20, 0] = NAN
    out = _ev(f"group_count_valid(field('x'), 'g', {window})", {"close": x.fillna(1), "x": x, "g": g})
    pd.testing.assert_frame_equal(out, _count_valid_reference(x, g, window))


def test_group_count_valid_is_fast_and_checks_its_window():
    import time
    rng = np.random.default_rng(10)
    idx = pd.bdate_range("2010-01-01", periods=1000)
    x = pd.DataFrame(rng.normal(size=(1000, 2000)), index=idx)
    g = pd.DataFrame(rng.integers(0, 11, (1000, 2000)).astype(str), index=idx)
    start = time.perf_counter()
    _ev("group_count_valid(field('x'), 'g', 5)", {"close": x, "x": x, "g": g})
    assert time.perf_counter() - start < 10                        # the loop took about 2 s for 150 days of 3,000 names
    with pytest.raises(ValueError, match="window"):
        qex("group_count_valid(close(), 'sector', 0)")
