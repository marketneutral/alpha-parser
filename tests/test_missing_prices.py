"""How qex turns prices into returns when prices are missing (halts, delistings, listings).

One rule everywhere: a return is a ratio of two observed prices, NaN on a day with no price,
and measured from the last valid price (no more than MAX_PRICE_GAP rows old). A backtest holds
a position through days the name can't be traded. These tests must pass on pandas 2 and 3.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from qex import qex, compute_context, price_returns, forward_returns, MAX_PRICE_GAP
from evaluation import Backtest, QuantileAnalysis, hold_untradable

NAN = np.nan


def _dates(n):
    return pd.bdate_range("2024-01-01", periods=n)


def _frame(cols, n=None):
    n = n or len(next(iter(cols.values())))
    return pd.DataFrame({k: np.asarray(v, dtype=float) for k, v in cols.items()}, index=_dates(n))


# A halts for two days (rows 2-3) and reopens at 104; B trades every day.
HALT = _frame({
    "A": [100, 101, NAN, NAN, 104, 105, 106, 107],
    "B": [50, 51, 52, 53, 54, 55, 56, 57],
})


# --- price_returns --------------------------------------------------------------------------

def test_halt_days_are_nan_and_reopen_carries_whole_move():
    r = price_returns(HALT)["A"].to_numpy()
    assert np.isnan(r[0])
    assert r[1] == pytest.approx(0.01)
    assert np.isnan(r[2]) and np.isnan(r[3])          # no fake 0% days
    assert r[4] == pytest.approx(104 / 101 - 1)       # the gap move is not lost
    assert r[5] == pytest.approx(105 / 104 - 1)


def test_n_day_return_uses_last_valid_start_price():
    r = price_returns(HALT, 2)["A"].to_numpy()
    assert np.isnan(r[2]) and np.isnan(r[3])          # today missing
    assert r[4] == pytest.approx(104 / 101 - 1)       # start (row 2) missing: row 1 used
    assert r[5] == pytest.approx(105 / 101 - 1)       # start (row 3) missing: row 1 used
    assert r[6] == pytest.approx(106 / 104 - 1)


def test_delisted_name_is_nan_forever():
    p = _frame({"A": [10, 11, 12, NAN, NAN, NAN, NAN]})
    r1, r3 = price_returns(p)["A"], price_returns(p, 3)["A"]
    assert r1.iloc[3:].isna().all() and r3.iloc[3:].isna().all()


def test_before_listing_is_nan():
    p = _frame({"A": [NAN, NAN, 20, 21, 22]})
    r = price_returns(p)["A"].to_numpy()
    assert np.isnan(r[:3]).all()
    assert r[3] == pytest.approx(0.05)


def test_daily_returns_compound_to_n_day_return():
    rng = np.random.default_rng(0)
    p = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.02, (300, 6)), axis=0)), index=_dates(300))
    p = p.mask(rng.random(p.shape) < 0.15)            # 15% of prices missing, short gaps
    n = 20
    daily, nday = price_returns(p, max_gap=1000), price_returns(p, n, max_gap=1000)
    compounded = np.log1p(daily).rolling(n, min_periods=1).sum().pipe(np.expm1)
    first = p.notna().cumsum()
    # where today's price exists and there is an earlier valid price at or before t - n
    ok = p.notna() & (first.shift(n) > 0)
    assert ok.to_numpy().sum() > 500
    np.testing.assert_allclose(compounded[ok].to_numpy()[ok.to_numpy()], nday[ok].to_numpy()[ok.to_numpy()])


@pytest.mark.parametrize("max_gap", [1, 3, MAX_PRICE_GAP])
def test_max_gap_limits_how_far_a_price_is_bridged(max_gap):
    for missing, bridged in [(max_gap, True), (max_gap + 1, False)]:
        p = _frame({"A": [100.0] + [NAN] * missing + [110.0, 121.0]})
        r = price_returns(p, max_gap=max_gap)["A"]
        reopen = r.iloc[missing + 1]
        if bridged:
            assert reopen == pytest.approx(0.10)
        else:
            assert np.isnan(reopen)
        assert r.iloc[-1] == pytest.approx(0.10)      # the day after is normal again


def test_default_max_gap_is_the_module_constant():
    assert MAX_PRICE_GAP == 10
    p = _frame({"A": [100.0] + [NAN] * 11 + [110.0]})
    assert np.isnan(price_returns(p)["A"].iloc[-1])
    p = _frame({"A": [100.0] + [NAN] * 10 + [110.0]})
    assert price_returns(p)["A"].iloc[-1] == pytest.approx(0.10)


def test_max_gap_zero_never_bridges():
    r = price_returns(HALT, max_gap=0)["A"]
    assert np.isnan(r.iloc[4])
    assert r.iloc[5] == pytest.approx(105 / 104 - 1)


def test_series_input():
    s = HALT["A"]
    pd.testing.assert_series_equal(price_returns(s), price_returns(HALT)["A"])


def test_no_pandas_warnings():
    # pct_change's default fill_method warns on pandas 2.x; the helper does not use it
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        price_returns(HALT, 3)
        forward_returns(HALT, 2)


def test_invalid_arguments():
    with pytest.raises(ValueError, match="period"):
        price_returns(HALT, 0)
    with pytest.raises(ValueError, match="max_gap"):
        price_returns(HALT, 1, max_gap=-1)


def test_integer_prices():
    p = pd.DataFrame({"A": [100, 110, 121]}, index=_dates(3))
    assert price_returns(p)["A"].iloc[2] == pytest.approx(0.10)


# --- forward_returns ------------------------------------------------------------------------

def test_forward_returns_is_shifted_price_returns():
    for h in (1, 3):
        pd.testing.assert_frame_equal(forward_returns(HALT, h), price_returns(HALT, h).shift(-h))


def test_forward_returns_put_gap_move_on_last_halt_day():
    f = forward_returns(HALT)["A"].to_numpy()
    assert np.isnan(f[1]) and np.isnan(f[2])          # the next day has no price
    assert f[3] == pytest.approx(104 / 101 - 1)       # last halt day -> reopen
    # summed over the halt, a held position earns exactly the move from 101 to 104
    assert np.nansum(f[1:4]) == pytest.approx(104 / 101 - 1)


# --- signals --------------------------------------------------------------------------------

def test_returns_signal_uses_the_rule():
    data = {"close": HALT}
    with compute_context():
        for n in (1, 2, 5):
            pd.testing.assert_frame_equal(qex(f"returns({n})").evaluate(data), price_returns(HALT, n))


def test_returns_signal_on_market_series():
    market = pd.Series([100, 102, NAN, 105, 106, 107, 108, 109], index=HALT.index, dtype=float)
    out = qex("returns(1, 'market')").evaluate({"close": HALT, "market": market})
    expected = price_returns(market).to_numpy()
    for col in out.columns:
        np.testing.assert_allclose(out[col].to_numpy(), expected, equal_nan=True)


def test_volatility_ignores_halt_days():
    rng = np.random.default_rng(1)
    p = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 80))), index=_dates(80))
    p.iloc[30:40] = NAN                                # a 10-day halt
    vol = qex("volatility(60)").evaluate({"close": p.to_frame("A")})["A"]
    r = price_returns(p)
    expected = r.iloc[-60:].dropna().std() * np.sqrt(252)
    assert vol.iloc[-1] == pytest.approx(expected)
    # carrying the price forward (pandas 2) would add ten 0% days and pull volatility down
    padded = p.ffill().pct_change(fill_method=None).iloc[-60:].std() * np.sqrt(252)
    assert vol.iloc[-1] > padded


def test_rank_drops_halted_and_delisted_names():
    p = _frame({
        "A": [10, 11, 12, 13, 14, 15],
        "B": [20, 21, NAN, NAN, 24, 25],              # halted
        "C": [30, 31, 32, NAN, NAN, NAN],              # delists
    })
    ranks = qex("rank(returns(1))").evaluate({"close": p})
    assert ranks["B"].iloc[2:4].isna().all()
    assert ranks["C"].iloc[3:].isna().all()
    assert ranks["A"].iloc[1:].notna().all()


def test_risk_model_returns_use_the_rule():
    FactorRiskModel = pytest.importorskip("qex.risk").FactorRiskModel
    model = FactorRiskModel(include_industries=False)
    pd.testing.assert_frame_equal(model._compute_returns({"close": HALT}), price_returns(HALT))


# --- hold_untradable ------------------------------------------------------------------------

def test_halted_position_is_held_then_target_resumes():
    w = _frame({"A": [0.5, 0.5, NAN, NAN, -0.3, -0.3, 0.1, 0.1], "B": [-0.5] * 8})
    held = hold_untradable(w, HALT)
    np.testing.assert_allclose(held["A"], [0.5, 0.5, 0.5, 0.5, -0.3, -0.3, 0.1, 0.1])
    np.testing.assert_allclose(held["B"], [-0.5] * 8)


def test_target_on_halt_day_is_ignored():
    # a halted name can't be traded, whatever the signal says that day
    w = _frame({"A": [0.5, 0.5, -0.9, 0.7, -0.3, -0.3, 0.1, 0.1], "B": [-0.5] * 8})
    held = hold_untradable(w, HALT)
    np.testing.assert_allclose(held["A"].iloc[:4], [0.5, 0.5, 0.5, 0.5])


def test_delisted_position_held_for_max_gap_then_dropped():
    p = _frame({"A": [10, 11, 12] + [NAN] * 6})
    w = _frame({"A": [0.2, 0.3, 0.4] + [NAN] * 6})
    held = hold_untradable(w, p, max_gap=4)["A"].to_numpy()
    np.testing.assert_allclose(held, [0.2, 0.3, 0.4, 0.4, 0.4, 0.4, 0.4, 0, 0])


def test_no_weight_before_listing_and_nan_target_is_flat():
    p = _frame({"A": [NAN, NAN, 10, 11, NAN, 12]})
    w = _frame({"A": [0.7, 0.7, 0.3, NAN, 0.5, 0.2]})
    held = hold_untradable(w, p)["A"].to_numpy()
    # before listing: 0; a NaN target on a tradable day is flat, and that flat position is
    # what gets held through the next halt (not an older weight)
    np.testing.assert_allclose(held, [0, 0, 0.3, 0, 0, 0.2])


def test_max_gap_zero_does_not_hold():
    w = _frame({"A": [0.5] * 8, "B": [-0.5] * 8})
    held = hold_untradable(w, HALT, max_gap=0)
    np.testing.assert_allclose(held["A"], [0.5, 0.5, 0, 0, 0.5, 0.5, 0.5, 0.5])


def test_names_missing_from_prices_get_no_weight():
    w = _frame({"A": [0.5] * 8, "Z": [0.5] * 8})
    held = hold_untradable(w, HALT)
    assert (held["Z"] == 0).all()


# --- backtest -------------------------------------------------------------------------------

def _sig_data(prices, sig):
    return {"close": prices, "sig": _frame(sig, len(prices))}


def _pnl(weights, prices, h=1):
    return (weights * forward_returns(prices, h).reindex_like(weights).fillna(0)).sum(axis=1)


def test_backtest_pnl_includes_move_across_halt():
    # long A, short B; A halts and the signal for A is missing while it's halted
    data = _sig_data(HALT, {"A": [1, 1, NAN, NAN, 1, 1, 1, 1], "B": [-1] * 8})
    r = Backtest(qex("field('sig')")).run(data)
    np.testing.assert_allclose(r.weights["A"], 0.5)   # held through the halt, not closed
    pd.testing.assert_series_equal(r.pnl, _pnl(r.weights, HALT), check_names=False)
    # the A leg earns the whole halt move (101 -> 104), booked the day before the reopen
    a_leg = r.weights["A"] * forward_returns(HALT)["A"].reindex(r.weights.index).fillna(0)
    assert a_leg.iloc[1:4].sum() == pytest.approx(0.5 * (104 / 101 - 1))
    assert a_leg.loc[HALT.index[3]] == pytest.approx(0.5 * (104 / 101 - 1))


def test_backtest_without_holding_would_lose_the_halt_move():
    # what the backtest did before 0.5.1: weight 0 on halt days, so the gap move never counted
    data = _sig_data(HALT, {"A": [1, 1, NAN, NAN, 1, 1, 1, 1], "B": [-1] * 8})
    r = Backtest(qex("field('sig')")).run(data)
    with compute_context():
        target = qex("field('sig')").to_weights(data).fillna(0)
    old_a = (target["A"] * HALT["A"].pct_change(fill_method=None).shift(-1).fillna(0)).iloc[:-1]
    new_a = r.weights["A"] * forward_returns(HALT)["A"].reindex(r.weights.index).fillna(0)
    # the only difference is the halt move, which the old backtest dropped
    assert new_a.sum() - old_a.sum() == pytest.approx(0.5 * (104 / 101 - 1))


def test_backtest_no_turnover_on_held_name_while_halted():
    data = _sig_data(HALT, {"A": [1, 1, NAN, NAN, 1, 1, 1, 1], "B": [-1] * 8})
    r = Backtest(qex("field('sig')"), transaction_cost=0.01).run(data)
    assert (r.weights["A"].diff().iloc[1:] == 0).all()


def test_backtest_holds_delisted_position_then_drops_it():
    p = _frame({"A": [10, 11, 12] + [NAN] * 14, "B": list(range(50, 67))})
    data = _sig_data(p, {"A": [1, 1, 1] + [NAN] * 14, "B": [-1] * 17})
    r = Backtest(qex("field('sig')")).run(data)
    a = r.weights["A"].to_numpy()
    np.testing.assert_allclose(a[:3 + MAX_PRICE_GAP], 0.5)
    assert (a[3 + MAX_PRICE_GAP:] == 0).all()
    pd.testing.assert_series_equal(r.pnl, _pnl(r.weights, p), check_names=False)
    # no PnL is booked on A after its last price
    a_fwd = forward_returns(p)["A"].reindex(r.weights.index)
    assert a_fwd.iloc[2:].isna().all()


def test_backtest_multi_day_holding_period_bridges_halt():
    data = _sig_data(HALT, {"A": [1] * 8, "B": [-1] * 8})
    r = Backtest(qex("field('sig')"), holding_period=2).run(data)
    np.testing.assert_allclose(r.weights["A"], [0.25] + [0.5] * 6)   # half the book on day one, then all of it
    pd.testing.assert_series_equal(r.pnl, _pnl(r.weights, HALT), check_names=False)
    # A is held through its halt and the day before it reopens books the whole move, once
    assert r.pnl.loc[HALT.index[3]] == pytest.approx(0.5 * (104 / 101 - 1) - 0.5 * (54 / 53 - 1))


def test_backtest_without_missing_prices_is_unchanged():
    rng = np.random.default_rng(2)
    p = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (60, 5)), axis=0)),
                     index=_dates(60), columns=list("ABCDE"))
    sig = qex("rank(-returns(3)) - 0.5")
    r = Backtest(sig).run({"close": p})
    with compute_context():
        w = sig.to_weights({"close": p})
    old_fwd = p.pct_change(fill_method=None).shift(-1)
    old = (w * old_fwd).sum(axis=1).loc[r.pnl.index]
    pd.testing.assert_series_equal(r.pnl, old, check_names=False)


# --- quantile / IC analysis -----------------------------------------------------------------

def test_quantile_returns_include_move_across_halt():
    data = _sig_data(HALT, {"A": [1] * 8, "B": [0] * 8})
    result = QuantileAnalysis(qex("field('sig')"), n_quantiles=2).run(data)
    top = result.quantile_returns.iloc[:, -1]
    fwd = forward_returns(HALT)["A"]
    # the day before A reopens, the top quantile (A) earns the whole halt move
    assert top.loc[HALT.index[3]] == pytest.approx(104 / 101 - 1)
    # halt days with no next price contribute no fake 0% return
    shared = top.index.intersection(fwd.dropna().index)
    np.testing.assert_allclose(top.loc[shared], fwd.loc[shared])


def test_ic_summary_uses_forward_returns_across_halt():
    rng = np.random.default_rng(3)
    n = 40
    p = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.02, (n, 6)), axis=0)),
                     index=_dates(n), columns=list("ABCDEF"))
    p.iloc[10:13, 0] = NAN                             # A halts for three days
    sig = pd.DataFrame(rng.normal(size=(n, 6)), index=p.index, columns=p.columns)
    data = {"close": p, "sig": sig}
    out = QuantileAnalysis(qex("field('sig')"), n_quantiles=2, holding_period=2).ic_summary(data)
    fwd = forward_returns(p, 2)
    ics = []
    for d in p.index:
        s_row, f_row = sig.loc[d], fwd.loc[d]
        ok = s_row.notna() & f_row.notna()
        if ok.sum() >= 5:
            ics.append(s_row[ok].corr(f_row[ok], method="spearman"))
    assert out["Mean Rank IC"] == pytest.approx(np.mean(ics))


def test_ic_summary_with_too_few_names_returns_nan():
    data = _sig_data(HALT, {"A": [1, 2, 3, 4, 5, 6, 7, 8], "B": [8, 7, 6, 5, 4, 3, 2, 1]})
    out = QuantileAnalysis(qex("field('sig')"), n_quantiles=2).ic_summary(data)
    assert np.isnan(out["Mean Rank IC"])
