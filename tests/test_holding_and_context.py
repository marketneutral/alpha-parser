"""Backtest with a holding period longer than a day (overlapping books, each day's move counted once), and a
compute_context that knows its data by identity (a new panel with the same shape is new data)."""
import numpy as np
import pandas as pd
import pytest

from qex import qex, compute_context, LazyData, forward_returns
from evaluation import Backtest


def _panel(seed, n=60, k=6):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n)
    cols = list("ABCDEF")[:k]
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n, k)), axis=0)), index=idx, columns=cols)
    sig = pd.DataFrame(rng.normal(size=(n, k)), index=idx, columns=cols)
    return {"close": close, "sig": sig}


# --- holding period ---------------------------------------------------------------------------------------------------

def test_a_constant_book_earns_the_same_whatever_the_holding_period():
    """Before 0.7.1 a 5-day holding period summed overlapping 5-day returns: about 5 times the P&L."""
    data = _panel(1)
    data["sig"] = data["sig"].iloc[:1].reindex(data["sig"].index, method="ffill")       # the same target every day
    daily = Backtest(qex("field('sig')")).run(data)
    held = Backtest(qex("field('sig')"), holding_period=5).run(data)
    after = daily.pnl.index[4:]                                                       # once all five books exist
    pd.testing.assert_series_equal(held.pnl.loc[after], daily.pnl.loc[after])
    pd.testing.assert_series_equal(held.returns, daily.returns)                      # partly in cash, not diluted


def test_the_held_book_is_the_average_of_the_last_h_targets():
    data, h = _panel(2), 3
    r = Backtest(qex("field('sig')"), holding_period=h).run(data)
    with compute_context():
        targets = qex("field('sig')").to_weights(data).fillna(0)
    books = sum(targets.shift(i).fillna(0) for i in range(h)) / h                  # h books, each a 1/h slice
    pd.testing.assert_frame_equal(r.weights, books.loc[r.weights.index], check_freq=False)
    fwd = forward_returns(data["close"], 1).reindex_like(r.weights).fillna(0)
    pd.testing.assert_series_equal(r.pnl, (r.weights * fwd).sum(axis=1))
    to = r.weights.diff().abs().sum(axis=1)
    to.iloc[0] = r.weights.iloc[0].abs().sum()
    assert r.avg_turnover == pytest.approx(to.mean())


def test_a_longer_holding_period_trades_less():
    data = _panel(3)
    turnover = [Backtest(qex("field('sig')"), holding_period=h).run(data).avg_turnover for h in (1, 5, 20)]
    assert turnover[0] > turnover[1] > turnover[2]


def test_holding_period_must_be_positive():
    with pytest.raises(ValueError, match="holding_period"):
        Backtest(qex("field('sig')"), holding_period=0).run(_panel(4))


# --- compute_context --------------------------------------------------------------------------------------------------

def test_a_new_panel_with_the_same_shape_is_new_data():
    """Before 0.7.1 the context compared shapes and the ends of the index, and returned the old panel's result."""
    a, b = _panel(5), _panel(6)
    expr = qex("rank(field('sig'))")
    with compute_context():
        ra, rb = expr.evaluate(a), expr.evaluate(b)
    with compute_context():
        assert rb.equals(expr.evaluate(b)) and not ra.equals(rb)


def test_a_new_dict_around_the_same_frames_keeps_the_cache():
    data = _panel(7)
    expr = qex("rank(field('sig'))")
    with compute_context():
        first = expr.evaluate(data)
        assert expr.evaluate(dict(data)) is first                       # cached: the same frames
        changed = dict(data, sig=data["sig"] * -1)
        assert expr.evaluate(changed) is not first                      # a replaced field is new data


def test_a_change_in_place_needs_clear():
    data = _panel(8)
    expr = qex("field('sig') * 2")
    with compute_context() as ctx:
        before = expr.evaluate(data).iloc[0, 0]
        data["sig"].iloc[0, 0] = 100.0
        assert expr.evaluate(data).iloc[0, 0] == before                  # not detected: documented
        ctx.clear()
        assert expr.evaluate(data).iloc[0, 0] == 200.0


def test_lazy_data_loads_only_the_fields_used_in_a_context():
    data, loaded = _panel(9), []

    def loader(name):
        def load():
            loaded.append(name)
            return data[name]
        return load

    lazy = LazyData({"close": loader("close"), "sig": loader("sig")})
    with compute_context():
        qex("rank(field('sig'))").evaluate(lazy)
    assert loaded == ["sig"]
