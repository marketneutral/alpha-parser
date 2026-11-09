"""days_since(x): trading days since x was last nonzero, with no window, through missing days, missing before the
first event, and never ahead of the data."""
import numpy as np
import pandas as pd
import pytest

from qex import SignalGraph, alpha, qex
from qex.adjust import check

N = np.nan


@pytest.fixture
def data():
    idx = pd.bdate_range("2020-01-01", periods=10)
    ev = pd.DataFrame({
        "two": [0, 1, 0, 0, 1, 0, 0, 0, 0, 0],      # two events in a 10-day window
        "never": [0] * 10,                          # never true
        "halt": [0, 1, 0, N, N, 0, 0, 0, 0, 0],     # a halt inside the gap
        "signs": [-2, 0, 0, np.inf, 0, 0, 0, 0, 0, 0],   # a negative value and an infinity are events
        "late": [N, N, N, 0, 0, 0, 1, 0, 0, 0],     # listed later, one event
    }, index=idx, dtype=float)
    return {"close": ev * 0 + 10.0, "ev": ev}


def _col(data, expr, name):
    return qex(expr).evaluate(data)[name].tolist()


def _same(got, want):
    assert len(got) == len(want)
    for g, w in zip(got, want):
        assert (np.isnan(g) and np.isnan(w)) or g == w, (got, want)


def test_counts_from_the_latest_event(data):
    _same(_col(data, "days_since(field('ev'))", "two"), [N, 0, 1, 2, 0, 1, 2, 3, 4, 5])


def test_ts_argmax_counts_back_to_the_older_of_two(data):
    """The reason for days_since: ts_argmax with two events in its window points at the older one."""
    am = qex("ts_argmax(field('ev'), 10)").evaluate(data)["two"].iloc[-1]
    ds = qex("days_since(field('ev'))").evaluate(data)["two"].iloc[-1]
    assert am == 8 and ds == 5


def test_missing_before_the_first_event_and_never_true(data):
    _same(_col(data, "days_since(field('ev'))", "never"), [N] * 10)
    _same(_col(data, "days_since(field('ev'))", "late"), [N] * 6 + [0, 1, 2, 3])


def test_a_halt_neither_resets_nor_stops_the_count(data):
    _same(_col(data, "days_since(field('ev'))", "halt"), [N, 0, 1, 2, 3, 4, 5, 6, 7, 8])


def test_negative_values_and_infinity_are_events(data):
    _same(_col(data, "days_since(field('ev'))", "signs"), [0, 1, 2, 0, 1, 2, 3, 4, 5, 6])


def test_a_condition_and_delay(data):
    _same(_col(data, "days_since(field('ev') > 0)", "signs"), [N, N, N, 0, 1, 2, 3, 4, 5, 6])
    _same(_col(data, "delay(days_since(field('ev')), 1)", "two"), [N, N, 0, 1, 2, 0, 1, 2, 3, 4])


def test_no_look_ahead(data):
    """Each day's answer depends only on that day and earlier: cutting the data later changes nothing before."""
    full = qex("days_since(field('ev'))").evaluate(data)
    cut = {k: v.iloc[:6] for k, v in data.items()}
    early = qex("days_since(field('ev'))").evaluate(cut)
    pd.testing.assert_frame_equal(full.iloc[:6], early)


def test_series_input():
    idx = pd.bdate_range("2020-01-01", periods=5)
    d = {"close": pd.DataFrame({"A": [1.0] * 5}, index=idx), "m": pd.Series([0, 1, 0, 0, 1], index=idx, dtype=float)}
    out = qex("days_since(field('m'))").evaluate(d)
    assert out.iloc[:, 0].tolist()[1:] == [0, 1, 2, 0]


def test_constant_is_refused(data):
    with pytest.raises(Exception, match="days_since"):
        qex("days_since(1)").evaluate(data)


def test_parses_counts_and_is_unit_free(data):
    a = alpha("days_since(field('ev'))")
    g = SignalGraph(); g.add_signal(a, name="a")
    assert g.stats()["total_nodes"] == 2
    assert a._cache_key() == qex("days_since( field('ev') )")._cache_key()
    fields = {"ev": dict(unit="number", adjustment=0), "close": dict(unit="usd_per_share", adjustment=1)}
    assert check(qex("rank(days_since(field('ev')))"), fields, mode="warn") == []
    assert check(qex("rank(days_since(close() > ts_max(close(), 5)))"), fields, mode="warn") == []
