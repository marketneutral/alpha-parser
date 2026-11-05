"""Event windows (ts_mean_events, ts_std_events, ts_sum_events) with a minimum number of events."""
import numpy as np
import pandas as pd
import pytest

from qex import qex

N = np.nan


def old_loop(values, n, func):
    """The cell-by-cell implementation before 0.10.0, as the reference for the default (all n events)."""
    out = pd.DataFrame(index=values.index, columns=values.columns, dtype=float)
    for col in values.columns:
        seen, res = [], np.full(len(values), np.nan)
        for i, v in enumerate(values[col].to_numpy()):
            if not np.isnan(v):
                seen.append(v)
            if len(seen) >= n:
                res[i] = func(seen[-n:])
        out[col] = res
    return out


@pytest.fixture
def x():
    rng = np.random.default_rng(7)
    v = rng.normal(size=(200, 5))
    v[rng.random(v.shape) < 0.4] = np.nan
    return pd.DataFrame(v, index=pd.bdate_range("2020-01-01", periods=200), columns=list("ABCDE"))


@pytest.mark.parametrize("op, func", [
    ("ts_mean_events", np.mean),
    ("ts_sum_events", np.sum),
    ("ts_std_events", lambda s: np.std(s, ddof=1) if len(s) > 1 else np.nan),
])
def test_the_default_is_unchanged(x, op, func):
    got = qex(f"{op}(field('x'), 6)").evaluate({"x": x})
    pd.testing.assert_frame_equal(got, old_loop(x, 6, func), check_freq=False)


def test_a_minimum_gives_a_value_from_the_min_events_th_event():
    x = pd.DataFrame({"A": [1, N, 2, 3, N, 4, 5]}, index=pd.bdate_range("2024-01-01", periods=7), dtype=float)
    full = qex("ts_mean_events(field('x'), 4)").evaluate({"x": x})["A"].tolist()
    part = qex("ts_mean_events(field('x'), 4, 2)").evaluate({"x": x})["A"].tolist()
    assert np.allclose(full, [N, N, N, N, N, 2.5, 3.5], equal_nan=True)
    assert np.allclose(part, [N, N, 1.5, 2, 2, 2.5, 3.5], equal_nan=True)


def test_a_partial_sum_over_sessions():
    """The universe rule's dollar volume: sum over the last 20 sessions, a zero-volume session counting as one."""
    idx = pd.bdate_range("2024-01-01", periods=5)
    d = {"p": pd.DataFrame({"A": [10, 10, N, 10, 10]}, index=idx, dtype=float),
         "v": pd.DataFrame({"A": [100, 0, N, 0, 300]}, index=idx, dtype=float)}
    traded = qex("ts_sum_events(field('v') > 0, 3, 1)").evaluate(d)["A"].tolist()
    dollars = qex("ts_sum_events(field('p') * field('v'), 3, 1)").evaluate(d)["A"].tolist()
    assert traded == [1, 1, 1, 1, 1]                      # day 4's last 3 sessions are (100, 0, 0); day 5's (0, 0, 300)
    assert dollars == [1000, 1000, 1000, 1000, 3000]      # a day with no session (day 3) carries the window


@pytest.mark.parametrize("expr", ["ts_mean_events(close(), 5, 6)", "ts_mean_events(close(), 5, 0)",
                                  "ts_sum_events(close(), 0)", "ts_std_events(close(), 5, 2.5)"])
def test_bad_counts_are_refused(expr):
    with pytest.raises(ValueError, match="whole number"):
        qex(expr)
