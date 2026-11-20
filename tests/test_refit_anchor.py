"""Fits re-made on a calendar (qex/schedule.py): the same day has the same fit whatever row the data begins on.

Before 0.19.0, ts_match and pca_factor counted their refits in rows from the data's first row, so two evaluations of
one expression, on data loaded from different dates, chose different twins on the same day (23% the same on real data).
"""
import numpy as np
import pandas as pd
import pytest

from qex import compute_context, qex
from qex.schedule import EPOCH, refit_rows


def _data(start="2019-01-01", T=520, N=24, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=T)
    idx = idx[~idx.isin(pd.to_datetime(["2019-07-04", "2019-12-25", "2020-01-01", "2020-07-03", "2020-12-25"]))]
    common = rng.normal(0, 0.01, (len(idx), 4))
    load = rng.normal(0, 1, (4, N))
    r = common @ load * 0.5 + rng.normal(0, 0.01, (len(idx), N))
    close = pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), idx, [f"S{i:02d}" for i in range(N)])
    sector = pd.DataFrame([np.where(np.arange(N) < N // 2, "A", "B")] * len(idx), index=idx, columns=close.columns)
    return {"close": close, "r": close.pct_change(), "sector": sector}


def _ev(expr, d):
    with compute_context():
        return qex(expr).evaluate(d)


def _from(d, day):
    """The same data, loaded from `day` on."""
    return {k: (v.loc[day:] if isinstance(v, pd.DataFrame) else v) for k, v in d.items()}


def test_the_schedule_is_the_calendar_s():
    idx = pd.bdate_range("2020-01-01", periods=200)
    rows = refit_rows(idx, 0, 21)
    days = np.busday_count(EPOCH, idx.values.astype("datetime64[D]"))
    blocks = days[rows] // 21
    assert (np.diff(blocks) == 1).all()                                   # one refit per block, none skipped
    assert all(days[r] % 21 == 0 for r in rows)                           # on the block's first weekday (no holidays)
    later = refit_rows(idx[37:], 0, 21) + 37
    assert set(later) == set(rows[rows >= 37])                            # a later start: the same days


def test_a_holiday_moves_the_refit_to_the_next_trading_day():
    idx = pd.bdate_range("2020-01-01", periods=120)
    full = refit_rows(idx, 0, 21)
    hole = idx[full[2]]
    gappy = idx.drop(hole)
    rows = refit_rows(gappy, 0, 21)
    assert gappy[rows[2]] == idx[full[2] + 1]                             # the block's first trading day


def test_the_old_rule_for_an_index_that_is_not_dates():
    assert list(refit_rows(pd.RangeIndex(100), 10, 25)) == [10, 35, 60, 85]


@pytest.mark.parametrize("expr", [
    "ts_match(identifier(), returns(1), 120, within='sector', matching='exclusive', rebalance=21)",
    "ts_match(identifier(), returns(1), 120, metric='path', within='sector', rebalance=21)",
    "ts_match_score(returns(1), 120, metric='corr', within='sector', rebalance=21)",
])
def test_ts_match_does_not_depend_on_where_the_data_starts(expr):
    d = _data()
    a = _ev(expr, d)
    for later in ("2019-01-24", "2019-03-07"):                            # different rows of the same calendar
        b = _ev(expr, _from(d, later))
        common = b.index[b.notna().any(axis=1)]
        assert len(common) > 200
        x, y = a.loc[common], b.loc[common]
        both = x.notna() & y.notna()
        assert both.to_numpy().sum() > 1000
        if pd.api.types.is_numeric_dtype(x.dtypes.iloc[0]):
            assert np.allclose(x[both].to_numpy(float), y[both].to_numpy(float), equal_nan=True)
        else:                                                             # labels: the twins themselves
            assert (x[both] == y[both]).to_numpy()[both.to_numpy()].all()


def test_pca_factor_s_decompositions_do_not_depend_on_where_the_data_starts():
    d = _data()
    a = _ev("pca_factor(field('r'), 120, 1, 21)", d)
    b = _ev("pca_factor(field('r'), 120, 1, 21)", _from(d, "2019-02-14"))
    common = b.index[b.notna().any(axis=1)]
    assert len(common) > 200
    # the same decomposition on the same days: the same betas up to the factor's sign, which is followed from the
    # first refit in the data (the docs say so)
    assert np.allclose(np.abs(a.loc[common].to_numpy()), np.abs(b.loc[common].to_numpy()), equal_nan=True)
