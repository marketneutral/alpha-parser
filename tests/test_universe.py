"""Cross-sectional universe: cross-sectional operators compare members only; time-series operators see all history."""

import numpy as np
import pandas as pd
import pytest

from qex import qex, compute_context, cross_section_universe
from evaluation import BookSettings, build_book


@pytest.fixture
def uni_data():
    """Prices for 30 names; membership that changes over time (names join and leave); sectors; a market."""
    rng = np.random.default_rng(5)
    dates = pd.date_range('2021-01-01', periods=300, freq='B')
    tickers = [f'T{i}' for i in range(30)]
    m = rng.normal(0, 0.01, len(dates))
    r = m[:, None] * np.linspace(0.3, 1.7, 30)[None, :] + rng.normal(0, 0.015, (300, 30))
    close = pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=dates, columns=tickers)
    close.iloc[40:50, 4] = np.nan
    member = pd.DataFrame(1.0, index=dates, columns=tickers)
    member.iloc[:, 20:] = np.nan                         # T20-T29 never members
    member.iloc[:150, 10:15] = np.nan                    # T10-T14 join on day 150
    member.iloc[200:, 0:3] = np.nan                      # T0-T2 leave on day 200
    sectors = pd.DataFrame(np.tile(np.array(['A', 'B', 'C'] * 10, dtype=object), (300, 1)), index=dates, columns=tickers)
    return {'close': close, 'volume': close * 0 + 1e6, 'member': member, 'sector': sectors,
            'market': pd.Series(100 * np.exp(np.cumsum(m)), index=dates)}


def _members(data):
    return data['member'].notna()


CROSS_SECTIONAL = [
    ("rank(returns(5))", lambda x: x.rank(axis=1, pct=True)),
    ("demean(returns(5))", lambda x: x.sub(x.mean(axis=1), axis=0)),
    ("zscore(returns(5))", lambda x: x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1), axis=0)),
    ("scale(returns(5))", lambda x: x.div(x.abs().sum(axis=1), axis=0)),
]


@pytest.mark.parametrize("expr,ref", CROSS_SECTIONAL)
def test_cross_sectional_ops_compare_members_only(uni_data, expr, ref):
    data = uni_data
    x = qex("returns(5)").evaluate(data)
    with cross_section_universe(_members(data)):
        out = qex(expr).evaluate(data)
    expected = ref(x.where(_members(data)))
    pd.testing.assert_frame_equal(out, expected)
    assert out.where(~_members(data)).isna().all().all()


def test_unrestricted_by_default(uni_data):
    data = uni_data
    out = qex("rank(returns(5))").evaluate(data)
    pd.testing.assert_frame_equal(out, qex("returns(5)").evaluate(data).rank(axis=1, pct=True))
    assert out.iloc[-1, 25] == out.iloc[-1, 25]          # a never-member is ranked when no universe is set


def test_group_ops_compare_members_only(uni_data):
    data = uni_data
    x = qex("returns(5)").evaluate(data)
    with cross_section_universe(_members(data)):
        gd = qex("group_demean(returns(5), 'sector')").evaluate(data)
        gr = qex("group_rank(returns(5), 'sector')").evaluate(data)
    xm = x.where(_members(data))
    for d in [data['close'].index[100], data['close'].index[250]]:
        for s in 'ABC':
            cols = data['sector'].loc[d] == s
            v = xm.loc[d, cols]
            np.testing.assert_allclose(gd.loc[d, cols].to_numpy(), (v - v.mean()).to_numpy(), equal_nan=True)
            np.testing.assert_allclose(gr.loc[d, cols].to_numpy(), v.rank(pct=True).to_numpy(), equal_nan=True)


def test_regression_neut_fits_on_members(uni_data):
    data = uni_data
    with cross_section_universe(_members(data)):
        out = qex("regression_neut(returns(5), ts_beta(returns(1), returns(1, 'market'), 60))").evaluate(data)
        b = qex("ts_beta(returns(1), returns(1, 'market'), 60)").evaluate(data)
    d = data['close'].index[-1]
    mem = _members(data).loc[d] & out.loc[d].notna()
    assert abs((out.loc[d, mem] * (b.loc[d, mem] - b.loc[d, mem].mean())).sum()) < 1e-12
    assert out.loc[d, ~_members(data).loc[d]].isna().all()


def test_time_series_ops_see_full_history(uni_data):
    data = uni_data
    plain = qex("ts_beta(returns(1), returns(1, 'market'), 60)").evaluate(data)
    with cross_section_universe(_members(data)):
        masked = qex("ts_beta(returns(1), returns(1, 'market'), 60)").evaluate(data)
    pd.testing.assert_frame_equal(plain, masked)
    assert masked.iloc[160, 12] == masked.iloc[160, 12]  # T12 joined on day 150 and already has a 60-day beta


def test_time_series_of_cross_sectional_starts_at_join(uni_data):
    data = uni_data
    with cross_section_universe(_members(data)):
        out = qex("ts_mean(rank(returns(1)), 20)").evaluate(data)
    assert out.iloc[140:150, 12].isna().all()            # before T12 joined
    assert out.iloc[150 + 19, 12] == out.iloc[150 + 19, 12]


def test_cache_never_crosses_universes(uni_data):
    data = uni_data
    with compute_context():
        a = qex("rank(returns(5))").evaluate(data)
        with cross_section_universe(_members(data)):
            b = qex("rank(returns(5))").evaluate(data)
        c = qex("rank(returns(5))").evaluate(data)
    assert not a.equals(b)
    pd.testing.assert_frame_equal(a, c)


def test_universe_field_values_accepted(uni_data):
    data = uni_data
    with cross_section_universe(_members(data)):
        a = qex("rank(returns(5))").evaluate(data)
    with cross_section_universe(data['member']):           # 1 in, NaN out
        b = qex("rank(returns(5))").evaluate(data)
    pd.testing.assert_frame_equal(a, b)


def test_ts_beta_shrunk_prior_from_universe(uni_data):
    data = uni_data
    raw = qex("ts_beta(returns(1), returns(1, 'market'), 60)").evaluate(data)
    with cross_section_universe(_members(data)):
        shr = qex("ts_beta_shrunk(returns(1), returns(1, 'market'), 60)").evaluate(data)
    d = data['close'].index[-1]
    prior = raw.loc[d][_members(data).loc[d]].mean()
    # every name, member or not, keeps a beta and is pulled toward the members' mean
    assert shr.loc[d].notna().all()
    assert ((shr.loc[d] - prior).abs() <= (raw.loc[d] - prior).abs() + 1e-12).all()


def test_build_book_masking(uni_data):
    data = uni_data
    s = BookSettings(delay=1, universe='member', truncation=0.2)
    off = build_book(qex("power(rank(returns(5)), 3)"), data, s)
    on = build_book(qex("power(rank(returns(5)), 3)"), data, BookSettings(delay=1, universe='member', truncation=0.2, mask_cross_section=True))
    # the masked book equals a book from the same signal computed by hand among members
    x = qex("returns(5)").evaluate(data).where(_members(data))
    manual = build_book(qex("field('r')"), {**data, 'r': x.rank(axis=1, pct=True) ** 3}, s)
    pd.testing.assert_frame_equal(on, manual)
    assert not np.allclose(on.to_numpy(), off.to_numpy())
    # off is unchanged from evaluating without any universe context
    with cross_section_universe(None):
        again = build_book(qex("power(rank(returns(5)), 3)"), data, s)
    pd.testing.assert_frame_equal(off, again)


def test_mask_cross_section_needs_universe():
    with pytest.raises(ValueError, match="needs a universe"):
        BookSettings(mask_cross_section=True)
