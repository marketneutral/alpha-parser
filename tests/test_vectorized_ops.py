"""The vectorized group operators and decay_linear match the per-date loops they replaced."""

import pytest
import numpy as np
import pandas as pd

from qex import alpha


# Reference implementations: the original per-date / rolling-apply code, kept verbatim in
# behaviour so the vectorized versions can be checked against it. Inputs come from qex's own
# returns(1), whose handling of missing prices follows the installed pandas (pandas 2 pads
# gaps in pct_change by default, pandas 3 does not).

def _ref_group_apply(values, group_df, fn):
    result = pd.DataFrame(np.nan, index=values.index, columns=values.columns)
    for date in values.index:
        date_values = values.loc[date]
        date_groups = group_df.loc[date] if date in group_df.index else group_df.iloc[-1]
        for group_name in date_groups.unique():
            if pd.isna(group_name):
                continue
            mask = date_groups == group_name
            result.loc[date, mask] = fn(date_values[mask])
    return result


def ref_group_rank(values, group_df):
    return _ref_group_apply(values, group_df, lambda g: g.rank(pct=True))


def ref_group_demean(values, group_df):
    return _ref_group_apply(values, group_df, lambda g: g - g.mean())


def ref_group_sum(values, group_df):
    return _ref_group_apply(values, group_df, lambda g: g.sum())


def ref_decay_linear(values, period):
    weights = np.arange(1, period + 1, dtype=float)
    weights = weights / weights.sum()

    def _weighted_mean(arr):
        valid = ~np.isnan(arr)
        if valid.sum() == 0:
            return np.nan
        w = weights[valid]
        return np.sum(arr[valid] * w) / w.sum()

    return values.rolling(period).apply(_weighted_mean, raw=True)


@pytest.fixture
def messy_data():
    """Random data with NaN values, NaN groups, a group that changes over time, and ties."""
    rng = np.random.default_rng(7)
    dates = pd.date_range('2020-01-01', periods=60, freq='B')
    tickers = [f'T{i}' for i in range(25)]
    close = pd.DataFrame(100 * np.exp(rng.normal(0, 0.02, (60, 25)).cumsum(axis=0)),
                         index=dates, columns=tickers)
    close = close.mask(rng.random(close.shape) < 0.1)             # 10% missing prices
    close.iloc[:, 3] = np.nan                                     # a name with no data at all
    close.iloc[10, 5:9] = close.iloc[10, 4]                       # ties within a row
    labels = rng.choice(['Tech', 'Energy', 'Health', 'Utilities'], size=(60, 25)).astype(object)
    sector = pd.DataFrame(labels, index=dates, columns=tickers)
    sector = sector.mask(rng.random(sector.shape) < 0.1)          # 10% with no group
    sector.iloc[:, 0] = 'Solo'                                    # a group of one
    return {'close': close, 'sector': sector}


@pytest.mark.parametrize('name, ref', [
    ('group_rank', ref_group_rank),
    ('group_demean', ref_group_demean),
    ('group_sum', ref_group_sum),
])
def test_group_ops_match_loop(messy_data, name, ref):
    expr = f"{name}(returns(1), 'sector')"
    got = alpha(expr).evaluate(messy_data)
    want = ref(alpha("returns(1)").evaluate(messy_data), messy_data['sector'])
    pd.testing.assert_frame_equal(got, want, check_exact=False, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize('name, ref', [
    ('group_rank', ref_group_rank),
    ('group_demean', ref_group_demean),
    ('group_sum', ref_group_sum),
])
def test_group_ops_missing_dates_use_last_row(messy_data, name, ref):
    """Dates absent from the group frame fall back to its last row, as before."""
    sector = messy_data['sector'].iloc[::3]                       # two dates in three are missing
    data = {'close': messy_data['close'], 'sector': sector}
    got = alpha(f"{name}(close(), 'sector')").evaluate(data)
    want = ref(messy_data['close'], sector)
    pd.testing.assert_frame_equal(got, want, check_exact=False, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize('period', [1, 2, 5, 10, 59, 60, 61])
def test_decay_linear_matches_rolling_apply(messy_data, period):
    got = alpha(f"decay_linear(returns(1), {period})").evaluate(messy_data)
    want = ref_decay_linear(alpha("returns(1)").evaluate(messy_data), period)
    pd.testing.assert_frame_equal(got, want, check_exact=False, rtol=1e-12, atol=1e-15)


def test_group_demean_sums_to_zero_within_groups(messy_data):
    out = alpha("group_demean(close(), 'sector')").evaluate(messy_data)
    stacked = pd.DataFrame({'v': out.stack(), 'g': messy_data['sector'].stack()}).dropna()
    sums = stacked.groupby([stacked.index.get_level_values(0), 'g'])['v'].sum()
    assert np.allclose(sums, 0, atol=1e-9)
