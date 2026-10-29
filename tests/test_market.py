"""market(), date-only (Series) fields, returns/volatility of another price field, and regression_neut."""

import numpy as np
import pandas as pd
import pytest

from qex import qex, compute_context, LazyData, MARKET_FIELD


@pytest.fixture
def mkt_data():
    """Stocks driven by a market factor with known betas, NaN gaps, and a market series with a missing day."""
    rng = np.random.default_rng(11)
    dates = pd.date_range('2020-01-01', periods=400, freq='B')
    tickers = [f'T{i}' for i in range(30)]
    m = rng.normal(0.0003, 0.01, len(dates))
    betas = np.linspace(0.2, 1.8, len(tickers))
    r = m[:, None] * betas[None, :] + rng.normal(0, 0.01, (len(dates), len(tickers)))
    close = pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=dates, columns=tickers)
    close.iloc[50:60, 3] = np.nan
    close.iloc[200, 7] = np.nan
    market = pd.Series(100 * np.exp(np.cumsum(m)), index=dates)
    return {'close': close, 'volume': close * 0 + 1e6, 'market': market}, pd.Series(betas, index=tickers)


def test_market_field_name():
    assert MARKET_FIELD == 'market'


def test_market_broadcasts_series(mkt_data):
    data, _ = mkt_data
    out = qex("market()").evaluate(data)
    assert out.shape == data['close'].shape
    assert out.index.equals(data['close'].index) and out.columns.equals(data['close'].columns)
    for t in ['T0', 'T3', 'T29']:
        np.testing.assert_array_equal(out[t].to_numpy(), data['market'].to_numpy())


def test_field_broadcasts_any_series_and_missing_dates_are_nan(mkt_data):
    data, _ = mkt_data
    s = data['market'].drop(data['market'].index[10])
    data = {**data, 'spy': s}
    out = qex("field('spy')").evaluate(data)
    assert out.iloc[10].isna().all()
    np.testing.assert_array_equal(out.iloc[11].to_numpy(), np.full(30, s.iloc[10]))


def test_series_field_without_close_uses_first_dataframe(mkt_data):
    data, _ = mkt_data
    data = {'volume': data['volume'], 'market': data['market']}
    out = qex("market()").evaluate(data)
    assert out.shape == data['volume'].shape


def test_series_field_with_no_panel_raises():
    with pytest.raises(ValueError, match="no DataFrame field"):
        qex("market()").evaluate({'market': pd.Series([1.0, 2.0])})


def test_missing_market_raises(mkt_data):
    data, _ = mkt_data
    with pytest.raises(ValueError, match="'market' not found"):
        qex("market()").evaluate({'close': data['close']})


def test_market_works_with_lazydata(mkt_data):
    data, _ = mkt_data
    lazy = LazyData({'close': lambda: data['close'], 'market': lambda: data['market']})
    out = qex("returns(1, 'market')").evaluate(lazy)
    np.testing.assert_allclose(out['T5'].to_numpy(), data['market'].pct_change().to_numpy(), equal_nan=True)


def test_dataframe_fields_unchanged(mkt_data):
    data, _ = mkt_data
    assert qex("field('close')").evaluate(data) is data['close']
    pd.testing.assert_frame_equal(qex("returns(5)").evaluate(data), data['close'].pct_change(5))
    pd.testing.assert_frame_equal(qex("returns(5, 'close')").evaluate(data), data['close'].pct_change(5))


def test_market_returns_and_volatility(mkt_data):
    data, _ = mkt_data
    mr = data['market'].pct_change()
    out = qex("returns(1, 'market')").evaluate(data)
    np.testing.assert_allclose(out['T0'].to_numpy(), mr.to_numpy(), equal_nan=True)
    vol = qex("volatility(63, 'market')").evaluate(data)
    np.testing.assert_allclose(vol['T9'].to_numpy(), (mr.rolling(63, min_periods=20).std() * np.sqrt(252)).to_numpy(), equal_nan=True)


def test_market_beta_recovers_true_betas(mkt_data):
    data, betas = mkt_data
    b = qex("ts_beta(returns(1), returns(1, 'market'), 252)").evaluate(data).iloc[-1]
    assert np.corrcoef(b, betas)[0, 1] > 0.95
    assert np.abs(b - betas).max() < 0.25


def test_market_cache_keys_distinct(mkt_data):
    data, _ = mkt_data
    with compute_context():
        a = qex("returns(1)").evaluate(data)
        m = qex("returns(1, 'market')").evaluate(data)
    assert not np.allclose(a['T1'].fillna(0), m['T1'].fillna(0))


def _random_panel(seed, nan_frac=0.1, shape=(40, 25)):
    rng = np.random.default_rng(seed)
    v = pd.DataFrame(rng.normal(size=shape), index=pd.date_range('2021-01-01', periods=shape[0], freq='B'),
                     columns=[f'T{i}' for i in range(shape[1])])
    return v.mask(rng.random(shape) < nan_frac)


def test_regression_neut_matches_ols_residual():
    y, x = _random_panel(1), _random_panel(2)
    data = {'close': y.abs() + 1, 'y': y, 'x': x}
    out = qex("regression_neut(field('y'), field('x'))").evaluate(data)
    for d in y.index[:10]:
        ok = y.loc[d].notna() & x.loc[d].notna()
        X = np.column_stack([np.ones(ok.sum()), x.loc[d, ok]])
        coef, *_ = np.linalg.lstsq(X, y.loc[d, ok], rcond=None)
        np.testing.assert_allclose(out.loc[d, ok].to_numpy(), (y.loc[d, ok] - X @ coef).to_numpy(), atol=1e-12)
        assert out.loc[d, ~ok].isna().all()


def test_regression_neut_zero_mean_and_zero_covariance():
    y, x = _random_panel(3), _random_panel(4)
    out = qex("regression_neut(field('y'), field('x'))").evaluate({'close': y.abs() + 1, 'y': y, 'x': x})
    xv = x.where(out.notna())
    np.testing.assert_allclose(out.mean(axis=1), 0, atol=1e-12)
    np.testing.assert_allclose((out * xv.sub(xv.mean(axis=1), axis=0)).sum(axis=1), 0, atol=1e-12)


def test_regression_neut_constant_x_is_demean():
    y = _random_panel(5, nan_frac=0)
    x = y * 0 + 3.0
    out = qex("regression_neut(field('y'), field('x'))").evaluate({'close': y.abs() + 1, 'y': y, 'x': x})
    pd.testing.assert_frame_equal(out, y.sub(y.mean(axis=1), axis=0))


def test_regression_neut_removes_market_beta(mkt_data):
    data, betas = mkt_data
    # a signal that is mostly beta: after neutralizing, its weights have ~zero ex-ante beta
    e = "let b = ts_beta(returns(1), returns(1, 'market'), 252) in regression_neut(b + 0.1 * returns(5), b)"
    out = qex(e).evaluate(data).iloc[-1]
    b = qex("ts_beta(returns(1), returns(1, 'market'), 252)").evaluate(data).iloc[-1]
    assert abs((out * b).sum()) < 1e-10 and abs(out.sum()) < 1e-10


def test_min_periods_optional_and_default_unchanged(mkt_data):
    data, _ = mkt_data
    e = "ts_beta(returns(1), returns(1, 'market'), 252)"
    assert qex(e).min_periods == 20 and qex("ts_corr(returns(1), close(), 60)").min_periods == 20
    short = qex("ts_beta(returns(1), returns(1, 'market'), 252, 200)").evaluate(data)
    full = qex(e).evaluate(data)
    assert short.iloc[:200].isna().all().all() and full.iloc[25:200].notna().any().any()
    # where the stricter estimate exists it is the same number; it is missing only where fewer than 200 days are paired
    pd.testing.assert_frame_equal(short.dropna(how='all'), full.where(short.notna()).dropna(how='all'))
    paired = (qex("returns(1)").evaluate(data).notna()).astype(float).rolling(252, min_periods=1).sum()
    assert (short.isna() == (paired < 200) | full.isna()).all().all()
    assert qex("ts_cov(returns(1), close(), 60, 40)").min_periods == 40
    assert qex("ts_corr(returns(1), close(), 60, 999)").min_periods == 60     # capped at the window


def _ref_shrunk(y, x, period, min_periods):
    """Per-name, per-date Vasicek shrinkage computed directly from the window's paired observations."""
    raw, se2 = pd.DataFrame(np.nan, index=y.index, columns=y.columns), pd.DataFrame(np.nan, index=y.index, columns=y.columns)
    for c in y.columns:
        for i in range(len(y)):
            yy, xx = y[c].iloc[max(0, i - period + 1):i + 1], x[c].iloc[max(0, i - period + 1):i + 1]
            ok = yy.notna() & xx.notna()
            n = ok.sum()
            if n < min_periods or xx[ok].var() == 0:
                continue
            yy, xx = yy[ok].to_numpy(), xx[ok].to_numpy()
            b = np.cov(yy, xx)[0, 1] / np.var(xx, ddof=1)
            res = yy - yy.mean() - b * (xx - xx.mean())
            raw.iloc[i, raw.columns.get_loc(c)] = b
            se2.iloc[i, se2.columns.get_loc(c)] = (res @ res / (n - 2)) / ((n - 1) * np.var(xx, ddof=1))
    tau2 = raw.var(axis=1)
    w = tau2.values[:, None] / (tau2.values[:, None] + se2)
    return w * raw + (1 - w) * raw.mean(axis=1).values[:, None]


def test_ts_beta_shrunk_matches_reference(mkt_data):
    data, _ = mkt_data
    small = {'close': data['close'].iloc[:90, :8], 'market': data['market'].iloc[:90]}
    small['close'].iloc[30:45, 2] = np.nan
    out = qex("ts_beta_shrunk(returns(1), returns(1, 'market'), 40, 10)").evaluate(small)
    y = qex("returns(1)").evaluate(small)
    x = qex("returns(1, 'market')").evaluate(small)
    ref = _ref_shrunk(y, x, 40, 10)
    np.testing.assert_allclose(out.to_numpy(), ref.to_numpy(), rtol=1e-9, atol=1e-12, equal_nan=True)


def test_ts_beta_shrunk_pulls_noisy_betas_in(mkt_data):
    data, betas = mkt_data
    raw = qex("ts_beta(returns(1), returns(1, 'market'), 252)").evaluate(data)
    shr = qex("ts_beta_shrunk(returns(1), returns(1, 'market'), 252)").evaluate(data)
    # shrinkage never moves a beta past the mean, and the cross-section is tighter
    d = data['close'].index[-1]
    mean = raw.loc[d].mean()
    assert ((shr.loc[d] - mean).abs() <= (raw.loc[d] - mean).abs() + 1e-12).all()
    assert shr.loc[d].std() < raw.loc[d].std()
    # a new listing with 25 days of history is shrunk much harder than names with a full year
    late = data['close'].copy(); late.iloc[:-25, 0] = np.nan
    s2 = qex("ts_beta_shrunk(returns(1), returns(1, 'market'), 252)").evaluate({**data, 'close': late}).iloc[-1]
    r2 = qex("ts_beta(returns(1), returns(1, 'market'), 252)").evaluate({**data, 'close': late}).iloc[-1]
    m2 = r2.mean()
    pull = 1 - (s2 - m2).abs() / (r2 - m2).abs()
    assert pull['T0'] > 3 * pull.drop('T0').median()


def test_ts_beta_uses_paired_days_only(mkt_data):
    # T3 has a 10-day gap (11 missing returns); var(x) must skip those days too, not just cov(y, x)
    data, _ = mkt_data
    y = qex("returns(1)").evaluate(data)
    x = qex("returns(1, 'market')").evaluate(data)
    both = y.notna() & x.notna()
    yp, xp = y.where(both), x.where(both)
    out = qex("ts_beta(returns(1), returns(1, 'market'), 60)").evaluate(data)
    ref = yp.rolling(60, min_periods=20).cov(xp) / xp.rolling(60, min_periods=20).var()
    np.testing.assert_allclose(out.to_numpy(), ref.to_numpy(), rtol=1e-12, equal_nan=True)
    ew = qex("ts_beta_ewma(returns(1), returns(1, 'market'), 20)").evaluate(data)
    ref_ew = yp.ewm(halflife=20).cov(xp) / xp.ewm(halflife=20).var()
    np.testing.assert_allclose(ew.to_numpy(), ref_ew.to_numpy(), rtol=1e-12, equal_nan=True)
    # the unpaired version differs on the gapped name inside the window, and matches elsewhere
    unpaired = y.rolling(60, min_periods=20).cov(x) / x.rolling(60, min_periods=20).var()
    assert not np.allclose(out['T3'].iloc[70:100], unpaired['T3'].iloc[70:100])
    np.testing.assert_allclose(out['T1'].to_numpy(), unpaired['T1'].to_numpy(), rtol=1e-12, equal_nan=True)
