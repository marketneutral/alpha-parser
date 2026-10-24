"""Book construction (BookSettings, build_book), the Finding Alphas metrics, and backward compatibility."""

import pytest
import numpy as np
import pandas as pd

from qex import alpha, compute_context, delay, close, field
from evaluation import (
    Backtest,
    BacktestResult,
    BookSettings,
    build_book,
    turnover,
    fitness,
    margin_bps,
    sharpe_ratio,
    annualized_return,
)


@pytest.fixture
def book_data():
    """60 names over 300 days, with missing prices, a universe that changes, and sectors with gaps."""
    rng = np.random.default_rng(11)
    dates = pd.date_range('2020-01-01', periods=300, freq='B')
    names = [f'N{i}' for i in range(60)]
    close_ = pd.DataFrame(100 * np.exp(rng.normal(0, 0.02, (300, 60)).cumsum(axis=0)), index=dates, columns=names)
    close_ = close_.mask(rng.random(close_.shape) < 0.03)
    close_.iloc[:, 7] = np.nan                                        # never priced
    member = pd.DataFrame(np.where(rng.random(close_.shape) < 0.85, 1.0, np.nan), index=dates, columns=names)
    labels = rng.choice(['Tech', 'Energy', 'Health', 'Utilities'], size=(300, 60)).astype(object)
    sector = pd.DataFrame(labels, index=dates, columns=names).mask(rng.random(close_.shape) < 0.05)
    return {'close': close_, 'member': member, 'sector': sector}


SIGNALS = ["-returns(5)", "power(returns(20), 3)", "rank(returns(10)) - 0.5"]


def _check_book(w, cap, data, settings):
    gross = w.abs().sum(axis=1)
    live = gross > 0
    assert live.sum() > 200
    assert np.allclose(gross[live], 1.0, atol=1e-12)
    if cap is not None:
        assert w.abs().max().max() <= cap + 1e-9
    priced = data['close'].reindex_like(w).notna()
    assert (w.where(~priced).fillna(0) == 0).all().all()
    if settings.universe:
        known = data[settings.universe].shift(settings.delay).reindex_like(w).notna()
        assert (w.where(~known).fillna(0) == 0).all().all()


@pytest.mark.parametrize('expr', SIGNALS)
def test_market_neutral_constant_book_truncated(book_data, expr):
    s = BookSettings(delay=1, universe='member', neutralize='market', truncation=0.05)
    w = build_book(alpha(expr), book_data, s)
    _check_book(w, 0.05, book_data, s)
    assert np.allclose(w.sum(axis=1), 0, atol=1e-12)


@pytest.mark.parametrize('expr', SIGNALS)
def test_group_neutral(book_data, expr):
    s = BookSettings(delay=1, universe='member', neutralize='sector', truncation=0.05)
    w = build_book(alpha(expr), book_data, s)
    _check_book(w, 0.05, book_data, s)
    sec = book_data['sector'].reindex_like(w)
    assert (w.where(sec.isna()).fillna(0) == 0).all().all()          # no group, no weight
    stacked = pd.DataFrame({'w': w.stack(), 'g': sec.stack()}).dropna()
    sums = stacked.groupby([stacked.index.get_level_values(0), 'g'])['w'].sum()
    assert np.allclose(sums, 0, atol=1e-12)


def test_no_neutralization_no_truncation(book_data):
    s = BookSettings(delay=1, neutralize=None, truncation=None)
    w = build_book(alpha("rank(returns(10))"), book_data, s)        # all positive: a long-only book
    _check_book(w, None, book_data, s)
    assert (w >= 0).all().all()


def test_truncation_binds(book_data):
    """The cubic signal concentrates weight; without the cap some names exceed 5%."""
    loose = build_book(alpha("power(returns(20), 3)"), book_data, BookSettings(truncation=None))
    assert loose.abs().max().max() > 0.05
    tight = build_book(alpha("power(returns(20), 3)"), book_data, BookSettings(truncation=0.05))
    assert tight.abs().max().max() <= 0.05 + 1e-9


def test_delay_one_ignores_same_day_data(book_data):
    """Changing every close on day t moves day t's weights at delay 0 but not at delay 1."""
    t = book_data['close'].index[150]
    t1 = book_data['close'].index[151]
    bumped = dict(book_data)
    c = book_data['close'].copy()
    c.loc[t] = c.loc[t] * np.exp(np.random.default_rng(3).normal(0, 0.05, c.shape[1]))
    bumped['close'] = c
    moved = lambda a, b, day: (a.loc[day] - b.loc[day]).abs().max() > 1e-6
    s1 = BookSettings(delay=1, universe='member')
    a, b = build_book(alpha("-returns(5)"), book_data, s1), build_book(alpha("-returns(5)"), bumped, s1)
    assert not moved(a, b, t) and moved(a, b, t1)                   # delay 1: day t trades on t-1's data
    s0 = BookSettings(delay=0, universe='member')
    a, b = build_book(alpha("-returns(5)"), book_data, s0), build_book(alpha("-returns(5)"), bumped, s0)
    assert moved(a, b, t)                                            # delay 0: day t trades on day t's data


def test_flat_or_empty_day_is_zero(book_data):
    data = dict(book_data)
    data['close'] = book_data['close'].copy()
    data['close'].iloc[100] = np.nan                                  # nothing tradable on day 100
    w = build_book(alpha("-returns(5)"), data, BookSettings(delay=0))
    assert (w.iloc[100] == 0).all()
    w = build_book(alpha("0 * close() + 1"), book_data, BookSettings())
    assert (w == 0).all().all()                                       # a flat signal has no book


def test_start_end_after_delay(book_data):
    s = BookSettings()
    full = build_book(alpha("-returns(5)"), book_data, s)
    part = build_book(alpha("-returns(5)"), book_data, s, start='2020-06-01', end='2020-09-30')
    pd.testing.assert_frame_equal(part, full.loc['2020-06-01':'2020-09-30'])


def test_out_of_rounds_keeps_neutral_and_gross(book_data):
    """With too few rounds for the cubic signal, neutrality and gross still hold exactly."""
    w = build_book(alpha("power(returns(20), 3)"), book_data, BookSettings(universe='member', rounds=2))
    gross = w.abs().sum(axis=1)
    assert np.allclose(gross[gross > 0], 1.0, atol=1e-12)
    assert np.allclose(w.sum(axis=1), 0, atol=1e-12)
    assert w.abs().max().max() > 0.05                                # the cap is what gives way


def test_settings_validation():
    with pytest.raises(ValueError):
        BookSettings(delay=-1)
    with pytest.raises(ValueError):
        BookSettings(truncation=0)
    with pytest.raises(ValueError):
        BookSettings(rounds=0)


# Reference: alpha_lab's eval.py (commit 211a577), which build_book generalizes. It stops
# when every day is within the cap (to 1e-9) and until then clips every day, so a day already
# within tolerance can be nudged again while others converge; build_book stops each day on
# its own. Both meet the same constraints and agree to the scale of that tolerance.

def _lab_to_book(raw, cap=0.05, rounds=1000):
    w = raw.to_numpy(dtype=float).copy()
    ok = np.isfinite(w)
    w[~ok] = 0.0
    n = ok.sum(axis=1, keepdims=True)
    for _ in range(rounds):
        w = np.where(ok, w - w.sum(axis=1, keepdims=True) / np.maximum(n, 1), 0.0)
        g = np.abs(w).sum(axis=1, keepdims=True)
        w = np.divide(w, g, out=np.zeros_like(w), where=g > 0)
        if np.abs(w).max() <= cap + 1e-9:
            break
        w = np.clip(w, -cap, cap)
    return pd.DataFrame(w, index=raw.index, columns=raw.columns)


@pytest.mark.parametrize('expr', SIGNALS)
def test_matches_alpha_lab_reference(book_data, expr):
    with compute_context():
        raw = delay(alpha(expr) + 0 * close() + 0 * field("member"), 1).evaluate(book_data)
    want = _lab_to_book(raw.where(book_data["close"].notna()).loc['2020-03-01':])
    got = build_book(alpha(expr), book_data, BookSettings(delay=1, universe='member', truncation=0.05),
                     start='2020-03-01')
    pd.testing.assert_frame_equal(got, want, check_exact=False, rtol=0, atol=1e-8)


# Metrics

def test_turnover_first_day_zero():
    w = pd.DataFrame([[0.5, -0.5], [0.25, -0.75], [0.25, -0.75]])
    pd.testing.assert_series_equal(turnover(w), pd.Series([0.0, 0.5, 0.0]))


def test_fitness_and_margin_by_hand():
    rng = np.random.default_rng(5)
    pnl = pd.Series(rng.normal(0.0004, 0.005, 500))
    to = pd.Series(np.full(500, 0.3))
    want = sharpe_ratio(pnl) * np.sqrt(abs(annualized_return(pnl)) / 0.3)
    assert np.isclose(fitness(pnl, to), want)
    slow = pd.Series(np.full(500, 0.05))                             # below the 0.125 floor
    assert np.isclose(fitness(pnl, slow), sharpe_ratio(pnl) * np.sqrt(abs(annualized_return(pnl)) / 0.125))
    assert np.isclose(margin_bps(pnl, to), pnl.sum() / to.sum() * 1e4)
    assert fitness(-pnl, to) < 0 < fitness(pnl, to)
    assert np.isnan(margin_bps(pnl, pd.Series(np.zeros(500))))


# Backward compatibility with 0.3.1

def _backtest_031(signal, data, holding_period=1, transaction_cost=0.0):
    """Backtest.run as of qex 0.3.1, the parts that produce the numbers."""
    with compute_context():
        weights = signal.to_weights(data)
    forward_returns = data["close"].pct_change(holding_period).shift(-holding_period)
    idx = weights.index.intersection(forward_returns.index)
    weights, forward_returns = weights.loc[idx], forward_returns.loc[idx]
    valid = weights.notna().any(axis=1) & forward_returns.notna().any(axis=1)
    weights, forward_returns = weights.loc[valid].fillna(0), forward_returns.loc[valid].fillna(0)
    pnl = (weights * forward_returns).sum(axis=1)
    to = weights.diff().abs().sum(axis=1)
    to.iloc[0] = weights.iloc[0].abs().sum()
    if transaction_cost > 0:
        pnl = pnl - to * transaction_cost
    returns = (pnl / weights.abs().sum(axis=1).replace(0, np.nan)).fillna(0)
    return weights, pnl, returns, to


@pytest.fixture
def backtest_data():
    """The fixture of test_evaluation.TestBacktest."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=252, freq='B')
    tickers = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    c = pd.DataFrame(np.exp(np.random.randn(252, 8).cumsum(axis=0) * 0.02 + 4), index=dates, columns=tickers)
    v = pd.DataFrame(np.random.lognormal(10, 0.5, (252, 8)), index=dates, columns=tickers)
    return {'close': c, 'volume': v}


@pytest.mark.parametrize('tc', [0.0, 0.001])
def test_default_backtest_unchanged(backtest_data, tc):
    signal = alpha("rank(returns(20)) - 0.5")
    weights, pnl, returns, to = _backtest_031(signal, backtest_data, transaction_cost=tc)
    r = Backtest(signal, transaction_cost=tc).run(backtest_data)
    pd.testing.assert_frame_equal(r.weights, weights, check_exact=True)
    pd.testing.assert_series_equal(r.pnl, pnl, check_exact=True)
    pd.testing.assert_series_equal(r.returns, returns, check_exact=True)
    assert r.sharpe == sharpe_ratio(returns)
    assert r.avg_turnover == to.mean()


def test_backtest_result_constructs_with_031_fields():
    s = pd.Series([0.0])
    r = BacktestResult(pnl=s, cumulative_pnl=s, weights=pd.DataFrame(), returns=s, total_return=0.0,
                       sharpe=0.0, annual_return=0.0, annual_volatility=0.0, max_drawdown=0.0, calmar=0.0,
                       sortino=0.0, return_gmv=0.0, drawdowns=[], avg_long_count=0.0, avg_short_count=0.0,
                       avg_turnover=0.0)
    assert r.fitness is None and r.margin_bps is None


def test_backtest_with_settings(book_data):
    r = Backtest(alpha("-returns(5)"), settings=BookSettings(delay=1, universe='member')).run(book_data)
    gross = r.weights.abs().sum(axis=1)
    assert np.allclose(gross[gross > 0], 1.0)
    assert r.weights.abs().max().max() <= 0.05 + 1e-9
    assert r.fitness is not None and r.margin_bps is not None
