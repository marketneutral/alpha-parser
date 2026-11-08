"""Backtest starts on the first day the book holds positions (0.12.2).

Before, the days before a signal had any value (a 60-day volatility's first 60 days) counted as days with a flat
book: the daily returns were right, but the summary diluted. Return on GMV divided total P&L by an average gross
that included those zero days, and the Sharpe ratio, annualized return and volatility averaged in zeros.
"""

import numpy as np
import pandas as pd
import pytest

from qex import qex
from evaluation import Backtest, BookSettings
from evaluation.metrics import sharpe_ratio, annualized_return


WARM = "scale(demean(rank(-returns(5) / ts_std(returns(1), 60))))"      # no value for its first 61 rows


def _active(signal, data):
    return qex(signal).evaluate(data).notna().any(axis=1)


def test_the_backtest_starts_when_the_book_does(sample_data):
    result = Backtest(qex(WARM)).run(sample_data)
    first_held = _active(WARM, sample_data).idxmax()
    assert result.returns.index[0] == first_held
    assert (result.weights.abs().sum(axis=1) > 0).iloc[0]


def test_summary_is_over_the_days_held(sample_data):
    result = Backtest(qex(WARM)).run(sample_data)
    r, pnl, w = result.returns, result.pnl, result.weights
    assert result.return_gmv == pytest.approx(pnl.sum() / w.abs().sum(axis=1).mean())
    assert result.return_gmv == pytest.approx(r.sum(), rel=1e-9)            # gross is 1 every day held
    assert result.sharpe == pytest.approx(sharpe_ratio(r))
    assert result.annual_return == pytest.approx(annualized_return(r))


def test_daily_returns_are_unchanged(sample_data):
    """Only the leading flat days go: every day held keeps the return it had."""
    signal = qex(WARM)
    result = Backtest(signal).run(sample_data)
    w = signal.to_weights(sample_data)
    from qex.primitives import forward_returns
    fwd = forward_returns(sample_data["close"], 1)
    expected = (w * fwd.fillna(0)).sum(axis=1).loc[result.returns.index]
    pd.testing.assert_series_equal(result.returns, expected, check_names=False, check_freq=False)


def test_a_flat_day_after_the_start_still_counts(sample_data):
    """A day the signal is flat once the book is running is a real flat day, and stays."""
    dates = sample_data["close"].index
    gap = dates[200]
    s = qex(WARM).evaluate(sample_data)
    s.loc[gap] = 0.0

    class Fixed:
        def to_weights(self, data):
            return s.fillna(0)

    result = Backtest(Fixed()).run(sample_data)
    assert gap in result.returns.index and result.returns.loc[gap] == 0


def test_the_settings_path_starts_when_the_book_does(sample_data):
    settings = BookSettings(delay=1)
    result = Backtest(qex(WARM), settings=settings).run(sample_data)
    assert (result.weights.abs().sum(axis=1) > 0).iloc[0]


def test_a_book_that_never_trades_does_not_fail(sample_data):
    result = Backtest(qex("returns(1) * 0")).run(sample_data)
    assert result.return_gmv == 0
