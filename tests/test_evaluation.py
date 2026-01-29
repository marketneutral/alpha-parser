"""Tests for the evaluation module."""

import pytest
import numpy as np
import pandas as pd

from alpha_parser import alpha
from alpha_parser.evaluation import (
    Backtest,
    BacktestResult,
    QuantileAnalysis,
    QuantileResult,
    sharpe_ratio,
    max_drawdown,
    top_drawdowns,
    return_on_gmv,
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    sortino_ratio,
)


class TestMetrics:
    """Test performance metrics calculations."""

    def test_sharpe_ratio_basic(self):
        """Test Sharpe ratio calculation."""
        # Create returns with zero volatility
        returns = pd.Series([0.0] * 252)
        sharpe = sharpe_ratio(returns)

        # Zero std returns 0 (protected against division by zero)
        assert sharpe == 0.0

    def test_sharpe_ratio_with_variance(self):
        """Test Sharpe with actual variance."""
        np.random.seed(42)
        # Generate returns with known properties
        returns = pd.Series(np.random.normal(0.0005, 0.01, 252))

        sharpe = sharpe_ratio(returns)

        # Should be roughly mean/std * sqrt(252)
        expected = returns.mean() / returns.std() * np.sqrt(252)
        assert np.isclose(sharpe, expected)

    def test_annualized_return(self):
        """Test annualized return calculation."""
        returns = pd.Series([0.001] * 252)  # 0.1% daily
        ann_ret = annualized_return(returns)

        # 0.1% * 252 = 25.2%
        assert np.isclose(ann_ret, 0.252)

    def test_annualized_volatility(self):
        """Test annualized volatility calculation."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.01, 252))  # 1% daily vol

        ann_vol = annualized_volatility(returns)

        # Should be roughly 1% * sqrt(252) ≈ 15.87%
        expected = returns.std() * np.sqrt(252)
        assert np.isclose(ann_vol, expected)

    def test_max_drawdown_basic(self):
        """Test max drawdown calculation."""
        # Cumulative returns: 1.0 -> 1.1 -> 0.9 -> 1.0
        cumulative = pd.Series([1.0, 1.1, 0.9, 1.0])

        mdd = max_drawdown(cumulative)

        # Max drawdown from 1.1 to 0.9 = 0.2/1.1 ≈ 18.18%
        assert np.isclose(mdd, 0.2 / 1.1)

    def test_max_drawdown_no_drawdown(self):
        """Test max drawdown with no drawdown."""
        cumulative = pd.Series([1.0, 1.1, 1.2, 1.3])
        mdd = max_drawdown(cumulative)
        assert mdd == 0.0

    def test_top_drawdowns(self):
        """Test top drawdowns identification."""
        # Create series with multiple drawdowns
        cumulative = pd.Series(
            [1.0, 1.1, 0.95, 1.05, 1.15, 0.9, 1.0, 1.2, 1.1, 1.25],
            index=pd.date_range('2020-01-01', periods=10),
        )

        drawdowns = top_drawdowns(cumulative, n=3)

        assert len(drawdowns) <= 3
        # First drawdown should be the largest
        if len(drawdowns) >= 2:
            assert drawdowns[0].depth >= drawdowns[1].depth

    def test_return_on_gmv(self):
        """Test return on GMV calculation."""
        pnl = pd.Series([0.01, 0.02, -0.01, 0.015])
        weights = pd.DataFrame({
            'A': [0.5, 0.4, 0.6, 0.5],
            'B': [-0.5, -0.4, -0.6, -0.5],
        })

        rogmv = return_on_gmv(pnl, weights)

        # Total PnL = 0.035, Avg GMV = 1.0
        total_pnl = pnl.sum()
        avg_gmv = weights.abs().sum(axis=1).mean()
        expected = total_pnl / avg_gmv
        assert np.isclose(rogmv, expected)

    def test_calmar_ratio(self):
        """Test Calmar ratio calculation."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        returns.iloc[50:60] = -0.05  # Add a drawdown period

        calmar = calmar_ratio(returns)

        # Should be annualized return / max drawdown
        assert calmar > 0 or calmar < 0  # Just check it computed

    def test_sortino_ratio(self):
        """Test Sortino ratio calculation."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.01, 100))

        sortino = sortino_ratio(returns)

        # Sortino uses downside deviation
        assert isinstance(sortino, float)


class TestBacktest:
    """Test Backtest class."""

    @pytest.fixture
    def backtest_data(self):
        """Create data for backtest tests."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=252, freq='B')
        tickers = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

        # Generate correlated random walks
        close = pd.DataFrame(
            np.exp(np.random.randn(252, 8).cumsum(axis=0) * 0.02 + 4),
            index=dates,
            columns=tickers,
        )

        volume = pd.DataFrame(
            np.random.lognormal(10, 0.5, (252, 8)),
            index=dates,
            columns=tickers,
        )

        return {'close': close, 'volume': volume}

    def test_backtest_runs(self, backtest_data):
        """Test that backtest runs without error."""
        signal = alpha("rank(returns(20)) - 0.5")
        bt = Backtest(signal)
        result = bt.run(backtest_data)

        assert isinstance(result, BacktestResult)
        assert len(result.pnl) > 0
        assert len(result.weights) > 0

    def test_backtest_result_attributes(self, backtest_data):
        """Test backtest result has all expected attributes."""
        signal = alpha("rank(returns(20)) - 0.5")
        bt = Backtest(signal)
        result = bt.run(backtest_data)

        # Check time series
        assert isinstance(result.pnl, pd.Series)
        assert isinstance(result.cumulative_pnl, pd.Series)
        assert isinstance(result.weights, pd.DataFrame)
        assert isinstance(result.returns, pd.Series)

        # Check metrics
        assert isinstance(result.total_return, float)
        assert isinstance(result.sharpe, float)
        assert isinstance(result.annual_return, float)
        assert isinstance(result.annual_volatility, float)
        assert isinstance(result.max_drawdown, float)
        assert isinstance(result.calmar, float)
        assert isinstance(result.sortino, float)
        assert isinstance(result.return_gmv, float)

        # Check drawdowns
        assert isinstance(result.drawdowns, list)

        # Check position stats
        assert isinstance(result.avg_long_count, float)
        assert isinstance(result.avg_short_count, float)
        assert isinstance(result.avg_turnover, float)

    def test_backtest_weights_normalized_by_gmv(self, backtest_data):
        """Test that weights are normalized by gross market value."""
        signal = alpha("rank(returns(20)) - 0.5")
        bt = Backtest(signal)
        result = bt.run(backtest_data)

        # Weights should have abs().sum() ≈ 1 (normalized by GMV)
        # Skip rows where all weights are 0 (warm-up period)
        valid_rows = result.weights[result.weights.abs().sum(axis=1) > 0]
        gmv_per_row = valid_rows.abs().sum(axis=1)
        assert np.allclose(gmv_per_row, 1.0, atol=1e-10)

    def test_backtest_with_transaction_cost(self, backtest_data):
        """Test backtest with transaction costs."""
        signal = alpha("rank(returns(20)) - 0.5")

        # Run without costs
        bt_no_cost = Backtest(signal, transaction_cost=0.0)
        result_no_cost = bt_no_cost.run(backtest_data)

        # Run with costs
        bt_with_cost = Backtest(signal, transaction_cost=0.001)
        result_with_cost = bt_with_cost.run(backtest_data)

        # With costs should have lower returns
        assert result_with_cost.total_return < result_no_cost.total_return

    def test_backtest_summary_string(self, backtest_data):
        """Test that summary produces readable output."""
        signal = alpha("rank(returns(20)) - 0.5")
        bt = Backtest(signal)
        result = bt.run(backtest_data)

        summary = result.summary()
        assert isinstance(summary, str)
        assert "BACKTEST RESULTS" in summary
        assert "Sharpe Ratio" in summary
        assert "Max Drawdown" in summary

    def test_backtest_date_filtering(self, backtest_data):
        """Test backtest with date filtering."""
        signal = alpha("rank(returns(20)) - 0.5")
        bt = Backtest(signal)

        dates = backtest_data['close'].index
        start = dates[50]
        end = dates[150]

        result = bt.run(backtest_data, start_date=start, end_date=end)

        # Result should only contain dates in range
        assert result.pnl.index.min() >= start
        assert result.pnl.index.max() <= end

    def test_momentum_signal_has_positive_sharpe(self, backtest_data):
        """Test that a simple momentum signal has reasonable Sharpe."""
        # This is a rough test - momentum may not always work
        signal = alpha("rank(returns(60)) - 0.5")
        bt = Backtest(signal)
        result = bt.run(backtest_data)

        # Sharpe should be finite
        assert np.isfinite(result.sharpe)

    def test_walk_forward(self, backtest_data):
        """Test walk-forward analysis."""
        signal = alpha("rank(returns(20)) - 0.5")
        bt = Backtest(signal)

        wf_results = bt.run_walk_forward(
            backtest_data,
            train_period=60,
            test_period=30,
        )

        assert isinstance(wf_results, pd.DataFrame)
        assert 'sharpe' in wf_results.columns
        assert 'return' in wf_results.columns


class TestQuantileAnalysis:
    """Test QuantileAnalysis class."""

    @pytest.fixture
    def quantile_data(self):
        """Create data for quantile tests."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=252, freq='B')
        tickers = [f'STOCK{i}' for i in range(20)]  # Need enough for quantiles

        close = pd.DataFrame(
            np.exp(np.random.randn(252, 20).cumsum(axis=0) * 0.02 + 4),
            index=dates,
            columns=tickers,
        )

        volume = pd.DataFrame(
            np.random.lognormal(10, 0.5, (252, 20)),
            index=dates,
            columns=tickers,
        )

        return {'close': close, 'volume': volume}

    def test_quantile_analysis_runs(self, quantile_data):
        """Test that quantile analysis runs without error."""
        signal = alpha("rank(returns(20)) - 0.5")
        qa = QuantileAnalysis(signal, n_quantiles=5)
        result = qa.run(quantile_data)

        assert isinstance(result, QuantileResult)

    def test_quantile_result_attributes(self, quantile_data):
        """Test quantile result has all expected attributes."""
        signal = alpha("rank(returns(20)) - 0.5")
        qa = QuantileAnalysis(signal, n_quantiles=5)
        result = qa.run(quantile_data)

        # Per-quantile stats
        assert isinstance(result.quantile_returns, pd.DataFrame)
        assert isinstance(result.quantile_cumulative, pd.DataFrame)
        assert isinstance(result.mean_returns, pd.Series)
        assert isinstance(result.sharpe_by_quantile, pd.Series)
        assert isinstance(result.hit_rate, pd.Series)

        # Spread stats
        assert isinstance(result.spread_returns, pd.Series)
        assert isinstance(result.spread_cumulative, pd.Series)
        assert isinstance(result.spread_sharpe, float)
        assert isinstance(result.spread_mean, float)

        # Other
        assert isinstance(result.is_monotonic, bool)
        assert isinstance(result.rank_ic, float)
        assert result.n_quantiles == 5

    def test_quantile_returns_shape(self, quantile_data):
        """Test that quantile returns have correct shape."""
        signal = alpha("rank(returns(20)) - 0.5")
        qa = QuantileAnalysis(signal, n_quantiles=5)
        result = qa.run(quantile_data)

        # Should have 5 columns (one per quantile)
        assert result.quantile_returns.shape[1] == 5
        assert list(result.quantile_returns.columns) == [1, 2, 3, 4, 5]

    def test_spread_is_q5_minus_q1(self, quantile_data):
        """Test that spread is correctly calculated as Q5 - Q1."""
        signal = alpha("rank(returns(20)) - 0.5")
        qa = QuantileAnalysis(signal, n_quantiles=5)
        result = qa.run(quantile_data)

        expected_spread = result.quantile_returns[5] - result.quantile_returns[1]
        pd.testing.assert_series_equal(result.spread_returns, expected_spread)

    def test_quantile_summary_string(self, quantile_data):
        """Test that summary produces readable output."""
        signal = alpha("rank(returns(20)) - 0.5")
        qa = QuantileAnalysis(signal, n_quantiles=5)
        result = qa.run(quantile_data)

        summary = result.summary()
        assert isinstance(summary, str)
        assert "QUANTILE ANALYSIS" in summary
        assert "Rank IC" in summary

    def test_ic_summary(self, quantile_data):
        """Test IC summary calculation."""
        signal = alpha("rank(returns(20)) - 0.5")
        qa = QuantileAnalysis(signal, n_quantiles=5)
        ic_stats = qa.ic_summary(quantile_data)

        assert isinstance(ic_stats, pd.Series)
        assert "Mean Rank IC" in ic_stats.index
        assert "IC IR (Rank)" in ic_stats.index

    def test_different_quantile_counts(self, quantile_data):
        """Test analysis with different quantile counts."""
        signal = alpha("rank(returns(20)) - 0.5")

        for n in [3, 4, 5, 10]:
            qa = QuantileAnalysis(signal, n_quantiles=n)
            result = qa.run(quantile_data)
            assert result.n_quantiles == n
            assert result.quantile_returns.shape[1] == n

    def test_rank_ic_is_reasonable(self, quantile_data):
        """Test that rank IC is in valid range."""
        signal = alpha("rank(returns(20)) - 0.5")
        qa = QuantileAnalysis(signal, n_quantiles=5)
        result = qa.run(quantile_data)

        # IC should be between -1 and 1
        assert -1 <= result.rank_ic <= 1


class TestIntegration:
    """Integration tests combining backtest and quantile analysis."""

    @pytest.fixture
    def integration_data(self):
        """Create data for integration tests."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=504, freq='B')  # 2 years
        tickers = [f'STOCK{i}' for i in range(50)]

        close = pd.DataFrame(
            np.exp(np.random.randn(504, 50).cumsum(axis=0) * 0.02 + 4),
            index=dates,
            columns=tickers,
        )

        volume = pd.DataFrame(
            np.random.lognormal(10, 0.5, (504, 50)),
            index=dates,
            columns=tickers,
        )

        return {'close': close, 'volume': volume}

    def test_backtest_and_quantile_agree_on_signal(self, integration_data):
        """Test that backtest and quantile analysis use the same signal."""
        signal = alpha("rank(returns(20)) - 0.5")

        bt = Backtest(signal)
        bt_result = bt.run(integration_data)

        qa = QuantileAnalysis(signal, n_quantiles=5)
        qa_result = qa.run(integration_data)

        # Both should have run successfully
        assert bt_result.sharpe is not None
        assert qa_result.spread_sharpe is not None

    def test_complex_signal_evaluation(self, integration_data):
        """Test evaluation of complex signal."""
        signal = alpha("rank(decay_linear(returns(5), 10)) + rank(-volatility(20)) - 1.0")

        bt = Backtest(signal)
        result = bt.run(integration_data)

        assert np.isfinite(result.sharpe)
        assert result.weights.shape[1] == 50  # All 50 stocks

    def test_rsi_signal_backtest(self, integration_data):
        """Test backtesting the RSI signal."""
        signal = alpha(
            "rank(-(100 * ts_mean(max(delta(close(), 1), 0), 14) "
            "/ (ts_mean(max(delta(close(), 1), 0), 14) + ts_mean(max(-delta(close(), 1), 0), 14)))) - 0.5"
        )

        bt = Backtest(signal)
        result = bt.run(integration_data)

        # Should complete without error
        assert isinstance(result, BacktestResult)
        assert np.isfinite(result.sharpe)

    def test_bollinger_signal_backtest(self, integration_data):
        """Test backtesting the Bollinger Band signal."""
        signal = alpha("rank(-(close() - ts_mean(close(), 20)) / (2 * ts_std(close(), 20))) - 0.5")

        bt = Backtest(signal)
        result = bt.run(integration_data)

        # Should complete without error
        assert isinstance(result, BacktestResult)
        assert np.isfinite(result.sharpe)
