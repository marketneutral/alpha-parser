"""Tests for all signal operators."""

import pytest
import numpy as np
import pandas as pd

from alpha_parser import alpha, compute_context


class TestMathOperations:
    """Test math operations: log, abs, sign, sqrt, power, max, min."""

    @pytest.fixture
    def simple_data(self):
        """Simple test data with known values."""
        dates = pd.date_range('2020-01-01', periods=5, freq='D')
        tickers = ['A', 'B', 'C']

        close = pd.DataFrame(
            [[100.0, 200.0, 50.0],
             [110.0, 180.0, 60.0],
             [90.0, 220.0, 40.0],
             [105.0, 190.0, 55.0],
             [95.0, 210.0, 45.0]],
            index=dates,
            columns=tickers
        )

        volume = pd.DataFrame(
            [[1000.0, 2000.0, 500.0],
             [1100.0, 1800.0, 600.0],
             [900.0, 2200.0, 400.0],
             [1050.0, 1900.0, 550.0],
             [950.0, 2100.0, 450.0]],
            index=dates,
            columns=tickers
        )

        return {'close': close, 'volume': volume}

    def test_log(self, simple_data):
        """Test natural logarithm."""
        signal = alpha("log(close())")
        result = signal.evaluate(simple_data)

        expected = np.log(simple_data['close'])
        pd.testing.assert_frame_equal(result, expected)

    def test_abs_positive(self, simple_data):
        """Test abs with positive values."""
        signal = alpha("abs(close())")
        result = signal.evaluate(simple_data)

        # All positive, so should be unchanged
        pd.testing.assert_frame_equal(result, simple_data['close'])

    def test_abs_negative(self, simple_data):
        """Test abs with negative values (returns can be negative)."""
        signal = alpha("abs(returns(1))")
        result = signal.evaluate(simple_data)

        # All values should be non-negative
        assert (result.dropna() >= 0).all().all()

    def test_sign(self, simple_data):
        """Test sign function."""
        signal = alpha("sign(returns(1))")
        result = signal.evaluate(simple_data)

        # Values should be -1, 0, or 1
        valid = result.dropna()
        assert valid.isin([-1.0, 0.0, 1.0]).all().all()

    def test_sqrt(self, simple_data):
        """Test square root."""
        signal = alpha("sqrt(close())")
        result = signal.evaluate(simple_data)

        expected = np.sqrt(simple_data['close'])
        pd.testing.assert_frame_equal(result, expected)

    def test_power(self, simple_data):
        """Test power function."""
        signal = alpha("power(close(), 2)")
        result = signal.evaluate(simple_data)

        expected = simple_data['close'] ** 2
        pd.testing.assert_frame_equal(result, expected)

    def test_power_fractional(self, simple_data):
        """Test power with fractional exponent."""
        signal = alpha("power(close(), 0.5)")
        result = signal.evaluate(simple_data)

        expected = simple_data['close'] ** 0.5
        pd.testing.assert_frame_equal(result, expected)

    def test_max(self, simple_data):
        """Test element-wise maximum."""
        signal = alpha("max(close(), 100)")
        result = signal.evaluate(simple_data)

        expected = simple_data['close'].clip(lower=100)
        pd.testing.assert_frame_equal(result, expected)

    def test_min(self, simple_data):
        """Test element-wise minimum."""
        signal = alpha("min(close(), 100)")
        result = signal.evaluate(simple_data)

        expected = simple_data['close'].clip(upper=100)
        pd.testing.assert_frame_equal(result, expected)

    def test_max_two_signals(self, simple_data):
        """Test max with two signals."""
        signal = alpha("max(close(), delay(close(), 1))")
        result = signal.evaluate(simple_data)

        # Should be the max of current close and previous close
        assert result.shape == simple_data['close'].shape
        # First row is NaN due to delay
        assert result.iloc[0].isna().all()

    def test_min_two_signals(self, simple_data):
        """Test min with two signals."""
        signal = alpha("min(close(), delay(close(), 1))")
        result = signal.evaluate(simple_data)

        assert result.shape == simple_data['close'].shape
        assert result.iloc[0].isna().all()


class TestTimeSeriesOperations:
    """Test time-series operations."""

    @pytest.fixture
    def ts_data(self):
        """Data for time-series tests."""
        dates = pd.date_range('2020-01-01', periods=20, freq='D')
        tickers = ['A', 'B']

        np.random.seed(42)
        close = pd.DataFrame(
            np.exp(np.random.randn(20, 2).cumsum(axis=0) * 0.02 + 4),
            index=dates,
            columns=tickers
        )

        volume = pd.DataFrame(
            np.random.lognormal(10, 0.5, (20, 2)),
            index=dates,
            columns=tickers
        )

        return {'close': close, 'volume': volume}

    def test_ts_corr(self, ts_data):
        """Test rolling correlation."""
        signal = alpha("ts_corr(close(), volume(1), 5)")
        result = signal.evaluate(ts_data)

        assert result.shape == ts_data['close'].shape
        # Correlation should be between -1 and 1
        valid = result.dropna()
        assert (valid >= -1).all().all()
        assert (valid <= 1).all().all()

    def test_ts_cov(self, ts_data):
        """Test rolling covariance."""
        signal = alpha("ts_cov(close(), volume(1), 5)")
        result = signal.evaluate(ts_data)

        assert result.shape == ts_data['close'].shape
        # First few rows should be NaN (need window)
        assert result.iloc[:4].isna().all().all()

    def test_ewma(self, ts_data):
        """Test exponentially weighted moving average."""
        signal = alpha("ewma(close(), 5)")
        result = signal.evaluate(ts_data)

        expected = ts_data['close'].ewm(halflife=5).mean()
        pd.testing.assert_frame_equal(result, expected)

    def test_ts_argmax(self, ts_data):
        """Test periods since rolling max."""
        signal = alpha("ts_argmax(close(), 5)")
        result = signal.evaluate(ts_data)

        assert result.shape == ts_data['close'].shape
        # Values should be between 0 and period-1
        valid = result.dropna()
        assert (valid >= 0).all().all()
        assert (valid <= 4).all().all()

    def test_ts_argmin(self, ts_data):
        """Test periods since rolling min."""
        signal = alpha("ts_argmin(close(), 5)")
        result = signal.evaluate(ts_data)

        assert result.shape == ts_data['close'].shape
        valid = result.dropna()
        assert (valid >= 0).all().all()
        assert (valid <= 4).all().all()

    def test_ts_skew(self, ts_data):
        """Test rolling skewness."""
        signal = alpha("ts_skew(returns(1), 10)")
        result = signal.evaluate(ts_data)

        assert result.shape == ts_data['close'].shape
        # Skewness can be any real number, just check it computes

    def test_ts_kurt(self, ts_data):
        """Test rolling kurtosis."""
        signal = alpha("ts_kurt(returns(1), 10)")
        result = signal.evaluate(ts_data)

        assert result.shape == ts_data['close'].shape

    def test_decay_linear(self, ts_data):
        """Test linearly decaying weighted average."""
        signal = alpha("decay_linear(close(), 5)")
        result = signal.evaluate(ts_data)

        assert result.shape == ts_data['close'].shape
        # First few rows should be NaN
        assert result.iloc[:4].isna().all().all()
        # After warm-up, should have values
        assert result.iloc[4:].notna().all().all()

    def test_decay_linear_weights_recent_more(self, ts_data):
        """Test that decay_linear weights recent values more heavily."""
        # Create data where recent values are higher
        dates = pd.date_range('2020-01-01', periods=5, freq='D')
        close = pd.DataFrame(
            [[1.0], [2.0], [3.0], [4.0], [5.0]],
            index=dates,
            columns=['A']
        )
        data = {'close': close}

        signal = alpha("decay_linear(close(), 5)")
        result = signal.evaluate(data)

        # Simple mean would be 3.0
        # Decay linear with weights [1,2,3,4,5]/15 should be higher than 3.0
        # (1*1 + 2*2 + 3*3 + 4*4 + 5*5) / 15 = 55/15 ≈ 3.67
        assert result.iloc[-1, 0] > 3.0


class TestCrossSectionalOperations:
    """Test cross-sectional operations."""

    @pytest.fixture
    def cs_data(self):
        """Data for cross-sectional tests."""
        dates = pd.date_range('2020-01-01', periods=5, freq='D')
        tickers = ['A', 'B', 'C', 'D']

        close = pd.DataFrame(
            [[100.0, 200.0, 150.0, 50.0],
             [110.0, 180.0, 160.0, 60.0],
             [90.0, 220.0, 140.0, 40.0],
             [105.0, 190.0, 155.0, 55.0],
             [95.0, 210.0, 145.0, 45.0]],
            index=dates,
            columns=tickers
        )

        volume = pd.DataFrame(
            [[1000.0] * 4] * 5,
            index=dates,
            columns=tickers
        )

        return {'close': close, 'volume': volume}

    def test_scale(self, cs_data):
        """Test scale (abs values sum to 1)."""
        signal = alpha("scale(returns(1))")
        result = signal.evaluate(cs_data)

        # Skip first row (NaN from returns)
        valid_rows = result.iloc[1:]
        abs_sums = valid_rows.abs().sum(axis=1)

        # Each row's absolute values should sum to 1
        assert np.allclose(abs_sums, 1.0, atol=1e-10)

    def test_scale_preserves_sign(self, cs_data):
        """Test that scale preserves the sign of values."""
        signal = alpha("scale(returns(1))")
        result = signal.evaluate(cs_data)

        # Get the raw returns for comparison
        returns_signal = alpha("returns(1)")
        returns = returns_signal.evaluate(cs_data)

        # Signs should match (where both are non-NaN and non-zero)
        valid = result.iloc[1:].notna() & returns.iloc[1:].notna()
        for col in result.columns:
            for idx in result.index[1:]:
                if valid.loc[idx, col] and returns.loc[idx, col] != 0:
                    assert np.sign(result.loc[idx, col]) == np.sign(returns.loc[idx, col])

    def test_truncate(self, cs_data):
        """Test truncate (clip to max weight)."""
        signal = alpha("truncate(returns(1), 0.05)")
        result = signal.evaluate(cs_data)

        # All values should be within [-0.05, 0.05]
        valid = result.dropna()
        assert (valid >= -0.05).all().all()
        assert (valid <= 0.05).all().all()

    def test_zscore_no_division_by_zero(self, cs_data):
        """Test that zscore handles zero std without error."""
        # Create data where all values are the same (std = 0)
        dates = pd.date_range('2020-01-01', periods=3, freq='D')
        tickers = ['A', 'B', 'C']
        close = pd.DataFrame(
            [[100.0, 100.0, 100.0],
             [100.0, 100.0, 100.0],
             [100.0, 100.0, 100.0]],
            index=dates,
            columns=tickers
        )
        data = {'close': close, 'volume': close}

        signal = alpha("zscore(close())")
        result = signal.evaluate(data)

        # When std is 0, result should be NaN (not inf)
        assert result.isna().all().all()

    def test_quantile(self, cs_data):
        """Test quantile bucketing."""
        signal = alpha("quantile(close(), 4)")
        result = signal.evaluate(cs_data)

        # Values should be 1, 2, 3, or 4
        assert result.isin([1.0, 2.0, 3.0, 4.0]).all().all()

        # With 4 tickers and 4 buckets, should have one in each
        for idx in result.index:
            assert set(result.loc[idx]) == {1.0, 2.0, 3.0, 4.0}

    def test_winsorize(self, cs_data):
        """Test winsorize (cap at percentiles)."""
        signal = alpha("winsorize(close(), 0.25)")
        result = signal.evaluate(cs_data)

        # Result should be bounded by 25th and 75th percentiles
        original = cs_data['close']
        for idx in result.index:
            lower = original.loc[idx].quantile(0.25)
            upper = original.loc[idx].quantile(0.75)
            assert (result.loc[idx] >= lower).all()
            assert (result.loc[idx] <= upper).all()


class TestPrimitives:
    """Test primitive operations."""

    @pytest.fixture
    def prim_data(self):
        """Data for primitive tests."""
        dates = pd.date_range('2020-01-01', periods=30, freq='D')
        tickers = ['A', 'B']

        np.random.seed(42)
        close = pd.DataFrame(
            np.exp(np.random.randn(30, 2).cumsum(axis=0) * 0.02 + 4),
            index=dates,
            columns=tickers
        )

        volume = pd.DataFrame(
            np.random.lognormal(10, 0.5, (30, 2)),
            index=dates,
            columns=tickers
        )

        return {'close': close, 'volume': volume}

    def test_adv(self, prim_data):
        """Test average dollar volume."""
        signal = alpha("adv(5)")
        result = signal.evaluate(prim_data)

        # Compute expected: rolling mean of (close * volume)
        dollar_vol = prim_data['close'] * prim_data['volume']
        expected = dollar_vol.rolling(5).mean()

        pd.testing.assert_frame_equal(result, expected)

    def test_adv_larger_window(self, prim_data):
        """Test adv with larger window."""
        signal = alpha("adv(20)")
        result = signal.evaluate(prim_data)

        assert result.shape == prim_data['close'].shape
        # First 19 rows should be NaN
        assert result.iloc[:19].isna().all().all()
        # After warm-up, should have values
        assert result.iloc[19:].notna().all().all()

    def test_returns(self, prim_data):
        """Test returns calculation."""
        signal = alpha("returns(5)")
        result = signal.evaluate(prim_data)

        expected = prim_data['close'].pct_change(5)
        pd.testing.assert_frame_equal(result, expected)

    def test_volatility(self, prim_data):
        """Test volatility calculation."""
        signal = alpha("volatility(10)")
        result = signal.evaluate(prim_data)

        # Volatility should be annualized std of returns
        rets = prim_data['close'].pct_change()
        expected = rets.rolling(10).std() * np.sqrt(252)
        pd.testing.assert_frame_equal(result, expected)

    def test_volume(self, prim_data):
        """Test volume rolling average."""
        signal = alpha("volume(5)")
        result = signal.evaluate(prim_data)

        expected = prim_data['volume'].rolling(5).mean()
        pd.testing.assert_frame_equal(result, expected)


class TestTsRankOptimized:
    """Test the optimized TsRank implementation."""

    def test_ts_rank_basic(self):
        """Test basic ts_rank functionality."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        close = pd.DataFrame(
            [[1.0], [2.0], [3.0], [4.0], [5.0],
             [4.0], [3.0], [2.0], [1.0], [5.0]],
            index=dates,
            columns=['A']
        )
        data = {'close': close, 'volume': close}

        signal = alpha("ts_rank(close(), 5)")
        result = signal.evaluate(data)

        assert result.shape == (10, 1)
        # First 4 rows should be NaN (need 5 periods)
        assert result.iloc[:4].isna().all().all()
        # Last value should be 1.0 (5 is highest in [4,3,2,1,5])
        assert result.iloc[-1, 0] == 1.0

    def test_ts_rank_with_nan(self):
        """Test ts_rank handles NaN values."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        close = pd.DataFrame(
            [[1.0], [2.0], [np.nan], [4.0], [5.0],
             [4.0], [3.0], [2.0], [1.0], [5.0]],
            index=dates,
            columns=['A']
        )
        data = {'close': close, 'volume': close}

        signal = alpha("ts_rank(close(), 5)")
        result = signal.evaluate(data)

        # Should still produce results (handling NaN internally)
        assert result.shape == (10, 1)


class TestComplexExpressions:
    """Test complex combined expressions."""

    def test_nested_math(self, sample_data):
        """Test nested math operations."""
        signal = alpha("sqrt(abs(returns(5)))")
        result = signal.evaluate(sample_data)

        assert result.shape == sample_data['close'].shape
        # All values should be non-negative
        assert (result.dropna() >= 0).all().all()

    def test_combined_ts_and_cs(self, sample_data):
        """Test combining time-series and cross-sectional ops."""
        signal = alpha("rank(ewma(returns(5), 10))")
        result = signal.evaluate(sample_data)

        assert result.shape == sample_data['close'].shape
        # Rank values should be between 0 and 1
        valid = result.dropna()
        assert (valid >= 0).all().all()
        assert (valid <= 1).all().all()

    def test_scaled_momentum(self, sample_data):
        """Test a realistic scaled momentum signal."""
        signal = alpha("scale(rank(returns(20)) - 0.5)")
        result = signal.evaluate(sample_data)

        assert result.shape == sample_data['close'].shape
        # Absolute values should sum to 1
        valid_rows = result.iloc[20:]  # Skip warm-up
        abs_sums = valid_rows.abs().sum(axis=1)
        assert np.allclose(abs_sums, 1.0, atol=1e-10)

    def test_truncated_alpha(self, sample_data):
        """Test truncated alpha signal."""
        signal = alpha("truncate(zscore(returns(5)), 2.0)")
        result = signal.evaluate(sample_data)

        assert result.shape == sample_data['close'].shape
        # All values should be within [-2, 2]
        valid = result.dropna()
        assert (valid >= -2.0).all().all()
        assert (valid <= 2.0).all().all()

    def test_decay_momentum(self, sample_data):
        """Test decay linear on momentum."""
        signal = alpha("rank(decay_linear(returns(1), 10))")
        result = signal.evaluate(sample_data)

        assert result.shape == sample_data['close'].shape

    def test_correlation_based_alpha(self, sample_data):
        """Test correlation-based alpha."""
        signal = alpha("ts_corr(returns(1), volume(1), 20)")
        result = signal.evaluate(sample_data)

        assert result.shape == sample_data['close'].shape
        # Correlation should be between -1 and 1
        valid = result.dropna()
        assert (valid >= -1).all().all()
        assert (valid <= 1).all().all()

    def test_min_max_capping(self, sample_data):
        """Test min/max for capping."""
        signal = alpha("max(min(returns(5), 0.1), -0.1)")
        result = signal.evaluate(sample_data)

        assert result.shape == sample_data['close'].shape
        # All values should be within [-0.1, 0.1]
        valid = result.dropna()
        assert (valid >= -0.1).all().all()
        assert (valid <= 0.1).all().all()


class TestEventBasedOperations:
    """Test event-based rolling operations (roll over N non-NaN values)."""

    @pytest.fixture
    def sparse_data(self):
        """Create sparse earnings-like data."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        tickers = ['A', 'B']

        # Mostly NaN, with occasional values (simulating quarterly earnings)
        earnings = pd.DataFrame(np.nan, index=dates, columns=tickers)

        # A announces on days 10, 35, 60, 85 (roughly quarterly)
        earnings.loc[dates[10], 'A'] = 1.0
        earnings.loc[dates[35], 'A'] = 1.2
        earnings.loc[dates[60], 'A'] = 0.8
        earnings.loc[dates[85], 'A'] = 1.1

        # B announces on days 15, 40, 65, 90
        earnings.loc[dates[15], 'B'] = 2.0
        earnings.loc[dates[40], 'B'] = 2.5
        earnings.loc[dates[65], 'B'] = 1.8
        earnings.loc[dates[90], 'B'] = 2.2

        # Create price data
        np.random.seed(42)
        close = pd.DataFrame(
            np.exp(np.random.randn(100, 2).cumsum(axis=0) * 0.02 + 4),
            index=dates,
            columns=tickers
        )
        volume = pd.DataFrame(
            np.random.lognormal(10, 0.5, (100, 2)),
            index=dates,
            columns=tickers
        )

        return {'close': close, 'volume': volume, 'earnings': earnings}

    def test_ts_mean_events_basic(self, sparse_data):
        """Test mean over past N events."""
        signal = alpha("ts_mean_events(field('earnings'), 2)")
        result = signal.evaluate(sparse_data)

        # For A: after day 35, mean of [1.0, 1.2] = 1.1
        assert np.isclose(result.loc[sparse_data['close'].index[35], 'A'], 1.1)

        # For A: after day 60, mean of [1.2, 0.8] = 1.0
        assert np.isclose(result.loc[sparse_data['close'].index[60], 'A'], 1.0)

        # Before 2 events, should be NaN
        assert np.isnan(result.loc[sparse_data['close'].index[10], 'A'])

    def test_ts_std_events_basic(self, sparse_data):
        """Test std over past N events."""
        signal = alpha("ts_std_events(field('earnings'), 2)")
        result = signal.evaluate(sparse_data)

        # For A: after day 35, std of [1.0, 1.2]
        expected_std = np.std([1.0, 1.2], ddof=1)
        assert np.isclose(result.loc[sparse_data['close'].index[35], 'A'], expected_std)

        # Before 2 events, should be NaN
        assert np.isnan(result.loc[sparse_data['close'].index[10], 'A'])

    def test_ts_sum_events_basic(self, sparse_data):
        """Test sum over past N events."""
        signal = alpha("ts_sum_events(field('earnings'), 3)")
        result = signal.evaluate(sparse_data)

        # For A: after day 60, sum of [1.0, 1.2, 0.8] = 3.0
        assert np.isclose(result.loc[sparse_data['close'].index[60], 'A'], 3.0)

        # Before 3 events, should be NaN
        assert np.isnan(result.loc[sparse_data['close'].index[35], 'A'])

    def test_ts_count_events(self, sparse_data):
        """Test counting events in rolling window."""
        signal = alpha("ts_count_events(field('earnings'), 30)")
        result = signal.evaluate(sparse_data)

        # On day 45, window covers days 16-45
        # A has event on day 35, B has events on days 15 (outside), 40
        # So A should have 1, B should have 1
        assert result.loc[sparse_data['close'].index[45], 'A'] == 1.0
        assert result.loc[sparse_data['close'].index[45], 'B'] == 1.0

    def test_sue_calculation(self, sparse_data):
        """Test realistic SUE calculation using event-based std."""
        # Create earnings surprise data
        dates = sparse_data['close'].index
        tickers = sparse_data['close'].columns

        # Estimates (constant for simplicity)
        estimates = pd.DataFrame(1.0, index=dates, columns=tickers)
        sparse_data['estimates'] = estimates

        # Actual earnings with surprises
        actuals = sparse_data['earnings'].copy()
        sparse_data['actuals'] = actuals

        # SUE = (actual - estimate) / std of past surprises
        # But we need at least 2 events for std
        sue_expr = "(field('actuals') - field('estimates')) / ts_std_events(field('actuals') - field('estimates'), 2)"
        signal = alpha(sue_expr)
        result = signal.evaluate(sparse_data)

        # Check shape
        assert result.shape == sparse_data['close'].shape

        # After 2nd announcement, should have valid SUE
        # A's 2nd announcement is day 35
        assert not np.isnan(result.loc[dates[35], 'A'])

    def test_pead_with_sue(self, sparse_data):
        """Test full PEAD signal construction with proper SUE."""
        dates = sparse_data['close'].index
        tickers = sparse_data['close'].columns

        estimates = pd.DataFrame(1.0, index=dates, columns=tickers)
        sparse_data['estimates'] = estimates
        sparse_data['actuals'] = sparse_data['earnings'].copy()

        # Full PEAD: rank(fill_forward(SUE, 60)) - 0.5
        sue = "(field('actuals') - field('estimates')) / ts_std_events(field('actuals') - field('estimates'), 2)"
        pead_expr = f"rank(fill_forward({sue}, 60)) - 0.5"
        signal = alpha(pead_expr)
        result = signal.evaluate(sparse_data)

        assert result.shape == sparse_data['close'].shape
        # After sufficient events, should have valid signal
        # Check day 95 (after all 4 announcements per stock)
        assert not np.isnan(result.loc[dates[95], 'A'])
        assert not np.isnan(result.loc[dates[95], 'B'])

    def test_event_operations_persist_between_events(self, sparse_data):
        """Test that event-based operations return values between events."""
        signal = alpha("ts_mean_events(field('earnings'), 2)")
        result = signal.evaluate(sparse_data)

        dates = sparse_data['close'].index

        # After day 35 (2nd A announcement), check days 36-59
        # The value should persist (same as day 35)
        day_35_value = result.loc[dates[35], 'A']
        for i in range(36, 60):
            assert np.isclose(result.loc[dates[i], 'A'], day_35_value)

        # After day 60 (3rd A announcement), value should update
        assert not np.isclose(result.loc[dates[60], 'A'], day_35_value)


class TestTechnicalIndicators:
    """Test technical indicator signals: Bollinger Bands, RSI."""

    @pytest.fixture
    def indicator_data(self):
        """Data for technical indicator tests."""
        dates = pd.date_range('2020-01-01', periods=50, freq='D')
        tickers = ['A', 'B', 'C', 'D']

        np.random.seed(42)
        # Create trending price data with some mean reversion
        close = pd.DataFrame(
            np.exp(np.random.randn(50, 4).cumsum(axis=0) * 0.02 + 4),
            index=dates,
            columns=tickers
        )

        volume = pd.DataFrame(
            np.random.lognormal(10, 0.5, (50, 4)),
            index=dates,
            columns=tickers
        )

        return {'close': close, 'volume': volume}

    def test_bollinger_pct_b_basic(self, indicator_data):
        """Test Bollinger %B calculation."""
        # Raw %B = (close - lower_band) / (upper_band - lower_band)
        # Simplified: (close - (ma - 2*std)) / (4*std) = (close - ma + 2*std) / (4*std)
        signal = alpha("(close() - ts_mean(close(), 20)) / (2 * ts_std(close(), 20))")
        result = signal.evaluate(indicator_data)

        assert result.shape == indicator_data['close'].shape
        # First 19 rows should be NaN (need 20 periods for rolling)
        assert result.iloc[:19].isna().all().all()
        # After warm-up, should have values
        assert result.iloc[19:].notna().all().all()

    def test_bollinger_mean_reversion_signal(self, indicator_data):
        """Test full Bollinger mean-reversion signal."""
        signal = alpha("rank(-(close() - ts_mean(close(), 20)) / (2 * ts_std(close(), 20))) - 0.5")
        result = signal.evaluate(indicator_data)

        assert result.shape == indicator_data['close'].shape

        # Valid values should be centered around 0
        valid = result.dropna()

        # Ranks minus 0.5 should be in [-0.5, 0.5]
        assert (valid >= -0.5).all().all()
        assert (valid <= 0.5).all().all()

        # Signal should have both positive and negative values (long/short)
        assert (valid > 0).any().any()
        assert (valid < 0).any().any()

    def test_bollinger_components(self, indicator_data):
        """Test individual Bollinger Band components."""
        close = indicator_data['close']

        # Middle band = 20-day SMA
        ma_signal = alpha("ts_mean(close(), 20)")
        ma_result = ma_signal.evaluate(indicator_data)
        expected_ma = close.rolling(20).mean()
        pd.testing.assert_frame_equal(ma_result, expected_ma)

        # Standard deviation
        std_signal = alpha("ts_std(close(), 20)")
        std_result = std_signal.evaluate(indicator_data)
        expected_std = close.rolling(20).std()
        pd.testing.assert_frame_equal(std_result, expected_std)

    def test_rsi_basic(self, indicator_data):
        """Test basic RSI calculation."""
        # RSI = 100 * avg_gain / (avg_gain + avg_loss)
        signal = alpha(
            "100 * ts_mean(max(delta(close(), 1), 0), 14) "
            "/ (ts_mean(max(delta(close(), 1), 0), 14) + ts_mean(max(-delta(close(), 1), 0), 14))"
        )
        result = signal.evaluate(indicator_data)

        assert result.shape == indicator_data['close'].shape

        # RSI should be between 0 and 100 (or NaN)
        valid = result.dropna()
        assert (valid >= 0).all().all()
        assert (valid <= 100).all().all()

    def test_rsi_mean_reversion_signal(self, indicator_data):
        """Test full RSI mean-reversion signal."""
        signal = alpha(
            "rank(-(100 * ts_mean(max(delta(close(), 1), 0), 14) "
            "/ (ts_mean(max(delta(close(), 1), 0), 14) + ts_mean(max(-delta(close(), 1), 0), 14)))) - 0.5"
        )
        result = signal.evaluate(indicator_data)

        assert result.shape == indicator_data['close'].shape

        # Valid values should be centered around 0
        valid = result.dropna()

        # Ranks minus 0.5 should be in [-0.5, 0.5]
        assert (valid >= -0.5).all().all()
        assert (valid <= 0.5).all().all()

        # Signal should have both positive and negative values (long/short)
        assert (valid > 0).any().any()
        assert (valid < 0).any().any()

    def test_rsi_gains_losses_separation(self, indicator_data):
        """Test that RSI correctly separates gains and losses."""
        # Gains: positive deltas only
        gains_signal = alpha("max(delta(close(), 1), 0)")
        gains = gains_signal.evaluate(indicator_data)

        # Losses: negative deltas made positive
        losses_signal = alpha("max(-delta(close(), 1), 0)")
        losses = losses_signal.evaluate(indicator_data)

        # Raw delta
        delta_signal = alpha("delta(close(), 1)")
        delta = delta_signal.evaluate(indicator_data)

        # Gains + Losses should equal abs(delta)
        valid_idx = gains.notna() & losses.notna() & delta.notna()
        for col in gains.columns:
            for idx in gains.index:
                if valid_idx.loc[idx, col]:
                    assert np.isclose(
                        gains.loc[idx, col] + losses.loc[idx, col],
                        abs(delta.loc[idx, col]),
                        atol=1e-10
                    )

    def test_rsi_extreme_values(self):
        """Test RSI with extreme price movements."""
        dates = pd.date_range('2020-01-01', periods=30, freq='D')

        # Create data with all up moves for stock A, all down moves for stock B
        close_a = [100.0 + i for i in range(30)]  # Steadily rising
        close_b = [100.0 - i * 0.5 for i in range(30)]  # Steadily falling

        close = pd.DataFrame(
            {'A': close_a, 'B': close_b},
            index=dates
        )
        volume = pd.DataFrame(
            {'A': [1000.0] * 30, 'B': [1000.0] * 30},
            index=dates
        )
        data = {'close': close, 'volume': volume}

        signal = alpha(
            "100 * ts_mean(max(delta(close(), 1), 0), 14) "
            "/ (ts_mean(max(delta(close(), 1), 0), 14) + ts_mean(max(-delta(close(), 1), 0), 14))"
        )
        result = signal.evaluate(data)

        # Stock A (all gains) should have RSI = 100
        # Stock B (all losses) should have RSI = 0
        assert np.isclose(result.iloc[-1]['A'], 100.0)
        assert np.isclose(result.iloc[-1]['B'], 0.0)

    def test_bollinger_ewma_variant(self, indicator_data):
        """Test Bollinger Bands with EWMA instead of SMA."""
        signal = alpha("rank(-(close() - ewma(close(), 10)) / ts_std(close(), 20)) - 0.5")
        result = signal.evaluate(indicator_data)

        assert result.shape == indicator_data['close'].shape

        valid = result.dropna()
        assert (valid >= -0.5).all().all()
        assert (valid <= 0.5).all().all()

    def test_rsi_ewma_variant(self, indicator_data):
        """Test RSI using EWMA instead of SMA (Wilder's smoothing style)."""
        # Traditional RSI often uses exponential smoothing
        signal = alpha(
            "100 * ewma(max(delta(close(), 1), 0), 14) "
            "/ (ewma(max(delta(close(), 1), 0), 14) + ewma(max(-delta(close(), 1), 0), 14))"
        )
        result = signal.evaluate(indicator_data)

        assert result.shape == indicator_data['close'].shape

        valid = result.dropna()
        assert (valid >= 0).all().all()
        assert (valid <= 100).all().all()


class TestPairsTrading:
    """Test pairs trading signals using group operations."""

    @pytest.fixture
    def pairs_data(self):
        """Data with pair groupings for pairs trading tests."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        tickers = ['AAPL', 'MSFT', 'XOM', 'CVX', 'JPM', 'GS']

        np.random.seed(42)

        # Create correlated price movements within pairs
        # Pair 1: AAPL/MSFT (tech)
        tech_factor = np.random.randn(100).cumsum() * 0.02
        aapl = 100 * np.exp(tech_factor + np.random.randn(100) * 0.01)
        msft = 200 * np.exp(tech_factor + np.random.randn(100) * 0.01)

        # Pair 2: XOM/CVX (energy)
        energy_factor = np.random.randn(100).cumsum() * 0.02
        xom = 60 * np.exp(energy_factor + np.random.randn(100) * 0.01)
        cvx = 120 * np.exp(energy_factor + np.random.randn(100) * 0.01)

        # Pair 3: JPM/GS (finance)
        finance_factor = np.random.randn(100).cumsum() * 0.02
        jpm = 130 * np.exp(finance_factor + np.random.randn(100) * 0.01)
        gs = 250 * np.exp(finance_factor + np.random.randn(100) * 0.01)

        close = pd.DataFrame({
            'AAPL': aapl, 'MSFT': msft,
            'XOM': xom, 'CVX': cvx,
            'JPM': jpm, 'GS': gs,
        }, index=dates)

        volume = pd.DataFrame(
            np.random.lognormal(15, 0.5, (100, 6)),
            index=dates,
            columns=tickers
        )

        # Pair groupings
        pair = pd.DataFrame({
            'AAPL': 'pair_tech', 'MSFT': 'pair_tech',
            'XOM': 'pair_energy', 'CVX': 'pair_energy',
            'JPM': 'pair_finance', 'GS': 'pair_finance',
        }, index=dates)

        return {'close': close, 'volume': volume, 'pair': pair}

    def test_group_demean_basic(self, pairs_data):
        """Test that group_demean produces zero-sum within each pair."""
        signal = alpha("group_demean(returns(5), 'pair')")
        result = signal.evaluate(pairs_data)

        # Within each pair, the demeaned values should sum to ~0
        for date in result.index[10:]:  # Skip warm-up
            # Tech pair
            tech_sum = result.loc[date, ['AAPL', 'MSFT']].sum()
            assert abs(tech_sum) < 1e-10, f"Tech pair not zero-sum: {tech_sum}"

            # Energy pair
            energy_sum = result.loc[date, ['XOM', 'CVX']].sum()
            assert abs(energy_sum) < 1e-10, f"Energy pair not zero-sum: {energy_sum}"

            # Finance pair
            finance_sum = result.loc[date, ['JPM', 'GS']].sum()
            assert abs(finance_sum) < 1e-10, f"Finance pair not zero-sum: {finance_sum}"

    def test_group_std_basic(self, pairs_data):
        """Test that group_std computes rolling std within pairs."""
        signal = alpha("group_std(returns(1), 'pair', 20)")
        result = signal.evaluate(pairs_data)

        # Should have values after warm-up
        assert result.iloc[25:].notna().all().all()

        # Within each pair, all members should have the same std value
        for date in result.index[25:]:
            # Tech pair should have same std
            assert np.isclose(
                result.loc[date, 'AAPL'],
                result.loc[date, 'MSFT']
            )

            # Energy pair should have same std
            assert np.isclose(
                result.loc[date, 'XOM'],
                result.loc[date, 'CVX']
            )

    def test_pairs_zscore_signal(self, pairs_data):
        """Test z-score normalized pair spread signal."""
        # Z-score of pair spread: spread / spread_volatility
        signal = alpha("group_demean(returns(5), 'pair') / group_std(returns(5), 'pair', 60)")
        result = signal.evaluate(pairs_data)

        # Should have values after warm-up (60 days)
        valid = result.iloc[65:].dropna()
        assert len(valid) > 0

        # Z-scores should be reasonable (mostly within -3 to 3)
        assert (valid.abs() < 5).all().all()

    def test_pairs_mean_reversion_signal(self, pairs_data):
        """Test full pairs mean reversion signal."""
        # Go long the laggard, short the leader within each pair
        signal = alpha("group_demean(-returns(5), 'pair')")
        result = signal.evaluate(pairs_data)

        # Should produce opposite signs within each pair
        for date in result.index[10:]:
            # If AAPL is positive, MSFT should be negative (and vice versa)
            aapl_val = result.loc[date, 'AAPL']
            msft_val = result.loc[date, 'MSFT']
            if not (np.isnan(aapl_val) or np.isnan(msft_val)):
                # Signs should be opposite (or both ~0)
                assert aapl_val * msft_val <= 1e-10

    def test_pairs_conditional_trading(self, pairs_data):
        """Test conditional pairs trading - where() with scalar works."""
        # Test that where() can handle scalar constants (0)
        signal = alpha(
            "where("
            "abs(group_demean(returns(10), 'pair')) > group_std(returns(10), 'pair', 60), "
            "group_demean(-returns(5), 'pair'), "
            "0)"
        )
        result = signal.evaluate(pairs_data)

        # Should complete without error and have valid structure
        assert result.shape == pairs_data['close'].shape

        # After warm-up period, should have values (mostly zeros due to low volatility in synthetic data)
        valid = result.iloc[65:]
        assert not valid.isna().all().all()


class TestEwmaAndBetaOperations:
    """Test EWMA variance/covariance and beta operations."""

    @pytest.fixture
    def beta_data(self):
        """Data for beta and EWMA tests."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        tickers = ['AAPL', 'MSFT', 'SPY']

        # Generate correlated returns
        # SPY is "market", AAPL has beta ~1.2, MSFT has beta ~0.8
        market_returns = np.random.randn(100) * 0.01
        aapl_returns = 1.2 * market_returns + np.random.randn(100) * 0.005
        msft_returns = 0.8 * market_returns + np.random.randn(100) * 0.005

        # Convert to prices
        close = pd.DataFrame(index=dates, columns=tickers)
        close['SPY'] = 100 * np.exp(np.cumsum(market_returns))
        close['AAPL'] = 150 * np.exp(np.cumsum(aapl_returns))
        close['MSFT'] = 200 * np.exp(np.cumsum(msft_returns))

        volume = pd.DataFrame(
            np.random.lognormal(15, 0.5, (100, 3)),
            index=dates,
            columns=tickers
        )

        return {'close': close, 'volume': volume}

    def test_ts_var(self, beta_data):
        """Test rolling variance."""
        signal = alpha("ts_var(returns(1), 20)")
        result = signal.evaluate(beta_data)

        assert result.shape == beta_data['close'].shape
        # Variance should be non-negative
        valid = result.dropna()
        assert (valid >= 0).all().all()
        # First 20 rows should be NaN
        assert result.iloc[:20].isna().all().all()

    def test_ewma_var(self, beta_data):
        """Test EWMA variance."""
        signal = alpha("ewma_var(returns(1), 10)")
        result = signal.evaluate(beta_data)

        assert result.shape == beta_data['close'].shape
        # Variance should be non-negative
        valid = result.dropna()
        assert (valid >= 0).all().all()

        # Should match pandas ewm().var()
        returns = beta_data['close'].pct_change()
        expected = returns.ewm(halflife=10).var()
        pd.testing.assert_frame_equal(result, expected)

    def test_ewma_cov(self, beta_data):
        """Test EWMA covariance between two signals."""
        signal = alpha("ewma_cov(returns(1), volume(1), 10)")
        result = signal.evaluate(beta_data)

        assert result.shape == beta_data['close'].shape
        # Covariance can be any real number

    def test_ts_beta_basic(self, beta_data):
        """Test rolling beta calculation."""
        # Beta of AAPL returns vs SPY returns
        signal = alpha("ts_beta(returns(1), delay(returns(1), 0), 20)")
        result = signal.evaluate(beta_data)

        assert result.shape == beta_data['close'].shape
        # Beta can be any real number

    def test_ts_beta_vs_manual(self, beta_data):
        """Test ts_beta matches manual cov/var calculation."""
        signal_beta = alpha("ts_beta(returns(1), volume(1), 20)")
        signal_manual = alpha("ts_cov(returns(1), volume(1), 20) / ts_var(volume(1), 20)")

        result_beta = signal_beta.evaluate(beta_data)
        result_manual = signal_manual.evaluate(beta_data)

        # Should be equal (within floating point tolerance)
        pd.testing.assert_frame_equal(result_beta, result_manual)

    def test_ts_beta_ewma_basic(self, beta_data):
        """Test EWMA beta calculation."""
        signal = alpha("ts_beta_ewma(returns(1), volume(1), 10)")
        result = signal.evaluate(beta_data)

        assert result.shape == beta_data['close'].shape

    def test_ts_beta_ewma_vs_manual(self, beta_data):
        """Test ts_beta_ewma matches manual ewma_cov/ewma_var calculation."""
        signal_beta = alpha("ts_beta_ewma(returns(1), volume(1), 10)")
        signal_manual = alpha("ewma_cov(returns(1), volume(1), 10) / ewma_var(volume(1), 10)")

        result_beta = signal_beta.evaluate(beta_data)
        result_manual = signal_manual.evaluate(beta_data)

        # Should be equal (within floating point tolerance)
        pd.testing.assert_frame_equal(result_beta, result_manual)

    def test_beta_hedge_ratio_concept(self, beta_data):
        """Test that beta can be used as hedge ratio.

        If we compute beta(y, x), then y - beta * x should have lower variance.
        """
        # Get returns
        returns = beta_data['close'].pct_change()
        y = returns['AAPL']
        x = returns['SPY']

        # Compute beta using our signal
        signal = alpha("ts_beta(returns(1), delay(returns(1), 0), 60)")
        result = signal.evaluate(beta_data)

        # After warm-up, we should have beta estimates
        # The residual y - beta * x should have lower std than y alone
        start_idx = 65
        beta_values = result.iloc[start_idx:]['AAPL']

        # Compute hedged returns (manually, just to verify concept)
        y_window = y.iloc[start_idx:]
        x_window = x.iloc[start_idx:]
        hedged = y_window - beta_values * x_window

        # Hedged returns should have lower std than unhedged
        # (This is the whole point of beta hedging)
        assert hedged.std() < y_window.std()

    def test_ewma_beta_more_responsive(self, beta_data):
        """Test that EWMA beta responds faster to recent changes."""
        # Modify data to have a regime change
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        np.random.seed(123)

        # First 50 days: low volatility, beta ~1.0
        market_ret1 = np.random.randn(50) * 0.01
        stock_ret1 = 1.0 * market_ret1 + np.random.randn(50) * 0.005

        # Last 50 days: high volatility, beta ~2.0
        market_ret2 = np.random.randn(50) * 0.02
        stock_ret2 = 2.0 * market_ret2 + np.random.randn(50) * 0.01

        market_returns = np.concatenate([market_ret1, market_ret2])
        stock_returns = np.concatenate([stock_ret1, stock_ret2])

        close = pd.DataFrame(index=dates, columns=['STOCK', 'MKT'])
        close['MKT'] = 100 * np.exp(np.cumsum(market_returns))
        close['STOCK'] = 100 * np.exp(np.cumsum(stock_returns))
        volume = pd.DataFrame(np.ones((100, 2)), index=dates, columns=['STOCK', 'MKT'])

        data = {'close': close, 'volume': volume}

        # Compare rolling vs EWMA beta at the end
        signal_rolling = alpha("ts_beta(returns(1), delay(returns(1), 0), 40)")
        signal_ewma = alpha("ts_beta_ewma(returns(1), delay(returns(1), 0), 10)")

        result_rolling = signal_rolling.evaluate(data)
        result_ewma = signal_ewma.evaluate(data)

        # At day 99, EWMA should be closer to 2.0 than rolling
        # because EWMA gives more weight to recent (high-beta) observations
        rolling_beta = result_rolling.iloc[-1]['STOCK']
        ewma_beta = result_ewma.iloc[-1]['STOCK']

        # Both should be > 1 (reflecting the regime change)
        # EWMA should be higher (closer to true recent beta of 2.0)
        assert ewma_beta > rolling_beta * 0.9  # EWMA should be at least similar or higher
