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
