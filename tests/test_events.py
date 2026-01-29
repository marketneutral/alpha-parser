"""Tests for event-based operations (sparse data handling)."""

import pytest
import numpy as np
import pandas as pd

from alpha_parser import alpha, compute_context


class TestIsValid:
    """Test is_valid operation."""

    def test_is_valid_detects_nan(self):
        """Test that is_valid returns 1 for non-NaN, 0 for NaN."""
        dates = pd.date_range('2020-01-01', periods=5, freq='D')
        tickers = ['A', 'B', 'C']

        # Sparse data - only some values present
        sparse_data = pd.DataFrame(
            [[1.0, np.nan, np.nan],
             [np.nan, 2.0, np.nan],
             [np.nan, np.nan, 3.0],
             [4.0, np.nan, np.nan],
             [np.nan, 5.0, np.nan]],
            index=dates,
            columns=tickers
        )

        data = {'sparse': sparse_data}

        signal = alpha("is_valid(field('sparse'))")
        result = signal.evaluate(data)

        # Check shape
        assert result.shape == (5, 3)

        # Check values - should be 1.0 where not NaN, 0.0 where NaN
        expected = pd.DataFrame(
            [[1.0, 0.0, 0.0],
             [0.0, 1.0, 0.0],
             [0.0, 0.0, 1.0],
             [1.0, 0.0, 0.0],
             [0.0, 1.0, 0.0]],
            index=dates,
            columns=tickers
        )
        pd.testing.assert_frame_equal(result, expected)


class TestFillForward:
    """Test fill_forward operation."""

    def test_fill_forward_basic(self):
        """Test basic forward fill with limit."""
        dates = pd.date_range('2020-01-01', periods=7, freq='D')
        tickers = ['A', 'B']

        # Sparse data
        sparse_data = pd.DataFrame(
            [[1.0, np.nan],
             [np.nan, np.nan],
             [np.nan, 2.0],
             [np.nan, np.nan],
             [np.nan, np.nan],
             [np.nan, np.nan],
             [3.0, np.nan]],
            index=dates,
            columns=tickers
        )

        data = {'sparse': sparse_data}

        # Fill forward with limit of 2
        signal = alpha("fill_forward(field('sparse'), 2)")
        result = signal.evaluate(data)

        # A: 1.0 should fill forward for 2 days, then NaN
        # B: 2.0 should fill forward for 2 days, then NaN
        expected = pd.DataFrame(
            [[1.0, np.nan],
             [1.0, np.nan],   # filled from A
             [1.0, 2.0],      # filled from A, new value B
             [np.nan, 2.0],   # A expired, B filled
             [np.nan, 2.0],   # A expired, B filled
             [np.nan, np.nan], # both expired
             [3.0, np.nan]],   # new A value
            index=dates,
            columns=tickers
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_fill_forward_longer_limit(self):
        """Test that fill forward respects limit."""
        dates = pd.date_range('2020-01-01', periods=5, freq='D')
        tickers = ['A']

        sparse_data = pd.DataFrame(
            [[1.0], [np.nan], [np.nan], [np.nan], [np.nan]],
            index=dates,
            columns=tickers
        )

        data = {'sparse': sparse_data}

        # Fill forward with limit of 10 (longer than data)
        signal = alpha("fill_forward(field('sparse'), 10)")
        result = signal.evaluate(data)

        # Should fill all the way
        expected = pd.DataFrame(
            [[1.0], [1.0], [1.0], [1.0], [1.0]],
            index=dates,
            columns=tickers
        )
        pd.testing.assert_frame_equal(result, expected)


class TestGroupCountValid:
    """Test group_count_valid operation."""

    def test_group_count_valid_basic(self, sample_data):
        """Test counting valid values within groups."""
        # Create sparse earnings-like data
        dates = sample_data['close'].index[:10]
        tickers = sample_data['close'].columns

        # Sparse earnings data - only a few events
        earnings = pd.DataFrame(
            np.nan,
            index=dates,
            columns=tickers
        )
        # AAPL reports day 2, MSFT reports day 4 (both Tech)
        # JPM reports day 3 (Finance)
        earnings.loc[dates[2], 'AAPL'] = 0.5
        earnings.loc[dates[4], 'MSFT'] = 0.3
        earnings.loc[dates[3], 'JPM'] = 0.2

        sample_data['earnings'] = earnings

        signal = alpha("group_count_valid(field('earnings'), 'sector', 3)")
        result = signal.evaluate(sample_data)

        # Check that Tech stocks get Tech count, Finance stocks get Finance count
        # On day 4: Tech should have 2 (AAPL day 2, MSFT day 4), Finance should have 1 (JPM day 3)
        tech_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        finance_tickers = ['JPM', 'BAC', 'GS']

        # Day 4 (index 4): window is days 2,3,4
        # Tech: AAPL on day 2, MSFT on day 4 = 2 events
        # Finance: JPM on day 3 = 1 event
        for ticker in tech_tickers:
            assert result.loc[dates[4], ticker] == 2.0
        for ticker in finance_tickers:
            assert result.loc[dates[4], ticker] == 1.0


class TestPEADAlpha:
    """Test the full PEAD-style alpha construction."""

    def test_pead_alpha_construction(self, sample_data):
        """Test that PEAD alpha can be constructed and evaluated."""
        dates = sample_data['close'].index[:20]
        tickers = sample_data['close'].columns

        # Create earnings estimate data (constant for simplicity)
        estimates = pd.DataFrame(
            1.0,
            index=dates,
            columns=tickers
        )

        # Create sparse reported earnings
        reported = pd.DataFrame(
            np.nan,
            index=dates,
            columns=tickers
        )
        # A few earnings events with surprises
        reported.loc[dates[5], 'AAPL'] = 1.2   # positive surprise
        reported.loc[dates[5], 'JPM'] = 0.8    # negative surprise
        reported.loc[dates[10], 'MSFT'] = 1.5  # positive surprise

        # Subset the sample data to match dates
        test_data = {
            'close': sample_data['close'].loc[dates],
            'volume': sample_data['volume'].loc[dates],
            'earnings_estimate': estimates,
            'earnings_reported': reported,
            'groups': sample_data['groups'],
        }

        # Build the PEAD alpha
        # SUE (price-scaled surprise)
        sue_expr = "(field('earnings_reported') - field('earnings_estimate')) / close()"

        # Held for 5 days
        held_expr = f"fill_forward({sue_expr}, 5)"

        # Weighted by industry reporting count
        weight_expr = "group_count_valid(field('earnings_reported'), 'sector', 5)"

        # Combined alpha
        full_expr = f"({held_expr}) * {weight_expr}"

        signal = alpha(full_expr)
        result = signal.evaluate(test_data)

        # Basic checks
        assert result.shape == (20, 8)

        # On day 5, AAPL had positive surprise, should have non-zero value
        assert result.loc[dates[5], 'AAPL'] != 0

        # On day 5, AAPL's value should be positive (positive surprise)
        # The SUE is (1.2 - 1.0) / price, which is positive
        assert not np.isnan(result.loc[dates[5], 'AAPL'])

        # On day 6, AAPL should still have a value (fill_forward)
        assert not np.isnan(result.loc[dates[6], 'AAPL'])


class TestCombinedOperations:
    """Test combinations of new operations."""

    def test_is_valid_with_fill_forward(self):
        """Test using is_valid to check filled data."""
        dates = pd.date_range('2020-01-01', periods=5, freq='D')
        tickers = ['A']

        sparse_data = pd.DataFrame(
            [[1.0], [np.nan], [np.nan], [np.nan], [np.nan]],
            index=dates,
            columns=tickers
        )

        data = {'sparse': sparse_data}

        # First fill forward, then check validity
        signal = alpha("is_valid(fill_forward(field('sparse'), 2))")
        result = signal.evaluate(data)

        # After fill_forward(2): [1, 1, 1, nan, nan]
        # After is_valid: [1, 1, 1, 0, 0]
        expected = pd.DataFrame(
            [[1.0], [1.0], [1.0], [0.0], [0.0]],
            index=dates,
            columns=tickers
        )
        pd.testing.assert_frame_equal(result, expected)
