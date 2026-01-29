"""Tests based on the original example code."""

import pytest
import numpy as np

from alpha_parser import alpha, compute_context


class TestSimpleReversal:
    """Test simple reversal normalized by volatility."""

    def test_reversal_evaluates(self, sample_data):
        """Test that reversal signal evaluates correctly."""
        with compute_context() as ctx:
            signal = alpha("-returns(20) / volatility(60)")
            result = signal.evaluate(sample_data)

            assert result.shape == (1461, 8)
            assert len(ctx.cache) == 4


class TestMomentumWithFilter:
    """Test momentum filtered by low volatility."""

    def test_momentum_filter_evaluates(self, sample_data):
        """Test momentum with volatility filter."""
        signal = alpha("rank(returns(252)) * (volatility(60) < 0.3)")
        result = signal.evaluate(sample_data)

        assert result.shape == (1461, 8)
        # Values should be between 0 and 1 (rank * binary filter)
        assert result.max().max() <= 1.0
        assert result.min().min() >= 0.0


class TestConditionalMomentum:
    """Test conditional momentum using where()."""

    def test_conditional_evaluates(self, sample_data):
        """Test where() conditional logic."""
        signal = alpha("where(returns(5) > 0, rank(returns(20)), -rank(returns(20)))")
        result = signal.evaluate(sample_data)

        assert result.shape == (1461, 8)
        # Values should be between -1 and 1
        assert result.max().max() <= 1.0
        assert result.min().min() >= -1.0


class TestGroupOperations:
    """Test sector-neutral operations."""

    def test_group_rank_evaluates(self, sample_data):
        """Test group_rank for sector neutrality."""
        signal = alpha("group_rank(returns(20), 'sector')")
        result = signal.evaluate(sample_data)

        assert result.shape == (1461, 8)
        # Rank values should be between 0 and 1
        valid_values = result.dropna()
        assert valid_values.max().max() <= 1.0
        assert valid_values.min().min() >= 0.0


class TestSharedCache:
    """Test that multiple alphas share computation cache."""

    def test_cache_is_shared(self, sample_data):
        """Test cache sharing across multiple alphas."""
        alphas = {
            'Reversal/Vol': "-returns(20) / volatility(60)",
            'Momentum': "rank(returns(252))",
            'Volume': "rank(volume(20))",
            'Combined': "rank(-returns(20)/volatility(60)) + rank(returns(252))"
        }

        with compute_context() as ctx:
            cache_sizes = []
            for name, expr in alphas.items():
                sig = alpha(expr)
                sig.evaluate(sample_data)
                cache_sizes.append(len(ctx.cache))

            # Cache should grow as we add more signals
            assert cache_sizes[0] < cache_sizes[-1]
            # Final cache should have shared computations
            assert cache_sizes[-1] == 10


class TestPortfolioWeights:
    """Test conversion of signals to portfolio weights."""

    def test_weights_normalize(self, sample_data):
        """Test weight normalization."""
        signal = alpha("rank(-returns(20)/volatility(60))")
        weights = signal.to_weights(sample_data, normalize=True, long_only=False)

        assert weights.shape == (1461, 8)
        # Normalized long-short weights should sum to ~1 (abs sum)
        # Skip warm-up period (first 60 days for volatility)
        abs_sums = weights.iloc[60:].abs().sum(axis=1)
        assert np.allclose(abs_sums, 1.0, atol=1e-10)

    def test_weights_long_only(self, sample_data):
        """Test long-only weight conversion."""
        signal = alpha("rank(returns(20))")
        weights = signal.to_weights(sample_data, normalize=True, long_only=True)

        # All weights should be non-negative
        assert (weights >= 0).all().all()
        # Should sum to 1 (skip warm-up period)
        sums = weights.iloc[20:].sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-10)


class TestCustomField:
    """Test custom field access."""

    def test_custom_field_evaluates(self, sample_data):
        """Test accessing custom fields via field()."""
        signal = alpha("delta(field('market_cap'), 20) / ts_std(field('market_cap'), 60)")
        result = signal.evaluate(sample_data)

        assert result.shape == (1461, 8)


class TestParserEdgeCases:
    """Test parser edge cases."""

    def test_nested_operations(self, sample_data):
        """Test deeply nested operations."""
        signal = alpha("rank(ts_mean(returns(5), 10))")
        result = signal.evaluate(sample_data)
        assert result.shape == (1461, 8)

    def test_multiple_comparisons(self, sample_data):
        """Test multiple comparison operations."""
        signal = alpha("(returns(5) > 0) * (volatility(20) < 0.5)")
        result = signal.evaluate(sample_data)
        assert result.shape == (1461, 8)

    def test_arithmetic_with_constants(self, sample_data):
        """Test arithmetic operations with constants."""
        signal = alpha("returns(20) * 2 + 0.5")
        result = signal.evaluate(sample_data)
        assert result.shape == (1461, 8)
