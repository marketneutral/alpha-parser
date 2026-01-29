"""Integration tests using real FMP data.

Run with: pytest -m integration
Requires: FMP data fetched to data/fmp/ directory

To fetch data:
    python data/fetch_fmp.py --tickers AAPL MSFT GOOG AMZN NVDA META TSLA JPM V MA --start 2022-01-01
"""

import pytest
from pathlib import Path

# Skip all tests in this module if data not present
DATA_DIR = Path(__file__).parent.parent / "data" / "fmp"
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (DATA_DIR / "close.parquet").exists(),
        reason="FMP data not fetched. Run: python data/fetch_fmp.py --tickers AAPL MSFT GOOG"
    ),
]


@pytest.fixture(scope="module")
def fmp_data():
    """Load real FMP data."""
    import pandas as pd

    return {
        'close': pd.read_parquet(DATA_DIR / "close.parquet"),
        'volume': pd.read_parquet(DATA_DIR / "volume.parquet"),
        'open': pd.read_parquet(DATA_DIR / "open.parquet"),
        'high': pd.read_parquet(DATA_DIR / "high.parquet"),
        'low': pd.read_parquet(DATA_DIR / "low.parquet"),
    }


class TestAlphasWithRealData:
    """Test alpha signals with real market data."""

    def test_momentum_alpha(self, fmp_data):
        """Test basic momentum signal."""
        from alpha_parser import alpha

        signal = alpha("rank(returns(20))")
        weights = signal.to_weights(fmp_data)

        # Should have valid weights
        assert weights.shape[0] > 100  # At least 100 days
        assert weights.shape[1] >= 5   # At least 5 tickers
        assert not weights.iloc[-1].isna().all()  # Last row has data

    def test_reversal_alpha(self, fmp_data):
        """Test mean reversion signal."""
        from alpha_parser import alpha

        signal = alpha("rank(-returns(5) / volatility(20))")
        weights = signal.to_weights(fmp_data)

        assert weights.shape[0] > 100
        assert not weights.iloc[-1].isna().all()

    def test_volume_adjusted_reversal(self, fmp_data):
        """Test volume-adjusted reversal (Alpha #1 from discussion)."""
        from alpha_parser import alpha

        signal = alpha("rank(-returns(5) * (volume(5) / adv(20)))")
        weights = signal.to_weights(fmp_data)

        assert weights.shape[0] > 100
        # Weights should be roughly balanced (long/short)
        last_weights = weights.iloc[-1].dropna()
        assert abs(last_weights.sum()) < 0.1  # Near zero sum


class TestBacktestWithRealData:
    """Test backtesting with real market data."""

    def test_backtest_momentum(self, fmp_data):
        """Backtest momentum strategy."""
        from alpha_parser import alpha, Backtest

        signal = alpha("rank(returns(20)) - 0.5")
        bt = Backtest(signal, transaction_cost=0.001)
        result = bt.run(fmp_data)

        # Should complete without error and have results
        assert result.total_return is not None
        assert result.sharpe_ratio is not None
        assert len(result.daily_returns) > 100

        # Print for manual inspection
        print("\n" + result.summary())

    def test_backtest_reversal(self, fmp_data):
        """Backtest reversal strategy."""
        from alpha_parser import alpha, Backtest

        signal = alpha("rank(-returns(5)) - 0.5")
        bt = Backtest(signal, transaction_cost=0.001)
        result = bt.run(fmp_data)

        assert result.total_return is not None
        print("\n" + result.summary())

    def test_backtest_combined_alpha(self, fmp_data):
        """Backtest combined momentum + reversal."""
        from alpha_parser import alpha, Backtest

        # Blend short-term reversal with medium-term momentum
        signal = alpha("rank(-returns(5)) * 0.5 + rank(returns(60)) * 0.5 - 0.5")
        bt = Backtest(signal, transaction_cost=0.001)
        result = bt.run(fmp_data)

        assert result.total_return is not None
        print("\n" + result.summary())


class TestQuantileAnalysisWithRealData:
    """Test quantile analysis with real market data."""

    def test_quantile_momentum(self, fmp_data):
        """Quantile analysis of momentum signal."""
        from alpha_parser import alpha, QuantileAnalysis

        signal = alpha("rank(returns(20))")
        qa = QuantileAnalysis(signal, n_quantiles=5)
        result = qa.run(fmp_data)

        # Should have 5 quantiles
        assert len(result.quantile_returns) == 5

        # Print for manual inspection
        print("\n" + result.summary())

    def test_ic_analysis(self, fmp_data):
        """Test Information Coefficient calculation."""
        from alpha_parser import alpha, QuantileAnalysis

        signal = alpha("rank(returns(20))")
        qa = QuantileAnalysis(signal, n_quantiles=5)
        ic_stats = qa.ic_summary(fmp_data)

        # Should have IC statistics
        assert 'mean_ic' in ic_stats
        assert 'ic_ir' in ic_stats
        assert -1 <= ic_stats['mean_ic'] <= 1

        print(f"\nIC Stats: {ic_stats}")


class TestComplexAlphasWithRealData:
    """Test more complex alpha expressions."""

    def test_52week_high_proximity(self, fmp_data):
        """Test 52-week high proximity signal."""
        from alpha_parser import alpha

        signal = alpha("rank(close / ts_max(close, 252))")
        weights = signal.to_weights(fmp_data)

        # Need at least 252 days of data for this signal
        valid_weights = weights.dropna(how='all')
        assert len(valid_weights) > 0

    def test_volatility_mean_reversion(self, fmp_data):
        """Test volatility regime mean reversion."""
        from alpha_parser import alpha

        signal = alpha("rank(where(volatility(20) > ts_mean(volatility(20), 60), -returns(5), returns(5)))")
        weights = signal.to_weights(fmp_data)

        valid_weights = weights.dropna(how='all')
        assert len(valid_weights) > 0

    def test_trend_aligned_reversal(self, fmp_data):
        """Test trend-aligned reversal signal."""
        from alpha_parser import alpha

        signal = alpha("rank(sign(returns(60)) * -returns(5))")
        weights = signal.to_weights(fmp_data)

        assert weights.shape[0] > 60
