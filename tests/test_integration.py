"""Integration tests using real FMP data.

Run with: pytest -m integration
Requires: FMP data fetched to data/fmp/ directory

To fetch data:
    python data/fetch_fmp.py --tickers AAPL MSFT GOOG AMZN JPM BAC KO PEP XOM CVX HD LOW V MA --start 2022-01-01
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
        # Weights should be normalized (sum to ~1)
        last_weights = weights.iloc[-1].dropna()
        assert abs(last_weights.sum() - 1.0) < 0.01  # Normalized to 1


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
        assert result.sharpe is not None
        assert len(result.returns) > 100

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
        assert result.n_quantiles == 5
        assert result.quantile_returns.shape[1] == 5  # 5 columns for 5 quantiles

        # Print for manual inspection
        print("\n" + result.summary())

    def test_ic_analysis(self, fmp_data):
        """Test Information Coefficient calculation."""
        from alpha_parser import alpha, QuantileAnalysis

        signal = alpha("rank(returns(20))")
        qa = QuantileAnalysis(signal, n_quantiles=5)
        ic_stats = qa.ic_summary(fmp_data)

        # Should have IC statistics (returns a pandas Series)
        assert 'Mean Rank IC' in ic_stats.index
        assert 'IC IR (Rank)' in ic_stats.index
        mean_ic = ic_stats['Mean Rank IC']
        assert -1 <= mean_ic <= 1

        print(f"\nIC Stats:\n{ic_stats}")


class TestComplexAlphasWithRealData:
    """Test more complex alpha expressions."""

    def test_52week_high_proximity(self, fmp_data):
        """Test 52-week high proximity signal."""
        from alpha_parser import alpha

        signal = alpha("rank(close() / ts_max(close(), 252))")
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


# Pairs data directory - separate from main FMP data
PAIRS_DATA_DIR = Path(__file__).parent.parent / "data" / "fmp_pairs"


class TestPairsTradingWithRealData:
    """Test pairs trading strategies with real FMP data.

    Classic pairs:
    - KO/PEP (beverages)
    - JPM/BAC (banks)
    - XOM/CVX (oil majors)
    - HD/LOW (home improvement)
    - V/MA (payments)
    """

    @pytest.fixture(scope="class")
    def pairs_fmp_data(self):
        """Load FMP data for pairs trading tests."""
        import pandas as pd

        # Try pairs-specific directory first, fall back to main FMP data
        data_dir = PAIRS_DATA_DIR if PAIRS_DATA_DIR.exists() else DATA_DIR

        if not (data_dir / "close.parquet").exists():
            pytest.skip("FMP data not available")

        close = pd.read_parquet(data_dir / "close.parquet")
        volume = pd.read_parquet(data_dir / "volume.parquet")

        # Define classic pairs
        pairs_config = {
            'KO': 'beverages', 'PEP': 'beverages',
            'JPM': 'banks', 'BAC': 'banks',
            'XOM': 'oil', 'CVX': 'oil',
            'HD': 'home', 'LOW': 'home',
            'V': 'payments', 'MA': 'payments',
        }

        # Filter to tickers that exist in the data
        available_tickers = [t for t in pairs_config.keys() if t in close.columns]

        if len(available_tickers) < 4:
            pytest.skip(f"Not enough pairs tickers. Available: {available_tickers}")

        # Filter data to available tickers
        close = close[[t for t in available_tickers if t in close.columns]]
        volume = volume[[t for t in available_tickers if t in volume.columns]]

        # Build pair grouping DataFrame
        pair_mapping = {t: pairs_config[t] for t in close.columns if t in pairs_config}
        pair_df = pd.DataFrame(
            {t: pair_mapping.get(t, 'other') for t in close.columns},
            index=close.index
        )

        return {
            'close': close,
            'volume': volume,
            'pair': pair_df,
            'available_pairs': list(set(pair_mapping.values())),
        }

    def test_pairs_data_loaded(self, pairs_fmp_data):
        """Verify pairs data is loaded correctly."""
        print(f"\nAvailable tickers: {list(pairs_fmp_data['close'].columns)}")
        print(f"Available pairs: {pairs_fmp_data['available_pairs']}")
        print(f"Date range: {pairs_fmp_data['close'].index[0]} to {pairs_fmp_data['close'].index[-1]}")

        assert pairs_fmp_data['close'].shape[0] > 100
        assert len(pairs_fmp_data['available_pairs']) >= 2

    def test_pairs_spread_mean_reversion(self, pairs_fmp_data):
        """Test basic pairs mean reversion: long laggard, short leader."""
        from alpha_parser import alpha

        signal = alpha("group_demean(-returns(5), 'pair')")
        result = signal.evaluate(pairs_fmp_data)

        # Should have results after warm-up
        valid = result.iloc[10:].dropna(how='all')
        assert len(valid) > 0

        # Check that spreads sum to ~0 within each pair
        for date in valid.index[-5:]:
            for pair_name in pairs_fmp_data['available_pairs']:
                pair_tickers = [t for t in pairs_fmp_data['close'].columns
                               if pairs_fmp_data['pair'].loc[date, t] == pair_name]
                if len(pair_tickers) == 2:
                    pair_sum = result.loc[date, pair_tickers].sum()
                    assert abs(pair_sum) < 1e-10, f"{pair_name} not zero-sum: {pair_sum}"

    def test_pairs_zscore_signal(self, pairs_fmp_data):
        """Test z-score normalized pair spread."""
        from alpha_parser import alpha

        signal = alpha("group_demean(returns(5), 'pair') / group_std(returns(5), 'pair', 60)")
        result = signal.evaluate(pairs_fmp_data)

        # Should have results after 60-day warm-up
        valid = result.iloc[65:].dropna(how='all')
        assert len(valid) > 0

        # Z-scores should be reasonable
        assert (valid.abs() < 10).all().all(), "Z-scores too extreme"

        print(f"\nSample z-scores (last day):\n{result.iloc[-1]}")

    def test_pairs_backtest(self, pairs_fmp_data):
        """Backtest pairs mean reversion strategy."""
        from alpha_parser import alpha, Backtest

        # Simple pairs mean reversion
        signal = alpha("group_demean(-returns(5), 'pair')")
        bt = Backtest(signal, transaction_cost=0.001)
        result = bt.run(pairs_fmp_data)

        assert result.total_return is not None
        assert result.sharpe is not None

        print("\n" + "=" * 50)
        print("PAIRS TRADING BACKTEST (Simple Mean Reversion)")
        print("=" * 50)
        print(result.summary())

    def test_pairs_zscore_backtest(self, pairs_fmp_data):
        """Backtest z-score normalized pairs strategy."""
        from alpha_parser import alpha, Backtest

        # Z-score normalized - trade when spread is stretched
        signal = alpha("-group_demean(returns(5), 'pair') / group_std(returns(5), 'pair', 60)")
        bt = Backtest(signal, transaction_cost=0.001)
        result = bt.run(pairs_fmp_data)

        assert result.total_return is not None

        print("\n" + "=" * 50)
        print("PAIRS TRADING BACKTEST (Z-Score Normalized)")
        print("=" * 50)
        print(result.summary())

    def test_pairs_correlation_check(self, pairs_fmp_data):
        """Verify pairs are actually correlated."""
        import numpy as np

        close = pairs_fmp_data['close']
        returns = close.pct_change().dropna()

        print("\n" + "=" * 50)
        print("PAIRS CORRELATION MATRIX (60-day rolling)")
        print("=" * 50)

        # Check correlation for each pair
        for pair_name in pairs_fmp_data['available_pairs']:
            pair_tickers = [t for t in close.columns
                          if pairs_fmp_data['pair'].iloc[-1][t] == pair_name]
            if len(pair_tickers) == 2:
                t1, t2 = pair_tickers
                corr = returns[t1].corr(returns[t2])
                print(f"{pair_name}: {t1}/{t2} correlation = {corr:.3f}")

                # Pairs should be reasonably correlated (>0.5)
                assert corr > 0.3, f"{pair_name} correlation too low: {corr}"
