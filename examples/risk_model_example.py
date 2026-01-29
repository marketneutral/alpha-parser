#!/usr/bin/env python3
"""Example: Multi-factor risk model using alpha-parser.

This script demonstrates how to:
1. Load market data
2. Compute style factor exposures
3. Estimate factor returns via cross-sectional regression
4. Build a factor covariance matrix
5. Decompose portfolio risk

Prerequisites:
    1. Fetch data using: python data/fetch_fmp.py --tickers AAPL MSFT GOOG AMZN META NVDA TSLA JPM BAC WMT
    2. Install dependencies: uv pip install -r requirements.txt

Usage:
    PYTHONPATH=src python examples/risk_model_example.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from alpha_parser import (
    alpha,
    LazyData,
    compute_context,
)
from alpha_parser.risk import (
    FactorRiskModel,
    FactorDefinition,
    PRICE_ONLY_FACTORS,
)


def create_sample_data():
    """Create synthetic data for demonstration."""
    np.random.seed(42)

    # 2 years of daily data
    dates = pd.date_range('2022-01-01', '2024-01-01', freq='B')
    tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'BAC', 'WMT']

    n_dates = len(dates)
    n_stocks = len(tickers)

    # Simulate correlated returns with factor structure
    # True factors: market, size, momentum
    market_returns = np.random.randn(n_dates) * 0.01
    size_returns = np.random.randn(n_dates) * 0.005
    momentum_returns = np.random.randn(n_dates) * 0.005

    # Stock factor loadings
    market_beta = np.array([1.2, 1.1, 1.3, 1.4, 1.5, 1.8, 2.0, 0.9, 1.0, 0.7])
    size_loading = np.array([0.5, 0.4, 0.3, 0.6, 0.2, 0.1, -0.2, 0.3, 0.2, 0.1])
    momentum_loading = np.array([0.3, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, -0.1, 0.0, -0.2])

    # Generate returns
    returns = np.zeros((n_dates, n_stocks))
    for i in range(n_stocks):
        factor_return = (
            market_beta[i] * market_returns +
            size_loading[i] * size_returns +
            momentum_loading[i] * momentum_returns
        )
        specific_return = np.random.randn(n_dates) * 0.015  # Idiosyncratic
        returns[:, i] = factor_return + specific_return

    # Convert to prices
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    close_df = pd.DataFrame(prices, index=dates, columns=tickers)

    # Volume (correlated with volatility)
    vol = np.abs(returns).mean(axis=0)
    base_volume = 1e7 * (1 + vol * 10)
    volume = np.random.poisson(base_volume, size=(n_dates, n_stocks))
    volume_df = pd.DataFrame(volume, index=dates, columns=tickers)

    # Market cap (size factor)
    market_caps = np.array([3e12, 2.8e12, 1.8e12, 1.5e12, 0.9e12, 1.2e12, 0.8e12, 0.5e12, 0.3e12, 0.4e12])
    # Add some time variation
    market_cap_growth = np.exp(np.cumsum(returns * 0.5, axis=0))
    market_cap_df = pd.DataFrame(
        market_caps * market_cap_growth,
        index=dates,
        columns=tickers
    )

    # Book-to-price (value factor)
    book_to_price = np.array([0.15, 0.12, 0.08, 0.05, 0.10, 0.06, 0.04, 0.80, 0.90, 0.60])
    # Add noise
    btp_noise = 1 + np.random.randn(n_dates, n_stocks) * 0.1
    book_to_price_df = pd.DataFrame(
        book_to_price * btp_noise,
        index=dates,
        columns=tickers
    )

    # Sector assignments
    sectors = ['Tech', 'Tech', 'Tech', 'Tech', 'Tech', 'Tech', 'Tech', 'Finance', 'Finance', 'Consumer']
    sector_df = pd.DataFrame(
        [sectors] * n_dates,
        index=dates,
        columns=tickers
    )

    return {
        'close': close_df,
        'volume': volume_df,
        'market_cap': market_cap_df,
        'book_to_price': book_to_price_df,
        'sector': sector_df,
    }


def main():
    print("=" * 60)
    print("Multi-Factor Risk Model Example")
    print("=" * 60)

    # Try to load real data, fall back to synthetic
    data_path = Path(__file__).parent.parent / 'data' / 'fmp'
    if (data_path / 'close.parquet').exists():
        print("\nLoading FMP data...")
        data = LazyData({
            'close': lambda: pd.read_parquet(data_path / 'close.parquet'),
            'volume': lambda: pd.read_parquet(data_path / 'volume.parquet'),
            'market_cap': lambda: pd.read_parquet(data_path / 'market_cap.parquet'),
            'book_to_price': lambda: pd.read_parquet(data_path / 'book_to_price.parquet'),
            'sector': lambda: pd.read_parquet(data_path / 'sector.parquet'),
        })
    else:
        print("\nNo FMP data found. Using synthetic data for demonstration.")
        print("To use real data, run: python data/fetch_fmp.py")
        data = create_sample_data()

    # Create the risk model
    print("\n" + "-" * 60)
    print("1. Creating Multi-Factor Risk Model")
    print("-" * 60)

    # Use simpler factors for the demo (less warm-up required)
    simple_factors = [
        FactorDefinition("momentum", "rank(returns(60))", "60-day momentum"),
        FactorDefinition("volatility", "rank(volatility(20))", "20-day volatility"),
        FactorDefinition("reversal", "rank(-returns(5))", "5-day reversal"),
    ]

    model = FactorRiskModel(
        style_factors=simple_factors,
        include_industries=True,
        industry_column='sector',
        half_life=63,  # 3-month half-life for covariance
    )

    # Fit the model
    print("\n" + "-" * 60)
    print("2. Fitting the Model (Cross-Sectional Regression)")
    print("-" * 60)

    model.fit(data, lookback=252)

    # Display factor returns summary
    print("\n" + "-" * 60)
    print("3. Factor Returns Summary")
    print("-" * 60)

    factor_returns = model.factor_returns
    print("\nAnnualized Factor Returns (mean):")
    print((factor_returns.mean() * 252).round(4))

    print("\nFactor Return Volatility (annualized):")
    print((factor_returns.std() * np.sqrt(252)).round(4))

    # Display factor covariance
    print("\n" + "-" * 60)
    print("4. Factor Covariance Matrix (subset)")
    print("-" * 60)

    # Show just style factors
    style_names = [f.name for f in model.style_factors]
    cov_style = model.factor_covariance.loc[
        model.factor_covariance.index.intersection(style_names),
        model.factor_covariance.columns.intersection(style_names)
    ]
    print("\nStyle Factor Covariance (annualized):")
    print(cov_style.round(6))

    # Factor correlations
    std = np.sqrt(np.diag(cov_style))
    corr = cov_style / np.outer(std, std)
    print("\nStyle Factor Correlations:")
    print(corr.round(3))

    # Specific risk
    print("\n" + "-" * 60)
    print("5. Specific Risk (Top 10)")
    print("-" * 60)

    specific_risk = model.specific_risk.sort_values(ascending=False)
    print("\nHighest specific risk stocks:")
    print(specific_risk.head(10).round(4))

    # Portfolio risk decomposition
    print("\n" + "-" * 60)
    print("6. Portfolio Risk Decomposition")
    print("-" * 60)

    # Create a sample portfolio
    tickers = model.results_.factor_exposures.index.tolist()[:10]
    weights = pd.Series(1.0 / len(tickers), index=tickers)
    weights = weights / weights.sum()  # Normalize

    print(f"\nEqual-weight portfolio of {len(tickers)} stocks:")
    print(weights.round(3))

    total_risk, factor_risk, specific_risk_port = model.portfolio_risk(weights)

    print(f"\nPortfolio Risk Decomposition:")
    print(f"  Total Risk (annualized):    {total_risk:.2%}")
    print(f"  Factor Risk:                {factor_risk:.2%}")
    print(f"  Specific Risk:              {specific_risk_port:.2%}")
    print(f"  Factor Risk %:              {(factor_risk/total_risk)**2:.1%} of variance")

    # Factor attribution
    print("\n" + "-" * 60)
    print("7. Factor Risk Attribution")
    print("-" * 60)

    attribution = model.factor_attribution(weights)
    attribution = attribution.sort_values(ascending=False)

    print("\nTop risk contributors (variance contribution):")
    for factor, contrib in attribution.head(10).items():
        print(f"  {factor:25s}: {contrib:.6f}")

    # Example: compute custom factor exposures
    print("\n" + "-" * 60)
    print("8. Custom Factor Analysis with alpha-parser")
    print("-" * 60)

    with compute_context():
        # Momentum factor
        momentum = alpha("zscore(returns(252) - returns(21))")
        momentum_exp = momentum.evaluate(data)

        # Volatility factor
        vol = alpha("zscore(volatility(60))")
        vol_exp = vol.evaluate(data)

        # Reversal factor
        reversal = alpha("zscore(-returns(5))")
        rev_exp = reversal.evaluate(data)

    # Get latest exposures
    latest_date = momentum_exp.index[-1]
    print(f"\nFactor Exposures as of {latest_date.date()}:")

    exposures_df = pd.DataFrame({
        'momentum': momentum_exp.loc[latest_date],
        'volatility': vol_exp.loc[latest_date],
        'reversal': rev_exp.loc[latest_date],
    }).dropna()

    print(exposures_df.head(10).round(3))

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
