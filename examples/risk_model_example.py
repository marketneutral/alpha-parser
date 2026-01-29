#!/usr/bin/env python3
"""Example: Multi-factor risk model using alpha-parser.

This script demonstrates how to:
1. Load market data from FMP
2. Compute style factor exposures
3. Estimate factor returns via cross-sectional regression
4. Build a factor covariance matrix
5. Decompose portfolio risk

Prerequisites:
    1. Fetch data using: python data/fetch_fmp.py
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


def main():
    print("=" * 60)
    print("Multi-Factor Risk Model Example")
    print("=" * 60)

    # Load FMP data
    data_path = Path(__file__).parent.parent / 'data' / 'fmp'
    if not (data_path / 'close.parquet').exists():
        print("\nError: No FMP data found.")
        print("Please fetch data first:")
        print("  python data/fetch_fmp.py")
        print("\nOr for a quick test with fewer tickers:")
        print("  python data/fetch_fmp.py --tickers AAPL MSFT GOOG AMZN META NVDA TSLA JPM BAC WMT")
        sys.exit(1)

    print("\nLoading FMP data...")
    data = LazyData({
        'close': lambda: pd.read_parquet(data_path / 'close.parquet'),
        'volume': lambda: pd.read_parquet(data_path / 'volume.parquet'),
        'market_cap': lambda: pd.read_parquet(data_path / 'market_cap.parquet'),
        'book_to_price': lambda: pd.read_parquet(data_path / 'book_to_price.parquet'),
        'sector': lambda: pd.read_parquet(data_path / 'sector.parquet'),
    })

    # Create the risk model
    print("\n" + "-" * 60)
    print("1. Creating Multi-Factor Risk Model")
    print("-" * 60)

    model = FactorRiskModel(
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

    # Create a sample portfolio (equal weight top 10 by market cap)
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
        momentum = alpha("rank(returns(252) - returns(21))")
        momentum_exp = momentum.evaluate(data)

        # Volatility factor
        vol = alpha("rank(volatility(60))")
        vol_exp = vol.evaluate(data)

        # Reversal factor
        reversal = alpha("rank(-returns(5))")
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
