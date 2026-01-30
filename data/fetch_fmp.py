#!/usr/bin/env python3
"""Fetch OHLCV, fundamental, and company data from Financial Modeling Prep API.

Usage:
    # Fetch S&P 500 constituents (default)
    python fetch_fmp.py

    # Fetch specific tickers
    python fetch_fmp.py --tickers AAPL MSFT GOOG

    # Fetch with custom date range
    python fetch_fmp.py --start 2020-01-01 --end 2024-01-01

    # Specify output directory
    python fetch_fmp.py --output ./my_data

    # Skip fundamentals (faster, price data only)
    python fetch_fmp.py --skip-fundamentals

Requires FMP_API_KEY environment variable or .env file.
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

# Load .env from script directory or parent
load_dotenv(Path(__file__).parent / '.env')
load_dotenv(Path(__file__).parent.parent / '.env')

BASE_URL = "https://financialmodelingprep.com/api/v3"

# Field descriptions for self-documenting data
FIELD_DESCRIPTIONS = {
    # Price data (OHLCV)
    'open': 'Daily opening price, split-adjusted',
    'high': 'Daily high price, split-adjusted',
    'low': 'Daily low price, split-adjusted',
    'close': 'Daily closing price, split-adjusted',
    'volume': 'Daily trading volume in shares',

    # Company classification
    'sector': 'GICS sector classification (e.g., "Technology", "Healthcare")',
    'industry': 'GICS industry classification (more granular than sector)',

    # Market data
    'market_cap': 'Market capitalization in USD (shares outstanding × price)',

    # Fundamental metrics (quarterly, forward-filled to daily)
    'bookValuePerShare': 'Book value per share from most recent quarterly report',
    'book_to_price': 'Book-to-price ratio = book value per share / close price (value factor)',
    'debtToEquity': 'Total debt / total equity from most recent quarterly report',
    'enterpriseValue': 'Market cap + debt - cash, measures total firm value',
    'peRatio': 'Price-to-earnings ratio = price / trailing 12-month EPS',
    'pbRatio': 'Price-to-book ratio = price / book value per share',
    'revenuePerShare': 'Trailing 12-month revenue per share',
    'netIncomePerShare': 'Trailing 12-month net income per share (similar to EPS)',
    'earningsYield': 'Earnings yield = EPS / price (inverse of P/E)',
    'dividendYield': 'Annual dividend per share / price',

    # Metadata
    'profiles': 'Company metadata: name, sector, industry, exchange',
}


def get_api_key() -> str:
    """Get FMP API key from environment."""
    key = os.getenv("FMP_API_KEY")
    if not key:
        print("Error: FMP_API_KEY not found in environment.")
        print("Set it in .env file or export FMP_API_KEY=your_key")
        sys.exit(1)
    return key


def fetch_sp500_constituents(api_key: str) -> list[str]:
    """Fetch current S&P 500 constituent tickers."""
    url = f"{BASE_URL}/sp500_constituent?apikey={api_key}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()
    return [item['symbol'] for item in data]


def fetch_company_profiles(tickers: list[str], api_key: str) -> pd.DataFrame:
    """Fetch company profiles (sector, industry) for tickers."""
    profiles = []

    # FMP allows batch requests of up to 500 symbols
    batch_size = 100
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        symbols = ','.join(batch)
        url = f"{BASE_URL}/profile/{symbols}?apikey={api_key}"

        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()

        for item in data:
            profiles.append({
                'ticker': item.get('symbol'),
                'name': item.get('companyName'),
                'sector': item.get('sector'),
                'industry': item.get('industry'),
                'exchange': item.get('exchangeShortName'),
            })

        # Rate limiting
        time.sleep(0.2)

    return pd.DataFrame(profiles)


def fetch_historical_prices(
    tickers: list[str],
    api_key: str,
    start_date: str,
    end_date: str,
) -> dict[str, pd.DataFrame]:
    """Fetch historical OHLCV data for tickers.

    Returns dict with keys: 'open', 'high', 'low', 'close', 'volume'
    Each value is a DataFrame with dates as index, tickers as columns.
    """
    all_data = []
    failed = []

    print(f"Fetching historical prices for {len(tickers)} tickers...")

    for i, ticker in enumerate(tickers):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i + 1}/{len(tickers)}")

        url = (
            f"{BASE_URL}/historical-price-full/{ticker}"
            f"?from={start_date}&to={end_date}&apikey={api_key}"
        )

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            if 'historical' not in data:
                failed.append(ticker)
                continue

            df = pd.DataFrame(data['historical'])
            df['ticker'] = ticker
            all_data.append(df)

        except Exception as e:
            print(f"  Warning: Failed to fetch {ticker}: {e}")
            failed.append(ticker)

        # Rate limiting
        time.sleep(0.1)

    if failed:
        print(f"  Failed tickers ({len(failed)}): {failed[:10]}{'...' if len(failed) > 10 else ''}")

    if not all_data:
        raise ValueError("No data fetched successfully")

    # Combine all data
    combined = pd.concat(all_data, ignore_index=True)
    combined['date'] = pd.to_datetime(combined['date'])

    # Pivot to wide format: dates as index, tickers as columns
    result = {}
    for field in ['open', 'high', 'low', 'close', 'volume']:
        pivot = combined.pivot(index='date', columns='ticker', values=field)
        pivot = pivot.sort_index()
        result[field] = pivot

    return result


def fetch_historical_market_cap(
    tickers: list[str],
    api_key: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch historical market cap data for tickers."""
    all_data = []
    failed = []

    print(f"Fetching historical market cap for {len(tickers)} tickers...")

    for i, ticker in enumerate(tickers):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i + 1}/{len(tickers)}")

        url = (
            f"{BASE_URL}/historical-market-capitalization/{ticker}"
            f"?from={start_date}&to={end_date}&apikey={api_key}"
        )

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            if not data:
                failed.append(ticker)
                continue

            df = pd.DataFrame(data)
            df['ticker'] = ticker
            all_data.append(df)

        except Exception as e:
            print(f"  Warning: Failed to fetch market cap for {ticker}: {e}")
            failed.append(ticker)

        # Rate limiting
        time.sleep(0.1)

    if failed:
        print(f"  Failed tickers ({len(failed)}): {failed[:10]}{'...' if len(failed) > 10 else ''}")

    if not all_data:
        return pd.DataFrame()

    # Combine all data
    combined = pd.concat(all_data, ignore_index=True)
    combined['date'] = pd.to_datetime(combined['date'])

    # Pivot to wide format
    pivot = combined.pivot(index='date', columns='ticker', values='marketCap')
    return pivot.sort_index()


def fetch_key_metrics(
    tickers: list[str],
    api_key: str,
    period: str = 'quarter',
    limit: int = 40,  # ~10 years of quarterly data
) -> dict[str, pd.DataFrame]:
    """Fetch key metrics (book value, debt/equity, etc.) for tickers.

    Returns dict with DataFrames for each metric, dates as index, tickers as columns.
    """
    all_data = []
    failed = []

    print(f"Fetching key metrics for {len(tickers)} tickers...")

    for i, ticker in enumerate(tickers):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i + 1}/{len(tickers)}")

        url = (
            f"{BASE_URL}/key-metrics/{ticker}"
            f"?period={period}&limit={limit}&apikey={api_key}"
        )

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            if not data:
                failed.append(ticker)
                continue

            df = pd.DataFrame(data)
            df['ticker'] = ticker
            all_data.append(df)

        except Exception as e:
            print(f"  Warning: Failed to fetch metrics for {ticker}: {e}")
            failed.append(ticker)

        # Rate limiting
        time.sleep(0.1)

    if failed:
        print(f"  Failed tickers ({len(failed)}): {failed[:10]}{'...' if len(failed) > 10 else ''}")

    if not all_data:
        return {}

    # Combine all data
    combined = pd.concat(all_data, ignore_index=True)
    combined['date'] = pd.to_datetime(combined['date'])

    # Extract key metrics for risk model
    metrics = {}
    metric_fields = [
        'bookValuePerShare',
        'debtToEquity',
        'enterpriseValue',
        'peRatio',
        'pbRatio',
        'revenuePerShare',
        'netIncomePerShare',
        'earningsYield',
        'dividendYield',
    ]

    for field in metric_fields:
        if field in combined.columns:
            pivot = combined.pivot(index='date', columns='ticker', values=field)
            pivot = pivot.sort_index()
            metrics[field] = pivot

    return metrics


def forward_fill_to_daily(
    quarterly_df: pd.DataFrame,
    daily_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Forward fill quarterly data to daily frequency."""
    # Reindex to daily dates and forward fill
    result = quarterly_df.reindex(daily_index, method='ffill')
    return result


def build_group_data(
    profiles: pd.DataFrame,
    price_data: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Build sector/industry DataFrames aligned with price data."""
    dates = price_data.index
    tickers = price_data.columns.tolist()

    # Create lookup from profiles
    sector_map = profiles.set_index('ticker')['sector'].to_dict()
    industry_map = profiles.set_index('ticker')['industry'].to_dict()

    # Build DataFrames with same structure as price data
    sectors = [sector_map.get(t, None) for t in tickers]
    industries = [industry_map.get(t, None) for t in tickers]

    sector_df = pd.DataFrame(
        [sectors] * len(dates),
        index=dates,
        columns=tickers,
    )

    industry_df = pd.DataFrame(
        [industries] * len(dates),
        index=dates,
        columns=tickers,
    )

    return {
        'sector': sector_df,
        'industry': industry_df,
    }


def save_parquet(data: dict[str, pd.DataFrame], output_dir: Path):
    """Save DataFrames to parquet files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, df in data.items():
        path = output_dir / f"{name}.parquet"
        df.to_parquet(path)
        print(f"  Saved {path} ({df.shape[0]} rows x {df.shape[1]} cols)")

    # Save field descriptions as JSON for loading later
    import json
    descriptions_path = output_dir / 'field_descriptions.json'
    # Only include descriptions for fields we actually saved
    saved_descriptions = {k: v for k, v in FIELD_DESCRIPTIONS.items() if k in data}
    with open(descriptions_path, 'w') as f:
        json.dump(saved_descriptions, f, indent=2)
    print(f"  Saved {descriptions_path}")


def load_fmp_data(data_dir: Path | str = None) -> 'LazyData':
    """Load FMP data as self-describing LazyData.

    Args:
        data_dir: Path to FMP data directory. Defaults to ./data/fmp

    Returns:
        LazyData with all available fields and descriptions.

    Example:
        from data.fetch_fmp import load_fmp_data

        data = load_fmp_data()
        print(data.describe())  # See all available fields

        signal = alpha("rank(returns(20) / volatility(60))")
        result = signal.evaluate(data)
    """
    import json
    from alpha_parser import LazyData

    if data_dir is None:
        data_dir = Path(__file__).parent / 'fmp'
    else:
        data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            f"Run 'python data/fetch_fmp.py' to download data first."
        )

    # Load descriptions
    desc_path = data_dir / 'field_descriptions.json'
    if desc_path.exists():
        with open(desc_path) as f:
            descriptions = json.load(f)
    else:
        descriptions = FIELD_DESCRIPTIONS

    # Build lazy loaders for all parquet files
    data = {}
    for parquet_file in data_dir.glob('*.parquet'):
        field_name = parquet_file.stem
        # Create closure with explicit path binding
        data[field_name] = (lambda p=parquet_file: pd.read_parquet(p))

    return LazyData(data=data, descriptions=descriptions)


def main():
    parser = argparse.ArgumentParser(description="Fetch FMP data for alpha-parser")
    parser.add_argument(
        '--tickers',
        nargs='+',
        help='Tickers to fetch (default: S&P 500 constituents)',
    )
    parser.add_argument(
        '--start',
        default=(datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d'),
        help='Start date (default: 5 years ago)',
    )
    parser.add_argument(
        '--end',
        default=datetime.now().strftime('%Y-%m-%d'),
        help='End date (default: today)',
    )
    parser.add_argument(
        '--output',
        default=Path(__file__).parent / 'fmp',
        type=Path,
        help='Output directory (default: ./data/fmp)',
    )
    parser.add_argument(
        '--skip-fundamentals',
        action='store_true',
        help='Skip fetching fundamental data (faster)',
    )

    args = parser.parse_args()
    api_key = get_api_key()

    # Get tickers
    if args.tickers:
        tickers = args.tickers
        print(f"Using provided tickers: {len(tickers)} symbols")
    else:
        print("Fetching S&P 500 constituents...")
        tickers = fetch_sp500_constituents(api_key)
        print(f"  Found {len(tickers)} constituents")

    # Fetch company profiles
    print("Fetching company profiles...")
    profiles = fetch_company_profiles(tickers, api_key)
    print(f"  Got profiles for {len(profiles)} companies")

    # Fetch historical prices
    prices = fetch_historical_prices(tickers, api_key, args.start, args.end)
    print(f"  Date range: {prices['close'].index.min()} to {prices['close'].index.max()}")

    # Build group data
    print("Building group data...")
    groups = build_group_data(profiles, prices['close'])

    # Combine all data
    all_data = {**prices, **groups}

    # Fetch fundamental data for risk model
    if not args.skip_fundamentals:
        daily_index = prices['close'].index

        # Market cap
        market_cap = fetch_historical_market_cap(tickers, api_key, args.start, args.end)
        if not market_cap.empty:
            # Align to daily index
            market_cap = market_cap.reindex(daily_index, method='ffill')
            all_data['market_cap'] = market_cap

        # Key metrics (quarterly, forward-filled to daily)
        metrics = fetch_key_metrics(tickers, api_key)
        for metric_name, metric_df in metrics.items():
            if not metric_df.empty:
                # Forward fill quarterly data to daily
                daily_metric = forward_fill_to_daily(metric_df, daily_index)
                all_data[metric_name] = daily_metric
                print(f"  Added {metric_name}")

        # Compute derived fields for risk model
        if 'bookValuePerShare' in all_data and 'close' in all_data:
            # Book-to-price ratio
            btp = all_data['bookValuePerShare'] / all_data['close']
            all_data['book_to_price'] = btp
            print("  Added book_to_price (derived)")

    # Save profiles separately
    all_data['profiles'] = profiles.set_index('ticker')

    # Save to parquet
    print(f"\nSaving to {args.output}/...")
    save_parquet(all_data, args.output)

    print("\nDone! Available fields:")
    for name in sorted(all_data.keys()):
        print(f"  - {name}")

    print(f"""
Example usage:

import json
import pandas as pd
from alpha_parser import alpha, LazyData

# Load field descriptions
with open('{args.output}/field_descriptions.json') as f:
    descriptions = json.load(f)

# Create self-describing LazyData
data = LazyData(
    data={{
        'close': lambda: pd.read_parquet('{args.output}/close.parquet'),
        'volume': lambda: pd.read_parquet('{args.output}/volume.parquet'),
        'market_cap': lambda: pd.read_parquet('{args.output}/market_cap.parquet'),
        'book_to_price': lambda: pd.read_parquet('{args.output}/book_to_price.parquet'),
        'sector': lambda: pd.read_parquet('{args.output}/sector.parquet'),
    }},
    descriptions=descriptions,
)

# Inspect available data
print(data.describe())

# Build and evaluate signals
signal = alpha("rank(returns(20))")
result = signal.evaluate(data)
""")


if __name__ == '__main__':
    main()
