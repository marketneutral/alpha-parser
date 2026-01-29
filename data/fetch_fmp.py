#!/usr/bin/env python3
"""Fetch OHLCV and company data from Financial Modeling Prep API.

Usage:
    # Fetch S&P 500 constituents (default)
    python fetch_fmp.py

    # Fetch specific tickers
    python fetch_fmp.py --tickers AAPL MSFT GOOG

    # Fetch with custom date range
    python fetch_fmp.py --start 2020-01-01 --end 2024-01-01

    # Specify output directory
    python fetch_fmp.py --output ./my_data

Requires FMP_API_KEY environment variable or .env file.
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# Load .env from script directory or parent
load_dotenv(Path(__file__).parent / '.env')
load_dotenv(Path(__file__).parent.parent / '.env')

BASE_URL = "https://financialmodelingprep.com/api/v3"


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

    # Save profiles separately
    all_data['profiles'] = profiles.set_index('ticker')

    # Save to parquet
    print(f"Saving to {args.output}/...")
    save_parquet(all_data, args.output)

    print("\nDone! Example usage:")
    print(f"""
from alpha_parser import alpha, LazyData
import pandas as pd

data = LazyData({{
    'close': lambda: pd.read_parquet('{args.output}/close.parquet'),
    'volume': lambda: pd.read_parquet('{args.output}/volume.parquet'),
    'sector': lambda: pd.read_parquet('{args.output}/sector.parquet'),
}})

signal = alpha("rank(-returns(20) / volatility(60))")
result = signal.evaluate(data)
""")


if __name__ == '__main__':
    main()
