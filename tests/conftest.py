"""Shared test fixtures."""

import numpy as np
import pandas as pd
import pytest


class GroupsContainer:
    """Simple container for group data that mimics DataFrame column access."""

    def __init__(self, **kwargs):
        self._groups = kwargs
        self.columns = list(kwargs.keys())

    def __getitem__(self, key):
        return self._groups[key]

    def __contains__(self, key):
        return key in self._groups


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'BAC', 'GS']

    np.random.seed(42)
    prices = pd.DataFrame(
        np.exp(np.random.randn(len(dates), len(tickers)).cumsum(axis=0) * 0.01 + 5),
        index=dates,
        columns=tickers
    )
    volumes = pd.DataFrame(
        np.random.lognormal(15, 1, (len(dates), len(tickers))),
        index=dates,
        columns=tickers
    )

    # Add custom fields
    market_caps = pd.DataFrame(
        np.random.lognormal(10, 0.5, (len(dates), len(tickers))),
        index=dates,
        columns=tickers
    )

    # Create sector groups
    sector_data = {}
    for ticker in tickers:
        sector_data[ticker] = 'Tech' if ticker in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'] else 'Finance'

    groups_df = pd.DataFrame(index=dates, columns=tickers)
    for ticker in tickers:
        groups_df[ticker] = sector_data[ticker]

    return {
        'close': prices,
        'volume': volumes,
        'market_cap': market_caps,
        'groups': GroupsContainer(sector=groups_df)
    }
