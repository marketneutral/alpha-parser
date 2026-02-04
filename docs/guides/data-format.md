# Data Format

How to structure data for qex.

## Basic Structure

Data is a `Dict[str, pd.DataFrame]`:

```python
data = {
    'close': close_df,
    'volume': volume_df,
    'sector': sector_df,
}
```

Each DataFrame must have:

- **Index**: `DatetimeIndex` (trading dates)
- **Columns**: Ticker symbols
- **Values**: Float (use `NaN` for missing)

## Required Fields

For price-based operations:

```python
data = {
    'close': pd.DataFrame(...),  # Required for returns(), volatility()
}
```

Optional:

```python
data = {
    'close': ...,
    'open': ...,     # For open()
    'high': ...,     # For high()
    'low': ...,      # For low()
    'volume': ...,   # For volume(), adv()
}
```

## Custom Fields

Access any field with `field('name')`:

```python
data['earnings_surprise'] = earnings_df
data['analyst_rating'] = rating_df

qex("field('earnings_surprise') * field('analyst_rating')")
```

## Group Data

For `group_rank`, `group_demean`, etc.:

```python
# Same index and columns as price data
# Values are group identifiers (strings)
sector_df = pd.DataFrame(
    [['Tech', 'Tech', 'Finance', 'Finance']] * len(dates),
    index=dates,
    columns=tickers
)

data['sector'] = sector_df

qex("group_demean(returns(20), 'sector')")
```

## Sparse Event Data

For events like earnings (most days are NaN):

```python
# NaN on non-event days
earnings = pd.DataFrame(np.nan, index=dates, columns=tickers)

# Fill in event values
earnings.loc['2024-01-15', 'AAPL'] = 1.25  # AAPL reported
earnings.loc['2024-01-18', 'MSFT'] = 2.10  # MSFT reported

data['earnings'] = earnings

# Forward fill to hold signal
qex("fill_forward(field('earnings'), 60)")

# Check if event occurred
qex("is_valid(field('earnings'))")
```

## LazyData for Large Datasets

Load fields on demand:

```python
from qex import LazyData

data = LazyData({
    'close': lambda: pd.read_parquet('data/close.parquet'),
    'volume': lambda: pd.read_parquet('data/volume.parquet'),
    'earnings': lambda: pd.read_parquet('data/earnings.parquet'),
})

# Only 'close' is loaded (volume, earnings stay on disk)
signal = qex("rank(returns(20))")
result = signal.evaluate(data)
```

### Self-Describing Data

Add descriptions for documentation:

```python
data = LazyData(
    data={
        'close': lambda: pd.read_parquet('close.parquet'),
        'earnings': lambda: pd.read_parquet('earnings.parquet'),
    },
    descriptions={
        'close': 'Daily adjusted close price',
        'earnings': 'Quarterly EPS surprise',
    }
)

# Inspect available fields
print(data.describe())
```

## Example: Complete Dataset

```python
import pandas as pd
import numpy as np

# Setup
dates = pd.date_range('2020-01-01', '2024-01-01', freq='B')
tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'NVDA', 'JPM', 'BAC']

# Price data
np.random.seed(42)
returns = np.random.randn(len(dates), len(tickers)) * 0.02
prices = 100 * np.exp(returns.cumsum(axis=0))

data = {
    'close': pd.DataFrame(prices, index=dates, columns=tickers),
    'volume': pd.DataFrame(
        np.random.randint(1_000_000, 10_000_000, (len(dates), len(tickers))),
        index=dates, columns=tickers
    ),
}

# Sector membership
sectors = ['Tech', 'Tech', 'Tech', 'Tech', 'Tech', 'Tech', 'Finance', 'Finance']
data['sector'] = pd.DataFrame(
    [sectors] * len(dates),
    index=dates, columns=tickers
)

# Sparse earnings (quarterly)
earnings = pd.DataFrame(np.nan, index=dates, columns=tickers)
for ticker in tickers:
    # Random quarterly dates
    quarter_dates = np.random.choice(len(dates), size=16, replace=False)
    earnings.iloc[quarter_dates, tickers.index(ticker)] = np.random.randn(16)
data['earnings'] = earnings

# Use it
signal = qex("group_demean(returns(20), 'sector')")
result = signal.evaluate(data)
```

## Data Alignment

All DataFrames must be aligned:

```python
# Check alignment
assert data['close'].index.equals(data['volume'].index)
assert data['close'].columns.equals(data['volume'].columns)
```

If your data isn't aligned, align it first:

```python
# Align to common index and columns
common_dates = data['close'].index.intersection(data['volume'].index)
common_tickers = data['close'].columns.intersection(data['volume'].columns)

data = {
    'close': data['close'].loc[common_dates, common_tickers],
    'volume': data['volume'].loc[common_dates, common_tickers],
}
```
