# Data Format

qex reads data as a Python dict of pandas objects: one entry per **field**, each a table of dates by stocks. This
page is the contract every other part of qex relies on. The pages after it cover labels, universes, event data,
large datasets, adjusted prices, and halts and delistings.

## Basic Structure

Data is a dict of fields:

```python
data = {
    'close': close_df,
    'volume': volume_df,
    'sector': sector_df,
}
```

Each DataFrame field has:

- **index**: a `DatetimeIndex` of trading dates, ascending;
- **columns**: one per stock (ticker symbols, or any stable identifier);
- **values**: numbers, with `NaN` for missing (labels are strings: see [Labels and Groups](labels.md)).

A field with a single value per day can instead be a date-indexed `pd.Series`
(see [Date-only fields](#date-only-fields-and-the-market)).

## Required Fields

For price-based operations:

```python
data = {
    'close': pd.DataFrame(...),  # Required for returns(), volatility()
}
```

Optional, for the operators that read them:

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

### Date-only fields and the market

A field with one value per day, such as an index level or a rate, can be stored as a
`pd.Series` indexed by date. `field('name')` broadcasts it to every ticker, on the dates and
tickers of `close` (dates the series lacks are NaN).

`market()` reads the field `'market'`. Point it at whatever market you hedge against:

```python
data['market'] = spy_total_return_close     # pd.Series, date-indexed

qex("returns(1, 'market')")                              # market daily return
qex("ts_beta(returns(1), returns(1, 'market'), 252)")    # each name's beta
qex("ts_corr(returns(1), returns(1, 'market'), 252)")    # each name's market correlation
```

Use a total-return series (dividends reinvested), so that ex-dividend days are not market
moves. [Market Beta](../examples/market-beta.md) covers choosing the market and estimating beta.

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

All DataFrame fields should share the same index and columns (a date-indexed Series is
aligned to `close` automatically):

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
