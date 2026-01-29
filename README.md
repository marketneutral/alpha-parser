# Alpha Parser

A DSL (Domain Specific Language) for defining quantitative trading signals and alpha factors.

## Setup

### Prerequisites

Install [uv](https://github.com/astral-sh/uv):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd alpha-parser
```

2. Create a virtual environment and install dependencies:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

### Running Tests

```bash
PYTHONPATH=src pytest tests/ -v
```

## Usage

```python
from alpha_parser import alpha, compute_weights, compute_context

# Parse a signal expression
signal = alpha("rank(-returns(20) / volatility(60))")

# Evaluate with data
result = signal.evaluate(data)

# Convert to portfolio weights
weights = signal.to_weights(data, normalize=True, long_only=False)

# Use compute context for caching across multiple signals
with compute_context():
    signal1 = alpha("-returns(20) / volatility(60)")
    signal2 = alpha("rank(returns(252))")
    result1 = signal1.evaluate(data)
    result2 = signal2.evaluate(data)
```

## Fetching Real Data (FMP)

A script is provided to fetch data from [Financial Modeling Prep](https://financialmodelingprep.com/):

```bash
# 1. Get an API key from https://financialmodelingprep.com/developer/docs/

# 2. Create .env file with your key
cp data/.env.example data/.env
# Edit data/.env and add your API key

# 3. Fetch S&P 500 data (last 5 years)
python data/fetch_fmp.py

# Or fetch specific tickers
python data/fetch_fmp.py --tickers AAPL MSFT GOOG AMZN

# Or specify date range
python data/fetch_fmp.py --start 2020-01-01 --end 2024-01-01
```

This creates parquet files in `data/fmp/`:
```
data/fmp/
├── open.parquet
├── high.parquet
├── low.parquet
├── close.parquet
├── volume.parquet
├── sector.parquet
├── industry.parquet
└── profiles.parquet
```

Use with LazyData:
```python
from alpha_parser import alpha, LazyData
import pandas as pd

data = LazyData({
    'close': lambda: pd.read_parquet('data/fmp/close.parquet'),
    'volume': lambda: pd.read_parquet('data/fmp/volume.parquet'),
    'sector': lambda: pd.read_parquet('data/fmp/sector.parquet'),
})

signal = alpha("group_rank(rank(-returns(20)), 'sector')")
result = signal.evaluate(data)
```

## Data Format

Data is passed as a `Dict[str, pd.DataFrame]` where each DataFrame has:
- **Index**: `DatetimeIndex` (trading dates)
- **Columns**: Ticker symbols (e.g., `'AAPL'`, `'MSFT'`)
- **Values**: Float values (use `NaN` for missing data)

### Required Fields

The following keys are expected for price data:

```python
data = {
    'close': pd.DataFrame(...),   # Closing prices
    'open': pd.DataFrame(...),    # Opening prices (optional)
    'high': pd.DataFrame(...),    # Daily highs (optional)
    'low': pd.DataFrame(...),     # Daily lows (optional)
    'volume': pd.DataFrame(...),  # Trading volume (optional)
}
```

### Custom Fields

Add any additional data using `field('name')`:

```python
data['earnings_estimate'] = earnings_estimate_df
data['earnings_reported'] = earnings_reported_df
data['analyst_rating'] = analyst_rating_df

# Access in expressions
signal = alpha("field('analyst_rating') * returns(20)")
```

### Group Data

For group operations (`group_rank`, `group_demean`, etc.), provide group membership:

```python
data['sector'] = sector_df  # Values like 'Tech', 'Finance', 'Healthcare'

# Use in expressions
signal = alpha("group_rank(returns(20), 'sector')")
```

Group DataFrames should have the same index/columns as price data, with string values indicating group membership.

### Sparse/Event Data

For event-driven signals (earnings, announcements), use `NaN` for dates without events:

```python
# Earnings data: NaN on non-reporting days, actual value on reporting days
#              AAPL   MSFT   GOOG
# 2024-01-15   NaN    NaN    NaN
# 2024-01-16   1.25   NaN    NaN    <- AAPL reported
# 2024-01-17   NaN    NaN    NaN
# 2024-01-18   NaN    2.10   NaN    <- MSFT reported

data['earnings_reported'] = earnings_df

# Forward fill to hold signal after event
signal = alpha("fill_forward(field('earnings_reported'), 5)")

# Check if data point exists
signal = alpha("is_valid(field('earnings_reported'))")
```

### Example: Building a Complete Dataset

```python
import pandas as pd
import numpy as np

# Create date index and tickers
dates = pd.date_range('2020-01-01', '2024-01-01', freq='B')
tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM']

# Price data (required)
data = {
    'close': pd.DataFrame(
        np.random.randn(len(dates), len(tickers)).cumsum(axis=0) + 100,
        index=dates, columns=tickers
    ),
    'volume': pd.DataFrame(
        np.random.randint(1000000, 10000000, (len(dates), len(tickers))),
        index=dates, columns=tickers
    ),
}

# Sector membership (for group operations)
sectors = ['Tech', 'Tech', 'Tech', 'Tech', 'Tech', 'Tech', 'Tech', 'Finance']
data['sector'] = pd.DataFrame(
    [sectors] * len(dates),
    index=dates, columns=tickers
)

# Sparse event data (earnings)
earnings = pd.DataFrame(np.nan, index=dates, columns=tickers)
# Simulate quarterly earnings
for ticker in tickers:
    report_dates = np.random.choice(len(dates), size=16, replace=False)
    earnings.iloc[report_dates, tickers.index(ticker)] = np.random.randn(16)
data['earnings_reported'] = earnings

# Now use it
signal = alpha("rank(-returns(20) / volatility(60))")
result = signal.evaluate(data)
```

### PEAD Example (Sparse Event Data)

You can build complex alphas either by composing Python strings or as a single expression with comments.

**Option 1: Compose with Python f-strings**

```python
# SUE (Standardized Unexpected Earnings) - price-scaled
sue = "(field('earnings_reported') - field('earnings_estimate')) / close()"

# Hold signal for 5 days after earnings announcement
held_sue = f"fill_forward({sue}, 5)"

# Weight by how many stocks in my industry reported this week
weight = "group_count_valid(field('earnings_reported'), 'sector', 5)"

# Final PEAD alpha
signal = alpha(f"rank({held_sue}) * {weight}")
```

**Option 2: Single multi-line string with comments**

```python
signal = alpha("""
    # PEAD: Post-Earnings Announcement Drift
    rank(
        fill_forward(
            # SUE = (actual - estimate) / price
            (field('earnings_reported') - field('earnings_estimate')) / close(),
            5  # hold for 5 days
        )
    ) * group_count_valid(field('earnings_reported'), 'sector', 5)  # weight by industry activity
""")
```

The parser uses Python's `ast.parse()`, so comments and whitespace are handled naturally.

### Lazy Loading (Large Datasets)

For large datasets, use `LazyData` to load fields on demand. Only fields actually used by the signal will be loaded:

```python
from alpha_parser import alpha, LazyData
import pandas as pd

# Define loaders - these are only called when the field is accessed
data = LazyData({
    'close': lambda: pd.read_parquet('data/close.parquet'),
    'volume': lambda: pd.read_parquet('data/volume.parquet'),
    'earnings': lambda: pd.read_parquet('data/earnings.parquet'),
})

# This signal only uses 'close', so 'volume' and 'earnings' are never loaded
signal = alpha("rank(returns(20))")
result = signal.evaluate(data)
```

**Benefits:**
- Only load what you need - unused fields stay on disk
- Automatic caching - each field loaded at most once per evaluation
- Drop-in replacement - works with all existing signals

**Recommended data layout:**
```
data/
├── close.parquet      # One file per field
├── volume.parquet
├── earnings.parquet
└── sector.parquet
```

## Project Structure

```
alpha-parser/
├── src/
│   └── alpha_parser/
│       ├── __init__.py       # Public API exports
│       ├── context.py        # Compute context and caching
│       ├── signal.py         # Base Signal class
│       ├── operators.py      # Arithmetic, comparison, logical, validity ops
│       ├── data.py           # Data field access
│       ├── primitives.py     # Returns, volatility, volume
│       ├── timeseries.py     # Time-series operations
│       ├── crosssection.py   # Cross-sectional operations
│       ├── groups.py         # Group operations
│       ├── conditional.py    # Conditional (where) operations
│       └── parser.py         # Expression parser
├── tests/
│   ├── conftest.py           # Test fixtures
│   ├── test_examples.py      # Example-based tests
│   └── test_events.py        # Event/sparse data tests
├── requirements.txt
├── LICENSE
└── README.md
```

## Available Functions

### Data Access
- `close()`, `open()`, `high()`, `low()` - Price fields
- `field('name')` - Access any named field

### Primitives
- `returns(period)` - Price returns over period
- `volatility(period)` - Rolling volatility
- `volume(period)` - Rolling average volume

### Time-Series Operations
- `ts_mean(signal, period)` - Rolling mean
- `ts_std(signal, period)` - Rolling standard deviation
- `ts_sum(signal, period)` - Rolling sum
- `ts_max(signal, period)` - Rolling maximum
- `ts_min(signal, period)` - Rolling minimum
- `delay(signal, period)` - Lag/shift signal
- `delta(signal, period)` - Difference from N periods ago
- `ts_rank(signal, period)` - Percentile rank within rolling window
- `fill_forward(signal, limit)` - Forward fill NaN for up to N periods

### Cross-Sectional Operations
- `rank(signal)` - Cross-sectional percentile rank (0-1, higher value → rank closer to 1)
- `quantile(signal, buckets)` - Assign to quantile buckets (1-n, higher value → higher bucket)
- `zscore(signal)` - Cross-sectional z-score
- `demean(signal)` - Subtract cross-sectional mean
- `winsorize(signal, limit)` - Cap extreme values at percentiles (e.g., 0.05 caps at 5th/95th)

> **Note:** Both `rank` and `quantile` are ascending: higher signal values produce higher ranks/buckets.

### Conditional
- `where(condition, if_true, if_false)` - Ternary operator

### Group Operations
- `group_rank(signal, 'group_name')` - Rank within groups
- `group_demean(signal, 'group_name')` - Demean within groups
- `group_neutralize(signal, 'group_name')` - Neutralize to groups
- `group_count_valid(signal, 'group_name', window)` - Count non-NaN within group over window

### Validity Operations
- `is_valid(signal)` - Returns 1 where not NaN, 0 otherwise

## Limitations

### Point-in-Time (PIT) Data

This package assumes data is **final and non-restated**. It does not support bi-temporal (point-in-time) data where historical values can be revised.

**The problem:**

```
Jan 8:  Company reports earnings = $1.00
Jan 31: Earnings restated to $3.00

When computing ts_mean(earnings, 20):
- On Jan 15: lookback to Jan 8 should see $1.00 (what was known then)
- On Feb 5:  lookback to Jan 8 should see $3.00 (the restated value)
```

A single DataFrame cell `(Jan 8, AAPL)` can only hold one value, but the "correct" value depends on when you're evaluating. This requires bi-temporal storage and computation, which is not currently supported.

**Workarounds:**

1. **Use only non-restated data** - Price data is rarely restated; fundamentals are more problematic
2. **Accept the bias** - For research/prototyping, using final values may be acceptable
3. **Pre-compute PIT externally** - Build separate DataFrames for each evaluation period in your data pipeline

**Future consideration:** Bi-temporal support would require storing snapshots of historical data at each knowledge date and row-by-row evaluation. This is a significant architectural change.

## License

Apache 2.0 - see [LICENSE](LICENSE) for details.
