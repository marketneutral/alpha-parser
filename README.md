# Alpha Parser

A DSL (Domain Specific Language) for defining quantitative trading signals and alpha factors.

## Why This Package?

Quantitative trading signals often involve complex combinations of operations:
- **Time-series**: rolling means, standard deviations, correlations, momentum
- **Cross-sectional**: ranking stocks, z-scoring, sector neutralization
- **Event-driven**: handling sparse data like earnings announcements

Writing these from scratch is tedious and error-prone. Alpha Parser lets you express complex signals in a single readable expression:

```python
# Instead of 50+ lines of pandas code:
signal = alpha("rank(ts_corr(returns(1), volume(1), 20))")

# Sector-neutral momentum with volatility scaling
signal = alpha("group_demean(returns(60) / volatility(60), 'sector')")

# Combine multiple factors
signal = alpha("0.5 * rank(returns(252)) + 0.5 * rank(-volatility(20))")
```

**Key benefits:**
- **Readable**: Signal logic is self-documenting
- **Composable**: Build complex signals from simple primitives
- **Efficient**: Built-in caching avoids redundant computation
- **Flexible**: Works with any DataFrame-based data pipeline

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

### PEAD Example (Post-Earnings Announcement Drift)

Post-Earnings Announcement Drift is a well-documented anomaly where stock prices continue to drift in the direction of earnings surprises for weeks after the announcement ([Ball & Brown, 1968](https://www.jstor.org/stable/2490232)).

**Basic PEAD Signal:**

```python
# Earnings surprise: actual minus estimate, scaled by price
surprise = "(field('earnings_actual') - field('earnings_estimate')) / close()"

# Hold the surprise signal for 60 trading days (one quarter)
# fill_forward propagates the signal from announcement day
held_surprise = f"fill_forward({surprise}, 60)"

# Rank cross-sectionally: go long positive surprises, short negative
signal = alpha(f"rank({held_surprise}) - 0.5")
```

**With Volatility Scaling:**

```python
# Scale surprise by historical earnings volatility for comparability
# This is closer to the academic "SUE" (Standardized Unexpected Earnings)
sue = """
    fill_forward(
        (field('earnings_actual') - field('earnings_estimate'))
        / ts_std(field('earnings_actual') - field('earnings_estimate'), 8),
        60
    )
"""
signal = alpha(f"rank({sue}) - 0.5")
```

**Notes on PEAD Research:**
- [Hirshleifer, Lim & Teoh (2009)](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.2009.01501.x) found drift is *stronger* on high-news days when investors are distracted
- [DellaVigna & Pollet (2009)](https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.2009.01447.x) found Friday announcements show 69% larger drift
- Recent research suggests PEAD has weakened significantly since 2006 in large-cap stocks

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
│       ├── operators.py      # Arithmetic, comparison, logical, validity, math ops
│       ├── data.py           # Data field access
│       ├── primitives.py     # Returns, volatility, volume, adv
│       ├── timeseries.py     # Time-series operations
│       ├── crosssection.py   # Cross-sectional operations
│       ├── groups.py         # Group operations
│       ├── conditional.py    # Conditional (where) operations
│       ├── parser.py         # Expression parser
│       └── risk.py           # Multi-factor risk model
├── tests/
│   ├── conftest.py           # Test fixtures
│   ├── test_examples.py      # Example-based tests
│   ├── test_events.py        # Event/sparse data tests
│   ├── test_lazy.py          # LazyData tests
│   └── test_operators.py     # Comprehensive operator tests
├── data/
│   ├── fetch_fmp.py          # FMP data fetcher
│   └── .env.example          # API key template
├── examples/
│   └── risk_model_example.py # Risk model usage example
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
- `volatility(period)` - Rolling volatility (annualized)
- `volume(period)` - Rolling average volume
- `adv(period)` - Average dollar volume (price × volume)

### Math Operations
- `log(signal)` - Natural logarithm
- `abs(signal)` - Absolute value
- `sign(signal)` - Sign (-1, 0, or 1)
- `sqrt(signal)` - Square root
- `power(signal, exponent)` - Raise to power
- `max(signal1, signal2)` - Element-wise maximum
- `min(signal1, signal2)` - Element-wise minimum

### Time-Series Operations
- `ts_mean(signal, period)` - Rolling mean
- `ts_std(signal, period)` - Rolling standard deviation
- `ts_sum(signal, period)` - Rolling sum
- `ts_max(signal, period)` - Rolling maximum
- `ts_min(signal, period)` - Rolling minimum
- `delay(signal, period)` - Lag/shift signal
- `delta(signal, period)` - Difference from N periods ago
- `ts_rank(signal, period)` - Percentile rank within rolling window (optimized with scipy)
- `fill_forward(signal, limit)` - Forward fill NaN for up to N periods
- `ts_corr(signal1, signal2, period)` - Rolling correlation
- `ts_cov(signal1, signal2, period)` - Rolling covariance
- `ewma(signal, halflife)` - Exponentially weighted moving average
- `ts_argmax(signal, period)` - Periods since rolling maximum
- `ts_argmin(signal, period)` - Periods since rolling minimum
- `ts_skew(signal, period)` - Rolling skewness
- `ts_kurt(signal, period)` - Rolling kurtosis
- `decay_linear(signal, period)` - Linearly decaying weighted average (recent values weighted more)

### Cross-Sectional Operations
- `rank(signal)` - Cross-sectional percentile rank (0-1, higher value → rank closer to 1)
- `quantile(signal, buckets)` - Assign to quantile buckets (1-n, higher value → higher bucket)
- `zscore(signal)` - Cross-sectional z-score (handles zero std gracefully)
- `demean(signal)` - Subtract cross-sectional mean
- `winsorize(signal, limit)` - Cap extreme values at percentiles (e.g., 0.05 caps at 5th/95th)
- `scale(signal)` - Scale so absolute values sum to 1
- `truncate(signal, max_weight)` - Clip values to [-max_weight, max_weight]

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

## Risk Model

The package includes a multi-factor risk model for portfolio risk estimation:

```python
from alpha_parser import FactorRiskModel, DEFAULT_STYLE_FACTORS

# Create risk model with default style factors
risk_model = FactorRiskModel(factors=DEFAULT_STYLE_FACTORS)

# Fit the model to historical data
results = risk_model.fit(data)

# Access results
print(results.factor_returns)      # Daily factor returns
print(results.factor_covariance)   # Factor covariance matrix
print(results.specific_risk)       # Stock-specific risk estimates
print(results.r_squared)           # Model fit quality per day
```

### Default Style Factors

The model includes these style factors (similar to industry-standard multi-factor models):

- **Size** - Market capitalization (log)
- **Value** - Book-to-price ratio
- **Momentum** - 12-month returns (excluding recent month)
- **Volatility** - 60-day rolling volatility
- **Liquidity** - Average dollar volume (log)
- **Short-term Reversal** - 5-day returns (negative)

### Price-Only Factors

For datasets without fundamental data, use `PRICE_ONLY_FACTORS`:

```python
from alpha_parser import FactorRiskModel, PRICE_ONLY_FACTORS

risk_model = FactorRiskModel(factors=PRICE_ONLY_FACTORS)
```

This includes: Size, Momentum, Volatility, Liquidity, and Short-term Reversal.

### Custom Factors

Define custom factors using signal expressions:

```python
from alpha_parser import FactorRiskModel, FactorDefinition

custom_factors = [
    FactorDefinition('earnings_yield', "field('earnings') / close()"),
    FactorDefinition('analyst_sentiment', "ts_mean(field('rating'), 20)"),
]

risk_model = FactorRiskModel(factors=custom_factors)
```

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
