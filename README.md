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

**Option 1: Install from Git (for use in other projects)**

```bash
# Basic install
uv pip install git+https://github.com/youruser/alpha-parser.git

# With optional dependencies
uv pip install "alpha-parser[all] @ git+https://github.com/youruser/alpha-parser.git"
```

**Option 2: Local development**

```bash
git clone <repository-url>
cd alpha-parser
uv venv
source .venv/bin/activate
uv pip install -e ".[all]"
```

**Optional dependency groups:**
- `dev` - pytest, pyarrow (for testing)
- `data` - requests, python-dotenv, pyarrow (for FMP data fetcher)
- `risk` - statsmodels (for risk model)
- `all` - everything

### Running Tests

```bash
pytest tests/ -v
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
# Earnings surprise: actual minus estimate
surprise = "field('earnings_actual') - field('earnings_estimate')"

# Hold the surprise signal for 60 trading days (one quarter)
# fill_forward propagates the signal from announcement day
held_surprise = f"fill_forward({surprise}, 60)"

# Rank cross-sectionally: go long positive surprises, short negative
signal = alpha(f"rank({held_surprise}) - 0.5")
```

**With Proper SUE (Standardized Unexpected Earnings):**

The academic definition of SUE divides the earnings surprise by its historical standard deviation. Since earnings are quarterly (sparse data), we use `ts_std_events` which computes the rolling std over the past N *announcements*, not the past N days:

```python
# SUE = surprise / std of past 8 earnings surprises
surprise = "field('earnings_actual') - field('earnings_estimate')"
sue = f"({surprise}) / ts_std_events({surprise}, 8)"

# Hold for 60 days, rank cross-sectionally
signal = alpha(f"rank(fill_forward({sue}, 60)) - 0.5")
```

**Notes on PEAD Research:**
- [Hirshleifer, Lim & Teoh (2009)](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.2009.01501.x) found drift is *stronger* on high-news days when investors are distracted
- [DellaVigna & Pollet (2009)](https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.2009.01447.x) found Friday announcements show 69% larger drift
- Recent research suggests PEAD has weakened significantly since 2006 in large-cap stocks

### Technical Indicator Examples

Alpha Parser makes it easy to express classic technical indicators as single-line expressions.

**Bollinger Band Mean-Reversion:**

Bollinger Bands measure how far price is from its moving average in terms of standard deviations. The %B indicator (0-1 scale) shows where price is within the bands. This signal goes long when price is near the lower band (oversold) and short when near the upper band (overbought):

```python
# Bollinger %B mean-reversion signal
# When %B is low (near 0), price is near lower band → go long
# When %B is high (near 1), price is near upper band → go short
signal = alpha("""
    rank(-(close() - ts_mean(close(), 20)) / (2 * ts_std(close(), 20))) - 0.5
""")
```

**RSI Mean-Reversion:**

The Relative Strength Index (RSI) measures the ratio of recent gains to total price movement. This signal goes long when RSI is low (oversold) and short when RSI is high (overbought):

```python
# RSI mean-reversion signal
# Low RSI (oversold) → go long, High RSI (overbought) → go short
signal = alpha("""
    rank(-(100 * ts_mean(max(delta(close(), 1), 0), 14)
        / (ts_mean(max(delta(close(), 1), 0), 14)
           + ts_mean(max(-delta(close(), 1), 0), 14)))) - 0.5
""")
```

Note: These signals use `rank()` to convert raw indicator values into cross-sectional percentile ranks, making them comparable across stocks with different price levels and volatilities. The `- 0.5` centers the signal around zero for long/short portfolios.

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
│   ├── alpha_parser/         # Signal DSL package
│   │   ├── __init__.py       # Public API exports
│   │   ├── context.py        # Compute context and caching
│   │   ├── signal.py         # Base Signal class
│   │   ├── operators.py      # Arithmetic, comparison, logical, validity, math ops
│   │   ├── data.py           # Data field access
│   │   ├── primitives.py     # Returns, volatility, volume, adv
│   │   ├── timeseries.py     # Time-series operations
│   │   ├── crosssection.py   # Cross-sectional operations
│   │   ├── groups.py         # Group operations
│   │   ├── conditional.py    # Conditional (where) operations
│   │   ├── parser.py         # Expression parser
│   │   └── risk.py           # Multi-factor risk model
│   └── evaluation/           # Backtesting & evaluation
│       ├── __init__.py
│       ├── backtest.py       # Backtest engine
│       ├── metrics.py        # Performance metrics
│       └── quantile.py       # Quantile analysis
├── tests/
│   ├── conftest.py           # Test fixtures
│   ├── test_examples.py      # Example-based tests
│   ├── test_events.py        # Event/sparse data tests
│   ├── test_lazy.py          # LazyData tests
│   ├── test_operators.py     # Comprehensive operator tests
│   └── test_evaluation.py    # Backtest & evaluation tests
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

### Event-Based Time-Series (for sparse data like earnings)
- `ts_mean_events(signal, n)` - Mean over past N non-NaN values (events)
- `ts_std_events(signal, n)` - Std over past N non-NaN values (events)
- `ts_sum_events(signal, n)` - Sum over past N non-NaN values (events)
- `ts_count_events(signal, period)` - Count of non-NaN values in rolling window

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
- `group_demean(signal, 'group_name')` - Demean within groups (subtract group mean)
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

## Signal Evaluation & Backtesting

The evaluation module provides WorldQuant-style backtesting and quantile analysis:

### Basic Backtest

```python
from alpha_parser import alpha, Backtest

# Create a signal
signal = alpha("rank(returns(20)) - 0.5")

# Run backtest
bt = Backtest(signal)
result = bt.run(data)

# Print summary
print(result.summary())
```

**Output:**
```
==================================================
BACKTEST RESULTS
==================================================

Performance Metrics:
  Total Return:        12.34%
  Annual Return:        6.52%
  Annual Volatility:   15.21%
  Sharpe Ratio:         0.43
  Sortino Ratio:        0.61
  Calmar Ratio:         0.38
  Max Drawdown:        17.12%
  Return on GMV:        5.87%

Position Statistics:
  Avg Long Count:       25.0
  Avg Short Count:      25.0
  Avg Daily Turnover:   8.52%

Top Drawdowns:
  1. Drawdown(17.1% from 2022-01-03 to 2022-06-15, duration=163d, recovery=89d)
  2. Drawdown(9.8% from 2020-02-20 to 2020-03-23, duration=32d, recovery=45d)
==================================================
```

### Backtest with Transaction Costs

```python
# Add 10bps round-trip transaction cost
bt = Backtest(signal, transaction_cost=0.001)
result = bt.run(data)
```

### Quantile Analysis

Understand signal performance across quintiles:

```python
from alpha_parser import alpha, QuantileAnalysis

signal = alpha("rank(returns(60)) - 0.5")

# Run quintile analysis
qa = QuantileAnalysis(signal, n_quantiles=5)
result = qa.run(data)

print(result.summary())
```

**Output:**
```
==================================================
QUANTILE ANALYSIS
==================================================

Quantiles: 5
Days: 1000
Avg Stocks/Day: 50

Returns by Quantile (annualized):
  Q1:  -3.21%  Sharpe: -0.25  Hit:  48.2%
  Q2:   1.05%  Sharpe:  0.08  Hit:  50.1%
  Q3:   2.41%  Sharpe:  0.18  Hit:  51.3%
  Q4:   4.12%  Sharpe:  0.31  Hit:  52.8%
  Q5:   7.85%  Sharpe:  0.59  Hit:  54.6%

Long-Short Spread (Q5 - Q1):
  Annual Return:   11.06%
  Sharpe Ratio:     0.84

Rank IC: 0.032
Monotonic: Yes
==================================================
```

### Information Coefficient (IC) Analysis

```python
ic_stats = qa.ic_summary(data)
print(ic_stats)
```

**Output:**
```
Mean Rank IC       0.032
Std Rank IC        0.089
IC IR (Rank)       0.360
% Positive IC      0.584
```

### Walk-Forward Analysis

Test signal stability across time:

```python
# Rolling out-of-sample tests
wf_results = bt.run_walk_forward(
    data,
    train_period=252,  # 1 year training
    test_period=63,    # 1 quarter testing
)

print(wf_results)
```

### Available Metrics

The evaluation module provides these metrics:

- **sharpe_ratio** - Annualized Sharpe ratio
- **sortino_ratio** - Sortino ratio (downside deviation)
- **calmar_ratio** - Return / max drawdown
- **max_drawdown** - Maximum peak-to-trough decline
- **top_drawdowns** - List of largest drawdowns with dates
- **annualized_return** - Mean return × 252
- **annualized_volatility** - Std × sqrt(252)
- **return_on_gmv** - Total PnL / average gross exposure

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
