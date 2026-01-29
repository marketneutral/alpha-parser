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
- `rank(signal)` - Cross-sectional percentile rank (0-1)
- `quantile(signal, buckets)` - Assign to quantile buckets (1=lowest, n=highest)
- `zscore(signal)` - Cross-sectional z-score
- `demean(signal)` - Subtract cross-sectional mean

### Conditional
- `where(condition, if_true, if_false)` - Ternary operator

### Group Operations
- `group_rank(signal, 'group_name')` - Rank within groups
- `group_demean(signal, 'group_name')` - Demean within groups
- `group_neutralize(signal, 'group_name')` - Neutralize to groups
- `group_count_valid(signal, 'group_name', window)` - Count non-NaN within group over window

### Validity Operations
- `is_valid(signal)` - Returns 1 where not NaN, 0 otherwise

## License

Apache 2.0 - see [LICENSE](LICENSE) for details.
