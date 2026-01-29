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

## Project Structure

```
alpha-parser/
├── src/
│   └── alpha_parser/
│       ├── __init__.py       # Public API exports
│       ├── context.py        # Compute context and caching
│       ├── signal.py         # Base Signal class
│       ├── operators.py      # Arithmetic, comparison, logical ops
│       ├── data.py           # Data field access
│       ├── primitives.py     # Returns, volatility, volume
│       ├── timeseries.py     # Time-series operations
│       ├── crosssection.py   # Cross-sectional operations
│       ├── groups.py         # Group operations
│       ├── conditional.py    # Conditional (where) operations
│       └── parser.py         # Expression parser
├── tests/
│   ├── conftest.py           # Test fixtures
│   └── test_examples.py      # Example-based tests
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

### Cross-Sectional Operations
- `rank(signal)` - Cross-sectional percentile rank
- `zscore(signal)` - Cross-sectional z-score
- `demean(signal)` - Subtract cross-sectional mean

### Conditional
- `where(condition, if_true, if_false)` - Ternary operator

### Group Operations
- `group_rank(signal, 'group_name')` - Rank within groups
- `group_demean(signal, 'group_name')` - Demean within groups
- `group_neutralize(signal, 'group_name')` - Neutralize to groups

## License

Apache 2.0 - see [LICENSE](LICENSE) for details.
