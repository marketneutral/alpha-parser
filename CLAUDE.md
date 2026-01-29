# Claude Project Guide

## Project Overview

Alpha Parser is a DSL for defining quantitative trading signals. It parses string expressions like `rank(-returns(20) / volatility(60))` into executable signal trees.

## Quick Commands

```bash
# Setup
uv venv && source .venv/bin/activate && uv pip install -r requirements.txt

# Run tests
PYTHONPATH=src pytest tests/ -v
```

## Architecture

- `src/alpha_parser/` - Main package
  - `signal.py` - Base `Signal` class with `evaluate()` and `to_weights()` methods
  - `parser.py` - `AlphaParser` converts string expressions to Signal trees
  - `operators.py` - Arithmetic (`Add`, `Sub`, `Mul`, `Div`), comparison, validity (`is_valid`), and math ops (`log`, `abs`, `sign`, `sqrt`, `power`, `max`, `min`)
  - `timeseries.py` - Rolling operations (`ts_mean`, `ts_std`, `delay`, `fill_forward`, `ts_corr`, `ts_cov`, `ewma`, `ts_argmax`, `ts_argmin`, `ts_skew`, `ts_kurt`, `decay_linear`)
  - `crosssection.py` - Cross-sectional operations (`rank`, `zscore`, `demean`, `quantile`, `winsorize`, `scale`, `truncate`)
  - `groups.py` - Group-neutral operations (`group_rank`, `group_demean`, `group_neutralize`, `group_count_valid`)
  - `primitives.py` - Basic signals (`returns`, `volatility`, `volume`, `adv`)
  - `data.py` - Data field access (`close`, `open`, `high`, `low`, `field`) and `LazyData`
  - `conditional.py` - Conditional logic (`where`)
  - `context.py` - `compute_context()` provides shared caching across signals
  - `risk.py` - Multi-factor risk model (`FactorRiskModel`, `FactorDefinition`)

## Key Patterns

- All signals inherit from `Signal` ABC and implement `_compute()` and `_cache_key()`
- Use `alpha("expression")` to parse strings into Signal objects
- Wrap multiple evaluations in `with compute_context():` for cache sharing
- Data is passed as `Dict[str, pd.DataFrame]` with keys like `'close'`, `'volume'`
- Use `LazyData` wrapper for large datasets - fields are loaded on demand
- Sparse data (e.g., earnings) uses NaN for missing values - use `fill_forward()` and `is_valid()`
- Group data can be accessed as `data['sector']` or `data['groups']['sector']`

## Available Operations

### Primitives
`returns(period)`, `volatility(period)`, `volume(period)`, `adv(period)`

### Math
`log(x)`, `abs(x)`, `sign(x)`, `sqrt(x)`, `power(x, n)`, `max(x, y)`, `min(x, y)`

### Time-Series
`ts_mean`, `ts_std`, `ts_sum`, `ts_max`, `ts_min`, `delay`, `delta`, `ts_rank`, `fill_forward`, `ts_corr`, `ts_cov`, `ewma`, `ts_argmax`, `ts_argmin`, `ts_skew`, `ts_kurt`, `decay_linear`

### Event-Based (for sparse data)
`ts_mean_events`, `ts_std_events`, `ts_sum_events`, `ts_count_events`

### Cross-Sectional
`rank`, `zscore`, `demean`, `quantile`, `winsorize`, `scale`, `truncate`

### Group
`group_rank`, `group_demean`, `group_count_valid`

### Conditional/Validity
`where`, `is_valid`

## Risk Model

```python
from alpha_parser import FactorRiskModel, DEFAULT_STYLE_FACTORS, PRICE_ONLY_FACTORS

# With fundamental data
risk_model = FactorRiskModel(factors=DEFAULT_STYLE_FACTORS)

# Price-only (no fundamentals)
risk_model = FactorRiskModel(factors=PRICE_ONLY_FACTORS)

# Fit and get results
results = risk_model.fit(data)
```

## Testing

Tests use pytest fixtures from `tests/conftest.py`. The `sample_data` fixture provides synthetic price/volume data with 8 tickers over 4 years.

- `test_examples.py` - Core functionality tests
- `test_events.py` - Sparse/event data tests (PEAD-style alphas)
- `test_lazy.py` - LazyData on-demand loading tests
- `test_operators.py` - Comprehensive tests for all operators (73 tests)

## Data Fetching

```bash
# Fetch S&P 500 data from FMP
python data/fetch_fmp.py

# Specific tickers
python data/fetch_fmp.py --tickers AAPL MSFT GOOG
```

Requires FMP API key in `data/.env`.
