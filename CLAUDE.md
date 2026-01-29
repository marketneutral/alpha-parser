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
  - `operators.py` - Arithmetic (`Add`, `Sub`, `Mul`, `Div`), comparison, and validity ops (`is_valid`)
  - `timeseries.py` - Rolling operations (`ts_mean`, `ts_std`, `delay`, `fill_forward`, etc.)
  - `crosssection.py` - Cross-sectional operations (`rank`, `zscore`, `demean`)
  - `groups.py` - Group-neutral operations (`group_rank`, `group_demean`, `group_count_valid`)
  - `context.py` - `compute_context()` provides shared caching across signals

## Key Patterns

- All signals inherit from `Signal` ABC and implement `_compute()` and `_cache_key()`
- Use `alpha("expression")` to parse strings into Signal objects
- Wrap multiple evaluations in `with compute_context():` for cache sharing
- Data is passed as `Dict[str, pd.DataFrame]` with keys like `'close'`, `'volume'`
- Sparse data (e.g., earnings) uses NaN for missing values - use `fill_forward()` and `is_valid()`

## Testing

Tests use pytest fixtures from `tests/conftest.py`. The `sample_data` fixture provides synthetic price/volume data with 8 tickers over 4 years.

- `test_examples.py` - Core functionality tests
- `test_events.py` - Sparse/event data tests (PEAD-style alphas)
