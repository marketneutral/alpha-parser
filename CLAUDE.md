# Claude Project Guide

## Project Overview

qex is a DSL for defining quantitative trading signals. It parses string expressions like `rank(-returns(20) / volatility(60))` into executable signal trees.

## Quick Commands

```bash
# Setup
uv venv && source .venv/bin/activate && uv pip install -r requirements.txt

# Run tests
PYTHONPATH=src pytest tests/ -v
```

## Architecture

- `src/qex/` - Signal DSL package
  - `signal.py` - Base `Signal` class with `evaluate()` and `to_weights()` methods
  - `parser.py` - `QexParser` converts string expressions to Signal trees
  - `operators.py` - Arithmetic (`Add`, `Sub`, `Mul`, `Div`), comparison, validity (`is_valid`), and math ops (`log`, `abs`, `sign`, `sqrt`, `power`, `max`, `min`)
  - `timeseries.py` - Rolling operations (`ts_mean`, `ts_std`, `delay`, `fill_forward`, `ts_corr`, `ts_cov`, `ewma`, `ts_argmax`, `ts_argmin`, `ts_skew`, `ts_kurt`, `decay_linear`)
  - `crosssection.py` - Cross-sectional operations (`rank`, `zscore`, `demean`, `quantile`, `winsorize`, `scale`, `truncate`)
  - `groups.py` - Group data access (`group`) and group-neutral operations (`group_rank`, `group_demean`, `group_std`, `group_count_valid`)
  - `primitives.py` - Basic signals (`returns`, `volatility`, `volume`, `adv`) and `price_returns`/`forward_returns`, the one rule for turning prices into returns
  - `data.py` - Data field access (`close`, `open`, `high`, `low`, `field`) and `LazyData`
  - `conditional.py` - Conditional logic (`where`)
  - `context.py` - `compute_context()` provides shared caching across signals
  - `distributed.py` - `SignalGraph` for DAG construction, `DistributedSignalEngine` for Dask
  - `risk.py` - Multi-factor risk model (`FactorRiskModel`, `FactorDefinition`)
- `src/evaluation/` - Backtesting and evaluation module
  - `backtest.py` - `Backtest` class with `BacktestResult`
  - `metrics.py` - Performance metrics (`sharpe_ratio`, `max_drawdown`, `top_drawdowns`, etc.)
  - `quantile.py` - `QuantileAnalysis` with `QuantileResult`

## Key Patterns

- All signals inherit from `Signal` ABC and implement `_compute()` and `_cache_key()`
- Use `qex("expression")` to parse strings into Signal objects (or `alpha()` for backwards compatibility)
- Wrap multiple evaluations in `with compute_context():` for cache sharing; the cache knows its data by identity (the frames a dict holds), so a value changed in place needs `ctx.clear()`
- Data is passed as `Dict[str, pd.DataFrame]` with keys like `'close'`, `'volume'`
- Use `LazyData` wrapper for large datasets - fields are loaded on demand
- Use `LazyData(data, descriptions={...})` for self-documenting data - call `data.describe()` to inspect
- Never call `pct_change` on prices: use `price_returns`/`forward_returns` (NaN on days with no price, bridged from the last valid price up to `MAX_PRICE_GAP` rows; identical on pandas 2 and 3). `Backtest` holds untradable names' weights via `hold_untradable`, and starts on the first day the book holds a position (since 0.12.2)
- Sparse data (e.g., earnings) uses NaN for missing values - use `fill_forward()` and `is_valid()`
- A universe is a mask, a field name, or a qex expression (nonzero = member), evaluated over all stocks with no universe active (`universe.py: universe_mask`); `latch(on, off)` is hysteresis (since 0.9.0)
- Expressions give numbers or labels (`group`, `group_cross`, strings); labels only compare (`==`, `!=`), group, or pass through `coalesce`/`where`, else the parse is refused (`kinds.py`)
- Group operators take a group's name or any labels expression (`quantile(field('mcap'), 5)`, `group_cross('sector', ...)`); `group_quantile` sorts within groups (since 0.8.0)
- Comparisons and logic are three-valued (since 0.7.0): a comparison with a missing side is missing, `&` `|` `~` follow Kleene (false & missing = 0, true | missing = 1), and `where` with a missing condition is missing; `coalesce(x, y)` (x where it has a value, else y) says what missing should mean, as `coalesce(field('price') > 8, 0)`
- `None` is the literal for "no value here, on purpose" (`where(cond, x, None)`); `0` is a value, and a position once a book demeans the signal. `explain_missing(signal, data, universe=...)` (`missing.py`) traces each missing cell to its reason and the node where it began
- Every group operator works on one day's cross-section; `group_std(x, g)` is the day's spread within each group (since 0.12.0; the old `group_std(x, g, n)` window form is refused). `group_count_valid(x, g, w)` alone still counts over a window
- `pca_factor(x, window, k[, refit=21])` is each stock's beta to the k-th statistical factor (correlation PCA across stocks, refit every `refit` rows and held, components tracked across refits); its input is an internal `PcaBasis` node (the decomposition, value a `PcaFits` list, not a frame), shared by every pca_factor on the same x, window and refit; `missing.py`, `adjust.py` and `distributed.py` special-case it
- `ts_alpha`/`ts_resid` are the intercept and today's residual of a rolling time-series fit per stock, `(y, x1, ..., xk, period[, min_periods])`: one regressor uses `ts_beta`'s fit on the paired days, several are one joint fit (`timeseries.py: _multi_fit`, at least k + 2 days); the `_shrunk` versions (one regressor) use `ts_beta_shrunk`'s slope, and all three shrunk fits take an optional group prior last, `(y, x, period[, min_periods][, groups])`; without groups `TsBetaShrunk`'s cache key is unchanged (since 0.13.0)
- `regression_neut(y, x1, x2, ...)` with several regressors is one joint least-squares fit; with one regressor its cache key is unchanged (downstream tools hash expressions by their cache key, so keys must stay stable)
- Group data can be accessed as `data['sector']` or `data['groups']['sector']`

## Available Operations

### Primitives
`returns(period)`, `volatility(period)`, `volume(period)`, `adv(period)`

### Math
`log(x)`, `exp(x)`, `abs(x)`, `sign(x)`, `sqrt(x)`, `power(x, n)`, `max(x, y)`, `min(x, y)`

### Time-Series
`ts_mean`, `ts_std`, `ts_sum`, `ts_max`, `ts_min`, `ts_var`, `ts_quantile`, `delay`, `delta`, `ts_rank`, `fill_forward`, `ts_corr`, `ts_cov`, `ewma`, `ewma_var`, `ewma_cov`, `ts_beta`, `ts_beta_shrunk`, `ts_beta_ewma`, `ts_alpha`, `ts_alpha_shrunk`, `ts_resid`, `ts_resid_shrunk`, `ts_argmax`, `ts_argmin`, `ts_skew`, `ts_kurt`, `decay_linear`

### Event-Based (for sparse data)
`ts_mean_events`, `ts_std_events`, `ts_sum_events`, `ts_count_events`

### Cross-Sectional
`rank`, `zscore`, `demean`, `quantile`, `winsorize`, `scale`, `truncate`, `regression_neut`, `pca_factor`

### Group
`group`, `group_rank`, `group_demean`, `group_std`, `group_sum`, `group_count_valid`

### Sector Filtering
Use `group()` with `where()` to filter signals by sector membership:

```python
# Price-to-book isn't meaningful for Financials - leave them out (None: no value, no position)
qex("where(group('sector')=='Financials', None, field('price_to_book'))")

# Only trade Technology stocks
qex("where(group('sector')=='Technology', rank(returns(20)), None)")
```

### Calendar
`day_of_week`, `day_of_month`, `month_of_year`

### Conditional/Validity
`where`, `is_valid`, `is_finite`

### Variable Bindings
Use `let ... in` syntax to define variables and avoid repeating complex expressions:

```python
# Single binding
qex("let s = returns(20) in rank(delta(s, 10)) * s")

# Multiple comma-separated bindings
qex("let mom = returns(20), vol = volatility(60) in rank(mom / vol) * sign(mom)")

# Dependent bindings (later can reference earlier)
qex("let mom = returns(20), sharpe = mom / volatility(60) in rank(sharpe) * sign(mom)")
```

## Risk Model

```python
from qex import FactorRiskModel, DEFAULT_STYLE_FACTORS, PRICE_ONLY_FACTORS

# With fundamental data (market_cap, book_to_price); style_factors=None means DEFAULT_STYLE_FACTORS
model = FactorRiskModel(style_factors=DEFAULT_STYLE_FACTORS)

# Price-only (no fundamentals)
model = FactorRiskModel(style_factors=PRICE_ONLY_FACTORS)

# fit() returns the model; read factor_returns, factor_covariance, specific_risk, portfolio_risk(w)
model.fit(data)
```

## Evaluation & Backtesting

```python
from qex import qex, Backtest, QuantileAnalysis

# Backtest a signal
signal = qex("rank(returns(20)) - 0.5")
bt = Backtest(signal, transaction_cost=0.001)
result = bt.run(data)
print(result.summary())

# Quantile analysis
qa = QuantileAnalysis(signal, n_quantiles=5)
qa_result = qa.run(data)
print(qa_result.summary())

# IC analysis
ic_stats = qa.ic_summary(data)
```

Key metrics available:
- `sharpe_ratio`, `sortino_ratio`, `calmar_ratio`
- `max_drawdown`, `top_drawdowns`
- `annualized_return`, `annualized_volatility`
- `return_on_gmv` (return on gross market value)

## Distributed Evaluation

For evaluating thousands of signals with shared sub-expressions:

```python
from qex import qex, SignalGraph, DistributedSignalEngine

# Build unified DAG - shared sub-expressions computed once
graph = SignalGraph()
graph.add_signal(qex("rank(returns(60))"), name="momentum")
graph.add_signal(qex("rank(returns(60) / volatility(60))"), name="sharpe")

# Visualize the graph (requires graphviz)
graph.visualize("signals.png")

# Compute all signals
results = graph.compute(data)

# For Dask distributed clusters
from dask.distributed import Client
client = Client("scheduler:8786")
engine = DistributedSignalEngine(client)
engine.load_data(data)  # Scatter once to all workers
results = engine.evaluate(["rank(returns(60))", "rank(-returns(5))"])
```

## Testing

Tests use pytest fixtures from `tests/conftest.py`. The `sample_data` fixture provides synthetic price/volume data with 8 tickers over 4 years.

- `test_examples.py` - Core functionality tests
- `test_events.py` - Sparse/event data tests (PEAD-style alphas)
- `test_lazy.py` - LazyData on-demand loading tests
- `test_operators.py` - Comprehensive tests for all operators
- `test_evaluation.py` - Backtest and quantile analysis tests
- `test_missing_prices.py` - Returns across halts/delistings, held weights in backtests, forward returns
- `test_pca_factor.py` - pca_factor: planted factors, the beta definition, refits, shared decomposition, tracking, coverage, universe
- `test_operators_012.py` - exp, is_finite, ts_quantile, regression_neut with several regressors, one-day group_std
- `test_parser_0121.py` - Comments to end of line, clear errors for unsupported syntax, constant-argument checks (`parser.py: _ARG_RULES`)
- `test_ts_alpha_resid.py` - ts_alpha, ts_resid, their shrunk versions and the group prior, against direct regressions
- `test_backtest_warmup.py` - Backtest starts on the first day the book holds a position (0.12.2)
- `test_excel.py` - The Excel to qex workbook: its formulas are current and its values match qex
- `test_docs.py` - The documentation: every operator documented, every qex block parses, Python blocks compile, worked pages run
- `test_bugfixes_052.py` - Regression tests for the 0.5.2 fixes (paired betas, infinities in cross-sections, risk model, ts_rank, math of numbers)
- `test_distributed.py` - Distributed evaluation and graph visualization tests
- `test_integration.py` - Integration tests with real FMP data (requires data fetch)

## Documentation

`docs/` (mkdocs, built with `mkdocs build --strict` in CI on every PR) has seven sections: Philosophy, Language (pure qex,
no Python; `language/reference.md` lists every operator in one table, and `language/ops/*.md` give each family's
operators an entry with worked examples on toy data: ```` ```toy ```` blocks that `docs/hooks/toy.py` evaluates at build time), Examples (pure qex), Python, Data, and Python
Examples, and last Excel to qex (a friendly walkthrough: one signal as a workbook and as qex;
`docs/excel/build_workbook.py --recalc` rebuilds `excel-to-qex.xlsx`, and `tests/test_excel.py` checks it against
qex). qex code goes in ```` ```qex ```` blocks, Python in ```` ```python ```` blocks. `tests/test_docs.py` checks that
every operator has a reference row (two-column `| \`name(args)\` | description |`, a stable form that tools can parse), that every qex block parses, that every Python block compiles and its `qex("...")` strings parse, and that the
Quick Start and Backtesting the Examples pages run. A new operator needs a reference row and an entry (`### name`) on its family's page.

## Data Fetching

```bash
# Fetch S&P 500 data from FMP
python data/fetch_fmp.py

# Specific tickers
python data/fetch_fmp.py --tickers AAPL MSFT GOOG
```

Requires FMP API key in `data/.env`.
