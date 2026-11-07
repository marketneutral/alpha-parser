# Installation

## Requirements

- Python 3.10+
- numpy, pandas, scipy

## Install from Git

```bash
# Basic
pip install git+https://github.com/marketneutral/alpha-parser.git

# With all optional dependencies
pip install "qex[all] @ git+https://github.com/marketneutral/alpha-parser.git"
```

## Install for Development

```bash
git clone https://github.com/marketneutral/alpha-parser
cd alpha-parser
pip install -e ".[all]"
```

## Project layout

```
src/qex/                 the language
    parser.py            reads an expression into a tree of Signal nodes
    signal.py            the Signal base class: evaluate, to_weights
    context.py           compute_context: the shared cache
    data.py              fields, close() and friends, market(), LazyData
    primitives.py        returns, volatility, volume, adv; price_returns, forward_returns
    operators.py         arithmetic, comparisons and logic, math, is_valid, is_finite, None
    conditional.py       where, coalesce
    timeseries.py        time-series operators, events, fill_forward, latch
    crosssection.py      cross-sectional operators, regression_neut, pca_factor
    groups.py            group operators and labels
    universe.py          the cross-sectional universe
    kinds.py             numbers versus labels, checked when an expression is read
    missing.py           explain_missing
    adjust.py            the look-ahead check for adjusted data
    distributed.py       SignalGraph and Dask
    risk.py              the factor risk model
src/evaluation/          books and backtests
    book.py              BookSettings, build_book, hold_untradable
    backtest.py          Backtest, BacktestResult
    quantile.py          QuantileAnalysis
    metrics.py           Sharpe, drawdowns, turnover, fitness, margin
tests/                   pytest; test_docs.py checks the documentation's examples
docs/                    this site (mkdocs)
examples/                cookbook.py, risk_model_example.py
data/fetch_fmp.py        the FMP data fetcher
```

## Optional Dependencies

| Group | Packages | Use Case |
|-------|----------|----------|
| `dev` | pytest, pyarrow | Running tests |
| `data` | requests, python-dotenv, pyarrow | FMP data fetcher |
| `distributed` | dask, distributed, graphviz | Dask evaluation, graph pictures |
| `risk` | statsmodels | Factor risk model |
| `docs` | mkdocs, mkdocs-material, mkdocstrings | Building this site |
| `all` | Everything above | Full installation |

Install specific groups (from cloned repo):

```bash
pip install -e ".[dev]"      # Just testing
pip install -e ".[risk]"     # With risk model
pip install -e ".[all]"      # Everything
```

## Verify Installation

```python
from qex import qex

signal = qex("rank(returns(20))")
print(type(signal).__name__)  # Rank
```
