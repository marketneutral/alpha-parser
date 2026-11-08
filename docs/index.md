# qex

<div class="hero" markdown>

## Express quant signals in one line

```qex
rank(-returns(20) / volatility(60))
```

[Quick Start](python-examples/quickstart.md){ .md-button .md-button--primary }
[Operator Reference](language/reference.md){ .md-button }

</div>

qex is a language for quantitative trading signals. A signal is one readable expression, applied to a whole panel of
stock data at once.

Sector-neutral, risk-adjusted momentum:

```qex
group_demean(returns(60) / volatility(60), 'sector')
```

Short-term reversal, leaving out the most volatile stocks:

```qex
where(volatility(20) > 0.6, None, rank(-returns(5)))
```

A blend of the two, with names:

```qex
let mom = rank(returns(60)) - 0.5,
    rev = rank(-returns(5)) - 0.5
in 0.7 * mom + 0.3 * rev
```

## Where to find what

| Section | For |
|---------|-----|
| [Philosophy](philosophy.md) | why signals are written this way, and how to go from an idea to an expression |
| [Excel to qex](excel-to-qex.md) | one signal built in a spreadsheet and in qex, step by step, with the workbook |
| [Language](language/syntax.md) | qex itself, with no Python: syntax, how expressions are evaluated, missing values, groups, universes, look-ahead, and the [operator reference](language/reference.md) |
| [Examples](examples/index.md) | signals written in qex: momentum, mean reversion, pairs, events, market beta, technical indicators |
| [Python](python/installation.md) | installing qex, parsing and evaluating, books, backtests, diagnostics, the risk model, Dask |
| [Data](data/format.md) | the data qex reads: format, labels, universes, events, large datasets, adjusted prices, gaps |
| [Python Examples](python-examples/quickstart.md) | worked programs: the quick start, backtesting, checking a book's beta |

## In Python

```python
from qex import qex, Backtest

signal = qex("demean(rank(returns(20)))")
result = Backtest(signal, transaction_cost=0.001).run(data)    # data: a dict of DataFrames, see Data
print(result.summary())
```

## Installation

```bash
pip install git+https://github.com/marketneutral/alpha-parser.git
```

Optional extras and a development install are in [Installation](python/installation.md).
