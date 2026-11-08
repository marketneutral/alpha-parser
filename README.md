# qex

qex is a language for quantitative trading signals. A signal is one readable expression, applied to a whole panel of
stock data at once:

```python
from qex import qex, Backtest

signal = qex("group_demean(returns(60) / volatility(60), 'sector')")    # sector-neutral, risk-adjusted momentum
result = Backtest(signal, transaction_cost=0.001).run(data)             # data: a dict of date x stock DataFrames
print(result.summary())
```

**Documentation: <https://marketneutral.github.io/alpha-parser/>**

| Section | For |
|---------|-----|
| [Philosophy](https://marketneutral.github.io/alpha-parser/philosophy/) | why signals are written this way |
| [Language](https://marketneutral.github.io/alpha-parser/language/syntax/) | the language: syntax, evaluation, missing values, groups, universes, look-ahead, and the [operator reference](https://marketneutral.github.io/alpha-parser/language/reference/) |
| [Examples](https://marketneutral.github.io/alpha-parser/examples/) | signals written in qex |
| [Python](https://marketneutral.github.io/alpha-parser/python/installation/) | parsing, books, backtests, diagnostics, the risk model, Dask |
| [Data](https://marketneutral.github.io/alpha-parser/data/format/) | the data qex reads |
| [Python Examples](https://marketneutral.github.io/alpha-parser/python-examples/quickstart/) | worked programs, starting with the quick start |
| [Excel to qex](https://marketneutral.github.io/alpha-parser/excel-to-qex/) | a friendly walkthrough for spreadsheet users: one signal built in Excel, then written in qex |

The same pages are in [`docs/`](docs/), and what changed in each version is in [`CHANGELOG.md`](CHANGELOG.md).

## Install

```bash
pip install git+https://github.com/marketneutral/alpha-parser.git

# for development, with every extra (tests, data fetching, risk model, Dask, docs)
git clone https://github.com/marketneutral/alpha-parser
cd alpha-parser
pip install -e ".[all]"
```

Python 3.10 or later.

## Tests and docs

```bash
PYTHONPATH=src pytest tests/ -m "not integration"     # unit tests, including the docs' examples
mkdocs serve                                           # the documentation site, locally
```

The integration tests need market data from FMP ([Getting Data](https://marketneutral.github.io/alpha-parser/data/getting-data/))
and skip themselves without it.

## License

Apache 2.0, see [LICENSE](LICENSE).
