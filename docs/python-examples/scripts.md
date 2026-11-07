# Example Scripts

Two runnable scripts live in the repository's `examples/` folder.

## `examples/cookbook.py`

A tour of qex in nine sections, each printing what it computes on synthetic data: data setup, momentum, mean
reversion, pairs trading, factor-neutral signals, event-driven signals, risk management patterns, `let` composition,
and common mistakes. It needs no data and no extras:

```bash
PYTHONPATH=src python examples/cookbook.py
```

## `examples/risk_model_example.py`

Fits the [risk model](../python/risk-model.md) to real data and prints factor returns, the factor covariance and
correlation, specific risk, and a portfolio's risk decomposition. It reads parquet files written by the FMP fetcher,
including `market_cap` and `book_to_price` for the default style factors ([Getting Data](../data/getting-data.md)),
and needs the `risk` extra:

```bash
pip install -e ".[risk,data]"
python data/fetch_fmp.py --tickers AAPL MSFT GOOG AMZN META NVDA TSLA JPM BAC WMT
PYTHONPATH=src python examples/risk_model_example.py
```

Without the data it says so and stops.
