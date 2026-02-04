# Quick Start

Get up and running in 5 minutes.

## 1. Create Sample Data

Qex expects data as a dict of DataFrames with DatetimeIndex and ticker columns:

```python
import pandas as pd
import numpy as np

# Generate sample data
dates = pd.date_range('2020-01-01', periods=252, freq='B')
tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN']

np.random.seed(42)
returns = np.random.randn(252, 4) * 0.02
prices = 100 * np.exp(returns.cumsum(axis=0))

data = {
    'close': pd.DataFrame(prices, index=dates, columns=tickers),
    'volume': pd.DataFrame(
        np.random.randint(1_000_000, 10_000_000, (252, 4)),
        index=dates, columns=tickers
    ),
}
```

## 2. Parse a Signal

```python
from qex import qex

# Simple momentum signal
signal = qex("returns(20)")
result = signal.evaluate(data)

print(result.tail())
```

Output:
```
            AAPL      MSFT      GOOG      AMZN
2020-12-24  0.0312   -0.0187    0.0456   -0.0234
2020-12-28  0.0289   -0.0145    0.0512   -0.0189
...
```

## 3. Rank Cross-Sectionally

```python
# Rank stocks each day (0 = lowest, 1 = highest)
signal = qex("rank(returns(20))")
result = signal.evaluate(data)
```

## 4. Convert to Weights

```python
# Center for long/short, normalize to sum to 1
signal = qex("rank(returns(20)) - 0.5")
weights = signal.to_weights(data, normalize=True)

print(weights.tail())
```

Output:
```
            AAPL      MSFT      GOOG      AMZN
2020-12-28 -0.167   -0.333    0.333     0.167
...
```

## 5. Run a Backtest

```python
from qex import Backtest

signal = qex("rank(returns(20)) - 0.5")
bt = Backtest(signal, transaction_cost=0.001)
result = bt.run(data)

print(result.summary())
```

Output:
```
==================================================
BACKTEST RESULTS
==================================================

Performance Metrics:
  Total Return:        8.42%
  Annual Return:       8.42%
  Annual Volatility:  12.31%
  Sharpe Ratio:        0.68
  Max Drawdown:        9.21%
==================================================
```

## 6. Use Compute Context for Caching

When evaluating multiple signals with shared components:

```python
from qex import qex, compute_context

with compute_context():
    # returns(20) computed once, reused
    signal1 = qex("rank(returns(20))")
    signal2 = qex("zscore(returns(20))")

    result1 = signal1.evaluate(data)
    result2 = signal2.evaluate(data)
```

## Next Steps

- [Concepts](concepts.md) - Understand signals, operations, and evaluation
- [Data Format](../guides/data-format.md) - Detailed data requirements
- [Operations Reference](../reference/operations.md) - All available operations
