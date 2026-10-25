# Quick Start

Get up and running in 5 minutes.

## 1. Create Sample Data

qex expects data as a dict of DataFrames with DatetimeIndex and ticker columns:

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
              AAPL    MSFT    GOOG    AMZN
2020-12-16  0.0901 -0.0524 -0.0807  0.0294
2020-12-17  0.0794 -0.0154 -0.0536  0.0348
```

## 3. Rank Cross-Sectionally

```python
# Rank stocks each day (0 = lowest, 1 = highest)
signal = qex("rank(returns(20))")
result = signal.evaluate(data)
```

## 4. Convert to Weights

```python
# Demean for long/short (longs and shorts equal), then scale so |weights| sum to 1
signal = qex("demean(rank(returns(20)))")
weights = signal.to_weights(data, normalize=True)

print(weights.tail(2))
```

Output:
```
             AAPL   MSFT   GOOG   AMZN
2020-12-16  0.375 -0.125 -0.375  0.125
2020-12-17  0.375 -0.125 -0.375  0.125
```

`rank()` is between 0 and 1, so `rank(x) - 0.5` is only roughly centred: its mean is
`1 / (2n)` over `n` names, which matters for small universes. `demean()` centres exactly.

## 5. Run a Backtest

```python
from qex import Backtest

signal = qex("demean(rank(returns(20)))")
bt = Backtest(signal, transaction_cost=0.001)
result = bt.run(data)

print(result.summary())
```

The summary reports performance (total and annual return, volatility, Sharpe, Sortino,
Calmar, maximum drawdown, return on gross), position counts and turnover, and the deepest
drawdowns. On this random data there is nothing to find, so expect a Sharpe near zero.

For books built the way *Finding Alphas* and WorldQuant BRAIN simulate them (delay,
universe, neutralization, truncation), see [Backtesting](../guides/backtesting.md#book-settings-finding-alphas).

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
- [Backtesting](../guides/backtesting.md) - Books, costs and statistics
- [Market Beta](../guides/market-beta.md) - Measuring and removing market exposure
- [Operations Reference](../reference/operations.md) - All available operations
