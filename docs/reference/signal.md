# Signal Class

The base Signal class and evaluation methods.

## Overview

All qex operations produce `Signal` objects. Signals are lazy - they build a computation tree that's evaluated when you call `.evaluate()` or `.to_weights()`.

## Basic Evaluation

```python
from qex import qex

signal = qex("rank(returns(20))")

# Evaluate with data
result = signal.evaluate(data)
```

The result is a DataFrame with the same shape as your input data.

## Portfolio Weights

```python
weights = signal.to_weights(
    data,
    normalize=True,    # Scale so |weights| sum to 1
    long_only=False,   # Allow negative weights
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | dict | required | Data dictionary |
| `normalize` | bool | `True` | Normalize weights to sum to 1 |
| `long_only` | bool | `False` | Clip negative weights to 0 |

### Long-Only Mode

```python
weights = signal.to_weights(data, long_only=True)
```

This clips negative weights to zero and normalizes the positive weights.

## Caching

Signals implement caching through `_cache_key()`:

```python
from qex import compute_context

with compute_context() as ctx:
    s1 = qex("rank(returns(20))")
    s2 = qex("zscore(returns(20))")

    r1 = s1.evaluate(data)  # returns(20) computed
    r2 = s2.evaluate(data)  # returns(20) reused

    print(f"Cached keys: {list(ctx.cache.keys())}")
```

## Creating Custom Signals

Subclass `Signal` and implement `_compute()` and `_cache_key()`:

```python
from qex import Signal
import pandas as pd

class MySignal(Signal):
    def __init__(self, child: Signal, param: int):
        self.child = child
        self.param = param

    def _compute(self, data: dict) -> pd.DataFrame:
        child_result = self.child.evaluate(data)
        return child_result * self.param

    def _cache_key(self) -> str:
        return f"MySignal({self.child._cache_key()}, {self.param})"
```

## API Reference

::: qex.signal.Signal
    options:
      members:
        - evaluate
        - to_weights
