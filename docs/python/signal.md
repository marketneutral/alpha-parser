# Signals

`qex()` returns a `Signal`: the expression's tree, not yet computed. It is computed when you call `.evaluate(data)`
or `.to_weights(data)`, and can be evaluated on any data in the [data format](../data/format.md).

## Basic Evaluation

```python
from qex import qex

signal = qex("rank(returns(20))")

# Evaluate with data
result = signal.evaluate(data)
```

The result is a DataFrame with the same dates and stocks as the data's `close`. An expression with no data in it,
such as `qex("2 * 3")`, evaluates to a number.

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

This clips negative weights to zero and normalizes the positive weights to sum to 1.

`to_weights` only scales the signal. For a book with a delay, a universe, neutralization and
a cap per name, use [`build_book`](books.md#book-settings).

## Caching: `compute_context`

Inside a `compute_context`, every part of every expression is computed once and reused: two signals that share
`returns(20)` compute it once, and so does an expression that uses a `let` name twice. `Backtest`, `build_book` and
`SignalGraph` open one for you; open your own when you evaluate several signals on the same data.

```python
from qex import compute_context

with compute_context() as ctx:
    s1 = qex("rank(returns(20))")
    s2 = qex("zscore(returns(20))")

    r1 = s1.evaluate(data)  # returns(20) computed
    r2 = s2.evaluate(data)  # returns(20) reused

    print(f"Cached keys: {list(ctx.cache.keys())}")
```

The cache is for one set of data, known by identity: the frames a dict holds (a new dict around the same frames is
the same data), or the object itself for a `LazyData`. Evaluating on other frames, even with the same shape and
dates, starts a fresh cache. A value changed in place inside a frame the context has seen is not detected (checking
values would read the whole panel at every node): after changing data in place, call `ctx.clear()` or open a new
context.

`get_context()` returns the active context, or `None`.

## Creating Custom Signals

A custom signal is usable from Python (not from qex text, whose operators are fixed). Subclass `Signal` and implement
`_compute()` and `_cache_key()`; the key must name everything the result depends on, since two signals with the same
key share a cached result:

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

    def _cache_key(self) -> tuple:
        return ('MySignal', self.child._cache_key(), self.param)
```

## API Reference

::: qex.signal.Signal
    options:
      members:
        - evaluate
        - to_weights
