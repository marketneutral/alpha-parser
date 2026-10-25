# Concepts

Understanding qex's core abstractions.

## Signals

A **Signal** is a lazy computation tree. When you write:

```python
signal = qex("rank(returns(20))")
```

You're not computing anything yet. You're building a tree:

```
Rank
 └── Returns(period=20)        reads data['close']
```

The computation happens when you call `.evaluate(data)`.

## Operations

Operations fall into categories by how they transform data:

### Time-Series Operations

Operate **along rows** (across time) for each column (ticker) independently.

```python
ts_mean(x, 20)   # 20-day rolling mean
ts_std(x, 20)    # 20-day rolling std
delay(x, 5)      # Lag by 5 days
delta(x, 10)     # Change over 10 days
```

!!! note "NaN Warmup"
    Time-series operations produce NaN for the warmup period.
    `ts_mean(x, 20)` has NaN for the first 19 rows.

### Cross-Sectional Operations

Operate **across columns** (across tickers) for each row (day) independently.

```python
rank(x)                 # Percentile rank (0-1)
zscore(x)               # Z-score across tickers
demean(x)               # Subtract cross-sectional mean
regression_neut(x, b)   # Remove x's cross-sectional fit on b (e.g. market beta)
```

Which tickers they compare is the *cross-sectional universe*. In a book built with
`BookSettings(universe='member')` it is the day's members, and names outside it get NaN;
evaluated on its own, a signal compares every ticker with data. Time-series operations
always see each ticker's full history. See
[the cross-sectional universe](../reference/operations.md#the-cross-sectional-universe).

### Group Operations

Like cross-sectional, but within groups (sectors, industries, etc.).

```python
group_rank(x, 'sector')    # Rank within sector
group_demean(x, 'sector')  # Demean within sector
```

## Data Format

Data is a dict of DataFrames, one per field:

```python
data = {
    'close': pd.DataFrame(...),   # DatetimeIndex, ticker columns
    'volume': pd.DataFrame(...),
    'sector': pd.DataFrame(...),  # For group operations
    'market': pd.Series(...),     # A date-only field: broadcast to every ticker
}
```

The DataFrames should share the same index and columns. A field with one value per day,
such as a market index, can be a date-indexed `pd.Series`.
See [Data Format](../guides/data-format.md).

## Evaluation

```python
signal = qex("rank(returns(20))")

# Returns DataFrame with same shape as input
result = signal.evaluate(data)

# Or convert to portfolio weights
weights = signal.to_weights(data, normalize=True, long_only=False)
```

## Caching

Use `compute_context()` for efficiency:

```python
from qex import compute_context

with compute_context() as ctx:
    # Shared sub-expressions computed once
    s1 = qex("rank(returns(20))")
    s2 = qex("zscore(returns(20))")

    r1 = s1.evaluate(data)  # returns(20) computed
    r2 = s2.evaluate(data)  # returns(20) reused from cache

    print(f"Cached results: {len(ctx.cache)}")   # 3: returns(20), the rank, the z-score
```

## Variable Bindings

Use `let ... in` to avoid repetition:

```python
# Without let (signal expression repeated):
qex("rank(delta(returns(5)/volatility(10), 20)) * (returns(5)/volatility(10))")

# With let (define once, use multiple times):
qex("let s = returns(5)/volatility(10) in rank(delta(s, 20)) * s")
```

Multiple bindings:

```python
qex("""
    let mom = returns(60),
        vol = volatility(60),
        sharpe = mom / vol
    in rank(sharpe) * sign(mom)
""")
```

## Long/Short Portfolios

`rank()` returns values in [0, 1]. For long/short:

```python
# Roughly centred (mean 1/(2n) over n names)
signal = qex("rank(returns(20)) - 0.5")

# Exactly centred: longs and shorts equal
signal = qex("demean(rank(returns(20)))")
```

Then `.to_weights(data, normalize=True)` scales so absolute weights sum to 1. A book built
with `build_book` is demeaned (or group-demeaned) for you; see
[Backtesting](../guides/backtesting.md#book-settings-finding-alphas).

Equal long and short dollars do not make a book market-neutral: see
[Market Beta](../guides/market-beta.md).
