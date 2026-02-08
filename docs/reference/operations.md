# All Operations

Complete reference of qex operations.

## Data Access

| Operation | Description |
|-----------|-------------|
| `close()` | Closing prices |
| `open()` | Opening prices |
| `high()` | Daily highs |
| `low()` | Daily lows |
| `field('name')` | Access any named field |

```python
qex("close()")
qex("field('earnings_surprise')")
```

## Primitives

| Operation | Description |
|-----------|-------------|
| `returns(period)` | Price returns over period |
| `volatility(period)` | Rolling annualized volatility |
| `volume(period)` | Rolling average volume |
| `adv(period)` | Average dollar volume (price × volume) |

```python
qex("returns(20)")         # 20-day returns
qex("volatility(60)")      # 60-day annualized vol
qex("volume(5) / adv(20)") # Volume ratio
```

## Math Operations

| Operation | Description |
|-----------|-------------|
| `log(x)` | Natural logarithm |
| `abs(x)` | Absolute value |
| `sign(x)` | Sign (-1, 0, or 1) |
| `sqrt(x)` | Square root |
| `power(x, n)` | Raise to power n |
| `max(x, y)` | Element-wise maximum |
| `min(x, y)` | Element-wise minimum |

```python
qex("log(close())")
qex("sign(returns(20)) * sqrt(abs(returns(20)))")
qex("max(returns(5), 0)")  # Positive returns only
```

## Time-Series Operations

Operate along time axis for each ticker independently.

### Rolling Statistics

| Operation | Description |
|-----------|-------------|
| `ts_mean(x, period)` | Rolling mean |
| `ts_std(x, period)` | Rolling standard deviation |
| `ts_sum(x, period)` | Rolling sum |
| `ts_max(x, period)` | Rolling maximum |
| `ts_min(x, period)` | Rolling minimum |
| `ts_var(x, period)` | Rolling variance |

```python
qex("ts_mean(returns(1), 20)")  # 20-day average return
qex("close() / ts_max(close(), 252)")  # Distance from 52-week high
```

### Lag and Difference

| Operation | Description |
|-----------|-------------|
| `delay(x, period)` | Lag/shift by N periods |
| `delta(x, period)` | Difference from N periods ago |

```python
qex("delay(close(), 1)")  # Yesterday's close
qex("delta(returns(20), 5)")  # Change in momentum
```

### Ranking

| Operation | Description |
|-----------|-------------|
| `ts_rank(x, period)` | Percentile rank within rolling window |

```python
qex("ts_rank(volume(1), 20)")  # Where does today's volume rank vs last 20 days?
```

### Correlation and Covariance

| Operation | Description |
|-----------|-------------|
| `ts_corr(x, y, period)` | Rolling correlation |
| `ts_cov(x, y, period)` | Rolling covariance |
| `ts_beta(y, x, period)` | Rolling beta (cov/var) |

```python
qex("ts_corr(returns(1), volume(1), 60)")  # Return-volume correlation
qex("ts_beta(returns(1), field('market'), 60)")  # Market beta
```

### Exponential Weighted

| Operation | Description |
|-----------|-------------|
| `ewma(x, halflife)` | Exponentially weighted moving average |
| `ewma_var(x, halflife)` | EWMA variance |
| `ewma_cov(x, y, halflife)` | EWMA covariance |
| `ts_beta_ewma(y, x, halflife)` | EWMA beta |

```python
qex("ewma(returns(1), 10)")  # Halflife of 10 days
```

!!! tip "Halflife vs Period"
    `halflife=10` means weights decay by 50% every 10 periods.
    Effective lookback is ~3× halflife.

### Higher Moments

| Operation | Description |
|-----------|-------------|
| `ts_skew(x, period)` | Rolling skewness |
| `ts_kurt(x, period)` | Rolling kurtosis |

### Special

| Operation | Description |
|-----------|-------------|
| `ts_argmax(x, period)` | Periods since rolling max |
| `ts_argmin(x, period)` | Periods since rolling min |
| `decay_linear(x, period)` | Linearly decaying weighted average |

```python
qex("ts_argmax(close(), 252)")  # Days since 52-week high
```

### Event-Based (for Sparse Data)

| Operation | Description |
|-----------|-------------|
| `ts_mean_events(x, n)` | Mean over past N non-NaN values |
| `ts_std_events(x, n)` | Std over past N non-NaN values |
| `ts_sum_events(x, n)` | Sum over past N non-NaN values |
| `ts_count_events(x, period)` | Count of non-NaN in rolling window |

```python
# For quarterly earnings (sparse data)
qex("ts_std_events(field('earnings'), 8)")  # Std of last 8 earnings
```

## Cross-Sectional Operations

Operate across tickers for each day independently.

| Operation | Description |
|-----------|-------------|
| `rank(x)` | Percentile rank (0-1), ascending |
| `zscore(x)` | Z-score |
| `demean(x)` | Subtract cross-sectional mean |
| `quantile(x, n)` | Assign to n buckets (1 to n), ascending |
| `winsorize(x, limit)` | Cap at percentiles (e.g., 0.05 caps at 5th/95th) |
| `scale(x)` | Scale so absolute values sum to 1 |
| `truncate(x, max)` | Clip to [-max, max] |

```python
qex("rank(returns(20))")  # Higher return → higher rank
qex("rank(-returns(20))")  # Higher return → lower rank (reversal)
qex("zscore(returns(20))")
qex("winsorize(returns(1), 0.01)")  # Cap 1% tails
```

!!! warning "rank() is Ascending"
    Higher values get higher ranks.
    For mean reversion (buy losers), use `rank(-returns(5))`.

## Group Operations

Like cross-sectional, but within groups.

| Operation | Description |
|-----------|-------------|
| `group('group')` | Access group data for filtering |
| `group_rank(x, 'group')` | Rank within each group |
| `group_demean(x, 'group')` | Demean within each group |
| `group_std(x, 'group', period)` | Rolling std within group |
| `group_sum(x, 'group')` | Sum within group |
| `group_count_valid(x, 'group', period)` | Count non-NaN within group |

```python
qex("group_demean(returns(20), 'sector')")  # Sector-neutral returns
qex("group_rank(returns(20), 'industry')")  # Rank within industry
```

### Sector Filtering

Use `group()` with `where()` to filter signals by sector membership:

```python
# Price-to-book isn't meaningful for Financials - zero them out
qex("where(group('sector')=='Financials', 0, field('price_to_book'))")

# Only trade Technology stocks
qex("where(group('sector')=='Technology', returns(20), 0)")
```

This is useful when factors don't apply to certain sectors (e.g., value metrics for financials, inventory for services).

## Conditional Operations

| Operation | Description |
|-----------|-------------|
| `where(cond, if_true, if_false)` | Ternary conditional |
| `is_valid(x)` | Returns 1 where not NaN, else 0 |
| `fill_forward(x, limit)` | Forward fill NaN for up to N periods |

```python
qex("where(volatility(20) > 0.3, 0, returns(5))")  # Zero out high-vol
qex("fill_forward(field('earnings'), 60)")  # Hold earnings for quarter
```

## Calendar Operations

| Operation | Description |
|-----------|-------------|
| `day_of_week()` | 0=Monday, 6=Sunday |
| `day_of_month()` | 1-31 |
| `month_of_year()` | 1-12 |

```python
qex("where(day_of_week() == 4, returns(5), 0)")  # Friday only
qex("where(month_of_year() == 1, 1, 0)")  # January indicator
```

## Arithmetic Operators

Standard Python operators work:

| Operator | Description |
|----------|-------------|
| `+`, `-`, `*`, `/` | Basic arithmetic |
| `**` | Power |
| `>`, `<`, `>=`, `<=`, `==`, `!=` | Comparison (returns 0/1) |
| `&`, `\|` | Logical AND/OR |

```python
qex("returns(20) / volatility(60)")
qex("(returns(5) > 0) & (volume(1) > adv(20))")
```

## Variable Bindings

Use `let ... in` to define variables:

```python
qex("""
    let mom = returns(60),
        vol = volatility(60),
        sharpe = mom / vol
    in rank(sharpe) * sign(mom)
""")
```

Rules:

- Later bindings can reference earlier ones
- Variable names cannot shadow function names
- Variables are scoped to the expression
