# Event-Driven Signals

Trading on sparse events like earnings.

## Handling Sparse Data

Event data has values only on event days, NaN otherwise:

```python
# Earnings: NaN most days, value on announcement
#            AAPL   MSFT
# 2024-01-15  NaN    NaN
# 2024-01-16  0.05   NaN   <- AAPL beat by 5%
# 2024-01-17  NaN    NaN
# 2024-01-18  NaN   -0.02  <- MSFT missed by 2%
```

## Key Operations

### Forward Fill

Hold a signal for N days after the event:

```python
qex("fill_forward(field('earnings_surprise'), 60)")
```

### Check Validity

Did an event occur?

```python
qex("is_valid(field('earnings_surprise'))")  # 1 if event, 0 otherwise
```

### Event-Based Rolling

Roll over N events (not N days):

```python
qex("ts_mean_events(field('earnings_surprise'), 4)")  # Mean of last 4 quarters
qex("ts_std_events(field('earnings_surprise'), 8)")   # Std of last 8 quarters
```

## PEAD Signal

Post-Earnings Announcement Drift:

```python
# Basic: go long positive surprises
qex("""
    let surprise = field('earnings_surprise'),
        held = fill_forward(surprise, 60)
    in rank(held) - 0.5
""")
```

## SUE (Standardized Unexpected Earnings)

Academic version normalizes by historical surprise volatility:

```python
qex("""
    let surprise = field('earnings_surprise'),
        surprise_std = ts_std_events(surprise, 8),
        sue = surprise / surprise_std,
        held = fill_forward(sue, 60)
    in rank(held) - 0.5
""")
```

## Earnings Momentum

Trend in earnings surprises:

```python
qex("""
    let surprise = field('earnings_surprise'),
        avg_surprise = ts_mean_events(surprise, 4),
        held = fill_forward(avg_surprise, 60)
    in rank(held) - 0.5
""")
```

## Announcement Returns

Trade on the announcement day return:

```python
qex("""
    let announce = is_valid(field('earnings_surprise')),
        ann_ret = where(announce, returns(1), 0),
        held = fill_forward(ann_ret, 60)
    in rank(-held) - 0.5
""")
```

This fades the immediate reaction.

## Event Count Filter

Only trade stocks with enough history:

```python
qex("""
    let surprise = field('earnings_surprise'),
        count = ts_count_events(surprise, 252),
        sue = surprise / ts_std_events(surprise, 8),
        held = fill_forward(sue, 60)
    in where(count >= 4, rank(held) - 0.5, 0)
""")
```

## Multiple Event Types

Combine different events:

```python
qex("""
    let earnings = fill_forward(field('earnings_surprise'), 60),
        guidance = fill_forward(field('guidance_revision'), 60)
    in 0.7 * rank(earnings) + 0.3 * rank(guidance) - 0.5
""")
```

## Data Setup

```python
# Sparse earnings data
earnings = pd.DataFrame(np.nan, index=dates, columns=tickers)

# Fill in event values
for ticker in tickers:
    # Quarterly dates
    quarter_dates = pd.date_range(
        dates[0], dates[-1], freq='QS'
    )
    for qd in quarter_dates:
        if qd in dates:
            earnings.loc[qd, ticker] = np.random.randn()

data['earnings_surprise'] = earnings
```

## Backtest Example

```python
from qex import qex, Backtest

signal = qex("""
    let surprise = field('earnings_surprise'),
        sue = surprise / ts_std_events(surprise, 8),
        held = fill_forward(sue, 60)
    in rank(held) - 0.5
""")

bt = Backtest(signal, transaction_cost=0.001)
result = bt.run(data)

print(result.summary())
```
