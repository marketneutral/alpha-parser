# Event-Driven Signals

Trading on sparse events like earnings.

## Handling Sparse Data

Event data has values only on event days, NaN otherwise:

```text
            AAPL    MSFT
2024-01-15   NaN     NaN
2024-01-16  0.05     NaN     AAPL beat by 5%
2024-01-17   NaN     NaN
2024-01-18   NaN   -0.02     MSFT missed by 2%
```

## Key Operations

### Forward Fill

Hold a signal for N days after the event:

```qex
fill_forward(field('earnings_surprise'), 60)
```

### Check Validity

Did an event occur?

```qex
is_valid(field('earnings_surprise'))    # 1 if event, 0 otherwise
```

### Event-Based Rolling

Roll over N events (not N days):

```qex
ts_mean_events(field('earnings_surprise'), 4)    # Mean of last 4 quarters
ts_std_events(field('earnings_surprise'), 8)    # Std of last 8 quarters
```

## PEAD Signal

Post-Earnings Announcement Drift:

```qex
# Basic: go long positive surprises

let surprise = field('earnings_surprise'),
    held = fill_forward(surprise, 60)
in rank(held) - 0.5
```

## SUE (Standardized Unexpected Earnings)

Academic version normalizes by historical surprise volatility:

```qex
let surprise = field('earnings_surprise'),
    surprise_std = ts_std_events(surprise, 8),
    sue = surprise / surprise_std,
    held = fill_forward(sue, 60)
in rank(held) - 0.5
```

## Earnings Momentum

Trend in earnings surprises:

```qex
let surprise = field('earnings_surprise'),
    avg_surprise = ts_mean_events(surprise, 4),
    held = fill_forward(avg_surprise, 60)
in rank(held) - 0.5
```

## Announcement Returns

Trade on the announcement day return:

```qex
let announce = is_valid(field('earnings_surprise')),
    ann_ret = where(announce, returns(1), None),
    held = fill_forward(ann_ret, 60)
in rank(-held) - 0.5
```

This fades the immediate reaction.

## Event Count Filter

Only trade stocks with enough history:

```qex
let surprise = field('earnings_surprise'),
    count = ts_count_events(surprise, 252),
    sue = surprise / ts_std_events(surprise, 8),
    held = fill_forward(sue, 60)
in where(count >= 4, rank(held) - 0.5, None)
```

## Multiple Event Types

Combine different events:

```qex
let earnings = fill_forward(field('earnings_surprise'), 60),
    guidance = fill_forward(field('guidance_revision'), 60)
in 0.7 * rank(earnings) + 0.3 * rank(guidance) - 0.5
```
