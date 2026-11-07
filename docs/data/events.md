# Event Data

Some data arrives now and then: earnings surprises once a quarter, analyst revisions, index changes. Store it as an
ordinary field that is `NaN` on every day without an event:

```python
surprise = pd.DataFrame(np.nan, index=dates, columns=tickers)
surprise.loc['2024-01-15', 'AAPL'] = 0.05      # AAPL beat by 5%
surprise.loc['2024-01-18', 'MSFT'] = -0.02     # MSFT missed by 2%
data['surprise'] = surprise
```

```
            AAPL   MSFT
2024-01-15  0.05    NaN
2024-01-16   NaN    NaN
2024-01-17   NaN    NaN
2024-01-18   NaN  -0.02
```

Then in qex:

```qex
fill_forward(field('surprise'), 60)            # hold each surprise for a quarter
is_valid(field('surprise'))                    # 1 on an event day
ts_std_events(field('surprise'), 8)            # spread of the last 8 surprises, however far back
ts_count_events(field('surprise'), 252)        # how many in the last year
```

Put each event on the day it became known (the announcement date, or the next trading day for an announcement after
the close), not the period it describes: a fiscal-quarter end date would let the signal see the number weeks early.

The event operators are in the [Operator Reference](../language/ops/events.md); worked signals are in
[Event-Driven](../examples/event-driven.md).
