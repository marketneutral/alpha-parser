# Events and State

For data that arrives now and then, such as quarterly earnings, and for signals that hold a state from one day to the
next. In the [toy data](../reference.md#the-toy-data), `field('surprise')` has four reports: **A** on Jan 03, **C** on
Jan 05, **B** and **D** on Jan 08.

## Holding values

### fill_forward

<p class="sig"><code>fill_forward(x[, limit])</code></p>

Carries each value forward over the missing days after it, for at most `limit` days; with no limit, until the next
value. Below, each surprise is held for one more day.

```toy
field('surprise')
fill_forward(field('surprise'), 1)
```

```qex
fill_forward(field('surprise'), 60)        # hold a surprise for a quarter
```

### latch

<p class="sig"><code>latch(on, off)</code></p>

A switch: 1 from a day `on` is true until a day `off` is true, then 0 until `on` is true again. It starts at 0, `off`
wins a tie, and a day where either is missing changes nothing. With the exit below the entry, small moves around a
threshold don't flip it back and forth. Below, a price filter: **A** trades above $11.50 on Jan 04 and switches on;
it slips to $11 on Jan 05, which turns the plain comparison off, but the latch stays on until a price under $10.50.

```toy
field('price') > 11.5
latch(field('price') > 11.5, field('price') < 10.5)
```

```qex
latch(field('mcap') >= 300e6, field('mcap') < 240e6)     # in at $300M, out under $240M
```

`latch` gives a universe hysteresis ([Universes](../universes.md#hysteresis-latch)), and holds a position from an
entry rule to an exit rule.

## Counting by events

A fixed window in days holds a varying number of quarterly reports. These operators count reports instead: the last
`n` values `x` has had, however far back, held between events.

### ts_mean_events

<p class="sig"><code>ts_mean_events(x, n[, min_n])</code></p>

The mean of the last `n` values, with a value every day once there are `min_n` (default `n`).

```toy
field('surprise')
ts_mean_events(field('surprise'), 4, 1)
```

```qex
ts_mean_events(field('surprise'), 4)       # average of the last four
```

### ts_std_events

<p class="sig"><code>ts_std_events(x, n[, min_n])</code></p>

The standard deviation of the last `n` values, once there are `min_n`.

```qex
field('surprise') / ts_std_events(field('surprise'), 8)     # a standardized surprise, on report days
```

### ts_sum_events

<p class="sig"><code>ts_sum_events(x, n[, min_n])</code></p>

The sum of the last `n` values, once there are `min_n`.

### ts_count_events

<p class="sig"><code>ts_count_events(x, period)</code></p>

How many values `x` has had in the last `period` days: an answer every day once `period` days have passed, 0 included.

```toy
field('surprise')
ts_count_events(field('surprise'), 3)
```

```qex
ts_count_events(field('surprise'), 252) >= 3    # three reports in a year
```

### days_since

<p class="sig"><code>days_since(x)</code></p>

How many trading days since `x` was last nonzero: 0 on a day it is, 1 the day after, and so on. A day counts when `x`
has a value that is not 0 (a negative value counts; so does an infinity, as in `where`). A missing day is not an
event, and it neither resets nor stops the count, so a halt between two events still counts its days. Before `x` is
first nonzero the answer is missing. There is no window: the count runs however far back the last event is.

Days are rows of the data, trading days, not calendar days. Below, **C**'s report on Jan 05 starts its count, and **B**
and **D** have no count until their reports on Jan 08.

```toy
field('surprise')
days_since(field('surprise'))
```

```qex
days_since(field('surprise')) <= 5                    # a report in the last week
where(days_since(field('surprise')) < 10, field('surprise'), None)    # the latest report, for ten days
days_since(close() >= ts_max(close(), 252))          # days since a 52-week high
```

For "days since the latest", use `days_since`, not `ts_argmax`: given two events in its window, `ts_argmax` counts
back to the older one.
