# How qex Evaluates

## Every value is a table of dates by stocks

Every part of a qex expression stands for a **panel**: one value per date and per stock. `close()` is each stock's
close on each day; `returns(5)` is each stock's five-day return on each day; `rank(returns(5))` is each stock's rank
among that day's stocks. A number such as `0.5` is the same value on every date and stock.

So an expression is not a calculation for one stock or one day. It is one formula applied to the whole panel at
once, and its result is again a panel: a signal, ready to become a book.

An expression is a tree of operators, read from the inside out:

```qex
rank(-returns(5) / volatility(20))
```

```
rank
 └── ÷
      ├── − returns(5)        each stock's 5-day return, negated
      └── volatility(20)      each stock's 20-day volatility
```

A part used twice, by name with `let` or written out twice, is the same part: qex recognizes it and computes it once.

## Four kinds of operator

| Kind | Works | Examples | A stock's result depends on |
|------|-------|----------|-----------------------------|
| **Element-wise** | on each date and stock separately | `+`, `log`, `abs`, `where`, comparisons | its own values that day |
| **Time-series** | down each stock's history | `ts_mean`, `delay`, `ts_rank`, `ewma`, `returns` | its own past |
| **Cross-sectional** | across the stocks on each day | `rank`, `zscore`, `demean`, `regression_neut` | the other stocks that day |
| **Group** | across the stocks *in its group* on each day | `group_demean`, `group_rank` | the stocks sharing its label |

Most signals mix all of them. In `rank(-returns(5) / volatility(20))`, `returns` and `volatility` look down each
stock's history, `/` and `-` work element by element, and `rank` compares the stocks on each day.

## Which stocks a cross-section compares

A cross-sectional or group operator compares the stocks in the **universe** that day, and gives every other stock no
value. Time-series operators are not restricted: a stock that joined the universe today still has its past returns,
volatility and beta. See [Universes](universes.md).

One consequence: a time-series operator applied to a cross-sectional result, such as `ts_mean(rank(x), 20)`, has no
history for a stock from before it joined. Smooth first, then compare, where the meaning allows:
`rank(ts_mean(x, 20))` has a value on a stock's first day in the universe.

## Windows and history

A time-series operator looks back over a window, and needs enough values in it. qex's operators follow three rules:

| Rule | Operators | Gives a value once | On a day its input is missing | After a gap |
|------|-----------|--------------------|-------------------------------|-------------|
| **Part of the window** | `ts_mean`, `ts_std`, `ts_sum`, `ts_max`, `ts_min`, `ts_var`, `ts_quantile`, `ts_corr`, `ts_cov`, `ts_beta`, `ts_beta_shrunk`, `volatility`, `volume`, `adv` | the window has `min(20, period)` values, or the `min_periods` given to an operator that takes one | still gives a value, from the rest of the window | recovers at once |
| **The whole window** | `ts_rank`, `ts_skew`, `ts_kurt`, `ts_argmax`, `ts_argmin`, `decay_linear` | every day of the window has a value | no value | no value until the gap has left the window |
| **Exponential** | `ewma`, `ewma_var`, `ewma_cov`, `ts_beta_ewma` | its first value | still gives a value | carries straight on |

So `ts_mean(x, 252)` has a value 20 days after a listing, from 20 days; `ts_rank(x, 252)` waits a full year, and one
missing day then hides it for a year. When a window's minimum matters, ask for it explicitly: `ts_corr`, `ts_cov`,
`ts_beta`, `ts_beta_shrunk` and `ts_quantile` take it as an argument, and
`where(ts_count_events(x, 252) >= 200, signal, None)` works for any operator.

`ts_count_events(x, period)` counts the values `x` has in its window, so it has an answer on every day once `period`
days have passed. The other event operators (`ts_mean_events`, `ts_std_events`, `ts_sum_events`) look back over the
last `n` values that exist, however far back, for data such as quarterly earnings. See the
[operator reference](reference.md#events).

## Point in time

A value on day t uses data up to and including day t, never later. There is no operator that looks forward: `delay`
and `delta` take only non-negative periods. A book trades on the signal with a delay, normally of one day, so a
signal computed from day t's close is first traded at day t+1's close.

Two things can still let the future in, and qex checks for both:

- **Adjusted price levels** fold in later splits and dividends. Comparing them across stocks, or with a fixed
  number, sees the future. See [Adjusted Data and Look-Ahead](adjusted-data.md).
- **Labels**: a stock's group on a past day must be the label it had then. Labels given on fewer dates than the
  prices hold from their date forward, never backward. See [Groups and Labels](groups.md).

## Long and short

A signal's values become a book only after the book demeans and scales them (see the Python section). What the
signal says is therefore relative: a stock is held long or short according to where its value sits against the
day's other values.

`rank(x)` gives values in (0, 1], so `rank(x)` and `rank(x) - 0.5` make the same book. A `0` is a value like any
other, so a stock at 0 is held against the day's average; a missing value is left out of the book. That difference
runs through the whole language: see [Missing Values](missing-values.md).

Equal long and short dollars don't make a book market-neutral; see [Market Beta](../examples/market-beta.md).
