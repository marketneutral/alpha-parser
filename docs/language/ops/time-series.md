# Time Series

Each stock's own history, one stock at a time. `period` counts rows of the data (trading days), not calendar days.

## Window rules

Every rolling operator follows one of three rules for how much history it needs and what a missing day does. The toy
data shows all three on **C**, which has no price on Jan 04.

| Rule | Operators | A value once the window holds | A missing day in the window |
|---|---|---|---|
| **Part of the window** | `ts_mean`, `ts_std`, `ts_sum`, `ts_max`, `ts_min`, `ts_var`, `ts_quantile`, the pair operators, `volatility`, `volume`, `adv` | `min(20, period)` values | is skipped, as long as enough values remain |
| **The whole window** | `ts_rank`, `ts_argmax`, `ts_argmin`, `ts_skew`, `ts_kurt`, `decay_linear` | `period` values | hides the result until it has left the window |
| **Exponential** | `ewma`, `ewma_var`, `ewma_cov`, `ts_beta_ewma` | one value | is skipped; the average carries on |

With a window of 20 days or less, part of the window needs every day too, as in the 3-day panels below. The difference
shows on longer windows: a 60-day mean needs 20 values and skips the gaps between them.
[How qex Evaluates](../concepts.md#windows-and-history) explains why each rule is what it is.

## Rolling statistics

### ts_mean

<p class="sig"><code>ts_mean(x, period)</code></p>

The mean of each stock's last `period` values. Part of the window.

```toy
close()
ts_mean(close(), 3)
```

```qex
ts_mean(returns(1), 20)                    # average daily return over a month
close() / ts_mean(close(), 200) - 1        # distance from the 200-day average
```

### ts_std

<p class="sig"><code>ts_std(x, period)</code></p>

The standard deviation of the window (sample, n − 1). Part of the window.

```qex
ts_std(returns(1), 20)                     # daily volatility, not annualized
```

### ts_var

<p class="sig"><code>ts_var(x, period)</code></p>

The variance of the window, `ts_std` squared. Part of the window.

### ts_sum

<p class="sig"><code>ts_sum(x, period)</code></p>

The sum of the window. Part of the window.

```qex
ts_sum(log(1 + returns(1)), 20)            # the 20-day log return, from daily returns
```

### ts_max

<p class="sig"><code>ts_max(x, period)</code></p>

The largest value in the window. Part of the window.

```toy
close()
ts_max(close(), 3)
```

```qex
close() / ts_max(close(), 252)             # how far below the 52-week high
```

### ts_min

<p class="sig"><code>ts_min(x, period)</code></p>

The smallest value in the window. Part of the window.

### ts_quantile

<p class="sig"><code>ts_quantile(x, period, q[, min_periods])</code></p>

The `q`-quantile of the window, `q` from 0 to 1, interpolating between the two nearest values: 0.5 is the median, 0
and 1 are `ts_min` and `ts_max`. Part of the window; `min_periods` changes the minimum.

```qex
-ts_quantile(returns(1), 252, 0.05)                                     # 5% value-at-risk, as a loss
ts_quantile(returns(1), 60, 0.75) - ts_quantile(returns(1), 60, 0.25)   # interquartile range
```

## Lag and difference

### delay

<p class="sig"><code>delay(x, period)</code></p>

The value `period` rows ago; `period` ≥ 0. There is no negative delay: a value can only come from the past.

```toy
close()
delay(close(), 1)
```

```qex
delay(returns(231), 21)                    # 12-month momentum, skipping the last month
```

### delta

<p class="sig"><code>delta(x, period)</code></p>

Today's value minus the value `period` rows ago: `x - delay(x, period)`. Missing when either is.

```qex
delta(returns(20), 5)                      # how momentum changed over a week
```

## Ranking and extremes

### ts_rank

<p class="sig"><code>ts_rank(x, period)</code></p>

Where today's value ranks among the stock's last `period` values, from `1/period` (the lowest) to `1` (the highest);
ties share their average rank. The whole window: after **C**'s missing day there is no rank until Jan 09.

```toy
close()
ts_rank(close(), 3)
```

```qex
ts_rank(volume(1), 20)                     # is today's volume unusual for this stock?
```

### ts_argmax

<p class="sig"><code>ts_argmax(x, period)</code></p>

How many days ago the window's largest value was, 0 if it is today. The whole window.

```toy
close()
ts_argmax(close(), 3)
```

```qex
ts_argmax(close(), 252)                    # days since the 52-week high
```

### ts_argmin

<p class="sig"><code>ts_argmin(x, period)</code></p>

How many days ago the window's smallest value was. The whole window.

## Pairs of series

The pair operators use only the days on which both series have a value. `min_periods` (default `min(20, period)`) is
the fewest such days a window needs: with the default, a stock 20 days after listing gets a "252-day" beta from 20
days. Ask for more when it matters.

### ts_corr

<p class="sig"><code>ts_corr(x, y, period[, min_periods])</code></p>

The correlation of `x` and `y` over the window. Part of the window.

```qex
ts_corr(abs(returns(1)), volume(1), 60)    # do big moves come with volume?
```

### ts_cov

<p class="sig"><code>ts_cov(x, y, period[, min_periods])</code></p>

The covariance of `x` and `y` over the window. Part of the window.

### ts_beta

<p class="sig"><code>ts_beta(y, x, period[, min_periods])</code></p>

The least-squares slope of `y` on `x` over the window, cov(y, x) / var(x). Part of the window.

```qex
ts_beta(returns(1), returns(1, 'market'), 252)          # market beta
ts_beta(returns(1), returns(1, 'market'), 252, 126)     # no value until 126 days are paired
```

### ts_beta_shrunk

<p class="sig"><code>ts_beta_shrunk(y, x, period[, min_periods])</code></p>

`ts_beta`, pulled toward the day's average beta in proportion to its own noise (Vasicek):
`w × b + (1 − w) × mean(b)`, with `w = var(b) / (var(b) + se(b)²)`. A beta from a short or noisy history moves furthest
toward the average, which is taken over the universe. [Market Beta](../../examples/market-beta.md#shrinking-beta)
derives it.

```qex
ts_beta_shrunk(returns(1), returns(1, 'market'), 252)
```

## Exponentially weighted

A day's weight halves every `halflife` days, so the effective look-back is about three half-lives. There is no warm-up:
a value from the first observation, carried through missing days.

### ewma

<p class="sig"><code>ewma(x, halflife)</code></p>

The exponentially weighted mean. Below, **C** keeps its average through Jan 04.

```toy
close()
ewma(close(), 2)
```

```qex
ewma(returns(1), 10)                       # a 10-day half-life average return
```

### ewma_var

<p class="sig"><code>ewma_var(x, halflife)</code></p>

The exponentially weighted variance.

```qex
sqrt(ewma_var(returns(1), 20) * 252)       # an annualized volatility that adapts quickly
```

### ewma_cov

<p class="sig"><code>ewma_cov(x, y, halflife)</code></p>

The exponentially weighted covariance, over the days both have a value.

### ts_beta_ewma

<p class="sig"><code>ts_beta_ewma(y, x, halflife)</code></p>

`ewma_cov(y, x) / ewma_var(x)`, over the days both have a value.

```qex
ts_beta_ewma(returns(1), returns(1, 'market'), 63)     # a beta that adapts within a quarter
```

## Shape of the window

### decay_linear

<p class="sig"><code>decay_linear(x, period)</code></p>

A weighted mean with weights `period`, `period − 1`, …, 1, the newest heaviest. The whole window. On Jan 04, **A** is
(3 × 12 + 2 × 11 + 1 × 10) / 6.

```toy
close()
decay_linear(close(), 3)
```

```qex
decay_linear(returns(1), 10)               # a recency-weighted 10-day average
```

### ts_skew

<p class="sig"><code>ts_skew(x, period)</code></p>

The skewness of the window. The whole window.

```qex
-ts_skew(returns(1), 60)                   # prefer stocks without lottery-like upside
```

### ts_kurt

<p class="sig"><code>ts_kurt(x, period)</code></p>

The excess kurtosis of the window (0 for a normal distribution). The whole window.
