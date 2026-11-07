# Operator Reference

Every operator in qex, by family. In the signatures:

- `x`, `y` are expressions (anything that gives a value per date and stock);
- `period`, `n`, `window`, `halflife` are whole numbers of days or values; `q`, `limit` are numbers;
- `'name'` is a quoted field or group name; `groups` is a group's name or any expression giving labels;
- `[, arg]` is an optional argument.

Each time-series operator follows one of three window rules (part of the window, the whole window, exponential),
explained in [How qex Evaluates](concepts.md#windows-and-history); the tables say which. Missing values and `None`
are covered in [Missing Values](missing-values.md).

## Data

| Operation | Description |
|-----------|-------------|
| `close()` | Each stock's closing price, adjusted for splits and dividends |
| `open()` | Opening price, adjusted |
| `high()` | Daily high, adjusted |
| `low()` | Daily low, adjusted |
| `field('name')` | Any field of the data by name, read as numbers (read labels with `group('name')`) |
| `market()` | The market's price level: the data's `market` series, the same for every stock |

A field with one value per day (an index, a rate) is given to every stock. `market()` is whatever market series the
data supplies: an index's total-return close, an equal-weight index of the universe, a future. `returns(1, 'market')`
is its daily return.

Adjusted prices (`close()`, `open()`, `high()`, `low()`) are for returns and a stock's own history; compare stocks, or
compare with a dollar amount, with as-traded fields such as `field('price')`. See
[Adjusted Data and Look-Ahead](adjusted-data.md).

## Primitives

| Operation | Description |
|-----------|-------------|
| `returns(period[, 'field'])` | Return over `period` days, of `close` or another price field: p[t] / p[last price at or before t − period] − 1 |
| `volatility(period[, 'field'])` | Annualized volatility: the standard deviation of daily returns over `period` days, times √252. Part of the window |
| `volume(period)` | Average daily volume over `period` days, in shares. Part of the window |
| `adv(period)` | Average daily dollar volume, close × volume, over `period` days. Part of the window |

```qex
returns(20)                 # 20-day return
returns(1, 'market')        # the market's daily return
volatility(60)              # 60-day annualized volatility
volume(5) / volume(60)      # recent volume against its norm
```

**Returns across gaps.** On a day with no price, a return has no value: a halted or delisted stock drops out of
`rank` and the group operators rather than counting as 0%. The day it trades again carries the whole move since its
last price. A price more than 10 days old no longer counts as a last price. The daily returns inside `volatility`
follow the same rule.

## Math

| Operation | Description |
|-----------|-------------|
| `log(x)` | Natural logarithm; missing for x ≤ 0 (−inf at 0) |
| `exp(x)` | e to the `x`, the inverse of `log`; inf when too large |
| `abs(x)` | Absolute value |
| `sign(x)` | −1, 0 or 1 |
| `sqrt(x)` | Square root; missing for x < 0 |
| `power(x, n)` | `x` to the power `n`; `n` can be a number or an expression |
| `max(x, y)` | The larger of `x` and `y`, element by element |
| `min(x, y)` | The smaller of `x` and `y`, element by element |

Any argument can be a number: `sqrt(5)`, `max(x, 0)`, `power(2, x)`.

```qex
log(close() / delay(close(), 20))           # log return
sign(returns(20)) * sqrt(abs(returns(20)))  # a signed square root
max(returns(5), 0)                          # positive returns, 0 otherwise
exp(ts_sum(log(1 + returns(1)), 20)) - 1    # daily returns compounded over 20 days
1 / (1 + exp(-zscore(returns(5))))          # a sigmoid
scale(exp(zscore(returns(5)) / 0.5))        # a cross-sectional softmax, temperature 0.5
```

## Time series

Each stock's own history, one stock at a time.

### Rolling statistics

| Operation | Description |
|-----------|-------------|
| `ts_mean(x, period)` | Mean over the last `period` days. Part of the window |
| `ts_std(x, period)` | Standard deviation over the window. Part of the window |
| `ts_sum(x, period)` | Sum over the window. Part of the window |
| `ts_max(x, period)` | Largest value in the window. Part of the window |
| `ts_min(x, period)` | Smallest value in the window. Part of the window |
| `ts_var(x, period)` | Variance over the window. Part of the window |
| `ts_quantile(x, period, q[, min_periods])` | The `q`-quantile of the window, `q` from 0 to 1: 0.5 is the median, 0 and 1 are `ts_min` and `ts_max`. Part of the window |

Part of the window means a value once the window holds `min(20, period)` values, and a value on a day `x` is missing
(from the rest of the window). `ts_quantile` interpolates between the two nearest values, and takes its minimum as an
optional last argument.

```qex
ts_mean(returns(1), 20)                                        # 20-day average daily return
close() / ts_max(close(), 252)                                 # distance from the 52-week high
-ts_quantile(returns(1), 252, 0.05)                            # 5% value-at-risk, as a loss
ts_quantile(returns(1), 60, 0.75) - ts_quantile(returns(1), 60, 0.25)    # interquartile range
```

### Lag and difference

| Operation | Description |
|-----------|-------------|
| `delay(x, period)` | The value `period` days ago; `period` ≥ 0 |
| `delta(x, period)` | Today's value minus the value `period` days ago: `x - delay(x, period)` |

There is no negative delay: a value can only come from the past.

```qex
delay(close(), 1)            # yesterday's close
delta(returns(20), 5)        # change in 20-day momentum over a week
```

### Ranking and extremes

| Operation | Description |
|-----------|-------------|
| `ts_rank(x, period)` | Today's percentile among the last `period` values, from 1/period to 1; ties share their average rank. The whole window |
| `ts_argmax(x, period)` | Days since the largest value in the window, 0 if it is today. The whole window |
| `ts_argmin(x, period)` | Days since the smallest value in the window. The whole window |

The whole window means every day of the window must have a value: a gap hides the result until it has left the
window.

```qex
ts_rank(volume(1), 20)       # where today's volume ranks against the last 20 days
ts_argmax(close(), 252)      # days since the 52-week high
```

### Pairs of series

| Operation | Description |
|-----------|-------------|
| `ts_corr(x, y, period[, min_periods])` | Correlation of `x` and `y` over the window. Part of the window |
| `ts_cov(x, y, period[, min_periods])` | Covariance over the window. Part of the window |
| `ts_beta(y, x, period[, min_periods])` | Least-squares slope of `y` on `x`, cov(y, x) / var(x), over the window. Part of the window |
| `ts_beta_shrunk(y, x, period[, min_periods])` | `ts_beta`, pulled toward the day's average beta in proportion to its noise (Vasicek). Part of the window |

All four use only the days on which both series have a value. `min_periods` (default `min(20, period)`) is the fewest
such days a window needs: with the default, a stock 20 days after listing gets a "252-day" beta from 20 days, so ask
for more when it matters, `ts_beta(returns(1), returns(1, 'market'), 252, 126)`.

`ts_beta_shrunk` gives `w × b + (1 − w) × mean(b)`, with `w = var(b) / (var(b) + se(b)²)`: a beta from a short or
noisy history moves furthest toward the day's average. The average is over the universe.
[Market Beta](../examples/market-beta.md#shrinking-beta) derives it.

```qex
ts_corr(returns(1), volume(1), 60)                       # return-volume correlation
ts_beta(returns(1), returns(1, 'market'), 252)           # market beta
ts_beta_shrunk(returns(1), returns(1, 'market'), 252)    # market beta, shrunk
```

### Exponentially weighted

| Operation | Description |
|-----------|-------------|
| `ewma(x, halflife)` | Exponentially weighted mean: a day's weight halves every `halflife` days. Exponential |
| `ewma_var(x, halflife)` | Exponentially weighted variance. Exponential |
| `ewma_cov(x, y, halflife)` | Exponentially weighted covariance. Exponential |
| `ts_beta_ewma(y, x, halflife)` | ewma_cov(y, x) / ewma_var(x), over the days both have a value. Exponential |

Exponential means a value from the first observation on (there is no warm-up), and a value carried through days
`x` is missing. The effective look-back is about three half-lives.

```qex
ewma(returns(1), 10)                                  # a 10-day half-life average return
ts_beta_ewma(returns(1), returns(1, 'market'), 63)    # a beta that adapts within a quarter
```

### Shape of the window

| Operation | Description |
|-----------|-------------|
| `ts_skew(x, period)` | Skewness of the window. The whole window |
| `ts_kurt(x, period)` | Excess kurtosis of the window. The whole window |
| `decay_linear(x, period)` | Weighted mean with weights period, period − 1, ..., 1, newest heaviest. The whole window |

```qex
ts_skew(returns(1), 60)      # 60-day return skew
decay_linear(returns(1), 10) # a recency-weighted 10-day average
```

### Events

| Operation | Description |
|-----------|-------------|
| `ts_mean_events(x, n[, min_n])` | Mean of the last `n` values `x` has had, however far back; a value every day once there are `min_n` (default `n`) |
| `ts_std_events(x, n[, min_n])` | Standard deviation of the last `n` values, once there are `min_n` |
| `ts_sum_events(x, n[, min_n])` | Sum of the last `n` values, once there are `min_n` |
| `ts_count_events(x, period)` | How many values `x` has in the last `period` days; an answer every day once `period` days have passed |

For data that arrives now and then, such as quarterly earnings, where a fixed window in days would hold a varying
number of events. Between events the result holds.

```qex
ts_std_events(field('surprise'), 8)              # standard deviation of the last 8 surprises
ts_mean_events(field('surprise'), 4, 2)          # average of up to 4, once there are 2
ts_count_events(field('surprise'), 252) >= 3     # at least three reports in a year
```

### Filling and state

| Operation | Description |
|-----------|-------------|
| `fill_forward(x[, limit])` | Carry each value forward over the missing days after it, at most `limit` days; with no limit, until the next value |
| `latch(on, off)` | 1 from a day `on` is true until a day `off` is true (off wins a tie), then 0 until `on` again; 0 before `on` is first true; a day where either is missing changes nothing |

```qex
fill_forward(field('surprise'), 60)                        # hold an earnings surprise for a quarter
latch(field('mcap') >= 300e6, field('mcap') < 240e6)       # enter at $300M, leave under $240M
```

`latch` is how a universe gets hysteresis ([Universes](universes.md#hysteresis-latch)), and how a signal holds a
position from an entry rule until an exit rule.

## Cross-sectional

Across the stocks on each day, one day at a time. They compare the stocks in the universe, and give every other stock
no value ([Universes](universes.md)).

| Operation | Description |
|-----------|-------------|
| `rank(x)` | Percentile rank among the day's stocks, from 1/n to 1, lowest first; ties share their average rank |
| `zscore(x)` | (x − mean) / standard deviation across the day's stocks; missing when they don't vary |
| `demean(x)` | x minus the day's mean |
| `quantile(x, n)` | Bucket 1 (lowest) to `n` by percentile rank |
| `winsorize(x, limit)` | Clip at the day's `limit` and 1 − `limit` quantiles: `winsorize(x, 0.05)` caps at the 5th and 95th percentiles |
| `scale(x)` | Divide by the day's sum of absolute values, so they sum to 1 |
| `truncate(x, max)` | Clip each value to [−max, max] (element by element, not across stocks) |
| `regression_neut(y, x1, x2, ...)` | y with its least-squares fit on the regressors (and an intercept) taken out, across the day's stocks |
| `pca_factor(x, window, k[, refit])` | Each stock's beta to the k-th statistical factor of `x`, from a principal component analysis across stocks, refit every `refit` days (default 21) |

```qex
rank(returns(20))                    # higher return, higher rank
rank(-returns(5))                    # higher return, lower rank: a reversal
zscore(returns(20))
winsorize(returns(1), 0.01)          # cap the 1% tails
quantile(field('mcap'), 5)           # size quintiles, 1 smallest
```

**Infinities.** `zscore`, `demean`, `winsorize`, `scale` and `regression_neut` leave an infinite value out of the day's
statistics and give that stock no value, so one division by zero (a zero-volume day in `volume(20) / volume(1)`)
can't empty the day. `rank` and `quantile` rank an infinity first or last.

### `regression_neut`

`regression_neut(y, x)` takes out of `y`, each day, the part that `x` explains: the residual of a least-squares fit
of `y` on `x` with an intercept, over the stocks where both have a value. The result has zero mean and zero
covariance with `x` across the day's stocks, which is what "neutral to x" means. A day on which `x` doesn't vary
gives `y` demeaned.

With several regressors, `regression_neut(y, beta, size, momentum)` is one fit on all of them at once: the result is
neutral to each. That is not the same as nesting, `regression_neut(regression_neut(y, beta), size)`: when `beta` and
`size` are correlated, the second step brings back some of the beta the first took out. Regressors that duplicate
each other are allowed (the fit of smallest norm is used).

A day needs at least k + 2 stocks with values for k regressors; with fewer, the fit passes through every point and
the residual would be rounding noise, so the day has no value.

```qex
let beta = ts_beta_shrunk(returns(1), returns(1, 'market'), 252)
in regression_neut(rank(returns(20)), beta)
```

Momentum with market beta taken out. With beta and size taken out together:

```qex
let beta = ts_beta_shrunk(returns(1), returns(1, 'market'), 252),
    size = log(field('mcap'))
in regression_neut(zscore(-returns(5)), beta, size)
```

The signal is exactly neutral; a book built from it is close to neutral, since a book's caps and scaling come after.

### `pca_factor`

`pca_factor(x, window, k, refit)` finds the common factors in `x` (usually `returns(1)`) from the data alone, by a
principal component analysis across stocks, and gives each stock's exposure to the k-th.

- **Point in time.** Every `refit` days (the first once `window` days exist), the analysis uses the last `window`
  days of `x`, ending that day; its result holds until the next refit. Before the first, there is no value.
- **Which stocks.** Those with values on at least 90% of the window, and in the universe that day. A stock not in a
  fit has no value until it is in one.
- **Correlation, not covariance.** Each stock is scaled by its own volatility first, so a few volatile stocks don't
  make up the first factors.
- **A beta.** The value is the stock's beta to the k-th factor's return, in `x`'s units: the exposure a position
  carries, and what `regression_neut` should take out.
- **Factors keep their identity.** Each refit's factors are matched to the previous refit's and signed to agree, so
  factor 2 stays factor 2 and doesn't flip sign. The first factor is the market, signed long. `k` is at most 20.
- **Shared.** `pca_factor` with the same `x`, `window` and `refit` and different `k` share one decomposition.

```qex
pca_factor(returns(1), 252, 1)       # each stock's beta to the first statistical factor

let r = returns(1)
in regression_neut(zscore(-returns(5)), pca_factor(r, 252, 1), pca_factor(r, 252, 2), pca_factor(r, 252, 3))
```

## Groups

Like the cross-sectional operators, but within groups: a stock is compared only with the stocks sharing its label that
day. [Groups and Labels](groups.md) works through each one on an example.

| Operation | Description |
|-----------|-------------|
| `group('name')` | The labels themselves, for comparing or grouping: `group('sector') == 'Technology'` |
| `group_rank(x, groups)` | Percentile rank within each group, from 1/n to 1 |
| `group_demean(x, groups)` | x minus its group's mean |
| `group_quantile(x, groups, n)` | Bucket 1 to `n` within each group: a dependent double sort |
| `group_std(x, groups)` | Standard deviation across the group's stocks that day; every member gets it |
| `group_sum(x, groups)` | Sum within the group; every member gets it; missing when the group has no values that day |
| `group_count_valid(x, groups, period)` | How many values the group's stocks have had over the last `period` days |
| `group_cross(a, b, ...)` | One grouping from several: labels joined with a bar, `'Technology\|3'` |

`groups` is a group's name (`'sector'`) or any expression giving labels: `quantile(field('mcap'), 5)`,
`coalesce(group('sector'), 'Other')`, `group_cross(...)`. A stock with no label, or outside the universe, gets no
value. A member whose own value is missing still gets its group's number from `group_std`, `group_sum` and
`group_count_valid`. Labels given on fewer dates than the prices hold from their date until the next, never earlier.

```qex
group_demean(returns(20), 'sector')                                      # sector-neutral momentum
group_demean(returns(20), quantile(field('mcap'), 5))                    # size-neutral momentum
group_demean(returns(5), 'sector') / group_std(returns(5), 'sector')     # a z-score within each sector
group_quantile(field('value'), quantile(field('mcap'), 3), 3)            # value terciles within size terciles
group_demean(returns(20), group_cross('sector', quantile(field('mcap'), 3)))    # sector- and size-neutral
volume(1) / group_sum(volume(1), 'sector')                               # share of the sector's volume
```

**Labels and numbers.** `group(...)`, `group_cross(...)` and quoted strings give labels. Labels can be compared with
`==` and `!=`, used as `groups`, and passed through `where` and `coalesce`; anything else is refused when the
expression is read. See [Groups and Labels](groups.md#numbers-and-labels).

## Conditions and missing values

| Operation | Description |
|-----------|-------------|
| `where(cond, if_true, if_false)` | `if_true` where `cond` is nonzero, `if_false` where it is 0, missing where it is missing |
| `coalesce(x, y)` | `x` where it has a value, else `y` (a number, label or expression) |
| `is_valid(x)` | 1 where `x` has a value, else 0; an infinity counts as a value |
| `is_finite(x)` | 1 where `x` is a finite number, 0 where it is missing or infinite |
| `None` | No value here, on purpose: `where(cond, x, None)` |

```qex
where(volatility(20) > 0.6, None, returns(5))                        # leave out the most volatile stocks
where(group('sector') == 'Financials', None, field('book_to_price')) # a ratio that means little for banks
coalesce(close() > 8, 0)                                             # missing counts as false
where(is_finite(volume(20) / volume(1)), volume(20) / volume(1), None)
```

`None` and `0` are different answers: a `0` is a value, and a book holds a stock at 0 against the day's average;
`None` takes the stock out. See [Missing Values](missing-values.md).

## Calendar

| Operation | Description |
|-----------|-------------|
| `day_of_week()` | 0 (Monday) to 6 (Sunday) |
| `day_of_month()` | 1 to 31 |
| `month_of_year()` | 1 to 12 |

The same for every stock on a day.

```qex
where(day_of_week() == 4, returns(5), None)                  # Fridays only
where(month_of_year() == 1, 1, 0)                            # a January indicator
month_of_year() != delay(month_of_year(), 1)                 # the first trading day of each month
```

## Arithmetic, comparisons and logic

| Operator | Description |
|----------|-------------|
| `+`, `-`, `*`, `/` | Arithmetic, element by element |
| `**` | Power, with a number as the exponent |
| `>`, `<`, `>=`, `<=`, `==`, `!=` | 1 or 0, missing where either side is missing |
| `&`, `\|`, `~` | And, or, not, three-valued |

Precedence and parentheses are in [Syntax](syntax.md#precedence); the three-valued rules in
[Missing Values](missing-values.md#comparisons-and-logic-three-values).
