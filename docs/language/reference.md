# Operator Reference

Every operator in qex, one line each. Each family also has a page where every operator has an entry: its signature,
what it does, how it treats missing values, and examples, most with a worked example on toy data.

In the signatures:

- `x`, `y` are expressions: anything that gives a value per date and stock;
- `period`, `n`, `window`, `halflife` are whole numbers of days or values; `q`, `limit` are numbers;
- `'name'` is a quoted field or group name; `groups` is a group's name or any expression giving labels;
- `[, arg]` is an optional argument.

## The toy data

Four stocks, **A** to **D**, over six trading days. **C** has no price on Jan 04. **A** and **B** are Tech, **C** and
**D** are Energy. `surprise` is an earnings surprise, reported on four days; `beta` is each stock's market beta (1.4, 0.7, 1.1 and 0.8);
`price` is the as-traded close (with no splits or dividends, the same as `close()`); and the data also has a `market`
series.
The worked examples evaluate their expressions on this data when the site is built, so every number is what qex
computes. A dot is a missing value.

```toy inputs
close()
volume(1)
field('surprise')
group('sector')
```

## Data and primitives

[Data and primitives in detail →](ops/data.md)

| Operation | Description |
|-----------|-------------|
| `close()` | Each stock's closing price, adjusted for splits and dividends |
| `open()` | Opening price, adjusted |
| `high()` | Daily high, adjusted |
| `low()` | Daily low, adjusted |
| `field('name')` | Any field of the data by name, read as numbers (read labels with `group('name')`) |
| `market()` | The market's price level: the data's `market` series, the same for every stock |
| `returns(period[, 'field'])` | Return over `period` days, of `close` or another price field: p[t] / p[last price at or before t − period] − 1 |
| `volatility(period[, 'field'])` | Annualized volatility: the standard deviation of daily returns over `period` days, times √252. Part of the window |
| `volume(period)` | Average daily volume over `period` days, in shares. Part of the window |
| `adv(period)` | Average daily dollar volume, close × volume, over `period` days. Part of the window |

## Math

[Math in detail →](ops/math.md)

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

## Time series

Each stock's own history. Each operator follows one of three [window rules](ops/time-series.md#window-rules). [Time series in detail →](ops/time-series.md)

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

### Lag and difference

| Operation | Description |
|-----------|-------------|
| `delay(x, period)` | The value `period` days ago; `period` ≥ 0 |
| `delta(x, period)` | Today's value minus the value `period` days ago: `x - delay(x, period)` |

### Ranking and extremes

| Operation | Description |
|-----------|-------------|
| `ts_rank(x, period)` | Today's percentile among the last `period` values, from 1/period to 1; ties share their average rank. The whole window |
| `ts_argmax(x, period)` | Days since the largest value in the window, 0 if it is today. The whole window |
| `ts_argmin(x, period)` | Days since the smallest value in the window. The whole window |

### Pairs of series

Regressions and correlations through time: one fit per stock, over its own window. For a regression across
the stocks on each day, see `regression_neut` under Cross-sectional.

| Operation | Description |
|-----------|-------------|
| `ts_corr(x, y, period[, min_periods])` | Correlation of `x` and `y` over the window. Part of the window |
| `ts_cov(x, y, period[, min_periods])` | Covariance over the window. Part of the window |
| `ts_beta(y, x, period[, min_periods])` | Least-squares slope of `y` on `x`, cov(y, x) / var(x), over the window. Part of the window |
| `ts_beta_shrunk(y, x, period[, min_periods][, groups])` | `ts_beta`, pulled toward the day's average beta (or with `groups`, its group's) in proportion to its noise (Vasicek). Part of the window |
| `ts_alpha(y, x1, ..., xk, period[, min_periods])` | Intercept of `y` on one or more regressors over the window, from one joint least-squares fit on the days all have a value; with one, mean(y) − beta × mean(x). Part of the window |
| `ts_alpha_shrunk(y, x, period[, min_periods][, groups])` | The intercept that goes with `ts_beta_shrunk`'s slope. Part of the window |
| `ts_resid(y, x1, ..., xk, period[, min_periods])` | Today's residual from the window's fit, y − alpha − Σ beta_j × x_j: the part of today's `y` the regressors don't explain. Part of the window |
| `ts_resid_shrunk(y, x, period[, min_periods][, groups])` | Today's residual from `ts_beta_shrunk`'s fit. Part of the window |

### Exponentially weighted

| Operation | Description |
|-----------|-------------|
| `ewma(x, halflife)` | Exponentially weighted mean: a day's weight halves every `halflife` days. Exponential |
| `ewma_var(x, halflife)` | Exponentially weighted variance. Exponential |
| `ewma_cov(x, y, halflife)` | Exponentially weighted covariance. Exponential |
| `ts_beta_ewma(y, x, halflife)` | ewma_cov(y, x) / ewma_var(x), over the days both have a value. Exponential |

### Shape of the window

| Operation | Description |
|-----------|-------------|
| `ts_skew(x, period)` | Skewness of the window. The whole window |
| `ts_kurt(x, period)` | Excess kurtosis of the window. The whole window |
| `decay_linear(x, period)` | Weighted mean with weights period, period − 1, ..., 1, newest heaviest. The whole window |

## Events and state

[Events and state in detail →](ops/events.md)

| Operation | Description |
|-----------|-------------|
| `fill_forward(x[, limit])` | Carry each value forward over the missing days after it, at most `limit` days; with no limit, until the next value |
| `latch(on, off)` | 1 from a day `on` is true until a day `off` is true (off wins a tie), then 0 until `on` again; 0 before `on` is first true; a day where either is missing changes nothing |
| `ts_mean_events(x, n[, min_n])` | Mean of the last `n` values `x` has had, however far back; a value every day once there are `min_n` (default `n`) |
| `ts_std_events(x, n[, min_n])` | Standard deviation of the last `n` values, once there are `min_n` |
| `ts_sum_events(x, n[, min_n])` | Sum of the last `n` values, once there are `min_n` |
| `ts_count_events(x, period)` | How many values `x` has in the last `period` days; an answer every day once `period` days have passed |

## Cross-sectional

Across the stocks in the universe, one day at a time. [Cross-sectional in detail →](ops/cross-sectional.md)

| Operation | Description |
|-----------|-------------|
| `rank(x)` | Percentile rank among the day's stocks, from 1/n to 1, lowest first; ties share their average rank |
| `zscore(x)` | (x − mean) / standard deviation across the day's stocks; missing when they don't vary |
| `demean(x)` | x minus the day's mean |
| `scale(x)` | Divide by the day's sum of absolute values, so they sum to 1 |
| `quantile(x, n)` | Bucket 1 (lowest) to `n` by percentile rank |
| `winsorize(x, limit)` | Clip at the day's `limit` and 1 − `limit` quantiles: `winsorize(x, 0.05)` caps at the 5th and 95th percentiles |
| `truncate(x, max)` | Clip each value to [−max, max] (element by element, not across stocks) |
| `regression_neut(y, x1, x2, ...)` | One regression per day, across the stocks (not through time): `y` with its least-squares fit on the regressors (and an intercept) taken out |
| `regression_alpha(y, x1, ..., xk)` | The intercept of `regression_neut`'s fit: one number per day, the same for every stock |
| `regression_beta(y, x1, ..., xk, j)` | The `j`-th slope of `regression_neut`'s fit: one number per day, the same for every stock; with one regressor `j` may be left out |
| `pca_factor(x, window, k[, refit])` | Each stock's beta to the k-th statistical factor of `x`, from a principal component analysis across stocks, refit every `refit` days (default 21) |

## Groups

Within groups of stocks sharing a label, one day at a time. [Groups in detail →](ops/groups.md)

| Operation | Description |
|-----------|-------------|
| `group('name')` | The labels themselves, for comparing or grouping: `group('sector') == 'Technology'` |
| `identifier()` | Each stock's own identifier, as a label; through `ts_match`, its twin's |
| `group_cross(a, b, ...)` | One grouping from several: labels joined with a bar, `'Technology\|3'` |
| `group_demean(x, groups)` | x minus its group's mean |
| `group_rank(x, groups)` | Percentile rank within each group, from 1/n to 1 |
| `group_quantile(x, groups, n)` | Bucket 1 to `n` within each group: a dependent double sort |
| `group_std(x, groups)` | Standard deviation across the group's stocks that day; every member gets it |
| `group_sum(x, groups)` | Sum within the group; every member gets it; missing when the group has no values that day |
| `group_count_valid(x, groups, period)` | How many values the group's stocks have had over the last `period` days |

## Matching

Each stock's statistical twin: one particular other stock, chosen from history. [Matching in detail →](ops/matching.md)

| Operation | Description |
|-----------|-------------|
| `ts_match(value, key, lookback, metric='corr', within=None, matching='nearest', rebalance=21, min_overlap=None)` | The `value` of each stock's twin: the stock whose `key` over the last `lookback` days was most alike, chosen every `rebalance` days from the past only; options by name |
| `ts_match_score(key, lookback, metric='corr', within=None, matching='nearest', rebalance=21, min_overlap=None)` | How alike each stock and its twin were when chosen: a correlation, or a distance for the distance metrics; reads the same selection as `ts_match`; options by name |

## Logic and conditions

[Logic and conditions in detail →](ops/logic.md)

| Operation | Description |
|-----------|-------------|
| `where(cond, if_true, if_false)` | `if_true` where `cond` is nonzero, `if_false` where it is 0, missing where it is missing |
| `coalesce(x, y)` | `x` where it has a value, else `y` (a number, label or expression) |
| `is_valid(x)` | 1 where `x` has a value, else 0; an infinity counts as a value |
| `is_finite(x)` | 1 where `x` is a finite number, 0 where it is missing or infinite |
| `None` | No value here, on purpose: `where(cond, x, None)` |

| Operator | Description |
|----------|-------------|
| `+`, `-`, `*`, `/` | Arithmetic, element by element |
| `**` | Power, with a number as the exponent |
| `>`, `<`, `>=`, `<=`, `==`, `!=` | 1 or 0, missing where either side is missing |
| `&`, <code>&#124;</code>, `~` | And, or, not, three-valued |

## Calendar

[Calendar in detail →](ops/calendar.md)

| Operation | Description |
|-----------|-------------|
| `day_of_week()` | 0 (Monday) to 6 (Sunday) |
| `day_of_month()` | 1 to 31 |
| `month_of_year()` | 1 to 12 |
