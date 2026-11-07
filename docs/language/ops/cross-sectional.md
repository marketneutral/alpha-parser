# Cross-Sectional

Across the stocks on each day, one day at a time. Each compares the stocks in the universe that day and gives every
other stock no value ([Universes](../universes.md)). A stock with no value takes no part: on Jan 04 below, the day has
three stocks, not four.

## Ranking and scaling

### rank

<p class="sig"><code>rank(x)</code></p>

Each stock's percentile among the day's stocks with a value, from `1/n` (the lowest) to `1` (the highest); ties share
their average rank. An infinity ranks first or last.

```toy
returns(1)
rank(returns(1))
```

```qex
rank(returns(20))                    # higher return, higher rank
rank(-returns(5))                    # higher return, lower rank: a reversal
```

### zscore

<p class="sig"><code>zscore(x)</code></p>

`(x − mean) / standard deviation` across the day's stocks. No value on a day the stocks don't vary.

```toy
returns(1)
zscore(returns(1))
```

### demean

<p class="sig"><code>demean(x)</code></p>

`x` minus the day's mean: the result sums to zero, the weights of a dollar-neutral book.

```toy
rank(returns(1))
demean(rank(returns(1)))
```

### scale

<p class="sig"><code>scale(x)</code></p>

`x` divided by the day's sum of absolute values, so the absolute values sum to 1: weights with a gross of one.

```toy
demean(returns(1))
scale(demean(returns(1)))
```

### quantile

<p class="sig"><code>quantile(x, n)</code></p>

The bucket each stock falls in, 1 (lowest) to `n`, by percentile rank. Buckets are numbers, and also labels: a
`quantile` can be the `groups` of a [group operator](groups.md).

```toy
returns(1)
quantile(returns(1), 2)
```

```qex
quantile(field('mcap'), 5)           # size quintiles, 1 the smallest
```

## Clipping

### winsorize

<p class="sig"><code>winsorize(x, limit)</code></p>

Clips each day's values at that day's `limit` and `1 − limit` quantiles: `winsorize(x, 0.05)` caps at the 5th and 95th
percentiles. Below, with four stocks and `limit` 0.25, **A** and **D** are pulled in to the 25th and 75th percentiles.

```toy
close()
winsorize(close(), 0.25)
```

```qex
winsorize(returns(1), 0.01)          # cap the 1% tails
```

### truncate

<p class="sig"><code>truncate(x, max)</code></p>

Clips each value to `[−max, max]`. Element by element, not across stocks: the limit is a number, not a percentile.

```toy
returns(1)
truncate(returns(1), 0.05)
```

## Neutralizing

### regression_neut

<p class="sig"><code>regression_neut(y, x1, x2, ...)</code></p>

Takes out of `y`, each day, the part the regressors explain: the residual of a least-squares fit of `y` on them, with
an intercept, over the stocks where all have a value. The result has zero mean and zero covariance with each
regressor across the day's stocks, which is what "neutral to x" means. Below, the day's returns with the part
explained by price level taken out.

```toy
returns(1)
close()
regression_neut(returns(1), close())
```

With several regressors the fit is joint, and the result is neutral to each. That is not the same as nesting,
`regression_neut(regression_neut(y, beta), size)`: when the regressors are correlated, the second step brings back
some of what the first took out. Regressors that duplicate each other are allowed. A day needs at least k + 2 stocks
for k regressors, or it has no value; a day a regressor doesn't vary gives `y` demeaned.

```qex
let beta = ts_beta_shrunk(returns(1), returns(1, 'market'), 252),
    size = log(field('mcap'))
in regression_neut(zscore(-returns(5)), beta, size)
```

The signal is exactly neutral; a book built from it is close to neutral, since a book's caps and scaling come after.

### pca_factor

<p class="sig"><code>pca_factor(x, window, k[, refit])</code></p>

Each stock's beta to the k-th statistical factor of `x` (usually `returns(1)`): the common factors are found from the
data alone, by a principal component analysis across stocks.

- **Point in time.** Every `refit` days (default 21; the first once `window` days exist), the analysis uses the last
  `window` days of `x`, ending that day, and holds until the next refit. Before the first, no value.
- **Which stocks.** Those with values on at least 90% of the window, and in the universe that day.
- **Correlation, not covariance.** Each stock is scaled by its own volatility first, so a few volatile stocks don't
  make up the first factors.
- **A beta.** The value is the stock's beta to the factor's return, in `x`'s units: the exposure a position carries,
  and what `regression_neut` should take out.
- **Factors keep their identity.** Each refit's factors are matched to the last refit's and signed to agree, so factor
  2 stays factor 2. The first factor is the market, signed long. `k` is at most 20.
- **Shared.** `pca_factor` with the same `x`, `window` and `refit` and different `k` share one decomposition.

```qex
let r = returns(1)
in regression_neut(zscore(-returns(5)), pca_factor(r, 252, 1), pca_factor(r, 252, 2), pca_factor(r, 252, 3))
```

## Infinities

`zscore`, `demean`, `winsorize`, `scale` and `regression_neut` leave an infinite value out of the day's statistics and
give that stock no value, so one division by zero (a zero-volume day in `volume(20) / volume(1)`) can't empty the
day. `rank` and `quantile` rank an infinity first or last.
