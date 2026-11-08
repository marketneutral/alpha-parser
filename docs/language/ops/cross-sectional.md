# Cross-Sectional

Across the stocks on each day, one day at a time. Each compares the stocks in the universe that day and gives every
other stock no value ([Universes](../universes.md)). A stock with no value takes no part: on Jan 04 below, the day has
three stocks, not four.

## Ranking and scaling

### rank

<p class="sig"><code>rank(x)</code></p>

Each stock's percentile among the day's stocks with a value, from `1/n` (the lowest) to `1` (the highest); ties share
their average rank. An infinity ranks first or last. With $n$ stocks:

$$
\operatorname{rank}(x)_i = \frac{1}{n}\left(\#\{j : x_j < x_i\} + \frac{\#\{j : x_j = x_i\} + 1}{2}\right)
$$

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

The distance from the day's mean in standard deviations. No value on a day the stocks don't vary.

$$
z_i = \frac{x_i - \bar x}{s},
\qquad
s^2 = \frac{1}{n - 1} \sum_j (x_j - \bar x)^2
$$

```toy
returns(1)
zscore(returns(1))
```

### gaussrank

<p class="sig"><code>gaussrank(x)</code></p>

The day's rank mapped through the inverse of the normal distribution: a z-score's shape without its outliers. With
$r_i$ the stock's rank from 1 to $n$ among the day's stocks with a value,

$$
\operatorname{gaussrank}(x)_i = \Phi^{-1}\!\left(\frac{r_i - 0.5}{n}\right)
$$

```toy
returns(1)
gaussrank(returns(1))
```

With four stocks the values are always $\Phi^{-1}$ of 1/8, 3/8, 5/8 and 7/8: ±1.150 and ±0.319. On Jan 04, with
three, they are ±0.967 and 0.

How it compares with its neighbors:

| | `rank(x)` | `zscore(x)` | `gaussrank(x)` |
|---|---|---|---|
| Shape | uniform, from 1/n to 1 | whatever x's shape is | normal, whatever x's shape is |
| One extreme stock | takes the top rank, no more | can dominate the day | takes the top score, no more |
| Uses x's values | only their order | their sizes | only their order |
| Sums to zero | no (centered on about ½) | yes | yes |

So `gaussrank` keeps `rank`'s robustness (only the order matters, so any increasing transform of `x` gives the same
result) and gives values on `zscore`'s scale, with the weight in the tails a normal sample has: the extremes count
more than in `rank`, and no single outlier counts more than a normal sample's extreme would. That makes it the natural
way to put several signals on one scale before combining them:

```qex
gaussrank(field('book_to_price')) + gaussrank(returns(231)) + gaussrank(-volatility(63))
```

Tied stocks share the average of the scores they occupy, so every day still sums to zero. An infinity ranks first or
last, as in `rank`.

### demean

<p class="sig"><code>demean(x)</code></p>

`x` minus the day's mean: the result sums to zero, the weights of a dollar-neutral book.

```toy
rank(returns(1))
demean(rank(returns(1)))
```

### scale

<p class="sig"><code>scale(x)</code></p>

`x` divided by the day's sum of absolute values, so the absolute values sum to 1: weights with a gross of one,

$$
w_i = \frac{x_i}{\sum_j |x_j|}
$$

Below,
a one-day reversal book: long yesterday's losers, short its winners, half the gross on each side.

```toy
demean(-returns(1))
scale(demean(-returns(1)))
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
percentiles. Below, with four stocks and `limit` 0.25, each day's highest and lowest returns are pulled in to the
25th and 75th percentiles: **A**'s 18% on Jan 08 no longer dominates the day.

```toy
returns(1)
winsorize(returns(1), 0.25)
```

```qex
winsorize(returns(1), 0.01)          # cap the 1% tails
```

### truncate

<p class="sig"><code>truncate(x, max)</code></p>

Clips each value to `[−max, max]`. Element by element, not across stocks: the limit is a number, not a percentile.
Its use is a position cap: below, the reversal book from [`scale`](#scale) with no position above 30% of the gross.
After the cap the gross is under one, and a book built from the signal scales it back.

```toy
scale(demean(-returns(1)))
truncate(scale(demean(-returns(1))), 0.3)
```

## Neutralizing

### regression_neut

<p class="sig"><code>regression_neut(y, x1, x2, ...)</code></p>

!!! note "A regression across stocks, not through time"
    `regression_neut` fits one regression **per day, across the stocks**: its data points are that day's stocks, and
    each `x` is a value per stock, such as each stock's beta or size. It is not a regression through time; for one
    stock's regression on the market over its history, use [`ts_beta`](time-series.md#ts_beta),
    [`ts_alpha`](time-series.md#ts_alpha) and [`ts_resid`](time-series.md#ts_resid).
    [Two kinds of regression](../concepts.md#two-kinds-of-regression) sets them side by side.

Takes out of `y`, each day, the part the regressors explain. Across the day's stocks with a value for all of them, it
fits

$$
y_i = a + \sum_{j=1}^{k} b_j \, x_{ij} + \varepsilon_i
$$

by least squares, and gives the residual $\hat\varepsilon_i = y_i - \hat a - \sum_j \hat b_j x_{ij}$. Least squares makes
the residual orthogonal to the intercept and to every regressor:

$$
\sum_i \hat\varepsilon_i = 0,
\qquad
\sum_i x_{ij} \, \hat\varepsilon_i = 0 \quad \text{for every } j
$$

so it has zero mean and zero covariance with each regressor, which is what "neutral to x" means. With one regressor
it is

$$
\hat\varepsilon_i = (y_i - \bar y) - \hat b \, (x_i - \bar x),
\qquad
\hat b = \frac{\sum_i (x_i - \bar x)(y_i - \bar y)}{\sum_i (x_i - \bar x)^2}
$$

Below, each stock's market beta is taken out
of the day's returns: what is left is the part of each return that beta doesn't explain. On Jan 03, **A** was up 10%,
but with a beta of 1.4 almost all of that lines up with beta, and 0.9% is left; **D**, up 5% with a beta of 0.8, has
the largest return beta doesn't explain.

```toy
returns(1)
field('beta')
regression_neut(returns(1), field('beta'))
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

### regression_alpha

<p class="sig"><code>regression_alpha(y, x1, ..., xk)</code></p>

The intercept $\hat a$ of `regression_neut`'s fit, the same regression across the day's stocks:

$$
y_i = a + \sum_{j=1}^{k} b_j \, x_{ij} + \varepsilon_i
\qquad\Longrightarrow\qquad
\hat a = \bar y - \sum_{j=1}^{k} \hat b_j \, \bar x_j
$$

**One number per day, the same for every stock.** The intercept belongs to the day's cross-section, not to a stock,
so every stock gets it, in the universe or not, and its history is the same for all. On a day with too few stocks
for a fit (fewer than k + 2), every stock has no value.

With today's returns as `y` and characteristics known at yesterday's close as the regressors, it is the return of a
stock with none of the characteristics: with betas, the zero-beta portfolio's return.

```qex
let b = delay(ts_beta_shrunk(returns(1), returns(1, 'market'), 252), 1)
in ts_mean(regression_alpha(returns(1), b), 252)       # the zero-beta portfolio's average daily return
```

### regression_beta

<p class="sig"><code>regression_beta(y, x1, ..., xk, j)</code></p>

The `j`-th slope $\hat b_j$ of the same fit (`j` from 1 to k; with one regressor it may be left out). Like the
intercept, one number per day for every stock. With today's returns on characteristics known at yesterday's close,
it is the day's **return to the j-th characteristic**, net of the others; its average over time is the
characteristic's premium (Fama and MacBeth, 1973).

A day's slope is the same for every stock, so on its own it is not a signal: a book demeans it away. Signals multiply a
characteristic by something learned from its premium's history:

```qex
let z    = zscore(delay(field('book_to_price'), 1)),
    prem = regression_beta(returns(1), z, 1)
in sign(ts_mean(prem, 126)) * z                        # factor momentum: lean into value while it pays
```

Lag the characteristics (`delay(..., 1)`): regressed on characteristics from the same day, the slope rewards
whatever moved with today's return, which is look-ahead. [Factor Premia and Timing](../../examples/factor-premia.md)
builds factor momentum, volatility-managed blends, Fama-MacBeth weights and beta timing from these.

`regression_alpha` and every `regression_beta` on the same `y` and regressors read one internal fit node, computed
once. `regression_neut` on the same inputs fits the same regression on its own, and its residual is exactly
$y - \hat a - \sum_j \hat b_j x_j$.

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
  and what `regression_neut` should take out (below).
- **Factors keep their identity.** Each refit's factors are matched to the last refit's and signed to agree, so factor
  2 stays factor 2. The first factor is the market, signed long. `k` is at most 20.
- **Shared.** `pca_factor` with the same `x`, `window` and `refit` and different `k` share one decomposition.

**The math.** At a refit, take the window's $T$ days of $x$ for the $N$ stocks in the fit, and standardize each stock
by its own mean and volatility over the window (a missing day counts as the stock's mean):

$$
z_{ti} = \frac{x_{ti} - \bar x_i}{\sigma_i}
$$

The correlation matrix and its eigenvectors, largest eigenvalue first, are

$$
C = \frac{1}{T - 1} Z^\top Z,
\qquad
C \, v_k = \lambda_k \, v_k,
\qquad
\lVert v_k \rVert = 1
$$

The k-th factor's return is the eigenportfolio $f_{kt} = \sum_i v_{ik} \, z_{ti}$, with variance $\lambda_k$. Since
$\operatorname{cov}(z_i, f_k) = (C v_k)_i = \lambda_k v_{ik}$, each stock's beta to it, in $x$'s units, is simply

$$
\beta_{ik} = \frac{\operatorname{cov}(x_i, f_k)}{\operatorname{var}(f_k)} = \sigma_i \, v_{ik}
$$

```qex
let r = returns(1)
in regression_neut(zscore(-returns(5)), pca_factor(r, 252, 1), pca_factor(r, 252, 2), pca_factor(r, 252, 3))
```

## Infinities

`zscore`, `demean`, `winsorize`, `scale` and `regression_neut` leave an infinite value out of the day's statistics and
give that stock no value, so one division by zero (a zero-volume day in `volume(20) / volume(1)`) can't empty the
day. `rank` and `quantile` rank an infinity first or last.
