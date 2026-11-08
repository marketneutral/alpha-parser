# Market Beta

Measuring each stock's market beta, making it more reliable, and taking it out of a signal.

## Dollar-neutral is not beta-neutral

A book whose long and short dollars are equal (`sum(w) = 0`) still has a market exposure
whenever its longs and shorts have different betas:

```
book beta = sum(w * beta)
```

Long low-beta stocks and short high-beta ones, and the book is short the market even though
it is dollar-neutral. The same goes for any signal that correlates with beta: low volatility,
size, many value and quality measures, and their opposites. Such a book can show a good
Sharpe ratio over a rising market and owe all of it to the market. If neutrality matters,
take beta out explicitly and check the result.

## The market series

Put the market in the data as a date-indexed `pd.Series`, under the key `'market'`:

The data supplies the market as a series with one price per day, given to every stock
([Data Format](../data/format.md#date-only-fields-and-the-market)). `market()` is that series, and `returns(n, 'market')` is the market's return
over the same `n` days as a stock's `returns(n)`:

```qex
returns(1, 'market')    # the market's daily return, on every ticker
volatility(63, 'market')    # its 63-day annualized volatility
```

Which market? Use the one you would actually hedge with, and keep it consistent between
the signal and the checks you run on the book:

| Choice | When |
|---|---|
| A tradable index (SPY, a future) | You hedge with it, or want the exposure an investor would see |
| The equal-weight average of your universe | You want the market your book is drawn from |
| A total-return series | Always: a price series that drops on ex-dividend days adds noise to every beta |

## Estimating beta

```qex
ts_beta(returns(1), returns(1, 'market'), 252)
```

`ts_beta(y, x, period[, min_periods])` is the rolling OLS slope `cov(y, x) / var(x)` over
the last `period` days on which both are valid.

**Window.** A year of daily returns (252) is the usual compromise. Shorter windows track
changes in beta but are noisier; longer ones are stabler but slow.

**History.** By default a window needs only `min(20, period)` paired observations, so a
stock 20 days after listing gets a "252-day" beta from 20 days. Ask for more:

```qex
ts_beta(returns(1), returns(1, 'market'), 252, 126)    # NaN until 126 days are paired
```

**Asynchronous prices.** A stock that trades less than the index responds to the market a
day late, which biases its daily beta toward zero. Multi-day returns absorb the lag:

```qex
ts_beta(returns(3), returns(3, 'market'), 756)    # 3-day returns over 3 years
```

Frazzini and Pedersen (2014) split the estimate in two, with a long, low-frequency window for
the correlation and a shorter daily one for the volatilities:

```qex
let rho = ts_corr(returns(3), returns(3, 'market'), 1260),
    vol = volatility(252) / volatility(252, 'market')
in rho * vol
```

**Outliers.** One earnings jump or deal announcement can move a year's beta a long way.
A robust (Huber) regression would down-weight those days, but a rolling robust fit for
every stock and every day is slow. Clipping each stock's returns at a few times its own
recent volatility before the regression does most of the same work:

```qex
let r = returns(1),
    s = ts_std(r, 63)
in ts_beta(min(max(r, -4 * s), 4 * s), returns(1, 'market'), 252)
```

!!! warning "Clip in time, not across names"
    `winsorize()` is cross-sectional: it clips each day's returns at that day's percentiles
    across names. With a universe set (the default in `build_book`), it is NaN outside the
    universe, so `ts_beta(winsorize(returns(1), 0.01), ...)` has no history for a stock
    before the day it joined. The time-series clip above has no such gap.

## Shrinking beta

Estimated betas are noisy, and the noise is not spread evenly: a stock with a short history,
or one whose returns are mostly idiosyncratic, has a much less certain beta than a large,
liquid stock. Extreme estimates are more often noise than extreme betas, so they tend
to move back toward the middle in the next period.

`ts_beta_shrunk` pulls each estimate toward the cross-sectional mean in proportion to
its own uncertainty (Vasicek 1973):

```qex
ts_beta_shrunk(returns(1), returns(1, 'market'), 252)
```

For stock *i* on day *t*:

```
b_shrunk = w * b + (1 - w) * mean(b)

w = var(b) / (var(b) + se(b)^2)
```

- `b` is the stock's OLS beta over the window, as from `ts_beta`.
- `mean(b)` and `var(b)` are the mean and variance of all the stocks' betas that day: the
  prior. With a universe set, only stocks in the universe enter them.
- `se(b)^2` is the sampling variance of the stock's own estimate:
  `residual variance / ((n - 1) * var(x))`, where `n` is the number of paired observations
  in the window and the residual variance is `(var(y) - cov(y, x)^2 / var(x)) * (n - 1) / (n - 2)`.

This is the Bayesian posterior mean under a normal prior centred on the cross-sectional mean.
When the stock's own estimate is precise (`se` small relative to the spread of betas),
`w` is near 1 and it keeps its beta. When the estimate is imprecise, `w` is small and the
stock moves most of the way to the mean. A new listing with 25 days of history is shrunk far
harder than a stock with a full year. Every stock keeps a value, inside or outside the
universe; only the prior is restricted.

**Why not the Bloomberg adjusted beta?** Bloomberg's adjustment, `0.67 * b + 0.33`
(after Blume 1971), shrinks every beta by the same fraction toward 1. That is a linear
transformation: it moves betas closer together but keeps their order and their relative
spacing. `rank(beta)` is unchanged, and so is `regression_neut(x, beta)`, because the
regression's intercept and slope absorb any linear transformation. The fixed blend only
matters where beta is used in levels, for example to size an index hedge. Vasicek
shrinkage differs by stock, so it changes the order, and with it what a signal or a
neutralization sees.

**Limitations.**

- The prior variance `var(b)` is the spread of the *estimated* betas, which includes their
  noise, so it overstates the true spread and the shrinkage is a little weaker than the ideal.
- By default there is one prior for all stocks. To shrink toward each sector's average beta
  instead, give the groups: `ts_beta_shrunk(returns(1), returns(1, 'market'), 252, 'sector')`.
- Shrinkage reduces noise in the estimate; it does not fix bias. Asynchronous trading and
  outliers still need the multi-day returns and the clipping above, and they combine:
  `ts_beta_shrunk(min(max(r, -4 * s), 4 * s), returns(1, 'market'), 252)`.

## Alpha and the residual

The same regression gives two more numbers. Like `ts_beta`, both come from each stock's own history, one
regression per stock through time (the next section's `regression_neut` is the other kind, across the stocks on
each day). `ts_alpha` is its intercept: the return a stock earned beyond what its
beta to the market explains, on average over the window. `ts_resid` is today's leftover: today's return minus what
the fit predicts from today's market move.

```qex
ts_alpha(returns(1), returns(1, 'market'), 252) * 252        # last year's alpha, annualized
ts_resid(returns(1), returns(1, 'market'), 252)              # today's idiosyncratic return
```

Residuals make better momentum and reversal signals than raw returns: a stock that rose with a rising market only
did what its beta said, and a signal built on the residual doesn't load up on beta. Residual momentum over the past
year, skipping the last month, scaled by the residuals' own volatility, after Blitz, Huij and Martens (2011), who
take out three factors where this takes out the market alone:

```qex
let e = ts_resid_shrunk(returns(1), returns(1, 'market'), 252, 'sector')
in delay(ts_sum(e, 231), 21) / ts_std(e, 252)
```

With more than one regressor, both are one joint fit, net of every regressor at once. Given daily factor returns in
the data (each a series with one value per day), a three-factor alpha and residual momentum net of all three:

```qex
ts_alpha(returns(1), field('mkt'), field('smb'), field('hml'), 252) * 252

let e = ts_resid(returns(1), field('mkt'), field('smb'), field('hml'), 252)
in delay(ts_sum(e, 231), 21) / ts_std(e, 252)
```

The `_shrunk` versions take one regressor and use the shrunk beta, and its alpha changes with it: what the shrinkage takes off the beta,
the alpha picks up, times the market's average return over the window
([`ts_alpha_shrunk`](../language/ops/time-series.md#ts_alpha_shrunk)).

## Taking beta out of a signal

`regression_neut(y, x)` regresses `y` on `x` across the stocks on each day (a cross-sectional regression, not one
through time), with an intercept, and
returns the residual. The result has zero mean and zero covariance with `x` across stocks,
so a book with those weights has no exposure to `x`:

```qex
let b = ts_beta_shrunk(returns(1), returns(1, 'market'), 252)
in regression_neut(rank(-returns(5)), b)
```

The regression is ordinary least squares on purpose. Only OLS residuals are exactly
orthogonal to `x`; a robust (Huber) fit would down-weight some stocks and leave the
residual with some covariance with `x`. Make the *beta* robust (above), and keep the
neutralization OLS.

**Neutralize among the stocks you trade.** The fit must run over the same stocks as the book.
`build_book` does this by default: with `BookSettings(universe=...)`, every cross-sectional
operator, `regression_neut` included, compares only the stocks in the universe that day (see
[the cross-sectional universe](../language/concepts.md#which-stocks-a-cross-section-compares)).
Evaluating a signal on its own, outside a book, needs the same restriction
([Universes from Python](../python/universes.md)): a fit over every stock with data is not neutral among the
members that are traded.

**Rank before you neutralize.** The book is built after the regression: the weights are
demeaned, scaled to gross one and clipped at the truncation, and clipping is not linear.
When most of a signal's weight sits in a few stocks, those stocks get clipped and the
hedging term spread across all the others keeps most of its weight; the book can end up
with a large beta of the *opposite* sign. Ranking first spreads the weight out, so little is
clipped:

```qex
let b = ts_beta_shrunk(returns(1), returns(1, 'market'), 252),
    spiky = returns(20) / power(volatility(63), 4)
in regression_neut(rank(spiky), b)
```

rather than `regression_neut(spiky, b)`, whose weight is concentrated in the few
lowest-volatility stocks.

**More than one exposure.** `regression_neut(y, beta, size)` removes several variables in one fit, so the result is
neutral to each. Nesting one `regression_neut` inside another is not the same: when the variables are correlated, the
second step brings back some of the first.

## Checking the book

The neutralization uses estimated betas; the book is exposed to realized ones. Always check the realized market
exposure of the book itself: its P&L's correlation with the market, its realized beta, and its ex-ante beta day by
day. [Checking a Book's Beta](../python-examples/beta-check.md) does it in a few lines.

Expect the realized correlation to be close to zero, not exactly zero: betas are estimated and they drift. A
correlation that stays large means the estimate is poor for this signal (try the shrunk or multi-day beta) or the
truncation is undoing the hedge (rank first).

## References

- Blitz, D., Huij, J. and Martens, M. (2011). Residual momentum. *Journal of Empirical
  Finance* 18(3), 506-521.
- Blume, M. (1971). On the assessment of risk. *Journal of Finance* 26(1), 1-10.
- Vasicek, O. (1973). A note on using cross-sectional information in Bayesian estimation of
  security betas. *Journal of Finance* 28(5), 1233-1239.
- Frazzini, A. and Pedersen, L. H. (2014). Betting against beta. *Journal of Financial
  Economics* 111(1), 1-25.
