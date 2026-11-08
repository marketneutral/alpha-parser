# Factor Premia and Timing

Signals built from the day's cross-sectional regression: what a characteristic paid each day, and how to trade on it.

## A factor's return, day by day

Regress the day's stock returns on a characteristic, across the stocks. The slope is how much more a stock with one
more unit of the characteristic earned that day: the characteristic's **return** that day. The intercept is what a
stock with none of it earned. `regression_beta` and `regression_alpha` give those two numbers
([Cross-Sectional](../language/ops/cross-sectional.md#regression_beta)):

```qex
let z = zscore(delay(field('book_to_price'), 1))
in regression_beta(returns(1), z, 1)          # value's return each day
```

This is the Fama-MacBeth procedure (Fama and MacBeth, 1973): one regression per day, and the average of the daily
slopes over time is the characteristic's **premium**. With several characteristics in one regression, each slope is
net of the others.

Two rules for everything on this page:

- **Lag the characteristics.** Regress today's return on characteristics known at yesterday's close, the
  `delay(..., 1)` above. A factor's return is what a portfolio formed *before* the day earned *during* it. Today's
  book-to-price moves with today's price, so a same-day regression mostly measures that mechanical link (price up,
  book-to-price down), not value's return. This is about what you measure, not about safety: the engine already rules
  out look-ahead ([No lookahead by construction](../philosophy.md#no-lookahead-by-construction)).
- **Average the premiums.** One day's slope is mostly noise; the signals below use averages over months.

A day's premium is one number for every stock, so on its own it is not a signal: a book demeans it away. Each signal
below multiplies a stock's characteristic by something learned from the premium's history.

## Factor momentum

Factors that have paid recently tend to keep paying (Ehsani and Linnainmaa, 2022). Hold a characteristic in the
direction its premium has had over the last six months:

```qex
let z    = zscore(delay(field('book_to_price'), 1)),
    prem = regression_beta(returns(1), z, 1)
in sign(ts_mean(prem, 126)) * z
```

Long value when value has been working, short it when it hasn't. `sign(...)` decides only the direction; the stock
weights come from `z`.

## Volatility-managed factors

A factor's risk moves around more than its return, so holding less of a factor when it is volatile improves its
Sharpe ratio (Moreira and Muir, 2017). In qex this works when it changes the **mix** of factors. Blend reversal and
momentum, each divided by its own premium's volatility over the last quarter:

```qex
let rev   = zscore(delay(-returns(5), 1)),
    mom   = zscore(delay(returns(231), 22)),
    p_rev = regression_beta(returns(1), rev, mom, 1),
    p_mom = regression_beta(returns(1), rev, mom, 2)
in rev / ts_std(p_rev, 63) + mom / ts_std(p_mom, 63)
```

When reversal turns turbulent its share of the book shrinks and momentum's grows, and the other way round: a
risk-parity blend that adapts. Both premiums come from one regression, so each is net of the other.

Scaling a **single** factor this way, `rev / ts_std(p_rev, 63)`, does nothing to a qex book: every stock is divided
by the same number that day, and the book is scaled back to a gross of one. The timing Moreira and Muir describe
changes the book's size, which is a decision about leverage, outside the signal.

## Letting the data weight the characteristics

Instead of an equal blend of value, momentum and size, weight each by its premium over the last year, from one
regression in which each premium is net of the others (Lewellen, 2015):

```qex
let v = zscore(delay(field('book_to_price'), 1)),
    m = zscore(delay(returns(231), 22)),
    s = zscore(delay(-log(field('mcap')), 1))
in ts_mean(regression_beta(returns(1), v, m, s, 1), 252) * v
 + ts_mean(regression_beta(returns(1), v, m, s, 2), 252) * m
 + ts_mean(regression_beta(returns(1), v, m, s, 3), 252) * s
```

Each stock's score is its expected return under the fitted model: its characteristics times what each has been
paying. The three `regression_beta`s read one fit: qex computes the regression once.

## Timing the beta premium

Regress returns on the stocks' betas. The slope is the day's return for bearing beta; the intercept, the return of a
zero-beta stock. In theory the slope averages to the market's excess return; in the data it is much flatter, which
is the low-beta anomaly (Frazzini and Pedersen, 2014). Trade it only while it is there:

```qex
let b    = delay(ts_beta_shrunk(returns(1), returns(1, 'market'), 252), 1),
    prem = regression_beta(returns(1), b, 1)
in sign(ts_mean(prem, 252)) * zscore(b)
```

When high beta has not been paid over the last year (the premium's average is negative), this is short high-beta
stocks and long low-beta ones; when beta has been paid, it flips. The zero-beta return is the other half of the
story:

```qex
let b = delay(ts_beta_shrunk(returns(1), returns(1, 'market'), 252), 1)
in ts_mean(regression_alpha(returns(1), b), 252)       # the zero-beta portfolio's average daily return
```

## Watching your own signal

Use a signal as the regressor and its slope is the signal's own return each day, net of an intercept. Its recent
average says whether the signal is still paying:

```qex
let s = zscore(delay(rank(-returns(5)), 1))
in ts_mean(regression_beta(returns(1), s, 1), 63)
```

That is a monitor rather than a trade, but other signals can be gated on it, for example
`where(ts_mean(regression_beta(returns(1), s, 1), 63) > 0, s, None)` inside the same `let`.

## References

- Ehsani, S. and Linnainmaa, J. T. (2022). Factor momentum and the momentum factor. *Journal of Finance* 77(3),
  1877-1919.
- Fama, E. F. and MacBeth, J. D. (1973). Risk, return, and equilibrium: Empirical tests. *Journal of Political
  Economy* 81(3), 607-636.
- Frazzini, A. and Pedersen, L. H. (2014). Betting against beta. *Journal of Financial Economics* 111(1), 1-25.
- Lewellen, J. (2015). The cross-section of expected stock returns. *Critical Finance Review* 4(1), 1-44.
- Moreira, A. and Muir, T. (2017). Volatility-managed portfolios. *Journal of Finance* 72(4), 1611-1644.
