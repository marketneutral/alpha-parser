# Changelog

## 0.6.0

Missing values get a literal and a reason. **No number changes**: both are additions, and expressions that do not
write `None` evaluate exactly as before.

### `None`: no value here, on purpose

`where(cond, x, None)` leaves `x` out where `cond` is false. `None` is missing like any NaN (cross-sections and a
book leave it out, arithmetic with it stays missing, `is_valid` is 0), which is what a `0` was often wrongly written
for: a book demeans its signal, so a `0` is a position. It replaces the `x * 0 / 0` trick. The documentation's
examples that wrote `0` to mean "not this stock" now write `None`.

### `explain_missing`: why a value is missing, and where it began

`explain_missing(signal, data, universe=...)` traces every missing cell of a signal through its graph and reports a
reason (no data, outside the universe, short history, excluded on purpose, no group, calculation error, infinite
value) and the node where it began. A short history says why the earlier days were missing, so a new member's
`ts_mean(rank(x), 10)` reads "short history: outside the universe on earlier days". `report.summary(within=...)`
gives each reason's share, and `report.on(date)` each missing name's reason that day. It reads the values the
signal's own evaluation produced and changes none of them.

## 0.5.2

Bug fixes, from the reports in qexarena's `IDEAS.md`. **Numbers change** for `ts_beta` and
`ts_beta_ewma` on names with missing values, for cross-sections containing an infinity, and for
everything `FactorRiskModel` produces.

### `ts_beta` and `ts_beta_ewma` pair their observations

The covariance was estimated on the days both inputs were valid, but the variance of `x` on all
of `x`'s days, so a hole in `y` (a new listing, a halt) biased the beta. Both now use only the
days where both are valid, as the documentation already said and as least squares does. Names
with no missing values give the same betas as before. The workaround
`ts_beta(y, x + 0 * y, n)` is no longer needed. (#20)

### One infinite value no longer empties a cross-section

`zscore`, `demean`, `winsorize`, `scale`, `regression_neut`, `group_demean`, `group_std` and
`group_sum` now leave infinite values out of the day's statistics, and the infinite name gets NaN.
Before, one zero volume in `zscore(volume(20) / volume(1))` gave NaN for every stock, `demean`
gave −inf for every other stock, and `scale` gave them all 0. Rank-based operators are unchanged.

### `FactorRiskModel`

Three fixes, each of which changed every number the model produced. On simulated data with one
market factor and a 1-day reversal factor (750 days, 300 stocks), before and after:

| | 0.5.1 | 0.5.2 |
|---|---|---|
| Reversal factor return Sharpe (true value 0) | −300 | −0.6 |
| `diag(factor_covariance) / (252 · var(factor_returns))` | 0.0013 | 1.0 |
| Equal-weight long book: forecast vol (realized 19.3%) | 0.0% | 19.4% |

- **No look-ahead.** Day t's returns are now regressed on the exposures known at day t−1's
  close. Before, same-day exposures let a factor built from day t's return (reversal,
  volatility) explain that return with hindsight.
- **Covariance and specific variance were about n times too small.** The weights were
  normalized to sum to one and then the weighted series passed to `.cov()`. They are now a
  proper weighted covariance (equal to `.cov()` with equal weights, and to `numpy.cov` with
  `aweights`).
- **The market factor is in the exposures.** It was in the factor covariance but not in
  `factor_exposures`, so `portfolio_risk` and `factor_attribution` left it out. Every stock now
  has exposure 1 to `market`, in the fitted exposures and in `get_exposures`.

### Smaller fixes

- `ts_rank` is vectorized: about 0.4 s for 600 names over 1,500 days with a 252-day window,
  against about a minute. Results are identical.
- Math functions accept numbers: `sqrt(5)`, `log(2)`, `abs(-3)`, `power(2.0, x)` and an exponent
  that is a signal no longer raise `AttributeError`.
- A signed number in a function argument is a number: `delay(close(), -1)` now says
  `got -1` instead of printing an object, and `power(x, -1)` works.
- CI installs the `risk` extra, so the risk model's tests run.

## 0.5.1

Bug fixes. **Numbers change** for any data with missing prices (halts, gaps, delistings):
on pandas 2 the signal side changes, on pandas 3 backtest and quantile/IC results change,
and backtests change on both. Data with no missing prices gives the same results as 0.5.0.

### Returns no longer depend on the pandas version (#21)

Every return was `pct_change` with the pandas default for missing prices, which changed
between pandas 2 (carry the last price forward: fake 0% returns during halts and forever
after a delisting) and pandas 3 (no bridging: the move across a halt is lost). All six call
sites (`returns`, `volatility`, the risk model's returns, and the forward returns of
`Backtest`, `QuantileAnalysis.run` and `QuantileAnalysis.ic_summary`) now use one rule,
`qex.price_returns`:

- a day with no price has a NaN return;
- a missing start price is replaced by the last valid price before it, so the reopen day
  carries the whole move since the last price, and daily returns compound to n-day returns;
- a price more than `MAX_PRICE_GAP = 10` rows old no longer counts.

New: `price_returns`, `forward_returns`, `MAX_PRICE_GAP`.

### Backtests hold positions in names that can't be traded

A name with no close on a day (halted, or delisted) used to get weight 0 that day, which
closed the position at a price that doesn't exist and dropped the move across the halt
from P&L. `Backtest` now holds the previous weight, marked at the last price, until the
name trades again, for at most `MAX_PRICE_GAP` days; after that the position is dropped
with no P&L for the missing move (delisting returns are not modelled). Held weights are not
rescaled, so gross can differ slightly from 1 on those days. `build_book` still returns the
exact target book. New: `hold_untradable`.

### Python 3.9 is no longer supported

`requires-python` is now `>=3.10`. Python 3.9 reached end of life in October 2025, and
GitHub's current `ubuntu-latest` runner no longer provides it. CI tests 3.10, 3.11 and 3.12,
and one failing version no longer cancels the others (`fail-fast: false`).

### Other fixes

- `QuantileAnalysis.ic_summary` raised `KeyError` when no day had at least 5 names; it now
  returns NaN statistics.
