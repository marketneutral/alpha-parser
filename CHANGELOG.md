# Changelog

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
