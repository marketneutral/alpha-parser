# Adjusted data and look-ahead

Most price data comes in two forms, and they answer different questions.

| | what it is | use it for | never use it for |
|---|---|---|---|
| **as traded** (`field('price')`, `field('volume_traded')`, `field('shares')`, `field('mcap')`) | the number printed or filed that day | dollar floors, price levels, market cap, dollars traded | returns: it jumps at every split and dividend |
| **adjusted** (`close()`, `open()`, `high()`, `low()`, `volume`) | the as-traded number rescaled for every split and dividend *after* that day | returns, volatility, a stock against its own past | comparing stocks, or comparing with a fixed number |

## Why adjusted levels look ahead

An adjusted price is the price as traded times a factor A(t) that folds in every split and dividend after day t, up
to the end of the data. On 2000-01-03 AAPL traded at $111.94 and NVDA at $46.81; their adjusted closes for that day are
0.84 and 0.09. NVDA's is tiny because NVDA later rose enough to split six times, and every split divides its past.

So on any past day, a low adjusted level partly means *this stock will do well*. `rank(-close())` (buy the lowest
adjusted price) had a validation Sharpe of 1.34 in one backtest; `rank(-field('price'))`, the same idea on the price that
traded, had 0.34. The difference is the future.

Returns are safe. Between two days the part of A that differs is the events between them, which the later day
knows, so `close() / delay(close(), 21)` and `returns(21)` are point in time.

## The rule qex checks

Each field declares an adjustment exponent: +1 for an adjusted price, −1 for adjusted volume (as traded ÷ A), 0 for
anything as traded or as filed. The check follows the exponent through the expression:

- multiplying adds exponents and dividing subtracts them: `close() * volume(20)` is 0, dollars traded;
- `delay` and the time-series operators keep it: `ts_mean(close(), 20)` is still an adjusted level;
- scale-free time-series statistics (`ts_rank`, `ts_corr`, `ts_argmax`, ...) give 0;
- `log` turns the exponent into a shift, and a difference over time cancels it: `log(close()) - log(delay(close(), 5))`;
- adding, subtracting and comparing need the same exponent on both sides;
- comparing with a constant needs 0;
- every cross-sectional and group operator needs 0, and so does an alpha's value (its book compares stocks).

| expression | |
|---|---|
| `close() > 5` | refused: compare `field('price') > 5` |
| `rank(close())`, `rank(log(close()))`, `zscore(close())` | refused |
| `group_demean(close(), 'sector')` | refused |
| `field('price') * volume(20)` | refused: the adjustments don't cancel |
| `field('price') + field('shares')` | refused: dollars per share plus shares |
| `close() / ts_max(close(), 252)` | fine |
| `returns(5)`, `volatility(63)`, `close() * volume(20)` | fine |
| `field('price') > 5`, `rank(field('mcap'))` | fine |

An expression the rule refuses is refused before it runs, with a message saying what to write instead. From Python
the check is `qex.adjust.check` ([Diagnostics](../python/diagnostics.md#adjusted-data-and-look-ahead-qexadjustcheck)),
and the declarations come with the data ([Data Format](../data/format.md)). A field without a declaration is not
checked, and neither is anything computed from it.
