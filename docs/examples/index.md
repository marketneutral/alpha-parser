# About the Examples

These pages are signals written in qex: momentum, mean reversion, pairs, events, market beta, technical indicators,
and patterns that combine them. Every code block is a qex expression, not Python; paste one into
`qex("...")`. Each page explains the idea, gives the expression, says how it reads, and covers
variants and what to watch for.

What each operator does is in the [Operator Reference](../language/reference.md). To run an example in a backtest,
see [Backtesting the Examples](../python-examples/example-backtests.md).

A few rules run through all of them:

- **`None`, not `0`, for "no position".** A book demeans its signal, so a `0` is a position against the day's average;
  `where(cond, signal, None)` leaves a stock out ([Missing Values](../language/missing-values.md)).
- **Rank or z-score before combining.** Raw values on different scales let one term dominate; `rank` and `zscore` put
  them on one scale.
- **Compare stocks only on point-in-time values.** Returns and ratios over time, not adjusted price levels
  ([Adjusted Data and Look-Ahead](../language/adjusted-data.md)).
