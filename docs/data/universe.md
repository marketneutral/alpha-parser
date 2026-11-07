# The Universe Field

A universe can be written as a qex expression over the raw data ([Universes](../language/universes.md)), or supplied
as a field: a table that is 1 (or any finite value) for the stocks in the universe on each day and `NaN` otherwise.

```python
data['member'] = member_df     # same dates and stocks as close; 1.0 or NaN
```

`BookSettings(universe='member')` uses it to restrict both the book and the signal's cross-sectional operators to the
day's members (see [Books](../python/books.md) and [Universes from Python](../python/universes.md)).

**Keep every stock's full history.** `close` and the other fields should hold each stock's whole history, including
days before it joined the universe and after it left. Time-series operators use it, so a stock that joins today has
its past returns, volatility and beta on its first day. Only the cross-sectional operators and the book look at the
universe.
