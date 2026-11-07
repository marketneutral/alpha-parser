# Books

A signal says which stocks look good and bad. A **book** is the weights you would actually hold: restricted to a
universe, traded with a delay, neutral to the market or to sectors, scaled to a fixed size and capped per stock.
`build_book` turns one into the other; `Backtest` uses it when given `BookSettings`.

## Book settings

`BookSettings` and `build_book` turn a signal into the book *Finding Alphas* and WorldQuant
BRAIN simulate: restricted to a universe, delayed, neutralized, scaled to a gross of one and
truncated.

```python
from qex import qex, BookSettings, build_book

settings = BookSettings(delay=1, universe='member', neutralize='market', truncation=0.05)
w = build_book(qex("rank(-returns(5))"), data, settings)
```

| Setting | Default | Description |
|---------|---------|-------------|
| `delay` | `1` | Day t's weights use data up to day t − delay |
| `universe` | `None` | A universe: a field's name, a mask, or a universe expression ([Universes from Python](universes.md)); `None` is every name with a price |
| `neutralize` | `"market"` | `"market"`, a group field such as `"sector"`, or `None` |
| `truncation` | `0.05` | Largest absolute weight of one name, as a fraction of the book; `None` for no cap |
| `rounds` | `1000` | Most neutralize, scale and clip rounds per day |
| `mask_cross_section` | `True` | With a universe, cross-sectional operators in the signal compare only the day's universe ([details](../language/concepts.md#which-stocks-a-cross-section-compares)); `False` applies the universe to the finished signal only, as before 0.5 |

## Untradable names

A name with no close that day can't be traded. `Backtest` holds its previous weight until it
trades again, for at most `MAX_PRICE_GAP` (10) days, and then drops it (a delisting).
`hold_untradable` applies the same rule to any weights:

```python
from qex import hold_untradable, forward_returns

held = hold_untradable(w, data['close'])                 # what Backtest holds
pnl = (held * forward_returns(data['close']).reindex_like(held).fillna(0)).sum(axis=1)
```

`Backtest` books P&L daily, on `forward_returns(close, 1)`. `QuantileAnalysis` measures
`forward_returns(close, holding_period)`. Both are NaN when the end price is missing, and measured from the last
valid price when the start price is.

## Book statistics

```python
from qex import turnover, fitness, margin_bps

to = turnover(w)          # sum(|w_t - w_{t-1}|) each day
fitness(pnl, to)          # Sharpe * sqrt(|annual return| / max(mean turnover, 0.125))
margin_bps(pnl, to)       # total P&L / total traded, in basis points
```

## Returns and forward returns

The returns behind every P&L and every forward-looking statistic follow one rule
([Halts, Delistings and Gaps](../data/gaps.md)), and are available directly:

```python
from qex import price_returns, forward_returns, MAX_PRICE_GAP

r = price_returns(data['close'], 5)       # what returns(5) gives: p[t] / p[last price at or before t-5] - 1
f = forward_returns(data['close'], 1)     # day t's return to day t+1, as Backtest books P&L
```

Both take `max_gap` (default `MAX_PRICE_GAP`, 10): the most rows a last price is carried over a gap.

Never compute returns with pandas' `pct_change` on prices: how it fills gaps changed between pandas 2 and 3.

## API reference

::: evaluation.book.BookSettings

::: evaluation.book.build_book

::: evaluation.book.hold_untradable

::: qex.primitives.price_returns

::: qex.primitives.forward_returns
