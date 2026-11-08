# Universes

A **universe** says which stocks count on each day: which ones the cross-sectional operators compare (`rank`,
`demean`, the group operators...) and which ones a book may hold. In qex a universe can be written as a qex
expression, like a signal. The data stays raw (prices, volume, shares outstanding, labels), and the rule that turns it
into membership is code you can read, version, and change.

## A universe is an expression

A universe expression is an ordinary qex expression read as yes or no: a stock is a member on a day the expression is
nonzero, and not a member where it is 0 or missing.

```qex
field('mcap') >= 300e6                                          # $300M market cap and up
(field('price') >= 2) & (field('price') * field('shares') >= 2e9)      # a $2 price and $2B market cap
```

A data field that is 1 for members and missing otherwise is a universe too. How to hand a universe to a book or an
evaluation from Python is in [Universes from Python](../python/universes.md).

## How a universe expression is evaluated

- **Over every stock, with no universe active.** The expression defines the cross-section, so it cannot be computed
  inside one: `rank(field('mcap')) > 0.5` ranks every stock in the data, even when evaluated inside another universe.
- **It must give numbers.** Nonzero (and not missing) is a member; 0 or missing is not. A universe that gives labels
  is refused: compare them instead (`group('category') == 'Domestic Common Stock'`).
- **Point in time**, like a signal: membership on day t uses data up to day t. A book's delay applies on top, so a
  stock that joins today can first be traded tomorrow.

## Hysteresis: `latch`

A universe defined by a hard line flips a stock in and out as its market cap wanders around the line, and every flip
is a forced trade. Index providers use **bands** instead: a stock enters above one line and leaves only below a lower
one. That is a latch:

`latch(on, off)` is 1 from a day `on` is true until a day `off` is true, then 0 until `on` is true again, and 0
before `on` is first true. A day where `on` or `off` is missing changes nothing, and on a day both are true, `off`
wins.

| market cap ($M) | 290 | 305 | 295 | 302 | 298 | 310 | 285 | 230 | 250 | 301 | 296 | 304 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `mcap >= 300` | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 1 |
| `latch(mcap >= 300, mcap < 240)` | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 1 | 1 | 1 |

The hard line changes state 9 times in twelve days; the band 3, and only for the real drop to 230.

Because a missing day changes nothing, a stock that stops trading stays latched in. Say when it leaves: put the
hard exits (no price for ten days, a price under $2) in `off`.

A liquidity universe written as an expression. Dollar volume is averaged over the last 20 *sessions* (days the stock
was quoted, traded or not), counting only the sessions with trades, once there are 10 of them:

```qex
let price  = field('price'),
    vol    = field('volume_raw'),
    traded = ts_sum_events(vol > 0, 20, 1),
    dvol   = ts_sum_events(price * vol, 20, 1) / traded,
    mcap   = price * field('shares')
in is_valid(close()) & (price >= 2) & (traded >= 10) & (dvol >= 5e6) & (mcap >= 300e6)
```

- `price` is the as-traded close, on every quoted session;
- `vol` is as-traded volume, 0 on a session with no trades;
- `traded` counts the sessions with trades among the last 20;
- `dvol` is dollar volume per session with trades.

With a band, and the hard exits in `off`:

```qex
let price = field('price'),
    mcap  = price * field('shares'),
    dvol  = ts_mean(price * field('volume_raw'), 20),
    alive = ts_count_events(price, 10) > 0,
    ok    = alive & (price >= 2) & (group('category') == 'Domestic Common Stock'),
    enter = ok & (mcap >= 300e6) & (dvol >= 5e6),
    leave = ~coalesce(ok, 0) | (mcap < 240e6) | (dvol < 4e6)
in latch(enter, leave)
```

`latch` is useful in signals too: hold a position from an entry signal until an exit signal.

## Reconstituting on a schedule

To decide membership once a month and hold it in between, keep the rule's value on the first trading day of each
month and carry it forward:

```qex
let first = month_of_year() != delay(month_of_year(), 1),
    rule  = (field('price') * field('shares') >= 300e6)
in coalesce(fill_forward(where(first, rule, None)), 0)
```

`fill_forward(x)` with no limit carries each value until the next one; `fill_forward(x, n)` carries it at most `n`
days.

## See also

- [How qex Evaluates](concepts.md#which-stocks-a-cross-section-compares): what the universe restricts, and what it
  does not.
- [Universes from Python](../python/universes.md): giving a universe to a book, an evaluation or a diagnostic.
- [Group Operations](groups.md), whose operators compare within groups inside the universe.
