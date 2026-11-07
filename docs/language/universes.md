# Universes

A **universe** says which stocks count on each day: which ones the cross-sectional operators compare (`rank`,
`demean`, the group operators...) and which ones a book may hold. In qex a universe can be written as a qex
expression, like a signal. The data stays raw (prices, volume, shares outstanding, labels), and the rule that turns it
into membership is code you can read, version, and change.

## Three ways to give a universe

Everywhere qex takes a universe (`cross_section_universe`, `BookSettings(universe=...)`,
`explain_missing(universe=...)`), it takes any of:

| Form | Example | Member when |
|------|---------|-------------|
| a mask | `data['member'].notna()` | True |
| a data field's name | `'member'` | the field has a value (1 in, NaN out) |
| a universe expression | `"field('mcap') >= 300e6"`, or a parsed `qex(...)` | the expression is nonzero |

```python
from qex import qex, cross_section_universe, universe_mask
from evaluation import BookSettings, build_book

LARGE = "(close() >= 2) & (field('price') * field('shares') >= 2e9)"

with cross_section_universe(LARGE, data):                       # an expression needs the data
    r = qex("rank(returns(5))").evaluate(data)                  # ranked among the large stocks only

book = build_book(qex("rank(-returns(5))"), data, BookSettings(delay=1, universe=LARGE))
members = universe_mask(LARGE, data)                            # the True/False frame itself
```

Text that names a field of the data is that field (`'member'` stays the field it always was); anything else is
parsed as qex.

## How a universe expression is evaluated

- **Over every stock, with no universe active.** The expression defines the cross-section, so it cannot be computed
  inside one: `rank(field('mcap')) > 0.5` ranks every stock in the data, even when evaluated inside another universe.
- **It must give numbers.** Nonzero (and not missing) is a member; 0 or missing is not. A universe that gives labels
  is refused: compare them instead (`group('category') == 'Domestic Common Stock'`).
- **Point in time**, like a signal: membership on day t uses data up to day t. A book's delay applies on top, so a
  stock that joins today can first be traded tomorrow.
- **Once per context.** Inside one `compute_context`, the membership is computed once and shared by every signal
  evaluated in it.

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

qexarena's universe until now, written as an expression, reproduces its old membership file exactly (13,138,466
member-days, no difference). Dollar volume is averaged over the last 20 *sessions* (days the stock was quoted, traded
or not), counting only the sessions with trades, once there are 10 of them:

```python
TODAY = """
let price  = field('price'),                              # as-traded close, on every quoted session
    vol    = field('volume_raw'),                         # as-traded volume, 0 on a session with no trades
    traded = ts_sum_events(vol > 0, 20, 1),               # sessions with trades among the last 20
    dvol   = ts_sum_events(price * vol, 20, 1) / traded,
    mcap   = price * field('shares')
in is_valid(close()) & (price >= 2) & (traded >= 10) & (dvol >= 5e6) & (mcap >= 300e6)
"""
```

With a band, and the hard exits in `off`:

```python
STANDARD = """
let price = field('price'),
    mcap  = price * field('shares'),
    dvol  = ts_mean(price * field('volume_raw'), 20),
    alive = ts_count_events(price, 10) > 0,
    ok    = alive & (price >= 2) & (group('category') == 'Domestic Common Stock'),
    enter = ok & (mcap >= 300e6) & (dvol >= 5e6),
    leave = ~coalesce(ok, 0) | (mcap < 240e6) | (dvol < 4e6)
in latch(enter, leave)
"""
```

`latch` is useful in signals too: hold a position from an entry signal until an exit signal.

## Reconstituting on a schedule

To decide membership once a month and hold it in between, keep the rule's value on the first trading day of each
month and carry it forward:

```python
"""
let first = month_of_year() != delay(month_of_year(), 1),
    rule  = (field('price') * field('shares') >= 300e6)
in coalesce(fill_forward(where(first, rule, None)), 0)
"""
```

`fill_forward(x)` with no limit carries each value until the next one; `fill_forward(x, n)` carries it at most `n`
days.

## See also

- [All operations](reference.md#the-cross-sectional-universe): what the cross-sectional universe
  restricts, and what it does not.
- [Group Operations](groups.md), whose operators compare within groups inside the universe.
