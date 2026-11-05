# Missing Values

Every panel of stock data has holes: days before a listing, halts, delistings, stocks with no sector, windows that
have not filled yet. A signal language has to say what happens at each one, because its numbers become positions.
qex follows one rule throughout: **a missing value means "not known", and not knowing is never turned into an
answer behind your back.** When you want an answer there, you write it.

## Missing is not zero

A book demeans its signal and scales it. A `0` is a value like any other: if the day's average is 0.3, a stock at 0
is 0.3 below average and the book is short it. A missing value is left out: the stock takes no position and does not
move the average.

| | A | B | C | D |
|---|---|---|---|---|
| signal | 1 | 0.2 | 0 | — |
| demeaned (what the book holds, before scaling) | 0.6 | −0.2 | −0.4 | — |

C, at 0, is short. D, missing, is out of the book. So never write 0 for "this does not apply to this stock".

## `None`: no value here, on purpose

`None` is the literal for a value you leave out on purpose:

```python
qex("where(group('sector') == 'Financials', None, field('price_to_book'))")   # book value means little for banks
qex("where(volatility(20) > 0.6, None, returns(5))")                          # leave out the most volatile names
```

`None` behaves like any missing value: cross-sections and the book leave it out, and arithmetic with it stays
missing. What differs is the reason `explain_missing` gives (below): *excluded on purpose*, not *no data*.

Write `0` only for a real zero: an indicator (`1` on an event day, `0` otherwise), a non-event in a sum, a neutral
bucket. (Older code wrote `x * 0 / 0` to make a missing value; write `None`.)

## Why a value is missing

`explain_missing` traces each missing cell of an expression back to the node where it began, and gives a reason:

```python
from qex import explain_missing
report = explain_missing(qex("ts_mean(rank(returns(1)), 10)"), data, universe=data['member'].notna())
report.summary(within=data['member'].notna())   # each reason's share of the missing member-days
report.on('2003-06-02')                          # that day's missing names, each with its reason and node
report.reason('2003-06-02', 'AAI')               # one cell: 'short history: outside the universe on earlier days'
```

| Reason | Means |
|--------|-------|
| no data | a field has no value that day: before a listing, during a halt, after a delisting |
| outside the universe | a cross-sectional operator leaves out a stock not in the universe that day |
| short history | a window has too few values yet; when its earlier days were missing, it says why |
| excluded on purpose | `None`, chosen by a `where` |
| no group | a group operator, for a stock with no label that day |
| calculation error | NaN from finite inputs: `0 / 0`, `log` or `sqrt` of a negative, a z-score with no spread |
| infinite value | a division by zero |

A missing input passes its reason on; a node gives its own reason only where all its inputs had values. In qexarena,
**why missing?** under a qex cell shows the same report, and `x.missing` does in Python.

## Comparisons and logic: three values

A comparison has three possible answers: true (1), false (0), and *not known* (missing). `close() > 8` on a day with
no price is not known. Before qex 0.7.0 it was 0, a confident "no", and everything downstream acted on it:
`where(close() > 8, 1, -1)` shorted a stock with no price.

Two stocks over three days, A with no price on day 2 and B none on day 3:

| | day 1 A | day 1 B | day 2 A | day 2 B | day 3 A | day 3 B |
|---|---|---|---|---|---|---|
| `close()` | 10 | 5 | — | 6 | 12 | — |
| `close() > 8` | 1 | 0 | — | 0 | 1 | — |
| `~(close() > 8)` | 0 | 1 | — | 1 | 0 | — |
| `where(close() > 8, 1, -1)` | 1 | −1 | — | −1 | 1 | — |
| `is_valid(close()) & (close() > 8)` | 1 | 0 | 0 | 0 | 1 | 0 |
| `coalesce(close() > 8, 0)` | 1 | 0 | 0 | 0 | 1 | 0 |
| `where(close() > 8, 1, 0)` | 1 | 0 | — | 0 | 1 | — |

The rules:

- A comparison (`>`, `<`, `>=`, `<=`, `==`, `!=`) with a missing side is missing.
- `&`, `|` and `~` follow Kleene's three-valued logic, as pandas' `NA` and SQL's `NULL` do. An answer that is
  known whatever the missing side would be survives; any other is missing:

  | `a` | `b` | `a & b` | `a \| b` | `~a` |
  |-----|-----|---------|-----------|------|
  | 1 | missing | missing | 1 | 0 |
  | 0 | missing | 0 | missing | 1 |
  | missing | missing | missing | missing | missing |

- `where` with a missing condition is missing: not knowing the condition is not knowing which branch to take.
  `where(close() > 8, 1, 0)` is therefore missing where the price is, not 0.

So `is_valid(x) & (x > k)` is 0 where `x` is missing: `is_valid` is known false there, and false & anything is
false.

## `coalesce`: saying what missing means

`coalesce(x, y)` is `x` where it has a value and `y` where it does not. `y` is a number or another expression.

```python
qex("coalesce(close() > 8, 0)")                    # the comparison, and 0 (false) where it is not known
qex("coalesce(field('eps_estimate'), field('eps_trailing'))")   # a fallback field
qex("coalesce(group('sector'), 'Other')")          # labels, with 'Other' where there is none
qex("coalesce(group_demean(x, 'sector'), 0)")      # a stock with no sector sits at its sector's average
```

`coalesce` is the explicit way to get the pre-0.7.0 behaviour of a comparison, `coalesce(cond, 0)`, where you want
it. Use it deliberately: a `coalesce(..., 0)` on a signal puts back exactly the zero that a book reads as a position.

## Where the difference shows

Most expressions are unchanged by three-valued logic, because their comparisons only ever see values that are
present. Two patterns change:

- **A comparison of a series still warming up.** `where(ts_std(mk, 21) > ts_std(mk, 252), a, b)` has no 252-day
  volatility in its first year. Before 0.7.0 the comparison was 0 and the expression took `b` (the "calm" branch)
  for that year without saying so; now it has no value there. To keep a default branch, say which:
  `where(coalesce(ts_std(mk, 21) > ts_std(mk, 252), 0), a, b)`.
- **A condition on a different field from the value it gates.** In `returns(1) * (field('volume') > 60)`, a day with
  a return but no volume is missing; before, it was 0, which the book demeaned into a position.

## Missing labels

A group operator gives a stock with no label no value, with the reason *no group*. See
[Group Operations](groups.md#a-stock-with-no-label) for what that means and how to keep such a stock.

## Rules of thumb

1. "Does not apply here" is `None`, never `0`, and never `x * condition`.
2. Leave cross-sectional signals missing rather than filling them; fill forward only slow fields, with a limit
   (`fill_forward(x, 5)`).
3. When missing should count as false, write `coalesce(cond, 0)`; when it should count as true, `coalesce(cond, 1)`.
4. When a signal has fewer values than you expect, ask why: `explain_missing`, or **why missing?** in qexarena.
5. Smooth before the cross-section where the meaning allows: `group_demean(ts_mean(x, 5), 'sector')` has a value on a
   stock's first day in the universe; `ts_mean(group_demean(x, 'sector'), 5)` waits five member days.

## See also

- [All operations](../reference/operations.md#none-and-why-a-value-is-missing): `None`, the reasons table, and the
  comparison operators.
- [Group Operations](groups.md).
