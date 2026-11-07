# Labels and Groups

A **label field** gives each stock a category on each day: a sector, an industry, an exchange, a pair name. The group
operators group by it, and `group('name')` reads it for comparing. (What the group operators do is in
[Groups and Labels](../language/groups.md).)

```python
sector = pd.DataFrame(
    [['Tech', 'Tech', 'Finance', 'Finance']] * len(dates),
    index=dates, columns=tickers,
)
data['sector'] = sector          # or data['groups']['sector']
```

```qex
group_demean(returns(20), 'sector')
group('sector') == 'Finance'
```

- **Shape.** The same dates and stocks as the prices, with a string (or any value) per cell, and `NaN` for a stock
  with no label that day. A stock with no label is in no group and gets no value from a group operator.
- **Where.** Either at the top level, `data['sector']`, or in a nested container, `data['groups']['sector']`.
- **Labels over time.** A label is what the stock was on that day. Labels given on fewer dates than the prices
  (monthly, say) hold from their date until the next row: each day takes the latest row on or before it, never a
  later one, and a day before the first row has no label.
- **Current-only labels look ahead.** A label field that repeats today's sector on every past day groups a stock in
  2005 by a sector it moved into in 2020. If your vendor only has current labels, know that the group operators see
  the future through them, and prefer point-in-time labels where you have them.
- **Labels or numbers.** Read a label field with `group('name')`. `field('name')` reads numbers, and a field of
  strings read that way can't be ranked or averaged.

A label can also come from an expression, such as `quantile(field('mcap'), 5)`, with no field at all: see
[Labels from any expression](../language/groups.md#labels-from-any-expression).
