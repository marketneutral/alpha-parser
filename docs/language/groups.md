# Group Operations

A cross-sectional operator (`rank`, `demean`, `zscore`, `quantile`) compares each stock with every other stock that
day. A **group operator** does the same thing inside each group: a stock is compared only with the stocks that share
its label. That is how a signal is made sector-neutral, size-neutral, or neutral to anything else you can label.

Every example on this page uses the same nine stocks on one day:

| | A | B | C | D | E | F | G | H | I |
|---|---|---|---|---|---|---|---|---|---|
| `field('value')` | 9 | 1 | 5 | 2 | 8 | 3 | 7 | 4 | 6 |
| `field('size')` | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| `group('sector')` | Tech | Tech | Fin | Fin | Tech | Fin | Tech | — | Tech |

H has no sector label (— is missing).

## The operators

| Operation | Each stock gets |
|-----------|-----------------|
| `group_rank(x, groups)` | its percentile within its group, in (0, 1] |
| `group_demean(x, groups)` | `x` minus its group's mean |
| `group_quantile(x, groups, n)` | its bucket within its group, 1 (lowest) to `n` |
| `group_sum(x, groups)` | its group's total (every member gets the same number; missing if the group has no values) |
| `group_std(x, groups)` | the standard deviation of `x` across its group's members that day |
| `group_count_valid(x, groups, period)` | how many of its group's values are present, over a rolling window |

`groups` is either a group's **name** (`'sector'`, looked up in the data), or any **expression that gives each stock
a label** (below).

Every group operator works on one day's cross-section, except `group_count_valid`, which counts over a window. Each
member gets its group's number from that day, even a member whose own value is missing. A stock with no label, or
outside the universe, gets no value.

**Labels given on fewer dates than the data** (monthly sector labels on daily prices) hold from their date until the
next row: each day takes the latest label row on or before it, never a later one, and a day before the first row has
no label. (Before qex 0.12.0 a missing date took the *last* row, a look-ahead.)

Compare each operator with its whole-market version:

| | A | B | C | D | E | F | G | H | I |
|---|---|---|---|---|---|---|---|---|---|
| `demean(field('value'))` | 4 | −4 | 0 | −3 | 3 | −2 | 2 | −1 | 1 |
| `group_demean(field('value'), 'sector')` | 2.8 | −5.2 | 1.67 | −1.33 | 1.8 | −0.33 | 0.8 | — | −0.2 |
| `rank(field('value'))` | 1 | 0.11 | 0.56 | 0.22 | 0.89 | 0.33 | 0.78 | 0.44 | 0.67 |
| `group_rank(field('value'), 'sector')` | 1 | 0.2 | 1 | 0.33 | 0.8 | 0.67 | 0.6 | — | 0.4 |
| `group_sum(field('value'), 'sector')` | 31 | 31 | 10 | 10 | 31 | 10 | 31 | — | 31 |

C has the highest value among the financials, so its `group_rank` is 1, though it is only in the middle of the market.
The Tech mean is 6.2 and the Fin mean is 3.33, so `group_demean` takes out the difference between the two sectors:
a book built from it is long and short within each sector, with no net bet on Tech against Fin.

```python
qex("group_demean(returns(20), 'sector')")      # sector-neutral momentum
qex("group_rank(-returns(5), 'sector')")        # reversal, ranked within each sector
qex("volume(1) / group_sum(volume(1), 'sector')")  # each stock's share of its sector's volume
```

## A stock with no label

A stock with no label is in no group, so a group operator gives it **no value**, even when every other field is
present. H above has none, and `explain_missing` (and **why missing?** in qexarena) says why:

```python
explain_missing(qex("group_demean(field('value'), 'sector')"), data).reason(day, "H")
# 'no group, at group_demean(field('value'), 'sector')'
```

The other stocks are not affected: H is in no group, so it moves no group's mean or ranks. A book takes no position
in H. Two ways to keep it, when you want to:

| | H | the others |
|---|---|---|
| `coalesce(group_demean(field('value'), 'sector'), 0)` | 0 | unchanged |
| `group_demean(field('value'), coalesce(group('sector'), 'Other'))` | 0 | unchanged |

The first says "a stock with no sector sits at its sector's average". The second puts every unlabelled stock in a
group of its own, named `'Other'`, and here gives the same answer, because H is alone in it and a group of one
demeans to 0. With two unlabelled stocks the second would demean them against each other: a pair trade between two
unrelated names, chosen by a gap in the data. Prefer the first.

## Labels from any expression

Anything that gives each stock a label on each day can be a grouping: a number, a string, the result of a
comparison. The common case is a bucket:

| | A | B | C | D | E | F | G | H | I |
|---|---|---|---|---|---|---|---|---|---|
| `quantile(field('size'), 3)` | 1 | 1 | 1 | 2 | 2 | 2 | 3 | 3 | 3 |
| `group_demean(field('value'), quantile(field('size'), 3))` | 4 | −4 | 0 | −2.33 | 3.67 | −1.33 | 1.33 | −1.67 | 0.33 |

```python
qex("group_demean(returns(20), quantile(field('mcap'), 5))")          # size-neutral momentum
qex("group_rank(x, quantile(volatility(60), 3))")                       # rank within volatility terciles
qex("let size = quantile(field('mcap'), 3) in group_demean(x, size)")  # a grouping can be named
```

A name and the expression that gives the same labels are the same grouping:
`group_demean(x, 'sector')` and `group_demean(x, group('sector'))` agree everywhere.

Things to know:

- **Buckets are recomputed every day.** A stock near a boundary switches groups from one day to the next, and each
  switch moves its position: a source of turnover, like a universe that flickers. Bucket a slow quantity
  (`quantile(ts_mean(field('mcap'), 21), 5)` rather than today's market cap) to damp it.
- **Buckets come from the universe.** `quantile` is computed over the universe's members, so each bucket holds its
  share of the members, and a stock outside the universe has no bucket.
- **A missing input means no label.** A stock with no market cap has no size bucket, so a size-neutral signal has
  no value for it; `explain_missing` says `no data, at field('mcap')`.
- **Labels compare by value.** `1` and `1.0` are one group; `0` is a label like any other.
- A grouping must give a label per stock per day: `group_demean(x, 3)` is refused, with a message saying what
  `groups` can be.

## Double sorts

A **double sort** buckets stocks on two characteristics. There are two kinds, and they differ.

**Independent**: bucket each characteristic over the whole market. Each stock's cell is the pair.

```python
qex("quantile(field('size'), 3)")     # size tercile
qex("quantile(field('value'), 3)")    # value tercile, among all stocks
```

**Dependent** (conditional): bucket on the first, then on the second *within* each bucket of the first. Value is
then measured against stocks of similar size:

```python
qex("group_quantile(field('value'), quantile(field('size'), 3), 3)")
```

| | A | B | C | D | E | F | G | H | I |
|---|---|---|---|---|---|---|---|---|---|
| size tercile | 1 | 1 | 1 | 2 | 2 | 2 | 3 | 3 | 3 |
| value tercile, **independent** | 3 | 1 | 2 | 1 | 3 | **1** | 3 | **2** | 2 |
| value tercile, **dependent** | 3 | 1 | 2 | 1 | 3 | **2** | 3 | **1** | 2 |

F's value of 3 is in the bottom third of the market but in the middle of its size bucket (2, 3, 8); H's 4 is the
reverse. In a dependent sort every size bucket splits into value buckets of the same size (as nearly as the counts allow), so a long-short
on value is not secretly a bet on size.

A long-short book from the dependent sort, long the top value tercile and short the bottom within each size bucket:

```python
qex("""
    let v = group_quantile(field('value'), quantile(field('size'), 3), 3)
    in where(v == 3, 1, where(v == 1, -1, None))
""")
```

The middle tercile gets `None`, so it takes no position.

`group_quantile(x, groups, n)` is `quantile` applied inside each group: the same as writing out, for each bucket,
`quantile(where(b == k, x, None), n)`, which is how a dependent sort had to be written before qex 0.8.0. Like
`quantile`, it buckets by percentile, so a group with fewer stocks than buckets skips some buckets, and a group of one
is in the top bucket.

## Crossing groupings

`group_cross(a, b, ...)` combines two or more groupings into one: each stock's label is its labels joined with `|`.
Its arguments are names or label expressions.

| | A | B | C | D | E | F | G | H | I |
|---|---|---|---|---|---|---|---|---|---|
| `group_cross('sector', quantile(field('size'), 3))` | Tech\|1 | Tech\|1 | Fin\|1 | Fin\|2 | Tech\|2 | Fin\|2 | Tech\|3 | — | Tech\|3 |
| `group_demean(field('value'), group_cross(...))` | 4 | −4 | 0 | −0.5 | 0 | 0.5 | 0.5 | — | −0.5 |

```python
qex("group_demean(returns(20), group_cross('sector', quantile(field('mcap'), 3)))")   # sector- and size-neutral
qex("where(group_cross('sector', quantile(field('mcap'), 3)) == 'Technology|3', 1, None)")
```

A stock missing any part has no label (H has no sector). Crossing multiplies the number of groups and shrinks each
one: here C and E are alone in their cells, so they demean to 0 and take no position. Count the stocks per cell
before trusting a cross on a real universe (`group_count_valid(x, groups, 1)` gives each stock its cell's count).
A label containing `|` is refused, since `'a|b'` with `'c'` and `'a'` with `'b|c'` would join to the same label.

## Numbers and labels

Every qex expression gives a value per date and stock. Most give **numbers**. A few give **labels**:
`group('sector')`, `group_cross(...)`, a string such as `'Other'`, and `coalesce` or `where` whose branches are labels.
Labels can be:

- compared, with `==` and `!=`: `group('sector') == 'Technology'` is 1, 0, or missing;
- used to group: the `groups` of any group operator, and the parts of `group_cross`;
- passed through `coalesce` and `where`, as long as both sides are labels (or one is `None`).

Anywhere else needs numbers, and qex refuses the expression when it is parsed, before anything runs:

```python
qex("rank(group('sector'))")
# ValueError: group('sector') gives labels, not numbers, and rank() needs numbers. Compare labels
# (group('sector') == 'Technology') or group by them (group_demean(x, group('sector'))).

qex("coalesce(group('sector'), 0)")
# ValueError: coalesce(group('sector'), 0) mixes labels and numbers: both sides must give the same kind
```

Numbers can serve as labels (a `quantile` bucket is a number), so a group operator accepts either. A field read with
`field('name')` is taken to be numbers; read a field of labels with `group('name')`. An alpha must give numbers:
`qex.kinds.numbers_only(signal)` says why one cannot.

## Inside the cross-sectional universe

Like every cross-sectional operator, a group operator compares only the members of the cross-sectional universe
(`cross_section_universe`); a stock outside it gets no value. Labels from an expression are computed the same way, so
`quantile(field('mcap'), 5)` buckets the members only.

## See also

- [All operations](reference.md#group-operations), for the table of every operator.
- [Missing values](missing-values.md): `None`, why a value is missing, three-valued logic and `coalesce`.
