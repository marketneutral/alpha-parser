# Statistical Twins

Every other cross-sectional operator compares a stock with a **crowd**: the market (`demean`), its sector
(`group_demean`), all its peers at once (`rank`). `ts_match` compares a stock with **one particular other stock**: its
closest relative, chosen from history, and `ts_match_score` says how close. The panels on this page run on the
[Excel to qex dataset](../../excel-to-qex.md): six stocks over 30 days, AAA, BBB and CCC in Tech, DDD, EEE and FFF in
Energy.

### ts_match

<p class="sig"><code>ts_match(value, key, lookback, metric='corr', within=None, matching='nearest', rebalance=21, min_overlap=None)</code></p>

For each stock, find the stock whose history of `key` over the last `lookback` days is most similar (its **twin**),
and return the twin's `value`. Like every operator it gives an ordinary dates × stocks table: the cell for a stock
holds its twin's value that day.

```toy data=excel tail=5
returns(5)
ts_match(returns(5), returns(1), 15, within='sector', rebalance=5)
```

Read the result column by column. AAA's twin is BBB, so AAA's cell holds BBB's five-day return; CCC's twin is AAA, so
CCC's cell holds AAA's. The windows here are short so the toy data has twins; in practice `lookback` is a year.

## Arguments

`ts_match` and `ts_match_score` are the qex operators that take **arguments by name**: after the first three, write
any option as `name=value`, in any order, and leave out the ones you don't need. Writing an option at its default is the same
expression as leaving it out.

| Argument | Default | Meaning |
|---|---|---|
| `value` | | What to read from the twin: any expression, numbers or labels |
| `key` | | The expression whose history decides who is alike |
| `lookback` | | How many days of `key` to compare, at least 2 |
| `metric` | `'corr'` | How to measure alike: `'corr'`, `'spearman'`, `'cosine'`, `'euclidean'`, `'manhattan'` |
| `within` | `None` | A group the twin must share: a group's name (`'sector'`) or any labels expression |
| `matching` | `'nearest'` | `'nearest'`: each stock's closest; `'exclusive'`: reciprocal pairs that don't overlap |
| `rebalance` | `21` | Choose twins every this many days, and hold them in between |
| `min_overlap` | `lookback` | The fewest days a pair must share for it to be compared, at least 2 |

```qex
ts_match(returns(5), returns(1), 252)                                            # the defaults
ts_match(returns(5), returns(1), 252, within='sector', matching='exclusive')     # options by name
ts_match(returns(5), returns(1), 252, metric='spearman', rebalance=63, min_overlap=200)
```

The options can also go by position, in the order above, but a name says what it is.

## How a twin is chosen

1. **On a rebalance day.** The first is the day `lookback` days of data exist; then every `rebalance` days.
2. **From the past only.** The window is the `lookback` days *before* the rebalance day: twins used on day $t$ are
   chosen from `key` up to day $t - 1$, so a twin can't be chosen for having moved with the stock on the day it is
   used.
3. **Among the eligible stocks.** A candidate must be in the universe that day ([Universes](../universes.md)), share
   the stock's group when `within` is given (the group on the rebalance day), and have at least `min_overlap` days in
   the window on which both it and the stock have a value of `key`. A stock is never its own twin.
4. **The most similar**, by `metric` (below), and by `matching`: the stock's closest, or a pairing of the whole
   group.
5. **Held until the next rebalance.** A twin doesn't change between rebalances, except that a match ends the day
   either stock leaves the universe.

Then each day, the stock's cell is its twin's `value` **that day**: today's value, like any other input.

## Measuring alike: the metric

Each pair is compared over the days in the window on which both have a value, $n$ of them. With $x$ and $y$ the two
stocks' `key` on those days:

| `metric` | Similarity | Compares |
|---|---|---|
| `'corr'` | $\displaystyle \frac{\sum (x - \bar x)(y - \bar y)}{\sqrt{\sum (x - \bar x)^2 \sum (y - \bar y)^2}}$ | how they move together, whatever their size |
| `'spearman'` | the correlation of their ranks | the same, robust to a few extreme days |
| `'cosine'` | $\displaystyle \frac{\sum x\,y}{\sqrt{\sum x^2 \sum y^2}}$ | the correlation without removing the means |
| `'euclidean'` | $\displaystyle -\sqrt{\tfrac{1}{n} \sum (x - y)^2}$ | how far apart they are, day by day |
| `'manhattan'` | $\displaystyle -\tfrac{1}{n} \sum \lvert x - y \rvert$ | the same, less swayed by one large gap |

Higher is closer, so the distances are negated. They are divided by $n$, so a pair with a few gaps isn't favored over
one with none. (Spearman ranks each stock's whole window, then correlates the ranks on the shared days: exact when
the pair shares every day.)

Correlation and Spearman ask "do they move together?" and ignore size: a quiet utility and a wild biotech can be
twins. The distances also ask "by the same amounts?", so a quiet stock's twin is another quiet stock. For returns,
correlation is usually the question; distances suit a key already on a common scale, such as volatility-scaled
returns.

## Nearest or exclusive: the matching

With `'nearest'`, each stock picks its single closest stock. That choice isn't mutual: CCC's closest stock can be AAA
while AAA's closest is BBB. A popular stock can be several stocks' twin at once.

With `'exclusive'`, the group is paired off: the closest pair is matched first, then the closest pair among the
stocks still free, and so on. Every twin is mutual, no stock is in two pairs, and in a group with an odd number of
stocks one is left without a twin:

```toy data=excel tail=5 inputs
ts_match(identifier(), returns(1), 15, within='sector', rebalance=5)
ts_match(identifier(), returns(1), 15, within='sector', rebalance=5, matching='exclusive')
```

In the first, CCC chose AAA, which chose BBB. In the second the pairs are mutual, and CCC and FFF, one stock too many
in each sector, have none.

Exclusive pairs make **self-contained trades**: in `returns(5) - ts_match(returns(5), …, matching='exclusive')`, each
pair's two values are equal and opposite, so every pair is a dollar-neutral bet before the book is built. The price is
some closeness (a stock whose best match is taken settles for its second choice) and some stability (one changed
pair can rearrange others).

## What the result holds, and when it has no value

- **The twin's value today.** Not the value on the rebalance day: the twin is fixed, its value is live.
- **No value without a twin**: no eligible stock (a group of one, too short a history, outside the universe), or the
  pairing left the stock out.
- **No value when the twin's value is missing**, for example a halted twin. The stock doesn't pick another twin until
  the next rebalance.
- **Labels gather as labels.** `ts_match(group('sector'), …)` is the twin's sector. What you can do with the result
  depends on what `value` is, like `where` and `coalesce`.

[Missing Values](../missing-values.md) reports a cell with no twin as **no twin**, and a missing twin value with the
twin's own reason.

## Seeing the twins: `identifier()`

`identifier()` is each stock's own identifier, as a label. Read through `ts_match`, it is each stock's twin's
identifier: the matches themselves, as an ordinary table.

```toy data=excel tail=5 inputs
identifier()
ts_match(identifier(), returns(1), 15, within='sector', rebalance=5)
```

This is how to see and check the twins in a research notebook. It reads the same cached selection as every other
`ts_match` with the same `key` and options, so looking costs nothing extra. Because the result is labels, it can be
compared exactly:

```qex
let key = delay(returns(1), 5)
in ts_match(identifier(), key, 252, within='sector') == ts_match(identifier(), key, 252, metric='spearman', within='sector')
```

That is 1 where correlation and Spearman pick the same twin, 0 where they don't: a direct check that a match isn't one
metric's accident.

### ts_match_score

<p class="sig"><code>ts_match_score(key, lookback, metric='corr', within=None, matching='nearest', rebalance=21, min_overlap=None)</code></p>

How alike each stock and its twin were when the twin was chosen. It takes `ts_match`'s arguments without the `value`,
and with the same `key` and options it reads the same selection: the score was computed while choosing, so it costs
nothing more.

```toy data=excel tail=5 inputs
ts_match(identifier(), returns(1), 15, within='sector', rebalance=5)
ts_match_score(returns(1), 15, within='sector', rebalance=5)
```

AAA and BBB are each other's twins, with one score, their correlation over the matching window; CCC's twin is AAA, a
weaker match. The score is in the metric's own units:

| `metric` | The score |
|---|---|
| `'corr'`, `'spearman'`, `'cosine'` | the correlation (or cosine), from −1 to 1, higher is closer |
| `'euclidean'`, `'manhattan'` | the distance, 0 or more, **smaller** is closer |

Like the twin, the score is fixed at the rebalance and held until the next, and there is no score without a twin. It
says how good the match was when chosen; whether the pair still moves together is a different question, asked of the
recent past, `ts_corr(r, ts_match(r, key, 252), 63)`. Both are useful, and they can disagree.

## Choosing the key

The key decides who is alike; anything you want to compare goes into it, as ordinary qex.

- **Exclude the move you trade.** To trade the last five days, match on returns that stop five days ago,
  `delay(returns(1), 5)`; otherwise the stocks that moved together this week are chosen as twins for having done so.
- **Take out what everyone shares.** Matching on raw returns finds stocks that share the market. Match on residuals
  to find stocks that share something besides it:
  `ts_resid(returns(1), returns(1, 'market'), 252)`.
- **Put stocks on one scale** before a distance: `returns(1) / ts_std(returns(1), 63)`.

## Holding and discontinuities

A twin is held between rebalances, so the result is continuous between them. At a rebalance the twin can change, and
the result jumps to a different stock's series. A rolling operator on the result, such as `ts_mean(twin, 20)` or
`ts_corr(r, twin, 63)`, mixes two stocks across a change. That is often fine, and sometimes the point (`ts_corr(r,
twin, 63)` checks whether the current twin still moves with the stock), but it is never one stock's history.

## Sharing and cost

Choosing twins is the expensive part; reading a value from them is a lookup. qex keeps the choice in an internal node
of its own, shared by every `ts_match` and `ts_match_score` with the same `key`, `lookback`, `metric`, `within`,
`matching`, `rebalance` and `min_overlap`, whatever its `value`. The twin's return, its volume, its identifier and the
match's score cost one selection
([Graphs and Dask](../../python/distributed.md#what-is-shared) shows the node).

Each rebalance compares every pair within each group over the window: about $\text{lookback} \times m^2$ for a group
of $m$ stocks. Measured on 1,500 stocks over five years with a 252-day window and monthly rebalancing:

| | `within='sector'` | no groups |
|---|---|---|
| `'nearest'` | 1.1 s | 4.6 s |
| `'exclusive'` | 2.3 s | 26 s |

`within` is the biggest saving, and usually what you want anyway. `'manhattan'` compares one stock at a time and is
the slowest metric (6.5 s within sectors); a longer `rebalance` divides the cost.

## Look-ahead

- The window ends the day before the rebalance, so a twin is chosen only from the past.
- The result is another stock's value, compared with this stock's: like any comparison across stocks, it must not be
  an adjusted price level ([Adjusted Data and Look-Ahead](../adjusted-data.md)). The look-ahead check flags
  `ts_match(close(), …)`, as it flags `rank(close())`; `ts_match(returns(5), …)` is fine.
- A key may be an adjusted price level for `'corr'`, `'spearman'` and `'cosine'`, which ignore each stock's scale,
  but not for the distances, which compare levels across stocks.

## Trading on twins

**The twin that fell behind.** Pair stocks on returns that stop five days ago, and bet that a stock which outran its
twin over the last week gives some of it back:

```qex
let key  = delay(returns(1), 5),
    twin = ts_match(returns(5), key, 252, within='sector', matching='exclusive')
in -(returns(5) - twin)
```

With exclusive pairs each pair is long one stock and short the other in equal measure.

**In the pair's own noise.** A gap of 3% means more for two utilities than for two biotechs. Divide by the spread's
recent volatility:

```qex
let key  = delay(returns(1), 5),
    r    = returns(1),
    twin = ts_match(r, key, 252, within='sector', matching='exclusive'),
    gap  = returns(5) - ts_match(returns(5), key, 252, within='sector', matching='exclusive')
in -gap / ts_std(r - twin, 63)
```

Both `ts_match`s share one selection.

**Weighting by the match.** Trade the closest pairs hardest: weight each pair's bet by how alike it was when chosen.

```qex
let key   = delay(returns(1), 5),
    twin  = ts_match(returns(5), key, 252, within='sector', matching='exclusive'),
    score = ts_match_score(key, 252, within='sector', matching='exclusive')
in -(returns(5) - twin) * max(score, 0)
```

`max(score, 0)` gives a pair that was negatively correlated no weight. With exclusive matching both stocks of a pair
have the same score, so the pair stays dollar-neutral.

**Weighting robust pairs.** A twin found by a year of correlation can be an accident. Three checks, each one line:
the pair still moves together (`near`), another metric picks the same twin (`agree`), and the twin trades within a
factor of three of the stock's volume (`liquid`). Used as weights rather than a gate, so no pair is dropped and robust
pairs trade bigger:

```qex
let key    = delay(returns(1), 5),
    r      = returns(1),
    twin   = ts_match(r, key, 252, within='sector', matching='exclusive'),
    near   = ts_corr(r, twin, 63) > 0.5,
    agree  = ts_match(identifier(), key, 252, within='sector', matching='exclusive')
          == ts_match(identifier(), key, 252, metric='spearman', within='sector', matching='exclusive'),
    liquid = abs(log(adv(20) / ts_match(adv(20), key, 252, within='sector', matching='exclusive'))) < log(3),
    gap    = returns(5) - ts_match(returns(5), key, 252, within='sector', matching='exclusive')
in -gap / ts_std(r - twin, 63) * (1 + near + agree + liquid)
```

Gating on all three checks at once (`where(near & agree & liquid, …, None)`) keeps only the best pairs, and usually
too few of them: breadth is part of the edge. [Pairs Trading](../../examples/pairs-trading.md) has more pairs
strategies, with fixed pairs.

## Errors

Every mistake is refused when the expression is read, with a message that says what to write:

| Written | Refused because |
|---|---|
| `ts_match(r, key, 252, metrc='corr')` | there is no option `metrc`; the options are listed |
| `ts_match(r, key, 252, 'corr', metric='corr')` | `metric` is given twice, by position and by name |
| `ts_match(r, key, 252, metric='pearson')` | the metrics are `'corr'`, `'spearman'`, `'cosine'`, `'euclidean'`, `'manhattan'` |
| `ts_match(r, key, 252, min_overlap=300)` | `min_overlap` can't be more than `lookback` |
| `rank(ts_match(identifier(), key, 252))` | the result is labels, and `rank` needs numbers |
| `ts_mean(r, period=20)` | only `ts_match` takes arguments by name |
