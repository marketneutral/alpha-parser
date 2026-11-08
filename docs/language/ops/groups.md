# Groups

Like the cross-sectional operators, but within groups: each stock is compared only with the stocks sharing its label
that day. In the [toy data](../reference.md#the-toy-data), **A** and **B** are Tech, **C** and **D** are Energy.

`groups` is a group's name (`'sector'`) or any expression giving labels: `quantile(field('mcap'), 5)`,
`coalesce(group('sector'), 'Other')`, `group_cross(...)`. A stock with no label, or outside the universe, gets no value.
Labels given on fewer dates than the prices hold from their date until the next, never earlier.
[Groups and Labels](../groups.md) works through them on an example.

## Labels

### group

<p class="sig"><code>group('name')</code></p>

The labels themselves. Labels can be compared with `==` and `!=`, used as `groups`, and passed through `where` and
`coalesce`; anything else (arithmetic, `rank`) is refused when the expression is read.

```toy
group('sector')
group('sector') == 'Tech'
```

```qex
where(group('sector') == 'Financials', None, field('book_to_price'))    # not for banks
```

### identifier

<p class="sig"><code>identifier()</code></p>

Each stock's own identifier (its name in the data), as a label, on every date. On its own it says nothing new; its use
is reading a stock's identity through an operator that moves values between stocks. `ts_match(identifier(), key,
252)` is each stock's twin's identifier: the way to see the twins ([Statistical Twins](matching.md#seeing-the-twins-identifier)).

```toy data=excel tail=3 inputs
identifier()
```

### group_cross

<p class="sig"><code>group_cross(a, b, ...)</code></p>

One grouping from several: each stock's labels joined with a bar, such as `'Tech|2'`. Below, sector crossed with
beta halves: the high-beta stock in each sector is in half 2.

```toy
group_cross('sector', quantile(field('beta'), 2))
```

```qex
group_demean(returns(20), group_cross('sector', quantile(field('mcap'), 3)))    # sector- and size-neutral
```

## Within each group

### group_demean

<p class="sig"><code>group_demean(x, groups)</code></p>

`x` minus its group's mean that day: neutral to the group.

```toy
returns(1)
group_demean(returns(1), 'sector')
```

```qex
group_demean(returns(20), 'sector')                         # sector-neutral momentum
group_demean(returns(20), quantile(field('mcap'), 5))       # size-neutral momentum
```

### group_rank

<p class="sig"><code>group_rank(x, groups)</code></p>

The percentile rank within each group, from `1/n` to 1.

```toy
returns(1)
group_rank(returns(1), 'sector')
```

### group_gaussrank

<p class="sig"><code>group_gaussrank(x, groups)</code></p>

[`gaussrank`](cross-sectional.md#gaussrank) within each group: $\Phi^{-1}((r_i - 0.5)/n)$ with $r_i$ the stock's
rank and $n$ the number of stocks with a value in its group that day. Normal scores within each sector, summing to
zero in each; a stock alone in its group gets 0.

```toy
returns(1)
group_gaussrank(returns(1), 'sector')
```

```qex
group_gaussrank(field('book_to_price'), 'sector') + group_gaussrank(returns(231), 'sector')   # sector-neutral, on one scale
```

### group_quantile

<p class="sig"><code>group_quantile(x, groups, n)</code></p>

The bucket 1 to `n` within each group: a dependent double sort.

```qex
group_quantile(field('value'), quantile(field('mcap'), 3), 3)    # value terciles within size terciles
```

## Group statistics

Every member of a group gets the group's number, even a member whose own value is missing: **C** has no dollar
volume on Jan 04 and still gets the Energy sum.

### group_std

<p class="sig"><code>group_std(x, groups)</code></p>

The standard deviation across the group's stocks that day; no value for a group with only one value, as Energy on
Jan 04.

```toy
returns(1)
group_std(returns(1), 'sector')
```

```qex
group_demean(returns(5), 'sector') / group_std(returns(5), 'sector')    # a z-score within each sector
```

### group_sum

<p class="sig"><code>group_sum(x, groups)</code></p>

The sum within the group; no value when the group has no values that day.

```toy
adv(1)
group_sum(adv(1), 'sector')
```

```qex
adv(20) / group_sum(adv(20), 'sector')         # each stock's share of its sector's trading
```

### group_count_valid

<p class="sig"><code>group_count_valid(x, groups, period)</code></p>

How many values the group's stocks have had over the last `period` days.

```qex
group_count_valid(field('surprise'), 'sector', 63)     # how many reports the sector had this quarter
```
