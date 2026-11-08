# Data and Primitives

Where every expression starts: prices, fields from the data, and the returns, volatility and volume built from them.
The panels run on the [toy data](../reference.md#the-toy-data).

## Prices and fields

### close

<p class="sig"><code>close()</code>  <code>open()</code>  <code>high()</code>  <code>low()</code></p>

Each stock's daily prices, adjusted for splits and dividends. Use them for returns and for a stock's own history; to
compare stocks with each other, or with a dollar amount, use an as-traded field such as `field('price')`
([Adjusted Data and Look-Ahead](../adjusted-data.md)).

```toy
close()
```

### field

<p class="sig"><code>field('name')</code></p>

Any field of the data, by name, read as numbers. A field given once per day (an index level, a rate) is given to every
stock. Labels such as a sector are read with [`group('name')`](groups.md#group) instead.

```toy
field('surprise')
```

```qex
field('mcap')                    # market capitalization, if the data has it
1 / field('price_to_book')       # book-to-price
```

### market

<p class="sig"><code>market()</code></p>

The market's price level: the data's `market` series, the same for every stock. It is whatever market the data
supplies, such as an index's total-return close or an equal-weight index of the universe.

```toy
market()
```

## Returns, risk and volume

### returns

<p class="sig"><code>returns(period[, 'field'])</code></p>

The return over `period` days: today's price over the last price at or before `period` days ago, minus 1. The price is
`close` unless another field is named; `returns(1, 'market')` is the market's daily return.

On a day with no price there is no return, so a halted stock drops out of `rank` and the group operators instead of
counting as 0%. The day it trades again carries the whole move since its last price: below, **C** on Jan 05 is
31 / 30.5 − 1, measured from Jan 03. A price more than 10 days old no longer counts.

```toy
close()
returns(1)
```

```qex
returns(20)                 # one month
returns(1, 'market')        # the market's daily return
-returns(5)                 # last week, reversed: a reversal signal
```

### volatility

<p class="sig"><code>volatility(period[, 'field'])</code></p>

Annualized volatility: the standard deviation of daily returns over `period` days, times √252. Daily returns follow the
rule above. Part of the window ([window rules](time-series.md#window-rules)).

```qex
volatility(60)
returns(60) / volatility(60)    # risk-adjusted momentum
```

### volume

<p class="sig"><code>volume(period)</code></p>

Average daily volume over `period` days, in shares; `volume(1)` is the day's volume. Part of the window.

```toy
volume(1)
```

```qex
volume(5) / volume(60)          # recent volume against its norm
```

### adv

<p class="sig"><code>adv(period)</code></p>

Average daily dollar volume, close × volume, over `period` days. Part of the window. Dollar volume compares across
stocks; share volume does not.

```qex
log(adv(20))                    # liquidity, on a log scale
where(adv(20) > 5e6, rank(-returns(5)), None)    # trade only the liquid names
```
