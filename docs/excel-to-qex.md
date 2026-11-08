# Excel to qex

*A walkthrough for spreadsheet people.*

If you know how to use Excel, you are well on your way to expressing quant ideas in the qex language! A sheet of
prices, a few columns of formulas, a ranking at the end: that's how qex thinks too. It just writes it down
differently.

To be clear about what qex is: it isn't an Excel add-in, and it doesn't run inside a spreadsheet. It's a short
language for describing a signal. You write the steps as one line, and qex works them out for every stock and every
day at once.

This page builds one signal twice: first the way you would in Excel, then the way you'd write it in qex. Same idea,
same numbers. If you'd like to follow along, the workbook has every step as ordinary Excel formulas.

[Download the Excel workbook](excel/excel-to-qex.xlsx){ .md-button .md-button--primary }

## The idea

**Short-term reversal.** Stocks that fell hard last week tend to bounce back, and stocks that rose tend to give some
of it back. A portfolio manager would ask for three refinements before trading it:

1. **Scale by volatility.** A 5% drop is news for a utility and noise for a biotech. Measure last week's move in
   units of the stock's own daily volatility.
2. **Compare within the sector.** If every energy stock fell because oil fell, that is not an overreaction to buy.
   Compare each stock with its own sector.
3. **Rank.** Turn the scores into percentiles, so one extreme stock doesn't dominate the book.

The data: six stocks over 30 trading days, AAA, BBB and CCC in Tech, DDD, EEE and FFF in Energy. The tables below
show the last five days.

```toy inputs data=excel tail=5
close()
```

## Step by step in Excel

The workbook has one sheet per step. Every sheet has the same layout: dates down column A from row 3, one stock per
column from B to G, and each stock's sector in row 2. Each step is one formula, written in one cell and filled across
the grid. The formulas below are the ones in cell B23 (AAA on Mar 29, the first day every step has a value).

### 1. Daily returns

Sheet **Returns**: today's close over yesterday's, minus 1.

```text
=Prices!B23/Prices!B22-1
```

In qex: `returns(1)`.

```toy data=excel tail=5
returns(1)
```

### 2. Last week's return

Sheet **Return5**: today's close over the close five rows up, minus 1.

```text
=Prices!B23/Prices!B18-1
```

In qex: `returns(5)`.

```toy data=excel tail=5
returns(5)
```

### 3. Volatility

Sheet **Vol**: the standard deviation of the last 20 daily returns. This is why the signal starts on row 23: the
first 20 returns run from row 4 to row 23.

```text
=STDEV.S(Returns!B4:B23)
```

In qex: `ts_std(returns(1), 20)`. Any qex operator whose name starts with `ts_` looks back over a window of rows like
this one, for each stock.

```toy data=excel tail=5
ts_std(returns(1), 20)
```

### 4. The score

Sheet **Score**: last week's return, flipped (a loss is a buy), in units of daily volatility.

```text
=-Return5!B23/Vol!B23
```

In qex: `-returns(5) / ts_std(returns(1), 20)`. Arithmetic works on the whole grid at once, cell by cell, as if
you had filled the formula everywhere.

```toy data=excel tail=5
-returns(5) / ts_std(returns(1), 20)
```

### 5. Sector-neutral

Sheet **Neutral**: the score minus the average score of the stock's sector that day. `AVERAGEIFS` averages the
day's row over the columns whose sector (row 2) matches this stock's.

```text
=Score!B23-AVERAGEIFS(Score!$B23:$G23,Prices!$B$2:$G$2,Prices!B$2)
```

In qex: `group_demean(score, 'sector')`. Any qex operator whose name starts with `group_` works across the row, one
sector at a time. In each sector the three numbers now add up to zero: a long-short bet inside Tech and another
inside Energy, with no bet on Tech against Energy.

```toy data=excel tail=5
group_demean(-returns(5) / ts_std(returns(1), 20), 'sector')
```

### 6. Rank

Sheet **Rank**: each stock's percentile across the row, from 1/6 for the lowest to 1 for the highest.

```text
=RANK.AVG(Neutral!B23,Neutral!$B23:$G23,1)/COUNT(Neutral!$B23:$G23)
```

In qex: `rank(...)`. The highest rank is the stock that fell furthest last week, for its volatility, against its
sector: the one to buy.

```toy data=excel tail=5
rank(group_demean(-returns(5) / ts_std(returns(1), 20), 'sector'))
```

## The same signal in qex

Now the qex side. Everything you just built, six sheets of formulas, is one line:

```qex
rank(group_demean(-returns(5) / ts_std(returns(1), 20), 'sector'))
```

If you prefer to see the steps, give each one a name, the way each sheet has one. This is the same signal:

```qex
let daily   = returns(1),                       # sheet Returns
    week    = returns(5),                       # sheet Return5
    vol     = ts_std(daily, 20),                # sheet Vol
    score   = -week / vol,                      # sheet Score
    neutral = group_demean(score, 'sector')     # sheet Neutral
in rank(neutral)                                # sheet Rank
```

The numbers are the same too: the workbook's values match qex's to the 15th decimal place.

You don't type this into Excel. You hand it to qex together with your price data, and qex fills in every stock and
every day. The [Quick Start](python-examples/quickstart.md) shows how.

## The translation

| In Excel | In qex |
|---|---|
| A sheet for each step | A name in `let`, or just nesting |
| Filling a formula down the dates and across the stocks | Nothing: every operator works on the whole grid |
| A formula over the last 20 rows, `STDEV.S(B4:B23)` | A time-series operator, `ts_std(x, 20)` |
| `AVERAGEIFS` over the stocks in the same sector | A group operator, `group_demean(x, 'sector')` |
| `RANK.AVG` over the day's row | A cross-sectional operator, `rank(x)` |
| Starting the formulas on row 23, once there is enough history | Automatic: no value until the window has its data |

## What qex does for you

The workbook works because its data is clean. Real data isn't, and this is where a spreadsheet gets hard:

- **A stock that didn't trade.** A blank price breaks `=B23/B22-1`. qex leaves that day's return empty, and the next
  day's return spans the gap. See [Missing Values](language/missing-values.md).
- **Looking ahead by accident.** One formula that points a row down uses tomorrow's price, and the backtest looks
  wonderful. qex has no way to refer to the future.
- **Size.** 3,000 stocks over 20 years is 15 million cells a sheet. qex evaluates the expression in seconds.
- **Changing your mind.** A 60-day volatility, or industries instead of sectors, is one edit in qex, not a rebuilt
  workbook.

## Next

- [Philosophy](philosophy.md): why signals are written this way.
- [Syntax](language/syntax.md) and the [Operator Reference](language/reference.md): the whole language, with a worked
  example for most operators.
- [Examples](examples/index.md): momentum, mean reversion, pairs, earnings and more, written in qex.
