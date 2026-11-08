# Evaluation

Backtests, quantile and IC analysis, and performance metrics. Everything here is importable from `qex` as well as
from `evaluation`. How a signal becomes a book is in [Books](books.md).

## Backtest

Run a backtest on any signal:

```python
from qex import qex, Backtest

signal = qex("demean(rank(returns(20)))")

bt = Backtest(signal, transaction_cost=0.001)
result = bt.run(data)

print(result.summary())
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `signal` | Signal | required | Signal to backtest |
| `holding_period` | int | `1` | Days each day's target book is held: the book held is the average of the last `holding_period` targets (below) |
| `transaction_cost` | float | `0.0` | Cost per dollar traded (0.001 = 10 bps), charged on daily turnover |
| `settings` | BookSettings | `None` | Build the book with [book settings](books.md#book-settings); `None` uses `signal.to_weights()` at delay 0 |

`run(data, start_date=None, end_date=None)` returns a `BacktestResult`.

The backtest starts on the first day the book holds a position. A signal has no value while its windows fill (a
60-day volatility, its first 60 days), and those days are not counted as a flat book; a flat day once the book is
running is. Every series and summary number below covers the days from that start. Each day's weights earn the next
day's return, so the last day of the data has no return and is left out.

### BacktestResult

| Attribute | Type | Description |
|-----------|------|-------------|
| `pnl` | Series | Daily P&L, after costs |
| `cumulative_pnl` | Series | Cumulative P&L |
| `returns` | Series | Daily returns (P&L / gross) |
| `weights` | DataFrame | Daily position weights |
| `total_return` | float | Total compounded return |
| `sharpe` | float | Annualized Sharpe ratio |
| `annual_return` | float | Annualized return |
| `annual_volatility` | float | Annualized volatility |
| `max_drawdown` | float | Maximum drawdown (positive decimal) |
| `calmar` | float | Calmar ratio |
| `sortino` | float | Sortino ratio |
| `return_gmv` | float | Total P&L / average gross |
| `drawdowns` | list | The five deepest drawdowns, as `Drawdown` objects |
| `avg_long_count`, `avg_short_count` | float | Average number of long and short positions |
| `avg_turnover` | float | Average daily turnover |
| `fitness` | float | Sharpe × √(\|annual return\| / max(turnover, 0.125)), before costs |
| `margin_bps` | float | P&L per dollar traded, in basis points, before costs |

```python
print(f"Sharpe: {result.sharpe:.2f}")
print(f"Max Drawdown: {result.max_drawdown:.1%}")
```

### Date Filtering

```python
result = bt.run(data, start_date='2022-01-01', end_date='2023-12-31')
```

### Walk-Forward Analysis

```python
results = bt.run_walk_forward(
    data,
    train_period=252,  # days skipped before the first window
    test_period=63,    # days per window
)
```

Returns a DataFrame with one row per window: `start`, `end`, `sharpe`, `return`,
`volatility`, `max_drawdown`. Nothing is fitted on the first `train_period` days.

### Holding period

With `holding_period=h > 1`, `Backtest` holds each day's target book for h days, as h overlapping books
(Jegadeesh and Titman, 1993): the book held on a day is the average of the last h days' targets, so 1/h of it is
replaced each day. P&L is that book times one-day returns and turnover is its change, so every statistic reads as
for a daily book, and a longer holding period trades less. In the first h - 1 days the book is partly in cash
(k/h after k days); `returns` are measured on the gross held, so they are not diluted.

```python
held = Backtest(signal, holding_period=5).run(data)       # 5 books, each a fifth, each held a week
```

(Before qex 0.7.1, `holding_period=h` rebalanced daily on h-day forward returns and summed them as daily P&L, which
counted each day's move up to h times.)

`QuantileAnalysis(holding_period=h)` is unchanged: its IC on h-day forward returns is the standard measure, and its
quantile return series overlap (a day's move is in h consecutive values), so read them as h-day returns, not as P&L.


## Quantile Analysis

Analyze signal performance by quantile:

```python
from qex import qex, QuantileAnalysis

signal = qex("rank(returns(60)) - 0.5")

qa = QuantileAnalysis(signal, n_quantiles=5)
result = qa.run(data)

print(result.summary())
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `signal` | Signal | required | Signal to analyze |
| `n_quantiles` | int | `5` | Number of quantile buckets |
| `holding_period` | int | `1` | Forward return period, in days |

### QuantileResult

| Attribute | Type | Description |
|-----------|------|-------------|
| `quantile_returns` | DataFrame | Daily returns by quantile |
| `quantile_cumulative` | DataFrame | Cumulative returns by quantile |
| `mean_returns` | Series | Mean daily return by quantile |
| `sharpe_by_quantile` | Series | Sharpe ratio by quantile |
| `hit_rate` | Series | Fraction of days with a positive return, by quantile |
| `spread_returns` | Series | Top minus bottom quantile, daily |
| `spread_sharpe` | float | Sharpe of the long-short spread |
| `rank_ic` | float | Mean rank information coefficient |
| `is_monotonic` | bool | Whether returns increase (or decrease) with quantile |

### IC Analysis

Information Coefficient (correlation of the signal with forward returns, across names):

```python
ic_stats = qa.ic_summary(data)
print(ic_stats)
```

A Series: mean, standard deviation and IR of the rank and the Pearson IC, the fraction of
days with a positive rank IC, and the average number of names.

## Metrics

Individual metric functions:

```python
from qex import (
    sharpe_ratio,
    max_drawdown,
    top_drawdowns,
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    sortino_ratio,
    return_on_gmv,
)

returns = result.returns
cumulative = (1 + returns).cumprod()

print(f"Sharpe: {sharpe_ratio(returns):.2f}")
print(f"Max DD: {max_drawdown(cumulative):.1%}")      # takes cumulative returns
print(f"Calmar: {calmar_ratio(returns):.2f}")
print(f"Return on GMV: {return_on_gmv(result.pnl, result.weights):.1%}")
```

### Top Drawdowns

```python
for dd in top_drawdowns(cumulative, n=5):
    print(f"{dd.depth:.1%} from {dd.start:%Y-%m-%d} to {dd.end:%Y-%m-%d}")
```

Each `Drawdown` has `start`, `trough`, `end` (recovery, or the last date), `depth`,
`duration` (calendar days from start to trough) and `recovery` (calendar days from trough to
recovery, or −1 if it has not recovered).

## API Reference

::: evaluation.backtest.Backtest
    options:
      members:
        - run
        - run_walk_forward

::: evaluation.quantile.QuantileAnalysis
    options:
      members:
        - run
        - ic_summary
