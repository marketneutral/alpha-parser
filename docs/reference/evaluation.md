# Evaluation Module

Backtesting, book construction and quantile analysis tools. Everything here is importable
from `qex` as well as from `evaluation`.

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
| `settings` | BookSettings | `None` | Build the book with [book settings](#book-settings); `None` uses `signal.to_weights()` at delay 0 |

`run(data, start_date=None, end_date=None)` returns a `BacktestResult`.

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

## Book Settings

`BookSettings` and `build_book` turn a signal into the book *Finding Alphas* and WorldQuant
BRAIN simulate: restricted to a universe, delayed, neutralized, scaled to a gross of one and
truncated.

```python
from qex import qex, BookSettings, build_book

settings = BookSettings(delay=1, universe='member', neutralize='market', truncation=0.05)
w = build_book(qex("rank(-returns(5))"), data, settings)
```

| Setting | Default | Description |
|---------|---------|-------------|
| `delay` | `1` | Day t's weights use data up to day t − delay |
| `universe` | `None` | A data field, 1 for names in the universe that day and NaN otherwise; `None` is every name with a price |
| `neutralize` | `"market"` | `"market"`, a group field such as `"sector"`, or `None` |
| `truncation` | `0.05` | Largest absolute weight of one name, as a fraction of the book; `None` for no cap |
| `rounds` | `1000` | Most neutralize, scale and clip rounds per day |
| `mask_cross_section` | `True` | With a universe, cross-sectional operators in the signal compare only the day's universe ([details](operations.md#the-cross-sectional-universe)); `False` applies the universe to the finished signal only, as before 0.5 |

### Untradable names

A name with no close that day can't be traded. `Backtest` holds its previous weight until it
trades again, for at most `MAX_PRICE_GAP` (10) days, and then drops it (a delisting).
`hold_untradable` applies the same rule to any weights:

```python
from qex import hold_untradable, forward_returns

held = hold_untradable(w, data['close'])                 # what Backtest holds
pnl = (held * forward_returns(data['close']).reindex_like(held).fillna(0)).sum(axis=1)
```

`Backtest` books P&L daily, on `forward_returns(close, 1)`. `QuantileAnalysis` measures
`forward_returns(close, holding_period)`. Both are NaN when the end price is missing, and measured from the last
valid price when the start price is.

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

### Book statistics

```python
from qex import turnover, fitness, margin_bps

to = turnover(w)          # sum(|w_t - w_{t-1}|) each day
fitness(pnl, to)          # Sharpe * sqrt(|annual return| / max(mean turnover, 0.125))
margin_bps(pnl, to)       # total P&L / total traded, in basis points
```

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

::: evaluation.book.BookSettings

::: evaluation.book.build_book

::: evaluation.quantile.QuantileAnalysis
    options:
      members:
        - run
        - ic_summary
