# Evaluation Module

Backtesting and quantile analysis tools.

## Backtest

Run a backtest on any signal:

```python
from qex import qex, Backtest

signal = qex("rank(returns(20)) - 0.5")

bt = Backtest(signal, transaction_cost=0.001)
result = bt.run(data)

print(result.summary())
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `signal` | Signal | required | Signal to backtest |
| `transaction_cost` | float | `0.0` | Round-trip cost (e.g., 0.001 = 10bps) |

### BacktestResult

The `result` object contains:

| Attribute | Type | Description |
|-----------|------|-------------|
| `returns` | Series | Daily returns |
| `cumulative_returns` | Series | Cumulative returns |
| `weights` | DataFrame | Daily position weights |
| `total_return` | float | Total return |
| `sharpe` | float | Annualized Sharpe ratio |
| `max_dd` | float | Maximum drawdown |
| `volatility` | float | Annualized volatility |
| `turnover` | Series | Daily turnover |

```python
print(f"Sharpe: {result.sharpe:.2f}")
print(f"Max Drawdown: {result.max_dd:.1%}")
```

### Date Filtering

```python
result = bt.run(data, start='2022-01-01', end='2023-12-31')
```

### Walk-Forward Analysis

```python
results = bt.run_walk_forward(
    data,
    train_period=252,  # 1 year training
    test_period=63,    # 1 quarter testing
)
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

### QuantileResult

| Attribute | Type | Description |
|-----------|------|-------------|
| `quantile_returns` | DataFrame | Daily returns by quantile |
| `quantile_cumulative` | DataFrame | Cumulative returns by quantile |
| `spread_returns` | Series | Q5 - Q1 returns |
| `spread_sharpe` | float | Sharpe of long-short spread |
| `is_monotonic` | bool | Whether returns increase with quantile |

### IC Analysis

Information Coefficient (rank correlation with forward returns):

```python
ic_stats = qa.ic_summary(data)
print(ic_stats)
```

Output:
```
Mean Rank IC       0.032
Std Rank IC        0.089
IC IR (Rank)       0.360
% Positive IC      0.584
```

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

print(f"Sharpe: {sharpe_ratio(returns):.2f}")
print(f"Max DD: {max_drawdown(returns):.1%}")
print(f"Calmar: {calmar_ratio(returns):.2f}")
```

### Top Drawdowns

```python
drawdowns = top_drawdowns(returns, n=5)
for dd in drawdowns:
    print(f"{dd.depth:.1%} from {dd.start} to {dd.end}")
```

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
