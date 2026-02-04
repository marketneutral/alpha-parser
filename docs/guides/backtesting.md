# Backtesting

Evaluating signal performance.

## Basic Backtest

```python
from qex import qex, Backtest

signal = qex("rank(returns(20)) - 0.5")

bt = Backtest(signal)
result = bt.run(data)

print(result.summary())
```

## With Transaction Costs

```python
# 10 basis points round-trip
bt = Backtest(signal, transaction_cost=0.001)
result = bt.run(data)
```

## Understanding Results

```python
result = bt.run(data)

# Key metrics
print(f"Total Return: {result.total_return:.1%}")
print(f"Sharpe: {result.sharpe:.2f}")
print(f"Max Drawdown: {result.max_dd:.1%}")
print(f"Avg Turnover: {result.turnover.mean():.1%}")

# Time series
result.returns          # Daily returns
result.cumulative       # Cumulative returns
result.weights          # Daily position weights
```

## Date Filtering

```python
result = bt.run(
    data,
    start='2022-01-01',
    end='2023-12-31'
)
```

## Walk-Forward Analysis

Test stability over time:

```python
results = bt.run_walk_forward(
    data,
    train_period=252,  # 1 year training window
    test_period=63,    # 1 quarter out-of-sample
)

for period, result in results.items():
    print(f"{period}: Sharpe = {result.sharpe:.2f}")
```

## Quantile Analysis

Understand signal predictive power:

```python
from qex import QuantileAnalysis

qa = QuantileAnalysis(signal, n_quantiles=5)
result = qa.run(data)

print(result.summary())
```

Output:
```
Returns by Quantile (annualized):
  Q1:  -3.21%  Sharpe: -0.25
  Q2:   1.05%  Sharpe:  0.08
  Q3:   2.41%  Sharpe:  0.18
  Q4:   4.12%  Sharpe:  0.31
  Q5:   7.85%  Sharpe:  0.59

Long-Short Spread (Q5 - Q1):
  Annual Return:   11.06%
  Sharpe Ratio:     0.84
```

## IC Analysis

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

Key metrics:
- **Mean IC**: Average predictive power (0.03+ is decent)
- **IC IR**: IC / Std(IC), consistency of prediction
- **% Positive**: How often signal predicts correctly

## Comparing Signals

```python
signals = {
    'momentum': qex("rank(returns(60)) - 0.5"),
    'reversal': qex("rank(-returns(5)) - 0.5"),
    'combined': qex("0.5*rank(returns(60)) + 0.5*rank(-returns(5)) - 0.5"),
}

for name, signal in signals.items():
    result = Backtest(signal, transaction_cost=0.001).run(data)
    print(f"{name}: Sharpe = {result.sharpe:.2f}, MaxDD = {result.max_dd:.1%}")
```

## Signal Decay Analysis

How quickly does signal alpha decay?

```python
from qex import qex, QuantileAnalysis

signal = qex("rank(returns(20))")

# Test at different holding periods
for hold in [1, 5, 10, 20]:
    # Simple decay test: evaluate at T, measure returns at T+hold
    qa = QuantileAnalysis(signal, n_quantiles=5)
    result = qa.run(data)
    print(f"Hold {hold}d: Spread Sharpe = {result.spread_sharpe:.2f}")
```

## Individual Metrics

```python
from qex import (
    sharpe_ratio,
    max_drawdown,
    top_drawdowns,
    calmar_ratio,
    sortino_ratio,
)

returns = result.returns

print(f"Sharpe: {sharpe_ratio(returns):.2f}")
print(f"Sortino: {sortino_ratio(returns):.2f}")
print(f"Calmar: {calmar_ratio(returns):.2f}")

# Top drawdowns with dates
for dd in top_drawdowns(returns, n=3):
    print(f"{dd.depth:.1%} from {dd.start} to {dd.end}")
```

## Tips

1. **Always use transaction costs** - Real performance is lower
2. **Check turnover** - High turnover + high TC = bad
3. **Use quantile analysis** - Verify monotonic relationship
4. **Walk-forward test** - Avoid overfitting to in-sample
5. **Check IC stability** - Consistent > high average
