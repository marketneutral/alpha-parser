# Backtesting

Evaluating signal performance.

## Basic Backtest

```python
from qex import qex, Backtest

signal = qex("demean(rank(returns(20)))")

bt = Backtest(signal)
result = bt.run(data)

print(result.summary())
```

## With Transaction Costs

`transaction_cost` is charged on every dollar traded: each day's P&L is reduced by
`turnover * transaction_cost`, where turnover is `sum(|w_t - w_{t-1}|)`.

```python
# 10 basis points per dollar traded
bt = Backtest(signal, transaction_cost=0.001)
result = bt.run(data)
```

## Book Settings (Finding Alphas)

`Backtest` turns a signal into weights with `signal.to_weights()` by default: the signal
scaled so absolute weights sum to one, traded at the same close it was computed on. Pass
`settings` to build the book the way *Finding Alphas* (and WorldQuant BRAIN) do instead:

```python
from qex import qex, Backtest, BookSettings

settings = BookSettings(
    delay=1,              # day t's weights use data up to day t-1's close
    universe="member",    # a data field, 1 for names in the universe that day, NaN otherwise
    neutralize="sector",  # "market", a group field, or None
    truncation=0.05,      # no name above 5% of the book
)
result = Backtest(qex("-returns(5)"), settings=settings).run(data)
print(result.fitness, result.margin_bps)
```

Each day the weights are neutralized, scaled to a book of exactly one (gross), and clipped
at the truncation, repeated until all three hold. The size of the book never changes, so
a signal's magnitude only moves weight between names; it cannot time the book.

### The universe inside the signal

With a universe set, the signal's cross-sectional operators (`rank`, `demean`, `zscore`,
`regression_neut`, the group operators, ...) compare only the names in the universe that
day, as BRAIN does, and give NaN outside it. Time-series operators still see every name's
full history, so a name that joined today has its past returns and beta. See
[the cross-sectional universe](../reference/operations.md#the-cross-sectional-universe).

`BookSettings(mask_cross_section=False)` applies the universe only to the finished signal
instead, so `rank(x)` ranks a member among every name with data. That reproduces books from
qex before 0.5.

### Market beta

Dollar-neutral is not beta-neutral. A signal correlated with beta (low volatility, size,
many value measures) gives a book with a market exposure, and over a rising market its
Sharpe can come from the market alone. Regress beta out before the book is built:

```python
data["market"] = spy_close                                   # a date-indexed Series
signal = qex("""
    let b = ts_beta_shrunk(returns(1), returns(1, 'market'), 252)
    in regression_neut(rank(-returns(5)), b)
""")
w = build_book(signal, data, BookSettings(delay=1, universe="member", truncation=0.05))
```

The book is then close to beta-neutral, not exactly: betas are estimated, and truncation
comes after the regression. Check the realized correlation of the book's P&L with the
market. [Market Beta](market-beta.md) covers estimating, shrinking and removing beta.

### Weights and statistics on their own

```python
from qex import build_book, turnover, fitness, margin_bps

w = build_book(signal, data, settings)            # dates x names
pnl = (w * data["close"].pct_change(fill_method=None).shift(-1)).sum(axis=1)
to = turnover(w)                                  # fraction of the book traded each day
fitness(pnl, to)      # Sharpe * sqrt(|annual return| / max(turnover, 0.125))
margin_bps(pnl, to)   # P&L per dollar traded, in basis points
```

## Understanding Results

```python
result = bt.run(data)

# Summary statistics
print(f"Total Return: {result.total_return:.1%}")
print(f"Sharpe: {result.sharpe:.2f}")
print(f"Max Drawdown: {result.max_drawdown:.1%}")
print(f"Avg Turnover: {result.avg_turnover:.1%}")

# Time series
result.returns          # Daily returns (P&L / gross)
result.pnl              # Daily P&L
result.cumulative_pnl   # Cumulative P&L
result.weights          # Daily position weights
result.drawdowns        # The five deepest drawdowns
```

## Date Filtering

```python
result = bt.run(
    data,
    start_date='2022-01-01',
    end_date='2023-12-31',
)
```

## Walk-Forward Analysis

Score consecutive out-of-sample windows. Nothing is fitted: the first `train_period` days
are skipped (as warm-up), then the signal is scored on each block of `test_period` days.

```python
results = bt.run_walk_forward(
    data,
    train_period=252,  # days skipped before the first window
    test_period=63,    # days per window
)
print(results)        # DataFrame: start, end, sharpe, return, volatility, max_drawdown
```

## Quantile Analysis

Understand signal predictive power:

```python
from qex import QuantileAnalysis

qa = QuantileAnalysis(signal, n_quantiles=5)
result = qa.run(data)

print(result.summary())
```

The summary lists, for each quantile, its annualized return, Sharpe and hit rate; then the
long-short spread (top minus bottom quantile), the rank IC, and whether returns are
monotonic in the quantile. On 50 random-walk prices, where there is nothing to find:

```
Returns by Quantile (annualized):
  Q1:   5.28%  Sharpe:  0.52  Hit: 50.2%
  Q2:   8.51%  Sharpe:  0.82  Hit: 53.2%
  Q3:   2.20%  Sharpe:  0.21  Hit: 49.5%
  Q4:   4.80%  Sharpe:  0.49  Hit: 52.4%
  Q5:  -0.99%  Sharpe: -0.10  Hit: 47.8%

Long-Short Spread (Q5 - Q1):
  Annual Return:  -6.27%
  Sharpe Ratio:    -0.43

Rank IC: -0.005
Monotonic: No
```

`QuantileAnalysis(signal, n_quantiles=5, holding_period=1)` measures forward returns over
`holding_period` days.

## IC Analysis

Information Coefficient (correlation of the signal with forward returns, across names, each day):

```python
ic_stats = qa.ic_summary(data)
print(ic_stats)
```

`ic_summary` returns a Series with the mean, standard deviation and IR (mean / std) of the
rank and Pearson IC, the fraction of days with a positive IC, and the average number of names.

Key metrics:

- **Mean IC**: Average predictive power (0.03+ is decent)
- **IC IR**: IC / Std(IC), consistency of prediction
- **% Positive**: How often the signal's ordering agrees with the next return's

## Comparing Signals

```python
signals = {
    'momentum': qex("rank(returns(60)) - 0.5"),
    'reversal': qex("rank(-returns(5)) - 0.5"),
    'combined': qex("0.5*rank(returns(60)) + 0.5*rank(-returns(5)) - 0.5"),
}

for name, signal in signals.items():
    result = Backtest(signal, transaction_cost=0.001).run(data)
    print(f"{name}: Sharpe = {result.sharpe:.2f}, MaxDD = {result.max_drawdown:.1%}")
```

## Signal Decay Analysis

How quickly does the signal's edge fade? Measure the spread over longer forward returns:

```python
from qex import qex, QuantileAnalysis

signal = qex("rank(returns(20))")

for hold in [1, 5, 10, 20]:
    result = QuantileAnalysis(signal, n_quantiles=5, holding_period=hold).run(data)
    print(f"Hold {hold}d: Spread Sharpe = {result.spread_sharpe:.2f}, Rank IC = {result.rank_ic:.3f}")
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
cumulative = (1 + returns).cumprod()          # drawdowns are measured on cumulative returns

print(f"Sharpe: {sharpe_ratio(returns):.2f}")
print(f"Sortino: {sortino_ratio(returns):.2f}")
print(f"Calmar: {calmar_ratio(returns):.2f}")
print(f"Max DD: {max_drawdown(cumulative):.1%}")

# Top drawdowns with dates
for dd in top_drawdowns(cumulative, n=3):
    print(f"{dd.depth:.1%} from {dd.start:%Y-%m-%d} to {dd.end:%Y-%m-%d}")
```

## Tips

1. **Always use transaction costs** - Real performance is lower
2. **Check turnover** - High turnover + high TC = bad
3. **Use quantile analysis** - Verify monotonic relationship
4. **Check market exposure** - A dollar-neutral book can still be a market bet
5. **Score out of sample** - Use walk-forward windows and a held-out period
6. **Check IC stability** - Consistent > high average
