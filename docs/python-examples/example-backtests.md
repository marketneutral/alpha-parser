# Backtesting the Examples

The [Examples](../examples/momentum.md) are written in qex alone. This page runs one from each in a backtest, and
builds the extra data two of them need. The data is any dataset in the [data format](../data/format.md); the
[complete dataset example](../data/format.md#example-complete-dataset) is enough to run all of them.

## Momentum

A volatility-adjusted momentum signal ([Momentum](../examples/momentum.md)):

```python
from qex import qex, Backtest

signal = qex("rank(returns(60) / volatility(60)) - 0.5")

bt = Backtest(signal, transaction_cost=0.001)
result = bt.run(data)

print(result.summary())
```

## Mean reversion

A volatility-adjusted short-term reversal ([Mean Reversion](../examples/mean-reversion.md)):

```python
from qex import qex, Backtest

signal = qex("""
    let ret = returns(5),
        vol = volatility(20)
    in rank(-ret / vol) - 0.5
""")

bt = Backtest(signal, transaction_cost=0.001)
result = bt.run(data)

print(result.summary())
```

## Pairs

Pairs need a label per stock naming its pair ([Labels and Groups](../data/labels.md)):

```python
# Define pair groupings
pairs = {
    'KO': 'beverages', 'PEP': 'beverages',
    'JPM': 'banks', 'BAC': 'banks',
    'XOM': 'oil', 'CVX': 'oil',
}

pair_df = pd.DataFrame(
    [[pairs.get(t, 'other') for t in tickers]] * len(dates),
    index=dates, columns=tickers
)

data['pair'] = pair_df
```

A z-scored spread, traded only when it is stretched ([Pairs Trading](../examples/pairs-trading.md)):

```python
from qex import qex, Backtest

signal = qex("""
    let spread = group_demean(returns(5), 'sector'),
        spread_vol = ts_std(spread, 60),
        z = spread / spread_vol
    in where(abs(z) > 1.5, -z, None)
""")

bt = Backtest(signal, transaction_cost=0.001)
result = bt.run(data)

print(result.summary())
```

## Earnings events

Event data is `NaN` except on announcement days ([Event Data](../data/events.md)). Synthetic quarterly surprises:

```python
# Sparse earnings data
earnings = pd.DataFrame(np.nan, index=dates, columns=tickers)

# Fill in event values
for ticker in tickers:
    # Quarterly dates
    quarter_dates = pd.date_range(
        dates[0], dates[-1], freq='QS'
    )
    for qd in quarter_dates:
        if qd in dates:
            earnings.loc[qd, ticker] = np.random.randn()

data['earnings_surprise'] = earnings
```

Standardized unexpected earnings, held for a quarter ([Event-Driven](../examples/event-driven.md)):

```python
from qex import qex, Backtest

signal = qex("""
    let surprise = field('earnings_surprise'),
        sue = surprise / ts_std_events(surprise, 8),
        held = fill_forward(sue, 60)
    in rank(held) - 0.5
""")

bt = Backtest(signal, transaction_cost=0.001)
result = bt.run(data)

print(result.summary())
```

## Comparing them

```python
from qex import qex, Backtest, compute_context

signals = {
    "momentum": "rank(returns(60) / volatility(60))",
    "reversal": "rank(-returns(5) / volatility(20))",
}
with compute_context():                     # shared parts computed once across the signals
    for name, expr in signals.items():
        r = Backtest(qex(expr), transaction_cost=0.001).run(data)
        print(f"{name:10s} Sharpe {r.sharpe:5.2f}  turnover {r.avg_turnover:6.1%}")
```
