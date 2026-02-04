# Momentum Strategies

Building momentum signals with qex.

## Basic Momentum

Buy winners, sell losers:

```python
# 20-day returns, ranked cross-sectionally
signal = qex("rank(returns(20)) - 0.5")
```

Higher returns → higher rank → long position.

## Time-Scale Variations

```python
# Short-term (1 week)
qex("rank(returns(5)) - 0.5")

# Medium-term (1 month)
qex("rank(returns(20)) - 0.5")

# Long-term (6 months)
qex("rank(returns(126)) - 0.5")

# Annual (skip recent month to avoid reversal)
qex("rank(delay(returns(252), 20)) - 0.5")
```

## Volatility-Adjusted Momentum

Sharpe-like signal:

```python
qex("rank(returns(60) / volatility(60)) - 0.5")
```

## Momentum Acceleration

Bet on improving momentum:

```python
# Change in momentum over 20 days
qex("rank(delta(returns(60), 20)) - 0.5")
```

## Trend-Weighted Momentum

Overweight when momentum is trending:

```python
qex("""
    let mom = returns(60),
        trend = rank(delta(mom, 20))
    in trend * sign(mom)
""")
```

## Volume-Adjusted Momentum

Weight by unusual volume:

```python
qex("rank(returns(20) * (volume(5) / adv(20))) - 0.5")
```

## Sector-Neutral Momentum

Remove sector effects:

```python
qex("group_demean(rank(returns(60)), 'sector') ")
```

Or rank within sector:

```python
qex("group_rank(returns(60), 'sector') - 0.5")
```

## Conditional Momentum

Only trade in calm markets:

```python
qex("""
    where(
        volatility(20) < ts_mean(volatility(20), 60),
        rank(returns(60)) - 0.5,
        0
    )
""")
```

## Multi-Horizon Composite

Blend multiple timeframes:

```python
qex("""
    let short = rank(returns(20)) - 0.5,
        medium = rank(returns(60)) - 0.5,
        long = rank(returns(126)) - 0.5
    in 0.2 * short + 0.5 * medium + 0.3 * long
""")
```

## EWMA Momentum

Exponentially weighted returns:

```python
# More responsive to recent data
qex("rank(ewma(returns(1), 20)) - 0.5")
```

## Breakout Momentum

Distance from highs:

```python
# Proximity to 52-week high
qex("rank(close() / ts_max(close(), 252)) - 0.5")

# Days since high (lower = more recent breakout)
qex("rank(-ts_argmax(close(), 252)) - 0.5")
```

## Decay-Weighted Momentum

Recent returns weighted more:

```python
qex("rank(decay_linear(returns(1), 20)) - 0.5")
```

## Backtest Example

```python
from qex import qex, Backtest

signal = qex("rank(returns(60) / volatility(60)) - 0.5")

bt = Backtest(signal, transaction_cost=0.001)
result = bt.run(data)

print(result.summary())
```
