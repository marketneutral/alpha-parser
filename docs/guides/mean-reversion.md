# Mean Reversion

Building reversal and mean-reversion signals.

## Basic Reversal

Buy losers, sell winners:

```python
# Negate returns so losers get high rank
signal = qex("rank(-returns(5)) - 0.5")
```

## Volatility-Adjusted Reversal

```python
qex("rank(-returns(5) / volatility(20)) - 0.5")
```

## Volume-Weighted Reversal

More conviction when volume is high:

```python
qex("rank(-returns(5) * (volume(5) / adv(20))) - 0.5")
```

## Z-Score Reversal

Trade extreme moves:

```python
# Z-score of recent returns vs history
qex("""
    let ret = returns(5),
        z = (ret - ts_mean(ret, 60)) / ts_std(ret, 60)
    in rank(-z) - 0.5
""")
```

## Conditional Reversal

Only in high-vol regime:

```python
qex("""
    let ret = returns(5),
        vol = volatility(20),
        high_vol = vol > ts_mean(vol, 60)
    in where(high_vol, rank(-ret) - 0.5, 0)
""")
```

## Trend-Aligned Reversal

Reverse only against the trend:

```python
qex("""
    let trend = sign(returns(60)),
        short_ret = returns(5)
    in rank(trend * -short_ret) - 0.5
""")
```

This buys:
- Oversold stocks in uptrends
- Avoids catching falling knives in downtrends

## Bollinger Band Reversion

```python
qex("""
    let price = close(),
        ma = ts_mean(price, 20),
        std = ts_std(price, 20),
        z = (price - ma) / (2 * std)
    in rank(-z) - 0.5
""")
```

## RSI Reversal

```python
qex("""
    let up = ts_mean(max(delta(close(), 1), 0), 14),
        down = ts_mean(max(-delta(close(), 1), 0), 14),
        rsi = 100 * up / (up + down)
    in rank(-rsi) - 0.5
""")
```

## Sector-Neutral Reversal

```python
qex("group_demean(rank(-returns(5)), 'sector')")
```

## Event-Driven Reversal

Fade earnings overreactions:

```python
qex("""
    let surprise = field('earnings_surprise'),
        held = fill_forward(surprise, 5)
    in rank(-held) - 0.5
""")
```

## Decay-Weighted Reversal

Weight recent days more:

```python
qex("rank(-decay_linear(returns(1), 10)) - 0.5")
```

## Combined with Momentum

```python
qex("""
    let mom = rank(returns(60)) - 0.5,
        rev = rank(-returns(5)) - 0.5
    in 0.5 * mom + 0.5 * rev
""")
```

## Backtest Example

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
