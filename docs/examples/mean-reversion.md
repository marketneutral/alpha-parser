# Mean Reversion

Building reversal and mean-reversion signals.

## Basic Reversal

Buy losers, sell winners:

```qex
# Negate returns so losers get high rank
rank(-returns(5)) - 0.5
```

## Volatility-Adjusted Reversal

```qex
rank(-returns(5) / volatility(20)) - 0.5
```

## Volume-Weighted Reversal

More conviction when volume is high relative to its norm:

```qex
rank(-returns(5) * (volume(5) / volume(60))) - 0.5
```

## Z-Score Reversal

Trade extreme moves:

```qex
# Z-score of recent returns vs history

let ret = returns(5),
    z = (ret - ts_mean(ret, 60)) / ts_std(ret, 60)
in rank(-z) - 0.5
```

## Conditional Reversal

Only in high-vol regime:

```qex
let ret = returns(5),
    vol = volatility(20),
    high_vol = vol > ts_mean(vol, 60)
in where(high_vol, rank(-ret) - 0.5, None)
```

## Trend-Aligned Reversal

Reverse only against the trend:

```qex
let trend = sign(returns(60)),
    short_ret = returns(5)
in rank(trend * -short_ret) - 0.5
```

In an uptrend this buys short-term losers (dips). In a downtrend the sign flips: it buys
short-term *winners*, so it never buys a stock that is falling in a falling trend.

## Bollinger Band Reversion

```qex
let price = close(),
    ma = ts_mean(price, 20),
    std = ts_std(price, 20),
    z = (price - ma) / (2 * std)
in rank(-z) - 0.5
```

## RSI Reversal

```qex
let up = ts_mean(max(delta(close(), 1), 0), 14),
    down = ts_mean(max(-delta(close(), 1), 0), 14),
    rsi = 100 * up / (up + down)
in rank(-rsi) - 0.5
```

## Sector-Neutral Reversal

```qex
group_demean(rank(-returns(5)), 'sector')
```

## Event-Driven Reversal

Fade earnings overreactions:

```qex
let surprise = field('earnings_surprise'),
    held = fill_forward(surprise, 5)
in rank(-held) - 0.5
```

## Decay-Weighted Reversal

Weight recent days more:

```qex
rank(-decay_linear(returns(1), 10)) - 0.5
```

## Combined with Momentum

```qex
let mom = rank(returns(60)) - 0.5,
    rev = rank(-returns(5)) - 0.5
in 0.5 * mom + 0.5 * rev
```
