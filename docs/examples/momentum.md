# Momentum Strategies

Building momentum signals with qex.

## Basic Momentum

Buy winners, sell losers:

```qex
# 20-day returns, ranked cross-sectionally
rank(returns(20)) - 0.5
```

Higher returns → higher rank → long position. `rank(x) - 0.5` is roughly centred; a book
built with `build_book` is demeaned exactly, and `demean(rank(x))` centres it yourself.

## Time-Scale Variations

```qex
# Short-term (1 week)
rank(returns(5)) - 0.5
# Medium-term (1 month)
rank(returns(20)) - 0.5
# Long-term (6 months)
rank(returns(126)) - 0.5
# Annual (skip recent month to avoid reversal)
rank(delay(returns(252), 20)) - 0.5
```

## Volatility-Adjusted Momentum

Sharpe-like signal:

```qex
rank(returns(60) / volatility(60)) - 0.5
```

## Momentum Acceleration

Bet on improving momentum:

```qex
# Change in momentum over 20 days
rank(delta(returns(60), 20)) - 0.5
```

## Trend-Weighted Momentum

Overweight when momentum is trending:

```qex
let mom = returns(60),
    trend = rank(delta(mom, 20))
in trend * sign(mom)
```

## Volume-Adjusted Momentum

Weight by unusual volume (recent volume relative to its own norm):

```qex
rank(returns(20) * (volume(5) / volume(60))) - 0.5
```

## Sector-Neutral Momentum

Remove sector effects:

```qex
group_demean(rank(returns(60)), 'sector')
```

Or rank within sector:

```qex
group_rank(returns(60), 'sector') - 0.5
```

## Conditional Momentum

Only trade in calm markets:

```qex
where(
    volatility(20) < ts_mean(volatility(20), 60),
    rank(returns(60)) - 0.5,
    0
)
```

## Multi-Horizon Composite

Blend multiple timeframes:

```qex
let short = rank(returns(20)) - 0.5,
    medium = rank(returns(60)) - 0.5,
    long = rank(returns(126)) - 0.5
in 0.2 * short + 0.5 * medium + 0.3 * long
```

## EWMA Momentum

Exponentially weighted returns:

```qex
# More responsive to recent data
rank(ewma(returns(1), 20)) - 0.5
```

## Breakout Momentum

Distance from highs:

```qex
# Proximity to 52-week high
rank(close() / ts_max(close(), 252)) - 0.5
# Days since high (lower = more recent breakout)
rank(-ts_argmax(close(), 252)) - 0.5
```

## Decay-Weighted Momentum

Recent returns weighted more:

```qex
rank(decay_linear(returns(1), 20)) - 0.5
```

## Beta-Neutral Momentum

Momentum books pick up the market's direction: after a rally the winners are the
high-beta names. Regress beta out (see [Market Beta](market-beta.md)):

```qex
let b = ts_beta_shrunk(returns(1), returns(1, 'market'), 252)
in regression_neut(rank(delay(returns(252), 20)), b)
```
