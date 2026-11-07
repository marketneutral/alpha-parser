# Pairs Trading

Statistical arbitrage with grouped instruments.

## Concept

Pairs trading bets on mean reversion within groups:

1. Group related stocks (sector, industry, pairs)
2. Compute relative value (spread)
3. Trade the spread back to equilibrium

## Basic Pairs Signal

Long laggards, short leaders within each sector:

```qex
group_demean(-returns(5), 'sector')
```

`group_demean` subtracts the group mean, so:
- Underperformers (negative deviation) → positive signal
- Outperformers (positive deviation) → negative signal

## Z-Score Normalized

Normalize the spread by its own volatility over time:

```qex
let spread = group_demean(returns(5), 'sector'),
    spread_vol = ts_std(spread, 60)
in -spread / spread_vol
```

`group_std(returns(5), 'sector')` is a different number: how far apart the sector's members are on
one day. Dividing by it, `group_demean(x, 'sector') / group_std(x, 'sector')`, is a z-score within the
sector that day, not a measure of how stretched a spread is against its own history.

## Threshold Trading

Only trade when spread is stretched:

```qex
let spread = group_demean(returns(5), 'sector'),
    spread_vol = ts_std(spread, 60),
    z = spread / spread_vol
in where(abs(z) > 1.5, -spread, None)
```

## Volatility-Matched Pairs

For pairs with different volatilities, compare risk-adjusted moves, so the quieter stock's
move counts as much as the noisier one's:

```qex
group_demean(returns(5) / volatility(60), 'pair')
```

## Partner Returns

For a group of two, the other member's return is the group sum minus your own. That gives
each stock its partner's return with no extra data:

```qex
let r = returns(1),
    partner = group_sum(r, 'pair') - r
in ts_beta(r, partner, 60)
```

## Multi-Leg Pairs

For groups with >2 members:

```qex
# Long the bottom 40% of each sector by recent return, short the top 40%

let ret = returns(5),
    grp_rank = group_rank(ret, 'sector')
in where(grp_rank < 0.4, 1, where(grp_rank > 0.6, -1, None))
```

## Correlation Filter

Only trade correlated pairs:

```qex
let r = returns(1),
    partner = group_sum(r, 'pair') - r,
    corr = ts_corr(r, partner, 60),
    spread = group_demean(returns(5), 'pair')
in where(corr > 0.7, -spread, None)
```
