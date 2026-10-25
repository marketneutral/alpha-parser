# Pairs Trading

Statistical arbitrage with grouped instruments.

## Concept

Pairs trading bets on mean reversion within groups:

1. Group related stocks (sector, industry, pairs)
2. Compute relative value (spread)
3. Trade the spread back to equilibrium

## Basic Pairs Signal

Long laggards, short leaders within each sector:

```python
qex("group_demean(-returns(5), 'sector')")
```

`group_demean` subtracts the group mean, so:
- Underperformers (negative deviation) → positive signal
- Outperformers (positive deviation) → negative signal

## Z-Score Normalized

Normalize by historical spread volatility:

```python
qex("""
    let spread = group_demean(returns(5), 'sector'),
        spread_vol = group_std(returns(5), 'sector', 60)
    in -spread / spread_vol
""")
```

## Threshold Trading

Only trade when spread is stretched:

```python
qex("""
    let spread = group_demean(returns(5), 'sector'),
        spread_vol = group_std(returns(5), 'sector', 60),
        z = spread / spread_vol
    in where(abs(z) > 1.5, -spread, 0)
""")
```

## Volatility-Matched Pairs

For pairs with different volatilities, compare risk-adjusted moves, so the quieter stock's
move counts as much as the noisier one's:

```python
qex("group_demean(returns(5) / volatility(60), 'pair')")
```

## Partner Returns

For a group of two, the other member's return is the group sum minus your own. That gives
each stock its partner's return with no extra data:

```python
qex("""
    let r = returns(1),
        partner = group_sum(r, 'pair') - r
    in ts_beta(r, partner, 60)
""")
```

## Setting Up Pair Data

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

## Multi-Leg Pairs

For groups with >2 members:

```python
# Long the bottom 40% of each sector by recent return, short the top 40%
qex("""
    let ret = returns(5),
        grp_rank = group_rank(ret, 'sector')
    in where(grp_rank < 0.4, 1, where(grp_rank > 0.6, -1, 0))
""")
```

## Correlation Filter

Only trade correlated pairs:

```python
qex("""
    let r = returns(1),
        partner = group_sum(r, 'pair') - r,
        corr = ts_corr(r, partner, 60),
        spread = group_demean(returns(5), 'pair')
    in where(corr > 0.7, -spread, 0)
""")
```

## Backtest Example

```python
from qex import qex, Backtest

signal = qex("""
    let spread = group_demean(returns(5), 'sector'),
        spread_vol = group_std(returns(5), 'sector', 60),
        z = spread / spread_vol
    in where(abs(z) > 1.5, -z, 0)
""")

bt = Backtest(signal, transaction_cost=0.001)
result = bt.run(data)

print(result.summary())
```
