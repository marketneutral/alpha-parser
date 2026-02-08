# Advanced Patterns

Complex signal constructions using qex.

## Variable Bindings with `let`

### DRY Signals

Define once, use multiple times:

```python
# Without let (repeated expression)
qex("rank(delta(returns(5)/volatility(10), 20)) * (returns(5)/volatility(10))")

# With let (clean)
qex("""
    let s = returns(5) / volatility(10)
    in rank(delta(s, 20)) * s
""")
```

### Dependent Bindings

Later variables can reference earlier ones:

```python
qex("""
    let mom = returns(60),
        vol = volatility(60),
        sharpe = mom / vol,
        trend = rank(delta(sharpe, 20))
    in trend * sign(mom)
""")
```

## Regime-Dependent Signals

### Volatility Regimes

```python
qex("""
    let vol = volatility(20),
        vol_ma = ts_mean(vol, 60),
        high_vol = vol > vol_ma,
        mom_signal = rank(returns(60)) - 0.5,
        rev_signal = rank(-returns(5)) - 0.5
    in where(high_vol, rev_signal, mom_signal)
""")
```

### Trend Regimes

```python
qex("""
    let price = close(),
        ma50 = ts_mean(price, 50),
        ma200 = ts_mean(price, 200),
        uptrend = ma50 > ma200,
        base_signal = rank(returns(20)) - 0.5
    in where(uptrend, base_signal, -base_signal)
""")
```

## Multi-Factor Composites

### Blended Signals

```python
qex("""
    let mom = rank(returns(60)) - 0.5,
        rev = rank(-returns(5)) - 0.5,
        vol_inv = rank(-volatility(20)) - 0.5
    in 0.4 * mom + 0.3 * rev + 0.3 * vol_inv
""")
```

### Quality-Adjusted Momentum

```python
qex("""
    let mom = rank(returns(60)),
        quality = rank(field('roe')),
        combo = 0.6 * mom + 0.4 * quality
    in rank(combo) - 0.5
""")
```

## Adaptive Parameters

### Volatility-Scaled Position

```python
qex("""
    let signal = rank(returns(20)) - 0.5,
        vol = volatility(20),
        target_vol = 0.15,
        vol_scale = target_vol / vol
    in signal * vol_scale
""")
```

### Dynamic Lookback

```python
qex("""
    let vol = volatility(20),
        fast_mom = returns(20),
        slow_mom = returns(60),
        use_fast = vol > ts_mean(vol, 60)
    in rank(where(use_fast, fast_mom, slow_mom)) - 0.5
""")
```

## Sector-Level Signals

### Sector Momentum

```python
qex("""
    let sector_ret = group_sum(returns(20) * close(), 'sector') /
                     group_sum(close(), 'sector'),
        sector_mom = ts_mean(sector_ret, 10)
    in rank(sector_mom) - 0.5
""")
```

### Industry Rotation

```python
qex("""
    let ind_rank = group_rank(returns(60), 'industry')
    in where(ind_rank > 0.7, 1, where(ind_rank < 0.3, -1, 0))
""")
```

## Sector Filtering

Use `group()` with `where()` to exclude sectors where a signal doesn't apply.

### Price-to-Book for Non-Financials

Price-to-book ratio isn't meaningful for Financials (banks, insurance) because their balance sheets are fundamentally different. Zero them out:

```python
qex("where(group('sector')=='Financials', 0, field('price_to_book'))")
```

### Sector-Specific Factors

Some factors only make sense for certain sectors:

```python
# Inventory turnover - not relevant for services/financials
qex("""
    let inv_turn = field('inventory_turnover'),
        is_relevant = (group('sector') != 'Financials') &
                      (group('sector') != 'Technology')
    in where(is_relevant, rank(inv_turn) - 0.5, 0)
""")

# CapEx/Sales - not meaningful for asset-light businesses
qex("""
    where(group('sector')=='Technology', 0,
        rank(field('capex_to_sales')) - 0.5)
""")
```

### Long-Only Sector Bets

Only trade stocks in a specific sector:

```python
# Only trade Technology stocks (long-only)
qex("where(group('sector')=='Technology', rank(returns(20)), 0)")
```

## Event Integration

### Earnings + Momentum

```python
qex("""
    let surprise = fill_forward(field('earnings_surprise'), 60),
        mom = returns(60),
        aligned = sign(surprise) == sign(mom)
    in where(aligned, rank(mom) - 0.5, 0)
""")
```

Only trade when earnings and momentum agree.

### Announcement Fade with Filter

```python
qex("""
    let is_event = is_valid(field('earnings_surprise')),
        event_ret = where(is_event, returns(1), 0),
        held = fill_forward(event_ret, 5),
        vol = volatility(20),
        high_vol = vol > ts_mean(vol, 60)
    in where(high_vol, rank(-held) - 0.5, 0)
""")
```

## Risk-Controlled Signals

### Drawdown-Aware

```python
qex("""
    let signal = rank(returns(60)) - 0.5,
        cum_ret = ts_sum(returns(1), 20),
        in_drawdown = cum_ret < ts_min(cum_ret, 60)
    in where(in_drawdown, signal * 0.5, signal)
""")
```

### Maximum Position Cap

```python
qex("""
    let raw = rank(returns(20)) - 0.5
    in truncate(scale(winsorize(raw, 0.05)), 0.1)
""")
```

Layers: winsorize outliers → scale to sum=1 → cap max position at 10%.

## Cross-Asset Signals

### Equity-Bond Relationship

```python
qex("""
    let stock_ret = returns(20),
        bond_ret = field('bond_returns'),
        corr = ts_corr(stock_ret, bond_ret, 60),
        decorrelated = corr < 0
    in where(decorrelated, rank(stock_ret) - 0.5, 0)
""")
```

## Seasonality

### Month-End Effect

```python
qex("""
    let dom = day_of_month(),
        month_end = dom > 25,
        base = rank(returns(20)) - 0.5
    in where(month_end, base * 1.5, base)
""")
```

### January Effect

```python
qex("""
    let month = month_of_year(),
        is_jan = month == 1,
        small_cap = rank(-field('market_cap'))
    in where(is_jan, rank(small_cap) - 0.5, 0)
""")
```

## Signal Combination Techniques

### Ensemble Average

```python
qex("""
    let s1 = rank(returns(20)) - 0.5,
        s2 = rank(-returns(5)) - 0.5,
        s3 = rank(ts_corr(returns(1), volume(1), 60)) - 0.5
    in (s1 + s2 + s3) / 3
""")
```

### Rank of Ranks

```python
qex("""
    let r1 = rank(returns(60)),
        r2 = rank(-returns(5)),
        r3 = rank(-volatility(20)),
        combo = r1 + r2 + r3
    in rank(combo) - 0.5
""")
```

### Intersection (AND)

```python
qex("""
    let mom_good = rank(returns(60)) > 0.7,
        vol_good = rank(-volatility(20)) > 0.5,
        both = mom_good & vol_good
    in where(both, 1, 0) - 0.5
""")
```
