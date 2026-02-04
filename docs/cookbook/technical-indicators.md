# Technical Indicators

Classic indicators expressed as qex signals.

## Bollinger Bands

```python
# %B indicator (position within bands)
qex("""
    let price = close(),
        ma = ts_mean(price, 20),
        std = ts_std(price, 20),
        upper = ma + 2*std,
        lower = ma - 2*std,
        pct_b = (price - lower) / (upper - lower)
    in rank(-pct_b) - 0.5
""")
```

Mean reversion: buy when near lower band, sell when near upper.

## RSI (Relative Strength Index)

```python
qex("""
    let up = ts_mean(max(delta(close(), 1), 0), 14),
        down = ts_mean(max(-delta(close(), 1), 0), 14),
        rsi = 100 * up / (up + down)
    in rank(-rsi) - 0.5
""")
```

Mean reversion: buy oversold (low RSI), sell overbought (high RSI).

## MACD

```python
qex("""
    let fast = ewma(close(), 12),
        slow = ewma(close(), 26),
        macd = fast - slow,
        signal_line = ewma(macd, 9),
        histogram = macd - signal_line
    in rank(histogram) - 0.5
""")
```

## Moving Average Crossover

```python
qex("""
    let ma_fast = ts_mean(close(), 10),
        ma_slow = ts_mean(close(), 50),
        cross = ma_fast - ma_slow
    in rank(cross) - 0.5
""")
```

## ATR (Average True Range)

Using high-low range as proxy:

```python
qex("""
    let range = high() - low(),
        atr = ts_mean(range, 14)
    in rank(-atr) - 0.5
""")
```

Lower ATR = less volatile = rank higher.

## Stochastic Oscillator

```python
qex("""
    let price = close(),
        low_14 = ts_min(price, 14),
        high_14 = ts_max(price, 14),
        k = (price - low_14) / (high_14 - low_14)
    in rank(-k) - 0.5
""")
```

## Money Flow Index

Volume-weighted RSI:

```python
qex("""
    let price = close(),
        mf = price * volume(1),
        pos_mf = where(delta(price, 1) > 0, mf, 0),
        neg_mf = where(delta(price, 1) < 0, mf, 0),
        mfi = 100 * ts_sum(pos_mf, 14) / (ts_sum(pos_mf, 14) + ts_sum(neg_mf, 14))
    in rank(-mfi) - 0.5
""")
```

## On-Balance Volume Trend

```python
qex("""
    let obv_change = where(returns(1) > 0, volume(1), -volume(1)),
        obv_trend = ts_sum(obv_change, 20)
    in rank(obv_trend) - 0.5
""")
```

## Price Momentum Oscillator

```python
qex("""
    let roc1 = ewma(returns(1), 35),
        roc2 = ewma(roc1, 20),
        pmo = roc2 * 10
    in rank(pmo) - 0.5
""")
```

## Keltner Channel

```python
qex("""
    let ma = ewma(close(), 20),
        range = high() - low(),
        atr = ewma(range, 20),
        upper = ma + 2*atr,
        lower = ma - 2*atr,
        position = (close() - lower) / (upper - lower)
    in rank(-position) - 0.5
""")
```

## Williams %R

```python
qex("""
    let high_14 = ts_max(high(), 14),
        low_14 = ts_min(low(), 14),
        wr = (high_14 - close()) / (high_14 - low_14) * -100
    in rank(-wr) - 0.5
""")
```

## Commodity Channel Index (CCI)

```python
qex("""
    let tp = (high() + low() + close()) / 3,
        ma = ts_mean(tp, 20),
        md = ts_mean(abs(tp - ma), 20),
        cci = (tp - ma) / (0.015 * md)
    in rank(-cci) - 0.5
""")
```

## EWMA-Based Variants

More responsive versions using EWMA:

```python
# EWMA Bollinger
qex("""
    let price = close(),
        ma = ewma(price, 20),
        var = ewma_var(price, 20),
        std = sqrt(var),
        z = (price - ma) / (2 * std)
    in rank(-z) - 0.5
""")

# EWMA RSI
qex("""
    let up = ewma(max(delta(close(), 1), 0), 14),
        down = ewma(max(-delta(close(), 1), 0), 14),
        rsi = 100 * up / (up + down)
    in rank(-rsi) - 0.5
""")
```
