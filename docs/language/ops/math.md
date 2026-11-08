# Math

Element by element: each date and stock on its own. Any argument can be a number, so `sqrt(5)`, `max(x, 0)` and
`power(2, x)` all work.

```toy
returns(1)
abs(returns(1))
```

### log

<p class="sig"><code>log(x)</code></p>

The natural logarithm. No value for `x` < 0, and −∞ at 0.

```qex
log(1 + returns(20))                        # a 20-day log return
log(field('mcap'))                          # size, on a log scale
```

### exp

<p class="sig"><code>exp(x)</code></p>

e to the `x`, the inverse of `log`; ∞ when too large.

```qex
exp(ts_sum(log(1 + returns(1)), 20)) - 1    # daily returns compounded over 20 days
1 / (1 + exp(-zscore(returns(5))))          # a sigmoid, from 0 to 1
scale(exp(zscore(returns(5)) / 0.5))        # a cross-sectional softmax, temperature 0.5
```

### abs

<p class="sig"><code>abs(x)</code></p>

The absolute value.

```qex
ts_mean(abs(returns(1)), 20)                # average daily move, either way
```

### sign

<p class="sig"><code>sign(x)</code></p>

−1, 0 or 1.

```qex
sign(returns(20)) * sqrt(abs(returns(20)))  # keeps the direction, damps the outliers
```

### sqrt

<p class="sig"><code>sqrt(x)</code></p>

The square root. No value for `x` < 0.

### power

<p class="sig"><code>power(x, n)</code></p>

`x` to the power `n`; `n` can be a number or an expression. `x ** 2` is the same with a number as the exponent.

```qex
power(volatility(63), 2)                    # variance, annualized
```

### max

<p class="sig"><code>max(x, y)</code></p>

The larger of `x` and `y`, element by element. Not the largest across stocks or over time: for those, use
[`ts_max`](time-series.md#ts_max) or [`rank`](cross-sectional.md#rank).

```qex
ts_mean(max(returns(1), 0), 14) / ts_mean(abs(returns(1)), 14)    # the up share of recent moves, as in RSI
```

### min

<p class="sig"><code>min(x, y)</code></p>

The smaller of `x` and `y`, element by element.

```qex
sqrt(ts_mean(power(min(returns(1), 0), 2), 60) * 252)              # downside deviation, annualized
min(max(returns(1), -0.1), 0.1)                                      # clip at ±10%
```
