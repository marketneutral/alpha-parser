# Logic and Conditions

Arithmetic, comparisons, logic, and the operators that choose between values or say whether there is one. The
rules for missing values are in [Missing Values](../missing-values.md); precedence is in
[Syntax](../syntax.md#precedence).

## Operators

| Operator | Description |
|----------|-------------|
| `+`, `-`, `*`, `/` | Arithmetic, element by element |
| `**` | Power, with a number as the exponent |
| `>`, `<`, `>=`, `<=`, `==`, `!=` | 1 or 0, missing where either side is missing |
| `&`, <code>&#124;</code>, `~` | And, or, not, three-valued |

### Comparisons

A comparison gives 1 or 0, and no value where either side has none: there are no returns on Jan 02.

```toy
returns(1)
returns(1) > 0
```

### And, or, not

Three-valued: false `&` missing is 0, true `|` missing is 1, and otherwise missing with missing is missing. On Jan 02,
**A**, **B** and **C** close at 30 or less, so the `&` is 0 without a return; **D** closes above 30 and has no answer.

```toy
returns(1) > 0
close() > 30
(returns(1) > 0) & (close() > 30)
```

`&` and `|` bind tighter than comparisons, so the parentheses are needed.

## Choosing values

### where

<p class="sig"><code>where(cond, if_true, if_false)</code></p>

`if_true` where `cond` is nonzero, `if_false` where it is 0, and no value where it is missing.

```toy
returns(1) > 0
where(returns(1) > 0, close(), None)
```

```qex
where(volatility(20) > 0.6, None, rank(-returns(5)))      # leave out the most volatile stocks
```

### coalesce

<p class="sig"><code>coalesce(x, y)</code></p>

`x` where it has a value, else `y`: a number, a label or an expression. It says what missing should mean.

```toy
returns(1)
coalesce(returns(1), 0)
```

```qex
coalesce(close() > 8, 0)                     # a missing comparison counts as false
coalesce(group('sector'), 'Other')           # a label for stocks without one
```

### None

<p class="sig"><code>None</code></p>

No value here, on purpose: `where(cond, x, None)`. `None` and `0` are different answers. A `0` is a value, and once
a book demeans the signal it is a position against the day's average; `None` takes the stock out.

## Is there a value?

### is_valid

<p class="sig"><code>is_valid(x)</code></p>

1 where `x` has a value, else 0. An infinity counts as a value.

```toy
close()
is_valid(close())
```

```qex
is_valid(field('surprise'))                  # 1 on report days
```

### is_finite

<p class="sig"><code>is_finite(x)</code></p>

1 where `x` is a finite number, 0 where it is missing or infinite.

```qex
where(is_finite(volume(20) / volume(1)), volume(20) / volume(1), None)
```
