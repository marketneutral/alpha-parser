# Syntax

A qex expression is one formula. It reads like arithmetic over named operators, and every part of it stands for a
value per date and stock (see [How qex Evaluates](concepts.md)).

```qex
rank(-returns(5) / volatility(20))
```

## Function calls

An operator is called by name with its arguments in parentheses, separated by commas. Arguments are positional;
there are no keyword arguments.

```qex
returns(20)                          # one argument
ts_corr(returns(1), volume(1), 60)   # several
close()                              # none
```

An argument is either an **expression** (anything on this page) or a **constant** the operator needs: a window length,
a number of buckets, a field or group name. The [operator reference](reference.md) gives each operator's arguments.

## Values

| Write | Means |
|-------|-------|
| `20`, `0.05`, `1e-3`, `-1` | a number, the same on every date and stock |
| `'sector'`, `"sector"` | a string: a field or group name, or a label to compare with (`group('sector') == 'Technology'`) |
| `None` | no value here, on purpose (see [Missing Values](missing-values.md)) |

A number in an expression is broadcast: `returns(5) - 0.01` subtracts 1% from every stock's return.

## Arithmetic

| Operator | Means |
|----------|-------|
| `a + b`, `a - b`, `a * b`, `a / b` | element-wise arithmetic |
| `-a` | negation |
| `a ** n` | `a` to the power `n`; `n` must be a plain number. For any other exponent write `power(a, b)` |

Division by zero gives an infinity, not an error; see [Missing Values](missing-values.md).

## Comparisons and logic

| Operator | Means |
|----------|-------|
| `a > b`, `a < b`, `a >= b`, `a <= b` | 1 or 0, and missing where either side is missing |
| `a == b`, `a != b` | the same; also compares labels |
| `a & b`, `a \| b` | and, or |
| `~a` | not |

`&`, `|` and `~` are three-valued: false & missing is false, true | missing is true, and anything else with a missing
side is missing. [Missing Values](missing-values.md#comparisons-and-logic-three-values) has the full table.

Comparisons can't be chained: write `(a < b) & (b < c)`, not `a < b < c`.

## Precedence

From tightest to loosest, as in ordinary arithmetic:

1. function calls and parentheses
2. `**`
3. `-a`, `~a`
4. `*`, `/`
5. `+`, `-`
6. `&`
7. `|`
8. `>`, `<`, `>=`, `<=`, `==`, `!=`

`&` and `|` bind tighter than comparisons, so **comparisons inside `&` and `|` need parentheses**:

```text
(returns(5) > 0) & (volume(1) > volume(20))    right
returns(5) > 0 & volume(1) > volume(20)        refused: it reads as a chained comparison
returns(5) > 0 & volume(1)                     no error, but reads as returns(5) > (0 & volume(1))
```

## Names: `let ... in`

`let` names a part of an expression so it can be used more than once:

```qex
let mom = returns(60),
    vol = volatility(60),
    sharpe = mom / vol
in rank(sharpe) * sign(mom)
```

- Bindings are separated by commas; a later binding can use an earlier one.
- A name is used only in the `in` body and in later bindings.
- A name must be a valid identifier (letters, digits, `_`, not starting with a digit) and can't be an operator's
  name: `let rank = ...` is refused.
- A named part is computed once, however many times it is used.

## Layout

Spaces and line breaks don't matter outside quoted strings, so an expression can be laid out across lines anywhere,
including after an operator:

```qex
let sharpe = returns(60) /
             volatility(60)
in rank(sharpe)
```

!!! warning "Don't use `#` comments in a multi-line expression"
    qex joins the lines of an expression before reading it, so a `#` comments out **everything after it, on every
    following line**. `rank(returns(5))  # reversal` followed by a line `* -1` reads as `rank(returns(5))`, with no
    error. A `#` comment is only safe on the last line. Explain an expression next to it, not inside it.

## What is refused

An expression that qex can't read, or that asks for something it doesn't do, is refused before anything is computed,
with a message:

| Written | Refused because |
|---------|-----------------|
| `foo(1)` | `Unknown function: foo` |
| `a + b` with no `let` for `b` | `Unknown variable: b` |
| `let rank = returns(1) in rank` | `Cannot shadow function name: rank` |
| `rank(returns(1), 2)` | the wrong number of arguments for the operator |
| `ts_mean(returns(1), period=20)` | keyword arguments aren't supported |
| `1 < returns(1) < 2` | `Chained comparisons not supported` |
| `returns(1) ** returns(2)` | `The exponent of ** must be a number, as in power(x, n)` |
| `rank(group('sector'))`, `group('sector') + 1` | labels where numbers are needed ([Groups and Labels](groups.md#numbers-and-labels)) |
| `group_std(x, 'pair', 60)` | the window form of `group_std` was removed in qex 0.12.0; the message gives the replacement |
| `and`, `or`, `not`, `%`, `//`, `[...]`, `x.y`, `if`/`else` | not part of qex (use `&`, `\|`, `~` for logic) |

Most argument checks happen when the expression is read: a negative `delay` or `delta`, a `ts_quantile` level outside
0 to 1, a `pca_factor` component beyond 20, a window of less than 1 for `group_count_valid`. A `winsorize` level
outside 0 to 1 is refused only when the expression is computed. A few values are not checked at all and give
meaningless results rather than an error: a window of 0 (`ts_mean(x, 0)`), `quantile(x, 0)`, a negative
`truncate` limit.
