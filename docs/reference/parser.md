# Parser & Syntax

The qex parser and expression syntax.

## Basic Usage

```python
from qex import qex

signal = qex("rank(returns(20))")
```

The `qex()` function parses a string expression into a Signal object.

## Backwards Compatibility

`alpha()` is an alias for `qex()`:

```python
from qex import alpha

signal = alpha("rank(returns(20))")  # Same as qex()
```

## Expression Syntax

### Function Calls

```python
qex("returns(20)")                        # One argument
qex("ts_corr(returns(1), volume(1), 60)")  # Multiple arguments
qex("close()")                            # No arguments
```

### Arithmetic

```python
qex("returns(20) / volatility(60)")
qex("-returns(5)")           # Unary minus
qex("returns(20) ** 2")      # Power, as power(returns(20), 2); the exponent must be a number
```

### Comparisons

```python
qex("returns(5) > 0")        # Returns 0 or 1
qex("volume(1) >= volume(20)")
```

### Logical Operations

```python
qex("(returns(5) > 0) & (volume(1) > volume(20))")  # AND
qex("(returns(5) > 0.1) | (returns(5) < -0.1)")  # OR
```

### Nested Expressions

```python
qex("rank(ts_mean(returns(1), 20) / ts_std(returns(1), 20))")
```

## Variable Bindings

### Single Binding

```python
qex("let s = returns(20) in rank(delta(s, 10)) * s")
```

### Multiple Bindings

Comma-separated, later bindings can reference earlier ones:

```python
qex("""
    let mom = returns(60),
        vol = volatility(60),
        sharpe = mom / vol
    in rank(sharpe) * sign(mom)
""")
```

### Rules

1. Variable names must be valid Python identifiers
2. Cannot shadow function names (e.g., `let rank = ...` is an error)
3. Variables are only in scope within the `in` body
4. Each binding expression is parsed independently

## String Arguments

For field names and group names, use quotes:

```python
qex("field('earnings_surprise')")
qex("group_demean(returns(5), 'sector')")
```

## Multi-line Expressions

Whitespace is insignificant outside quoted strings, so an expression can break across lines
anywhere, including after an operator. Use Python string concatenation or triple quotes:

```python
# String concatenation
qex(
    "let mom = returns(60), "
    "    vol = volatility(60) "
    "in rank(mom / vol)"
)

# Triple quotes
qex("""
    let mom = returns(60),
        vol = volatility(60)
    in rank(mom / vol)
""")

# A break after an operator is fine
qex("""
    let sharpe = returns(60) /
                 volatility(60)
    in rank(sharpe)
""")
```

## API Reference

::: qex.parser.qex

::: qex.parser.QexParser
