# Parsing

`qex(expression)` reads an expression and returns a `Signal`: the expression's tree, ready to evaluate. Nothing is
computed yet.

```python
from qex import qex

signal = qex("rank(-returns(5) / volatility(20))")
```

The expression is checked as it is read: an unknown operator, a misplaced label, a negative delay and the other
mistakes listed in [Syntax](../language/syntax.md#what-is-refused) raise `ValueError` or `TypeError` with a message,
before any data is touched.

Multi-line expressions are ordinary Python strings:

```python
signal = qex("""
    let mom = returns(60),
        vol = volatility(60)
    in rank(mom / vol)
""")
```

`#` comments run to the end of their line, so a multi-line expression can be commented line by line
([Syntax](../language/syntax.md#layout)).

`alpha()` is the same function under its older name, and `QexParser` is the class behind both.

## API reference

::: qex.parser.qex

::: qex.parser.QexParser
