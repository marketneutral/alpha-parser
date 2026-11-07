# Universes from Python

What a universe is, and how to write one as an expression, is in [Universes](../language/universes.md). From Python,
everywhere qex takes a universe (`cross_section_universe`, `BookSettings(universe=...)`,
`explain_missing(universe=...)`) it takes any of three forms:

| Form | Example | Member when |
|------|---------|-------------|
| a mask | `data['member'].notna()` | True |
| a data field's name | `'member'` | the field has a value (1 in, NaN out) |
| a universe expression | `"field('mcap') >= 300e6"`, or a parsed `qex(...)` | the expression is nonzero |

Text that names a field of the data is that field (`'member'` stays the field it always was); anything else is
parsed as qex.

```python
from qex import qex, cross_section_universe, universe_mask
from evaluation import BookSettings, build_book

LARGE = "(close() >= 2) & (field('price') * field('shares') >= 2e9)"

with cross_section_universe(LARGE, data):                       # an expression needs the data
    r = qex("rank(returns(5))").evaluate(data)                  # ranked among the large stocks only

book = build_book(qex("rank(-returns(5))"), data, BookSettings(delay=1, universe=LARGE))
members = universe_mask(LARGE, data)                            # the True/False frame itself
```

Inside one `compute_context`, a universe expression's membership is computed once and shared by every signal
evaluated in it. `BookSettings(mask_cross_section=False)` applies the universe only to the finished signal, as qex did
before 0.5.

::: qex.universe.cross_section_universe

::: qex.universe.universe_mask
