# Diagnostics

Three tools say what an expression does with your data before you trust its numbers.

## Why a value is missing: `explain_missing`

`explain_missing(signal, data, universe=None)` traces each missing cell of an expression back to the node where it
began, and gives the reason (the reasons are listed in [Missing Values](../language/missing-values.md#why-a-value-is-missing)):

```python
from qex import qex, explain_missing

report = explain_missing(qex("ts_mean(rank(returns(1)), 10)"), data, universe=data['member'].notna())
report.summary(within=data['member'].notna())   # each reason's share of the missing member-days
report.on('2003-06-02')                          # that day's missing names, each with its reason and node
report.reason('2003-06-02', 'AAI')               # one cell: 'short history: outside the universe on earlier days'
```

Evaluate it in the universe the signal is used in: pass `universe` (a mask, a field name or a universe expression),
or call it inside `cross_section_universe`. Explaining changes nothing: it reads the values the expression's own
evaluation produced.

For a stock with no group, `report.reason(day, "H")` gives `'no group, at group_demean(field('value'), 'sector')'`.

::: qex.missing.explain_missing

::: qex.missing.MissingReport

## Adjusted data and look-ahead: `qex.adjust.check`

`check(signal, fields, mode="refuse")` follows each field's adjustment exponent through an expression (the rule is
in [Adjusted Data and Look-Ahead](../language/adjusted-data.md#the-rule-qex-checks)):

```python
from qex import qex
from qex.adjust import check

check(qex("rank(close())"), fields)              # raises LookAhead, saying what to write instead
check(qex("rank(close())"), fields, mode="warn") # returns the problems and lets it run
```

`fields` is the data's declarations, `{"close": {"adjustment": 1, "unit": "usd_per_share"}, ...}`; a panel's
manifest carries them. A field without a declaration is not checked, and neither is anything computed from it.
`mode="off"` skips the check. `alpha=False` checks an expression that is not itself an alpha (a regressor, a
universe), whose value need not be comparable across stocks.

::: qex.adjust.check

## Numbers and labels: `qex.kinds`

`qex()` already refuses an expression that uses labels where numbers are needed. `qex.kinds.numbers_only(signal)`
says why an expression can't be an alpha (it gives labels), or returns `None` when it can.
