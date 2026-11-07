# Checking a Book's Beta

A signal made beta-neutral with `regression_neut` uses *estimated* betas; the book is exposed to *realized* ones, after
its caps and scaling. This checks how much market exposure is really left. Why it can be large, and how to fix it, is
in [Market Beta](../examples/market-beta.md).

```python
from qex import qex, forward_returns, BookSettings, build_book, hold_untradable

signal = qex("""
    let b = ts_beta_shrunk(returns(1), returns(1, 'market'), 252)
    in regression_neut(rank(-returns(5)), b)
""")

settings = BookSettings(delay=1, universe='member', truncation=0.05)
w = hold_untradable(build_book(signal, data, settings), data['close'])   # the book actually held
fwd = forward_returns(data['close'])
pnl = (w * fwd.reindex_like(w).fillna(0)).sum(axis=1)
mkt = forward_returns(data['market']).reindex(w.index)

pnl.corr(mkt)                                  # realized correlation with the market
pnl.cov(mkt) / mkt.var()                       # realized beta of the book

b = qex("ts_beta_shrunk(returns(1), returns(1, 'market'), 252)").evaluate(data)
(w * b.shift(1).reindex_like(w)).sum(axis=1)   # ex-ante beta, day by day
```

The data needs a `market` series ([Data Format](../data/format.md#date-only-fields-and-the-market)) and a `member`
universe field. To evaluate the signal on its own with the same universe the book uses:

```python
from qex import cross_section_universe

with cross_section_universe(data['member'].notna()):
    z = signal.evaluate(data)          # regression_neut fits over the members only
```
