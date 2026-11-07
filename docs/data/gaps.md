# Halts, Delistings and Gaps

Prices go missing: before a listing, during a halt, after a delisting, on a vendor's bad day. qex treats every gap
the same way, on the signal side and in P&L.

## Returns across a gap

Every return in qex follows one rule (`price_returns`):

- a return is the ratio of two observed prices; **a day with no price has no return**, not 0%;
- a missing start price is replaced by the last price before it, so **the day a halted stock trades again carries
  the whole move** since its last price;
- a price more than **10 rows old** (`MAX_PRICE_GAP`) no longer counts, so a stock back after a long suspension gets
  no return for that day rather than one enormous one.

So a halted or delisted stock drops out of `rank` and the group operators instead of counting as a 0% day, and
`volatility` and `ts_beta` don't see fake quiet days. The rule gives the same answer on pandas 2 and 3.

```
close         100   101    —     —    104   105
returns(1)     —    1.0%   —     —    3.0%  1.0%      the reopen day carries 101 → 104
```

## Positions across a gap

A stock with no price can't be traded. `Backtest` holds the position from the stock's last priced day
(`hold_untradable`), marked at its last price, so P&L is 0 during a halt and the move across it is booked when the
stock trades again. A position is held for at most 10 days with no price; after that (a delisting, or a long
suspension) it is dropped with no P&L for the missing move.

**Delisting returns are not modelled.** A stock that delists after bad news usually loses part of its last price on
the way out; qex books nothing for it. That flatters a long book in distressed stocks. If your data has delisting
returns, apply them before running a backtest.

Held weights are not rescaled, so on a day a halted stock is held, the book's gross can be a little above or below 1.
`build_book` itself still returns the exact target book. See [Books](../python/books.md#untradable-names).
