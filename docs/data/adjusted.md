# Adjusted and As-Traded Prices

Price data usually comes in two forms. They answer different questions, and mixing them up lets the future into a
signal.

| | What it is | Use it for | Never use it for |
|---|---|---|---|
| **as traded** (`field('price')`, `field('volume_traded')`, `field('shares')`, `field('mcap')`) | the number printed or filed that day | dollar floors, price levels, market cap, dollars traded | returns: it jumps at every split and dividend |
| **adjusted** (`close`, `open`, `high`, `low`, `volume`) | the as-traded number rescaled for every split and dividend *after* that day | returns, volatility, a stock against its own past | comparing stocks, or comparing with a fixed number |

`close()`, `open()`, `high()`, `low()` and `volume(n)` read the adjusted fields. Name the as-traded ones however your
data does; the names above are a convention.

Why adjusted levels see the future, and the rule qex applies, is in
[Adjusted Data and Look-Ahead](../language/adjusted-data.md).

## Declaring each field

The check needs to know which fields are adjusted. Each field declares an **adjustment exponent** and, optionally, a
**unit**:

```python
fields = {
    "close":         {"adjustment": 1,  "unit": "usd_per_share"},   # adjusted price
    "volume":        {"adjustment": -1, "unit": "shares"},          # adjusted volume (as traded / factor)
    "price":         {"adjustment": 0,  "unit": "usd_per_share"},   # as traded
    "volume_traded": {"adjustment": 0,  "unit": "shares"},
    "shares":        {"adjustment": 0,  "unit": "shares"},
    "mcap":          {"adjustment": 0,  "unit": "usd"},
}
```

| Exponent | Means |
|----------|-------|
| `1` | an adjusted price (as traded × the factor) |
| `-1` | adjusted volume (as traded ÷ the factor) |
| `0` | as traded or as filed |

Units are `usd_per_share`, `usd`, `shares`, `count`, `ratio`, `percent` or `number`; a field without a unit is not
checked for units. A field with no declaration is not checked at all, and neither is anything computed from it. A
data panel's manifest is the natural home for these declarations; pass them to `qex.adjust.check`
([Diagnostics](../python/diagnostics.md#adjusted-data-and-look-ahead-qexadjustcheck)).
