# qex

<div class="hero" markdown>

## Express quant signals in one line

```python
signal = qex("rank(-returns(20) / volatility(60))")
```

[Get Started](getting-started/quickstart.md){ .md-button .md-button--primary }
[Operations Reference](reference/operations.md){ .md-button }

</div>

<div class="features" markdown>

<div class="feature" markdown>
### Readable
Signal logic is self-documenting. No more deciphering 50 lines of pandas.
</div>

<div class="feature" markdown>
### Composable
Build complex signals from simple primitives. Nest operations freely.
</div>

<div class="feature" markdown>
### Efficient
Built-in caching. Shared sub-expressions computed once.
</div>

<div class="feature" markdown>
### Flexible
Works with any DataFrame pipeline. Lazy loading for large datasets.
</div>

</div>

---

## What is qex?

Qex (Quant Expression) is a DSL for defining quantitative trading signals. Instead of writing verbose pandas code, express signals as readable one-liners:

```python
from qex import qex

# Sector-neutral momentum
signal = qex("group_demean(returns(60) / volatility(60), 'sector')")

# Mean reversion with volatility filter
signal = qex("where(volatility(20) > 0.3, -returns(5), 0)")

# Multi-factor composite with variable bindings
signal = qex("""
    let mom = rank(returns(60)) - 0.5,
        rev = rank(-returns(5)) - 0.5
    in 0.7 * mom + 0.3 * rev
""")
```

## Quick Example

```python
from qex import qex, Backtest
import pandas as pd

# Load your data
data = {
    'close': pd.read_parquet('close.parquet'),
    'volume': pd.read_parquet('volume.parquet'),
}

# Define a signal
signal = qex("rank(returns(20)) - 0.5")

# Backtest it
bt = Backtest(signal, transaction_cost=0.001)
result = bt.run(data)
print(result.summary())
```

## Operations at a Glance

| Category | Operations |
|----------|------------|
| **Primitives** | `returns`, `volatility`, `volume`, `adv` |
| **Time-Series** | `ts_mean`, `ts_std`, `delay`, `delta`, `ewma`, `ts_corr`, `ts_beta` |
| **Cross-Sectional** | `rank`, `zscore`, `demean`, `quantile`, `winsorize`, `scale` |
| **Group** | `group_rank`, `group_demean`, `group_std` |
| **Math** | `log`, `abs`, `sign`, `sqrt`, `power`, `max`, `min` |
| **Conditional** | `where`, `is_valid`, `fill_forward` |

[See all operations →](reference/operations.md)

## Installation

```bash
pip install qex
```

Or from source:

```bash
git clone https://github.com/marketneutral/alpha-parser
cd alpha-parser
pip install -e ".[all]"
```
