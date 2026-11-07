# Risk Model

`FactorRiskModel` is a fundamental-style multi-factor risk model. Each day it regresses the stocks' returns on their
factor exposures, which are qex expressions, plus a market factor and, optionally, industry dummies. The regression
coefficients are that day's factor returns. From them it estimates a factor covariance matrix and each stock's
specific risk, and uses those to forecast a portfolio's risk and attribute it to the factors.

It needs the `risk` extra: `pip install "qex[risk]"` (statsmodels).

```python
from qex import FactorRiskModel, PRICE_ONLY_FACTORS
import pandas as pd

model = FactorRiskModel(style_factors=PRICE_ONLY_FACTORS, include_industries=True, industry_column='sector')
model.fit(data, lookback=252)                     # returns the model

model.factor_returns          # dates x factors: each day's factor returns
model.factor_covariance       # factors x factors, annualized
model.specific_risk           # each stock's annualized specific (idiosyncratic) volatility

weights = pd.Series({'AAPL': 0.1, 'MSFT': -0.1, 'JPM': 0.05, 'BAC': -0.05})
total, factor, specific = model.portfolio_risk(weights)     # annualized volatilities
model.factor_attribution(weights)                            # each factor's contribution to variance
```

## How it is estimated

- **Exposures** are the style factors' expressions, evaluated on the data, plus 1 for every stock on the `market`
  factor and a 0/1 dummy per industry when `include_industries` is set.
- **Point in time.** Day t's returns are regressed on the exposures known at the close of day t − 1, so a factor
  built from day t's own return (short-term reversal, volatility) can't explain that return with hindsight.
- **Returns** follow qex's rule for gaps ([Halts, Delistings and Gaps](../data/gaps.md)): a day with no price has no
  return and leaves that day's regression.
- **The last `lookback` days** are fitted, one cross-sectional regression a day, over the stocks with every exposure
  and a return; a day with fewer than `min_observations` stocks (default 10) is skipped.
- **Covariances** are exponentially weighted, with a `half_life` (default 63 days), and annualized (× 252). They are
  proper weighted covariances: with equal weights, the factor covariance is exactly that of the factor returns.
- **Forecasts** use the exposures at the close of the last fitted day, so `portfolio_risk` forecasts the risk of a
  book held from that day on.

## Factors

A factor is a `FactorDefinition(name, expression, description)`: any qex expression that gives each stock a
number each day. Two sets come with qex:

`DEFAULT_STYLE_FACTORS` (needs fundamental fields `market_cap` and `book_to_price`):

| Factor | Expression |
|--------|------------|
| `size` | `rank(field('market_cap'))` |
| `value` | `rank(winsorize(field('book_to_price'), 0.05))` |
| `momentum` | `rank(returns(252) - returns(21))` |
| `volatility` | `rank(volatility(60))` |
| `liquidity` | `rank(ts_mean(volume(1), 20))`: average volume in shares (its description says dollar volume; it is not) |
| `short_term_reversal` | `rank(-returns(5))` |

`PRICE_ONLY_FACTORS` (prices and volume only):

| Factor | Expression |
|--------|------------|
| `momentum` | `rank(returns(252) - returns(21))` |
| `volatility` | `rank(volatility(60))` |
| `liquidity` | `rank(ts_mean(volume(1), 20))` |
| `short_term_reversal` | `rank(-returns(5))` |
| `medium_term_reversal` | `rank(-returns(20))` |

Your own:

```python
from qex import FactorRiskModel, FactorDefinition

factors = [
    FactorDefinition('earnings_yield', "rank(field('earnings') / field('price'))", 'Earnings yield rank'),
    FactorDefinition('beta', "rank(ts_beta_shrunk(returns(1), returns(1, 'market'), 252))", 'Market beta rank'),
]
model = FactorRiskModel(style_factors=factors).fit(data)
```

`style_factors=None` uses `DEFAULT_STYLE_FACTORS`. A factor that can't be computed on the data (a missing field) is
skipped with a warning.

For a statistical model instead, with factors found from the returns themselves, see `pca_factor` in the
[Operator Reference](../language/reference.md#pca_factor).

## API reference

::: qex.risk.FactorRiskModel

::: qex.risk.FactorDefinition
