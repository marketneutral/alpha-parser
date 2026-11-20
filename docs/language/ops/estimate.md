# Estimates

Some values are *fitted*: a beta, a factor loading, the residual of a regression. Each comes from an **estimator** that
fits something on a window of days, refits on a schedule, and gives one or more **outputs**. `estimate` reaches any of
them by name, with one vocabulary for the window and the schedule.

### estimate

<p class="sig"><code>estimate(name, output, inputs..., window=, refit=, options...)</code></p>

The `output` of the estimator `name`, fitted on the inputs: a value per date and stock like any other, so it can be
ranked, added, bound in a `let`. The name and the output come first, quoted; then the inputs, each an expression; then,
by name, `window=` (days of history in each fit), `refit=` (how often it is fitted again, in weekdays on a fixed
calendar, as `ts_match` and `pca_factor` do) and the estimator's own options.

```qex
estimate('ols', 'beta1', returns(1), returns(1, 'market'), window=252)       # a market beta: ts_beta(..., 252)
estimate('pca', 'beta2', returns(1), window=252, refit=21)                 # the second statistical factor
estimate('ols', 'beta2', returns(1), returns(1, 'market'), returns(1, 'smb'), window=252)   # a joint fit's 2nd slope
```

Calls with the same estimator, inputs, window, refit and options share one fit, whatever output each reads, so asking
for three outputs of one fit costs one fit.

**Prefer the named operator where there is one.** `ts_beta(y, x, 252)` says what it is in a way
`estimate('ols', 'beta1', y, x, window=252)` does not, and it is that estimate, written short. Write `estimate` for what
has no name: an estimator such as the stump, or an output such as one slope of a joint fit.

`name@version` picks a version (`estimate('ols@1', ...)`); the name alone is the latest. An unknown estimator, output or
option is refused when the expression is read, with the list of what there is.

## The operators are estimates

Every fitted operator in qex is **sugar for `estimate`**: `ts_beta(y, x, 252)` calls
`estimate('ols', 'beta1', y, x, window=252)`, and the maths is the `ols` estimator's, written once. Both spellings are
one node: the same numbers, one cached computation, the same identity (each keeps the cache key it always had, so an
alpha's identity does not change). And every output of one fit reads that fit: `ts_beta`, `ts_alpha` and `ts_resid` on
the same `y`, `x` and window are one regression, as are `regression_neut`, `regression_alpha` and `regression_beta` on
the same inputs, and every `pca_factor` on one `x`, window and refit is one decomposition.

**For readability, write the named operator** wherever the left column has one; the right column is what it means,
and the way to the outputs that have no name, such as one slope of a joint fit.

| Named operator | As `estimate` |
|---|---|
| `ts_beta(y, x, 252)` | `estimate('ols', 'beta1', y, x, window=252)` |
| `ts_alpha(y, x1, x2, 252)` | `estimate('ols', 'alpha', y, x1, x2, window=252)` |
| `ts_resid(y, x, 252, 60)` | `estimate('ols', 'resid', y, x, window=252, min_periods=60)` |
| (none: a joint fit's j-th slope) | `estimate('ols', 'beta2', y, x1, x2, window=252)` |
| `ts_beta_shrunk(y, x, 252, 'sector')` | `estimate('ols_shrunk', 'beta1', y, x, window=252, groups='sector')` |
| `ts_alpha_shrunk`, `ts_resid_shrunk` | `estimate('ols_shrunk', 'alpha' / 'resid', ...)` |
| `regression_neut(y, b1, b2)` | `estimate('xs_ols', 'resid', y, b1, b2)` |
| `regression_alpha(y, b1)` | `estimate('xs_ols', 'alpha', y, b1)` |
| `regression_beta(y, b1, b2, 2)` | `estimate('xs_ols', 'beta2', y, b1, b2)` |
| `pca_factor(x, 252, 2, 21)` | `estimate('pca', 'beta2', x, window=252, refit=21)` |

| Estimator | Fits | Inputs | Outputs | `window` / `refit` | Options |
|---|---|---|---|---|---|
| `ols` | each stock's rolling least squares | y, then one or more regressors | `alpha`, `resid`, `beta1` ... `betak` | required / 1 | `min_periods` |
| `ols_shrunk` | the same, the slope shrunk toward the day's (or its group's) mean | y, x | `alpha`, `resid`, `beta1` | required / 1 | `min_periods`, `groups` |
| `xs_ols` | each day's least squares across stocks | y, then one or more regressors | `resid`, `alpha`, `beta1` ... `betak` | 1 / 1 | |
| `pca` | principal components across stocks | x | `beta1` ... `beta20` | required / 21 | |
| `stump` | a learned switch on a market state (below) | state, r1, r2 | `w1`, `threshold`, `gap`, `p` | required / 21 | `min_leaf`, `quantiles`, `permutations` |

`ts_match` stays its own operator: it reads *another* stock's value through the twins it chose, which is not an
output of a fit.

## The stump: a learned switch between two signals

A switching model in its smallest learned form: one market state, one threshold, and on each side the signal that has
done better there. It is threshold regression (Hansen 2000), a decision stump:

```qex
let v   = sqrt(ewma(power(returns(1, 'market'), 2), 11)),          # the state: market volatility
    mom = zscore(rank(delay(returns(252), 20))),
    rev = zscore(rank(-returns(5))),
    pm  = regression_beta(returns(1), delay(mom, 1)),               # each signal's daily return
    pr  = regression_beta(returns(1), delay(rev, 1)),
    w   = estimate('stump', 'w1', v, pm, pr, window=1260, refit=63)
in w * mom + (1 - w) * rev
```

**What it fits.** On each refit day, over the window, it pairs the state on day *u* − 1 with the two signals' returns
on day *u*, and tries thresholds at the state's quantiles. For each, the days split into those at or below it and
those above; the threshold kept is the one that best separates how `r1` did against `r2` on the two sides (the largest
cut in the squared error of `r1 − r2` around each side's mean: least squares with one break). Each side keeps the
signal that earned more there.

| Output | Is |
|---|---|
| `w1` | 1 when today's state is on the side where `r1` earned more, else 0: the weight on the first signal |
| `threshold` | the threshold in force, in the state's units; watch it drift |
| `gap` | `r1`'s edge over `r2` above the threshold, less its edge below: a day's return |
| `p` | a permutation test inside the window: how often a split on the state, shuffled, separates as well |

| Option | Default | |
|---|---|---|
| `min_leaf` | a fifth of the window | the fewest paired days on each side |
| `quantiles` | 20 | the thresholds tried: the state's quantiles at 1/20, 2/20, ... |
| `permutations` | 200 | the shuffles behind `p` (0: no `p`) |

**One series.** The stump fits market-level series: each input must be the same for every stock on a day
(`returns(1, 'market')`, an `ewma` of it, a `regression_beta`). Its outputs are given to every stock. A stock's own
series is refused, with a message.

**Small samples.** Regimes are rare, and a side that learned from two crises learned two anecdotes: hence `min_leaf`,
and `p` to say whether the switch is real. Fix the state for an economic reason and let the stump learn only where to
cut it; trying state after state is a search, and each is a trial.

## Point in time, by protocol

An estimator can be written to a protocol (or vectorized, as the built-ins are: below). The engine owns the schedule and
hands the estimator only what was known: `fit` gets the rows of the window ending on the refit day, and `step` gets one
day's row, with the parameters fixed until the next refit. So an estimator cannot use a later day, even by mistake;
an HMM written this way can only give filtered probabilities, never the smoothed ones that leak the future. A
randomized fit (the stump's shuffles) draws from a generator seeded by the call and the refit's date (and, per stock,
the stock's name), so one expression always gives one number, whatever day the data starts on.

What `fit` and `step` are handed follows the estimator's **axis**:

| Axis | `fit(window, ...)` gets | `step(state, row, ...)` gets | Each output |
|---|---|---|---|
| `series` | rows × inputs, market-level (the same for every stock) | that day's inputs | one number, for every stock |
| `stock` | one stock's rows × inputs, called for each stock | that stock's inputs that day | the stock's number |
| `cross_section` | rows × stocks × inputs, and `members` (rows × stocks: in the universe, with every input) | the day's stocks × inputs, and that day's `members` | a number per stock, none outside the universe |
| `panel` | the same as `cross_section`, over a window of more than a day | the same | the same |

The window and the rows are read-only. A per-stock estimator may also define `fit_many(window, hyper, rngs)` and
`step_many(states, row, hyper)`, the same for every stock at once (the window stocks × rows × inputs, the row stocks ×
inputs), to fit in one vectorized step instead of a loop over stocks; by default they call `fit` and `step` stock by
stock.

## Writing an estimator (Python)

An estimator is a class, registered once; an expression names it. Python never goes inside the string. It implements
one of two things:

- **`compute(values, window, refit, hyper, data)`**: every output at once, vectorized over the panel. This is how the
  built-ins are written (rolling sums for `ols`, a least squares per day for `xs_ols`, an SVD per refit for `pca`).
  Each output may be a frame, or a function `get -> frame` computed when first read, where `get('name')` gives
  another output; so a fit read only for its slope never computes its residual. Fast, and the estimator is
  responsible for using only the past.
- **`fit` and `step`**: the protocol above. The engine owns the schedule and hands over only what was known, so the
  estimator cannot look ahead. The way to write a model whose maths qex does not have.

A protocol estimator, market-level:

```python
import numpy as np
import qex

class MeanGap(qex.Estimator):
    """How far the market's return today is from its mean over the window, in the window's standard deviations."""
    name, version, axis = "mean_gap", "1", "series"
    n_inputs, outputs = (1, 1), ("z",)
    window, refit, refit_default = "required", "optional", 21

    def fit(self, window, hyper, rng):             # rows x inputs: the window ending on the refit day
        x = window[:, 0][np.isfinite(window[:, 0])]
        return None if len(x) < 20 else (x.mean(), x.std(ddof=1))

    def step(self, state, row, hyper):             # one day's row, the fit in force
        mean, sd = state
        return {"z": (row[0] - mean) / sd}

qex.register_estimator(MeanGap)
qex.qex("estimate('mean_gap', 'z', returns(1, 'market'), window=252)")
```

Per stock, the same idea against each stock's own history, and across stocks against the day's universe:

```python
class OwnGap(qex.Estimator):
    """Each stock's value today against its own mean over the window."""
    name, axis, n_inputs, outputs = "own_gap", "stock", (1, 1), ("gap",)

    def fit(self, window, hyper, rng):             # one stock's rows x inputs
        x = window[:, 0][np.isfinite(window[:, 0])]
        return None if len(x) < 20 else x.mean()

    def step(self, state, row, hyper):             # that stock's inputs today
        return {"gap": row[0] - state}

class MedianGap(qex.Estimator):
    """Each stock's value today against the median of the universe's values over the window."""
    name, axis, n_inputs, outputs = "median_gap", "cross_section", (1, 1), ("gap",)

    def fit(self, window, hyper, rng, members=None):     # rows x stocks x inputs; members: rows x stocks
        v = window[:, :, 0][members]
        return None if v.size < 5 else float(np.median(v))

    def step(self, state, row, hyper, members=None):     # the day's stocks x inputs
        return {"gap": row[:, 0] - state}
```

`qex.estimators()` lists what is registered. A name and version can be registered once; a change ships as a new
version beside the old one, so an expression that names `'mean_gap@1'` keeps its numbers.

**Reference.** Hansen, B. E. (2000). Sample splitting and threshold estimation. *Econometrica* 68(3), 575-603.
