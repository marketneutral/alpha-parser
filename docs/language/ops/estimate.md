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
`estimate('ols', 'beta1', y, x, window=252)` does not, and they are the same node. Write `estimate` for what has no
name: an estimator such as the stump, or an output such as one slope of a joint fit.

`name@version` picks a version (`estimate('ols@1', ...)`); the name alone is the latest. An unknown estimator, output or
option is refused when the expression is read, with the list of what there is.

## The operators are estimates too

Every fitted operator qex has is a built-in estimator, and both ways of writing it are **the same node**: the same
numbers, one cached computation, the same identity. **For readability, write the named operator** wherever the left
column has one; the right column is what it means, and the way to the outputs that have no name, such as one slope of a
joint fit.

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

An estimator that is not built from qex's own operators is written to a protocol. The engine owns the schedule and
hands the estimator only what was known: `fit` gets the rows of the window ending on the refit day, and `step` gets one
day's row, with the parameters fixed until the next refit. So an estimator cannot use a later day, even by mistake;
an HMM written this way can only give filtered probabilities, never the smoothed ones that leak the future. A
randomized fit (the stump's shuffles) draws from a generator seeded by the call and the refit's date, so one expression
always gives one number, whatever day the data starts on.

## Writing an estimator (Python)

An estimator is a class, registered once; an expression names it. Python never goes inside the string.

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

`qex.estimators()` lists what is registered. A name and version can be registered once; a change ships as a new
version beside the old one, so an expression that names `'mean_gap@1'` keeps its numbers. The protocol covers the
`series` axis in this version; the built-ins cover each stock's own series, the cross-section and the panel.

**Reference.** Hansen, B. E. (2000). Sample splitting and threshold estimation. *Econometrica* 68(3), 575-603.
