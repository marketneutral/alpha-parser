# Changelog

## 0.23.0

**`qex.complexity(signal, data)`: realized dof.** With data, the charge for an estimator is the dof its fits actually
used, averaged over the refits, where it measures them (`Estimator.realized_dof(state, fit)`); without data, or for an
estimator that does not measure them, the declared dof as before. The tree measures them: the splits it made and each
leaf's commitment |2w - 1|, so a regularized tree is charged less (on noise, `alpha` about 1 instead of 10).

**`tree`: the stump, grown to a depth, on several market states.**
`estimate('tree', 'w1', state1, state2, ..., r1, r2, window=, refit=, depth=2)` grows a decision tree greedily by the
stump's rule (at each node, over every state and its quantiles, the split that most reduces the squared error of
r1 - r2 on each side), each split fitted only on the days the one above sent to its side. Each leaf holds the signal
that earned more there. Outputs `w1`, `leaf` (which leaf today is in) and `p` (a permutation test of the whole tree).
Regularizers, each off by default: `alpha` (a split only if its own permutation test passes), `soft` (a leaf's weight is its confidence, Phi(t)), `honest` (splits on alternate months, leaves judged on the others); measured on made-up data in `ops/estimate.md`. Options `depth` (1 to 3), `min_leaf` (the fewest days on each side of a split; by default a fifth of the window,
halved per level; given, the same at every level), `quantiles`, `permutations`. With one state and depth 1 it is the
stump, exactly (tested). dof per fit: each split's threshold, and its choice of state when there are several, and
each leaf's side: depth 2 on two states is 10. Point in time, start-independent and repeatable (tested).
`ops/estimate.md` draws a tree learned on US stocks, 2011-2015: momentum when calm, reversal when the market is
nervous.

## 0.22.1

- `qex.estimators()` listed nothing until an estimator had been used: it now registers the built-ins first, and its table gains each estimator's `dof`.

## 0.22.0

**`stump_xs`: a learned split of the stocks between two signals, on each stock's own state.** The stump learns
*when*, on one market state; `stump_xs` learns *which stocks*: "when a stock is in the top 10% by x, rank it by y,
otherwise by z", with the 10% learned. `estimate('stump_xs', 'w1', state, y, z, returns(1), window=, refit=)` pools
every (stock, day) of the window (state, y and z on day u - 1, each signal z-scored across the day's stocks, and the
return on day u), and finds the threshold on the state that best separates where y beat z (least squares with one
break on d = (y - z) x return). Outputs `w1` (per stock), `threshold`, `gap` (the average day's gap) and `p`: a
Fama-MacBeth t-test on the days' gaps, Bonferroni over the thresholds tried, since a day's stocks are not independent
draws (a sampled permutation test was tried first and gave p from 0.01 to 0.88 on nearly the same window). Options
`min_share` (default 0.05), `quantiles` (20). Written to the protocol (`panel` axis): point in time and independent of
where the data starts, tested; deterministic; dof 3. On the US 3000, five years, about half a second a fit.

## 0.21.0

**`qex.complexity(signal)`: what a complexity charge should count.** qexarena and alpha_lab charged for every node of
an expression's SignalGraph, internal nodes included, so the charge moved when qex's internals did (0.20.0 put a fit
node above `ts_beta` and took the hidden slope from under `ts_resid`) though no number moved. `complexity` returns:

- `operators`: the distinct operations written (a shared subexpression once), not the internal nodes (an estimator's
  fit, `ts_match`'s selection). `ts_beta`, `ts_resid` and `pca_factor` count one each, and
  `estimate('ols', 'beta1', ...)` is the same one as `ts_beta`.
- `dof`: the degrees of freedom the expression's estimators learn by searching the data, per distinct fit: the stump's
  threshold and two sides (3); a beta, residual or component, none. **Every estimator now declares `dof`**
  (`Estimator.dof`, or `dof_of(fit)` when it depends on the options); registering one without it is refused.

`Complexity.charge(per_operator, per_dof)` prices them. `SignalGraph.stats()` gains `operators` and `dof`.
Documented in `ops/estimate.md` (What an estimate costs).

## 0.20.0

**`estimate(name, output, inputs..., window=, refit=, options...)`: every fitted value through one operator** (new
family *Estimates*, `ops/estimate.md`). qex had built the same thing four times (rolling least squares, its shrunk
form, the daily cross-sectional fit, the principal components): each fits on a window, refits on a schedule and is
read by small operators. An *estimator* is that pattern with a name, registered in Python and called by name.

- **The fitted operators are sugar for `estimate`.** Their maths moved, unchanged, into four built-in estimators
  (`qex/fits.py`): `ols` (`ts_beta`, `ts_alpha`, `ts_resid`), `ols_shrunk` (the shrunk fits, the group prior as
  `groups=`), `xs_ols` (`regression_neut`, `regression_alpha`, `regression_beta`) and `pca` (`pca_factor`). Each
  named operator now calls `estimate`: `ts_beta(y, x, 252)` is `estimate('ols', 'beta1', y, x, window=252)`. Their
  classes (`TsBeta`, `RegressionNeut`, `PcaFactor`, ...) remain as views of one output with no maths of their own,
  so `isinstance` checks, labels and the DAG's names still work.
- **Nothing moves.** Every output keeps its operator's cache key, so an expression's identity is what it was
  (qexarena's `expr_hash`, cached results). Checked against 0.19.0: 27 expressions over the four estimators, with and
  without a universe, give identical cache keys and identical values bit for bit; `explain_missing` gives identical
  reasons, and the look-ahead check identical refusals; the cache keys of all 894 distinct expressions in alpha_lab's
  runs and of every qexarena submission are unchanged; qexarena's sealed-alpha gate passes; and the speed is the
  same.
- **What does change: sharing.** Every output of one fit reads it. `regression_neut` now shares its fit with
  `regression_alpha` and `regression_beta` on the same inputs, and `ts_beta` with `ts_alpha` and `ts_resid`: the
  internal fit node is an `EstimateFit` (keyed `TsFit`, `RegressionFit` or `PcaBasis`, as before), so the DAG shows
  one more node above a `ts_beta` or a `regression_neut`. A fit's outputs are computed when first read, so a fit read
  only for its slope never computes its residual. The kind check's message now names the operator written
  (`ts_alpha() needs numbers`), not the internal fit.
- **For readability, prefer the named operator wherever one exists**; `estimate` is for what has no name.
- **One output that had no name:** `estimate('ols', 'betaj', y, x1, ..., xk, window=)`, the j-th slope of a joint
  rolling fit (ts_alpha and ts_resid used it; nothing read it).
- **A new estimator, `stump`**: a learned switch between two signals on a market state, threshold regression (Hansen
  2000). Outputs `w1` (the weight on the first signal), `threshold`, `gap` and `p` (a permutation test inside the
  window); options `min_leaf`, `quantiles`, `permutations`.
- **Two ways to write an estimator:** `compute`, every output at once over the panel, vectorized (how the
  built-ins are written), or **the protocol**, for a model whose maths qex does not have: `fit` gets only the rows of the window ending on the
  refit day and `step` one day's row, the schedule is the engine's (on the calendar, `schedule.py`), and randomness is
  seeded by the call and the refit's date. Point in time by construction, deterministic, and independent of where the
  data starts. Every axis: `series` (market-level inputs, each output given to every stock), `stock` (`fit` and
  `step` per stock, with optional vectorized `fit_many` and `step_many`), and `cross_section` and `panel` (`fit` gets
  the window's rows x stocks x inputs and the universe's members, `step` the day's cross-section; no value outside the
  universe). The three properties are tested on the stump and on a per-stock and a cross-sectional estimator.
- **The registry:** `qex.Estimator`, `qex.register_estimator`, `qex.estimators()`, `qex.get_estimator`;
  `'name@version'` picks a version; a name and version register once. `explain_missing` names estimate nodes as
  written and calls the days before a first fit a short history; the look-ahead and kind checks cover them.
- `estimate` takes its options by name (any the estimator declares; unknown ones are refused with the list).

Tests in `tests/test_estimate.py`.

## 0.19.0

**Refits on a calendar: `ts_match` and `pca_factor` no longer depend on where the data starts.** Both re-fit on a
schedule (`rebalance=`, `refit=`, default 21), which was counted in rows from the data's first row. So one expression
was different books on different data: evaluated on data loaded from 2020-01-01 and from 2020-01-20, a `ts_match` chose
the same twin for only 23% of stocks on 2021-12-31 (the US 3000, `metric='path'`, within sector, exclusive). Nothing
looked ahead, but a backtest's start date changed every number after it, and a validation, a test and a live record
that load from different dates scored different twins.

Now the refit days come from a fixed calendar (`qex/schedule.py`): the weekdays counted from Monday 1970-01-05, cut into
blocks of `rebalance` (or `refit`) weekdays, a refit on the first trading day of each block (a holiday moves it to the
next trading day), the first one with a full window before it. The same day has the same refit and, given the window's
history, the same fit, whatever row the data begins on: on the case above, the same twin for 100% of stocks, and the
same `pca_factor` betas. An index that is not dates keeps the old rule.

What changes for an existing expression:

- **The numbers move.** Refits fall on different days, so twins and factors differ from 0.18's on most days. Sealed
  or published results that use `ts_match`, `ts_match_score` or `pca_factor` should be rescored.
- **Blocks are weekdays, not rows**: `rebalance=21` is 21 weekdays, about 20 trading days in a year with holidays.
- **The first fit can come later**: the first block start with a full window, up to `rebalance - 1` days after the old
  first fit, with no value before it.
- **`pca_factor` still follows its factors' identity from the first refit in the data** (matched to the previous refit,
  signed to agree), so a later start can number or sign the factors differently, though each refit's decomposition is
  now the same. The docs say so.

Tests in `tests/test_refit_anchor.py`; the tests that hard-coded the old schedule read it from `refit_rows`. Docs
(`ops/matching.md`, `ops/cross-sectional.md`, the reference) and the shipped reference regenerated.

## 0.18.0

`days_since(x)`, in Events and state: trading days since `x` was last nonzero, 0 on the day, 1 the day after. A day
counts when `x` has a value that is not 0 (a negative value or an infinity counts); a missing day neither resets nor
stops the count; before the first event it is missing; there is no window. It answers "days since the latest event"
(an earnings release, a 52-week high), which `ts_argmax` does not: given two events in its window, `ts_argmax` counts
back to the older one (its entry now says so). Unit-free for the adjustment check, like `ts_count_events`. Tests in
`tests/test_days_since.py`; the shipped reference is regenerated. No change to any other operator.

## 0.17.0

The language reference ships in the package: `qex/docs/OPERATORS.md` lists every operator, one line each, each line
linked to the operator's own entry in `qex/docs/ops/<family>.md`, with the other language pages beside them (syntax,
concepts, missing values, groups, adjusted data, universes). Plain Markdown, readable in an editor, on GitHub, or by an
agent that has only the installed package: the worked examples are evaluated into tables, and links to the Python
pages point to the site. `scripts/build_agent_docs.py` writes it from `docs/language/`, and `tests/test_agent_docs.py`
checks it is current, that every operator has a linked line, and that every link resolves. The language only: no
Python, and no book settings. No change to any operator.

## 0.16.1

Messages name `gaussrank` and `group_gaussrank` as they are written (they said `gauss_rank()`). Both are
cross-sectional, so the adjustment check refuses them on an adjusted level (`gaussrank(close())`), as it does `rank`;
tests added.

## 0.16.0

- **`gaussrank(x)`**: the day's rank mapped through the inverse normal, Φ⁻¹((r − 0.5)/n), exact (scipy's `ndtri`). A
  z-score's shape without its outliers: robust like `rank` (only the order of x matters), on `zscore`'s scale, with a
  normal sample's tails. Tied stocks share the average of the scores they occupy, so each day sums to zero exactly.
  About twice `rank`'s cost (0.7 s on 3,000 stocks over five years); days with ties are averaged separately.
- **`group_gaussrank(x, groups)`**: the same within each group; a stock alone in its group gets 0.
- **`ts_match`'s `'path'` and `'coint'` check that their key is daily returns**, which they compound into price
  paths. `returns(n)` with n above 1 is refused when the expression is read; a key with a value below -1, or whose
  typical size (median absolute value) is above 0.25, when it is evaluated. That refuses ranks, z-scores, prices and
  volumes, and accepts any daily-return-like key: `returns(1)`, excess returns, `ts_resid` residuals, PCA residuals.
  The look-ahead check now says plainly that the key is daily returns, not a price.

## 0.15.0

`ts_match`, each stock's statistical twin, and the first keyword arguments in qex. Existing expressions are unchanged.

- **`ts_match(value, key, lookback, metric='corr', within=None, matching='nearest', rebalance=21,
  min_overlap=None)`**: for each stock, the stock whose `key` over the last `lookback` days was most alike, and that
  twin's `value`. Twins are chosen every `rebalance` days from the days before (point in time), among stocks in the
  universe and, with `within`, in the same group; never the stock itself; held in between, and ended when either
  stock leaves the universe. Metrics: `'corr'` (who moves with the stock) and `'euclidean'` (who is closest on the
  key, such as z-scored fundamentals), each over the pair's shared days; `'path'`, Gatev, Goetzmann and
  Rouwenhorst's distance between price paths compounded from 100 at each window's start; and `'coint'`, Engle-Granger:
  the Dickey-Fuller t of the pair's spread, so pairs are chosen by evidence that they revert to each other (pairs
  with a non-positive hedge ratio are not considered). For `'path'` and `'coint'` the key is daily returns. In a test
  against decoys that correlate more closely day to day but drift away, `'coint'` found every true partner and
  `'corr'` none. `'exclusive'` pairs the group off into mutual twins. Designed and tested as a
  mock in qexarena's notebook "ts_match: mocking up a new operator in Python"; the semantics tests are ported.
- **The selection is one internal node**, `MatchSelection`, shared by every `ts_match` and `ts_match_score` on the
  same key and options whatever its value: a twin's return, volume, identifier and score cost one selection.
- **`ts_match_score(key, lookback, ...)`**: how alike each stock and its twin were when chosen, `ts_match`'s
  arguments without the value. It reads the same selection, so it costs nothing more: the correlation, the distance,
  or for `'coint'` the Dickey-Fuller t, or with `stat=` the pair's hedge ratio, half-life or spread width; held
  between rebalances, none without a twin.
- **`identifier()`**: each stock's own identifier, as a label. `ts_match(identifier(), key, 252)` is each stock's
  twin's identifier, the way to see and check the twins; `ts_match` gathers labels as labels.
- **Keyword arguments**, for `ts_match` and `ts_match_score` only: their options can be given by name, in any order. Every other operator
  still takes its arguments by position. Writing an option at its default is the same expression (and cache key) as
  leaving it out.
- A missing cell with no twin is reported as "no twin"; a missing twin value, with the twin's own reason.

## 0.14.0

The coefficients of the cross-sectional regression, and the graph and caching fixes found while testing many signals
together. Existing expressions keep their values and cache keys.

- **`regression_alpha(y, x1, ..., xk)`** and **`regression_beta(y, x1, ..., xk, j)`**: the intercept and the j-th
  slope of `regression_neut`'s daily fit, one number per day given to every stock. On today's returns and yesterday's
  characteristics they are the Fama-MacBeth intercept and factor returns: factor momentum, volatility-managed blends,
  data-weighted characteristics and beta timing are on the new Factor Premia and Timing page. A day with fewer than
  k + 2 names has no value, reported as "too few names".
- **Shared fits.** `regression_alpha` and every `regression_beta` on the same inputs read one internal
  `RegressionFit` node; `ts_alpha`, `ts_resid` and their shrunk versions read one `TsFit` node, whose slope, with one
  regressor, is the `ts_beta` (or `ts_beta_shrunk`) node itself. Each fit is computed once, in a compute_context, in
  SignalGraph and on Dask, and the DAG shows it shared.
- **The DAG connects every operator to its inputs.** `SignalGraph` found inputs by a fixed list of attribute names
  and missed the pair operators' (`ts_corr`, `ts_cov`, `ts_beta`, `ts_beta_shrunk`, `ewma_cov`, `ts_beta_ewma`),
  `latch`'s conditions and `power`'s exponent: the graph drew them without inputs, and on Dask their inputs were
  recomputed inside the node. It now takes every attribute that is a signal.
- **A compute context belongs to its thread.** It was one module global: threads entering and leaving contexts at
  overlapping times (Dask's threaded workers) could leave one context active for the whole process, so later
  evaluations read a stale cache, and a node's inputs could land in another thread's cache and be computed again.
  It is now a `contextvars.ContextVar`.
- `tests/test_sharing.py` puts six overlapping signals (two statistical factors, a cross-sectional fit read for its
  intercept and a slope, a sector-prior shrunk beta used three ways, a two-regressor time-series fit) through one
  compute context, `SignalGraph.compute` and Dask, and checks every node is computed exactly once and every result
  equals the signal evaluated alone.

## 0.13.0

New operators for the intercept and residual of a rolling regression, and a group prior for shrunk betas. Existing
expressions keep their values and cache keys.

- **`ts_alpha(y, x1, ..., xk, period[, min_periods])`**: the window's intercept. With one regressor, mean(y) − beta
  × mean(x), both means over the days both series have a value; written by hand with two `ts_mean`s it is wrong
  whenever `y` has a gap `x` doesn't. With several, one joint least-squares fit over the days all have a value (at
  least k + 2), a multi-factor alpha: `ts_alpha(returns(1), field('mkt'), field('smb'), field('hml'), 252)`.
  Duplicated regressors get the minimum-norm fit. Three factors on 3,000 stocks over five years take a few seconds.
- **`ts_resid(y, x1, ..., xk, period[, min_periods])`**: today's residual, y − alpha − Σ beta_j × x_j, from the
  window ending today: the part of today's `y` the regressors don't explain, the building block of residual momentum
  and reversal.
- **`ts_alpha_shrunk` and `ts_resid_shrunk`**: the same with `ts_beta_shrunk`'s slope (one regressor). For a fixed slope the
  least-squares intercept is still the line through the means, so alpha_shrunk − alpha = (beta − beta_shrunk) ×
  mean(x).
- **A group prior for the shrunk fits**: `ts_beta_shrunk(y, x, period[, min_periods][, groups])`, and the same for
  `ts_alpha_shrunk` and `ts_resid_shrunk`. With groups, a beta is pulled toward its own group's mean beta by that
  group's spread; a stock with no group has no value, and a group with fewer than two betas keeps its own. Without
  groups nothing changes, cache key included. A number written as a number is `min_periods` and anything else is the
  groups, so groups come last and `min_periods` is written out (`126`, not `252 / 2`); either mistake is refused.

## 0.12.2

**Numbers change** for every `Backtest` summary on a signal with a warm-up, for the better.

- **A backtest starts on the first day the book holds a position.** The days before a signal has a value (a 60-day
  volatility's first 60 days) were counted as days with a flat book. Daily returns were right, but every summary
  diluted: `return_gmv` divided total P&L by an average gross that included those zero days, so it was too large by
  the ratio of all days to days held, and `sharpe`, `annual_return`, `annual_volatility`, `sortino`, `calmar` and
  `avg_turnover` averaged in zeros. On 9 days held after 20 of warm-up, return on GMV read 3.5% instead of 1.1%.
  Those leading days are now left out of every series and number; each day from the first one held is exactly as
  before, and a flat day once the book is running still counts.

## 0.12.1

Fixes found while rewriting the documentation. Expressions that parsed before give the same tree, except those with a
`#` comment followed by more lines, which now parse in full.

- **`#` comments end at the end of their line.** The parser joined an expression's lines before removing comments, so
  a comment hid every line after it: `rank(returns(5))  # reversal` followed by a line `* -1` parsed, silently, as
  `rank(returns(5))`. Comments are now removed line by line first (a `#` inside a quoted string is kept), in
  `_collapse_whitespace`, which tools that join several lines into one expression can use too.
- **Clear errors for what qex doesn't have.** `and`, `or`, `not`, `%`, `//`, keyword arguments, lists, indexing,
  attributes and `if`/`else` raised `KeyError` or "Unsupported node type"; each now says what to write instead
  (`&`, `|`, `~`, `where`, arguments by position). Unary `+x` is `x`.
- **Constant arguments are checked when an expression is read**: windows and counts must be whole numbers of at
  least 1 (a lag, 0 or more), half-lives and `truncate`'s limit above 0, `winsorize`'s level in [0, 0.5], names
  quoted strings. `returns('5')`, `delay(x, 1.5)`, `quantile(x, 0)`, `ts_mean(x, 0)` and `truncate(x, -1)` were
  accepted and gave errors later or meaningless numbers. Values pass through unchanged (`20.0` stays `20.0`), so no
  cache key changes.
- **The risk model's `liquidity` factor is dollar volume.** It was described as "20-day average dollar volume" but
  computed `rank(ts_mean(volume(1), 20))`, share volume; it is now `rank(adv(20))`. Risk-model numbers that use it
  change.

## 0.12.0

**Breaking: `group_std` changes meaning.** Everything else here is new and changes no existing expression.

### `group_std(x, groups)` is one day's spread within each group

Every other group operator (`group_rank`, `group_demean`, `group_sum`, `group_quantile`) works on one day's
cross-section; `group_std` was the exception. `group_std(x, g, n)` pooled n days of every member's values into one
sample, so it mostly measured how much the whole group moved over the window, not how far apart its members are.
Now `group_std(x, g)` is the standard deviation across the group's members that day (ddof 1), broadcast to every
member, so `group_demean(x, g) / group_std(x, g)` is a within-group z-score. It is vectorized (the old form looped in
Python over every day and group).

The old three-argument form **is refused when the expression is parsed**, with a message naming the replacement; it
never silently gives different numbers. For a pair's spread against its own history, which is what the old pairs
examples meant: `let s = group_demean(x, 'pair') in s / ts_std(s, n)`. The pairs-trading guide, the cookbook and the
README now use that. `group_count_valid(x, g, w)` keeps its window.

Before adopting this version, check stored expressions for `group_std`.

### The other group operators: three fixes that change numbers, and one that doesn't

Each change touches only the cases named; everywhere else the numbers are identical.

- **Labels are taken as of each date (a look-ahead fix).** When the labels have fewer dates than the data (monthly
  sector labels on daily prices), a date without a label row took the label frame's *last* row, so a stock was
  grouped by a label from the future. Now each date takes the latest row on or before it, and a date before the first
  row has no group. This is in the shared label alignment, so it covers every group operator, `group_cross`,
  `build_book`'s group neutralization and `explain_missing`. Labels given every day are unaffected.
- **`group_sum` of a group with no values that day is missing, not 0.** 0 is a value, and a position once a book is
  built; `group_demean` and `group_rank` already gave NaN there.
- **Names outside the universe get NaN from `group_sum`, `group_std` and `group_count_valid`,** as every other
  cross-sectional operator gives them. They used to get their group's number. A member whose own value is missing
  still does.
- **`group_count_valid` is vectorized,** with results identical to the old loop (tested against it), which took
  about 2 s per 150 days of 3,000 names. Its window must now be at least 1.

### New operators

- `exp(x)`: the inverse of `log`. With it, compounding (`exp(ts_sum(log(1 + r), n)) - 1`), a sigmoid
  (`1 / (1 + exp(-x))`) and a cross-sectional softmax (`scale(exp(x / T))`) need no operators of their own. Too large
  an argument gives inf.
- `ts_quantile(x, period, q[, min_periods])`: the rolling q-quantile (0.5 the median, 0 and 1 `ts_min` and
  `ts_max`), interpolated linearly, with the same `min_periods` default as the other rolling operators. Value-at-risk
  is `-ts_quantile(returns(1), 252, 0.05)`; an interquartile range no single print can move is the 0.75 quantile less
  the 0.25.
- `is_finite(x)`: 1 where `x` is a finite number, 0 where it is missing or infinite (`is_valid` counts an infinity as
  valid).
- `regression_neut(y, x1, x2, ...)`: several regressors in one least-squares fit per day, so the residual has zero
  covariance with each. Nesting `regression_neut(regression_neut(y, x1), x2)` is not the same when `x1` and `x2` are
  correlated: the second step brings back some of the `x1` exposure. With one regressor nothing changes, including
  its cache key.

  A day needs at least k + 2 valid names for k regressors (three for one). With fewer, the fit passes through every
  point and the residual is rounding noise (around 1e-16) that `rank` or `zscore` would turn into full-size
  positions; such a day is now NaN, for the one-regressor form too (it gave that noise before), and
  `explain_missing` reports a new reason, `too few names`. On a cross-section of hundreds of names it never applies.

- `pca_factor(x, window, k[, refit])`: each stock's beta to the k-th statistical factor of `x`, from a correlation
  PCA across stocks on the trailing `window` days, refit every `refit` days (default 21) and held in between.
  Components are tracked across refits (matched by overlap, signs aligned), so a factor keeps its identity; names
  need values on 90% of the window; inside a universe only members enter the fit. Every `pca_factor` on the same `x`,
  `window` and `refit` shares one decomposition per refit: it is a node of its own in the graph (`pca_basis`), so the
  DAG shows it shared and Dask computes it once. With the joint `regression_neut`, a statistical-factor
  residual is one line: `regression_neut(y, pca_factor(r, 252, 1), ..., pca_factor(r, 252, 5))`.

The look-ahead check (`qex.adjust`) knows the new operators: `ts_quantile` keeps its input's adjustment, `exp` undoes
`log`, and `is_finite` gives a plain number.

### Fixed

- `SignalGraph.compute_dask` (and `DistributedSignalEngine` on a cluster) gave each node's task only the data, so every
  task recomputed its whole subtree: a five-node chain computed its leaf five times. Each task now receives its
  inputs' results and computes only its own node. On a PCA residual reversal with three factors, that is one
  decomposition and one `returns(1)` instead of 15 and 20, with identical results. CI now installs the `distributed`
  extra, so the Dask tests run.

- The cookbook's event-driven section (Section 6) set its earnings surprises by chained assignment
  (`df.iloc[i][col] = v`), which pandas 3's copy-on-write discards: the data stayed all NaN and the section ran on
  no events. It now sets them in one step (`df.loc[date, col] = v`) and finds its 20 events on pandas 2 and 3.

## 0.11.0

### Adjusted data and look-ahead: `qex.adjust`

`check(signal, fields, mode="refuse" | "warn" | "off")` finds expressions that use an adjusted level in a way that
sees the future: comparing it across stocks (`rank(close())`), with a fixed number (`close() > 5`), or mixing it with
data adjusted differently (`field('price') * volume(20)`). Each field declares an adjustment exponent (+1 an adjusted
price, -1 adjusted volume, 0 as traded); the check follows it through the expression, so ratios over time
(`close() / ts_max(close(), 252)`) and returns pass. It also refuses adding values in different units. `refuse` raises
`LookAhead` with what to write instead; `warn` returns the problems. Nothing changes unless a caller passes field
declarations. Guide: *Adjusted Data*.

## 0.10.0

### A minimum number of events

`ts_mean_events(x, n, min_n)`, `ts_std_events` and `ts_sum_events` take an optional minimum: the window gives a value
once it has `min_n` events, not only all `n` (the default, unchanged). A dollar-volume average over the last 20
sessions that needs only 10 is `ts_sum_events(price * vol, 20, 1) / ts_sum_events(vol > 0, 20, 1)`, guarded by
`ts_sum_events(vol > 0, 20, 1) >= 10`. A count outside 1 to `n` is refused.

### Faster event windows

The event windows were a Python loop over every cell; they are now pandas rolling windows over each stock's events,
with identical results (tested against the old loop): about five times faster on a full panel.

## 0.9.0

Universes as qex expressions. Nothing that worked before changes: a mask or a field's name is a universe as before.

### A universe can be an expression

`cross_section_universe`, `BookSettings(universe=...)` and `explain_missing(universe=...)` take a mask, a data
field's name (as before), or a universe expression: qex text or a parsed signal, nonzero for a member. It is
evaluated over every stock with no universe active (it defines the cross-section), must give numbers, and is computed
once per `compute_context`. `universe_mask(universe, data)` gives the membership itself. The data can stay raw, and
the rule that makes a universe is code.

### `latch(on, off)`

1 from a day `on` is true until a day `off` is true, 0 before; a missing day changes nothing, and `off` wins a tie.
Hysteresis is `latch(mcap >= 300e6, mcap < 240e6)`: on noise around one line, a hard cutoff changes state 9 times in
twelve days where the band changes 3.

### `fill_forward(x)` with no limit

Carries each value until the next one; `fill_forward(x, n)` still stops after `n` days.

### Docs

A new guide, *Universes*: the three forms, how an expression is evaluated, hysteresis with `latch`, and
reconstituting on a schedule.

## 0.8.0

Groupings from any expression, sorts within groups, and crossed groupings. Nothing that worked before changes.

### Group operators take a labels expression

`groups` in `group_rank`, `group_demean`, `group_sum`, `group_std` and `group_count_valid` is a group's name, as
before, or any expression that gives each stock a label on each day: `group_demean(x, quantile(field('mcap'), 5))`
is size-neutral, `group_demean(x, coalesce(group('sector'), 'Other'))` gives unlabelled stocks a group, and a named
`let` binding works too. A stock with no label is in no group and gets no value, with the reason `no group`, or the
label's own reason (`no data, at field('mcap')`). Anything else is refused with a message saying what `groups` can be
(before, an expression failed with `unhashable type`).

### `group_quantile(x, groups, n)`

Quantile buckets within each group, 1 to n, as `quantile` within each group. `group_quantile(value, quantile(size,
3), 3)` is a dependent double sort, which before had to be written as one `where` branch per bucket.

### `group_cross(a, b, ...)`

One grouping from several: each stock's label is its labels joined with `|` (`'Technology|3'`), and missing where any
part is. `group_demean(x, group_cross('sector', quantile(field('mcap'), 3)))` is sector- and size-neutral. A label
containing `|` is refused, since two different pairs could join to the same label.

### Numbers and labels, checked when parsed

Every expression gives numbers or labels (`group(...)`, `group_cross(...)`, a string, and `coalesce` or `where` of
labels). Labels can be compared with `==` and `!=`, used to group, and passed through `coalesce` and `where`; an
expression that hands labels to an operator needing numbers is now refused when it is parsed, naming the part and
saying what to write instead. Before, `rank(group('sector'))` silently ranked the sector names alphabetically, and
`group('sector') + 1` failed with a pandas error. `qex.kinds.kind(signal)` says which an expression gives, and
`qex.kinds.numbers_only(signal)` why one cannot be an alpha.

### Fixes

- `explain_missing` failed on any expression with `group(...)` in it, such as
  `where(group('sector') == 'Financials', None, x)`: it read the labels as numbers. It now reads a label node as
  present or missing.
- The graph view (`SignalGraph`) shows a grouping expression as an input of its group operator.

### Docs

Two new guides: *Missing Values* (`None`, why a value is missing, three-valued logic, `coalesce`) and *Group
Operations* (every operator, labels from expressions, missing labels, double sorts, crossed groupings).

## 0.7.1

Two bug fixes. Nothing changes for a daily book (`holding_period=1`, the default) or for code that evaluates each
panel in its own context.

### `Backtest(holding_period=h)` holds overlapping books (#24)

`holding_period=h > 1` rebalanced every day on h-day forward returns and summed those as daily P&L, so each day's
move was counted up to h times: total and annual return about h times too big, volatility inflated, a strongly
autocorrelated P&L, and turnover charged as for a daily book. Now each day's target is held for h days, as h
overlapping books (Jegadeesh and Titman, 1993): the book held is the average of the last h targets, P&L is that book
times one-day returns, and turnover is its change. `QuantileAnalysis(holding_period=h)` is unchanged.

### `compute_context` knows its data by identity

The context checked its cache against the data's shape and the ends of its index only, so evaluating a second panel
with the same shape and dates in one context returned the first panel's results. It now compares the data's frames by
identity (a dict's values; a `LazyData` itself), and holds them so an id cannot be reused. A new dict around the same
frames keeps the cache. A value changed in place is still not detected (checking values would cost about a second per
node on a real panel); call `ctx.clear()`. The old check also read every field, so under a context a `LazyData`
loaded all its fields; it now loads only the ones an expression uses.

## 0.7.0

Comparisons and logic become three-valued. **Numbers change** wherever a comparison meets a missing value.

### A comparison with a missing side is missing

`>`, `<`, `>=`, `<=`, `==` and `!=` give 1 or 0 as before, and NaN where either side is missing (before, 0). A 0
there was a confident "no": `where(close() > 8, 1, -1)` shorted a stock with no price, and `~(close() > 8)` said
the same stock was not above 8. `group('sector') == 'Financials'` is missing for a name with no sector.

### `&`, `|` and `~` follow Kleene's logic

As pandas' `NA` does: false & missing is false, true | missing is true, and every other combination with a missing
side is missing; `~` keeps missing missing. So guards keep their answer: `is_valid(x) & (x > k)` is 0 where `x` is
missing.

### `where` with a missing condition is missing

`where` read its condition with `astype(bool)`, so a missing condition counted as true and took `if_true`. Now it
gives a missing value. `explain_missing` passes the condition's reason on.

### `coalesce(x, y)`

`x` where it has a value, else `y` (a number or a signal). It is how an expression says what a missing value should
mean: `coalesce(close() > 8, 0)` is the comparison where it is known and 0 where it is not, which is what a
comparison gave everywhere before this release. A new operator: nothing that does not use it changes.

### What moves

An expression whose comparisons only see values that are present is unchanged. One that compares a series still
warming up (`where(ts_std(x, 21) > ts_std(x, 252), a, b)` in its first year) or a field that can be missing on
days the gated value is not (`returns(1) * (field('volume') > 60)`) now has no value there, where before it took
the "false" answer.

## 0.6.0

Missing values get a literal and a reason. **No number changes**: both are additions, and expressions that do not
write `None` evaluate exactly as before.

### `None`: no value here, on purpose

`where(cond, x, None)` leaves `x` out where `cond` is false. `None` is missing like any NaN (cross-sections and a
book leave it out, arithmetic with it stays missing, `is_valid` is 0), which is what a `0` was often wrongly written
for: a book demeans its signal, so a `0` is a position. It replaces the `x * 0 / 0` trick. The documentation's
examples that wrote `0` to mean "not this stock" now write `None`.

### `explain_missing`: why a value is missing, and where it began

`explain_missing(signal, data, universe=...)` traces every missing cell of a signal through its graph and reports a
reason (no data, outside the universe, short history, excluded on purpose, no group, calculation error, infinite
value) and the node where it began. A short history says why the earlier days were missing, so a new member's
`ts_mean(rank(x), 10)` reads "short history: outside the universe on earlier days". `report.summary(within=...)`
gives each reason's share, and `report.on(date)` each missing name's reason that day. It reads the values the
signal's own evaluation produced and changes none of them.

## 0.5.2

Bug fixes. **Numbers change** for `ts_beta` and
`ts_beta_ewma` on names with missing values, for cross-sections containing an infinity, and for
everything `FactorRiskModel` produces.

### `ts_beta` and `ts_beta_ewma` pair their observations

The covariance was estimated on the days both inputs were valid, but the variance of `x` on all
of `x`'s days, so a hole in `y` (a new listing, a halt) biased the beta. Both now use only the
days where both are valid, as the documentation already said and as least squares does. Names
with no missing values give the same betas as before. The workaround
`ts_beta(y, x + 0 * y, n)` is no longer needed. (#20)

### One infinite value no longer empties a cross-section

`zscore`, `demean`, `winsorize`, `scale`, `regression_neut`, `group_demean`, `group_std` and
`group_sum` now leave infinite values out of the day's statistics, and the infinite name gets NaN.
Before, one zero volume in `zscore(volume(20) / volume(1))` gave NaN for every stock, `demean`
gave −inf for every other stock, and `scale` gave them all 0. Rank-based operators are unchanged.

### `FactorRiskModel`

Three fixes, each of which changed every number the model produced. On simulated data with one
market factor and a 1-day reversal factor (750 days, 300 stocks), before and after:

| | 0.5.1 | 0.5.2 |
|---|---|---|
| Reversal factor return Sharpe (true value 0) | −300 | −0.6 |
| `diag(factor_covariance) / (252 · var(factor_returns))` | 0.0013 | 1.0 |
| Equal-weight long book: forecast vol (realized 19.3%) | 0.0% | 19.4% |

- **No look-ahead.** Day t's returns are now regressed on the exposures known at day t−1's
  close. Before, same-day exposures let a factor built from day t's return (reversal,
  volatility) explain that return with hindsight.
- **Covariance and specific variance were about n times too small.** The weights were
  normalized to sum to one and then the weighted series passed to `.cov()`. They are now a
  proper weighted covariance (equal to `.cov()` with equal weights, and to `numpy.cov` with
  `aweights`).
- **The market factor is in the exposures.** It was in the factor covariance but not in
  `factor_exposures`, so `portfolio_risk` and `factor_attribution` left it out. Every stock now
  has exposure 1 to `market`, in the fitted exposures and in `get_exposures`.

### Smaller fixes

- `ts_rank` is vectorized: about 0.4 s for 600 names over 1,500 days with a 252-day window,
  against about a minute. Results are identical.
- Math functions accept numbers: `sqrt(5)`, `log(2)`, `abs(-3)`, `power(2.0, x)` and an exponent
  that is a signal no longer raise `AttributeError`.
- A signed number in a function argument is a number: `delay(close(), -1)` now says
  `got -1` instead of printing an object, and `power(x, -1)` works.
- CI installs the `risk` extra, so the risk model's tests run.

## 0.5.1

Bug fixes. **Numbers change** for any data with missing prices (halts, gaps, delistings):
on pandas 2 the signal side changes, on pandas 3 backtest and quantile/IC results change,
and backtests change on both. Data with no missing prices gives the same results as 0.5.0.

### Returns no longer depend on the pandas version (#21)

Every return was `pct_change` with the pandas default for missing prices, which changed
between pandas 2 (carry the last price forward: fake 0% returns during halts and forever
after a delisting) and pandas 3 (no bridging: the move across a halt is lost). All six call
sites (`returns`, `volatility`, the risk model's returns, and the forward returns of
`Backtest`, `QuantileAnalysis.run` and `QuantileAnalysis.ic_summary`) now use one rule,
`qex.price_returns`:

- a day with no price has a NaN return;
- a missing start price is replaced by the last valid price before it, so the reopen day
  carries the whole move since the last price, and daily returns compound to n-day returns;
- a price more than `MAX_PRICE_GAP = 10` rows old no longer counts.

New: `price_returns`, `forward_returns`, `MAX_PRICE_GAP`.

### Backtests hold positions in names that can't be traded

A name with no close on a day (halted, or delisted) used to get weight 0 that day, which
closed the position at a price that doesn't exist and dropped the move across the halt
from P&L. `Backtest` now holds the previous weight, marked at the last price, until the
name trades again, for at most `MAX_PRICE_GAP` days; after that the position is dropped
with no P&L for the missing move (delisting returns are not modelled). Held weights are not
rescaled, so gross can differ slightly from 1 on those days. `build_book` still returns the
exact target book. New: `hold_untradable`.

### Python 3.9 is no longer supported

`requires-python` is now `>=3.10`. Python 3.9 reached end of life in October 2025, and
GitHub's current `ubuntu-latest` runner no longer provides it. CI tests 3.10, 3.11 and 3.12,
and one failing version no longer cancels the others (`fail-fast: false`).

### Other fixes

- `QuantileAnalysis.ic_summary` raised `KeyError` when no day had at least 5 names; it now
  returns NaN statistics.
