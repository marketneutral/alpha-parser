"""Several complex signals at once: everything they have in common is computed once, and the DAG shows it shared.

The signals below overlap on purpose:

- two statistical factors from one PCA (pca_factor k = 1 and 2 share one PcaBasis);
- a cross-sectional fit read for its intercept and a slope (one RegressionFit);
- a sector-prior shrunk beta, used directly and inside the fit behind ts_resid_shrunk and ts_alpha_shrunk (one
  TsBetaShrunk, one TsFit);
- a two-regressor time-series fit read for its alpha and its residual (one TsFit);
- returns(1) everywhere.

For each, in one compute_context, in SignalGraph.compute and on Dask: every node is computed exactly once, and every
result is what evaluating the signal on its own gives.
"""

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from qex import qex, compute_context
from qex.signal import Signal
from qex.distributed import SignalGraph
from qex.missing import children

SIGNALS = {
    # residual reversal, neutral to the first two statistical factors
    "pca_neutral": "let r = returns(1) "
                   "in regression_neut(zscore(-returns(5)), pca_factor(r, 120, 1), pca_factor(r, 120, 2))",
    # the second factor's premium, timed by its own recent sign (Fama-MacBeth on the two factor betas)
    "factor_timing": "let r = returns(1), p1 = delay(pca_factor(r, 120, 1), 1), p2 = delay(pca_factor(r, 120, 2), 1) "
                     "in sign(ts_mean(regression_beta(r, p1, p2, 2), 60)) * zscore(p2)",
    # the zero-exposure return of the same fit, smoothed
    "factor_alpha": "let r = returns(1), p1 = delay(pca_factor(r, 120, 1), 1), p2 = delay(pca_factor(r, 120, 2), 1) "
                    "in ts_mean(regression_alpha(r, p1, p2), 60)",
    # idiosyncratic reversal on a sector-prior shrunk beta, and that beta taken out of a reversal, and its alpha
    "idio_reversal": "let e = ts_resid_shrunk(returns(1), returns(1, 'market'), 120, 'sector') in rank(-ts_sum(e, 5))",
    "beta_neutral": "let b = ts_beta_shrunk(returns(1), returns(1, 'market'), 120, 'sector') "
                    "in regression_neut(rank(-returns(5)), b) + ts_alpha_shrunk(returns(1), returns(1, 'market'), "
                    "120, 'sector')",
    # a two-regressor alpha and residual on the same fit
    "two_factor": "let r = returns(1) in ts_alpha(r, returns(1, 'market'), field('style'), 120) * 252 "
                  "+ ts_resid(r, returns(1, 'market'), field('style'), 120)",
}


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(11)
    T, N = 320, 30
    idx = pd.bdate_range("2022-01-03", periods=T)
    names = [f"S{i:02d}" for i in range(N)]
    f = rng.normal(0, 0.01, (T, 2))                                   # two common factors
    load = rng.normal([1.0, 0.0], [0.3, 1.0], (N, 2))
    r = f @ load.T + rng.normal(0, 0.012, (T, N))
    close = pd.DataFrame(40 * np.cumprod(1 + r, axis=0), index=idx, columns=names)
    close.iloc[150:154, 3] = np.nan                                   # a halt
    return {
        "close": close,
        "market": pd.Series(100 * np.cumprod(1 + f[:, 0]), index=idx),
        "style": pd.Series(f[:, 1], index=idx),
        "sector": pd.DataFrame([[("A", "B", "C")[i % 3] for i in range(N)]] * T, index=idx, columns=names),
    }


def _graph():
    g = SignalGraph()
    for name, expr in SIGNALS.items():
        g.add_signal(qex(expr), name=name)
    return g


def _alone(data):
    """Each signal evaluated by itself, with no shared cache."""
    return {name: qex(expr).evaluate(data) for name, expr in SIGNALS.items()}


class _CountComputes:
    """Counts each node's _compute by its cache key, for every class in the graph."""

    def __init__(self, classes):
        self.classes, self.calls, self.saved = classes, Counter(), {}

    def __enter__(self):
        originals = {cls: cls._compute for cls in self.classes}     # before any is wrapped: a subclass inherits
        for cls in self.classes:
            original = originals[cls]

            def counted(this, data, _original=original):
                self.calls[this._cache_key()] += 1
                return _original(this, data)

            self.saved[cls] = cls.__dict__.get("_compute")
            cls._compute = counted
        return self

    def __exit__(self, *exc):
        for cls, original in self.saved.items():
            if original is None:
                del cls._compute
            else:
                cls._compute = original


def _classes(g):
    return {type(n) for n in g.nodes.values()}


def test_the_shared_parts_are_single_nodes():
    g = _graph()
    kinds = Counter(k[0] for k in g.nodes)
    assert kinds["PcaBasis"] == 1                    # pca_factor k = 1 and k = 2, in three signals
    assert kinds["PcaFactor"] == 2
    assert kinds["RegressionFit"] == 1               # regression_alpha and regression_beta on the same inputs
    assert kinds["TsBetaShrunk"] == 1                # written directly, and the slope of the shrunk fit
    assert kinds["TsFit"] == 2                       # the shrunk one-regressor fit and the two-regressor fit
    assert sum(1 for k in g.nodes if k == ("Returns", 1, "close")) == 1

    users = Counter(d for deps in g.dependencies.values() for d in deps)
    basis = next(k for k in g.nodes if k[0] == "PcaBasis")
    fit = next(k for k in g.nodes if k[0] == "RegressionFit")
    assert users[basis] == 2 and users[fit] == 2     # both factors read one basis; alpha and beta read one fit
    shrunk = next(k for k in g.nodes if k[0] == "TsBetaShrunk")
    assert users[shrunk] == 2                        # the shrunk fit, and regression_neut's regressor


def test_every_node_is_wired_to_all_its_inputs():
    g = _graph()
    for key, node in g.nodes.items():
        expected = {c._cache_key() for c in children(node)}
        assert set(g.dependencies[key]) == expected, key[0]
        if not expected:
            assert key[0] in ("Field", "Returns", "Constant", "Group"), f"{key[0]} has no inputs"


def test_one_compute_context_computes_each_node_once(data):
    g = _graph()
    alone = _alone(data)
    with _CountComputes(_classes(g)) as counter, compute_context():
        together = {name: qex(expr).evaluate(data) for name, expr in SIGNALS.items()}
    repeated = {k[0]: n for k, n in counter.calls.items() if n > 1}
    assert repeated == {}
    for name in SIGNALS:
        pd.testing.assert_frame_equal(together[name], alone[name])


def test_signal_graph_computes_each_node_once(data):
    g = _graph()
    alone = _alone(data)
    with _CountComputes(_classes(g)) as counter:
        out = g.compute(data)
    assert {k[0]: n for k, n in counter.calls.items() if n > 1} == {}
    assert set(counter.calls) <= set(g.nodes)
    for name in SIGNALS:
        pd.testing.assert_frame_equal(out[name], alone[name])


def test_dask_computes_each_node_once(data):
    distributed = pytest.importorskip("dask.distributed")
    g = _graph()
    alone = _alone(data)
    with _CountComputes(_classes(g)) as counter, distributed.Client(processes=False) as client:
        out = g.compute_dask(data, client=client)
    assert {k[0]: n for k, n in counter.calls.items() if n > 1} == {}
    for name in SIGNALS:
        pd.testing.assert_frame_equal(out[name], alone[name])


def test_the_counter_sees_repeats_without_a_shared_cache(data):
    """The control: with no compute context nothing is shared, so the same nodes are computed again."""
    g = _graph()
    with _CountComputes(_classes(g)) as counter:
        for expr in SIGNALS.values():
            qex(expr).evaluate(data)
    repeated = {k[0] for k, n in counter.calls.items() if n > 1}
    assert {"Returns", "PcaBasis", "RegressionFit"} <= repeated


def test_compute_contexts_in_parallel_threads_never_mix():
    """Before 0.14.0 the context was one module global: threads entering and leaving contexts at overlapping times
    could leave one active for the whole process. Each thread now has its own."""
    import threading
    from qex.context import get_context
    barrier = threading.Barrier(8)
    seen, errors = [], []

    def work():
        try:
            with compute_context() as mine:
                barrier.wait()                              # every thread inside its context at once
                seen.append(get_context() is mine)
                barrier.wait()
            seen.append(get_context() is None)
        except Exception as e:                              # pragma: no cover - reported below
            errors.append(e)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and all(seen) and len(seen) == 16
    assert get_context() is None                            # nothing left behind in the main thread
