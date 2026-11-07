"""pca_factor(x, window, k, refit): each stock's beta to the k-th statistical factor of x."""

import numpy as np
import pandas as pd
import pytest

from qex import qex, compute_context, explain_missing
import qex.crosssection as cs
from qex.universe import cross_section_universe

NAN = np.nan


def _ev(expr, data):
    with compute_context():
        return qex(expr).evaluate(data)


def _two_factor_data(T=600, N=80, seed=0):
    """A market factor with betas around 1, and a factor long the first half of names, short the second."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=T)
    cols = [f"S{i}" for i in range(N)]
    market, split = rng.normal(0, 0.01, T), rng.normal(0, 0.006, T)
    beta = rng.uniform(0.5, 1.5, N)
    side = np.where(np.arange(N) < N // 2, 1.0, -1.0)
    r = pd.DataFrame(market[:, None] * beta + split[:, None] * side + rng.normal(size=(T, N)) * rng.uniform(0.01, 0.03, N),
                     index=idx, columns=cols)
    return {"close": 100 * (1 + r).cumprod(), "r": r}, beta, side


def test_recovers_planted_factors():
    d, beta, side = _two_factor_data()
    f1 = _ev("pca_factor(field('r'), 252, 1)", d).iloc[-1]
    f2 = _ev("pca_factor(field('r'), 252, 2)", d).iloc[-1]
    assert (f1 > 0).all()                                            # the market, signed long
    assert np.corrcoef(f1, beta)[0, 1] > 0.8
    assert abs(np.corrcoef(f2, side)[0, 1]) > 0.95


def test_value_is_the_beta_to_the_eigenportfolio():
    d, _, _ = _two_factor_data(T=300, N=30)
    r = d["r"]
    out = _ev("pca_factor(field('r'), 120, 2, 1000)", d)               # one fit, at row 119
    window = r.iloc[:120]
    z = (window - window.mean()) / window.std()
    _, _, vt = np.linalg.svd(z.to_numpy(), full_matrices=False)
    factor = z.to_numpy() @ vt[1]                                     # the eigenportfolio's return, standardized space
    slope = [np.cov(window[c], factor)[0, 1] / factor.var(ddof=1) for c in r.columns]
    got = out.iloc[119].to_numpy()
    assert np.allclose(np.abs(got), np.abs(slope), rtol=1e-8)         # sigma_i * v_i, up to the component's sign
    assert np.allclose(got * np.sign(got[0]), np.array(slope) * np.sign(slope[0]), rtol=1e-8)


def test_held_between_refits_and_starts_with_a_full_window():
    d, _, _ = _two_factor_data(T=320)
    out = _ev("pca_factor(field('r'), 252, 1, 21)", d)
    assert out.iloc[:251].isna().all().all() and out.iloc[251].notna().all()
    changes = out.iloc[251:].diff().abs().sum(axis=1) > 0
    refits = np.flatnonzero(changes.to_numpy())
    assert set(refits) == {21, 42, 63}                               # rows 272, 293, 314: every 21 rows
    daily = _ev("pca_factor(field('r'), 252, 1, 1)", d).iloc[251:]
    assert (daily.diff().abs().sum(axis=1).iloc[1:] > 0).all()


def test_one_decomposition_shared_by_every_k(monkeypatch):
    d, _, _ = _two_factor_data(T=300, N=20)
    calls = []
    real = cs._pca_fits
    monkeypatch.setattr(cs, "_pca_fits", lambda *a: calls.append(1) or real(*a))
    with compute_context():
        for k in range(1, 6):
            qex(f"pca_factor(field('r'), 120, {k})").evaluate(d)
        qex("pca_factor(field('r'), 120, 1, 5)").evaluate(d)          # another refit: another decomposition
    assert len(calls) == 2


def test_components_keep_their_identity_when_their_size_order_swaps():
    """Two group factors whose variances cross half way: k = 1 follows the same group throughout."""
    rng = np.random.default_rng(3)
    T, N = 700, 40
    idx = pd.bdate_range("2015-01-01", periods=T)
    group_a = np.arange(N) < N // 2
    vol_a = np.where(np.arange(T) < T // 2, 0.020, 0.008)
    vol_b = np.where(np.arange(T) < T // 2, 0.008, 0.020)
    fa, fb = rng.normal(size=T) * vol_a, rng.normal(size=T) * vol_b
    r = pd.DataFrame(np.where(group_a, fa[:, None], fb[:, None]) + rng.normal(0, 0.01, (T, N)), index=idx)
    out = _ev("pca_factor(field('r'), 120, 1, 21)", {"close": r + 100, "r": r})
    fits = out.drop_duplicates().dropna()
    on_a = fits.where(np.tile(group_a, (len(fits), 1))).abs().mean(axis=1)
    on_b = fits.where(np.tile(~group_a, (len(fits), 1))).abs().mean(axis=1)
    pos = np.array([idx.get_loc(t) for t in fits.index])
    clean = (pos < T // 2) | (pos - 119 >= T // 2)                  # windows wholly in one regime
    # where the two factors are equally large any mix of them is a first component; elsewhere k = 1 is group a,
    # also after group a's factor has become the smaller one (strict ordering would switch to group b there)
    assert (on_a[clean] > 2 * on_b[clean]).all()
    assert (pos[clean] - 119 >= T // 2).sum() >= 5
    # and the sign never flips between refits
    corr = [np.corrcoef(fits.iloc[i], fits.iloc[i + 1])[0, 1] for i in range(len(fits) - 1)]
    assert min(corr) > 0


def test_coverage_and_infinities():
    d, _, _ = _two_factor_data(T=300, N=20)
    r = d["r"].copy()                                                  # fits at rows 119 and 299 (window 180..299)
    r.iloc[180:300:20, 0] = NAN                                        # S0: 5% of the window missing: still fitted
    r.iloc[180:300:5, 1] = NAN                                         # S1: 20% missing: left out
    r.iloc[250, 2] = np.inf                                            # one infinity in S2: treated as missing
    out = _ev("pca_factor(field('r'), 120, 1, 180)", dict(d, r=r)).iloc[-1]
    assert np.isfinite(out["S0"]) and np.isnan(out["S1"]) and np.isfinite(out["S2"])


def test_universe_fits_on_members_only():
    d, _, _ = _two_factor_data(T=300, N=30)
    member = pd.DataFrame(True, index=d["r"].index, columns=d["r"].columns)
    member.iloc[:, :5] = False
    with compute_context(), cross_section_universe(member):
        inside = qex("pca_factor(field('r'), 120, 1, 1000)").evaluate(d)
    assert inside.iloc[:, :5].isna().all().all()
    alone = _ev("pca_factor(field('r'), 120, 1, 1000)", {"close": d["close"].iloc[:, 5:], "r": d["r"].iloc[:, 5:]})
    pd.testing.assert_frame_equal(inside.iloc[:, 5:], alone)


def test_neutralizing_to_statistical_factors():
    d, _, _ = _two_factor_data()
    expr = ("let r = field('r'), f1 = pca_factor(r, 252, 1), f2 = pca_factor(r, 252, 2) "
            "in regression_neut(zscore(r), f1, f2)")
    with compute_context():
        res = qex(expr).evaluate(d)
        f1 = qex("pca_factor(field('r'), 252, 1)").evaluate(d)
        f2 = qex("pca_factor(field('r'), 252, 2)").evaluate(d)
    day = res.index[-1]
    for f in (f1, f2):
        x = f.loc[day] - f.loc[day].mean()
        assert abs((res.loc[day] * x).sum()) < 1e-9


def test_explain_missing_before_the_first_fit():
    d, _, _ = _two_factor_data(T=300, N=10)
    report = explain_missing(qex("pca_factor(field('r'), 120, 1)"), d)
    assert set(report.on(d["r"].index[50])["reason"]) == {"short history"}


@pytest.mark.parametrize("bad, match", [
    ("pca_factor(returns(1), 252, 0)", "k must"), ("pca_factor(returns(1), 252, 21)", "first 20"),
    ("pca_factor(returns(1), 2, 1)", "window must"), ("pca_factor(returns(1), 252, 1, 0)", "refit must"),
])
def test_arguments_are_checked(bad, match):
    with pytest.raises(ValueError, match=match):
        qex(bad)


def test_adjusted_levels_are_refused():
    from qex.adjust import check, LookAhead
    f = {"close": {"adjustment": 1, "unit": "usd_per_share"}}
    check(qex("pca_factor(returns(1), 252, 1)"), f)
    with pytest.raises(LookAhead):
        check(qex("pca_factor(close(), 252, 1)"), f, alpha=False)


def test_the_decomposition_is_a_shared_node_of_the_graph():
    from qex import SignalGraph
    from qex.missing import label
    e = qex("let r = returns(1) in regression_neut(r, pca_factor(r, 252, 1), pca_factor(r, 252, 2), pca_factor(r, 252, 3, 5))")
    g = SignalGraph()
    g.add_signal(e, name="a")
    bases = [k for k in g.nodes if k[0] == "PcaBasis"]
    assert len(bases) == 2                                            # refit 21 (shared by k = 1, 2) and refit 5
    shared = ("PcaBasis", ("Returns", 1, "close"), 252, 21)
    users = [k for k, deps in g.dependencies.items() if shared in deps]
    assert sorted(k[3] for k in users) == [1, 2]
    assert label(e) == ("regression_neut(returns(1), pca_factor(returns(1), 252, 1), pca_factor(returns(1), 252, 2), "
                        "pca_factor(returns(1), 252, 3, 5))")
    assert label(g.nodes[shared]) == "pca_basis(returns(1), 252)"


def test_dask_computes_each_node_once(monkeypatch):
    dd = pytest.importorskip("dask.distributed")
    from qex import SignalGraph
    import qex.primitives as pr
    d, _, _ = _two_factor_data(T=300, N=30)
    calls = {"svd": 0, "returns": 0}
    real_fits, real_returns = cs._pca_fits, pr.Returns._compute
    monkeypatch.setattr(cs, "_pca_fits", lambda *a: calls.__setitem__("svd", calls["svd"] + 1) or real_fits(*a))

    def counted(self, data):
        calls["returns"] += 1
        return real_returns(self, data)
    monkeypatch.setattr(pr.Returns, "_compute", counted)
    expr = ("let r = returns(1) in zscore(-ts_sum(regression_neut(r, pca_factor(r, 120, 1), pca_factor(r, 120, 2), "
            "pca_factor(r, 120, 3)), 5))")
    g = SignalGraph()
    g.add_signal(qex(expr), name="a")
    with dd.Client(processes=False, dashboard_address=None) as client:
        out = g.compute_dask(d, client=client)["a"]
    assert calls == {"svd": 1, "returns": 1}                          # was 15 and 20: every task redid its subtree
    pd.testing.assert_frame_equal(out, _ev(expr, d))
