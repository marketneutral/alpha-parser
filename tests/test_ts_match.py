"""ts_match: each stock's statistical twin (0.15.0), and keyword arguments, which only ts_match takes.

The semantics are the ones qexarena's notebook "ts_match: mocking up a new operator in Python" specified and tested
on its mock: point in time, no self-matches, reciprocal exclusive pairs, matches held between rebalances, a match
ended by leaving the universe, no value without a twin, and a selection shared by every value read through it.
"""

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from qex import qex, compute_context, cross_section_universe
from qex.match import similarity, METRICS, MatchSelection
from qex.distributed import SignalGraph
from qex.missing import explain_missing, label

LOOK = 60


@pytest.fixture
def pairs_data():
    """24 stocks in 12 planted pairs (each pair shares a component), three sectors of four pairs, with gaps."""
    rng = np.random.default_rng(21)
    T, N = 260, 24
    idx = pd.bdate_range("2023-01-02", periods=T)
    names = [f"S{i:02d}" for i in range(N)]
    common = rng.normal(0, 0.012, (T, N // 2))
    r = np.repeat(common, 2, axis=1) + rng.normal(0, 0.004, (T, N))
    close = pd.DataFrame(30 * np.cumprod(1 + r, axis=0), index=idx, columns=names)
    close.iloc[100:104, 5] = np.nan                                   # a halt
    sector = pd.DataFrame([[("A", "B", "C")[(i // 2) % 3] for i in range(N)]] * T, index=idx, columns=names)
    return {"close": close, "sector": sector, "market": pd.Series(100 * np.cumprod(1 + r.mean(axis=1)), idx)}


def ev(expr, data):
    with compute_context():
        return qex(expr).evaluate(data)


def peers(expr_options, data, key="returns(1)"):
    """The selection's peer columns for ts_match(..., key, LOOK, options)."""
    s = qex(f"ts_match(returns(5), {key}, {LOOK}{expr_options})")
    with compute_context():
        return pd.DataFrame(s.selection.evaluate(data).peers, index=data["close"].index, columns=data["close"].columns)


def test_planted_twins_are_found(pairs_data):
    for opts in ("", ", matching='exclusive'", ", within='sector'"):
        p = peers(opts, pairs_data).iloc[-1]
        partner = np.arange(24) ^ 1                                    # 0<->1, 2<->3, ...
        assert (p.to_numpy() == partner).mean() > 0.9, opts


def test_the_value_read_is_the_twins(pairs_data):
    twin = ev(f"ts_match(returns(5), returns(1), {LOOK})", pairs_data)
    r5 = ev("returns(5)", pairs_data)
    p = peers("", pairs_data)
    t = len(r5) - 1
    for i, c in enumerate(r5.columns):
        assert twin.iloc[t, i] == r5.iloc[t, p.iloc[t, i]] if p.iloc[t, i] >= 0 else np.isnan(twin.iloc[t, i])


@pytest.mark.parametrize("metric", METRICS)
def test_each_metric_is_the_pairwise_definition(metric):
    """similarity() against a pair-by-pair computation over each pair's shared days, with gaps."""
    rng = np.random.default_rng(3)
    w = rng.normal(0, 0.02 if metric == "path" else 1, (30, 9))
    w[rng.random(w.shape) < 0.15] = np.nan
    s = similarity(w, metric, 10)
    for i in range(9):
        for j in range(9):
            both = np.isfinite(w[:, i]) & np.isfinite(w[:, j])
            if i == j or both.sum() < 10:
                assert np.isnan(s[i, j])
                continue
            a, b = w[both, i], w[both, j]
            if metric == "corr":
                expected = np.corrcoef(a, b)[0, 1]
            elif metric == "spearman":                                  # ranks within each full column, as documented
                ra = pd.Series(w[:, i]).rank().to_numpy()[both]
                rb = pd.Series(w[:, j]).rank().to_numpy()[both]
                expected = np.corrcoef(ra, rb)[0, 1]
            elif metric == "cosine":
                expected = a @ b / np.sqrt((a @ a) * (b @ b))
            elif metric == "euclidean":
                expected = -np.sqrt(np.mean((a - b) ** 2))
            elif metric == "path":                                     # paths from 100, a missing day flat
                pi = 100 * np.cumprod(1 + np.nan_to_num(w[:, i]))
                pj = 100 * np.cumprod(1 + np.nan_to_num(w[:, j]))
                expected = -np.sqrt(np.mean((pi[both] - pj[both]) ** 2))
            else:
                expected = -np.mean(np.abs(a - b))
            assert s[i, j] == pytest.approx(expected, abs=1e-10)


def test_point_in_time(pairs_data):
    """Scramble the key from a date on: no match chosen before that date may change."""
    d = pairs_data
    cut = 180
    scrambled = d["close"].copy()
    rng = np.random.default_rng(0)
    scrambled.iloc[cut:] = scrambled.iloc[cut:].to_numpy()[:, rng.permutation(24)]
    before = peers("", d).iloc[:cut + 1]                               # day `cut` uses key up to cut - 1
    after = peers("", dict(d, close=scrambled)).iloc[:cut + 1]
    pd.testing.assert_frame_equal(before, after)


def test_no_stock_is_its_own_twin(pairs_data):
    p = peers("", pairs_data).to_numpy()
    assert not (p == np.arange(24)[None, :]).any()


def test_exclusive_pairs_are_reciprocal(pairs_data):
    p = peers(", matching='exclusive'", pairs_data).to_numpy()
    for row in p[LOOK:]:
        has = np.flatnonzero(row >= 0)
        assert (row[row[has]] == has).all()
    twin = ev(f"ts_match(returns(5), returns(1), {LOOK}, matching='exclusive')", pairs_data)
    score = ev("returns(5)", pairs_data) - twin
    paired = ev(f"ts_match(returns(5) - ts_match(returns(5), returns(1), {LOOK}, matching='exclusive'), "
                f"returns(1), {LOOK}, matching='exclusive')", pairs_data)
    both = score.notna() & paired.notna()
    assert both.to_numpy().sum() > 0
    assert np.nanmax(np.abs((score + paired)[both].to_numpy())) < 1e-15   # each pair's scores are opposite


def test_matches_change_only_on_rebalance_days(pairs_data):
    p = peers(", rebalance=21", pairs_data)
    changed = (p != p.shift(1)) & (p.shift(1) >= 0) & (p >= 0)
    rebalance_days = set(range(LOOK, len(p), 21))
    off = [t for t in range(LOOK + 1, len(p)) if changed.iloc[t].any() and t not in rebalance_days]
    assert off == []


def test_leaving_the_universe_ends_a_match(pairs_data):
    d = pairs_data
    members = pd.DataFrame(True, index=d["close"].index, columns=d["close"].columns)
    members.iloc[200:, 0] = False                                     # S00 leaves on day 200
    with cross_section_universe(members), compute_context():
        p = qex(f"ts_match(returns(5), returns(1), {LOOK})").selection.evaluate(d).peers
    assert p[199, 1] == 0 and p[200, 1] == -1 and (p[200:205, 0] == -1).all()


def test_no_twin_and_a_missing_twin_value(pairs_data):
    d = pairs_data
    sector = d["sector"].copy()
    sector["S00"] = "alone"                                           # a group of one: no twin
    twin = ev(f"ts_match(returns(5), returns(1), {LOOK}, within='sector')", dict(d, sector=sector))
    assert twin["S00"].isna().all()
    report = explain_missing(qex(f"ts_match(returns(5), returns(1), {LOOK}, within='sector')"), dict(d, sector=sector))
    assert report.text(int(report.codes["S00"].iloc[-1])) == "no twin"
    p = peers("", d)
    t = 102                                                           # S05 is halted; whoever has S05 as twin
    holders = np.flatnonzero(p.iloc[t].to_numpy() == 5)
    twin = ev(f"ts_match(close(), returns(1), {LOOK})", d)
    assert all(np.isnan(twin.iloc[t, h]) for h in holders)            # no value, and no other twin picked


def test_groups_can_be_a_labels_expression_and_value_a_series(pairs_data):
    out = ev(f"ts_match(returns(5), returns(1), {LOOK}, within=quantile(returns(20), 2))", pairs_data)
    assert out.notna().any().any()
    m = ev(f"ts_match(returns(1, 'market'), returns(1), {LOOK})", pairs_data)
    assert m.notna().any().any()


def test_one_selection_for_every_value(pairs_data):
    e1 = f"ts_match(returns(5), returns(1), {LOOK}, within='sector')"
    e2 = f"ts_match(returns(20), returns(1), {LOOK}, metric='corr', within='sector')"
    g = SignalGraph()
    g.add_signal(qex(e1), name="a")
    g.add_signal(qex(e2), name="b")
    users = Counter(d for deps in g.dependencies.values() for d in deps)
    sel = [k for k in g.nodes if k[0] == "MatchSelection"]
    assert len(sel) == 1 and users[sel[0]] == 2
    calls = []
    original = MatchSelection._compute
    MatchSelection._compute = lambda self, data: calls.append(1) or original(self, data)
    try:
        with compute_context():
            qex(e1).evaluate(pairs_data)
            qex(e2).evaluate(pairs_data)
    finally:
        MatchSelection._compute = original
    assert len(calls) == 1


def test_keyword_defaults_give_the_same_expression():
    a = qex(f"ts_match(returns(5), returns(1), {LOOK})")
    b = qex(f"ts_match(returns(5), returns(1), {LOOK}, metric='corr', matching='nearest', rebalance=21, "
            f"min_overlap={LOOK})")
    c = qex(f"ts_match(returns(5), returns(1), {LOOK}, 'corr', None, 'nearest')")
    assert a._cache_key() == b._cache_key() == c._cache_key()
    assert label(qex(f"ts_match(returns(5), returns(1), {LOOK}, within='sector', matching='exclusive')")) == \
        f"ts_match(returns(5), returns(1), {LOOK}, within='sector', matching='exclusive')"


@pytest.mark.parametrize("expr, message", [
    ("ts_match(returns(5), returns(1), 60, metrc='corr')", "no argument named 'metrc'"),
    ("ts_match(returns(5), returns(1), 60, 'corr', metric='corr')", "both by position and by name"),
    ("ts_match(returns(5), returns(1), 60, metric='pearson')", "metric is one of"),
    ("ts_match(returns(5), returns(1), 60, matching='best')", "'nearest'.*'exclusive'"),
    ("ts_match(returns(5), returns(1), 60, min_overlap=90)", "can't be more than its lookback"),
    ("ts_match(returns(5), returns(1), 1)", "lookback must be a whole number, at least 2"),
    ("ts_match(returns(5), returns(1), 60, rebalance=0)", "rebalance must be a whole number"),
    ("rank(returns(5), x=2)", "takes its arguments by position"),
    ("ts_mean(returns(1), period=20)", "takes its arguments by position"),
])
def test_bad_arguments_are_refused_clearly(expr, message):
    with pytest.raises(ValueError, match=message):
        qex(expr)


def test_look_ahead_rules():
    from qex.adjust import check
    fields = {"close": dict(adjustment=1, unit="usd_per_share")}
    kinds = lambda e: [f["kind"] for f in check(qex(e), fields, mode="warn")]
    assert kinds("ts_match(returns(5), close(), 60)") == []                       # correlation ignores scale
    assert kinds("ts_match(returns(5), close(), 60, metric='euclidean')") != []   # a distance compares levels
    assert kinds("ts_match(close(), returns(1), 60)") != []                       # another stock's adjusted level


# --- identifier(): seeing the twins ------------------------------------------------------------------------------------

def test_identifier_is_each_stocks_own_name(pairs_data):
    ids = ev("identifier()", pairs_data)
    assert (ids.to_numpy() == np.asarray(pairs_data["close"].columns)[None, :]).all()


def test_the_twins_identifier_is_the_selection(pairs_data):
    names = np.asarray(pairs_data["close"].columns, dtype=object)
    twin = ev(f"ts_match(identifier(), returns(1), {LOOK}, within='sector')", pairs_data)
    p = peers(", within='sector'", pairs_data).to_numpy()
    expected = np.where(p >= 0, names[np.maximum(p, 0)], None)
    got = twin.to_numpy()
    assert all((e is None and pd.isna(g)) or e == g for e, g in zip(expected.ravel(), got.ravel()))


def test_identifier_reads_the_cached_selection(pairs_data):
    calls = []
    original = MatchSelection._compute
    MatchSelection._compute = lambda self, data: calls.append(1) or original(self, data)
    try:
        with compute_context():
            ev_ = lambda e: qex(e).evaluate(pairs_data)
            ev_(f"ts_match(returns(5), returns(1), {LOOK}, within='sector')")
            ev_(f"ts_match(identifier(), returns(1), {LOOK}, within='sector')")
    finally:
        MatchSelection._compute = original
    assert len(calls) == 1


def test_the_same_partner_check_is_exact(pairs_data):
    same = ev(f"ts_match(identifier(), returns(1), {LOOK}) == ts_match(identifier(), returns(1), {LOOK}, "
              f"metric='spearman')", pairs_data)
    values = same.to_numpy()
    assert set(np.unique(values[np.isfinite(values)])) <= {0.0, 1.0}
    assert same.iloc[-1].mean() > 0.8                                   # planted pairs: the metrics agree


@pytest.mark.parametrize("expr", ["rank(ts_match(identifier(), returns(1), 60))", "identifier() + 1",
                                  "ts_mean(identifier(), 5)"])
def test_identifiers_are_labels(expr):
    with pytest.raises(ValueError, match="gives labels, not numbers"):
        qex(expr)


def test_a_twins_labels_are_gathered_as_labels(pairs_data):
    twin_sector = ev(f"ts_match(group('sector'), returns(1), {LOOK})", pairs_data)
    assert set(twin_sector.stack().dropna().unique()) <= {"A", "B", "C"}
    with pytest.raises(ValueError, match="gives labels, not numbers"):
        qex(f"rank(ts_match(group('sector'), returns(1), {LOOK}))")


def test_identifiers_group_and_compare():
    qex(f"group_demean(returns(5), ts_match(identifier(), returns(1), {LOOK}, matching='exclusive'))")
    qex(f"where(ts_match(identifier(), returns(1), {LOOK}) == 'S01', returns(5), None)")


# --- ts_match_score: how alike each stock and its twin were --------------------------------------------------------

@pytest.mark.parametrize("metric", ["corr", "euclidean", "manhattan"])
def test_the_score_is_the_pairs_similarity_at_selection(pairs_data, metric):
    d = pairs_data
    score = ev(f"ts_match_score(returns(1), {LOOK}, metric='{metric}', min_overlap=40)", d)
    p = peers(f", metric='{metric}', min_overlap=40", d)
    r = ev("returns(1)", d)
    for t in range(LOOK, len(r), 21):                                  # each rebalance day
        window = r.iloc[t - LOOK:t]
        for i in range(24):
            j = p.iloc[t, i]
            if j < 0:
                assert np.isnan(score.iloc[t, i])
                continue
            both = window.iloc[:, [i, j]].dropna()
            a, b = both.iloc[:, 0].to_numpy(), both.iloc[:, 1].to_numpy()
            expected = {"corr": lambda: np.corrcoef(a, b)[0, 1],
                        "euclidean": lambda: np.sqrt(np.mean((a - b) ** 2)),
                        "manhattan": lambda: np.mean(np.abs(a - b))}[metric]()
            assert score.iloc[t, i] == pytest.approx(expected, abs=1e-10)


def test_the_score_is_held_and_missing_without_a_twin(pairs_data):
    d = pairs_data
    score = ev(f"ts_match_score(returns(1), {LOOK}, matching='exclusive')", d)
    twin = ev(f"ts_match(identifier(), returns(1), {LOOK}, matching='exclusive')", d)
    assert (score.isna() == twin.isna()).all().all()
    block = score.iloc[LOOK:LOOK + 21]
    assert (block.nunique(dropna=True) <= 1).all()                     # constant between rebalances
    sector = d["sector"].copy()
    sector["S00"] = "alone"
    report = explain_missing(qex(f"ts_match_score(returns(1), {LOOK}, within='sector')"), dict(d, sector=sector))
    assert report.text(int(report.codes["S00"].iloc[-1])) == "no twin"


def test_the_score_reads_the_same_selection(pairs_data):
    g = SignalGraph()
    g.add_signal(qex(f"ts_match(returns(5), returns(1), {LOOK}, within='sector')"), name="twin")
    g.add_signal(qex(f"ts_match_score(returns(1), {LOOK}, within='sector')"), name="score")
    assert sum(1 for k in g.nodes if k[0] == "MatchSelection") == 1
    calls = []
    original = MatchSelection._compute
    MatchSelection._compute = lambda self, data: calls.append(1) or original(self, data)
    try:
        with compute_context():
            for e in (f"ts_match(returns(5), returns(1), {LOOK}, within='sector')",
                      f"ts_match_score(returns(1), {LOOK}, within='sector')"):
                qex(e).evaluate(pairs_data)
    finally:
        MatchSelection._compute = original
    assert len(calls) == 1


@pytest.mark.parametrize("expr, message", [
    ("ts_match_score(returns(1), 60, metrc='corr')", "no argument named 'metrc'"),
    ("ts_match_score(returns(1), 60, metric='pearson')", "metric is one of"),
    ("ts_match_score(group('sector'), 60)", "gives labels, not numbers"),
])
def test_bad_score_arguments(expr, message):
    with pytest.raises(ValueError, match=message):
        qex(expr)


# --- metric='path': Gatev, Goetzmann and Rouwenhorst's distance -------------------------------------------------------

def _tracking_pairs(idio):
    """24 stocks in 12 pairs whose prices track each other: a shared return plus `idio` of daily noise each."""
    rng = np.random.default_rng(21)
    T, N = 260, 24
    idx = pd.bdate_range("2023-01-02", periods=T)
    r = np.repeat(rng.normal(0, 0.012, (T, N // 2)), 2, axis=1) + rng.normal(0, idio, (T, N))
    return {"close": pd.DataFrame(30 * np.cumprod(1 + r, axis=0), idx, [f"S{i:02d}" for i in range(N)])}


def test_path_finds_pairs_whose_prices_track_each_other():
    d = _tracking_pairs(0.002)
    p = peers(", metric='path', matching='exclusive'", d).iloc[-1]
    assert (p.to_numpy() == (np.arange(24) ^ 1)).all()


def test_path_twins_are_the_closest_paths():
    """The defining property: each stock's nearest twin is the stock whose path from 100 was closest."""
    d = _tracking_pairs(0.004)                                         # loose pairs: path and correlation differ
    p = peers(", metric='path'", d)
    r = ev("returns(1)", d)
    t = LOOK + 21 * 2
    paths = 100 * (1 + r.iloc[t - LOOK:t].fillna(0)).cumprod()
    valid = r.iloc[t - LOOK:t].notna()
    for i in range(24):
        dist = [np.sqrt(((paths.iloc[:, i] - paths.iloc[:, j])[valid.iloc[:, i] & valid.iloc[:, j]] ** 2).mean())
                if j != i else np.inf for j in range(24)]
        assert p.iloc[t, i] == int(np.argmin(dist))


def test_path_score_is_the_distance_between_paths_in_index_points(pairs_data):
    d = pairs_data
    score = ev(f"ts_match_score(returns(1), {LOOK}, metric='path', min_overlap=40)", d)
    p = peers(", metric='path', min_overlap=40", d)
    r = ev("returns(1)", d)
    t = LOOK + 21 * 3                                                  # a rebalance day
    window = r.iloc[t - LOOK:t]
    for i in range(24):
        j = p.iloc[t, i]
        if j < 0:
            continue
        paths = 100 * (1 + window.iloc[:, [i, j]].fillna(0)).cumprod()
        both = window.iloc[:, [i, j]].notna().all(axis=1)
        expected = np.sqrt(((paths.iloc[:, 0] - paths.iloc[:, 1])[both] ** 2).mean())
        assert score.iloc[t, i] == pytest.approx(expected, abs=1e-9) and score.iloc[t, i] >= 0


def test_path_needs_returns_as_its_key():
    from qex.adjust import check
    fields = {"close": dict(adjustment=1, unit="usd_per_share")}
    assert [f["kind"] for f in check(qex("ts_match(returns(5), close(), 60, metric='path')"), fields, mode="warn")]
