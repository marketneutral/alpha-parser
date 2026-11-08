"""gaussrank and group_gaussrank (0.16.0): the day's rank mapped through the inverse normal."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from qex import qex, cross_section_universe


@pytest.fixture
def panel():
    rng = np.random.default_rng(8)
    T, N = 40, 12
    idx = pd.bdate_range("2024-01-01", periods=T)
    names = [f"S{i:02d}" for i in range(N)]
    x = pd.DataFrame(rng.standard_t(2, (T, N)), idx, names)          # heavy tails
    x.iloc[5, 3] = np.nan
    x.iloc[7, [1, 2]] = 0.25                                          # a tie
    x.iloc[9, 4] = np.inf
    sector = pd.DataFrame([[("A", "B", "C")[i % 3] for i in range(N)]] * T, idx, names)
    return {"close": x.abs() + 10, "x": x, "sector": sector}


def _expected(row):
    """Normal scores by first-come rank, then averaged over each set of tied values."""
    v = row.dropna()
    scores = pd.Series(norm.ppf((v.rank(method="first") - 0.5) / len(v)), index=v.index)
    return scores.groupby(v).transform("mean").reindex(row.index)


def test_gaussrank_is_the_inverse_normal_of_the_shifted_rank(panel):
    g = qex("gaussrank(field('x'))").evaluate(panel)
    for t in range(len(g)):
        np.testing.assert_allclose(g.iloc[t].to_numpy(), pd.Series(_expected(panel["x"].iloc[t]),
                                   index=g.columns).to_numpy(), atol=1e-12, equal_nan=True)


def test_ties_share_a_score_and_missing_names_are_left_out(panel):
    g = qex("gaussrank(field('x'))").evaluate(panel)
    assert g.iloc[7, 1] == g.iloc[7, 2]
    assert abs(g.iloc[7].sum()) < 1e-12                                 # a tie keeps the day summing to zero
    assert np.isnan(g.iloc[5, 3]) and g.iloc[5].notna().sum() == 11
    assert g.iloc[9, 4] == g.iloc[9].max()                              # an infinity ranks last


def test_every_day_sums_to_zero_and_ignores_scale(panel):
    g = qex("gaussrank(field('x'))").evaluate(panel)
    assert np.abs(g.sum(axis=1)).max() < 1e-12
    pd.testing.assert_frame_equal(g, qex("gaussrank(3 * field('x') + 1)").evaluate(panel))
    finite = panel["x"].where(np.isfinite(panel["x"]))
    d = dict(panel, x=finite)
    pd.testing.assert_frame_equal(qex("gaussrank(field('x'))").evaluate(d),
                                  qex("gaussrank(exp(field('x') / 10))").evaluate(d))   # any increasing transform


def test_the_universe_limits_who_is_ranked(panel):
    members = pd.DataFrame(True, index=panel["x"].index, columns=panel["x"].columns)
    members.iloc[:, :4] = False
    with cross_section_universe(members):
        g = qex("gaussrank(field('x'))").evaluate(panel)
    assert g.iloc[:, :4].isna().all().all()
    alone = qex("gaussrank(field('x'))").evaluate({k: (v.iloc[:, 4:] if isinstance(v, pd.DataFrame) else v)
                                                    for k, v in panel.items()})
    pd.testing.assert_frame_equal(g.iloc[:, 4:], alone)


def test_group_gaussrank_is_gaussrank_within_each_group(panel):
    g = qex("group_gaussrank(field('x'), 'sector')").evaluate(panel)
    for sector in ("A", "B", "C"):
        cols = [c for c in panel["x"].columns if panel["sector"].iloc[0][c] == sector]
        within = qex("gaussrank(field('x'))").evaluate({**panel, "x": panel["x"][cols], "close": panel["close"][cols]})
        np.testing.assert_allclose(g[cols].to_numpy(), within.to_numpy(), atol=1e-12, equal_nan=True)


def test_group_edge_cases(panel):
    sector = panel["sector"].copy()
    sector["S00"] = "alone"
    sector["S01"] = np.nan
    g = qex("group_gaussrank(field('x'), 'sector')").evaluate(dict(panel, sector=sector))
    assert (g["S00"].dropna() == 0).all()                               # a group of one: the middle
    assert g["S01"].isna().all()                                        # no group: no value


@pytest.mark.parametrize("expr", ["gaussrank(group('sector'))", "group_gaussrank(group('sector'), 'sector')"])
def test_labels_are_refused(expr):
    with pytest.raises(ValueError, match="gives labels, not numbers"):
        qex(expr)
