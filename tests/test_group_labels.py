"""Group operators take a group's name or any expression giving labels; group_quantile sorts within groups (a
dependent double sort); group_cross combines groupings; explain_missing reads label nodes."""
import numpy as np
import pandas as pd
import pytest

from qex import qex, explain_missing, SignalGraph

N = np.nan
COLS = list("ABCDEFGHI")
DAY = "2024-01-01"


@pytest.fixture
def data():
    idx = pd.bdate_range(DAY, periods=1)
    f = lambda row: pd.DataFrame([row], index=idx, columns=COLS, dtype=float)
    return {
        "close": f([9, 1, 5, 2, 8, 3, 7, 4, 6]),
        "value": f([9, 1, 5, 2, 8, 3, 7, 4, 6]),
        "size": f([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        "sector": pd.DataFrame([["t", "t", "f", "f", "t", "f", "t", None, "t"]], index=idx, columns=COLS),
    }


def row(expr, data):
    return qex(expr).evaluate(data).iloc[0].tolist()


def same(a, b):
    return np.array_equal(np.array(a, dtype=object).astype(str), np.array(b, dtype=object).astype(str))


def test_a_label_expression_groups_like_a_named_group(data):
    data = dict(data, bucket=qex("quantile(field('size'), 3)").evaluate(data))
    for op in ("group_demean", "group_rank", "group_sum"):
        by_name = qex(f"{op}(field('value'), 'bucket')").evaluate(data)
        by_expr = qex(f"{op}(field('value'), quantile(field('size'), 3))").evaluate(data)
        pd.testing.assert_frame_equal(by_expr, by_name)
    assert np.allclose(row("let b = quantile(field('size'), 3) in group_demean(field('value'), b)", data),
                       row("group_demean(field('value'), 'bucket')", data))


def test_group_quantile_is_a_dependent_double_sort(data):
    """Value terciles within size terciles, against the same sort written out with where and None."""
    dep = row("group_quantile(field('value'), quantile(field('size'), 3), 3)", data)
    by_hand = row("""let b = quantile(field('size'), 3), x = field('value')
        in where(b == 1, quantile(where(b == 1, x, None), 3),
           where(b == 2, quantile(where(b == 2, x, None), 3), quantile(where(b == 3, x, None), 3)))""", data)
    assert dep == by_hand == [3, 1, 2, 1, 3, 2, 3, 1, 2]
    assert row("quantile(field('value'), 3)", data) == [3, 1, 2, 1, 3, 1, 3, 2, 2]   # independent: F and H differ


def test_group_quantile_by_name_matches_quantile_within_each_sector(data):
    out = row("group_quantile(field('value'), 'sector', 2)", data)
    assert np.allclose(out, [2, 1, 2, 1, 2, 2, 2, N, 1], equal_nan=True)


def test_group_cross_joins_labels(data):
    labels = row("group_cross('sector', quantile(field('size'), 3))", data)
    assert same(labels, ["t|1", "t|1", "f|1", "f|2", "t|2", "f|2", "t|3", N, "t|3"])
    out = row("group_demean(field('value'), group_cross('sector', quantile(field('size'), 3)))", data)
    assert np.allclose(out, [4, -4, 0, -0.5, 0, 0.5, 0.5, N, -0.5], equal_nan=True)


def test_a_missing_label_can_be_named(data):
    out = row("group_demean(field('value'), coalesce(group('sector'), 'other'))", data)
    assert out[7] == 0.0                                       # H, with no sector, is the only 'other'
    assert np.isnan(row("group_demean(field('value'), 'sector')", data)[7])


def test_explain_missing_says_which_grouping_had_no_label(data):
    data = dict(data, size=data["size"].where(data["size"] != 9))     # I has no size
    expr = qex("group_demean(field('value'), group_cross('sector', quantile(field('size'), 3)))")
    report = explain_missing(expr, data)
    assert report.reason(DAY, "H") == "no group, at group('sector')"
    assert report.reason(DAY, "I") == "no data, at field('size')"


def test_explain_missing_reads_a_graph_with_labels(data):
    """Before 0.8.0 any graph with group('...') in it failed: the labels were read as numbers."""
    report = explain_missing(qex("where(group('sector') == 't', close(), None)"), data)
    assert report.reason(DAY, "C") == "excluded on purpose, at where(group('sector') == 't', close(), None)"


def test_the_graph_shows_a_label_expression_as_an_input():
    g = SignalGraph()
    kids = g._get_children(qex("group_demean(field('v'), quantile(field('size'), 3))"))
    assert [type(k).__name__ for k in kids] == ["Field", "Quantile"]
    kids = g._get_children(qex("group_cross('sector', quantile(field('size'), 3))"))
    assert [type(k).__name__ for k in kids] == ["Group", "Quantile"]


def test_the_cache_tells_groupings_apart(data):
    from qex import compute_context
    with compute_context():
        a = qex("group_demean(field('value'), quantile(field('size'), 2))").evaluate(data)
        b = qex("group_demean(field('value'), quantile(field('size'), 3))").evaluate(data)
    assert not a.equals(b)


@pytest.mark.parametrize("expr, error, message", [
    ("group_demean(close(), 3)", TypeError, "group's name"),
    ("group_demean(close(), 1 + 1)", ValueError, "label on each day"),
    ("group_quantile(close(), 'sector', 0)", ValueError, "whole number of buckets"),
    ("group_cross('sector')", ValueError, "two or more"),
])
def test_clear_errors(data, expr, error, message):
    with pytest.raises(error, match=message):
        qex(expr).evaluate(data)
