"""Corner cases: group labels from expressions, group_quantile, group_cross, and three-valued logic at the edges."""
import numpy as np
import pandas as pd
import pytest

from qex import qex, compute_context, explain_missing
from qex.universe import cross_section_universe

N = np.nan
IDX = pd.bdate_range("2024-01-01", periods=3)
COLS = list("ABCDEF")


def F(rows, cols=COLS, idx=IDX):
    return pd.DataFrame(rows, index=idx[:len(rows)], columns=cols, dtype=object if any(
        isinstance(v, str) for r in rows for v in r) else float)


@pytest.fixture
def data():
    return {
        "close": F([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1], [1, 1, 2, 2, 3, 3]]),
        "size":  F([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1], [1, 2, 3, 4, 5, N]]),
        "sector": F([["a", "a", "a", "b", "b", "b"]] * 3),
    }


def ev(expr, data):
    return qex(expr).evaluate(data)


def close_to(a, b):
    return np.allclose(np.asarray(a, float), np.asarray(b, float), equal_nan=True)


# --- labels from expressions -------------------------------------------------------------------------------------------

def test_labels_that_change_from_day_to_day(data):
    """Day 1 small names are A B C, day 2 they are D E F: each day groups by that day's labels."""
    out = ev("group_demean(close(), quantile(field('size'), 2))", data)
    assert close_to(out.iloc[0], [-1, 0, 1, -1, 0, 1])
    assert close_to(out.iloc[1], [1, 0, -1, 1, 0, -1])


def test_a_named_group_and_group_of_the_name_agree(data):
    a, b = ev("group_rank(close(), 'sector')", data), ev("group_rank(close(), group('sector'))", data)
    pd.testing.assert_frame_equal(a, b)


def test_a_condition_is_a_grouping(data):
    """close() > 2 labels each stock 1 or 0 (and missing where close is): demean within each side."""
    out = ev("group_demean(close(), close() > 2)", data)
    assert close_to(out.iloc[0], [-0.5, 0.5, -1.5, -0.5, 0.5, 1.5])


def test_float_and_integer_labels_are_one_group():
    idx = IDX[:1]
    d = {"close": F([[1, 2, 3, 4]], list("ABCD"), idx), "g": pd.DataFrame([[1, 1.0, 2, 2.0]], index=idx, columns=list("ABCD"))}
    assert close_to(ev("group_demean(close(), 'g')", d).iloc[0], [-0.5, 0.5, -0.5, 0.5])


def test_a_label_of_zero_is_a_label():
    idx = IDX[:1]
    d = {"close": F([[1, 3, 10, 20]], list("ABCD"), idx), "g": F([[0, 0, 1, 1]], list("ABCD"), idx)}
    assert close_to(ev("group_demean(close(), 'g')", d).iloc[0], [-1, 1, -5, 5])


def test_labels_with_fewer_dates_take_their_last_row(data):
    """As for a named group: a date missing from the labels takes their last row."""
    d = dict(data, sector=data["sector"].iloc[:1])
    pd.testing.assert_frame_equal(ev("group_demean(close(), 'sector')", d), ev("group_demean(close(), 'sector')", data))


def test_a_stock_missing_from_the_labels_has_no_group(data):
    d = dict(data, sector=data["sector"].drop(columns="F"))
    out = ev("group_demean(close(), 'sector')", d)
    assert out["F"].isna().all() and out["E"].notna().all()


def test_a_missing_value_with_a_label_is_missing_and_does_not_move_its_group(data):
    d = dict(data, close=data["close"].where(~((data["close"].index == IDX[0])[:, None] & (np.array(COLS) == "C"))))
    out = ev("group_demean(close(), 'sector')", d)
    assert close_to(out.iloc[0], [-0.5, 0.5, N, -1, 0, 1])


def test_the_universe_leaves_non_members_out_of_labels_and_groups(data):
    member = F([[1, 1, 1, 1, N, N]] * 3)
    with compute_context(), cross_section_universe(member):
        out = ev("group_demean(close(), quantile(field('size'), 2))", data)
    assert close_to(out.iloc[0], [-0.5, 0.5, -0.5, 0.5, N, N])        # buckets of the four members only


def test_time_series_group_operators_take_label_expressions(data):
    by_name = dict(data, bucket=ev("quantile(field('size'), 2)", data))
    for op in ("group_std(close(), {g}, 2)", "group_count_valid(close(), {g}, 2)"):
        a = ev(op.format(g="'bucket'"), by_name)
        b = ev(op.format(g="quantile(field('size'), 2)"), data)
        pd.testing.assert_frame_equal(a, b)


def test_an_unknown_group_name_says_so(data):
    with pytest.raises(ValueError, match="Group 'industry' not found"):
        ev("group_demean(close(), 'industry')", data)


def test_a_constant_is_not_a_grouping(data):
    with pytest.raises(ValueError, match="label on each day"):
        ev("group_demean(close(), 2 * 3)", data)


# --- group_quantile ----------------------------------------------------------------------------------------------------

def test_one_bucket_is_everyone(data):
    out = ev("group_quantile(close(), 'sector', 1)", data)
    assert (out == 1).all().all()


def test_a_group_of_one_is_in_the_top_bucket():
    """As quantile: a lone stock's percentile is 1, so it is in bucket n."""
    idx = IDX[:1]
    d = {"close": F([[1, 2, 3, 9]], list("ABCD"), idx), "g": pd.DataFrame([["x", "x", "x", "y"]], index=idx, columns=list("ABCD"))}
    assert close_to(ev("group_quantile(close(), 'g', 3)", d).iloc[0], [1, 2, 3, 3])


def test_ties_share_a_bucket():
    idx = IDX[:1]
    d = {"close": F([[1, 1, 2, 2]], list("ABCD"), idx), "g": pd.DataFrame([["x"] * 4], index=idx, columns=list("ABCD"))}
    out = ev("group_quantile(close(), 'g', 2)", d).iloc[0]
    assert out["A"] == out["B"] and out["C"] == out["D"] and out["A"] < out["C"]


def test_more_buckets_than_stocks_skips_buckets(data):
    out = ev("group_quantile(close(), 'sector', 10)", data).iloc[0]
    assert close_to(out, [4, 7, 10, 4, 7, 10])


def test_group_quantile_by_sector_is_quantile_within_each_sector(data):
    out = ev("group_quantile(close(), 'sector', 3)", data)
    by_hand = ev("where(group('sector') == 'a', quantile(where(group('sector') == 'a', close(), None), 3),"
                 " quantile(where(group('sector') == 'b', close(), None), 3))", data)
    pd.testing.assert_frame_equal(out, by_hand)


@pytest.mark.parametrize("n", [0, -1, 2.5, True])
def test_bucket_counts_are_whole_and_positive(n):
    with pytest.raises(ValueError, match="whole number"):
        qex(f"group_quantile(close(), 'sector', {n})")


# --- group_cross -------------------------------------------------------------------------------------------------------

def test_three_groupings(data):
    labels = ev("group_cross('sector', quantile(field('size'), 2), close() > 2)", data).iloc[0].tolist()
    assert labels == ["a|1|0", "a|1|0", "a|1|1", "b|2|1", "b|2|1", "b|2|1"]


def test_a_missing_part_leaves_the_stock_without_a_label(data):
    labels = ev("group_cross('sector', quantile(field('size'), 2))", data).iloc[2]
    assert pd.isna(labels["F"]) and labels.drop("F").notna().all()


def test_crossing_with_itself_is_the_same_grouping(data):
    a = ev("group_demean(close(), group_cross('sector', 'sector'))", data)
    pd.testing.assert_frame_equal(a, ev("group_demean(close(), 'sector')", data))


def test_a_label_with_the_separator_is_refused():
    idx = IDX[:1]
    d = {"close": F([[1, 2]], ["A", "B"], idx), "g": pd.DataFrame([["x|y", "z"]], index=idx, columns=["A", "B"]),
         "h": pd.DataFrame([["p", "q"]], index=idx, columns=["A", "B"])}
    with pytest.raises(ValueError, match="cannot contain one"):
        ev("group_cross('g', 'h')", d)


def test_cross_labels_compare_as_strings(data):
    out = ev("group_cross('sector', quantile(field('size'), 2)) == 'b|2'", data).iloc[0]
    assert close_to(out, [0, 0, 0, 1, 1, 1])


def test_explain_missing_through_group_quantile(data):
    report = explain_missing(qex("group_quantile(close(), quantile(field('size'), 2), 2)"), data)
    assert report.reason(IDX[2], "F") == "no data, at field('size')"


# --- three-valued logic at the edges -----------------------------------------------------------------------------------

def test_scalars_and_none():
    assert np.isnan(qex("None > 1").evaluate({}))
    assert qex("coalesce(None, None)").evaluate({}) is None or np.isnan(qex("coalesce(None, None)").evaluate({}))
    assert qex("(1 > 2) & None").evaluate({}) == 0.0
    assert qex("(2 > 1) | None").evaluate({}) == 1.0
    assert np.isnan(qex("(2 > 1) & None").evaluate({}))


def test_a_frame_against_a_missing_scalar_is_missing(data):
    assert ev("close() > None", data).isna().all().all()
    assert (ev("(close() > 100) & None", data) == 0).all().all()               # known false wins


def test_coalesce_aligns_a_fallback_with_fewer_names(data):
    close = data["close"].copy()
    close.loc[IDX[0], ["A", "C"]] = N                                  # A has a fallback, C does not
    d = dict(data, close=close, other=data["close"][["A", "B"]] * 100)
    out = ev("coalesce(close(), field('other'))", d)
    assert out.shape == close.shape
    assert out.loc[IDX[0], "A"] == 100.0 and np.isnan(out.loc[IDX[0], "C"]) and out.loc[IDX[0], "B"] == 2.0


def test_where_with_a_label_condition(data):
    out = ev("where(group('sector') == 'a', close(), None)", data).iloc[0]
    assert close_to(out, [1, 2, 3, N, N, N])
