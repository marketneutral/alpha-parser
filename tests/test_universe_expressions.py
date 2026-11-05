"""Universes as qex expressions (cross_section_universe, BookSettings.universe, explain_missing), latch, and
fill_forward with no limit."""
import numpy as np
import pandas as pd
import pytest

from qex import qex, compute_context, cross_section_universe, universe_mask, explain_missing
from evaluation import BookSettings, build_book

N = np.nan
IDX = pd.bdate_range("2024-01-01", periods=12)


def series(values):
    return pd.DataFrame({"A": values}, index=IDX[:len(values)], dtype=float)


# --- latch ---------------------------------------------------------------------------------------------------------

MCAP = [290, 305, 295, 302, 298, 310, 285, 230, 250, 301, 296, 304]     # noise around 300, one real drop


def test_latch_is_hysteresis():
    d = {"mc": series(MCAP)}
    hard = qex("field('mc') >= 300").evaluate(d)["A"].tolist()
    band = qex("latch(field('mc') >= 300, field('mc') < 240)").evaluate(d)["A"].tolist()
    assert hard == [0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1]
    assert band == [0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1]
    flips = lambda x: sum(a != b for a, b in zip(x, x[1:]))
    assert (flips(hard), flips(band)) == (9, 3)


def test_latch_missing_days_change_nothing_and_off_wins_a_tie():
    d = {"on": series([0, 1, N, 0, 1, 0]), "off": series([0, 0, N, N, 1, 0])}
    assert qex("latch(field('on'), field('off'))").evaluate(d)["A"].tolist() == [0, 1, 1, 1, 0, 0]


def test_latch_is_zero_before_the_first_on():
    d = {"on": series([N, N, 0, 1]), "off": series([N, 1, 1, 0])}
    assert qex("latch(field('on'), field('off'))").evaluate(d)["A"].tolist() == [0, 0, 0, 1]


def test_latch_with_a_constant_off_never_resets():
    d = {"on": series([0, 1, 0, 0])}
    assert qex("latch(field('on'), 0)").evaluate(d)["A"].tolist() == [0, 1, 1, 1]


def test_latch_equals_the_fill_forward_idiom():
    d = {"mc": series(MCAP)}
    a = qex("latch(field('mc') >= 300, field('mc') < 240)").evaluate(d)
    b = qex("coalesce(fill_forward(where(field('mc') < 240, 0, where(field('mc') >= 300, 1, None))), 0)").evaluate(d)
    pd.testing.assert_frame_equal(a, b)


def test_fill_forward_without_a_limit():
    d = {"x": series([1, N, N, N, N, N, 2, N])}
    assert qex("fill_forward(field('x'))").evaluate(d)["A"].tolist() == [1, 1, 1, 1, 1, 1, 2, 2]
    assert np.isnan(qex("fill_forward(field('x'), 2)").evaluate(d)["A"].iloc[3])


# --- universe expressions -----------------------------------------------------------------------------------------

@pytest.fixture
def data():
    idx = IDX[:3]
    cols = list("ABCDE")
    close = pd.DataFrame([[10, 20, 30, 40, 50], [11, 19, 31, 39, 51], [12, 18, 1, 38, 52]], index=idx, columns=cols, dtype=float)
    mcap = pd.DataFrame([[100, 200, 300, 400, 500]] * 3, index=idx, columns=cols, dtype=float)
    member = mcap.where(mcap >= 300) * 0 + 1                            # C, D, E as a field: 1 in, NaN out
    return {"close": close, "mcap": mcap, "member": member}


def test_a_universe_is_a_mask_a_field_or_an_expression(data):
    by_mask = universe_mask(data["member"].notna())
    by_field = universe_mask("member", data)
    by_text = universe_mask("field('mcap') >= 300", data)
    by_signal = universe_mask(qex("field('mcap') >= 300"), data)
    for m in (by_field, by_text, by_signal):
        pd.testing.assert_frame_equal(m, by_mask)


def test_cross_sections_compare_the_expression_s_members(data):
    with compute_context(), cross_section_universe("field('mcap') >= 300", data):
        by_expr = qex("rank(close())").evaluate(data)
    with compute_context(), cross_section_universe(data["member"]):
        by_field = qex("rank(close())").evaluate(data)
    pd.testing.assert_frame_equal(by_expr, by_field)
    assert by_expr[["A", "B"]].isna().all().all()


def test_a_universe_is_computed_over_every_stock(data):
    """rank inside the universe expression ranks all five stocks, even inside another universe."""
    with compute_context(), cross_section_universe(data["member"]):
        m = universe_mask("rank(field('mcap')) > 0.5", data)
    assert m.iloc[0].tolist() == [False, False, True, True, True]


def test_a_universe_is_computed_once_per_context(data, monkeypatch):
    from qex.operators import GreaterEqual as Greater
    calls = []
    original = Greater._compute
    monkeypatch.setattr(Greater, "_compute", lambda self, d: calls.append(1) or original(self, d))
    with compute_context():
        for _ in range(3):
            with cross_section_universe("field('mcap') >= 300", data):
                qex("rank(close())").evaluate(data)
    assert len(calls) == 1


def test_the_book_takes_a_universe_expression(data):
    sig = qex("close()")
    s_field = BookSettings(delay=0, universe="member", neutralize="market", truncation=1.0)
    s_expr = BookSettings(delay=0, universe="field('mcap') >= 300", neutralize="market", truncation=1.0)
    s_same = BookSettings(delay=0, universe="field('member')", neutralize="market", truncation=1.0)
    a, b, c = (build_book(sig, data, s) for s in (s_field, s_expr, s_same))
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(a, c)
    assert (a[["A", "B"]] == 0).all().all()


def test_a_zero_is_not_a_member(data):
    """A field holds 1 or NaN; an expression's 0 means out, as its NaN does."""
    m = universe_mask("where(field('mcap') >= 300, 1, 0)", data)
    assert m.iloc[0].tolist() == [False, False, True, True, True]


def test_explain_missing_with_a_universe_expression(data):
    r = explain_missing(qex("rank(close())"), data, universe="field('mcap') >= 300")
    assert r.reason(IDX[0], "A").startswith("outside the universe")


@pytest.mark.parametrize("universe, error, message", [
    ("group('sector')", ValueError, "not labels"),
    ("1 + 1", ValueError, "value on each day"),
    (42, TypeError, "a universe is a mask"),
])
def test_what_a_universe_cannot_be(data, universe, error, message):
    data = dict(data, sector=pd.DataFrame("x", index=data["close"].index, columns=data["close"].columns))
    with pytest.raises(error, match=message):
        universe_mask(universe, data)


def test_an_expression_needs_the_data():
    with pytest.raises(TypeError, match="needs the data"):
        with cross_section_universe("field('mcap') >= 300"):
            pass
