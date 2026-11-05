"""Three-valued logic: a comparison with a missing side is missing, & | ~ follow Kleene, and where with a missing
condition is missing. A missing value is not knowing; 0 would be a confident "no"."""
import numpy as np
import pandas as pd
import pytest

from qex import qex
from qex.missing import explain_missing

IDX = ["day1", "day2", "day3"]


@pytest.fixture
def data():
    close = pd.DataFrame({"A": [10.0, np.nan, 12.0], "B": [5.0, 6.0, np.nan]}, index=IDX)   # A missing day2, B day3
    sector = pd.DataFrame({"A": ["tech", "tech", None], "B": ["fin", "fin", "fin"]}, index=IDX)
    return {"close": close, "sector": sector}


def values(expr, data):
    return qex(expr).evaluate(data).to_numpy(dtype=float).tolist()


N = np.nan


def same(a, b):
    return np.array_equal(np.array(a, float), np.array(b, float), equal_nan=True)


@pytest.mark.parametrize("expr, expected", [
    ("close() > 8", [[1, 0], [N, 0], [1, N]]),
    ("close() <= 8", [[0, 1], [N, 1], [0, N]]),
    ("close() == 6", [[0, 0], [N, 1], [0, N]]),
    ("close() != 6", [[1, 1], [N, 0], [1, N]]),
    ("~(close() > 8)", [[0, 1], [N, 1], [0, N]]),
])
def test_a_comparison_with_a_missing_side_is_missing(data, expr, expected):
    assert same(values(expr, data), expected)


def test_kleene_and_or(data):
    # known false wins an &, known true wins an |; otherwise a missing side leaves the answer missing
    assert same(values("is_valid(close()) & (close() > 8)", data), [[1, 0], [0, 0], [1, 0]])
    assert same(values("(close() > 0) & (close() < 100)", data), [[1, 1], [N, 1], [1, N]])
    assert same(values("(close() > 8) | (close() > 0)", data), [[1, 1], [N, 1], [1, N]])
    assert same(values("(close() > 8) | ~is_valid(close())", data), [[1, 0], [1, 0], [1, 1]])


def test_where_with_a_missing_condition_is_missing(data):
    assert same(values("where(close() > 8, 1, -1)", data), [[1, -1], [N, -1], [1, N]])
    # the guard written before this change still gives the same answer
    assert same(values("where(is_valid(close()), close() > 8, None)", data), [[1, 0], [N, 0], [1, N]])


def test_group_labels_compare_the_same_way(data):
    assert same(values("group('sector') == 'tech'", data), [[1, 0], [1, 0], [N, 0]])


def test_numbers_compare_as_numbers():
    assert qex("2 > 1").evaluate({}) == 1.0 and qex("1 > 2").evaluate({}) == 0.0


def test_a_missing_condition_passes_on_its_reason(data):
    dated = {k: v.set_axis(pd.bdate_range("2020-01-01", periods=3)) for k, v in data.items()}
    report = explain_missing(qex("where(close() > 8, 1, -1)"), dated)
    assert report.reason("2020-01-02", "A") == "no data, at close()"


def test_coalesce_says_what_missing_means(data):
    """coalesce(x, y): x where it has a value, else y; how to ask for a 0 where a comparison is not known."""
    assert same(values("coalesce(close() > 8, 0)", data), [[1, 0], [0, 0], [1, 0]])
    assert same(values("where(close() > 8, 1, 0)", data), [[1, 0], [N, 0], [1, N]])     # where stays missing
    data = dict(data, open=pd.DataFrame({"A": [9.0, 9.5, np.nan], "B": [4.0, 6.5, 7.0]}, index=IDX))
    assert same(values("coalesce(close(), open())", data), [[10, 5], [9.5, 6], [12, 7]])
    assert same(values("coalesce(close(), None)", data), values("close()", data))
    assert qex("coalesce(None, 3)").evaluate({}) == 3
