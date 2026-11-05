"""Numbers and labels: an expression that hands labels to an operator needing numbers is refused when it is parsed."""
import pytest

from qex import qex
from qex.kinds import kind, numbers_only


@pytest.mark.parametrize("expr, expected", [
    ("rank(returns(5))", "numbers"),
    ("group('sector')", "labels"),
    ("group_cross('sector', quantile(field('mcap'), 3))", "labels"),
    ("coalesce(group('sector'), 'Other')", "labels"),
    ("where(returns(1) > 0, group('sector'), None)", "labels"),
    ("where(close() > 1, 'high', 'low')", "labels"),
    ("group('sector') == 'Technology'", "numbers"),
    ("group('sector') != 'Technology'", "numbers"),
    ("group_demean(returns(5), group('sector'))", "numbers"),
    ("group_demean(returns(5), coalesce(group('sector'), 'Other'))", "numbers"),
    ("group_quantile(returns(5), group_cross('sector', quantile(field('mcap'), 3)), 3)", "numbers"),
    ("group_demean(returns(5), quantile(field('mcap'), 3))", "numbers"),       # numbers can serve as labels
    ("let s = group('sector') in group_rank(returns(5), s)", "numbers"),
    ("where(group('sector') == 'Financials', None, returns(5))", "numbers"),
    ("coalesce(close(), None)", "numbers"),
    ("None", "either"),
])
def test_what_an_expression_gives(expr, expected):
    assert kind(qex(expr)) == expected


@pytest.mark.parametrize("expr, says", [
    ("rank(group('sector'))", "rank() needs numbers"),
    ("group('sector') + 1", "'+' needs numbers"),
    ("group('sector') * returns(1)", "'*' needs numbers"),
    ("-group('sector')", "'-' needs numbers"),
    ("ts_mean(group('sector'), 5)", "ts_mean() needs numbers"),
    ("group('sector') > 'A'", "'>' needs numbers"),
    ("~group('sector')", "'~' needs numbers"),
    ("group('sector') & (close() > 1)", "'&' needs numbers"),
    ("where(group('sector'), 1, 0)", "the condition of where needs numbers"),
    ("group_demean(group('sector'), 'sector')", "group_demean() needs numbers"),
    ("let s = group('sector') in rank(s)", "rank() needs numbers"),
    ("abs(coalesce(group('sector'), 'x'))", "abs() needs numbers"),
    ("rank(group_cross('sector', quantile(field('mcap'), 3)))", "rank() needs numbers"),
])
def test_labels_where_numbers_are_needed_are_refused(expr, says):
    with pytest.raises(ValueError, match="gives labels, not numbers") as e:
        qex(expr)
    assert says in str(e.value) and "== 'Technology'" in str(e.value)


@pytest.mark.parametrize("expr", ["coalesce(group('sector'), 0)", "where(close() > 1, 'high', 0)"])
def test_labels_and_numbers_do_not_mix(expr):
    with pytest.raises(ValueError, match="mixes labels and numbers"):
        qex(expr)


def test_refused_before_anything_runs():
    """No data is needed to refuse it: the check is on the expression."""
    with pytest.raises(ValueError, match="labels"):
        qex("rank(group('no_such_group'))")


def test_an_alpha_must_give_numbers():
    assert numbers_only(qex("rank(returns(5))")) is None
    assert "cannot be an alpha" in numbers_only(qex("group('sector')"))
