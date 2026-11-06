"""The adjustment check: which expressions use an adjusted level in a way that sees the future."""
import pytest

from qex import qex
from qex.adjust import LookAhead, check, exponent

FIELDS = {
    "open": dict(adjustment=1, unit="usd_per_share"), "high": dict(adjustment=1, unit="usd_per_share"),
    "low": dict(adjustment=1, unit="usd_per_share"), "close": dict(adjustment=1, unit="usd_per_share"),
    "volume": dict(adjustment=-1, unit="shares"), "price": dict(adjustment=0, unit="usd_per_share"),
    "volume_traded": dict(adjustment=0, unit="shares"), "shares": dict(adjustment=0, unit="shares"),
    "mcap": dict(adjustment=0, unit="usd"), "market": dict(adjustment=1, unit="usd_per_share"),
    "sector": dict(adjustment=None, unit="label"),
}

REFUSED = [
    "close() > 5",
    "5 < close()",
    "rank(close())",
    "rank(-close())",
    "zscore(close())",
    "rank(log(close()))",
    "demean(open())",
    "quantile(close(), 5)",
    "group_rank(close(), 'sector')",
    "group_demean(close(), 'sector')",
    "regression_neut(rank(returns(5)), close())",
    "field('price') * volume(20)",
    "field('price') - close()",
    "where(close() > ts_mean(close(), 20), close(), field('price'))",
    "rank(ts_mean(close(), 20))",
    "rank(delta(close(), 5))",
    "rank(ts_std(close(), 20))",
    "log(close()) > 3",
    "field('price') + field('shares')",
    "scale(close())",
]

FINE = [
    "returns(5)",
    "rank(returns(5))",
    "volatility(63)",
    "close() / ts_max(close(), 252)",
    "rank(close() / ts_max(close(), 252))",
    "close() > ts_mean(close(), 200)",
    "-returns(5)",
    "rank(close() * volume(20))",
    "close() * volume(20)",
    "adv(20) > 5e6",
    "field('price') > 5",
    "rank(field('mcap'))",
    "rank(log(field('price') * field('shares')))",
    "log(close()) - log(delay(close(), 5))",
    "rank(delta(log(close()), 21))",
    "ts_rank(close(), 252)",
    "rank(ts_corr(close(), volume(20), 63))",
    "group_rank(returns(21), 'sector')",
    "rank(-field('price'))",
    "rank(field('volume_traded') * field('price'))",
    "ts_beta(returns(1), returns(1, 'market'), 252)",
    "rank(close() / delay(close(), 21) - 1)",
    "is_valid(close()) & (field('price') >= 2)",
    "rank(field('unknown_field'))",                       # an undeclared field is not checked
    "delay(field('mcap'), 1) + 0 * returns(1)",            # a plain number adds to dollars
]


@pytest.mark.parametrize("expr", REFUSED)
def test_refused(expr):
    with pytest.raises(LookAhead):
        check(qex(expr), FIELDS)


@pytest.mark.parametrize("expr", FINE)
def test_fine(expr):
    assert check(qex(expr), FIELDS) == []


def test_warn_mode_returns_the_problems_and_runs():
    found = check(qex("rank(close()) + (close() > 5)"), FIELDS, mode="warn")
    kinds = [f["kind"] for f in found]
    assert "cross-section of an adjusted level" in kinds and "adjusted level against a number" in kinds
    assert all(f["message"] for f in found)


def test_the_message_says_what_to_write_instead():
    with pytest.raises(LookAhead, match=r"field\('price'\)"):
        check(qex("close() > 5"), FIELDS)
    with pytest.raises(LookAhead, match="rank\\(\\) compares close\\(\\) across stocks"):
        check(qex("rank(close())"), FIELDS)


def test_off_and_no_declarations_check_nothing():
    assert check(qex("rank(close())"), FIELDS, mode="off") == []
    assert check(qex("rank(close())"), {}) == []


def test_exponents():
    assert exponent(qex("close()"), FIELDS) == 1
    assert exponent(qex("close() * volume(20)"), FIELDS) == 0
    assert exponent(qex("ts_mean(close(), 20)"), FIELDS) == 1
    assert exponent(qex("close() / delay(close(), 5)"), FIELDS) == 0
    assert exponent(qex("volume(20)"), FIELDS) == -1
    assert exponent(qex("sqrt(close() * close())"), FIELDS) == 1
    assert exponent(qex("field('unknown')"), FIELDS) is None


def test_an_alpha_that_is_an_adjusted_level_is_refused_but_a_part_may_be():
    with pytest.raises(LookAhead, match="an alpha's value is compared across stocks"):
        check(qex("close()"), FIELDS)
    assert check(qex("ts_mean(close(), 20)"), FIELDS, alpha=False) == []      # e.g. a cell that only looks


def test_let_bindings_are_followed():
    with pytest.raises(LookAhead):
        check(qex("let p = close(), q = p * 2 in rank(q)"), FIELDS)
    assert check(qex("let p = close(), r = p / delay(p, 5) in rank(r)"), FIELDS) == []
