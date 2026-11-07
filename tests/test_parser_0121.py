"""qex 0.12.1: # comments end at the line's end, clear errors for what qex doesn't have, and argument checks."""

import numpy as np
import pandas as pd
import pytest

from qex import qex
from qex.missing import label
from qex.parser import _collapse_whitespace


# --- comments ----------------------------------------------------------------------------------

@pytest.mark.parametrize("src, reads", [
    ("rank(returns(5))  # reversal\n * -1", "rank(returns(5)) * -1"),
    ("let a = returns(1),   # one day\n    b = returns(5)      # a week\nin a - b", "returns(1) - returns(5)"),
    ("# a comment line first\nrank(returns(5))", "rank(returns(5))"),
    ("rank(returns(5))   # a comment on the last line", "rank(returns(5))"),
    ("group('sector') == 'A#1'   # a # inside quotes is kept", "group('sector') == 'A#1'"),
    ('group("sector") == "B#2"', "group('sector') == 'B#2'"),
])
def test_a_comment_ends_at_the_end_of_its_line(src, reads):
    assert label(qex(src)) == reads


def test_collapse_whitespace_strips_comments_first():
    # qexarena's notebook joins cells with this function: a comment must not swallow the lines after it
    assert _collapse_whitespace("a  # x\n + b") == "a + b"
    assert _collapse_whitespace("'#1' # c\n'#2'") == "'#1' '#2'"


def test_comments_change_no_cache_key():
    assert qex("rank(returns(5))  # note")._cache_key() == qex("rank(returns(5))")._cache_key()


# --- clear errors --------------------------------------------------------------------------------

@pytest.mark.parametrize("src, message", [
    ("returns(5) and returns(1)", "'and' is not qex: write &"),
    ("returns(5) or returns(1)", "'or' is not qex: write |"),
    ("not returns(1)", "'not' is not qex: write ~x"),
    ("returns(1) % 2", "'%' is not a qex operator"),
    ("returns(1) // 2", "'//' is not a qex operator"),
    ("ts_mean(returns(1), period=20)", "takes its arguments by position"),
    ("[1, 2]", "a list"),
    ("returns(1) if 1 else 2", "use where"),
    ("returns(1).x", "an attribute"),
    ("returns(1)[0]", "indexing"),
    ("x.rank(1)", "Only operators are called"),
])
def test_what_qex_does_not_have_says_so(src, message):
    with pytest.raises(ValueError, match=message.replace("(", r"\(").replace("|", r"\|")):
        qex(src)


def test_unary_plus_is_a_no_op():
    assert qex("+returns(5)")._cache_key() == qex("returns(5)")._cache_key()


# --- argument checks -----------------------------------------------------------------------------

@pytest.mark.parametrize("src, message", [
    ("returns('5')", r"returns\(\)'s first argument must be a whole number of days"),
    ("returns(0)", "at least 1"),
    ("returns(5, 3)", "a name in quotes"),
    ("volatility(2.5)", "whole number of days"),
    ("delay(returns(1), 1.5)", "0 or more"),
    ("delay(returns(1), -1)", "Negative delays cause lookahead bias"),   # delay's own message
    ("ts_mean(returns(1), 0)", "at least 1"),
    ("ts_mean(returns(1), returns(1))", r"\(got an expression\)"),
    ("quantile(returns(1), 0)", "a whole number, at least 1"),
    ("truncate(returns(1), -1)", "a number above 0"),
    ("winsorize(returns(1), 0.6)", "from 0 to 0.5"),
    ("ewma(returns(1), 0)", "above 0"),
    ("field(5)", "a name in quotes"),
    ("fill_forward(returns(1), 0)", "at least 1"),
    ("ts_corr(returns(1), returns(2), 60, 0)", "fourth argument"),
    ("ts_mean_events(returns(1), 0)", "at least 1"),
    ("pca_factor(returns(1), 252, 0)", "k must"),                         # pca_factor's own message
])
def test_bad_arguments_are_refused_when_read(src, message):
    with pytest.raises(ValueError, match=message):
        qex(src)


@pytest.mark.parametrize("src", [
    "delay(returns(1), 0)", "ts_mean(returns(1), 20.0)", "winsorize(returns(1), 0.5)", "winsorize(returns(1), 0)",
    "returns(1, 'market')", "fill_forward(returns(1))", "ts_corr(returns(1), returns(5), 60, 30)", "truncate(rank(returns(1)), 0.05)",
    "ewma(returns(1), 0.5)", "quantile(returns(1), 1)",
])
def test_good_arguments_pass(src):
    qex(src)


def test_values_pass_through_unchanged():
    # 20.0 stays 20.0: qexarena hashes an expression's cache key as its identity
    assert qex("ts_mean(returns(1), 20.0)")._cache_key() == ("TsMean", ("Returns", 1, "close"), 20.0, 20)
    assert repr(qex("ts_mean(returns(1), 20.0)")._cache_key()) != repr(qex("ts_mean(returns(1), 20)")._cache_key())


# --- the risk model's liquidity factor -----------------------------------------------------------

def test_liquidity_factor_is_dollar_volume():
    pytest.importorskip("statsmodels")
    from qex import DEFAULT_STYLE_FACTORS, PRICE_ONLY_FACTORS
    for factors in (DEFAULT_STYLE_FACTORS, PRICE_ONLY_FACTORS):
        liq = next(f for f in factors if f.name == "liquidity")
        assert liq.expression == "rank(adv(20))"
        assert "dollar volume" in liq.description
