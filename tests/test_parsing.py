"""Parsing details: whitespace anywhere, and the ** operator."""

import numpy as np
import pandas as pd
import pytest

from qex import qex


@pytest.fixture
def px():
    rng = np.random.default_rng(3)
    idx = pd.date_range('2022-01-03', periods=80, freq='B')
    return {'close': pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (80, 6)), axis=0)), index=idx,
                                  columns=list('ABCDEF'))}


def test_line_break_after_operator_in_binding(px):
    multi = qex("""
        let a = returns(5) /
                volatility(20),
            b = (a > 0) &
                (a < 1)
        in where(b, a, 0)
    """).evaluate(px)
    one = qex("let a = returns(5) / volatility(20), b = (a > 0) & (a < 1) in where(b, a, 0)").evaluate(px)
    pd.testing.assert_frame_equal(multi, one)


def test_tabs_and_newlines_in_plain_expression(px):
    pd.testing.assert_frame_equal(qex("rank(\n\treturns(5)\n)  -\n 0.5").evaluate(px), qex("rank(returns(5)) - 0.5").evaluate(px))


def test_whitespace_inside_strings_is_kept():
    from qex.parser import _collapse_whitespace
    assert _collapse_whitespace("field('a  b')  *\n 2") == "field('a  b') * 2"
    assert _collapse_whitespace('field("x\ty")') == 'field("x\ty")'


def test_power_operator(px):
    pd.testing.assert_frame_equal(qex("returns(5) ** 2").evaluate(px), qex("power(returns(5), 2)").evaluate(px))
    pd.testing.assert_frame_equal(qex("returns(5) ** 0.5").evaluate(px), qex("power(returns(5), 0.5)").evaluate(px))


def test_power_with_signal_exponent_raises():
    with pytest.raises(TypeError, match="must be a number"):
        qex("returns(5) ** returns(1)")
