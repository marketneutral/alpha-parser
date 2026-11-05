"""The None literal, and explain_missing: why each missing cell is missing, and where it began."""
import numpy as np
import pandas as pd
import pytest

from qex import compute_context, cross_section_universe, qex
from qex.missing import explain_missing

IDX = pd.bdate_range("2020-01-01", periods=12)
COLS = ["A", "B", "C", "D"]


@pytest.fixture
def data():
    close = pd.DataFrame(np.linspace(10, 20, 48).reshape(12, 4), IDX, COLS)
    close.loc[IDX[5], "B"] = np.nan                       # B halted one day
    close.loc[:IDX[3], "D"] = np.nan                      # D lists on day 4
    member = pd.DataFrame(1.0, IDX, COLS)
    member.loc[:IDX[7], "C"] = np.nan                     # C joins the universe on day 8
    sector = pd.DataFrame("tech", IDX, COLS)
    sector["A"] = np.nan                                  # A has no sector
    return {"close": close, "member": member, "sector": sector}


def evaluate(expr, data):
    with compute_context(), cross_section_universe(data["member"].notna()):
        return qex(expr).evaluate(data)


# --- the None literal --------------------------------------------------------------------------------------------------

def test_none_is_missing_on_purpose(data):
    v = evaluate("where(close() > 15, close(), None)", data)
    assert v[data["close"] <= 15].isna().all().all() and (v == data["close"]).where(data["close"] > 15).fillna(True).all().all()
    assert evaluate("close() + None", data).isna().all().all()                       # arithmetic stays missing
    assert evaluate("is_valid(where(close() > 15, close(), None))", data).where(data["close"] <= 15).fillna(0).eq(0).all().all()
    ranked = evaluate("rank(where(close() > 15, close(), None))", data)            # a cross-section leaves it out
    assert ranked.where(data["close"] <= 15).isna().all().all()
    assert qex("where(close() > 2, close(), None)")._cache_key()[-1] == ("None",)   # a node of its own
    assert qex("ts_mean(None, 3)")._cache_key()[1] == ("None",)                     # also as an argument


def test_none_differs_from_zero_in_a_book(data):
    """The case that motivates it: 0 is a position after demeaning, None is not."""
    with_none = evaluate("demean(where(close() > 15, close(), None))", data)
    with_zero = evaluate("demean(where(close() > 15, close(), 0))", data)
    low = (data["close"] <= 15) & data["member"].notna()
    assert with_none[low].isna().all().all() and with_zero[low].notna().any().any()


# --- explain_missing -----------------------------------------------------------------------------------------------------

def why(expr, data, day, name):
    return explain_missing(qex(expr), data, universe=data["member"].notna()).reason(IDX[day], name)


def test_each_reason_where_it_begins(data):
    assert why("close()", data, 5, "B") == "no data, at close()"
    assert why("rank(close())", data, 2, "C") == "outside the universe, at rank(close())"
    assert why("where(close() > 15, close(), None)", data, 0, "A").startswith("excluded on purpose, at where(")
    assert why("group_demean(close(), 'sector')", data, 9, "A") == "no group, at group_demean(close(), 'sector')"
    assert why("log(close() - 15)", data, 0, "A") == "calculation error, at log(close() - 15)"
    assert why("close() / (close() - close())", data, 9, "A").startswith("infinite value")
    assert why("returns(3)", data, 1, "A") == "short history, at returns(3)"
    assert why("rank(close())", data, 9, "C") == ""                                # a value: no reason


def test_a_window_says_why_its_history_is_short(data):
    """The new member: a time series of a cross-section is blank while its window fills, because the stock was
    outside the universe on the earlier days."""
    assert why("ts_mean(rank(returns(1)), 3)", data, 8, "C") == \
        "short history: outside the universe on earlier days, at ts_mean(rank(returns(1)), 3)"
    assert why("ts_mean(rank(returns(1)), 3)", data, 5, "D").startswith("short history: no data on earlier days")
    assert why("delay(rank(close()), 1)", data, 8, "C").startswith("short history: outside the universe")


def test_an_input_missing_today_passes_on_its_reason(data):
    assert why("ts_mean(close(), 3) + rank(close())", data, 2, "C") == "outside the universe, at rank(close())"
    assert why("-close() * 2", data, 5, "B") == "no data, at close()"


@pytest.mark.parametrize("expr", [
    "close()", "returns(3)", "rank(returns(1))", "ts_mean(rank(returns(1)), 3)", "delay(rank(close()), 2)",
    "delta(close(), 2)", "group_demean(close(), 'sector')", "where(close() > 15, close(), None)",
    "log(close() - 15)", "close() / (close() - close())", "zscore(ts_sum(returns(1), 3))", "fill_forward(close(), 1)",
    "ts_corr(close(), returns(1), 4)", "let r = returns(1) in demean(r) / ts_std(r, 3)",
])
def test_a_reason_exactly_where_a_value_is_missing(data, expr):
    report = explain_missing(qex(expr), data, universe=data["member"].notna())
    value = evaluate(expr, data).reindex_like(report.codes).to_numpy(dtype=float)
    assert np.array_equal(report.codes.to_numpy() != 0, ~np.isfinite(value))
    assert ((report.origins.to_numpy() >= 0) & (report.origins.to_numpy() < len(report.nodes))).all()


def test_explaining_changes_nothing(data):
    before = evaluate("ts_mean(rank(returns(1)), 3)", data)
    explain_missing(qex("ts_mean(rank(returns(1)), 3)"), data, universe=data["member"].notna())
    assert before.equals(evaluate("ts_mean(rank(returns(1)), 3)", data))


def test_summary_and_one_day(data):
    report = explain_missing(qex("ts_mean(rank(returns(1)), 3)"), data, universe=data["member"].notna())
    shares = report.summary(within=data["member"].notna())
    assert abs(shares.sum() - 1) < 1e-12 and "short history: outside the universe on earlier days" in shares.index
    day = report.on(IDX[8])
    assert list(day.columns) == ["reason", "at"] and "C" in day.index
