"""The Excel to qex workbook (docs/excel/excel-to-qex.xlsx) is what its builder writes, and its values are qex's.

The committed workbook was calculated by LibreOffice (build_workbook.py --recalc), so it holds each formula's value;
those must match qex evaluating the same steps on the same data. With LibreOffice installed, a fresh build is
calculated and checked too.
"""

import importlib.util
import shutil
from pathlib import Path

import numpy as np
import pytest

from qex import qex

openpyxl = pytest.importorskip("openpyxl")

DOCS = Path(__file__).resolve().parent.parent / "docs"
BOOK = DOCS / "excel" / "excel-to-qex.xlsx"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BUILD = _load("build_workbook", DOCS / "excel" / "build_workbook.py")
TOY = _load("toy", DOCS / "hooks" / "toy.py")

SCORE = "-returns(5) / ts_std(returns(1), 20)"
SHEETS = {
    "Returns": "returns(1)",
    "Return5": "returns(5)",
    "Vol": "ts_std(returns(1), 20)",
    "Score": SCORE,
    "Neutral": f"group_demean({SCORE}, 'sector')",
    "Rank": f"rank(group_demean({SCORE}, 'sector'))",
    "Demean": f"demean(rank(group_demean({SCORE}, 'sector')))",
    "Weights": f"scale(demean(rank(group_demean({SCORE}, 'sector'))))",
    "Contribution": f"delay(scale(demean(rank(group_demean({SCORE}, 'sector')))), 1) * returns(1)",
}
WEIGHTS = SHEETS["Weights"]


def _grid(ws, n_dates, n_stocks):
    def num(v):
        return np.nan if v in (None, "") else float(v)
    return np.array([[num(ws.cell(row=BUILD.FIRST + i, column=2 + j).value) for j in range(n_stocks)]
                     for i in range(n_dates)])


def _check_values(path):
    data = TOY.excel_data()
    wb = openpyxl.load_workbook(path, data_only=True)
    for sheet, expr in SHEETS.items():
        q = qex(expr).evaluate(data).to_numpy()
        x = _grid(wb[sheet], *q.shape)
        assert (np.isfinite(x) == np.isfinite(q)).all(), sheet
        np.testing.assert_allclose(x[np.isfinite(x)], q[np.isfinite(q)], rtol=0, atol=1e-12, err_msg=sheet)


def _check_pnl(path):
    """The PnL sheet is the qex backtest's daily return on GMV, on the days the book is held."""
    from evaluation import Backtest
    data = TOY.excel_data()
    result = Backtest(qex(WEIGHTS)).run(data)
    ws = openpyxl.load_workbook(path, data_only=True)["PnL"]
    rows = range(BUILD.PNL_FIRST, BUILD.FIRST + len(data["close"]))
    excel = np.array([ws[f"B{r}"].value for r in rows], dtype=float)
    # the backtest dates a return by the day its weights were set; the sheet, by the day it is earned
    np.testing.assert_allclose(excel, result.returns.to_numpy(), rtol=0, atol=1e-12)
    summary = {ws[f"E{r}"].value: ws[f"F{r}"].value for r in range(BUILD.FIRST, BUILD.FIRST + len(BUILD.SUMMARY))}
    assert summary["Days"] == len(result.returns)
    assert summary["Return on GMV, total"] == pytest.approx(result.return_gmv, abs=1e-12)
    assert summary["Growth of $1"] == pytest.approx(1 + result.total_return, abs=1e-12)
    assert summary["Sharpe ratio, annualized"] == pytest.approx(result.sharpe, abs=1e-9)


def test_committed_workbook_matches_qex():
    _check_values(BOOK)
    _check_pnl(BOOK)


def test_committed_workbook_is_current(tmp_path):
    fresh = BUILD.build(tmp_path / "fresh.xlsx")
    a, b = openpyxl.load_workbook(fresh), openpyxl.load_workbook(BOOK)
    assert a.sheetnames == b.sheetnames
    for name in a.sheetnames:
        for ra, rb in zip(a[name].iter_rows(), b[name].iter_rows()):
            for ca, cb in zip(ra, rb):
                va, vb = ca.value, cb.value
                if isinstance(va, float) or isinstance(vb, float):
                    assert va == pytest.approx(vb), (name, ca.coordinate)
                elif hasattr(va, "date") or hasattr(vb, "date"):
                    assert str(va)[:10] == str(vb)[:10], (name, ca.coordinate)
                else:
                    assert va == vb, (name, ca.coordinate)


@pytest.mark.skipif(not (shutil.which("soffice") or shutil.which("libreoffice")), reason="needs LibreOffice")
def test_fresh_workbook_calculates_to_qex(tmp_path):
    path = BUILD.recalc(BUILD.build(tmp_path / "fresh.xlsx"))
    _check_values(path)
    _check_pnl(path)


def test_page_pnl_table_matches_the_workbook():
    """The P&L table and summary on the Excel to qex page are the workbook's numbers."""
    import re
    page = (DOCS / "excel-to-qex.md").read_text()
    ws = openpyxl.load_workbook(BOOK, data_only=True)["PnL"]
    for r in range(BUILD.PNL_FIRST, BUILD.FIRST + len(TOY.excel_data()["close"])):
        day, ret, growth = ws[f"A{r}"].value, ws[f"B{r}"].value, ws[f"C{r}"].value
        row = f"| {day:%b %d} | {100 * ret:+.2f}% | {growth:.4f} |".replace("-", "−")
        assert row in page, row
    summary = {ws[f"E{r}"].value: ws[f"F{r}"].value for r in range(BUILD.FIRST, BUILD.FIRST + len(BUILD.SUMMARY))}
    assert f"returned {100 * summary['Return on GMV, total']:.2f}% on its gross" in page
    assert f"({100 * (summary['Growth of $1'] - 1):.2f}% compounded)" in page
    assert f"of {summary['Sharpe ratio, annualized']:.2f}." in page
