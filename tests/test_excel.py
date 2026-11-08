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
}


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


def test_committed_workbook_matches_qex():
    _check_values(BOOK)


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
