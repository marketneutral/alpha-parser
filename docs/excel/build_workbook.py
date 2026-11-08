"""Build excel-to-qex.xlsx: the Excel to qex page's signal, one sheet per step, as Excel formulas.

    scale(demean(rank(group_demean(-returns(5) / ts_std(returns(1), 20), 'sector'))))

The prices are the page's dataset (docs/hooks/toy.py: excel_data). Every sheet has dates down column A from row 3,
the six stocks across columns B to G, and the sectors in row 2. Run from the repository root:

    python docs/excel/build_workbook.py             # writes docs/excel/excel-to-qex.xlsx
    python docs/excel/build_workbook.py --recalc    # and has LibreOffice calculate it, so the file opens with values

Needs openpyxl (and LibreOffice for --recalc). tests/test_excel.py checks the calculated workbook against qex.
"""

import argparse
import importlib.util
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "excel-to-qex.xlsx"
FIRST = 3                                   # the first date's row
STD_WINDOW, RET_PERIOD = 20, 5

# (sheet, first row with a value, formula for column `c` and row `r`, number format, what it is)
STEPS = [
    ("Returns", FIRST + 1, lambda c, r: f"=Prices!{c}{r}/Prices!{c}{r - 1}-1", "0.0000",
     "returns(1): today's close over yesterday's, minus 1"),
    ("Return5", FIRST + RET_PERIOD, lambda c, r: f"=Prices!{c}{r}/Prices!{c}{r - RET_PERIOD}-1", "0.0000",
     "returns(5): today's close over the close 5 rows up, minus 1"),
    ("Vol", FIRST + STD_WINDOW, lambda c, r: f"=_xlfn.STDEV.S(Returns!{c}{r - STD_WINDOW + 1}:{c}{r})", "0.0000",
     "ts_std(returns(1), 20): the sample standard deviation of the last 20 daily returns"),
    ("Score", FIRST + STD_WINDOW, lambda c, r: f"=-Return5!{c}{r}/Vol!{c}{r}", "0.00",
     "-returns(5) / ts_std(returns(1), 20): last week's loss, in daily standard deviations"),
    ("Neutral", FIRST + STD_WINDOW,
     lambda c, r: f"=Score!{c}{r}-AVERAGEIFS(Score!$B{r}:$G{r},Prices!$B$2:$G$2,Prices!{c}$2)", "0.00",
     "group_demean(score, 'sector'): the score minus its sector's average that day"),
    ("Rank", FIRST + STD_WINDOW,
     lambda c, r: f"=_xlfn.RANK.AVG(Neutral!{c}{r},Neutral!$B{r}:$G{r},1)/COUNT(Neutral!$B{r}:$G{r})", "0.000",
     "rank(...): the day's percentile, 1/n to 1, ties at their average rank"),
    ("Demean", FIRST + STD_WINDOW, lambda c, r: f"=Rank!{c}{r}-AVERAGE(Rank!$B{r}:$G{r})", "0.000",
     "demean(...): the rank minus the day's average rank; positive is long, negative is short"),
    ("Weights", FIRST + STD_WINDOW,
     lambda c, r: f"=Demean!{c}{r}/SUMPRODUCT(ABS(Demean!$B{r}:$G{r}))", "0.000",
     "scale(...): the weights, divided by the day's total absolute value so long and short add up to 1"),
    ("Contribution", FIRST + STD_WINDOW + 1, lambda c, r: f"=Weights!{c}{r - 1}*Returns!{c}{r}", "0.0000",
     "What each position earned: yesterday's weight times today's return"),
]

# The PnL sheet: one row per day, summary on the right
PNL_FIRST = FIRST + STD_WINDOW + 1          # the first day with a return earned by the book
SUMMARY = [
    ("Days", lambda a, b: f"=COUNT({a}:{b})", "0"),
    ("Return on GMV, total", lambda a, b: f"=SUM({a}:{b})", "0.00%"),
    ("Growth of $1", None, "0.0000"),
    ("Average daily return", lambda a, b: f"=AVERAGE({a}:{b})", "0.000%"),
    ("Daily volatility", lambda a, b: f"=_xlfn.STDEV.S({a}:{b})", "0.000%"),
    ("Sharpe ratio, annualized", None, "0.00"),
]


def _data():
    spec = importlib.util.spec_from_file_location("toy", HERE.parent / "hooks" / "toy.py")
    toy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(toy)
    return toy.excel_data()


def build(path=OUT):
    from openpyxl import Workbook
    from openpyxl.styles import Font
    from openpyxl.utils import get_column_letter

    data = _data()
    close, sectors = data["close"], data["sector"].iloc[0]
    stocks = list(close.columns)
    cols = [get_column_letter(2 + j) for j in range(len(stocks))]
    last = FIRST + len(close) - 1
    bold, grey = Font(bold=True), Font(italic=True, color="808080")

    wb = Workbook()
    about = wb.active
    about.title = "About"
    lines = [
        "Excel to qex",
        "",
        "scale(demean(rank(group_demean(-returns(5) / ts_std(returns(1), 20), 'sector'))))",
        "",
        "Each sheet is one step of the signal. Dates run down column A from row 3, stocks across columns B to G,",
        "and row 2 holds each stock's sector. Each formula is written once and filled across the grid.",
        "",
    ] + [f"{name}: {what}" for name, _, _, _, what in STEPS] + [
        "",
        "Contribution and PnL: the P&L the weights earn. That part is a backtest's job, not the qex language's.",
        "",
        "From data to signal to weights, the whole workbook is the one line above.",
    ]
    for i, line in enumerate(lines, start=1):
        about.cell(row=i, column=1, value=line)
    about["A1"].font = Font(bold=True, size=14)
    about["A3"].font = Font(name="Consolas")
    about.column_dimensions["A"].width = 110

    def sheet(name):
        ws = wb.create_sheet(name)
        ws["A1"], ws["A2"] = "Date", "Sector"
        ws["A1"].font = ws["A2"].font = bold
        for j, (c, s) in enumerate(zip(cols, stocks)):
            ws[f"{c}1"] = s
            ws[f"{c}1"].font = bold
            ws[f"{c}2"] = sectors[s] if name == "Prices" else f"=Prices!{c}2"
            ws[f"{c}2"].font = grey
            ws.column_dimensions[c].width = 11
        for i, d in enumerate(close.index):
            ws.cell(row=FIRST + i, column=1, value=d.to_pydatetime().date()).number_format = "yyyy-mm-dd"
        ws.column_dimensions["A"].width = 12
        ws.freeze_panes = "B3"
        return ws

    prices = sheet("Prices")
    for i, d in enumerate(close.index):
        for c, s in zip(cols, stocks):
            cell = prices[f"{c}{FIRST + i}"]
            cell.value = float(close.loc[d, s])
            cell.number_format = "0.00"

    for name, start, formula, fmt, _ in STEPS:
        ws = sheet(name)
        for r in range(start, last + 1):
            for c in cols:
                ws[f"{c}{r}"] = formula(c, r)
                ws[f"{c}{r}"].number_format = fmt

    pnl = wb.create_sheet("PnL")
    for col, head in zip("ABC", ("Date", "Return on GMV", "Growth of $1")):
        pnl[f"{col}1"] = head
        pnl[f"{col}1"].font = bold
    pnl["A2"] = "Today's return on the book: the positions' P&L over the gross held"
    pnl["A2"].font = grey
    for i, d in enumerate(close.index):
        pnl.cell(row=FIRST + i, column=1, value=d.to_pydatetime().date()).number_format = "yyyy-mm-dd"
    span = f"{cols[0]}{{r}}:{cols[-1]}{{r}}"
    for r in range(PNL_FIRST, last + 1):
        pnl[f"B{r}"] = (f"=SUM(Contribution!{span.format(r=r)})"
                        f"/SUMPRODUCT(ABS(Weights!{span.format(r=r - 1)}))")
        pnl[f"B{r}"].number_format = "0.00%"
        pnl[f"C{r}"] = f"=(1+B{r})" if r == PNL_FIRST else f"=C{r - 1}*(1+B{r})"
        pnl[f"C{r}"].number_format = "0.0000"
    a, b = f"B{PNL_FIRST}", f"B{last}"
    for i, (label, formula, fmt) in enumerate(SUMMARY, start=FIRST):
        pnl[f"E{i}"] = label
        pnl[f"E{i}"].font = bold
        if label == "Growth of $1":
            pnl[f"F{i}"] = f"=C{last}"
        elif label.startswith("Sharpe"):
            pnl[f"F{i}"] = f"=F{i - 2}/F{i - 1}*SQRT(252)"
        else:
            pnl[f"F{i}"] = formula(a, b)
        pnl[f"F{i}"].number_format = fmt
    for col, width in zip("ABCDEF", (12, 15, 13, 3, 26, 12)):
        pnl.column_dimensions[col].width = width
    pnl.freeze_panes = "B3"

    wb.calculation.fullCalcOnLoad = True
    wb.save(path)
    return path


def recalc(path=OUT):
    """Have LibreOffice calculate the workbook and save it with values, in place."""
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        raise RuntimeError("LibreOffice (soffice) not found")
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([soffice, "--headless", f"-env:UserInstallation=file://{tmp}/profile", "--calc",
                        "--convert-to", "xlsx:Calc MS Excel 2007 XML", "--outdir", f"{tmp}/out", str(path)],
                       check=True, capture_output=True, timeout=180)
        shutil.copy(Path(tmp) / "out" / Path(path).name, path)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--recalc", action="store_true", help="calculate the workbook with LibreOffice")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    p = build(a.out)
    if a.recalc:
        recalc(p)
    print(p)
