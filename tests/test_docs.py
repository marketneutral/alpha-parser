"""The documentation stays true to the code.

- every qex function has an entry in the operator reference (parsed as qexarena's lang.py parses it);
- every ```qex block parses: a block is one or more expressions separated by blank lines, and each is either one
  multi-line expression or a list of one-line expressions, one per line (a `#` comment may end a line);
- every ```python block is valid Python, and every qex("...") string literal in it parses;
- the worked pages run end to end on synthetic data;
- every ```toy block in the operator reference evaluates on the toy data (docs/hooks/toy.py renders them at build).
"""

import ast
import contextlib
import textwrap
import io
import re
from pathlib import Path

import importlib.util

import pytest

from qex import qex
from qex.parser import QexParser

DOCS = Path(__file__).resolve().parent.parent / "docs"
PAGES = sorted(DOCS.rglob("*.md"))


def _blocks(lang):
    for page in PAGES:
        for m in re.finditer(rf"```{lang}\n(.*?)```", page.read_text(), re.S):
            line = page.read_text()[:m.start()].count("\n") + 2
            yield pytest.param(textwrap.dedent(m.group(1)), id=f"{page.relative_to(DOCS)}:{line}")


def _strip_comment(line):
    """A trailing `# comment` outside quotes."""
    quote = None
    for i, ch in enumerate(line):
        if ch in "'\"":
            quote = None if quote == ch else (quote or ch)
        elif ch == "#" and quote is None:
            return line[:i].rstrip()
    return line.rstrip()


def test_every_operator_is_in_the_reference():
    names = QexParser().functions
    found = set()
    for line in (DOCS / "language" / "reference.md").read_text().splitlines():
        m = re.match(r"\|\s*`([a-z_]+)(\([^`]*\))?`\s*\|\s*(.*?)\s*\|\s*$", line)      # qexarena/lang.py's rule
        if m and m.group(1) in names and m.group(3):
            found.add(m.group(1))
    assert sorted(set(names) - found) == []


@pytest.mark.parametrize("block", list(_blocks("qex")))
def test_qex_block_parses(block):
    assert block.strip(), "an empty qex block"
    for chunk in re.split(r"\n\s*\n", block.strip()):
        lines = [_strip_comment(l) for l in chunk.split("\n")]
        lines = [l for l in lines if l.strip()]
        if not lines:
            continue
        whole = "\n".join(lines)
        try:
            qex(whole)
        except Exception:
            for line in lines:                       # a list of one-line expressions
                qex(line)


@pytest.mark.parametrize("block", list(_blocks("python")))
def test_python_block_is_python_and_its_qex_parses(block):
    tree = ast.parse(block)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("qex", "alpha")
                and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
            qex(node.args[0].value)


def _run(page, setup=""):
    ns = {}
    exec(setup, ns)
    for block in re.findall(r"```python\n(.*?)```", (DOCS / page).read_text(), re.S):
        with contextlib.redirect_stdout(io.StringIO()):
            exec(textwrap.dedent(block), ns)
    return ns


def _complete_dataset():
    blocks = re.findall(r"```python\n(.*?)```", (DOCS / "data" / "format.md").read_text(), re.S)
    return "from qex import qex\n" + next(b for b in blocks if "tickers = [" in b)


@pytest.mark.parametrize("page, setup", [
    ("python-examples/quickstart.md", ""),
    ("python-examples/example-backtests.md", "COMPLETE"),
])
def test_worked_pages_run(page, setup):
    _run(page, _complete_dataset() if setup == "COMPLETE" else setup)


def _toy():
    spec = importlib.util.spec_from_file_location("toy", DOCS / "hooks" / "toy.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TOY = _toy()


@pytest.mark.parametrize("block, inputs", [pytest.param(b, f, id=f"{p.relative_to(DOCS)}:{i}")
                                           for p in PAGES for i, (b, f) in enumerate(TOY.blocks(p.read_text()))])
def test_toy_block_renders(block, inputs):
    out = TOY.render(block, inputs)
    assert out.count('class="toy-panel') == len([l for l in block.splitlines() if l.strip()])


def test_every_family_page_has_an_entry_for_each_of_its_operators():
    names = set(QexParser().functions)
    entries = set()
    for page in (DOCS / "language" / "ops").glob("*.md"):
        entries |= set(re.findall(r"^### ([a-z_]+)\s*$", page.read_text(), re.M))
    sig = set()
    for page in (DOCS / "language" / "ops").glob("*.md"):
        sig |= set(re.findall(r"<code>([a-z_]+)\(", page.read_text()))
    assert sorted(names - entries - sig) == []
