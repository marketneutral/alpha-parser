"""The language reference shipped in the package (src/qex/docs/, written by scripts/build_agent_docs.py) is complete,
its links resolve, it is the language only, and it is what the generator writes from docs/language/ today."""
import importlib.util
import re
from pathlib import Path

import pytest

from qex.parser import QexParser

ROOT = Path(__file__).resolve().parent.parent
SHIPPED = ROOT / "src" / "qex" / "docs"


def _gen():
    spec = importlib.util.spec_from_file_location("build_agent_docs", ROOT / "scripts" / "build_agent_docs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GEN = _gen()
PAGES = sorted(SHIPPED.rglob("*.md"))


def _anchors(page):
    text = re.sub(r"^```.*?^```", "", page.read_text(), flags=re.S | re.M)
    return {GEN.slug(h) for h in re.findall(r"^#{1,6} (.+)$", text, re.M)}


def test_every_operator_has_a_linked_line():
    text = (SHIPPED / "OPERATORS.md").read_text()
    linked = dict(re.findall(r"^- \[`([a-z_0-9]+)(?:\([^`]*\))?`\]\((ops/[^)]+)\): \S", text, re.M))
    missing = sorted(set(QexParser().functions) - set(linked))
    assert missing == [], f"operators with no linked line in OPERATORS.md: {missing}"
    for name, target in linked.items():
        page, _, anchor = target.partition("#")
        assert anchor, f"{name} links to a page, not to its entry"


@pytest.mark.parametrize("page", PAGES, ids=lambda p: str(p.relative_to(SHIPPED)))
def test_links_resolve(page):
    text = re.sub(r"^```.*?^```", "", page.read_text(), flags=re.S | re.M)
    for target in re.findall(r"\]\(([^)\s]+)\)", text):
        if re.match(r"[a-z]+:", target):
            continue
        path, _, anchor = target.partition("#")
        dest = (page.parent / path).resolve() if path else page
        assert dest.is_file(), f"{page.name}: {target} is not a shipped page"
        assert SHIPPED.resolve() in dest.parents or dest.parent == SHIPPED.resolve(), f"{page.name}: {target} leaves the docs"
        if anchor:
            assert anchor in _anchors(dest), f"{page.name}: {target} has no such heading"


def test_operators_is_one_line_each():
    """No tables and no toy data: a heading, a sentence of context, then one list line per operator."""
    text = (SHIPPED / "OPERATORS.md").read_text()
    assert not re.search(r"^\|", text, re.M), "a table row in OPERATORS.md"
    assert "Jan 02" not in text, "toy data in OPERATORS.md"
    lines = [l for l in text.splitlines() if l.startswith("- [`")]
    assert len(lines) >= len(QexParser().functions)


@pytest.mark.parametrize("page", PAGES, ids=lambda p: str(p.relative_to(SHIPPED)))
def test_plain_markdown_only(page):
    text = page.read_text()
    assert "```toy" not in text and "!!! " not in text, "a site-only construct left in"


@pytest.mark.parametrize("page", PAGES, ids=lambda p: str(p.relative_to(SHIPPED)))
def test_math_renders_on_github(page):
    """GitHub's math refuses some LaTeX macros and shows an error in their place: \\operatorname is one."""
    assert "\\operatorname" not in page.read_text(), "use \\mathrm{name} instead"


def test_operators_is_the_language_only():
    text = (SHIPPED / "OPERATORS.md").read_text()
    for word in ("```python", "import qex", "build_book", "BookSettings", "LazyData"):
        assert word not in text


def test_shipped_docs_are_up_to_date(tmp_path):
    written = GEN.build(tmp_path)
    shipped = {p.relative_to(SHIPPED).as_posix() for p in PAGES}
    assert shipped == set(written), "pages added or removed: run scripts/build_agent_docs.py"
    stale = [rel for rel, text in written.items() if (SHIPPED / rel).read_text() != text]
    assert stale == [], f"out of date, run scripts/build_agent_docs.py: {stale}"


def test_installed_package_finds_them():
    import qex
    here = Path(qex.__file__).parent / "docs"
    assert (here / "OPERATORS.md").is_file() and (here / "ops" / "time-series.md").is_file()
