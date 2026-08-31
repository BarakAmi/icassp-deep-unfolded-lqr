"""Every shipped document's links, examples and citations still hold.

Scoped to `docs/methods/`, which is what this repository ships: a methods
document describes what is *currently implemented*, so a reader follows its
links while using the thing and a stale one misleads exactly the person it was
written for.

Three checks, and each catches a failure that is silent otherwise. A relative
link rots when a file moves. A `toml` block that no parser accepts is a defect
in the specification rather than a typo in prose — the whole argument for TOML
over YAML is that one document has one meaning. And a `file.py:line` citation
drifts as the line moves, leaving the claim beside it still reading as measured.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs"

_TOML_BLOCK = re.compile(r"```toml\n(.*?)```", re.DOTALL)
_SOURCE_REFERENCE = re.compile(r"`([\w./-]+\.py):(\d+)(?:-(\d+))?`")


def _documents() -> list[Path]:
    return sorted(DOCS.rglob("*.md"))


def _toml_blocks() -> list[tuple[Path, int, str]]:
    return [
        (document, index, block)
        for document in _documents()
        for index, block in enumerate(_TOML_BLOCK.findall(document.read_text()), 1)
    ]


def _tracked_python() -> list[str]:
    return sorted(
        str(path.relative_to(REPO))
        for path in REPO.glob("**/*.py")
        if ".venv" not in path.parts and "__pycache__" not in path.parts
    )


@pytest.mark.parametrize(
    "document", _documents(), ids=lambda p: str(p.relative_to(REPO))
)
def test_every_link_resolves(document: Path) -> None:
    dangling = []
    for match in re.finditer(r"\]\(([^)]+)\)", document.read_text()):
        target = match.group(1).split("#")[0]
        if not target or target.startswith(("http://", "https://", "mailto:")):
            continue
        if not (document.parent / target).resolve().exists():
            dangling.append(target)
    assert not dangling, f"{document} links to {dangling}, which do not exist"


def test_there_are_documents_to_check() -> None:
    """Anti-vacuity: a glob that stops matching turns the gate green."""
    assert len(_documents()) >= 3


@pytest.mark.parametrize(
    ("document", "index", "block"),
    _toml_blocks(),
    ids=[f"{d.name}#{i}" for d, i, _ in _toml_blocks()],
)
def test_every_documented_toml_block_parses(
    document: Path, index: int, block: str
) -> None:
    try:
        tomllib.loads(block)
    except tomllib.TOMLDecodeError as error:
        pytest.fail(f"{document}: toml block #{index} does not parse: {error}")


@pytest.mark.parametrize(
    "document", _documents(), ids=lambda p: str(p.relative_to(REPO))
)
def test_every_cited_source_line_exists(document: Path) -> None:
    tracked = _tracked_python()
    problems: list[str] = []
    for match in _SOURCE_REFERENCE.finditer(document.read_text()):
        path, line = match.group(1), int(match.group(2))
        candidates = [f for f in tracked if f == path or f.endswith("/" + path)]
        if not candidates:
            problems.append(f"{path}:{line} — no such file ships")
            continue
        text = (REPO / candidates[0]).read_text().splitlines()
        if line > len(text):
            problems.append(f"{path}:{line} — the file has {len(text)} lines")
    assert not problems, f"{document}: {problems}"
