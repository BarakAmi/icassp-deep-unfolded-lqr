#!/usr/bin/env python3
"""Fail if any tracked notebook carries embedded outputs (decision D9).

Committed outputs are what put `.git` on a 16 GB growth curve: notebooks do
not delta-compress, so every commit of an executed notebook stores a fresh
multi-megabyte blob. Git tracks the output-free source; the executed copy
with figures is a build artifact.

Deliberately dependency-free -- it runs in the `static` CI job, which
installs no project dependencies so that style and hygiene fail in seconds
rather than after a multi-gigabyte wheel download.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

#: Fields whose presence means a cell was executed and its result stored.
DIRTY_CELL_FIELDS = ("outputs", "execution_count")


def tracked_notebooks() -> list[Path]:
    """Every `.ipynb` git knows about, relative to the repository root."""
    listing = subprocess.run(
        ["git", "ls-files", "*.ipynb"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [Path(line) for line in listing.stdout.splitlines() if line]


def dirty_cell_count(notebook: Path) -> int:
    """How many cells in `notebook` carry outputs or an execution count.

    A malformed notebook counts as dirty rather than clean: the check must
    never pass because a file could not be parsed.
    """
    try:
        document = json.loads(notebook.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"  {notebook}: unreadable ({error})", file=sys.stderr)
        return 1
    return sum(
        1
        for cell in document.get("cells", [])
        if any(cell.get(field) for field in DIRTY_CELL_FIELDS)
    )


def main() -> int:
    offenders = [
        (notebook, count)
        for notebook in tracked_notebooks()
        if (count := dirty_cell_count(notebook))
    ]
    if not offenders:
        print("All tracked notebooks are output-free.")
        return 0

    print(
        "Notebooks with embedded outputs (D9 forbids committing these):\n",
        file=sys.stderr,
    )
    for notebook, count in offenders:
        print(f"  {notebook}  ({count} executed cells)", file=sys.stderr)
    print(
        "\nStrip them before committing:\n"
        "    uvx nbstripout <notebook>...\n"
        "or install the pre-commit hooks, which do it automatically:\n"
        "    uvx pre-commit install",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
