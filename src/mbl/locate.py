"""Where this project's directories are, when nobody says.

One rule, in one place, because the same question is asked about two of them —
`store/` and `studies/` — and they have already been answered differently. The
studies directory was taught to search and the store was not, and the author
found both halves in one evening: first a notebook that could not locate its
study document, then a notebook that located the document and reported the
study **0 of 135 measured** against a complete store.

That second failure is the shape this module exists to prevent. A path resolved
against the wrong directory is not usually *malformed* — it is **absent**, and
absence is indistinguishable from "not produced yet". So the refusal was a
perfectly clear message telling the reader to run a command they had already
run.

**The working directory cannot be the only anchor.** It is right for exactly two
callers: the command line invoked from the project root, and a Jupyter kernel
whose notebook happens to sit at the root. A kernel's working directory is the
*notebook's own*, which for this project is two levels down.

**Nor can the directory's name be the anchor.** The first fix here walked up
looking for a directory *called* `store` or `studies`, and this repository
contains six such directories that are not either of those things:
`src/mbl/store/` and `tests/store/` are Python packages, `src/mbl/applications/studies/`
and `tests/applications/studies/` are more of them, and — the one that settles
it — **every store contains a `studies/` of its own**, because
`store/layout.py` files artifacts under `store/studies/<StudyID>/`. Name
matching is not merely fragile here, it is structurally unsound: a working
directory inside any store resolves the studies directory into that store.

So the anchor is the **project root**, found by its marker. That is the idiom
this repository already uses (`experiments/notebook_bootstrap.find_project_root`),
and it is immune to every namesake by construction.
"""

from __future__ import annotations

import os
from pathlib import Path

#: What identifies a project root. The same marker
#: `experiments/notebook_bootstrap.py` has always used; spelled again here
#: rather than imported, because that module belongs to the tier the deletion
#: ledger retires and this one must outlive it.
ROOT_MARKER = "pyproject.toml"

#: The repository this package was imported from — for an editable install, the
#: checkout the author is working in. `locate.py` sits at `src/mbl/locate.py`,
#: so the root is three parents up. When the package is installed somewhere
#: with no repository around it the directories below simply do not exist, and
#: the search falls through.
PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def project_root() -> Path | None:
    """The nearest directory at or above the working directory holding a
    project marker.

    Returns:
        The project root, or `None` when the working directory is not inside
        one at all — standing in `/tmp`, or in a store kept outside every
        worktree as Annex 05 §2.3 prescribes.
    """
    here = Path.cwd().resolve()
    return next(
        (
            directory
            for directory in (here, *here.parents)
            if (directory / ROOT_MARKER).is_file()
        ),
        None,
    )


def candidates(name: str) -> list[Path]:
    """Every directory called `name` that could be meant, in priority order.

    The project you are standing in first, the one this package was imported
    from second. The order is load-bearing in both directions:

    * **Standing-in first**, because a second worktree must read its own
      results. A search that reached past it would find a real store, resolve,
      and agree with itself completely — the silent direction.
    * **The package's repository second**, so that a caller outside any project
      — a notebook opened from elsewhere, a script in `/tmp` — still finds the
      one the package came from rather than nothing at all.

    Args:
        name: The directory's name, e.g. `store` or `studies`.

    Returns:
        At least one candidate, which may not exist. Never empty, so a refusal
        always has somewhere to point.
    """
    roots = [root for root in (project_root(), PACKAGE_ROOT) if root is not None]
    return [root / name for root in dict.fromkeys(roots)]


def project_directory(name: str, *, env: str) -> Path:
    """The directory called `name`, discovered.

    Args:
        name: The directory's name.
        env: An environment variable that overrides the search outright. Not
            merely first in the order — the *whole* answer, so that naming a
            store on a machine that has not built one yet says where it will go
            rather than being silently overruled by one that happens to exist.

    Returns:
        The first candidate that exists; else the first candidate, which is
        where a run would create it and what a refusal can name. There is no
        separate fallback: a defaulted path that no caller could reach would be
        a declaration taking no effect.
    """
    declared = os.environ.get(env)
    if declared:
        return Path(declared)
    options = candidates(name)
    return next((path for path in options if path.is_dir()), options[0])
