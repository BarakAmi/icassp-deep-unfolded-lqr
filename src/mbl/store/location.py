"""Where the store is, when nobody says.

The command line has always resolved `--store`, else `$MBL_STORE`, else
`./store`; Stage 6 gave a notebook the same question to answer, and a notebook
that answered it differently — or that answered it by carrying a directory name
in a cell — is the second, laxer surface this architecture keeps refusing to
grow.

The discovery itself lives in `mbl.locate` and is shared with the studies
directory, because it is one question asked twice and answering it twice is
exactly how the two came to disagree: `studies_root` was taught to search and
this was not, and the author found the consequence — a notebook reporting a
complete study as **0 of 135 measured**, because `./store` from a Jupyter
kernel is a directory two levels below the one holding the results.

This module is Tier 4, so both the command line and the replay seam reach it
without either reaching the other. That matters: Tier 8's own docstring records
why it does not import `cli.run` — the top tier and the command line are peers
over one pipeline, and an import would put argparse behind every notebook.
"""

from __future__ import annotations

from pathlib import Path

from ..locate import project_directory

#: Environment variable naming the store root. Annex 05 §2.3 puts the store
#: outside every worktree at a fixed absolute path, so that several worktrees
#: share one store and none retrains what another already produced; this is how
#: that path is declared.
STORE_ENV = "MBL_STORE"

#: The directory's name. On a fresh clone nothing exists to discover and the
#: store is created at the PROJECT ROOT rather than beside the working
#: directory -- which is the fix for the surprise
#: `docs/methods/mbl_command_line.md` lists as "a store appears where you did
#: not expect one".
DEFAULT_STORE = Path("store")


def default_store() -> Path:
    """The store root to use when none was given.

    Returns:
        `$MBL_STORE` if it is set; else the `store/` of the project the working
        directory is inside; else the one belonging to the repository this
        package was imported from. The first that exists, or the first named,
        which is where a run creates it.
    """
    return project_directory(DEFAULT_STORE.name, env=STORE_ENV)
