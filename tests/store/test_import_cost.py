"""The store's read path must not import PyTorch, pandas or NumPy.

`mbl models list` answers from `index.sqlite` and never opens a checkpoint, so
paying several seconds of tensor-library import on every invocation would make
the command line unpleasant enough not to be used -- which defeats the point of
building it (decision D8).

This is checked in a **subprocess**: the test session has already imported
torch by the time any test runs, so an in-process `sys.modules` check would pass
no matter what the import graph does. Measured before the fix, `import
mbl.store` cost 4.34 s against a 0.02 s interpreter baseline.
"""

from __future__ import annotations

import subprocess
import sys

#: Every module that reads the store without touching a checkpoint. Listed
#: exhaustively rather than by representative: the laziness is a property of the
#: package's `__getattr__` plus two deferred imports, and a regression in any one
#: of them would be invisible if only one entry point were probed.
READ_PATH = (
    "import mbl.store",
    "from mbl.store import StoreIndex",
    "from mbl.store import ModelRow, MeasurementRow",
    "from mbl.store import ModelID, ProblemID, semantic_name",
    "import mbl.store.index",
    "import mbl.store.ids",
    "import mbl.store.naming",
    "import mbl.cli",
    "import mbl.cli.render",
    "from mbl.cli.app import build_parser",
)

HEAVY = ("torch", "pandas", "numpy")


def _loaded(statement: str) -> set[str]:
    """Which heavy libraries `statement` drags in, in a fresh interpreter."""
    probe = (
        f"import sys\n{statement}\n"
        f"print(' '.join(m for m in {HEAVY!r} if m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    return set(result.stdout.split())


def test_no_read_path_import_pulls_a_tensor_library() -> None:
    offenders = {
        statement: loaded for statement in READ_PATH if (loaded := _loaded(statement))
    }
    assert not offenders, f"read-path imports pulled heavy libraries: {offenders}"


def test_the_probe_detects_a_heavy_import() -> None:
    """Negative control. Without this, a broken probe -- a typo in the module
    names, a subprocess whose output is never read -- would report every import
    as clean and the check above would be worthless."""
    assert _loaded("from mbl.store import ModelStore") >= {"torch"}
