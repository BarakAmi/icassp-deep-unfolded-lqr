"""The `mbl` command line -- the store's user interface (decision D8).

Read-only commands are safe by construction; every destructive command requires
confirmation or `--yes`.

`main` is imported lazily so that `import mbl.cli` costs nothing: the entry
point pays only for what the invoked subcommand actually touches.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .app import main

__all__ = ["main"]


def __getattr__(name: str) -> Any:
    if name == "main":
        from .app import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
