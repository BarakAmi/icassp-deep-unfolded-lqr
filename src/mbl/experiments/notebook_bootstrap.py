"""One-line environment bootstrap for research notebooks (NB03 overhaul,
Phase 1.2): the inline-backend registration, console-log configuration, and
project-root/`sys.path` wiring that every experiment notebook opens with,
factored out of the per-notebook boilerplate block into a single reusable
call so no notebook re-authors that infrastructure (and infrastructure code
stops cluttering the research narrative).

    from mbl.experiments.notebook_bootstrap import bootstrap_experiment_notebook
    PROJECT_ROOT = bootstrap_experiment_notebook()

The console log format deliberately omits the logger NAME (``%(name)s``): a
per-epoch training row must read ``Epoch: 1 | Cost (Loss): ...``, never
``src.engine.callbacks: Epoch: ...`` -- the module path is noise in a
narrated training trace. The per-run ``experiment.log`` file
(`experiments.run_logging`) is written regardless of this console handler.
"""

import logging
import sys
from collections.abc import Sequence
from pathlib import Path

#: Third-party loggers whose INFO chatter would clutter a notebook's figure
#: cells (matplotlib's per-GIF "Animation.save" line, font-manager scans, PIL
#: plugin debug) -- quieted to WARNING WITHOUT touching this project's own
#: training narration.
DEFAULT_QUIET_LOGGERS: tuple[str, ...] = (
    "matplotlib.animation",
    "matplotlib.font_manager",
    "PIL",
)

#: The console log format: timestamp + level + message, with NO logger-name
#: field, so a training row reads ``Epoch: ...`` rather than
#: ``src.engine.callbacks: Epoch: ...``.
NOTEBOOK_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(message)s"

#: The marker whose presence identifies the project root.
_ROOT_MARKER = "pyproject.toml"


def find_project_root(start: Path | None = None) -> Path:
    """The project root: the nearest ancestor of `start` (inclusive) that
    contains a ``pyproject.toml``.

    Args:
        start: Where to begin the upward search; defaults to the current
            working directory.

    Returns:
        The resolved project-root path.

    Raises:
        RuntimeError: If no ancestor contains a ``pyproject.toml``.
    """
    start = (start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / _ROOT_MARKER).exists():
            return candidate
    raise RuntimeError(
        f"Could not locate the project root: no {_ROOT_MARKER!r} found at or "
        f"above {start}."
    )


def enable_inline_backend() -> bool:
    """Register Jupyter's inline (Figure -> PNG) backend when running inside an
    IPython shell, so figures render inline rather than opening a GUI window
    (which would hang a headless ``nbconvert --execute`` run).

    A no-op outside IPython (plain script / pytest), where there is no shell to
    configure.

    Returns:
        ``True`` if the inline backend was enabled, ``False`` if no IPython
        shell was present.
    """
    try:
        from IPython import get_ipython
    except ImportError:  # pragma: no cover - IPython is a dev dependency
        return False
    shell = get_ipython()
    if shell is None:
        return False
    shell.run_line_magic("matplotlib", "inline")
    return True


def configure_notebook_logging(
    *,
    level: int = logging.INFO,
    quiet_loggers: Sequence[str] = DEFAULT_QUIET_LOGGERS,
) -> None:
    """Install the console log handler with `NOTEBOOK_LOG_FORMAT` (no logger
    name), so this project's INFO narration -- device/hardware transparency,
    structured per-epoch training rows, cache hit/miss decisions -- is visible
    inline, then quiet the noisy third-party loggers in `quiet_loggers`.

    Args:
        level: The root logging level for the console handler.
        quiet_loggers: Third-party logger names lifted to WARNING (see
            `DEFAULT_QUIET_LOGGERS`).
    """
    logging.basicConfig(level=level, format=NOTEBOOK_LOG_FORMAT, force=True)
    for noisy in quiet_loggers:
        logging.getLogger(noisy).setLevel(logging.WARNING)


def bootstrap_experiment_notebook(
    *,
    start: Path | None = None,
    level: int = logging.INFO,
    quiet_loggers: Sequence[str] = DEFAULT_QUIET_LOGGERS,
    configure_logging: bool = True,
) -> Path:
    """The single call an experiment notebook opens with: enable the inline
    backend, configure console logging (see `configure_notebook_logging`),
    locate the project root, and ensure it is importable, returning that root.

    Args:
        start: Where to begin the project-root search; defaults to the current
            working directory.
        level: The root logging level.
        quiet_loggers: Third-party loggers to quiet (see
            `DEFAULT_QUIET_LOGGERS`).
        configure_logging: Install the console log handler; pass ``False`` to
            leave an existing logging configuration untouched (e.g. under a
            test runner that owns log capture).

    Returns:
        The project-root path (also inserted at the front of ``sys.path`` if
        not already present).
    """
    enable_inline_backend()
    if configure_logging:
        configure_notebook_logging(level=level, quiet_loggers=quiet_loggers)
    project_root = find_project_root(start)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root
