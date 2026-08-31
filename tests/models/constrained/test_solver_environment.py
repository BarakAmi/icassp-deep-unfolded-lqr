"""Environment guard for the COCP solver seam.

Every COCP-backed contender needs `DIFFCP` -- it is the project's long-standing
reference solver and the fallback whenever MOREAU is unavailable or fails its
correctness canary. When `diffcp` cannot be imported, cvxpy reports the generic
`SolverError: The solver DIFFCP is not installed`, and roughly sixty downstream
tests fail with tracebacks that point at COCP rather than at the environment.

These two tests exist to fail *first*, and to say why. They are deliberately
version-agnostic: the concrete cause has already differed twice.

  1. A cold environment escalated diffcp's compile-time `SyntaxWarning` (its
     docstrings contain a raw `\\i`) into an import failure, because the
     zero-warning policy turns warnings into errors and a warm `__pycache__`
     had been hiding it locally.
  2. Python 3.14 reworded that same warning from ``invalid escape sequence
     '\\i'`` to ``"\\i" is an invalid escape sequence...``. The filter that
     silenced case 1 was anchored to the old wording, so it stopped matching
     and the failure returned -- on CI only, because nothing pinned the
     interpreter version.

Neither cause was a defect in this project's logic, and neither was visible
from the failing tests. Hence a direct assertion on the environment.
"""

from __future__ import annotations

import cvxpy


def test_diffcp_imports_under_the_projects_warning_filters() -> None:
    """The import must succeed with the suite's own filters in force.

    Runs under the configured `filterwarnings`, so it fails exactly when a
    warning diffcp emits at import or compile time is not covered -- whatever
    the interpreter version happens to word it as.
    """
    import diffcp  # noqa: PLC0415 -- the import IS the assertion

    assert diffcp is not None


def test_cvxpy_reports_diffcp_as_installed() -> None:
    """cvxpy's own view is what the COCP layer actually consults.

    `diffcp` importing is necessary but not sufficient: cvxpy discovers solvers
    through its own registry, and a failure there is what produces the
    misleading "not installed" message.
    """
    installed = cvxpy.installed_solvers()
    assert "DIFFCP" in installed, (
        "cvxpy does not report DIFFCP as installed, so every COCP contender will "
        f"fail with a misleading SolverError. Installed: {sorted(installed)}. "
        "Check that `diffcp` imports cleanly under the suite's warning filters."
    )
