"""The code-provenance stamp (REFACTOR_PLAN v3, T3.e — the v3 extension that
closes G7's cache half): every `ExperimentCache` key embeds this stamp, so a
committed change to released numeric semantics automatically invalidates every
prior cache entry — a stale result can never be silently revalidated by a
content-only signature match.

The stamp composes exactly two governed components:

* the **installed package version** (single source: the distribution
  metadata, falling back to ``pyproject.toml`` while the package root is not
  yet installed — the T1.a promotion makes the distribution authoritative);
* a deliberately-bumped **results-schema constant** owned by this layer, for
  mid-release semantic breaks.

Governance rule: any change affecting numerics, signatures, or
persisted-result semantics must bump at least the patch version (or
`RESULTS_SCHEMA_VERSION` for mid-release breaks) — a review-checklist
obligation aligned with the Conventional Commits discipline.

Deliberate non-choice: the stamp is **not** the git commit hash. Per-commit
invalidation would zero the cache's research value during normal iteration;
the stamp tracks *released semantics*, not editorial history.
"""

import tomllib
from functools import cache
from importlib import metadata
from pathlib import Path

#: The distribution name declared in ``pyproject.toml`` ([project] name).
PACKAGE_NAME = "model-based-learning-for-stochastic-control"

#: Bump deliberately for mid-release breaks to persisted-result semantics
#: (schema changes to cached payloads, cost-convention redefinitions, ...).
#: Either this or the package version changing orphans every prior cache
#: entry at once — exactly one migration boundary (REFACTOR_PLAN v3 §8 S4).
RESULTS_SCHEMA_VERSION = 1


@cache
def package_version() -> str:
    """The installed package version — one component of the provenance stamp.

    Resolution order: the installed distribution's metadata (authoritative
    once T1.a installs the package editable), then the repo's
    ``pyproject.toml`` (the pre-T1.a fallback, same single source the build
    would consume).

    Returns:
        The version string, e.g. ``"0.1.0"``.

    Raises:
        RuntimeError: If neither the distribution metadata nor
            ``pyproject.toml`` is reachable — provenance may never silently
            default.
    """
    try:
        return metadata.version(PACKAGE_NAME)
    except metadata.PackageNotFoundError:
        pass
    pyproject = Path(__file__).resolve().parents[3] / "pyproject.toml"
    if pyproject.is_file():
        with pyproject.open("rb") as f:
            return str(tomllib.load(f)["project"]["version"])
    raise RuntimeError(
        f"Cannot resolve the package version: distribution {PACKAGE_NAME!r} "
        f"is not installed and {pyproject} does not exist. The provenance "
        "stamp (T3.e) must never be guessed."
    )


def code_provenance_stamp() -> str:
    """The composite stamp embedded in every cache key and history record.

    Returns:
        ``"<package>-<version>/schema-<n>"``, e.g. ``"lqr-0.1.0/schema-1"``.
    """
    return f"{PACKAGE_NAME}-{package_version()}/schema-{RESULTS_SCHEMA_VERSION}"
