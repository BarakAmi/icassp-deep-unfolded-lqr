"""`mbl run` — a study document, executed into the store from a command line.

The producer has existed since Stage 2 Phase F2 and has been reachable only from
Python. This is the entry point that makes a study runnable by a person, and it
is deliberately **thin**: it reads the three editing levels of Annex 01 §2.3.1,
resolves them, and calls `run_study`. Annex 06's job DAG, resource classes and
persistent queue are deferred with reasons recorded in the Stages 3–5 slice
plan, and a command that grew a scheduler would be the thing that plan exists to
prevent.

**`--dry-run` answers from the specification alone**, which is the property the
identity split was built for and the one this module is most likely to lose.
Measured on the tracked NB04 study:

| Step | Cost |
|---|---|
| load the document and the catalogue, resolve, materialise all nine points | 0.00 s |
| `build_controller()` for **one** COCP contender | **1.37 s** |

`build_controller` is where D17's solver canary runs, so a preview that reaches
it is not slightly slower — it is a different kind of operation. The count comes
from `materialise()` and `ModelStore.exists()`, and nothing here constructs a
controller. Note which call that is: recipe *construction* is free and a dry run
does perform it, to derive identifiers at all.

**`--set` parses its value as TOML.** `--set training.seeds=2` must reach the
grammar as the integer 2, not the string `"2"` — the tier catalogue's one
per-path coercion refuses a string, and a study whose seed count arrived as text
would fail far from the flag that caused it. Parsing through `tomllib` makes the
command line agree with the document surface by construction rather than by a
second little parser that can drift from it.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..spec.errors import SpecificationError
from ..spec.loader import load_study, load_tier_catalogue
from ..spec.tiers import DEFAULT_TIER_CATALOGUE, ResolvedStudy, TierCatalogue

#: Annex 01 §2.3's default. A command that required `--tier` would make the
#: documented default unreachable.
DEFAULT_TIER = "standard"

#: Separates a `--set` path from its value. The value is TOML, so it may itself
#: contain `=` inside a string and the split is deliberately on the first only.
SET_SEPARATOR = "="


@dataclass(frozen=True)
class RunOutcome:
    """What a run would cost, or did.

    Attributes:
        study: The study's declared id.
        tier: The tier it was resolved at.
        points: Materialised points — the execution units.
        to_train: Points whose model the store does not hold.
        reused: Points whose model it already holds.
    """

    study: str
    tier: str
    points: int
    to_train: int
    reused: int

    def render(self, *, previewed: bool) -> str:
        """One line a person can act on."""
        verb = "would train" if previewed else "trained"
        return (
            f"{self.study} at {self.tier}: {self.points} point(s), "
            f"{verb} {self.to_train}, reused {self.reused}"
        )


def parse_overrides(assignments: Sequence[str]) -> dict[str, Any]:
    """`--set path=value` pairs, with each value parsed as TOML.

    Args:
        assignments: The raw `path=value` strings.

    Returns:
        Path to parsed value.

    Raises:
        SpecificationError: If an assignment has no `=`, or its value is not a
            TOML value. Both name the offending assignment, because a command
            line is the one surface with no file and line to point at.
    """
    overrides: dict[str, Any] = {}
    for assignment in assignments:
        path, separator, raw = assignment.partition(SET_SEPARATOR)
        if not separator or not path:
            raise SpecificationError(
                f"--set {assignment!r} is not an assignment; write "
                "--set path.to.field=value"
            )
        try:
            overrides[path] = tomllib.loads(f"value = {raw}")["value"]
        except tomllib.TOMLDecodeError as error:
            raise SpecificationError(
                f"--set {assignment!r} has a value TOML cannot read ({error}). "
                "Values use the document's own syntax, so a string needs "
                f"quoting: --set {path}='\"text\"'"
            ) from error
    return overrides


def _catalogue(path: str | None) -> TierCatalogue:
    """The tracked catalogue when one is named, else the shipped default.

    The two are asserted equal by `tests/spec/test_loader.py`, so which is used
    changes nothing — but a run whose tier came from a file the author did not
    name would be a run whose effort depends on the working directory.
    """
    return DEFAULT_TIER_CATALOGUE if path is None else load_tier_catalogue(path)


def plan_run(
    study: Path | str,
    *,
    store: Path,
    tier: str,
    catalogue: str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> tuple[ResolvedStudy, RunOutcome]:
    """Resolve a document and count what running it would cost.

    **Nothing here builds a controller.** The counts come from the identifiers
    `materialise()` derives and from `ModelStore.exists`, which is what makes a
    preview instant and what makes it honest: the model a run would reuse is the
    model whose identifier is already in the store, by definition rather than by
    estimate.

    Args:
        study: The study document.
        store: The store root. Not created here.
        tier: The tier to resolve at.
        catalogue: A tier catalogue file, or `None` for the shipped default.
        overrides: The per-invocation editing level.

    Returns:
        The resolved study and the outcome a run would have.

    Raises:
        SpecificationError: On any malformed document, unknown tier or
            forbidden override.
    """
    from ..experiments.bindings import DEFAULT_SPEC_BINDINGS
    from ..store.content_store import ModelStore

    document = load_study(study, bindings=DEFAULT_SPEC_BINDINGS)
    resolved = document.resolve(
        _catalogue(catalogue), tier, overrides=dict(overrides or {})
    )
    points = resolved.study.materialise()
    models = ModelStore(Path(store))
    held = sum(1 for point in points if models.exists(point.model_id))
    return resolved, RunOutcome(
        study=resolved.study.id,
        tier=tier,
        points=len(points),
        to_train=len(points) - held,
        reused=held,
    )


def preview_or_execute(
    study: Path | str,
    *,
    store: Path,
    tier: str,
    catalogue: str | None,
    assignments: Sequence[str],
    previewed: bool,
) -> tuple[bool, str]:
    """The whole of `mbl run`, as a value.

    Returns `(ok, message)` rather than printing and rather than raising,
    because the command line's print carve-out is deliberately **one module
    deep** and this is not that module. It also keeps `cli/app.py` free of any
    `mbl.spec` import: importing even `spec.errors` executes the grammar's
    package `__init__` and pulls numpy and torch, which would make
    `mbl models list` pay for a tensor library it never uses
    (`tests/store/test_import_cost.py`).

    Args:
        study: The study document.
        store: The store root; created by the run, not by the preview.
        tier: The tier to resolve at.
        catalogue: A tier catalogue file, or `None` for the shipped default.
        assignments: Raw `--set path=value` strings.
        previewed: `True` for `--dry-run`.

    Returns:
        `(ok, message)`. `ok` is `False` for any specification failure, whose
        message is the grammar's own — it names the offending key, which is the
        deliverable the loader was written for.
    """
    try:
        resolved, preview = plan_run(
            study,
            store=store,
            tier=tier,
            catalogue=catalogue,
            overrides=parse_overrides(assignments),
        )
    except SpecificationError as error:
        return False, str(error)
    if previewed:
        return True, preview.render(previewed=True)
    return True, execute_run(resolved, store=store, tier=tier).render(previewed=False)


def execute_run(resolved: ResolvedStudy, *, store: Path, tier: str) -> RunOutcome:
    """Run a resolved study, and report what it cost.

    Returns rather than prints, for the reason `render.py` does: the command
    line's print carve-out is deliberately one module deep, so everything below
    it stays testable against values instead of captured output.
    """
    from ..experiments.bindings import DEFAULT_SPEC_BINDINGS
    from ..runner.producer import run_study

    report = run_study(
        resolved.study,
        store=store,
        bindings=DEFAULT_SPEC_BINDINGS,
        provenance=resolved.provenance,
    )
    return RunOutcome(
        study=report.study,
        tier=tier,
        points=len(report.outcomes),
        to_train=len(report.trained),
        reused=len(report.outcomes) - len(report.trained),
    )
