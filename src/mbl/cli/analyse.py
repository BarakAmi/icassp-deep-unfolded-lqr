"""`mbl analyse` — a study's declared tables, computed from the store.

Tier 6 has existed since slice Phase B-2 and has been reachable only from
Python. This is the entry point that makes a table producible by a person, and
it is the same shape as `mbl run`: the decision is made here and returned as
`(ok, message)`, because `cli/app.py` is the only module on this surface
allowed to `print`, and because importing even `mbl.spec.errors` at app scope
executes the grammar's package `__init__` and pulls numpy and torch into
`mbl models list` (`tests/store/test_import_cost.py`).

**Analysing does not run.** The document is read to learn what varies — the
axes, the labels, the roles — and the store is read for the numbers. Nothing
trains, nothing evaluates, and no controller is constructed. That is not a
convention: `run_analyses` is handed a `MeasurementStore` and a `StudySpec`,
and neither the producer nor the recipe registry is reachable from them.

**The tier is not optional in effect, only in spelling.** A tier moves every
identifier (F2a), so analysing at a tier the study was not run at looks for
measurements that do not exist — and this command **fails** rather than
emitting a short table. A table missing a contender is the plausible-and-wrong
output the whole re-architecture exists to make impossible, and it is the last
thing between the store and a figure.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..spec.errors import SpecificationError

#: What a study is analysed at when nothing says otherwise. The same default
#: `mbl run` uses, so the two commands agree without either restating a tier.
DEFAULT_TIER = "standard"


@dataclass(frozen=True)
class AnalyseOutcome:
    """What analysing a study produced.

    Attributes:
        study: The study's declared id.
        tier: The tier it was resolved at.
        rows: `(analysis_id, kind, row count)` per analysis, in declaration
            order.
    """

    study: str
    tier: str
    rows: tuple[tuple[str, str, int], ...]

    def render(self) -> str:
        """One line per analysis, plus a header naming the study and tier.

        The counts travel beside the name they belong to rather than as a
        total, for the reason slice Phase A recorded: a message carrying two
        numbers separately reads the same when they are swapped.
        """
        if not self.rows:
            return (
                f"{self.study} at tier {self.tier}: declares no analyses, so "
                "nothing was computed. Add an [[analyses]] table to the "
                "document (Annex 01 §2.6)"
            )
        lines = [f"{self.study} at tier {self.tier}:"]
        lines += [
            f"  {analysis_id} ({kind}): {rows} row{'s' if rows != 1 else ''}"
            for analysis_id, kind, rows in self.rows
        ]
        return "\n".join(lines)


def analyse(
    study: Path | str,
    *,
    store: Path,
    tier: str,
    catalogue: str | None = None,
    overrides: Mapping[str, Any] | None = None,
    only: Sequence[str] | None = None,
) -> tuple[bool, str]:
    """The whole of `mbl analyse`, as a value.

    Args:
        study: The study document.
        store: The store root. Read, never created.
        tier: The tier to resolve at — it decides which measurements exist.
        catalogue: A tier catalogue file, or `None` for the shipped default.
        overrides: The per-invocation editing level, so a table can be
            computed for the same overrides the run used.
        only: Analysis ids to compute, or `None` for all of them.

    Returns:
        `(ok, message)`. `ok` is `False` for any specification failure, whose
        message is the grammar's or the analysis's own — it names the offending
        key, the missing measurement or the subsetted run, which is the whole
        deliverable of those refusals.
    """
    from ..analysis import run_analyses
    from .run import plan_run

    try:
        resolved, _ = plan_run(
            study, store=store, tier=tier, catalogue=catalogue, overrides=overrides
        )
        outcomes = run_analyses(resolved.study, store=store, only=only)
    except SpecificationError as error:
        return False, str(error)
    return True, AnalyseOutcome(
        study=resolved.study.id,
        tier=tier,
        rows=tuple(
            (outcome.analysis_id, outcome.kind, outcome.rows) for outcome in outcomes
        ),
    ).render()
