"""Evaluating the gates a study declares — Annex 01 §4.1.

Until this module, **no gate was ever evaluated**. The parse stage validated
that a gate was *sayable* (`spec/gates.py`, and the coherence check in
`spec/study.py`), and nothing anywhere checked that it *held*. That is the
fourth occurrence in this project of a declaration that takes no effect, and
the one it mattered for most is on the record in NB04's own document: the
`constraint_binds` gate exists because a box so wide it never bound would
silently degenerate the constrained study into an expensive repeat of the
unconstrained one — a failure the study nearly shipped, guarded by a check that
did not run.

`spec/gates.py` says where the evaluation belongs and this module is the whole
of it: **post-hoc, over a tidy table**, which is where §4's own table puts
`contenders_separate` and where the `axis_subset` refusal already lives.
`constraint_binds` joined it on 2026-08-22, when the statistic it needs became
retained content (Annex 03 §A.3.1a) and the annex moved it off a pre-flight
proxy that could not have seen the contenders it was being asked about.

**A gate this tier cannot decide reports `NOT_EVALUABLE` and never `PASSED`.**
That distinction is the whole of step B0: a green verdict on a check nobody
made is worse than no verdict, because a reader cannot tell them apart.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from ..spec.gates import GateKind, GateSpec, GateStage

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

#: Roles that take part in `contenders_separate`. A bound is excluded by
#: Annex 01 §4.1: it is a reference the study is bracketed against, not a
#: competitor, and including it would let a study whose contenders all tie pass
#: on its distance from a reference line alone.
EXCLUDED_ROLES = frozenset({"bound"})

#: Roles that take part in `constraint_binds` — the policies that actually
#: respect the box. A `reference` is infeasible **by construction**: the
#: unconstrained optimum exceeds the box wherever the box binds at all, so
#: admitting it would let a study report "the box binds" on the one loop that
#: never respects it. A `bound` attains no trajectory and has nothing to
#: measure. Annex 01 §4.1.
FEASIBLE_ROLES = frozenset({"contender", "baseline"})

#: What `constraint_binds` needs and a measurement written before Annex 03
#: §A.3.1a does not carry. Named in the outcome, because a refusal that does
#: not say what is absent cannot be acted on.
MISSING_FOR_CONSTRAINT_BINDS = (
    "the fraction of scalar control entries at the box, over the study's "
    "feasible contenders. A measurement written before Annex 03 §A.3.1a "
    "carries (batch_index, trajectory_index, trajectory_cost) and no "
    "constraint-activity statistic; a problem declaring no control bound has "
    "no box to bind. Re-evaluating the study retains it -- the models are "
    "reused and no identifier moves, because a measurement's payload takes no "
    "part in MeasurementID"
)


class GateStatus(StrEnum):
    """A gate's verdict, including the two that are not verdicts.

    `NOT_EVALUABLE` and `DECIDED_AT_PARSE` exist so that "checked and held",
    "checked and failed", "cannot be checked here" and "checked somewhere
    else" are four distinguishable things. Collapsing any of them into
    `PASSED` is the defect this module was written to end.
    """

    PASSED = "passed"
    FAILED = "failed"
    #: This tier cannot decide it; `detail` says what is missing.
    NOT_EVALUABLE = "not evaluable"
    #: Decided from the specification before any compute (`spec/study.py`).
    DECIDED_AT_PARSE = "decided at parse time"


@dataclass(frozen=True)
class GateOutcome:
    """One gate's verdict, with the number behind it.

    Attributes:
        kind: The gate.
        stage: When its verdict can be reached, from the declaration.
        status: The verdict.
        measured: The quantity compared, or `None` when there was none.
        threshold: What it was compared against, or `None`.
        detail: One sentence a reader can act on.
    """

    kind: GateKind
    stage: GateStage
    status: GateStatus
    measured: float | None
    threshold: float | None
    detail: str

    @property
    def blocking(self) -> bool:
        """Whether this outcome must stop a notebook (Annex 04 §4)."""
        return self.status is GateStatus.FAILED


def _participants(table: pd.DataFrame) -> dict[str, float]:
    """Each participating contender, reduced to its best aggregate.

    Best rather than mean, per Annex 01 §4.1: the study compares what each
    method *can achieve*, and a contender's worst axis point says nothing
    about whether the method is distinguishable from another method.
    """
    rows = table[~table["role"].isin(EXCLUDED_ROLES)]
    grouped = rows.groupby("contender")["aggregate"].min()
    return {str(name): float(value) for name, value in grouped.items()}


def _contenders_separate(gate: GateSpec, table: pd.DataFrame) -> GateOutcome:
    threshold = float(gate.config["min_relative_gap"])
    best_per_contender = _participants(table)
    if len(best_per_contender) < 2:
        return GateOutcome(
            kind=gate.kind,
            stage=gate.stage,
            status=GateStatus.NOT_EVALUABLE,
            measured=None,
            threshold=threshold,
            detail=(
                f"{len(best_per_contender)} participating contender(s); a gate "
                "about separation needs two things to separate"
            ),
        )
    best, worst = min(best_per_contender.values()), max(best_per_contender.values())
    measured = (worst - best) / abs(best)
    passed = measured >= threshold
    return GateOutcome(
        kind=gate.kind,
        stage=gate.stage,
        status=GateStatus.PASSED if passed else GateStatus.FAILED,
        measured=measured,
        threshold=threshold,
        detail=(
            f"best {best:.6g} to worst {worst:.6g} over "
            f"{len(best_per_contender)} contenders is a relative spread of "
            f"{measured:.3%}, against a declared minimum of {threshold:.3%}"
            + ("" if passed else "; the study may be measuring noise")
        ),
    )


def _constraint_binds(gate: GateSpec, activity: pd.DataFrame | None) -> GateOutcome:
    """The box must be active often enough to matter — Annex 01 §4.1.

    Reduced by the **maximum** over feasible participants, which is what the
    gate is for: the failure on this project's record is *a box so wide it
    never bound*, degenerating a constrained study into an expensive repeat of
    the unconstrained one, and one controller pinned to the boundary refutes
    that outright. A conservative controller that rarely saturates is a finding
    about that controller, and a minimum would fail the study for having
    discovered it.
    """
    threshold = float(gate.config["min_fraction"])
    if activity is None or activity.empty:
        return _not_evaluable_here(gate, MISSING_FOR_CONSTRAINT_BINDS)
    rows = activity[activity["role"].isin(FEASIBLE_ROLES)]
    if rows.empty:
        return GateOutcome(
            kind=gate.kind,
            stage=gate.stage,
            status=GateStatus.NOT_EVALUABLE,
            measured=None,
            threshold=threshold,
            detail=(
                "no contender or baseline carries the statistic; a reference "
                "is infeasible by construction and a bound attains no "
                "trajectory, so neither can witness that the box binds"
            ),
        )
    measured = float(rows["aggregate"].max())
    leader = rows.loc[rows["aggregate"].idxmax(), "contender"]
    passed = measured >= threshold
    return GateOutcome(
        kind=gate.kind,
        stage=gate.stage,
        status=GateStatus.PASSED if passed else GateStatus.FAILED,
        measured=measured,
        threshold=threshold,
        detail=(
            f"{leader} sits on the box for {measured:.3%} of its control "
            f"entries, the most of {rows['contender'].nunique()} feasible "
            f"contender(s), against a declared minimum of {threshold:.3%}"
            + (
                ""
                if passed
                else "; the box may be so wide it never binds, which makes "
                "this an expensive repeat of the unconstrained study"
            )
        ),
    )


def _not_evaluable_here(gate: GateSpec, missing: str) -> GateOutcome:
    return GateOutcome(
        kind=gate.kind,
        stage=gate.stage,
        status=GateStatus.NOT_EVALUABLE,
        measured=None,
        threshold=None,
        detail=missing,
    )


def evaluate_gates(
    gates: Sequence[GateSpec],
    *,
    table: pd.DataFrame,
    activity: pd.DataFrame | None = None,
) -> tuple[GateOutcome, ...]:
    """Decide every gate this tier can, and account for the ones it cannot.

    Args:
        gates: The study's declared gates, in declaration order.
        table: A `cost_vs_axis` tidy table — the measurements, aggregated.
        activity: The same shape over the **constraint-activity** quantity of
            Annex 03 §A.3.1a — `contender`, `role`, `axis_value`, `aggregate`
            — or `None` where the store carries none. A second frame rather
            than a second column because it is a different reduction of the
            same measurements, and folding two quantities into one `aggregate`
            is how a gate ends up comparing a saturated fraction against a
            cost.

    Returns:
        One outcome per declared gate, in declaration order. **Exactly one**:
        a gate silently dropped is indistinguishable from a gate that passed,
        which is the shape of the defect this module closes.
    """
    outcomes: list[GateOutcome] = []
    for gate in gates:
        if gate.kind is GateKind.CONTENDERS_SEPARATE:
            outcomes.append(_contenders_separate(gate, table))
        elif gate.kind is GateKind.CONSTRAINT_BINDS:
            outcomes.append(_constraint_binds(gate, activity))
        elif gate.stage is GateStage.PARSE:
            outcomes.append(
                GateOutcome(
                    kind=gate.kind,
                    stage=gate.stage,
                    status=GateStatus.DECIDED_AT_PARSE,
                    measured=None,
                    threshold=None,
                    detail=(
                        "decided from the specification before any compute; a "
                        "study reaching this point already satisfies it"
                    ),
                )
            )
        else:  # pragma: no cover - the kind set is closed and covered above
            outcomes.append(
                _not_evaluable_here(gate, f"no evaluator for {gate.kind.value!r}")
            )
    return tuple(outcomes)
