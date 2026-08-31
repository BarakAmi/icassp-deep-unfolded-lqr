"""Acceptance tests for gate evaluation — Stage 6, step B0.

Written before the implementation. Until now **no gate was ever evaluated**:
the parse stage validated that a gate was *sayable*, and nothing checked that
it *held*. That is the fourth occurrence in this project of a declaration that
takes no effect, and the gate NB04 declares exists to prevent a failure the
study nearly shipped.

Two properties carry the whole step, and both are tested in the failing
direction because the passing one says nothing:

1. **A gate that cannot be evaluated reports so, and never reports `passed`.**
   Reporting a check nobody made as green would be the exact defect §4 exists
   to prevent.
2. **A gate that fails, fails.** A threshold no study could miss is not a gate;
   every kind here is driven from both sides of its own threshold.

*Extended 2026-08-22, when `constraint_binds` went live.* Its statistic became
retained content (Annex 03 §A.3.1a), so the tests that pinned it as permanently
not-evaluable would now pin a defect. What survives is the property they were
protecting, in its stronger form: **absent statistic still reports not
evaluable**, and the gate must now also be shown to fail, to read its own
threshold, to exclude the roles Annex 01 §4.1 excludes, and to reduce by the
maximum rather than by anything else that happens to agree on one table.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mbl.analysis.gates import GateStatus, evaluate_gates
from mbl.spec.gates import GateKind, GateSpec

#: The tidy-table columns `cost_vs_axis` emits that this gate reads.
COLUMNS = ("contender", "role", "axis_value", "aggregate")


def _table(rows: list[tuple[str, str, float | None, float]]) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=list(COLUMNS))


def _separate(gap: float) -> GateSpec:
    return GateSpec(kind=GateKind.CONTENDERS_SEPARATE, config={"min_relative_gap": gap})


def _binds(fraction: float = 0.01) -> GateSpec:
    return GateSpec(kind=GateKind.CONSTRAINT_BINDS, config={"min_fraction": fraction})


#: The constraint-activity table `_activity_table` builds, whose `aggregate` is
#: a saturated fraction rather than a cost. A SEPARATE frame from the cost
#: table on purpose: folding two quantities into one `aggregate` column is how
#: a gate ends up comparing a fraction against a cost and reporting a verdict.
def _activity(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame([(c, r, 1.0, v) for c, r, v in rows], columns=list(COLUMNS))


#: A cost table that is present, well-formed and irrelevant — handed to every
#: `constraint_binds` case so that a verdict read off the WRONG frame shows up
#: as a wrong number rather than as a crash.
_ANY = _table([("a", "contender", 1.0, 2.0), ("b", "contender", 1.0, 3.0)])


class TestContendersSeparate:
    def test_it_passes_when_the_spread_clears_the_threshold(self) -> None:
        table = _table([("a", "contender", 1.0, 2.00), ("b", "contender", 1.0, 2.20)])
        (outcome,) = evaluate_gates((_separate(0.02),), table=table)
        assert outcome.status is GateStatus.PASSED
        assert outcome.measured == pytest.approx(0.10)

    def test_it_fails_when_the_spread_does_not(self) -> None:
        """The failing direction. A gate verified only where it passes is a
        gate nobody has tested."""
        table = _table([("a", "contender", 1.0, 2.000), ("b", "contender", 1.0, 2.001)])
        (outcome,) = evaluate_gates((_separate(0.02),), table=table)
        assert outcome.status is GateStatus.FAILED
        assert outcome.measured == pytest.approx(0.0005)

    def test_the_threshold_is_read_and_not_assumed(self) -> None:
        """Same table, two thresholds, opposite verdicts — so a hard-coded
        constant cannot satisfy both."""
        table = _table([("a", "contender", 1.0, 2.00), ("b", "contender", 1.0, 2.10)])
        lenient = evaluate_gates((_separate(0.02),), table=table)[0].status
        strict = evaluate_gates((_separate(0.20),), table=table)[0].status
        assert (lenient, strict) == (GateStatus.PASSED, GateStatus.FAILED)

    def test_a_bound_does_not_count_as_separation(self) -> None:
        """Annex 01 §4.1's exclusion, and the reason it exists: two tied
        contenders bracketed far from a bound must FAIL, or a study that
        measures noise passes on its distance from a reference line."""
        tied = [("a", "contender", 1.0, 2.000), ("b", "baseline", 1.0, 2.001)]
        without = evaluate_gates((_separate(0.02),), table=_table(tied))[0]
        with_bound = evaluate_gates(
            (_separate(0.02),), table=_table([*tied, ("lb", "bound", None, 1.0)])
        )[0]
        assert without.status is GateStatus.FAILED
        assert with_bound.status is GateStatus.FAILED
        assert with_bound.measured == pytest.approx(without.measured)

    def test_a_contender_is_reduced_to_its_best_over_the_axis(self) -> None:
        """§4.1: the study compares what each method CAN achieve.

        `b`'s worst depth is far from `a`, its best is adjacent to it. Reducing
        by mean or by worst would pass this table; reducing by best fails it,
        which is the declared rule.
        """
        table = _table(
            [
                ("a", "contender", 1.0, 2.000),
                ("b", "contender", 1.0, 9.000),
                ("b", "contender", 2.0, 2.001),
            ]
        )
        (outcome,) = evaluate_gates((_separate(0.02),), table=table)
        assert outcome.status is GateStatus.FAILED
        assert outcome.measured == pytest.approx(0.0005)

    def test_one_contender_cannot_separate_from_anything(self) -> None:
        table = _table([("a", "contender", 1.0, 2.0)])
        (outcome,) = evaluate_gates((_separate(0.02),), table=table)
        assert outcome.status is GateStatus.NOT_EVALUABLE


class TestConstraintBinds:
    """The gate that spent thirty study documents unable to fire."""

    def test_it_passes_when_a_feasible_contender_sits_on_the_box(self) -> None:
        activity = _activity([("a", "contender", 0.85), ("b", "baseline", 0.42)])
        (outcome,) = evaluate_gates((_binds(0.10),), table=_ANY, activity=activity)
        assert outcome.status is GateStatus.PASSED
        assert outcome.measured == pytest.approx(0.85)

    def test_it_fails_when_the_box_is_so_wide_it_never_binds(self) -> None:
        """The failing direction, and the failure this gate exists for: a box
        that never binds degenerates a constrained study into an expensive
        repeat of the unconstrained one."""
        activity = _activity([("a", "contender", 0.004), ("b", "baseline", 0.001)])
        (outcome,) = evaluate_gates((_binds(0.01),), table=_ANY, activity=activity)
        assert outcome.status is GateStatus.FAILED
        assert outcome.measured == pytest.approx(0.004)

    def test_the_threshold_is_read_and_not_assumed(self) -> None:
        """Same table, two thresholds, opposite verdicts — so a hard-coded
        comparison cannot pass both."""
        activity = _activity([("a", "contender", 0.30)])
        lenient = evaluate_gates((_binds(0.10),), table=_ANY, activity=activity)[0]
        strict = evaluate_gates((_binds(0.90),), table=_ANY, activity=activity)[0]
        assert (lenient.status, strict.status) == (
            GateStatus.PASSED,
            GateStatus.FAILED,
        )

    def test_it_reduces_by_the_maximum_over_participants(self) -> None:
        """Annex 01 §4.1. One controller pinned to the boundary refutes "the
        box never binds" outright; a conservative controller beside it is a
        finding about that controller, and a minimum — or a mean — would fail
        the study for having discovered it.

        The three candidate reductions are separated by construction: max
        0.90, mean 0.34, min 0.01.
        """
        activity = _activity(
            [
                ("saturating", "contender", 0.90),
                ("middling", "contender", 0.11),
                ("conservative", "baseline", 0.01),
            ]
        )
        (outcome,) = evaluate_gates((_binds(0.50),), table=_ANY, activity=activity)
        assert outcome.status is GateStatus.PASSED
        assert outcome.measured == pytest.approx(0.90)

    def test_a_reference_cannot_carry_the_verdict(self) -> None:
        """The exclusion that matters most. The unconstrained optimum exceeds
        the box wherever the box binds at all, so a gate that admitted it would
        report "the box binds" on the one loop that never respects it — and
        would do so on a cast whose feasible members never touch the bound."""
        activity = _activity([("free", "reference", 0.83), ("shy", "contender", 0.002)])
        (outcome,) = evaluate_gates((_binds(0.01),), table=_ANY, activity=activity)
        assert outcome.status is GateStatus.FAILED
        assert outcome.measured == pytest.approx(0.002)

    def test_a_bound_cannot_carry_the_verdict_either(self) -> None:
        """A bound attains no trajectory, so it has nothing to measure."""
        activity = _activity([("floor", "bound", 0.99), ("shy", "contender", 0.002)])
        (outcome,) = evaluate_gates((_binds(0.01),), table=_ANY, activity=activity)
        assert outcome.status is GateStatus.FAILED
        assert outcome.measured == pytest.approx(0.002)

    def test_a_cast_of_references_alone_is_not_evaluable(self) -> None:
        """Excluded is not the same as absent: with nothing feasible left, the
        gate has no witness and must say so rather than fail — a study cannot
        be refused for a check that was never possible."""
        activity = _activity([("free", "reference", 0.83), ("floor", "bound", 0.99)])
        (outcome,) = evaluate_gates((_binds(),), table=_ANY, activity=activity)
        assert outcome.status is GateStatus.NOT_EVALUABLE
        assert outcome.measured is None

    def test_it_reports_not_evaluable_when_the_store_carries_no_statistic(
        self,
    ) -> None:
        """What step B0's original test protected, in its durable form. A
        measurement written before Annex 03 §A.3.1a carries no
        constraint-activity statistic, and a green verdict on it would be a
        check nobody made."""
        (outcome,) = evaluate_gates((_binds(),), table=_ANY, activity=None)
        assert outcome.status is GateStatus.NOT_EVALUABLE
        assert outcome.measured is None

    def test_an_empty_activity_table_is_not_evaluable_either(self) -> None:
        """The shape `_activity_table` returns for a store with nothing to
        report — which must not be mistaken for a saturated fraction of zero."""
        (outcome,) = evaluate_gates((_binds(),), table=_ANY, activity=_activity([]))
        assert outcome.status is GateStatus.NOT_EVALUABLE

    def test_it_names_what_is_missing(self) -> None:
        """A refusal that does not say what is absent cannot be acted on."""
        (outcome,) = evaluate_gates((_binds(),), table=_ANY, activity=None)
        assert "constraint" in outcome.detail.lower()
        assert "measurementid" in outcome.detail.lower()

    def test_the_verdict_names_the_contender_behind_it(self) -> None:
        """The number without the contender is unactionable: an author reading
        a failure needs to know which policy was closest to the bound."""
        activity = _activity(
            [("saturating", "contender", 0.90), ("shy", "baseline", 0.01)]
        )
        (outcome,) = evaluate_gates((_binds(0.5),), table=_ANY, activity=activity)
        assert "saturating" in outcome.detail


class TestTheSetAsAWhole:
    def test_every_declared_gate_gets_exactly_one_outcome(self) -> None:
        """A gate silently dropped is indistinguishable from one that passed,
        which is the shape of the defect this step exists to close."""
        table = _table([("a", "contender", 1.0, 2.0), ("b", "contender", 1.0, 3.0)])
        gates = (_binds(), _separate(0.02))
        outcomes = evaluate_gates(gates, table=table)
        assert [o.kind for o in outcomes] == [g.kind for g in gates]

    def test_a_parse_stage_gate_reports_that_it_was_decided_earlier(self) -> None:
        """`stability` and `dimension_robust` are decided at parse time, so a
        post-hoc evaluator must neither re-check them nor claim them as its
        own — but it must still account for them, or a reader cannot tell a
        gate that was checked elsewhere from one that was forgotten."""
        table = _table([("a", "contender", 1.0, 2.0), ("b", "contender", 1.0, 3.0)])
        gate = GateSpec(
            kind=GateKind.DIMENSION_ROBUST, config={"control_dims": (1, 2, 3)}
        )
        (outcome,) = evaluate_gates((gate,), table=table)
        assert outcome.status is GateStatus.DECIDED_AT_PARSE
