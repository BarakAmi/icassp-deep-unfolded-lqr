"""Acceptance tests for the declarative pre-flight gates — Stage 2 Phase E2.

Written before the implementation. The checkpoint class is **degeneracy**: a
gate exists to catch a study that has quietly stopped testing what it claims,
so every way of writing a gate that cannot do its job must fail at parse time
and name what is wrong with it.

The recorded failure these prevent is in this project's own history: a box
constraint so wide it never bound, degenerating a constrained study into an
expensive repeat of the unconstrained one. A gate that is silently misspelled
is strictly worse than no gate, because it reads in the study file as a check
that is being performed.

Two properties beyond the schema carry weight here, and both are asserted
against the study rather than the gate:

* **`gates` participates in `StudyID`.** D15 forbids a tier from writing
  `gates.*` on the ground that a cheaper run must not be a less-checked one;
  that rule would be much weaker if two studies differing only in how strictly
  they are checked shared an identity.
* **A gate must be able to fire.** `constraint_binds` on an unconstrained
  problem, and `dimension_robust` over a set excluding the study's own control
  dimension, are gates whose verdict is fixed before any compute runs.

Every accepted kind, key and value below is written out as a literal rather
than read from the implementation's own tables: a suite that iterates the
constant it is checking agrees with whatever that constant says, including a
version that has let the defect back in.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from mbl.core.runtime import Backend, ComputeContext, Precision
from mbl.engine.training_plan import OptimizerSpec, TrainingPlan
from mbl.spec.data import DataSpec
from mbl.spec.errors import SpecificationError
from mbl.spec.evaluation import EvaluationSpec
from mbl.spec.gates import GateKind, GateSpec, GateStage
from mbl.spec.problem import ProblemData, ProblemSpec
from mbl.spec.study import StudySpec
from mbl.spec.training import BatchPlan, TrainingSpec

from .test_contender import FAMILIES, REGISTRY
from .test_study import _contender, _protocol

N, M, HORIZON = 4, 2, 6
CTX = ComputeContext(backend=Backend.TORCH, device="cpu", precision=Precision.FLOAT64)

#: The five gate declarations of Annex 01 §4, transcribed from the annex's own
#: YAML block rather than from `gates.py`.
ANNEX_GATES: dict[str, dict[str, Any]] = {
    "constraint_binds": {"min_fraction": 0.01},
    "contenders_separate": {"min_relative_gap": 0.02},
    "dimension_robust": {"control_dims": (1, 2, M)},
    "stability": {"max_spectral_radius": 1.0},
    "step_is_inverse_lipschitz": {"contenders": ("unfolded_a",)},
}

#: When each kind can be decided, from the annex's table. Restated, not
#: imported, for the reason in the module docstring.
ANNEX_STAGES: dict[str, GateStage] = {
    # Post-hoc since 2026-08-22, when the statistic became retained content
    # (Annex 03 §A.3.1a). The pre-flight definition it replaces nominated one
    # cheap rollout of a NON-LEARNED contender as the witness, which cannot see
    # the contenders the gate is asked about -- saturation is a property of a
    # policy's own trajectory, and four contenders on one plant read 89.65 %,
    # 87.62 %, 87.82 % and 94.25 %.
    "constraint_binds": GateStage.POST_HOC,
    "contenders_separate": GateStage.POST_HOC,
    "dimension_robust": GateStage.PARSE,
    "stability": GateStage.PARSE,
    "step_is_inverse_lipschitz": GateStage.PARSE,
}

EVERY_KIND = sorted(ANNEX_GATES)


def _problem(seed: int = 0, *, control_bound: float | None = 0.5) -> ProblemSpec:
    rng = np.random.default_rng(seed)
    return ProblemSpec(
        ProblemData(
            system={"A": rng.normal(size=(N, N)), "B": rng.normal(size=(N, M))},
            cost={
                "Q": np.repeat(np.eye(N)[None], HORIZON + 1, axis=0),
                "R": np.repeat(np.eye(M)[None], HORIZON, axis=0),
            },
            horizon=HORIZON,
            control_bound=control_bound,
        )
    )


def _study(**overrides: Any) -> StudySpec:
    fields: dict[str, Any] = {
        "id": "probe/gated",
        "problem": _problem(),
        "contenders": (
            _contender("unfolded", "unfolded_a"),
            _contender("truncated_riccati", "baseline"),
        ),
        "training": TrainingSpec(
            data=DataSpec(kind="gaussian", seed=1, params={"process_noise_std": 0.5}),
            plan=TrainingPlan(
                optimizer=OptimizerSpec(name="adam", learning_rate=1e-3), epochs=3
            ),
            batch=BatchPlan(effective_size=256),
            ctx=CTX,
        ),
        "evaluation": EvaluationSpec(problem=_problem(), protocol=_protocol(), ctx=CTX),
    }
    fields.update(overrides)
    return StudySpec(**fields)


def _gate(kind: str) -> GateSpec:
    return GateSpec(kind=GateKind(kind), config=ANNEX_GATES[kind])


# --------------------------------------------------------------------------
# The kinds the annex declares, and nothing else
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", EVERY_KIND)
def test_every_annex_gate_kind_is_accepted(kind: str) -> None:
    """All four, not a sample: a grammar that accepts three of the annex's own
    examples is a grammar the annex cannot be written against."""
    gate = _gate(kind)
    assert gate.kind.value == kind


def test_the_kind_set_is_exactly_the_annex_s() -> None:
    """The other direction. An extra kind is a promise nothing keeps, and a
    missing one silently drops a check every study was to inherit."""
    assert {member.value for member in GateKind} == set(ANNEX_GATES)


def test_an_unknown_kind_is_refused_naming_the_accepted_set() -> None:
    with pytest.raises(SpecificationError) as error:
        GateSpec(kind="constraint_bounds", config={"min_fraction": 0.01})
    message = str(error.value)
    assert "constraint_bounds" in message
    for kind in EVERY_KIND:
        assert kind in message, "the message must show what was meant instead"


@pytest.mark.parametrize("kind", EVERY_KIND)
def test_each_kind_declares_when_it_can_be_decided(kind: str) -> None:
    """The annex's table. `contenders_separate` in particular cannot abort a
    study before training, since whether two contenders separate is what the
    training was run to find out."""
    assert _gate(kind).stage is ANNEX_STAGES[kind]


# --------------------------------------------------------------------------
# Degeneracy — every way a gate can be written into silence
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", EVERY_KIND)
def test_an_unknown_configuration_key_is_refused(kind: str) -> None:
    """Covering every kind, not one: an unrecognised key is how a real gate
    turns into a decorative one, and a schema checked for a single kind is a
    schema three quarters unchecked."""
    with pytest.raises(SpecificationError) as error:
        GateSpec(kind=GateKind(kind), config={**ANNEX_GATES[kind], "tolerance": 0.5})
    assert "tolerance" in str(error.value)


@pytest.mark.parametrize("kind", EVERY_KIND)
def test_a_missing_configuration_key_is_refused(kind: str) -> None:
    """A gate with no threshold has no verdict; defaulting one would invent a
    scientific claim the author never made."""
    with pytest.raises(SpecificationError) as error:
        GateSpec(kind=GateKind(kind), config={})
    message = str(error.value)
    assert all(key in message for key in ANNEX_GATES[kind])


@pytest.mark.parametrize(
    ("kind", "config"),
    [
        ("constraint_binds", {"min_fraction": 0.0}),
        ("constraint_binds", {"min_fraction": 1.5}),
        ("contenders_separate", {"min_relative_gap": -0.02}),
        ("dimension_robust", {"control_dims": ()}),
        ("dimension_robust", {"control_dims": (0, 1)}),
        ("stability", {"max_spectral_radius": 0.0}),
    ],
    ids=[
        "fraction zero",
        "fraction above one",
        "negative gap",
        "no dimensions",
        "dimension zero",
        "radius zero",
    ],
)
def test_an_unsatisfiable_value_is_refused(kind: str, config: dict[str, Any]) -> None:
    """A threshold outside its own range is a gate that either never fires or
    always does. `min_fraction: 0.0` is the exact shape of the historical
    failure: a constraint-binding check that passes on a constraint which never
    binds."""
    with pytest.raises(SpecificationError) as error:
        GateSpec(kind=GateKind(kind), config=config)
    assert next(iter(config)) in str(error.value)


def test_two_gates_of_one_kind_are_refused() -> None:
    """Not a conjunction: a study carrying two `stability` thresholds says
    nothing about which one it claims to meet."""
    with pytest.raises(SpecificationError) as error:
        _study(
            gates=(
                GateSpec(GateKind.STABILITY, {"max_spectral_radius": 1.0}),
                GateSpec(GateKind.STABILITY, {"max_spectral_radius": 2.0}),
            )
        )
    assert "stability" in str(error.value)


# --------------------------------------------------------------------------
# Coherence — a gate whose verdict is fixed before any compute runs
# --------------------------------------------------------------------------


def test_constraint_binds_is_refused_on_an_unconstrained_problem() -> None:
    """The gate asks what fraction of the time the box is active. With no box
    there is no fraction, and the study would run to completion reporting a
    check it never made."""
    with pytest.raises(SpecificationError) as error:
        _study(
            problem=_problem(control_bound=None),
            gates=(_gate("constraint_binds"),),
        )
    message = str(error.value)
    assert "constraint_binds" in message and "control_bound" in message


def test_constraint_binds_is_accepted_on_a_constrained_problem() -> None:
    """The positive direction, so the refusal above cannot be passing because
    the gate is refused everywhere."""
    study = _study(gates=(_gate("constraint_binds"),))
    assert study.gates[0].kind is GateKind.CONSTRAINT_BINDS


def test_dimension_robust_must_cover_the_study_s_own_control_dimension() -> None:
    """The study runs at its own control dimension whatever else it sweeps, so
    a declared set excluding it claims robustness over dimensions the study
    never visits while saying nothing about the one it does."""
    with pytest.raises(SpecificationError) as error:
        _study(
            gates=(
                GateSpec(GateKind.DIMENSION_ROBUST, {"control_dims": (M + 1, M + 2)}),
            )
        )
    message = str(error.value)
    assert "dimension_robust" in message and str(M) in message


def test_dimension_robust_is_accepted_when_it_covers_that_dimension() -> None:
    study = _study(gates=(_gate("dimension_robust"),))
    assert study.gates[0].config["control_dims"] == (1, 2, M)


# --------------------------------------------------------------------------
# Identity — gates are content, and the tier whitelist depends on it
# --------------------------------------------------------------------------


def test_declaring_a_gate_changes_the_study_id() -> None:
    """D15 forbids a tier from writing `gates.*` because a cheaper run must not
    be a less-checked one. If a gate were invisible to identity, two studies
    checked to different standards would be one study."""
    assert _study().study_id != _study(gates=(_gate("stability"),)).study_id


def test_a_stricter_threshold_is_a_different_study() -> None:
    """The same argument one level down: relaxing a threshold is exactly the
    edit the whitelist exists to prevent, so it must not hash the same."""
    loose = _study(gates=(GateSpec(GateKind.STABILITY, {"max_spectral_radius": 2.0}),))
    strict = _study(gates=(GateSpec(GateKind.STABILITY, {"max_spectral_radius": 1.0}),))
    assert loose.study_id != strict.study_id


def test_the_order_gates_are_written_in_does_not_change_the_study_id() -> None:
    """Two authors listing the same checks in a different order must collide
    rather than diverge -- the same rule `metrics` already obeys."""
    kinds = [_gate(kind) for kind in EVERY_KIND]
    assert (
        _study(gates=tuple(kinds)).study_id
        == _study(gates=tuple(reversed(kinds))).study_id
    )


def test_a_study_carrying_every_gate_survives_the_strict_json_check() -> None:
    """`study_id` refuses anything `json.dumps` cannot represent, deliberately,
    so that a signature can never hash an object's address. `control_dims` is a
    tuple and is the member most likely to break it."""
    study = _study(
        problem=_problem(),
        gates=tuple(_gate(kind) for kind in EVERY_KIND),
    )
    assert len(str(study.study_id)) == 16


def test_gates_are_visible_in_the_signature_tree_under_their_own_key() -> None:
    """A structural companion to the identity tests: those would also pass if
    gates leaked into some other key, which would collide with a real field."""
    tree = _study(gates=(_gate("stability"),)).get_signature()
    assert [entry["kind"] for entry in tree["gates"]] == ["stability"]


def test_a_study_declares_no_gates_by_default() -> None:
    """Gates are opt-in. A default gate would apply a scientific claim to every
    study ever written, including the ones it is wrong for."""
    assert _study().gates == ()
    assert _study().get_signature()["gates"] == []


def test_the_contender_registry_is_still_the_real_one() -> None:
    """Anti-vacuity: every study above is built from the live recipe registry,
    so a fixture that quietly stopped resolving would make the identity
    assertions compare two empty trees."""
    assert set(FAMILIES) == set(REGISTRY.available())
