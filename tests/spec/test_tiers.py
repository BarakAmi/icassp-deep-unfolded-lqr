"""Acceptance tests for the tier catalogue (D15) — Stage 2 Phase E2.

Written before the implementation. The checkpoint class is **negative control
covering every instance**: a tier may write only to a closed whitelist of
paths, and the forbidden column is the whole point of the rule, so each of the
four forbidden classes is refused here rather than one representative. Where a
rule applies N times, verify N times.

D15 in one sentence: *a tier scales effort; it never changes content.* The
failure it prevents is a `publication` run that quietly means something the
reader of the resulting figure does not assume — a swept axis truncated, a
gate relaxed, a contender family swapped.

Three properties beyond the whitelist carry the phase:

* **The tier is not an opaque label.** It participates in identity only through
  the fields it writes, so two differently named tiers writing identical
  effective fields must produce identical identifiers. Otherwise renaming a
  tier would orphan every model built under it.
* **The `smoke_subset` exemption is stamped by what happened, not by what was
  permitted.** A smoke run whose axes were all shorter than the cap dropped
  nothing and stays analysable.
* **The analysis tier refuses stamped measurements.** That is the second guard,
  and it is what stops a truncated axis reaching a figure.

Every permitted and forbidden path below is written out as a **literal**. A
suite that iterated `PERMITTED_TIER_PATHS` would agree with whatever that
constant says, including a version with `problem.*` in it — the trap that made
the golden-master containment unfalsifiable on its own capture host.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from mbl.core.runtime import Backend, ComputeContext, Precision
from mbl.engine.training_plan import OptimizerSpec, TrainingPlan
from mbl.spec.contender import ContenderSpec
from mbl.spec.data import DataSpec
from mbl.spec.errors import SpecificationError
from mbl.spec.evaluation import EvaluationSpec
from mbl.spec.gates import GateKind, GateSpec
from mbl.spec.problem import ProblemData, ProblemSpec
from mbl.spec.study import Composition, StudySpec, SweepAxis
from mbl.spec.tiers import (
    AXIS_SUBSET_KEY,
    DEFAULT_TIER_CATALOGUE,
    SmokeSubset,
    Tier,
    TierCatalogue,
    require_analysable_measurement,
)
from mbl.spec.training import BatchPlan, TrainingSpec

from .test_contender import REGISTRY
from .test_study import _contender, _protocol

N, M, HORIZON = 4, 2, 6
CTX = ComputeContext(backend=Backend.TORCH, device="cpu", precision=Precision.FLOAT64)

#: NB04's real shape: eight unrolled depths across six contenders, of which
#: only the three unfolded ones carry a depth at all.
DEPTHS = (1, 2, 3, 5, 8, 10, 15, 20)
DEPTH_SWEPT = ("standard_pgd", "unfolded_alpha", "unfolded_alpha_p")
DEPTH_INVARIANT = ("truncated_riccati", "cocp", "cocp_lower_bound")

#: The five standard tiers of Annex 01 §2.3, transcribed from the annex --
#: in effort order, which the ordering tests below read as meaningful.
#: `publication_b16k` (added 2026-08-09) is `publication` with the training
#: batch at 16384 and nothing else, so it ties `publication` on epochs.
STANDARD_TIERS = (
    "smoke",
    "standard",
    "publication",
    "publication_b16k",
    "comprehensive",
)


def _problem(seed: int = 0) -> ProblemSpec:
    rng = np.random.default_rng(seed)
    return ProblemSpec(
        ProblemData(
            system={"A": rng.normal(size=(N, N)), "B": rng.normal(size=(N, M))},
            cost={
                "Q": np.repeat(np.eye(N)[None], HORIZON + 1, axis=0),
                "R": np.repeat(np.eye(M)[None], HORIZON, axis=0),
            },
            horizon=HORIZON,
            control_bound=0.5,
        )
    )


def _study(**overrides: Any) -> StudySpec:
    fields: dict[str, Any] = {
        "id": "probe/tiered",
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
        "sweep": (
            SweepAxis(
                path="contenders.*.config.num_iterations",
                values=(2, 4, 8),
                applies_to=("unfolded_a",),
            ),
        ),
    }
    fields.update(overrides)
    return StudySpec(**fields)


def _nb04_study(**overrides: Any) -> StudySpec:
    """NB04, declaratively. The E1 defect was invisible to fixtures chosen by
    the author and obvious on the first real study shape, so the subsetting
    counts below are measured on that shape rather than on a toy."""
    fields: dict[str, Any] = {
        "id": "box_lqr/depth_scaling",
        "contenders": (
            _contender("truncated_riccati", "truncated_riccati"),
            _contender("unfolded_fixed", "standard_pgd"),
            _contender("unfolded", "unfolded_alpha"),
            _contender("unfolded_warmstart", "unfolded_alpha_p"),
            _contender("cocp", "cocp"),
            _contender("cocp_lower_bound", "cocp_lower_bound"),
        ),
        "sweep": (
            SweepAxis(
                path="contenders.*.config.num_iterations",
                values=DEPTHS,
                applies_to=DEPTH_SWEPT,
            ),
        ),
    }
    fields.update(overrides)
    return _study(**fields)


def _catalogue(**tiers: Tier) -> TierCatalogue:
    return TierCatalogue(tiers=tiers, version="test-1")


# --------------------------------------------------------------------------
# THE CHECKPOINT — the four forbidden classes, each of them
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("problem.data.horizon", 3),
        ("problem.data.control_bound", 10.0),
        ("problem", "anything"),
        ("contenders.*.family", "riccati"),
        ("contenders.*.config.num_iterations", 1),
        ("contenders", ()),
        ("sweep.0.values", (1, 2)),
        ("sweep.*.values", (1, 2)),
        ("sweep", ()),
        ("gates.0.config.min_fraction", 0.0),
        ("gates.*.config.max_spectral_radius", 99.0),
        ("gates", ()),
    ],
    ids=[
        "problem horizon",
        "problem control bound",
        "problem whole",
        "contender family",
        "contender config",
        "contenders whole",
        "sweep values indexed",
        "sweep values wildcard",
        "sweep whole",
        "gate threshold indexed",
        "gate threshold wildcard",
        "gates whole",
    ],
)
def test_a_forbidden_path_is_refused_at_parse_time(path: str, value: Any) -> None:
    """All four forbidden classes, in every spelling an author would reach for.
    One representative would leave three quarters of the rule unverified, and
    the rule *is* the forbidden column: a tier that could rewrite
    `sweep.*.values` or relax `gates` would let `publication` mean something the
    reader of the figure does not assume."""
    with pytest.raises(SpecificationError) as error:
        Tier(name="cheap", overrides={path: value})
    message = str(error.value)
    assert path in message, "the message must name the path that was refused"


@pytest.mark.parametrize(
    "root", ["problem", "contenders", "sweep", "gates"], ids=lambda r: r
)
def test_each_forbidden_root_is_a_real_field_of_a_study(root: str) -> None:
    """Anti-vacuity, and the reason the test above is a *negative control*
    rather than a coincidence. Each forbidden root addresses a field that
    genuinely exists, so the refusal has to come from the whitelist and cannot
    be a path that would have failed to resolve anyway."""
    assert hasattr(_study(), root)


# --------------------------------------------------------------------------
# ... and the permitted column, every path of it, against a real study
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "value", "reader"),
    [
        ("training.seeds", 3, lambda s: s.training.seeds),
        ("training.plan.epochs", 200, lambda s: s.training.plan.epochs),
        (
            "training.plan.optimizer.learning_rate",
            1e-4,
            lambda s: s.training.plan.optimizer.learning_rate,
        ),
        (
            "training.batch.effective_size",
            8192,
            lambda s: s.training.batch.effective_size,
        ),
        ("training.batch.microbatch", 512, lambda s: s.training.batch.microbatch),
        (
            "evaluation.protocol.n_batches",
            8,
            lambda s: s.evaluation.protocol.n_batches,
        ),
        (
            "evaluation.protocol.batch_spec.batch_size",
            4096,
            lambda s: s.evaluation.protocol.batch_spec.batch_size,
        ),
    ],
    ids=[
        "seeds",
        "epochs",
        "learning rate",
        "effective batch",
        "microbatch",
        "eval batches",
        "eval batch size",
    ],
)
def test_every_permitted_path_resolves_against_a_real_study(
    path: str, value: Any, reader: Any
) -> None:
    """The other half of the checkpoint, and the check that would have caught
    the three stale paths the annex carried: a whitelist entry naming a field
    the grammar does not have permits nothing and is enforced by nobody."""
    catalogue = _catalogue(cheap=Tier(name="cheap", overrides={path: value}))
    resolved = catalogue.resolve(_study(), "cheap")
    written = reader(resolved.study)
    assert written == value or written == tuple(range(value))


def test_a_permitted_subtree_does_not_permit_its_own_root() -> None:
    """`training.plan.*` means strictly below the plan. Replacing the plan
    wholesale could swap an end-to-end plan for a layer-wise one, which is
    content, not effort."""
    Tier(name="ok", overrides={"training.plan.epochs": 9})
    with pytest.raises(SpecificationError):
        Tier(name="bad", overrides={"training.plan": "anything"})


@pytest.mark.parametrize(
    ("path", "value"),
    [("statistics.bootstrap_resamples", 10000), ("profiling.timing_repeats", 20)],
    ids=["statistics", "profiling"],
)
def test_a_forward_looking_path_parses_but_does_not_resolve_yet(
    path: str, value: int
) -> None:
    """`statistics` and `profiling` are Tier 6/7 fields arriving in Stages 4-5.
    They stay whitelisted so adding them later is a grammar change and not also
    a policy change — and until they exist, writing one must fail naming the
    missing field rather than being silently dropped."""
    catalogue = _catalogue(rich=Tier(name="rich", overrides={path: value}))
    with pytest.raises(SpecificationError) as error:
        catalogue.resolve(_study(), "rich")
    assert path.split(".")[0] in str(error.value)


# --------------------------------------------------------------------------
# The seed count — the one per-path coercion
# --------------------------------------------------------------------------


def test_a_seed_count_resolves_to_that_many_seeds() -> None:
    """The annex writes `training.seeds: 5` against a `tuple[int, ...]` field
    and its tier table reads "Seeds | 5"; the value is a count."""
    catalogue = _catalogue(pub=Tier(name="pub", overrides={"training.seeds": 5}))
    assert catalogue.resolve(_study(), "pub").study.training.seeds == (0, 1, 2, 3, 4)


def test_an_explicit_seed_sequence_passes_through() -> None:
    """A tier that needs particular seeds must be able to say so, or the
    coercion becomes a ceiling on what a tier can express."""
    catalogue = _catalogue(odd=Tier(name="odd", overrides={"training.seeds": [7, 11]}))
    assert catalogue.resolve(_study(), "odd").study.training.seeds == (7, 11)


@pytest.mark.parametrize("count", [0, -1], ids=["zero", "negative"])
def test_a_non_positive_seed_count_is_refused_by_the_tier_itself(count: int) -> None:
    """Zero seeds trains nothing, which is a specification error rather than a
    very cheap tier.

    The message is asserted, not just the exception type. `range(0)` and
    `range(-1)` are both the empty tuple, which `TrainingSpec` already refuses
    on its own account -- so a version of `_coerce` with no lower bound at all
    still raises, from two frames away, saying "seeds is empty" and naming
    neither the tier nor the path the author has to go and edit. Only the
    message distinguishes the check that exists from the one that does not."""
    with pytest.raises(SpecificationError) as error:
        _catalogue(none=Tier(name="none", overrides={"training.seeds": count})).resolve(
            _study(), "none"
        )
    assert "training.seeds" in str(error.value)


def test_a_boolean_is_not_a_seed_count() -> None:
    """`True` is an `int` in Python, and a tier file written by hand is exactly
    where a stray `yes` becomes one. One seed by accident is a silently
    underpowered study."""
    with pytest.raises(SpecificationError):
        _catalogue(oops=Tier(name="oops", overrides={"training.seeds": True})).resolve(
            _study(), "oops"
        )


# --------------------------------------------------------------------------
# A tier writes effort, and effort is identity — but the LABEL never is
# --------------------------------------------------------------------------


def test_two_tiers_writing_the_same_fields_give_the_same_identifiers() -> None:
    """The tier participates in identity through the fields it sets, *never* as
    an opaque label. If the name leaked in, renaming a tier — or adding one that
    happens to agree with another — would orphan every model built under it."""
    overrides = {"training.plan.epochs": 40, "training.seeds": 2}
    catalogue = _catalogue(
        alpha=Tier(name="alpha", overrides=overrides),
        beta=Tier(name="beta", overrides=dict(overrides)),
    )
    study = _study()
    left = catalogue.resolve(study, "alpha").study
    right = catalogue.resolve(study, "beta").study
    assert left.study_id == right.study_id
    assert [p.model_id for p in left.materialise()] == [
        p.model_id for p in right.materialise()
    ]


def test_a_tier_that_scales_effort_does_change_the_models() -> None:
    """The positive direction. Without it the test above would also pass if
    `resolve` returned the study untouched.

    Written against `contenders.*.config.plan.epochs`, not `training.plan.*`:
    since G-2 the study plan is a template materialised into the contenders
    before anything else, so a tier writing it changes no identifier — which is
    correct, and would have made this assertion measure nothing.

    And the models it must change are the **learned** ones only. The baseline
    solves in closed form and is not trained at all, so an epoch count moving
    its `ModelID` would be the G-2 defect in miniature: it used to, because
    `training.plan` was in every model's signature whether or not the family
    could read it.
    """
    catalogue = _catalogue(
        light=Tier(name="light", overrides={"contenders.*.config.plan.epochs": 5}),
        heavy=Tier(name="heavy", overrides={"contenders.*.config.plan.epochs": 500}),
    )
    study = _study()

    def _models(tier: str, label: str) -> set[Any]:
        return {
            point.model_id
            for point in catalogue.resolve(study, tier).study.materialise()
            if point.contender.resolved_label == label
        }

    assert _models("light", "unfolded_a").isdisjoint(_models("heavy", "unfolded_a"))
    assert _models("light", "baseline") == _models("heavy", "baseline"), (
        "a closed-form baseline was re-identified by an epoch count it never reads"
    )


def test_a_tier_writing_only_the_template_changes_no_identifier() -> None:
    """The G-2 statement of F2a's finding, and the reason the shipped catalogue
    writes both paths. `training.plan.*` keeps a resolved document from
    declaring a 200-epoch plan while running 5 — a declaration being kept
    honest, which must cost nothing."""
    catalogue = _catalogue(
        light=Tier(name="light", overrides={"training.plan.epochs": 5}),
        heavy=Tier(name="heavy", overrides={"training.plan.epochs": 500}),
    )
    study = _study()
    light = catalogue.resolve(study, "light").study
    heavy = catalogue.resolve(study, "heavy").study
    assert light.study_id == heavy.study_id
    assert [p.model_id for p in light.materialise()] == [
        p.model_id for p in heavy.materialise()
    ]


def test_the_tier_name_appears_nowhere_in_the_signature_tree() -> None:
    """Structural companion to the two above. A behavioural test only covers
    the collisions this module thought of; the absence of the name covers every
    one there will ever be."""
    catalogue = _catalogue(
        publication=Tier(name="publication", overrides={"training.plan.epochs": 40})
    )
    tree = repr(catalogue.resolve(_study(), "publication").study.get_signature())
    assert "publication" not in tree
    assert "test-1" not in tree


def test_an_unknown_tier_is_refused_naming_the_available_ones() -> None:
    catalogue = _catalogue(smoke=Tier(name="smoke"))
    with pytest.raises(SpecificationError) as error:
        catalogue.resolve(_study(), "publicatoin")
    message = str(error.value)
    assert "publicatoin" in message and "smoke" in message


def test_a_catalogue_entry_must_be_filed_under_its_own_name() -> None:
    """A catalogue is edited by hand; a tier filed under the wrong key would be
    reported by name in provenance as a tier nobody can look up."""
    with pytest.raises(SpecificationError):
        TierCatalogue(tiers={"publication": Tier(name="standard")})


# --------------------------------------------------------------------------
# The per-invocation overlay — the same whitelist, one level more local
# --------------------------------------------------------------------------


def test_an_overlay_obeys_the_same_whitelist() -> None:
    """Annex 01 §2.3.1: all three editing levels write to the same closed set of
    paths. A `--set` that escaped the whitelist would make the catalogue's
    enforcement decorative."""
    catalogue = _catalogue(smoke=Tier(name="smoke"))
    with pytest.raises(SpecificationError) as error:
        catalogue.resolve(_study(), "smoke", overrides={"sweep.0.values": (1,)})
    assert "sweep.0.values" in str(error.value)


def test_an_overlay_overrides_the_catalogue() -> None:
    """Increasing locality is the whole point of having three levels."""
    catalogue = _catalogue(
        smoke=Tier(name="smoke", overrides={"training.plan.epochs": 5})
    )
    resolved = catalogue.resolve(
        _study(), "smoke", overrides={"training.plan.epochs": 7}
    )
    assert resolved.study.training.plan.epochs == 7


# --------------------------------------------------------------------------
# F2a — the tier must reach the plan that RUNS
# --------------------------------------------------------------------------
#
# `TrainableRecipe.build_engine` trains for the plan in
# `contenders.*.config.plan` (or `.schedule`). Before F2a a tier could only
# write `training.plan.*`, which no family reads: resolved at `smoke`, NB04's
# study plan dropped to 2 epochs while `unfolded_alpha` still declared 200 --
# the tier moved every identifier and changed no effort at all, inverting D15
# rather than bending it.


def _plan_epochs(study: StudySpec, label: str) -> int:
    """The epochs a contender will actually be trained for."""
    plan = dict(next(c for c in study.contenders if c.resolved_label == label).config)[
        "plan"
    ]
    return int(plan.epochs)


def _schedule(study: StudySpec, label: str) -> tuple[int, int]:
    config = dict(next(c for c in study.contenders if c.resolved_label == label).config)
    schedule = config["schedule"]
    return int(schedule.warmup_epochs_per_layer), int(schedule.refinement_epochs)


def test_a_tier_reaches_the_plan_each_contender_actually_trains_under() -> None:
    """The decisive check, on NB04's real shape.

    Read off the contenders, not off `training.plan`: reading the study plan is
    exactly what made the original measurement look correct while every learned
    contender trained for its declared 200 epochs.
    """
    catalogue = _catalogue(
        cheap=Tier(
            name="cheap",
            overrides={
                "contenders.*.config.plan.epochs": 2,
                "contenders.*.config.schedule.warmup_epochs_per_layer": 1,
                "contenders.*.config.schedule.refinement_epochs": 1,
            },
        )
    )
    resolved = catalogue.resolve(_nb04_study(), "cheap").study

    assert _plan_epochs(resolved, "unfolded_alpha") == 2
    assert _plan_epochs(resolved, "cocp") == 2
    assert _schedule(resolved, "unfolded_alpha_p") == (1, 1)


def test_a_contender_that_never_trains_is_left_alone() -> None:
    """Three of NB04's six carry no plan. Writing one onto them would invent a
    fitting procedure for a closed-form solve."""
    catalogue = _catalogue(
        cheap=Tier(name="cheap", overrides={"contenders.*.config.plan.epochs": 2})
    )
    resolved = catalogue.resolve(_nb04_study(), "cheap").study

    for label in DEPTH_INVARIANT:
        config = dict(
            next(c for c in resolved.contenders if c.resolved_label == label).config
        )
        if label == "cocp":  # depth-invariant, but it does train
            continue
        assert "plan" not in config, f"{label} was given a plan it never declared"


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("contenders.*.config.plan", "anything"),
        ("contenders.*.config.schedule", "anything"),
        ("contenders.*.config.num_iterations", 2),
        ("contenders.*.config.kind", "learned_step_size"),
        ("contenders.*.label", "renamed"),
    ],
    ids=["plan root", "schedule root", "depth", "policy kind", "label"],
)
def test_the_new_wildcard_is_permitted_by_leaf_and_not_by_subtree(
    path: str, value: Any
) -> None:
    """`contenders.*.config.plan.*` opens the fitting effort and nothing else.
    The unrolling depth is the study's independent variable, `kind` is the
    policy class, and replacing a plan wholesale swaps end-to-end for
    layer-wise -- all content, all still refused."""
    with pytest.raises(SpecificationError) as error:
        Tier(name="cheap", overrides={path: value})
    assert path in str(error.value)


def test_a_wildcard_that_matches_nothing_is_tolerated_from_the_catalogue() -> None:
    """The catalogue is shared across every study and cannot know which
    families any one of them declares. A study with no layer-wise contender
    must still run at a tier whose catalogue entry mentions `schedule`."""
    catalogue = _catalogue(
        cheap=Tier(
            name="cheap",
            overrides={"contenders.*.config.schedule.refinement_epochs": 1},
        )
    )
    # `_study()` declares `unfolded` and `truncated_riccati`: no schedule.
    resolved = catalogue.resolve(_study(), "cheap")
    assert resolved.study.materialise()


def test_a_wildcard_that_matches_nothing_is_refused_from_an_overlay() -> None:
    """...and refused one level in, because a per-study overlay and a `--set`
    are written by someone looking at *this* study. A path matching nobody is a
    typo, and a knob nobody turns that provenance reports as one that was is
    the whole defect F2a exists to close."""
    catalogue = _catalogue(cheap=Tier(name="cheap"))
    with pytest.raises(SpecificationError) as error:
        catalogue.resolve(
            _study(), "cheap", overrides={"contenders.*.config.plan.epcohs": 2}
        )
    message = str(error.value)
    assert "contenders.*.config.plan.epcohs" in message
    assert "no contender" in message


def test_the_overlay_refusal_does_not_fire_when_the_path_does_match() -> None:
    """Anti-vacuity for the test above: the same overlay shape, spelled
    correctly, must resolve. Otherwise the refusal could be unconditional."""
    catalogue = _catalogue(cheap=Tier(name="cheap"))
    resolved = catalogue.resolve(
        _study(), "cheap", overrides={"contenders.*.config.plan.epochs": 2}
    )
    assert _plan_epochs(resolved.study, "unfolded_a") == 2


def test_a_wildcard_write_is_the_same_as_writing_each_contender_by_hand() -> None:
    """Identity. The fan-out is a convenience for the author, never a term:
    a study tiered through the wildcard must derive exactly the identifiers of
    the study written out longhand."""
    catalogue = _catalogue(
        cheap=Tier(name="cheap", overrides={"contenders.*.config.plan.epochs": 4})
    )
    fanned = catalogue.resolve(_study(), "cheap").study

    longhand = _study(
        contenders=tuple(
            ContenderSpec(
                family=c.family,
                config={**dict(c.config), "plan": replace(c.config["plan"], epochs=4)},
                label=c.resolved_label,
                registry=REGISTRY,
            )
            if "plan" in c.config
            else c
            for c in _study().contenders
        )
    )

    assert fanned.study_id == longhand.study_id
    assert [str(p.model_id) for p in fanned.materialise()] == [
        str(p.model_id) for p in longhand.materialise()
    ]


def test_the_shipped_tiers_reach_the_plans_that_run() -> None:
    """The catalogue this repository ships, against NB04's real shape, at every
    tier. `test_the_shipped_tiers_are_ordered_by_the_effort_they_buy` reads
    `training.plan`, which no family executes -- this reads what does."""
    epochs = [
        _plan_epochs(DEFAULT_TIER_CATALOGUE.resolve(_nb04_study(), tier).study, "cocp")
        for tier in STANDARD_TIERS
    ]
    assert epochs == sorted(epochs), "the tiers are not ordered by executed effort"
    assert len(set(epochs)) > 1, "no tier changes the effort that is actually spent"
    # That `smoke` is *cheaper than the study declares* is asserted against the
    # tracked document rather than here, in `test_loader.py`: this fixture's
    # plan runs three epochs, which no tier is obliged to undercut.


def test_the_shipped_smoke_tier_reaches_the_layerwise_schedule_too() -> None:
    """`unfolded_warmstart` declares `schedule`, not `plan`, with no `epochs`
    in it at all -- so a catalogue that only wrote `plan.epochs` would leave
    NB04's flagship contender at its full 25-per-layer warm-up."""
    smoked = DEFAULT_TIER_CATALOGUE.resolve(_nb04_study(), "smoke").study
    assert _schedule(smoked, "unfolded_alpha_p") != _schedule(
        _nb04_study(), "unfolded_alpha_p"
    ), "the schedule path reached nothing"
    assert _plan_epochs(smoked, "unfolded_alpha") != _plan_epochs(
        _nb04_study(), "unfolded_alpha"
    ), "and neither did the plan path"


# --------------------------------------------------------------------------
# The smoke_subset exemption, and its two guards
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["standard", "publication", "publication_b16k", "comprehensive", "overnight"],
)
def test_only_smoke_may_declare_a_subset(name: str) -> None:
    """Covering every other standard tier plus a user-defined one, because the
    exemption is justified *only* by smoke models being disposable: smoke also
    cuts epochs and batch size, so its models differ in effective fields and
    are never reused by the later full run. No other tier can say that."""
    with pytest.raises(SpecificationError) as error:
        Tier(name=name, overrides={}, smoke_subset=SmokeSubset(max_points_per_axis=2))
    message = str(error.value)
    assert name in message and "smoke" in message


def test_smoke_may_declare_a_subset() -> None:
    assert Tier(name="smoke", smoke_subset=SmokeSubset(2)).smoke_subset is not None


@pytest.mark.parametrize(
    ("cap", "expected"),
    [(2, (1, 20)), (3, (1, 8, 20)), (8, DEPTHS), (20, DEPTHS)],
    ids=["endpoints", "endpoints plus middle", "exactly the axis", "cap above axis"],
)
def test_the_subset_keeps_evenly_spaced_values_including_both_endpoints(
    cap: int, expected: tuple[int, ...]
) -> None:
    """Endpoints because the extremes are where plumbing breaks: a depth of 1
    and a depth of 20 exercise different code paths, two adjacent middle values
    exercise one. An axis at or below the cap is left whole."""
    catalogue = _catalogue(
        smoke=Tier(name="smoke", smoke_subset=SmokeSubset(max_points_per_axis=cap))
    )
    resolved = catalogue.resolve(_nb04_study(), "smoke")
    assert resolved.study.sweep[0].values == expected


def test_subsetting_nb04_cuts_the_trainings_it_claims_to() -> None:
    """The exemption's whole justification is a number: a wiring check over
    eight depths x six contenders is 48 trainings, over two depths it is 12.
    Measured on NB04's real shape, where three contenders are depth-invariant,
    the true figures are 27 and 9 — and the claim is *counted*, not asserted."""
    full = _nb04_study().materialise()
    catalogue = _catalogue(smoke=Tier(name="smoke", smoke_subset=SmokeSubset(2)))
    subset = catalogue.resolve(_nb04_study(), "smoke").study.materialise()
    assert len({p.model_id for p in full}) == len(DEPTH_SWEPT) * len(DEPTHS) + len(
        DEPTH_INVARIANT
    )
    assert len({p.model_id for p in subset}) == len(DEPTH_SWEPT) * 2 + len(
        DEPTH_INVARIANT
    )


def test_a_subsetted_run_is_stamped() -> None:
    """Guard 1. Without the stamp the second guard has nothing to refuse."""
    catalogue = _catalogue(smoke=Tier(name="smoke", smoke_subset=SmokeSubset(2)))
    resolved = catalogue.resolve(_nb04_study(), "smoke")
    assert resolved.axis_subset is True
    assert resolved.provenance[AXIS_SUBSET_KEY] is True


def test_a_run_that_dropped_nothing_is_not_stamped() -> None:
    """The stamp records what happened, not what was permitted. Stamping on the
    mere presence of `smoke_subset` would bar results that are complete, which
    teaches authors to route around the guard rather than trust it."""
    catalogue = _catalogue(smoke=Tier(name="smoke", smoke_subset=SmokeSubset(8)))
    resolved = catalogue.resolve(_nb04_study(), "smoke")
    assert resolved.axis_subset is False
    assert resolved.provenance[AXIS_SUBSET_KEY] is False


def test_a_study_with_no_sweep_is_not_stamped() -> None:
    """The degenerate case: nothing to subset means nothing was lost."""
    catalogue = _catalogue(smoke=Tier(name="smoke", smoke_subset=SmokeSubset(2)))
    assert catalogue.resolve(_study(sweep=()), "smoke").axis_subset is False


def test_a_tier_without_the_exemption_never_stamps() -> None:
    catalogue = _catalogue(standard=Tier(name="standard"))
    assert catalogue.resolve(_nb04_study(), "standard").axis_subset is False


def test_subsetting_does_not_touch_the_scientific_content() -> None:
    """The exemption cuts *points*, never what a point is. A subset that also
    moved the problem or a contender family would be the content change the
    whole whitelist exists to prevent."""
    catalogue = _catalogue(smoke=Tier(name="smoke", smoke_subset=SmokeSubset(2)))
    study = _nb04_study()
    resolved = catalogue.resolve(study, "smoke").study
    assert resolved.problem.problem_id == study.problem.problem_id
    assert [c.family for c in resolved.contenders] == [
        c.family for c in study.contenders
    ]
    assert resolved.gates == study.gates


# --------------------------------------------------------------------------
# Guard 2 — the analysis tier refuses stamped measurements
# --------------------------------------------------------------------------


def test_the_analysis_tier_refuses_a_stamped_measurement() -> None:
    """No analysis, table or figure may consume a truncated axis. It fails
    loudly here rather than rendering a plausible, wrong figure."""
    with pytest.raises(SpecificationError) as error:
        require_analysable_measurement({AXIS_SUBSET_KEY: True, "tier": "smoke"})
    message = str(error.value)
    assert AXIS_SUBSET_KEY in message and "smoke" in message


@pytest.mark.parametrize(
    "provenance",
    [{}, {AXIS_SUBSET_KEY: False}, {"tier": "publication", AXIS_SUBSET_KEY: False}],
    ids=["unstamped", "stamped false", "full provenance"],
)
def test_the_analysis_tier_accepts_an_unsubsetted_measurement(
    provenance: dict[str, Any],
) -> None:
    """The positive direction, including the absent-stamp case: measurements
    predating the stamp must stay analysable, or the guard retires the store."""
    require_analysable_measurement(provenance)


def test_the_stamp_key_is_the_one_the_annex_names() -> None:
    """A by-name contract with the producer that writes it and the analysis
    tier that reads it, in the same family as `seeds` and `id`. Restated as a
    literal so a rename has to be a decision rather than a refactor."""
    assert AXIS_SUBSET_KEY == "axis_subset"


# --------------------------------------------------------------------------
# The shipped catalogue
# --------------------------------------------------------------------------


def test_the_default_catalogue_carries_the_five_standard_tiers() -> None:
    """Annex 01 §2.3's table, restated. E3 moves these values into a tracked
    `studies/_tiers.yaml`; the constant becomes its fallback."""
    assert set(DEFAULT_TIER_CATALOGUE.tiers) == set(STANDARD_TIERS)


def test_publication_b16k_differs_from_publication_in_the_batch_alone() -> None:
    """Annex 01 §2.3 (2026-08-09): 'everything else identical' is the tier's
    whole contract -- a second divergent field would make the revision's
    studies differ from the campaign's in something no plan costed. Asserted
    over the override DICTS, so a future edit to either tier re-decides this
    consciously rather than by drift."""
    publication = DEFAULT_TIER_CATALOGUE.tiers["publication"].overrides
    b16k = DEFAULT_TIER_CATALOGUE.tiers["publication_b16k"].overrides
    assert b16k["training.batch.effective_size"] == 16384
    assert publication["training.batch.effective_size"] == 8192
    stripped = {k: v for k, v in b16k.items() if k != "training.batch.effective_size"}
    assert stripped == {
        k: v for k, v in publication.items() if k != "training.batch.effective_size"
    }


@pytest.mark.parametrize(
    ("tier", "seeds"),
    [
        ("smoke", 1),
        ("standard", 1),
        ("publication", 5),
        ("publication_b16k", 5),
        ("comprehensive", 10),
    ],
)
def test_the_standard_tiers_declare_the_seed_counts_the_annex_publishes(
    tier: str, seeds: int
) -> None:
    resolved = DEFAULT_TIER_CATALOGUE.resolve(_study(), tier)
    assert len(resolved.study.training.seeds) == seeds


def test_only_the_smoke_tier_of_the_shipped_catalogue_subsets() -> None:
    subsetting = [
        name
        for name, tier in DEFAULT_TIER_CATALOGUE.tiers.items()
        if tier.smoke_subset is not None
    ]
    assert subsetting == ["smoke"]


@pytest.mark.parametrize("tier", STANDARD_TIERS)
def test_every_shipped_tier_resolves_against_a_real_study(tier: str) -> None:
    """A catalogue entry that cannot be applied is a tier nobody can run. This
    is the check that fails if a shipped tier is written against one of the
    stale annex paths."""
    resolved = DEFAULT_TIER_CATALOGUE.resolve(_nb04_study(), tier)
    assert resolved.study.materialise()
    assert resolved.provenance["tier"] == tier


def test_the_shipped_tiers_are_ordered_by_the_effort_they_buy() -> None:
    """The tiers mean something as a sequence — `standard` must not be more
    expensive than `publication`, or `--tier` is a lottery.

    Read off the contender, not off `training.plan`. Since G-2 the study plan
    is a template materialised into each contender, and a tier writing it
    changes no effort at all — so this assertion went on holding while
    measuring something nobody is charged for. The NB04-shaped version of the
    same claim lives above; this one keeps the small fixture's coverage.
    """
    epochs = [
        _plan_epochs(DEFAULT_TIER_CATALOGUE.resolve(_study(), tier).study, "unfolded_a")
        for tier in STANDARD_TIERS
    ]
    assert epochs == sorted(epochs)
    assert len(set(epochs)) > 1, "no tier changes the effort that is spent"


def test_the_catalogue_version_is_recorded_and_never_signed() -> None:
    """Annex 01 §2.3.1: `tier_catalogue_version` is recorded in provenance for
    diagnosing a retrain, and is informational — the effective fields already
    carry the identity."""
    resolved = DEFAULT_TIER_CATALOGUE.resolve(_study(), "smoke")
    assert (
        resolved.provenance["tier_catalogue_version"] == DEFAULT_TIER_CATALOGUE.version
    )
    assert "tier_catalogue_version" not in repr(resolved.study.get_signature())


# --------------------------------------------------------------------------
# The interaction the whitelist exists to make impossible
# --------------------------------------------------------------------------


def test_no_tier_can_relax_a_gate() -> None:
    """The end-to-end statement of D15's forbidden column, on a study that
    actually declares a gate: a cheaper run must not be a less-checked one."""
    study = _study(gates=(GateSpec(GateKind.STABILITY, {"max_spectral_radius": 1.0}),))
    with pytest.raises(SpecificationError):
        Tier(name="smoke", overrides={"gates.0.config.max_spectral_radius": 99.0})
    smoked = DEFAULT_TIER_CATALOGUE.resolve(study, "smoke").study
    assert smoked.gates == study.gates


def test_subsetting_keeps_a_coupled_pair_coupled() -> None:
    """`_subset_axes` rebuilds every axis, so an omitted `compose` would default
    to PRODUCT and silently turn a zipped pair into a grid.

    That is a *scientific* change made by a tier -- the one thing D15 forbids
    outright -- and it is invisible in every other subsetting test, because
    they all sweep a single axis. Here two coupled axes of eight values each
    must survive the cut as two coupled values, giving 2 points and not 4.
    """
    coupled = (
        SweepAxis(
            path="contenders.*.config.num_iterations",
            values=DEPTHS,
            applies_to=("unfolded_alpha",),
        ),
        SweepAxis(
            path="evaluation.protocol.n_batches",
            values=tuple(range(1, len(DEPTHS) + 1)),
            applies_to=("unfolded_alpha",),
            compose=Composition.ZIP,
        ),
    )
    study = _nb04_study(sweep=coupled)
    catalogue = _catalogue(smoke=Tier(name="smoke", smoke_subset=SmokeSubset(2)))
    subset = catalogue.resolve(study, "smoke").study

    assert [axis.compose for axis in subset.sweep] == [
        Composition.PRODUCT,
        Composition.ZIP,
    ]
    swept = [
        point
        for point in subset.materialise()
        if point.contender.resolved_label == "unfolded_alpha"
    ]
    assert len(swept) == 2, "a coupled pair subset to 2 is 2 points, not 4"
