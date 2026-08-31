"""The cross-branch identity probe: does this branch move a single identifier?

Run it against `main` and against the branch, and diff the two outputs. Any
line that differs is an identifier that changed, which means every stored model
or measurement filed under the old one is orphaned -- a failure no test *on the
branch* can see, because every test on the branch agrees with the new value.

    git worktree add /tmp/mbl-main main
    PYTHONPATH=/tmp/mbl-main/src uv run python tools/probes/identity_probe.py \
        > /tmp/main.txt
    uv run python tools/probes/identity_probe.py > /tmp/branch.txt
    diff /tmp/main.txt /tmp/branch.txt

**Nothing here may write to a store, touch a network, read a clock or consult
an environment variable.** The output must be a pure function of the source
tree and the two committed data files, or a diff means nothing.

Coverage, in the order printed:

1. the tracked problem's `ProblemID`;
2. every registered contender family, signed through `study.materialise()`
   rather than through a bare spec, because materialisation is what a run
   actually does (G-2 pushes the study's default training plan into each
   contender there, so a bare spec signs something no run ever signs);
3. a synthetic `StudyID` and point count over those families;
4. NB04's declared study, at every tier, with the `axis_subset` stamp;
5. four real NB04 point identifiers;
6. six legacy `Experiment.contender_content_digest` values -- the pre-Stage-2
   cache key, which the migration of the 3,225 legacy runs has to match;
7. the structural surface: `dataclasses.fields` of every spec type, and the
   three identity-key constants. A behavioural probe cannot see a field that
   was added with a default, because a default shifts every derived identifier
   *uniformly* and so shifts nothing relative to anything else.
"""

from __future__ import annotations

import dataclasses
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
PROBLEM_NPZ = REPO / "studies" / "box_lqr" / "box_lqr_n4m2_N50_u0.5_s0.npz"
NB04_STUDY = REPO / "studies" / "box_lqr" / "depth_scaling.toml"
TIERS = ("smoke", "standard", "publication", "comprehensive")
HORIZON = 50


def line(label: str, value: object) -> str:
    """One `label value` line. The whole output format."""
    return f"{label} {value}"


# -- 2. every registered family ---------------------------------------------


def family_configs(plan: Any, schedule: Any) -> dict[str, dict[str, Any]]:
    """A minimal *valid* configuration per registered family.

    Every entry is the constructor's required parameters and nothing else, so
    that a family gaining a required parameter breaks this probe loudly rather
    than being silently omitted from the sweep. The values are arbitrary and
    fixed; what is being probed is the signature machinery, not the numbers.
    """
    unrolled = {
        "num_iterations": 8,
        "step_size_init": 0.05,
        "step_size_max": 1.0,
        "horizon": HORIZON,
    }
    return {
        "riccati": {"horizon": HORIZON},
        "truncated_riccati": {"horizon": HORIZON},
        "unfolded_fixed": dict(unrolled),
        "unfolded": {"kind": "learned_step_size", "plan": plan, **unrolled},
        "unfolded_warmstart": {
            "kind": "learned_step_size_and_matrix",
            "schedule": schedule,
            **unrolled,
        },
        "neural": {"hidden_dim": 32, "plan": plan},
        "cocp": {"plan": plan},
        "cocp_exact": {"plan": plan},
        "cocp_exact_lower_bound": {"process_noise_std": 0.5},
        "cocp_lower_bound": {"process_noise_std": 0.5},
    }


def probe_families() -> Iterator[str]:
    """Sign all eight families through a real materialised study.

    Every buildable object goes through `bindings.build` with the same `_build`
    vocabulary a TOML document uses, rather than through its concrete
    constructor. A probe that instantiated `TrainingPlan` directly would not
    notice a change in the document surface, which is exactly the surface a
    study author edits.
    """
    from mbl.core.runtime.compute_context import ComputeContext
    from mbl.experiments.bindings import DEFAULT_SPEC_BINDINGS as bindings
    from mbl.spec import (
        ContenderSpec,
        DataSpec,
        EvaluationSpec,
        ProblemSpec,
        StudySpec,
        TrainingSpec,
    )
    from mbl.spec.training import BatchPlan

    ctx = ComputeContext(backend="torch", device="cpu", precision="float64")
    problem = ProblemSpec.load(PROBLEM_NPZ)
    yield line("problem_id", problem.problem_id)
    context = {"state_dim": problem.state_dim, "horizon": problem.horizon}

    plan = bindings.build(
        "end_to_end", {"optimizer": "adam", "learning_rate": 0.01, "epochs": 7}, context
    )
    schedule = bindings.build(
        "layerwise",
        {
            "optimizer": "adam",
            "learning_rate": 0.01,
            "warmup_epochs_per_layer": 3,
            "refinement_epochs": 11,
        },
        context,
    )
    batch_spec = bindings.build(
        "gaussian",
        {
            "batch_size": 64,
            "seed": 0,
            "process_noise_std": 0.5,
            "initial_state_std": 1.0,
        },
        context,
    )
    protocol = bindings.build(
        "protocol", {"n_batches": 2, "batch_spec": batch_spec}, context
    )

    registry = bindings.registry
    contenders = tuple(
        ContenderSpec(family=name, config=config, label=name, registry=registry)
        for name, config in sorted(family_configs(plan, schedule).items())
    )
    training = TrainingSpec(
        data=DataSpec(kind="gaussian", seed=0, params={"process_noise_std": 0.5}),
        batch=BatchPlan(effective_size=128),
        plan=plan,
        ctx=ctx,
        seeds=(0,),
    )
    evaluation = EvaluationSpec(
        problem=problem, protocol=protocol, ctx=ctx, metrics=("expected_cost",)
    )
    study = StudySpec(
        id="probe/all_families",
        problem=problem,
        contenders=contenders,
        training=training,
        evaluation=evaluation,
    )
    yield line("all_families.study_id", study.study_id)
    points = study.materialise()
    yield line("all_families.points", len(points))
    for point in points:
        label = point.contender.resolved_label
        yield line(f"all_families.{label}.model_id", point.model_id)
        yield line(f"all_families.{label}.measurement_id", point.measurement_id)

    # -- the rehost surface (Annex 01 §2.4.1, D24). One aware measurement
    # identifier, with an override, pinned beside the blind ones above: the
    # aware line must differ from its blind sibling, and every blind line in
    # this file must be byte-identical to the pre-field era -- that asymmetry
    # IS the conditional-signing rule, held by the golden.
    import dataclasses as _dataclasses

    from mbl.spec import RehostOverride
    from mbl.spec.identity import derive_measurement_id

    first = points[0]
    aware = _dataclasses.replace(
        evaluation,
        rehost="aware",
        rehost_overrides=(
            RehostOverride(
                contender=first.contender.resolved_label,
                problem=problem,
                config=(("horizon", 9),),
            ),
        ),
    )
    yield line(
        f"rehost.aware.{first.contender.resolved_label}.measurement_id",
        derive_measurement_id(first.model_id, aware),
    )


# -- 4/5. the tracked NB04 study --------------------------------------------


def _point_key(point: Any) -> str:
    """A stable name for one materialised point, independent of its position.

    **The seed is part of the name, and that is not cosmetic.** Until it was,
    every identifier this probe pinned came from seed 0, so a mutation that is a
    no-op at zero was invisible: `seed * 2` inside `derive_model_id` moved 0 of
    62 lines while orphaning 108 of the 162 stored models (67 %, at seeds 1-4).
    A seed-derived stream offset or a `seed << 1` has the same shape. When every
    pinned case carries the same value of a parameter, none of them tests it.
    """
    depth = point.axis_values.get("contenders.*.config.num_iterations")
    label = point.contender.resolved_label
    named = label if depth is None else f"{label}@J{depth}"
    return f"{named}#s{point.seed}"


def probe_nb04() -> Iterator[str]:
    """NB04 as declared, then at every tier, plus four real point ids."""
    from mbl.experiments.bindings import DEFAULT_SPEC_BINDINGS as bindings
    from mbl.spec.loader import load_study
    from mbl.spec.tiers import AXIS_SUBSET_KEY, DEFAULT_TIER_CATALOGUE

    document = load_study(NB04_STUDY, bindings=bindings)
    declared = document.study
    yield line("nb04.declared.study_id", declared.study_id)
    yield line("nb04.declared.points", len(declared.materialise()))

    for tier in TIERS:
        resolved = document.resolve(DEFAULT_TIER_CATALOGUE, tier)
        points = resolved.study.materialise()
        yield line(f"nb04.{tier}.study_id", resolved.study.study_id)
        yield line(f"nb04.{tier}.points", len(points))
        yield line(
            f"nb04.{tier}.axis_subset",
            bool(dict(resolved.provenance).get(AXIS_SUBSET_KEY, False)),
        )

    # Real points, chosen to cross every kind of contender the study has:
    # depth-variant learned, depth-variant analytic, depth-invariant learned,
    # and the bound. Selected BY NAME, so a change in declaration order shows
    # up as a changed identifier rather than silently re-labelling the rows.
    #
    # `standard` declares one seed, so every point it can offer is seed 0. The
    # two `publication` points are here for that reason alone: they are the only
    # thing in this probe that can see a mutation which is the identity at zero
    # and something else everywhere the store actually is (§A0).
    for tier, wanted in (
        (
            "standard",
            (
                "unfolded_alpha_p@J20#s0",
                "standard_pgd@J8#s0",
                "cocp#s0",
                "cocp_lower_bound#s0",
            ),
        ),
        ("publication", ("unfolded_alpha_p@J20#s3", "cocp#s4")),
    ):
        study = document.resolve(DEFAULT_TIER_CATALOGUE, tier).study
        by_key = {_point_key(p): p for p in study.materialise()}
        for key in wanted:
            point = by_key[key]
            yield line(f"nb04.point.{key}.model_id", point.model_id)
            yield line(f"nb04.point.{key}.measurement_id", point.measurement_id)


# -- 6. the legacy cache key -------------------------------------------------


def probe_legacy_digests() -> Iterator[str]:
    """`Experiment.contender_content_digest` for NB04's six contenders.

    This is the key the 3,225 legacy runs are filed under. The migration has to
    reproduce it exactly to recognise what it is salvaging, so a refactor that
    moves it is a refactor that silently orphans the archive.
    """
    from mbl.applications.studies.nb04_box_constrained import (
        UnfoldedModelConfig,
        build_nb04_config,
        nb04_box_constrained_experiment,
    )
    from mbl.experiments.bindings import DEFAULT_SPEC_BINDINGS as bindings

    context: dict[str, Any] = {"state_dim": 4, "horizon": HORIZON}
    plan = bindings.build(
        "end_to_end", {"optimizer": "adam", "learning_rate": 0.01, "epochs": 7}, context
    )
    schedule = bindings.build(
        "layerwise",
        {
            "optimizer": "adam",
            "learning_rate": 0.01,
            "warmup_epochs_per_layer": 3,
            "refinement_epochs": 11,
        },
        context,
    )
    # Fourteen keyword-only arguments, spelled out. `build_nb04_config` takes no
    # positional ones by design, and the count is why: a legacy digest computed
    # from a differently-shaped config is a different digest, so every field
    # this probe pins has to be visible in it.
    config = build_nb04_config(
        state_dim=4,
        control_dim=2,
        horizon=HORIZON,
        u_max=0.5,
        num_unfolding_iterations=8,
        step_size_init=0.05,
        step_size_max=1.0,
        batch_size=128,
        n_eval_batches=2,
        seed=0,
        process_noise_std=0.5,
        unfolded_alpha=UnfoldedModelConfig(
            kind="learned_step_size",
            training_mode="end_to_end",
            plan=plan,
            schedule=None,
        ),
        unfolded_alpha_p=UnfoldedModelConfig(
            kind="learned_step_size_and_matrix",
            training_mode="layerwise",
            plan=None,
            schedule=schedule,
        ),
        cocp_plan=plan,
    )
    experiment = nb04_box_constrained_experiment(config)
    for spec in experiment.contenders:
        yield line(
            f"legacy.{spec.label}.content_digest",
            experiment.contender_content_digest(spec),
        )


# -- 7. the structural surface ----------------------------------------------


def _spec_types() -> Iterator[tuple[str, type]]:
    """Every dataclass whose fields reach a stored identifier.

    Seven of these were missing until §A0 counted them, and their absence was
    worse than an orphan. A field added to one of them **with a default** that
    its hand-written `get_signature` does not emit makes two genuinely different
    configurations derive the *same* `ModelID`: `mbl run` finds the stored
    record, reuses it, and publishes a measurement attributing the result to a
    model never trained under the declared setting. An orphaned store is visible
    the moment someone looks at it; a collision is not.
    """
    from mbl.applications.factories import GaussianBatchSpec
    from mbl.core.runtime.compute_context import ComputeContext
    from mbl.engine.training_plan import (
        LayerwiseTrainingPlan,
        OptimizerSpec,
        TrainingPlan,
    )
    from mbl.spec import (
        AnalysisSpec,
        ContenderSpec,
        DataSpec,
        EvaluationSpec,
        FigureSpec,
        ProblemSpec,
        RehostOverride,
        StudySpec,
        SweepAxis,
        TrainingSpec,
    )
    from mbl.experiments.experiment import EvaluationProtocol
    from mbl.spec.gates import GateSpec
    from mbl.spec.problem import ProblemData
    from mbl.spec.training import BatchPlan

    for cls in (
        AnalysisSpec,
        BatchPlan,
        ComputeContext,
        ContenderSpec,
        DataSpec,
        EvaluationProtocol,
        EvaluationSpec,
        FigureSpec,
        GateSpec,
        GaussianBatchSpec,
        LayerwiseTrainingPlan,
        OptimizerSpec,
        # `ProblemSpec` only delegates; `ProblemData` is what holds the system,
        # the cost, the horizon and the control bound.
        ProblemData,
        ProblemSpec,
        RehostOverride,
        StudySpec,
        SweepAxis,
        TrainingPlan,
        TrainingSpec,
    ):
        yield cls.__name__, cls


def probe_structure() -> Iterator[str]:
    """Fields and identity constants -- what behaviour alone cannot see."""
    from mbl.spec.tiers import PERMITTED_TIER_PATHS
    from mbl.store.ids import STUDY_KEYS_EXCLUDED_FROM_STUDY_ID
    from mbl.store.maintenance import MEASUREMENT_IDENTITY_KEYS, MODEL_IDENTITY_KEYS

    for name, cls in _spec_types():
        fields = ",".join(f.name for f in dataclasses.fields(cls))
        yield line(f"fields.{name}", fields)
    yield line("keys.MODEL_IDENTITY_KEYS", ",".join(MODEL_IDENTITY_KEYS))
    yield line("keys.MEASUREMENT_IDENTITY_KEYS", ",".join(MEASUREMENT_IDENTITY_KEYS))
    yield line(
        "keys.STUDY_KEYS_EXCLUDED_FROM_STUDY_ID",
        ",".join(STUDY_KEYS_EXCLUDED_FROM_STUDY_ID),
    )
    yield line("keys.PERMITTED_TIER_PATHS", ",".join(PERMITTED_TIER_PATHS))


def surface_lines() -> list[str]:
    """The whole identity surface, in order, as text.

    A value rather than a side effect, so `tests/architecture/
    test_identity_stability.py` can assert it in-process. Measured: this call
    costs 0.11 s inside a warm pytest session against 5.2 s for the subprocess,
    which is why the gate is free and never opt-in.
    """
    return [
        *probe_families(),
        *probe_nb04(),
        *probe_legacy_digests(),
        *probe_structure(),
    ]


def main() -> int:
    for text in surface_lines():
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
