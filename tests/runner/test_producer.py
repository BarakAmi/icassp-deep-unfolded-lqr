"""Acceptance tests for the minimal producer — Stage 2 Phase F2.

Written before the implementation. The checkpoint class is **end-to-end
closure**, and it is the gate the whole re-architecture turns on: train a model,
then evaluate it under a different protocol *and* a different problem, and
assert that **zero training occurs** — measured at the real training call and
never at a stub.

The measurement is what makes it a gate rather than a claim. `synthesize` is an
injected seam of the producer's own signature, and the double that counts it
**wraps `default_synthesize` and calls it**; it never replaces it. A counter on
a fake would assert something about the fake, which is precisely the weakness
that motivated including a producer in Stage 2 at all (plan §2.2).

Three properties beyond the count carry the phase:

* **A model whose record carries weights is never re-synthesized.** Skipping is
  not the mechanism: `load_trained_controller` is handed a bound
  `build_controller` and nothing else, so training is unreachable from its
  inputs — the reason `derive_model_id`'s absent parameter is stronger than any
  behavioural test of it. A negative control over recipes whose
  `build_synthesizer` raises covers the indirect route, and it covers **exactly
  the weight-bearing families, computed from the store** rather than named in a
  list. A record with no weights is a receipt for a closed-form solve and is
  re-derived, which is bit-exact where rebuilding is not: plan §9.6.7.
* **The reuse path is faithful.** A model rebuilt from stored weights must score
  *exactly* what the freshly trained one scored, or the store is a cache that
  returns different answers. Guarded by an explicit check that the assertion is
  capable of failing: an unloaded rebuild must score differently.
* **Training reads the training distribution.** `experiments/runner.py:234`
  builds its harness from `evaluation.batch_spec`, which is parent §2.2's defect
  being executed. F1 gave `TrainingSpec` its own distribution; this is the first
  suite that can observe the trainer actually consuming it, and it observes it
  in the *weights*, not in the identifier — F1 already proved the identifier
  moves, which is a strictly weaker statement.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
import torch

from mbl.applications.recipes.base import (
    EngineHarness,
    ModelRecipe,
    TrainedControllerArtifact,
)
from mbl.core.optimal_control_problem import OptimalControlProblem
from mbl.core.runtime import ComputeContext, Precision
from mbl.core.runtime.exceptions import DeviceUnavailableError
from mbl.experiments import DEFAULT_SPEC_BINDINGS, EvaluationProtocol
from mbl.experiments.evaluation import evaluate_synthesized_controller
from mbl.models.lifecycle import SynthesizedController, Synthesizer
from mbl.persistence.null_tracker import NullExperimentTracker
import mbl.runner.producer as producer_module
from mbl.runner.producer import (
    Disposition,
    RunReport,
    default_synthesize,
    load_trained_controller,
    run_study,
    training_batch_spec,
)
from mbl.runner.seeds import derive_replicate_streams
from mbl.spec.naming import derive_semantic_name
from mbl.spec.errors import SpecificationError
from mbl.spec.loader import SpecBindings, load_study
from mbl.spec.problem import ProblemData, ProblemSpec
from mbl.spec.study import StudyPoint, StudySpec
from mbl.spec.tiers import (
    AXIS_SUBSET_KEY,
    DEFAULT_TIER_CATALOGUE,
    require_analysable_measurement,
)
from mbl.store.content_store import MeasurementStore, ModelStore
from mbl.store.index import StoreIndex
from mbl.store.naming import split_semantic_name
from mbl.store.maintenance import (
    MEASUREMENT_IDENTITY_KEYS,
    MODEL_IDENTITY_KEYS,
    _recomputed,
)

N, M, HORIZON = 4, 2, 6

#: The distribution the study *trains* on, and the one it is *scored* on. They
#: are deliberately different in every field: a producer that fed the trainer
#: the evaluation batches -- which is the defect this stage exists to fix --
#: would be invisible under one shared set of numbers.
TRAIN_NOISE, TRAIN_STREAM, TRAIN_BATCH = 0.35, 11, 32
EVAL_NOISE, EVAL_STREAM, EVAL_BATCH = 0.90, 7, 16

#: The point `_learned_weights` reads, named once so every caller reads the
#: same model rather than whichever identifier happened to sort first.
PROBE_DEPTH, PROBE_SEED = 2, 0


def _problem_file(directory: Path, name: str, seed: int) -> Path:
    """A real box-constrained LQR instance, frozen to `.npz` as D19 requires."""
    rng = np.random.default_rng(seed)
    spec = ProblemSpec(
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
    path = directory / name
    spec.save(path)
    return path


#: One real study over two contender shapes that behave differently in the
#: store: `unfolded_a` trains and has weights to reload, `baseline` solves in
#: closed form and has none. The depth axis names only the first, so the
#: analytic contender is emitted once per seed rather than once per depth.
#:
#: Held as a format template rather than mutated by `str.replace`: a fixture
#: built by substitution goes silently inert when its pattern stops matching.
STUDY = """
id = "probe/producer"

[problem]
path = "problem.npz"

[compute]
backend = "torch"
device = "cpu"
precision = "float64"

[training]
seeds = [0, 1]

[training.data]
kind = "gaussian"
seed = {train_stream}
process_noise_std = {train_noise}
initial_state_std = 0.8

[training.plan]
_build = "end_to_end"
optimizer = "adam"
learning_rate = 0.05
epochs = 8

[training.batch]
effective_size = {train_batch}

[evaluation]
metrics = ["expected_cost"]

[evaluation.protocol]
_build = "protocol"
n_batches = 2

[evaluation.protocol.batch_spec]
_build = "gaussian"
batch_size = {eval_batch}
seed = {eval_stream}
process_noise_std = {eval_noise}
initial_state_std = 1.0

[[contenders]]
label = "unfolded_a"
family = "unfolded"

[contenders.config]
kind = "learned_step_size"
num_iterations = 3
step_size_init = 0.1
step_size_max = 1.0
horizon = 6

[contenders.config.plan]
_build = "end_to_end"
optimizer = "adam"
learning_rate = 0.05
epochs = 8

[[contenders]]
label = "baseline"
family = "truncated_riccati"
role = "baseline"

[contenders.config]
horizon = 6

[[sweep]]
path = "contenders.*.config.num_iterations"
values = {depths}
applies_to = ["unfolded_a"]
"""

DEFAULTS: dict[str, Any] = {
    "train_stream": TRAIN_STREAM,
    "train_noise": TRAIN_NOISE,
    "train_batch": TRAIN_BATCH,
    "eval_stream": EVAL_STREAM,
    "eval_noise": EVAL_NOISE,
    "eval_batch": EVAL_BATCH,
    "depths": [2, 4],
}

#: Points the fixture expands to: two depths x one swept contender x two seeds,
#: plus one unswept contender x two seeds (deduplicated across the depth axis).
EXPECTED_POINTS = 6


def _write(directory: Path, /, **overrides: Any) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    _problem_file(directory, "problem.npz", seed=0)
    path = directory / "study.toml"
    path.write_text(STUDY.format(**{**DEFAULTS, **overrides}))
    return path


def _study(
    directory: Path, /, bindings: SpecBindings | None = None, **overrides: Any
) -> StudySpec:
    return load_study(
        _write(directory, **overrides), bindings=bindings or DEFAULT_SPEC_BINDINGS
    ).study


class CountingSynthesize:
    """Wraps the **real** training call and counts it; never replaces it.

    Also records the global torch seed as observed *inside* the call, which is
    the only vantage point from which "the producer made this point's training
    seed reach the stream" is a fact rather than a line of code.
    """

    def __init__(self) -> None:
        self.synthesizers: list[str] = []
        self.observed_seeds: list[int] = []

    def __call__(
        self,
        synthesizer: Synthesizer,
        problem: OptimalControlProblem,
        ctx: ComputeContext,
    ) -> SynthesizedController:
        self.synthesizers.append(type(synthesizer).__name__)
        self.observed_seeds.append(torch.initial_seed())
        return default_synthesize(synthesizer, problem, ctx)

    @property
    def calls(self) -> int:
        return len(self.synthesizers)


def _run(
    study: StudySpec,
    store: Path,
    counter: CountingSynthesize | None = None,
    *,
    bindings: SpecBindings | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> RunReport:
    return run_study(
        study,
        store=store,
        bindings=bindings or DEFAULT_SPEC_BINDINGS,
        synthesize=counter or default_synthesize,
        provenance=provenance,
    )


def _record_digests(store: Path) -> dict[str, dict[str, str]]:
    """Every published record's manifest, so "republished nothing" is a
    measurement of the bytes rather than of a counter the producer keeps."""
    digests: dict[str, dict[str, str]] = {}
    for kind in ("models", "measurements"):
        root = store / kind
        for record in sorted(root.iterdir()) if root.is_dir() else ():
            manifest = record / "manifest.json"
            if manifest.is_file():
                digests[f"{kind}/{record.name}"] = json.loads(manifest.read_text())
    return digests


def _protocol(study: StudySpec) -> EvaluationProtocol:
    """The study's protocol, narrowed. `EvaluationSpec.protocol` is typed
    `Signable` so that Tier 3 need not import `experiments`; a test may know
    what the loader actually built."""
    return cast(EvaluationProtocol, study.evaluation.protocol)


def _shifted(study: StudySpec, other_problem: Path) -> StudySpec:
    """The same study, scored on a different problem under a different
    protocol. Both halves at once, because §7.2's gate names both."""
    protocol = _protocol(study)
    return replace(
        study,
        evaluation=replace(
            study.evaluation,
            problem=ProblemSpec.load(other_problem),
            protocol=replace(protocol, n_batches=protocol.n_batches * 3),
        ),
    )


def _probe_point(study: StudySpec) -> StudyPoint:
    """The one learned point every weight comparison reads."""
    return next(
        point
        for point in study.materialise()
        if point.contender.resolved_label == "unfolded_a"
        and point.seed == PROBE_SEED
        and point.contender.config["num_iterations"] == PROBE_DEPTH
    )


def _learned_weights(study: StudySpec, store: Path) -> Mapping[str, torch.Tensor]:
    """The probe model's weights, freshly trained into `store`."""
    _run(study, store)
    return ModelStore(store).get(str(_probe_point(study).model_id)).weights


def _checkpointed(store: Path) -> set[str]:
    """The stored models that carry weights, read off the store itself.

    The producer's two reuse paths are chosen by this exact predicate, so every
    test that reasons about them measures it rather than restating which
    families are supposed to learn.
    """
    models = ModelStore(store)
    return {mid for mid in models.list_ids() if models.get(mid).weights}


def _checkpointed_families(store: Path) -> set[str]:
    models = ModelStore(store)
    return {
        str(models.spec(mid)["family"])
        for mid in models.list_ids()
        if models.get(mid).weights
    }


# --------------------------------------------------------------------------
# The study runs, and what it leaves behind
# --------------------------------------------------------------------------


def test_every_point_leaves_one_model_and_one_measurement(tmp_path: Path) -> None:
    study = _study(tmp_path / "doc")
    store = tmp_path / "store"
    report = _run(study, store)

    points = study.materialise()
    assert len(points) == EXPECTED_POINTS, (
        "if the fixture's shape changed, every count below is measuring "
        "something other than what it names"
    )
    assert len(report.outcomes) == len(points)
    assert {str(o.model_id) for o in report.outcomes} == {
        str(p.model_id) for p in points
    }
    assert sorted(ModelStore(store).list_ids()) == sorted(
        {str(p.model_id) for p in points}
    )
    assert sorted(MeasurementStore(store).list_ids()) == sorted(
        {str(p.measurement_id) for p in points}
    )

    index = StoreIndex(store)
    assert len(index.models()) == len(points)
    assert len(index.measurements()) == len(points)
    assert all(o.metrics["eval_expected_cost"] > 0.0 for o in report.outcomes)


def test_the_first_run_trains_every_point_and_says_so(tmp_path: Path) -> None:
    study = _study(tmp_path / "doc")
    counter = CountingSynthesize()
    report = _run(study, tmp_path / "store", counter)

    assert counter.calls == EXPECTED_POINTS
    assert all(o.model is Disposition.TRAINED for o in report.outcomes)
    assert all(o.measurement is Disposition.EVALUATED for o in report.outcomes)


# --------------------------------------------------------------------------
# THE DECISIVE GATE
# --------------------------------------------------------------------------


def test_evaluating_differently_trains_nothing(tmp_path: Path) -> None:
    """Parent §7.2's gate, and the reason the whole identity split exists.

    A different evaluation problem *and* a different protocol, over models
    already in the store. The count is taken at the real training call.
    """
    document = tmp_path / "doc"
    study = _study(document)
    other = _problem_file(document, "other.npz", seed=99)
    store = tmp_path / "store"

    counter = CountingSynthesize()
    _run(study, store, counter)
    trained = counter.calls
    assert trained == EXPECTED_POINTS

    # What the store actually holds a checkpoint for. Measured, not assumed:
    # a record with no weights is a receipt for a closed-form solve and is
    # re-derived rather than reloaded (§9.6.7), so it is the *checkpointed*
    # models the gate is a claim about.
    checkpointed = _checkpointed(store)
    assert 0 < len(checkpointed) < trained, (
        "the fixture must hold both kinds of record, or this gate is measuring "
        "only one of the two reuse paths"
    )

    shifted = _shifted(study, other)
    assert shifted.evaluation.problem.problem_id != study.problem.problem_id, (
        "the shifted study must actually be scored on another problem, or the "
        "gate below is measured against a study that never shifted"
    )
    assert _protocol(shifted).n_batches != _protocol(study).n_batches
    report = _run(shifted, store, counter)

    resynthesized = counter.calls - trained
    assert resynthesized == trained - len(checkpointed), (
        f"evaluating differently ran {resynthesized} offline phase(s); only the "
        f"{trained - len(checkpointed)} weightless record(s) may be re-derived, "
        "and no checkpoint may be"
    )
    assert all(o.model is Disposition.REUSED for o in report.outcomes)
    assert all(o.measurement is Disposition.EVALUATED for o in report.outcomes)

    index = StoreIndex(store)
    assert len(index.measurements(shifted=True)) == EXPECTED_POINTS, (
        "every shifted measurement must be indexed as shifted; `is_shifted` is "
        "derived from the model's own training problem, so this also proves the "
        "producer indexed each model before its measurement (trap §11.3)"
    )
    assert len(index.models()) == trained, "scoring added a model row"


def test_re_running_the_whole_study_trains_nothing_and_publishes_nothing(
    tmp_path: Path,
) -> None:
    """Trap §11.1: `ModelStore.put` refuses differing bytes under an existing
    id, so the reuse path must check `exists()` *before* training rather than
    publish-and-reconcile after. D20 permits two runs of one `ModelID` to differ
    at float-reduction tolerance, so publish-and-reconcile would eventually
    raise `ContentConflictError` on a legitimate re-run.
    """
    study = _study(tmp_path / "doc")
    store = tmp_path / "store"
    counter = CountingSynthesize()
    _run(study, store, counter)
    trained = counter.calls
    before = _record_digests(store)
    assert before, "nothing was published, so 'republished nothing' is vacuous"

    report = _run(study, store, counter)  # must not raise ContentConflictError

    assert counter.calls == trained, "a second identical run trained something"
    assert _record_digests(store) == before, "a second identical run rewrote bytes"
    assert all(o.model is Disposition.REUSED for o in report.outcomes)
    assert all(o.measurement is Disposition.REUSED for o in report.outcomes), (
        "a point whose measurement is already stored must not be re-evaluated; "
        "re-scoring and re-putting identical bytes passes the letter of "
        "'republishes nothing' while paying for every rollout twice (§9.6.4)"
    )


def test_a_shifted_measurement_actually_scores_the_shifted_problem(
    tmp_path: Path,
) -> None:
    """The other half of the gate, and the half a disposition cannot show.

    Reusing the model and filing a measurement under a shifted identifier is
    worthless if the rollout still ran on the training problem: the store would
    report a distribution-shift result computed in-distribution. So the numbers
    themselves must differ.
    """
    document = tmp_path / "doc"
    study = _study(document)
    store = tmp_path / "store"
    nominal = _run(study, store)

    # The problem alone, holding the protocol fixed. `_shifted` moves the batch
    # count too, and a producer scoring on the *training* problem would still
    # report different numbers under it -- from averaging over three times as
    # many batches, not from the shift. Isolating the problem is what makes
    # this capable of failing.
    rehosted = replace(
        study,
        evaluation=replace(
            study.evaluation,
            problem=ProblemSpec.load(_problem_file(document, "other.npz", seed=99)),
        ),
    )
    assert _protocol(rehosted).get_signature() == _protocol(study).get_signature()
    shifted = _run(rehosted, store)

    by_model = {str(o.model_id): o.metrics for o in nominal.outcomes}
    assert set(by_model) == {str(o.model_id) for o in shifted.outcomes}
    for outcome in shifted.outcomes:
        assert outcome.metrics != by_model[str(outcome.model_id)], (
            f"{outcome.label}: the shifted measurement scored exactly what the "
            "nominal one did, so the evaluation problem never reached the rollout"
        )


def test_a_present_model_with_an_absent_measurement_is_the_shifted_case(
    tmp_path: Path,
) -> None:
    """The two reuse checks are independent, and this is the combination the
    split exists for: reuse the model, compute the measurement."""
    document = tmp_path / "doc"
    study = _study(document)
    store = tmp_path / "store"
    _run(study, store)

    rescored = _shifted(study, _problem_file(document, "other.npz", seed=99))
    report = _run(rescored, store)

    assert {(o.model, o.measurement) for o in report.outcomes} == {
        (Disposition.REUSED, Disposition.EVALUATED)
    }


def test_renaming_a_contender_costs_nothing(tmp_path: Path) -> None:
    """Phase G-1, stated end to end rather than as a digest comparison.

    A contender's `label` is the name a figure axis is drawn with, and it
    reached `ModelID` through the recipe's signature tree — so renaming
    `unfolded_a` to `alpha` retrained every model it had ever produced. The
    store is the only place that claim can be settled: rename, re-run, and the
    training count must not move.
    """
    study = _study(tmp_path / "doc")
    store = tmp_path / "store"
    counter = CountingSynthesize()
    _run(study, store, counter)
    trained = counter.calls
    before = _record_digests(store)

    # A sweep axis names the contenders it applies to, and `StudySpec` refuses
    # one naming a contender the study does not have -- so a rename is an edit
    # in two places, exactly as it would be in the document. That the axis
    # still resolves afterwards is part of what is being checked.
    def rename(label: str) -> str:
        return f"renamed_{label}"

    renamed = replace(
        study,
        contenders=tuple(
            replace(c, label=rename(c.resolved_label)) for c in study.contenders
        ),
        sweep=tuple(
            replace(axis, applies_to=tuple(rename(a) for a in axis.applies_to))
            for axis in study.sweep
        ),
    )
    assert {c.resolved_label for c in renamed.contenders} != {
        c.resolved_label for c in study.contenders
    }, "the rename did not take, so this measures nothing"
    assert len(renamed.materialise()) == len(study.materialise()), (
        "the renamed study expands to a different number of points, so the "
        "counts below are not comparable"
    )

    report = _run(renamed, store, counter)

    assert counter.calls == trained, (
        f"renaming trained {counter.calls - trained} model(s); a label is what "
        "a figure calls a contender, not what makes it a different model"
    )
    assert all(o.model is Disposition.REUSED for o in report.outcomes)
    assert all(o.measurement is Disposition.REUSED for o in report.outcomes), (
        "the measurements moved too, so a rename still invalidates results"
    )
    assert _record_digests(store) == before, "renaming republished bytes"


# --------------------------------------------------------------------------
# The reuse path never touches a Synthesizer
# --------------------------------------------------------------------------


def test_the_reuse_path_cannot_reach_a_synthesizer() -> None:
    """The structural half, and the stronger one.

    `load_trained_controller` is handed the single capability it needs -- a
    bound `build_controller` -- and there is no parameter through which a
    `Synthesizer`, a recipe that could build one, or the `synthesize` seam
    could be passed. Asserted as an EXACT parameter set, so adding such a
    parameter fails here even before anything calls it: the same argument that
    makes `derive_model_id`'s missing evaluation parameter stronger than any
    behavioural test of it.
    """
    parameters = inspect.signature(load_trained_controller).parameters
    assert set(parameters) == {"build_controller", "problem", "ctx", "record"}

    annotations = " ".join(str(p.annotation) for p in parameters.values())
    for forbidden in ("Synthesizer", "Synthesize", "Recipe", "harness", "Harness"):
        assert forbidden not in annotations


def test_a_checkpointed_model_is_never_re_synthesized(tmp_path: Path) -> None:
    """The negative control, covering what a parameter set cannot: an
    *indirect* route to a synthesizer.

    Every recipe of a **checkpointed** family is replaced by one whose
    `build_synthesizer` detonates -- and which families those are is read off
    the store rather than named here, so a family that starts or stops
    persisting weights cannot drift out of coverage.
    """
    document = tmp_path / "doc"
    store = tmp_path / "store"
    _run(_study(document), store)

    families = _checkpointed_families(store)
    assert families, "nothing was checkpointed, so this control guards nothing"

    exploding = _ExplodingBindings(DEFAULT_SPEC_BINDINGS, disarm=families)
    disarmed = load_study(document / "study.toml", bindings=exploding).study
    assert {str(p.model_id) for p in disarmed.materialise()} == set(
        ModelStore(store).list_ids()
    ), "disarming moved the identifiers, so this is not the same study"

    wrapped_before = exploding.recipes_disarmed
    report = _run(disarmed, store, bindings=exploding)

    assert exploding.recipes_disarmed > wrapped_before, (
        "the run resolved no checkpointed recipe through the disarming "
        "registry, so this control would have passed whatever the reuse path did"
    )
    assert all(o.model is Disposition.REUSED for o in report.outcomes)


def test_the_control_does_fire_on_the_training_path(tmp_path: Path) -> None:
    """...and that control is only worth having if it can fail. Against an
    empty store the very same bindings must detonate."""
    exploding = _ExplodingBindings(DEFAULT_SPEC_BINDINGS, disarm={"unfolded"})
    study = _study(tmp_path / "doc", bindings=exploding)
    with pytest.raises(AssertionError, match="build_synthesizer"):
        _run(study, tmp_path / "store", bindings=exploding)


def test_a_weightless_record_is_re_derived_rather_than_rebuilt(
    tmp_path: Path,
) -> None:
    """The other half of §9.6.7, stated as the reason it exists.

    `AnalyticRecipe.build_controller` ignores the `ComputeContext` it is handed
    and solves on NumPy, while the synthesizer honours it. So for a family with
    no weights to overwrite the difference away, rebuilding would score
    something the trained run did not -- and the producer re-derives instead.
    """
    document = tmp_path / "doc"
    study = _study(document)
    store = tmp_path / "store"
    _run(study, store)

    weightless = set(ModelStore(store).list_ids()) - _checkpointed(store)
    assert weightless, "the fixture has no weightless contender to test with"

    point = next(p for p in study.materialise() if str(p.model_id) in weightless)
    problem = point.problem.build()
    ctx = point.training.ctx
    recipe = cast(ModelRecipe, point.contender.resolve())

    rebuilt = recipe.build_controller(problem, ctx)
    synthesized = recipe.build_synthesizer(
        problem,
        EngineHarness(
            batch_spec=training_batch_spec(
                point.training,
                point.problem,
                DEFAULT_SPEC_BINDINGS,
                stream=derive_replicate_streams(
                    point.training.data.seed, point.seed
                ).data,
            ),
            tracker=NullExperimentTracker(),
            ctx=ctx,
        ),
    ).synthesize(problem, ctx)

    assert isinstance(synthesized.K_arr, torch.Tensor)
    assert isinstance(rebuilt.K_arr, np.ndarray)
    assert not np.array_equal(
        synthesized.K_arr.detach().cpu().numpy(),
        rebuilt.K_arr,
    ), (
        "the two construction paths now agree, so the re-derivation this test "
        "justifies is no longer needed -- see plan §9.6.7 before deleting it"
    )


# --------------------------------------------------------------------------
# The reuse path is faithful
# --------------------------------------------------------------------------


def test_a_reused_model_scores_exactly_what_the_trained_one_scored(
    tmp_path: Path,
) -> None:
    """Independent recomputation. One evaluation reached two ways: once against
    a model trained in that very run, once against the same model rebuilt from
    stored weights. A store that answers differently from the computation it
    replaces is worse than no store at all.
    """
    document = tmp_path / "doc"
    study = _study(document)
    rescored = _shifted(study, _problem_file(document, "other.npz", seed=99))

    fresh = _run(rescored, tmp_path / "fresh")  # trains, then evaluates

    warm = tmp_path / "warm"
    _run(study, warm)  # trains under the nominal evaluation
    reused = _run(rescored, warm)  # reuses those models, evaluates shifted

    assert all(o.model is Disposition.TRAINED for o in fresh.outcomes)
    assert all(o.model is Disposition.REUSED for o in reused.outcomes)

    by_id = {str(o.measurement_id): o.metrics for o in fresh.outcomes}
    assert set(by_id) == {str(o.measurement_id) for o in reused.outcomes}
    for outcome in reused.outcomes:
        assert outcome.metrics == by_id[str(outcome.measurement_id)], (
            f"{outcome.label}: the rebuilt model scored differently from the "
            "trained one"
        )


def test_that_the_fidelity_equality_can_fail(tmp_path: Path) -> None:
    """The guard on the test above.

    If eight epochs barely moved the parameters, a reuse path that forgot to
    load them would score the same and the equality would hold under a broken
    implementation. So: the stored weights must differ from a freshly
    constructed controller's, and an unloaded rebuild must score differently.
    """
    study = _study(tmp_path / "doc")
    store = tmp_path / "store"
    report = _run(study, store)

    point = _probe_point(study)
    trained = next(o for o in report.outcomes if str(o.model_id) == str(point.model_id))
    record = ModelStore(store).get(str(point.model_id))
    assert record.weights, "the learned contender stored no weights at all"

    problem = point.problem.build()
    ctx = point.training.ctx
    recipe = cast(ModelRecipe, point.contender.resolve())
    naked = recipe.build_controller(problem, ctx)
    initial = dict(naked.as_module().state_dict())

    assert set(initial) == set(record.weights)
    assert any(not torch.equal(initial[key], record.weights[key]) for key in initial), (
        "training moved no parameter at all, so loading them cannot matter"
    )

    batches = _protocol(study).build_batches(study.evaluation.ctx)
    unloaded, _ = evaluate_synthesized_controller(
        TrainedControllerArtifact(naked, ctx, {}), problem, batches
    )
    assert unloaded != trained.metrics, (
        "a rebuild that never loaded the stored weights scored identically, so "
        "the fidelity assertion cannot fail and is evidence of nothing"
    )


def test_a_partial_checkpoint_is_refused_rather_than_loaded(tmp_path: Path) -> None:
    """`dense_state_dict` drops non-strided entries when publishing, so the
    reload cannot be `strict=True` -- a cone layer's compiled constraint
    matrices are sparse and would make every such family unloadable. It must
    not therefore be lenient: a checkpoint missing a *parameter* would produce
    a plausible controller with untrained values, scored under an identifier
    claiming it was trained.
    """
    study = _study(tmp_path / "doc")
    store = tmp_path / "store"
    _run(study, store)

    point = _probe_point(study)
    record = ModelStore(store).get(str(point.model_id))
    recipe = cast(ModelRecipe, point.contender.resolve())
    problem = point.problem.build()
    dropped = sorted(record.weights)[0]

    with pytest.raises(SpecificationError, match=dropped):
        load_trained_controller(
            recipe.build_controller,
            problem,
            point.training.ctx,
            replace(
                record,
                weights={k: v for k, v in record.weights.items() if k != dropped},
            ),
        )

    with pytest.raises(SpecificationError, match="not_a_parameter"):
        load_trained_controller(
            recipe.build_controller,
            problem,
            point.training.ctx,
            replace(
                record,
                weights={**record.weights, "not_a_parameter": torch.zeros(1)},
            ),
        )


def test_weights_on_a_family_that_cannot_hold_them_are_refused(
    tmp_path: Path,
) -> None:
    """The store and the recipe must agree about whether a family learns
    anything. A weightless controller handed a checkpoint would drop it in
    silence."""
    study = _study(tmp_path / "doc")
    store = tmp_path / "store"
    _run(study, store)

    analytic = next(
        p
        for p in study.materialise()
        if p.contender.resolved_label == "baseline" and p.seed == PROBE_SEED
    )
    record = ModelStore(store).get(str(analytic.model_id))
    assert not record.weights, "the analytic contender stored weights after all"

    with pytest.raises(SpecificationError, match="exposes no module"):
        load_trained_controller(
            cast(ModelRecipe, analytic.contender.resolve()).build_controller,
            analytic.problem.build(),
            analytic.training.ctx,
            replace(record, weights={"invented": torch.zeros(1)}),
        )


def test_a_record_that_changed_category_is_refused(tmp_path: Path) -> None:
    """A weightless record is re-derived on the assumption that it is a receipt
    for a closed-form solve. If re-deriving it produces weights, that
    assumption is false and the record is not what the producer would write
    today -- refused rather than silently re-derived (§9.6.7)."""
    study = _study(tmp_path / "doc")
    store = tmp_path / "store"
    _run(study, store)

    point = _probe_point(study)
    models = ModelStore(store)
    original = models.get(str(point.model_id))
    assert original.weights

    # Rewrite the learned model's record as though it had been published by a
    # family that learns nothing. `put` refuses differing bytes under a live
    # id, so the record is removed first -- which is also the only way this
    # inconsistency could arise in a real store.
    models.delete(str(point.model_id))
    models.put(replace(original, weights={}))
    # Its measurement too: a point whose model AND measurement are both stored
    # costs nothing and never reaches the reuse path at all, so leaving the
    # measurement behind would make this test pass without exercising anything.
    MeasurementStore(store).delete(str(point.measurement_id))

    with pytest.raises(SpecificationError, match="stored with no weights"):
        _run(study, store)


# --------------------------------------------------------------------------
# Training reads the TRAINING distribution — parent §2.2, executed correctly
# --------------------------------------------------------------------------


def test_the_trainer_consumes_the_training_distribution(tmp_path: Path) -> None:
    """`experiments/runner.py:234` builds its harness from
    `evaluation.batch_spec`. This is the first caller able to write that line
    correctly, and the proof is in the *weights*: two studies differing only in
    `training.data.process_noise_std` must fit different models.

    F1 already showed the two derive different `ModelID`s. That is strictly
    weaker -- an identifier can move while the trainer still consumes the
    evaluation batches, which is exactly the state this stage inherited.
    """
    quiet = _study(tmp_path / "quiet", train_noise=0.05)
    loud = _study(tmp_path / "loud", train_noise=1.50)

    assert quiet.evaluation.get_signature() == loud.evaluation.get_signature(), (
        "the two studies must differ in the TRAINING distribution alone"
    )
    assert quiet.training.data.params != loud.training.data.params

    quiet_weights = _learned_weights(quiet, tmp_path / "quiet-store")
    loud_weights = _learned_weights(loud, tmp_path / "loud-store")

    assert set(quiet_weights) == set(loud_weights)
    assert any(
        not torch.equal(quiet_weights[key], loud_weights[key]) for key in quiet_weights
    ), (
        "training under two different process-noise levels fitted identical "
        "weights, so the trainer is not reading training.data at all"
    )


def test_the_trainer_consumes_the_declared_batch_count(tmp_path: Path) -> None:
    """The other half of the composition: `BatchPlan.effective_size` is where
    the count lives (F1), so it -- and not the evaluation's `batch_size` -- must
    be what the gradient is averaged over."""
    small = _study(tmp_path / "small", train_batch=8)
    large = _study(tmp_path / "large", train_batch=96)

    assert small.evaluation.get_signature() == large.evaluation.get_signature()

    assert any(
        not torch.equal(a, b)
        for a, b in zip(
            _learned_weights(small, tmp_path / "small-store").values(),
            _learned_weights(large, tmp_path / "large-store").values(),
            strict=True,
        )
    ), "the effective batch size did not reach the trainer"


@pytest.mark.parametrize("restated", ["batch_size", "seed"])
def test_a_distribution_may_not_restate_what_it_does_not_own(
    tmp_path: Path, restated: str
) -> None:
    """Degeneracy, and F1's decision enforced where it is composed.

    The count lives in `BatchPlan.effective_size` and the stream in
    `DataSpec.seed`. A `params` entry claiming either would be silently
    overridden by the producer's composition -- an author's declaration lost
    without a word, which is the failure E3 refused a restated dimension for.
    Both keys, because a rule that applies twice is verified twice.
    """
    study = _study(tmp_path / "doc")
    broken = replace(
        study,
        training=replace(
            study.training,
            data=replace(
                study.training.data,
                params={**study.training.data.params, restated: 4},
            ),
        ),
    )
    with pytest.raises(SpecificationError, match=restated):
        _run(broken, tmp_path / "store")


def test_the_same_training_distribution_fits_the_same_weights(tmp_path: Path) -> None:
    """The determinism control for the two tests above: without it, "the
    weights differ" would be satisfied by a nondeterministic trainer whatever
    distribution it read."""
    study = _study(tmp_path / "doc")
    first = _learned_weights(study, tmp_path / "first")
    second = _learned_weights(study, tmp_path / "second")

    assert set(first) == set(second)
    assert all(torch.equal(first[key], second[key]) for key in first)


# --------------------------------------------------------------------------
# What is written down
# --------------------------------------------------------------------------


def test_a_published_record_derives_its_own_identifier(tmp_path: Path) -> None:
    """Identity/provenance. Every record must carry the sub-trees its own
    identifier follows from -- `store.maintenance` recomputes exactly these
    keys, and calls a record that cannot supply them *unverifiable*, which
    today is 100 % of the store.
    """
    study = _study(tmp_path / "doc")
    store = tmp_path / "store"
    _run(study, store)

    models, measurements = ModelStore(store), MeasurementStore(store)
    assert list(models.list_ids()), "nothing was published, so nothing is checked"
    for model_id in models.list_ids():
        model_spec = models.spec(model_id)
        assert set(MODEL_IDENTITY_KEYS) <= set(model_spec)
        assert _recomputed(model_spec, MODEL_IDENTITY_KEYS, "model") == model_id
    assert list(measurements.list_ids())
    for measurement_id in measurements.list_ids():
        measurement_spec = measurements.spec(measurement_id)
        assert set(MEASUREMENT_IDENTITY_KEYS) <= set(measurement_spec)
        assert (
            _recomputed(measurement_spec, MEASUREMENT_IDENTITY_KEYS, "measurement")
            == measurement_id
        )


def test_the_index_row_survives_a_reindex(tmp_path: Path) -> None:
    """`reindex` rebuilds every row from `spec.json` alone, so anything the
    producer writes only into the index row is lost on the first rebuild (trap
    §11.2). Rebuild it and compare."""
    study = _study(tmp_path / "doc")
    store = tmp_path / "store"
    _run(study, store)

    index = StoreIndex(store)
    before = {row.model_id: row for row in index.models()}

    # The row's CONTENT first. Comparing a rebuild against the original only
    # shows the two agree -- they read the same `spec.json`, so a field the
    # producer never wrote is equally absent from both and the comparison holds
    # while the store answers `mbl models list` with blanks.
    assert {row.family for row in before.values()} == {"unfolded", "truncated_riccati"}
    assert {row.problem_id for row in before.values()} == {
        str(study.problem.problem_id)
    }
    assert {row.contender_id for row in before.values()} == {"unfolded_a", "baseline"}
    assert {row.seed for row in before.values()} == {0, 1}
    assert {
        (row.state_dim, row.control_dim, row.horizon) for row in before.values()
    } == {(N, M, HORIZON)}
    assert all(row.created_utc and row.stamp for row in before.values())

    index.reindex(ModelStore(store), MeasurementStore(store))
    after = {row.model_id: row for row in index.models()}

    assert set(before) == set(after)
    for model_id, row in before.items():
        rebuilt = after[model_id]
        assert (row.problem_id, row.family, row.contender_id, row.seed) == (
            rebuilt.problem_id,
            rebuilt.family,
            rebuilt.contender_id,
            rebuilt.seed,
        )
        assert (row.state_dim, row.control_dim, row.horizon) == (
            rebuilt.state_dim,
            rebuilt.control_dim,
            rebuilt.horizon,
        )


def test_scoring_at_another_precision_is_refused_before_anything_trains(
    tmp_path: Path,
) -> None:
    """D20's rider is declared but not executable, so the producer refuses it.

    `EvaluationSpec` carries its own `ComputeContext` and it is signed into
    `MeasurementID`, but nothing rehosts a synthesized controller onto one: a
    float64 model scored on float32 batches dies inside the rollout with
    ``expected m1 and m2 to have the same dtype`` -- measured for the learned
    and the analytic family alike. Refused where both contexts are in hand, and
    refused *before* any training is paid for.
    """
    study = _study(tmp_path / "doc")
    store = tmp_path / "store"
    mismatched = replace(
        study,
        evaluation=replace(
            study.evaluation,
            ctx=replace(study.evaluation.ctx, precision=Precision.FLOAT32),
        ),
    )
    counter = CountingSynthesize()

    with pytest.raises(SpecificationError, match="rehosts"):
        _run(mismatched, store, counter)

    assert counter.calls == 0, "the refusal came after something had been trained"
    assert not list(ModelStore(store).list_ids()), "a model was published anyway"


def test_the_training_seed_reaches_the_synthesis(tmp_path: Path) -> None:
    """`training.seeds` is Annex 01's first-class multi-seed, and a seed
    reaching nothing would make a five-seed study five identical models under
    five identifiers -- surfacing later as a zero-width confidence interval.

    What must be in effect inside the call is the replicate's **weights
    stream**, not the replicate index itself (Phase G-3): the index is an
    author's label for a repetition, and two studies whose roots differ must
    not initialise identically because both happened to number their
    replicates from zero.
    """
    study = _study(tmp_path / "doc")
    counter = CountingSynthesize()
    report = _run(study, tmp_path / "store", counter)

    expected = [
        derive_replicate_streams(study.training.data.seed, outcome.seed).weights
        for outcome in report.outcomes
    ]
    assert counter.observed_seeds == expected
    assert len(set(counter.observed_seeds)) == 2, (
        "the study declares two seeds; if every synthesis observed one value "
        "the assertion above would hold vacuously"
    )
    assert not set(counter.observed_seeds) & {o.seed for o in report.outcomes}, (
        "the replicate index itself reached the global stream, which is what "
        "F2 did and what G-3 replaced"
    )


def test_the_tier_stamp_reaches_the_analysis_guard(tmp_path: Path) -> None:
    """D15's second guard reads a measurement's provenance mapping. E2 built
    the rule and left the *stamping* to the producer, so a subsetted run must
    be refusable from what the store holds -- not from what a caller remembers.
    """
    # Four depths, so `smoke`'s cap of two actually truncates the axis. With a
    # two-value axis nothing is dropped and the run is correctly NOT stamped --
    # E2's "the stamp records what happened" -- which would leave this test
    # asserting the stamp of a run that was never subsetted.
    document = load_study(
        _write(tmp_path / "doc", depths=[1, 2, 4, 8]), bindings=DEFAULT_SPEC_BINDINGS
    )
    resolved = document.resolve(DEFAULT_TIER_CATALOGUE, "smoke")
    assert resolved.axis_subset, "smoke truncated nothing; the stamp would be absent"
    store = tmp_path / "store"

    _run(resolved.study, store, provenance=resolved.provenance)

    measurements = MeasurementStore(store)
    assert list(measurements.list_ids())
    for measurement_id in measurements.list_ids():
        stamped = measurements.spec(measurement_id)["provenance"]
        assert stamped["tier"] == "smoke"
        assert stamped[AXIS_SUBSET_KEY] is True
        with pytest.raises(SpecificationError, match=AXIS_SUBSET_KEY):
            require_analysable_measurement(stamped)


def test_an_unsubsetted_run_is_not_stamped_as_subsetted(tmp_path: Path) -> None:
    """The negative control for the stamp: `standard` truncates no axis, so its
    measurements must pass the very same guard."""
    document = load_study(_write(tmp_path / "doc"), bindings=DEFAULT_SPEC_BINDINGS)
    resolved = document.resolve(DEFAULT_TIER_CATALOGUE, "standard")
    store = tmp_path / "store"

    _run(resolved.study, store, provenance=resolved.provenance)

    measurements = MeasurementStore(store)
    assert list(measurements.list_ids())
    for measurement_id in measurements.list_ids():
        require_analysable_measurement(measurements.spec(measurement_id)["provenance"])


# --------------------------------------------------------------------------
# What a measurement carries (Annex 03 §A.3.1, slice Phase B step B0)
# --------------------------------------------------------------------------

#: Rows a measurement's `samples.parquet` must hold: the fixture's two
#: evaluation batches of sixteen trajectories each. Written as the product it
#: is, so a fixture change moves the expectation with it rather than silently
#: making the row-count assertion agree with a stale literal.
EXPECTED_SAMPLE_ROWS = 2 * EVAL_BATCH


def _samples(store: Path) -> dict[str, pd.DataFrame]:
    """Every measurement's stored samples, read back **through the store**.

    Read rather than taken from the record that was written: the payload has
    to survive a parquet round trip, and a test holding the in-memory frame
    would pass on a schema parquet cannot represent.
    """
    measurements = MeasurementStore(store)
    frames: dict[str, pd.DataFrame] = {}
    for measurement_id in measurements.list_ids():
        samples = measurements.get(measurement_id).samples
        assert samples is not None
        frames[measurement_id] = samples
    return frames


def test_a_measurement_stores_one_row_per_evaluation_trajectory(
    tmp_path: Path,
) -> None:
    """§A.3 names evaluation trajectories as the within-seed aggregation unit.

    Until B0 this function wrote one batch *mean* per batch -- two rows here,
    one at `smoke` -- which left §A.4's paired procedures with no operand and
    made "recompute the aggregate by hand" a comparison of one number with
    itself.
    """
    store = tmp_path / "store"
    _run(_study(tmp_path / "doc"), store)

    frames = _samples(store)
    assert len(frames) == EXPECTED_POINTS
    for measurement_id, samples in frames.items():
        # The first three are §A.3.1's required content, in order. What may
        # follow them is enumerated rather than waved through: §A.3.1a's two
        # constraint-activity columns and nothing else, so a fourth quantity
        # arriving by accident fails here instead of being read as declared.
        columns = list(samples.columns)
        assert columns[:3] == [
            "batch_index",
            "trajectory_index",
            "trajectory_cost",
        ], measurement_id
        assert set(columns[3:]) <= {
            "max_abs_control",
            "control_saturation",
        }, measurement_id
        assert len(samples) == EXPECTED_SAMPLE_ROWS, measurement_id
        # Asserted after the parquet round trip because the analysis groups by
        # the two index columns and reduces the third: a key that came back as
        # a float, or a cost that came back as a float32, would change what a
        # group is and what its mean is without changing any column name.
        assert samples["batch_index"].dtype == np.int64, measurement_id
        assert samples["trajectory_index"].dtype == np.int64, measurement_id
        assert samples["trajectory_cost"].dtype == np.float64, measurement_id


def test_the_pairing_key_is_unique_within_a_measurement(tmp_path: Path) -> None:
    store = tmp_path / "store"
    _run(_study(tmp_path / "doc"), store)

    for measurement_id, samples in _samples(store).items():
        keys = list(zip(samples["batch_index"], samples["trajectory_index"]))
        assert len(set(keys)) == len(keys), measurement_id
        assert set(samples["batch_index"]) == {0, 1}, measurement_id
        assert set(samples["trajectory_index"]) == set(range(EVAL_BATCH))


def test_the_pairing_key_is_common_across_contenders(tmp_path: Path) -> None:
    """What makes a paired difference paired.

    §A.2's common-random-numbers law says every contender is scored on the
    identical realisations, so `(batch_index, trajectory_index)` must name the
    same realisation in every measurement of the study. The costs must
    nevertheless differ -- two contenders scoring identically at every index
    would mean the key is common because nothing varies, which would satisfy
    the first assertion for the wrong reason.
    """
    store = tmp_path / "store"
    _run(_study(tmp_path / "doc"), store)

    frames = list(_samples(store).values())
    reference = frames[0]
    reference_key = list(zip(reference["batch_index"], reference["trajectory_index"]))
    for samples in frames[1:]:
        assert (
            list(zip(samples["batch_index"], samples["trajectory_index"]))
            == reference_key
        )
    assert any(
        not np.allclose(samples["trajectory_cost"], reference["trajectory_cost"])
        for samples in frames[1:]
    )


#: Base of the positional encoding below. Larger than any batch size the
#: fixture uses, so `batch * POSITION_BASE + trajectory` is injective.
POSITION_BASE = 1000


def test_each_row_carries_the_cost_of_the_realisation_its_key_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mapping, not the key set — and the two survivors that taught it.

    The first version of this section asserted that the keys are unique, that
    they span the right values, and that the costs aggregate to the stored
    scalar. Mutation testing killed ten of twelve mutants against it and the
    two survivors were the same real defect: `np.repeat`/`np.tile` swapped, and
    `costs.T.reshape(-1)`. Both keep every key, keep every cost, keep the
    aggregate, and hand each cost to the **wrong** realisation — which is
    exactly the failure §A.4's pairing cannot survive and no figure would show.

    So the payload is replaced by a positional encoding and read back through
    the store: row `(b, t)` must carry `b * POSITION_BASE + t`, or some
    permutation has been applied between the rollout and the parquet.
    """
    calls: list[int] = []
    original = producer_module.evaluate_synthesized_controller

    def positional(*args: Any, **kwargs: Any) -> Any:
        metrics, arrays = original(*args, **kwargs)
        n_batches, batch_size = arrays["eval_trajectory_costs"].shape
        calls.append(n_batches)
        encoded = (
            np.arange(n_batches)[:, None] * POSITION_BASE + np.arange(batch_size)[None]
        ).astype(np.float64)
        return metrics, {**arrays, "eval_trajectory_costs": encoded}

    monkeypatch.setattr(producer_module, "evaluate_synthesized_controller", positional)
    store = tmp_path / "store"
    _run(_study(tmp_path / "doc"), store)

    assert len(calls) == EXPECTED_POINTS, (
        "the producer did not route every point through the evaluation this "
        "test replaced, so the assertions below would hold vacuously"
    )
    for measurement_id, samples in _samples(store).items():
        expected = (
            samples["batch_index"].to_numpy() * POSITION_BASE
            + samples["trajectory_index"].to_numpy()
        )
        np.testing.assert_array_equal(
            samples["trajectory_cost"].to_numpy(),
            expected.astype(np.float64),
            err_msg=measurement_id,
        )


def test_the_stored_trajectories_aggregate_to_the_stored_scalar(
    tmp_path: Path,
) -> None:
    """The seam Phase B's analysis stands on, asserted at a stated tolerance.

    `eval_expected_cost` keeps `BATCH_MEAN`; the samples carry `PER_SAMPLE`.
    The two are distinct named association orders, so they answer the same
    question to floating-point tolerance and **not** bit-exactly, and an
    analysis has to declare which one it aggregated.
    """
    store = tmp_path / "store"
    _run(_study(tmp_path / "doc"), store)

    measurements = MeasurementStore(store)
    for measurement_id, samples in _samples(store).items():
        scalar = measurements.get(measurement_id).metrics["eval_expected_cost"]
        assert float(samples["trajectory_cost"].mean()) == pytest.approx(
            scalar, rel=1e-12
        )


# --------------------------------------------------------------------------
# The exploding-recipe control's machinery
# --------------------------------------------------------------------------


def _detonating(recipe: Any) -> Any:
    """The same recipe, with `build_synthesizer` replaced by a detonator.

    A real subclass of the recipe's own dataclass rather than a delegating
    wrapper, for two reasons. `isinstance` against a runtime-checkable
    `Protocol` uses `inspect.getattr_static`, which does not consult
    `__getattr__`, so a wrapper is not a `Recipe` and never reaches the
    producer. And `ModelRecipe.get_signature` emits `type(self).__name__`, so
    the subclass is given its parent's exact name -- otherwise disarming a
    recipe would move every `ModelID` derived from it and the control would be
    running against a different study.
    """
    origin = type(recipe)

    def build_synthesizer(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError(
            "the reuse path called build_synthesizer; a model already in the "
            "store must be rebuilt and loaded, never re-synthesized"
        )

    disarmed = type(
        origin.__name__, (origin,), {"build_synthesizer": build_synthesizer}
    )
    return disarmed(
        **{f.name: getattr(recipe, f.name) for f in fields(recipe) if f.init}
    )


class _ExplodingRegistry:
    """A registry whose products refuse to build a synthesizer -- but only for
    the families the owner names, so a run may still legitimately re-derive a
    weightless record."""

    def __init__(self, inner: Any, owner: _ExplodingBindings) -> None:
        self._inner = inner
        self._owner = owner

    def create(self, name: str, **kwargs: Any) -> Any:
        recipe = self._inner.create(name, **kwargs)
        if name not in self._owner.disarm:
            return recipe
        self._owner.recipes_disarmed += 1
        return _detonating(recipe)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._inner, item)


@dataclass
class _ExplodingBindings:
    """`DEFAULT_SPEC_BINDINGS` with a registry that refuses synthesis for the
    named families."""

    inner: Any
    disarm: frozenset[str] | set[str] = frozenset()
    recipes_disarmed: int = 0

    @property
    def registry(self) -> Any:
        return _ExplodingRegistry(self.inner.registry, self)

    def build(
        self, kind: str, declaration: Mapping[str, Any], context: Mapping[str, Any]
    ) -> Any:
        return self.inner.build(kind, declaration, context)


# --------------------------------------------------------------------------
# D20's unexecuted half — the microbatch is refused, not ignored
# --------------------------------------------------------------------------


def test_a_declared_microbatch_is_honoured_and_moves_no_identifier(
    tmp_path: Path,
) -> None:
    """**A declaration must take effect or be refused; never be decoration.**

    This asserted the refusal until Annex 06 §4.3's executor existed. D20 had
    settled what a chunked batch *means* for identity while nothing chunked
    one, so `microbatch` was parsed, whitelisted for tiers, excluded from
    `ModelID` -- and ignored. The producer refused it rather than reporting in
    provenance that it had been honoured.

    Now it is executed, and the property that replaces the refusal is the
    stronger one: a chunked run trains the same models under the same
    identifiers. `microbatch` is a resource decision, so changing it may not
    move a single identifier (D20).
    """
    study = _study(tmp_path / "doc")
    chunked = replace(
        study,
        training=replace(
            study.training, batch=replace(study.training.batch, microbatch=8)
        ),
    )
    assert chunked.training.batch.accumulation_steps == 4, (
        "the fixture must actually imply chunking, or this proves nothing"
    )

    plain = _run(study, tmp_path / "store_plain")
    split = _run(chunked, tmp_path / "store_split")

    assert len(split.trained) == EXPECTED_POINTS
    assert sorted(m.model_id for m in plain.trained) == sorted(
        m.model_id for m in split.trained
    ), "a microbatch moved a ModelID, which D20 forbids"


def test_a_microbatch_that_implies_one_pass_trains_exactly_as_none_does(
    tmp_path: Path,
) -> None:
    """The degenerate case: a microbatch at or above the effective size is one
    pass, so it must be indistinguishable from declaring none."""
    study = _study(tmp_path / "doc")
    whole = replace(
        study,
        training=replace(
            study.training,
            batch=replace(study.training.batch, microbatch=TRAIN_BATCH * 2),
        ),
    )
    assert whole.training.batch.accumulation_steps == 1
    report = _run(whole, tmp_path / "store")
    assert len(report.trained) == EXPECTED_POINTS


def test_a_study_that_declares_no_microbatch_still_runs(tmp_path: Path) -> None:
    """Anti-vacuity, and the property the refusal must not break: `None` means
    "the runner chooses", the runner chooses one pass, and that is what every
    study does today."""
    study = _study(tmp_path / "doc")
    assert study.training.batch.microbatch is None
    assert study.training.batch.accumulation_steps == 1
    report = _run(study, tmp_path / "store")
    assert len(report.trained) == EXPECTED_POINTS


def test_the_stored_record_carries_a_real_semantic_name(tmp_path: Path) -> None:
    """**F3's single deliverable**, and the one thing its projection suite
    cannot see: that the producer actually writes the name.

    Written into the stored `spec.json` and not only into the index row,
    because `reindex` reads it from that file -- a name living only in the
    index is lost on the first rebuild (trap §11.2), and `mbl models tree` then
    falls back to rendering the problem as a raw digest with `—` in the depth
    and training columns.
    """
    study = _study(tmp_path / "doc")
    store = tmp_path / "store"
    _run(study, store)

    models = ModelStore(store)
    stored = {
        model_id: models.get(model_id).spec["semantic_name"]
        for model_id in models.list_ids()
    }
    assert stored, "the fixture stored nothing"
    for model_id, name in stored.items():
        assert name, f"{model_id} was published with an empty semantic name"
        problem, contender, training, digest = split_semantic_name(name)
        assert problem and contender and training
        assert model_id.startswith(digest)

    expected = {
        str(point.model_id): derive_semantic_name(point, study)
        for point in study.materialise()
    }
    assert stored == expected, (
        "the stored name is not the one the grammar projects for that point"
    )


def test_the_index_row_agrees_with_the_stored_name(tmp_path: Path) -> None:
    """Anti-vacuity for the trap: the row and the file must say the same thing,
    or a `reindex` silently changes what the store displays."""
    study = _study(tmp_path / "doc")
    store = tmp_path / "store"
    _run(study, store)

    models = ModelStore(store)
    for row in StoreIndex(store).models():
        assert row.semantic_name == models.get(row.model_id).spec["semantic_name"]


class TestTheDeviceCapabilityProbe:
    """A run on a host that cannot honour its declared device is refused —
    D17 as extended 2026-08-23, Annex 06 §3.5.

    The availability check moved OUT of `ComputeContext.__post_init__`, where
    it made reading a document a probe of the reading machine and left thirteen
    CUDA studies unloadable on CPU-only CI. This suite is the other half of
    that move: the refusal has to still exist, still fire **before any work**,
    and still not be reachable by accident on a CPU study.

    Without these, the relocation would be indistinguishable from a deletion —
    which is the shape of "a declaration that takes no effect" this project has
    now found five times.
    """

    def test_a_run_declaring_an_unavailable_device_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        study = _study(tmp_path / "doc")
        cuda = replace(
            study,
            training=replace(
                study.training,
                ctx=ComputeContext(backend="torch", device="cuda", precision="float64"),
            ),
            evaluation=replace(
                study.evaluation,
                ctx=ComputeContext(backend="torch", device="cuda", precision="float64"),
            ),
        )
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with pytest.raises(DeviceUnavailableError, match="CUDA is not available"):
            run_study(cuda, store=tmp_path / "store", bindings=DEFAULT_SPEC_BINDINGS)

    def test_it_is_refused_before_anything_is_synthesized(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The whole point of a pre-flight check: a study that cannot run must
        not first spend an hour producing models for it. Measured at the real
        synthesis seam, which must be called ZERO times."""
        study = _study(tmp_path / "doc")
        cuda = replace(
            study,
            training=replace(
                study.training,
                ctx=ComputeContext(backend="torch", device="cuda", precision="float64"),
            ),
            evaluation=replace(
                study.evaluation,
                ctx=ComputeContext(backend="torch", device="cuda", precision="float64"),
            ),
        )
        calls = 0

        def counting(synthesizer, problem, ctx):
            nonlocal calls
            calls += 1
            return default_synthesize(synthesizer, problem, ctx)

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with pytest.raises(DeviceUnavailableError):
            run_study(
                cuda,
                store=tmp_path / "store",
                bindings=DEFAULT_SPEC_BINDINGS,
                synthesize=counting,
            )
        assert calls == 0

    def test_one_probe_suffices_because_the_contexts_must_match(
        self, tmp_path: Path
    ) -> None:
        """Why the runner probes ONE context and not two.

        Written after trying to test a second probe on the evaluation context
        and finding it unreachable: `require_scorable_context` runs first in
        the same loop and refuses any point whose two contexts differ, so a
        second probe could never fail. Rather than keep a line no test can
        reach, the probe is single and this pins the invariant that makes it
        sufficient — so a future relaxation of that refusal fails here and
        says what to restore.
        """
        study = _study(tmp_path / "doc")
        mismatched = replace(
            study,
            evaluation=replace(
                study.evaluation,
                ctx=ComputeContext(backend="torch", device="cpu", precision="float32"),
            ),
        )
        with pytest.raises(SpecificationError, match="scored under"):
            run_study(
                mismatched, store=tmp_path / "store", bindings=DEFAULT_SPEC_BINDINGS
            )

    def test_a_cpu_study_runs_with_cuda_absent(self, tmp_path: Path) -> None:
        """Anti-vacuity. A probe that refused everything would pass all three
        assertions above; the fixture is a CPU study and must be untouched."""
        report = run_study(
            _study(tmp_path / "doc"),
            store=tmp_path / "store",
            bindings=DEFAULT_SPEC_BINDINGS,
        )
        assert len(report.outcomes) == EXPECTED_POINTS
