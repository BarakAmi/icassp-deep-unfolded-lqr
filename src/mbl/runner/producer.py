"""The minimal producer: a study specification, executed into the store.

**The defect this closes, located in code.** `experiments/runner.py` builds its
training harness like this:

```python
harness = EngineHarness(batch_spec=experiment.evaluation.batch_spec, ...)
```

That single argument is parent §2.2's retraining defect being executed: the
evaluation protocol reaches the trainer, so it also reaches the cache key that
gates training, so raising an evaluation batch count retrains every model. F1
gave `TrainingSpec` its own distribution, and this module is the first caller
able to write that line correctly — it builds the harness from
`training.data` and `training.batch`, and an evaluation cannot reach it.

**What a point costs.** Per `StudyPoint`, in order: derive the `ModelID` and the
`MeasurementID`, then decide *per record* what is already there. Both checks
matter and they are independent:

| Model | Measurement | What runs |
|---|---|---|
| absent | absent | synthesize, publish, evaluate, publish |
| **present** | **absent** | **reuse the model, evaluate, publish** — the shifted case |
| present | present | nothing at all |

The middle row is the whole re-architecture. An out-of-distribution study over
ten shifted problems is one training and ten rollouts, because
`derive_model_id` has no parameter through which an evaluation could enter.

**A model whose record carries weights is never re-synthesized.** A store hit
returns weights, not a controller, and `evaluate_synthesized_controller` needs a
`SynthesizedController`. Re-synthesising would *train*, which is the one thing
the gate forbids; persisting the live artifact would need a pickle, which D3
forbids for the same reason it forbids matplotlib pickles. So the controller is
rebuilt untrained and the stored weights are loaded into it — and
`load_trained_controller` is handed nothing but a bound `build_controller`, so
training is not merely skipped but unreachable from its inputs.

A record carrying **no** weights is a different thing: a receipt for a
closed-form solve rather than a checkpoint. It is re-derived through its own
synthesizer, because rebuilding it would not be faithful —
`AnalyticRecipe.build_controller` ignores the `ComputeContext` it is handed and
solves on NumPy while the synthesizer honours it, so the two paths' gains differ
by one ULP and there are no weights to overwrite the difference away (plan
§9.6.7). Re-deriving costs a Riccati recursion, which is less than the read it
replaces.

**What a point records about itself.** Every published model carries the
per-epoch history of its own training (`training.parquet`) where a training
phase exists, and every published measurement carries the per-batch trace of
its own evaluation (`evaluation.parquet`) and the wall-clock that pass cost --
Annex 02 §2.2. None of it takes part in an identifier, so a record written
before these existed is complete, and is left exactly as it is.

**Known omissions, stated rather than discovered later.** Gates are declared and
validated but not evaluated (pre-flight is Stage 3, post-hoc is Stage 4).
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch

from ..applications.recipes.base import (
    AnalyticRecipe,
    EngineHarness,
    EngineTrainedSynthesizer,
    ModelRecipe,
    TrainableRecipe,
    TrainedControllerArtifact,
)
from ..core.optimal_control_problem import OptimalControlProblem
from ..core.profiling import (
    peak_gpu_memory_bytes,
    reset_gpu_peak_memory,
    wall_clock,
)
from ..core.runtime import ComputeContext
from ..core.runtime.compute_context import require_available_device
from ..experiments.evaluation import evaluate_synthesized_controller
from ..experiments.experiment import EvaluationProtocol
from ..experiments.history import utc_timestamp
from ..experiments.provenance import code_provenance_stamp
from ..experiments.run_logging import capture_log
from ..models.lifecycle import SynthesizedController, Synthesizer
from ..engine.callbacks import METRICS_HISTORY
from ..persistence.recording_tracker import RecordingExperimentTracker
from ..spec.errors import SpecificationError
from ..spec.evaluation import REHOST_AWARE, REHOST_FULL
from ..spec.loader import SpecBindings
from ..spec.naming import derive_semantic_name
from ..spec.problem import ProblemSpec
from ..spec.study import StudyPoint, StudySpec
from ..spec.training import TrainingSpec
from ..store.content_store import (
    MeasurementRecord,
    MeasurementStore,
    ModelRecord,
    ModelStore,
)
from ..store.ids import MeasurementID, ModelID, StudyID
from ..store.index import MeasurementRow, ModelRow, StoreIndex
from .seeds import derive_replicate_streams

logger = logging.getLogger(__name__)

#: The modes whose points roll out a rebuilt controller instead of the stored
#: blind artifact (Annex 01 §2.4.1). Membership decides WHETHER to rebuild;
#: `_rehost_for_evaluation` decides HOW, per mode and family.
_REHOSTED_MODES = (REHOST_AWARE, REHOST_FULL)

#: The offline phase, as an injectable seam. It exists so that the phase's gate
#: can be *measured*: a double wraps this and counts it, and because the double
#: calls through, what is counted is the real training and not a stub.
Synthesize = Callable[
    [Synthesizer, OptimalControlProblem, ComputeContext], SynthesizedController
]

#: The single capability the reuse path is given. A bound
#: `ModelRecipe.build_controller`; deliberately not the recipe, which could
#: build a synthesizer.
BuildController = Callable[[OptimalControlProblem, ComputeContext], Any]

#: Keys a distribution declaration may not carry, because they have a home of
#: their own and two homes for one quantity is how a specification starts
#: disagreeing with itself (F1). `batch_size` is `BatchPlan.effective_size` and
#: `seed` is `DataSpec.seed`; the producer composes both in.
COUNT_AND_STREAM_HAVE_THEIR_OWN_HOME = ("batch_size", "seed")


def default_synthesize(
    synthesizer: Synthesizer,
    problem: OptimalControlProblem,
    ctx: ComputeContext,
) -> SynthesizedController:
    """The real offline phase: solve, compile or train, and freeze."""
    return synthesizer.synthesize(problem, ctx)


class Disposition(StrEnum):
    """What a point cost."""

    #: The offline phase ran.
    TRAINED = "trained"
    #: The rollout ran.
    EVALUATED = "evaluated"
    #: The store already held it.
    REUSED = "reused"


@dataclass(frozen=True)
class PointOutcome:
    """One executed `StudyPoint`.

    Attributes:
        label: The contender's label.
        seed: The training seed this model was fitted with.
        model_id: What the model is filed under.
        measurement_id: What the evaluation is filed under.
        model: `TRAINED` or `REUSED`.
        measurement: `EVALUATED` or `REUSED`.
        metrics: The scalar results, whether computed or read back.
    """

    label: str
    seed: int
    model_id: ModelID
    measurement_id: MeasurementID
    model: Disposition
    measurement: Disposition
    metrics: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RunReport:
    """What one `run_study` did.

    Attributes:
        study: The name the study is filed under.
        study_id: Its identity, independent of that name.
        store: The store root written to.
        outcomes: One per materialised point, in declaration order.
    """

    study: str
    study_id: StudyID
    store: Path
    outcomes: tuple[PointOutcome, ...]

    @property
    def trained(self) -> tuple[PointOutcome, ...]:
        """The points whose offline phase actually ran."""
        return tuple(o for o in self.outcomes if o.model is Disposition.TRAINED)

    @property
    def evaluated(self) -> tuple[PointOutcome, ...]:
        """The points whose rollout actually ran."""
        return tuple(o for o in self.outcomes if o.measurement is Disposition.EVALUATED)


def load_trained_controller(
    build_controller: BuildController,
    problem: OptimalControlProblem,
    ctx: ComputeContext,
    record: ModelRecord,
) -> SynthesizedController:
    """A stored model, back as something that can be rolled out.

    Note the absent parameters: there is no synthesizer here, no recipe that
    could build one, and no `synthesize` seam. Training is not skipped by a
    branch, it is unreachable from this function's inputs — the same property
    that makes `derive_model_id`'s missing evaluation parameter stronger than
    any behavioural test of it.

    Args:
        build_controller: A bound `ModelRecipe.build_controller`. Constructs an
            *untrained* controller, which is cheap for every family whose
            offline phase is training; for a family whose construction is
            itself the solve (the SDP-seeded bound), it repeats that solve, and
            that is still strictly less than re-synthesising.
        problem: The problem the model was trained on.
        ctx: The context it was trained under; part of `ModelID`, so it is the
            context the stored tensors are shaped for.
        record: The stored model.

    Returns:
        A `SynthesizedController` carrying the stored parameters.

    Raises:
        SpecificationError: If the record's weights and the rebuilt
            controller's parameters do not correspond exactly. Refused rather
            than loaded leniently: a silently partial load produces a
            *plausible* controller with untrained parameters, whose measurement
            would enter the store under an identifier claiming otherwise.
    """
    controller = build_controller(problem, ctx)
    module = getattr(controller, "as_module", None)
    if module is None:
        _require_no_orphan_weights(record, type(controller).__name__)
        return TrainedControllerArtifact(
            controller=controller,
            context=ctx,
            synthesizer_signature=_stored_synthesizer_signature(record),
        )
    _load_weights(module(), record)
    return TrainedControllerArtifact(
        controller=controller,
        context=ctx,
        synthesizer_signature=_stored_synthesizer_signature(record),
    )


def run_study(
    study: StudySpec,
    *,
    store: Path,
    bindings: SpecBindings,
    synthesize: Synthesize = default_synthesize,
    provenance: Mapping[str, Any] | None = None,
) -> RunReport:
    """Execute one study, serially, into a content-addressed store.

    Args:
        study: The study to run, already tier-resolved if it is being run at a
            tier.
        store: The store root. Created if absent.
        bindings: The Tier-4 vocabulary. Supplies the recipe registry the
            contenders resolve through and builds the training sampler the
            `DataSpec` names — the same `_build` vocabulary the document
            surface uses, so a distribution written in TOML and one composed in
            Python build the same sampler.
        synthesize: The offline phase. Injected so the phase's gate can be
            measured at the real call; a double must *wrap* the default rather
            than replace it, or the count is a statement about the double.
        provenance: What to stamp into every measurement of this run —
            `ResolvedStudy.provenance` when a tier was applied. D15's second
            guard reads it back off the stored record, so a truncated sweep
            axis cannot reach a figure.

    Returns:
        The `RunReport`, one outcome per materialised point.

    Raises:
        SpecificationError: If a training distribution restates a quantity that
            has its own home, or if a stored model's weights do not correspond
            to the controller rebuilt for them.
    """
    root = Path(store)
    models, measurements = ModelStore(root), MeasurementStore(root)
    index = StoreIndex(root)
    stamp = code_provenance_stamp()
    run_provenance = dict(provenance or {})

    services = _Services(
        models=models,
        measurements=measurements,
        index=index,
        bindings=bindings,
        synthesize=synthesize,
        stamp=stamp,
        run_provenance=run_provenance,
    )
    points = study.materialise()
    for point in points:
        # Before anything is trained, not when the rollout reaches it: a study
        # that cannot be scored must not first spend an hour producing models
        # for it. The same applies to a declaration this tier cannot honour --
        # both refusals are of the same kind, and both are cheaper here than
        # after the training is paid for.
        require_scorable_context(point)
        # The capability probe D17 (as extended 2026-08-23) puts HERE rather
        # than where a document is read: the declared device SIGNS, so
        # resolving it against the reading machine would make `ModelID`
        # machine-dependent -- while a run that cannot honour it must still
        # fail before it computes anything. Annex 06 §3.5.
        #
        # ONE call, not two. The evaluation context looks like a second thing
        # to probe and is not: `require_scorable_context` above has just
        # refused this point unless the two contexts' signatures are EQUAL, so
        # a second probe could never fail and no test could reach it -- a line
        # that takes no effect, which is the defect shape this project keeps
        # finding. If that check is ever relaxed, the second probe comes back;
        # `test_one_probe_suffices_because_the_contexts_must_match` is what
        # says so.
        require_available_device(str(point.training.ctx.device))
    logger.info(
        "Study %r (%s) started: %d point(s) into %s.",
        study.id,
        study.study_id,
        len(points),
        root,
    )
    outcomes = tuple(
        _run_point(point, study=study, services=services) for point in points
    )
    report = RunReport(
        study=study.id, study_id=study.study_id, store=root, outcomes=outcomes
    )
    logger.info(
        "Study %r finished: %d trained, %d evaluated, %d point(s) total.",
        study.id,
        len(report.trained),
        len(report.evaluated),
        len(outcomes),
    )
    return report


@dataclass(frozen=True)
class _Services:
    """The run-wide collaborators one point needs.

    A bundle rather than eight parameters: the signature budget is six
    (`PLR0913`), and these travel together for the whole run.
    """

    models: ModelStore
    measurements: MeasurementStore
    index: StoreIndex
    bindings: SpecBindings
    synthesize: Synthesize
    stamp: str
    run_provenance: Mapping[str, Any]


def _run_point(
    point: StudyPoint, *, study: StudySpec, services: _Services
) -> PointOutcome:
    """One point: reuse what is there, compute what is not."""
    models, measurements = services.models, services.measurements
    model_id, measurement_id = point.model_id, point.measurement_id
    label = point.contender.resolved_label

    model_present = models.exists(model_id)
    # Checked BEFORE anything is built, and at both levels. A point whose model
    # and measurement are both stored costs nothing at all -- not even a
    # controller -- and a second identical run of a whole study therefore
    # neither trains nor rolls out (plan §9.6.4).
    if model_present and measurements.exists(measurement_id):
        logger.info("Point %r (%s): already stored.", label, measurement_id)
        return PointOutcome(
            label=label,
            seed=point.seed,
            model_id=model_id,
            measurement_id=measurement_id,
            model=Disposition.REUSED,
            measurement=Disposition.REUSED,
            metrics=dict(measurements.get(measurement_id).metrics),
        )

    recipe = cast("ModelRecipe", point.contender.resolve())
    problem = point.problem.build()
    if model_present:
        # A rehosted point never builds the blind artifact at all: the rebuild
        # below is the only controller this point rolls out (Annex 01 §2.4.1).
        if point.evaluation.rehost in _REHOSTED_MODES:
            artifact = _rehost_for_evaluation(point, recipe, services)
        else:
            artifact = _reuse(point, recipe, problem, services)
        model_disposition = Disposition.REUSED
        logger.info("Point %r (%s): model reused, scoring.", label, model_id)
    else:
        synthesis = _synthesize(point, recipe, problem, services)
        artifact = synthesis.artifact
        services.models.put(
            ModelRecord(
                model_id=model_id,
                spec=_model_spec(point, study),
                weights=_weights_of(artifact),
                # One row per epoch for a family that trains; `None` for one
                # whose offline phase never builds an engine (every analytic
                # and frozen family), which is what `ModelRecord.history` has
                # always meant.
                history=synthesis.history,
                log=synthesis.log,
                provenance=_model_provenance(synthesis, services),
            )
        )
        model_disposition = Disposition.TRAINED
        if point.evaluation.rehost in _REHOSTED_MODES:
            # The published model IS the nominal training above; only what is
            # rolled out is rebuilt against the evaluation problem.
            artifact = _rehost_for_evaluation(point, recipe, services)

    services.index.upsert_model(_model_row(models.get(model_id)))
    metrics = _evaluate_and_publish(point, artifact, services, study)
    return PointOutcome(
        label=label,
        seed=point.seed,
        model_id=model_id,
        measurement_id=measurement_id,
        model=model_disposition,
        measurement=Disposition.EVALUATED,
        metrics=metrics,
    )


def _reuse(
    point: StudyPoint,
    recipe: ModelRecipe,
    problem: OptimalControlProblem,
    services: _Services,
) -> SynthesizedController:
    """A model already in the store, back as something that can be rolled out.

    Two cases, and the store itself decides which (plan §9.6.7). A record
    carrying **weights** is a checkpoint: it is rebuilt and loaded, and its
    synthesizer is never touched -- that is the gate, and in this project
    everything expensive writes parameters. A record carrying **none** is a
    receipt for a closed-form solve, and rebuilding it would not be faithful:
    `AnalyticRecipe.build_controller` ignores the `ComputeContext` it is given
    and solves on NumPy, while the synthesizer honours it, so the two paths'
    gains differ by one ULP and there are no weights to overwrite the
    difference away. Re-deriving such a record through its own synthesizer is
    bit-exact and costs a Riccati recursion.
    """
    record = services.models.get(point.model_id)
    if record.weights:
        return load_trained_controller(
            recipe.build_controller, problem, point.training.ctx, record
        )
    artifact = _synthesize(point, recipe, problem, services).artifact
    if _weights_of(artifact):
        raise SpecificationError(
            f"{point.model_id} is stored with no weights, but re-deriving it "
            "produced some; a family cannot change category between runs, and "
            "re-derivation is only sound for a record that is a receipt for a "
            "closed-form solve rather than a checkpoint"
        )
    return artifact


def _rehost_for_evaluation(
    point: StudyPoint, recipe: ModelRecipe, services: _Services
) -> SynthesizedController:
    """The aware rebuild (Annex 01 §2.4.1, D24, as corrected 2026-08-08):
    offline frozen, online live.

    The governing principle is the practical offline/online separation — no
    offline computation may run at the online stage, analytic or not. The
    recipe's `build_rehosted_controller` performs the two-problem
    construction: everything computed offline derives from `point.problem`
    (the plant the model was told — Riccati stacks, the declared step), and
    the plant it is handed, `point.evaluation.problem`, enters only the
    per-step solver expressions. Trained tensors then transplant from the
    stored record for a recipe that trains; a recipe that trains nothing has
    nothing to load, and its stored constant tensors are deliberately NOT
    loaded — a weights-driven transplant would clobber a declared override
    with the nominal constant.

    **The `full` mode (Annex 01 §2.4.1 as amended 2026-08-09)** is the same
    doctrine's second half: a family that trains nothing has no offline stage
    to freeze, so an `AnalyticRecipe` under `full` re-synthesises WHOLE
    against the evaluation plant — with its per-plant override already
    applied — through the synthesizer path, never `build_controller`: the
    shortcut ignores the compute context and its gains differ by one ULP,
    which is exactly the no-op anchor this function's tests pin. A trainable
    recipe under `full` falls through to the identical two-problem
    construction as `aware`, deliberately sharing the code path.

    Nothing built here is ever published as a model; the rebuild is charged to
    the measurement.
    """
    rebuilt = _overridden_recipe(recipe, point)
    online_problem = point.evaluation.problem.build()
    if point.evaluation.rehost == REHOST_FULL and isinstance(rebuilt, AnalyticRecipe):
        logger.info(
            "Point %r (%s): full rehost — the whole synthesis re-runs against %s.",
            point.contender.resolved_label,
            point.model_id,
            point.evaluation.problem.problem_id,
        )
        return _synthesize(point, rebuilt, online_problem, services).artifact
    offline_problem = point.problem.build()
    logger.info(
        "Point %r (%s): %s rehost — offline quantities from %s, online "
        "expressions from %s.",
        point.contender.resolved_label,
        point.model_id,
        point.evaluation.rehost,
        point.problem.problem_id,
        point.evaluation.problem.problem_id,
    )
    # `require_scorable_context` has already refused a study whose scoring
    # context differs from the training one, so the training ctx below is
    # also the evaluation ctx; it is passed because it is the context the
    # stored tensors are shaped for.
    artifact = rebuilt.build_rehosted_controller(
        offline_problem, online_problem, point.training.ctx
    )
    if isinstance(rebuilt, TrainableRecipe):
        record = services.models.get(point.model_id)
        # Every trainable's rebuild is a `TrainedControllerArtifact`; the
        # narrowing is for the type checker, not a runtime branch.
        controller = cast("TrainedControllerArtifact", artifact).controller
        module = getattr(controller, "as_module", None)
        if module is None:  # pragma: no cover — every trainable exposes it
            _require_no_orphan_weights(record, type(controller).__name__)
        else:
            _load_weights(module(), record)
    return artifact


def _overridden_recipe(recipe: ModelRecipe, point: StudyPoint) -> ModelRecipe:
    """This point's recipe with its matching rehost override applied, if any.

    An override matches on the contender's label and the evaluation plant's
    `ProblemID` — never on filename or axis label, which take no part in any
    identifier. `StudySpec` has already refused unknown labels, duplicates and
    overrides no point can fire; what remains checkable only here is whether
    the named fields exist on the resolved recipe.
    """
    label = point.contender.resolved_label
    scored_on = str(point.evaluation.problem.problem_id)
    for override in point.evaluation.rehost_overrides:
        if override.contender != label:
            continue
        if str(override.problem.problem_id) != scored_on:
            continue
        replacements = dict(override.config)
        try:
            # `recipe` is typed at the abstract `ModelRecipe`, which mypy
            # cannot know is a dataclass -- the same narrowing the legacy
            # rehost seam documents for the same call.
            return dataclasses.replace(recipe, **replacements)  # type: ignore[type-var]
        except TypeError as error:
            available = sorted(
                f.name
                for f in dataclasses.fields(recipe)  # type: ignore[arg-type]
            )
            raise SpecificationError(
                f"rehost override for {label!r} names a field "
                f"{type(recipe).__name__} does not have ({error}); "
                f"available: {available}"
            ) from error
    return recipe


@dataclass(frozen=True)
class _Synthesis:
    """One offline phase, what it cost, and what it recorded on the way."""

    artifact: SynthesizedController
    wall_time_s: float
    history: pd.DataFrame | None
    log: str | None


def _synthesize(
    point: StudyPoint,
    recipe: ModelRecipe,
    problem: OptimalControlProblem,
    services: _Services,
) -> _Synthesis:
    """Run the offline phase. **The corrected line is the harness's.**"""
    ctx = point.training.ctx
    # NOT a null tracker. `MetricsHistoryCallback` is attached to every
    # trainable run and builds the per-epoch frame regardless; under the null
    # tracker it was built, handed to `save_artifact` and discarded, which is
    # why 162 stored records carry no `training.parquet` (Annex 02 §2.2). The
    # whitelist keeps that one frame and still discards the trajectory arrays
    # the same callback list hands over, which are the megabytes.
    tracker = RecordingExperimentTracker(frames=(METRICS_HISTORY,))
    # What distinguishes one replicate from another, and it has to reach both
    # streams: a replicate redraws everything a repetition of the experiment
    # would redraw (Annex 01 §2.3.3). Derived from the declared root and the
    # replicate alone, so every contender at this replicate trains on identical
    # batches.
    streams = derive_replicate_streams(point.training.data.seed, point.seed)
    harness = EngineHarness(
        # NOT `evaluation.batch_spec`. This is the whole of parent §2.2's fix:
        # the trainer draws from the study's own training distribution, so an
        # evaluation knob cannot reach a model and a training knob cannot reach
        # a measurement.
        batch_spec=training_batch_spec(
            point.training, point.problem, services.bindings, stream=streams.data
        ),
        tracker=tracker,
        ctx=ctx,
        # D20's resource half, executable at last (Annex 06 §4.3). It rides the
        # harness and not the training plan because the plan is signed into
        # `ModelID` and the microbatch must not be: which chunk size fits is a
        # property of the machine that ran the job, never of the experiment.
        microbatch=point.training.batch.microbatch,
    )
    # In effect before anything stochastic is constructed. This is the seam
    # through which a family that declares no seed field of its own -- half the
    # registry -- still gets a real replicate, and through which one added
    # later gets one while declaring nothing.
    torch.manual_seed(streams.weights)
    synthesizer = recipe.build_synthesizer(problem, harness)
    # The training WORLD (Annex 01 §2.3.5, D25): when declared, the engine
    # rolls out in it while the controller keeps deriving from `problem` --
    # the plant the contender is told. Refused per point for a contender
    # whose offline phase consumes no trajectories: the declaration is inert
    # there, and this project does not carry inert declarations. Scope the
    # axis with `applies_to`, or drop the contender.
    if point.training.problem is not None:
        if not isinstance(synthesizer, EngineTrainedSynthesizer):
            raise SpecificationError(
                f"contender {point.contender.resolved_label!r} declares a "
                "training-data plant, but its offline phase is "
                f"{type(synthesizer).__name__} and consumes no training "
                "trajectories -- the declaration is inert for it (Annex 01 "
                "§2.3.5). Scope the training.problem axis with applies_to, "
                "or drop the contender from the document"
            )
        synthesizer = dataclasses.replace(
            synthesizer, data_problem=point.training.problem.build()
        )

    # The capture opens BEFORE the line that names the point, so a stored
    # `synthesis.log` says which model it belongs to on its first line. A log
    # that begins mid-training is a log a reader has to correlate by hand.
    with capture_log() as captured:
        logger.info(
            "Point %r (%s): synthesizing on %d-trajectory batches from %r.",
            point.contender.resolved_label,
            point.model_id,
            point.training.batch.effective_size,
            point.training.data.kind,
        )
        reset_gpu_peak_memory()
        started = wall_clock()
        artifact = services.synthesize(synthesizer, problem, ctx)
        elapsed = wall_clock() - started
    return _Synthesis(
        artifact=artifact,
        wall_time_s=elapsed,
        history=tracker.frame(METRICS_HISTORY),
        log=captured.getvalue(),
    )


@dataclass(frozen=True)
class _Online:
    """One evaluation pass, and what it cost.

    Annex 04 §7 has asked for "total offline and online compute" since it was
    written, and §7's row said *not recorded* against a fully populated store
    because nothing on this path held a clock.
    """

    wall_time_s: float
    peak_vram: int | None


def _evaluate_and_publish(
    point: StudyPoint,
    artifact: SynthesizedController,
    services: _Services,
    study: StudySpec,
) -> dict[str, float]:
    """Score `artifact` under the study's evaluation and publish the result."""
    protocol = cast(EvaluationProtocol, point.evaluation.protocol)
    reset_gpu_peak_memory()
    started = wall_clock()
    # Inside the timed region: building the batches is work this pass paid for,
    # it happens once per measurement, and nothing else records it.
    with capture_log() as captured:
        batches = protocol.build_batches(point.evaluation.ctx)
        metrics, arrays = evaluate_synthesized_controller(
            artifact, point.evaluation.problem.build(), batches
        )
    online = _Online(
        wall_time_s=wall_clock() - started, peak_vram=peak_gpu_memory_bytes()
    )
    spec = _measurement_spec(point, study, services, online)
    services.measurements.put(
        MeasurementRecord(
            measurement_id=point.measurement_id,
            spec=spec,
            metrics=metrics,
            samples=_measurement_samples(arrays),
            trace=_measurement_trace(arrays),
            log=captured.getvalue(),
        )
    )
    # The SAME spec into the index. It used to write "{}" here while
    # `rebuild_from` wrote the real thing, so a run and a rebuild produced
    # different indexes and every query against a measurement's spec came back
    # empty -- silently, because an empty object is a legal answer. Nothing in
    # this package reads it, which is why it survived; it cost two wrong
    # conclusions in one audit (2026-08-13), where `rehost` read as absent on
    # every row and the informed cells looked as though they had never run.
    # The model path has always written its spec on both routes.
    services.index.upsert_measurement(
        MeasurementRow(
            measurement_id=point.measurement_id,
            model_id=point.model_id,
            eval_problem_id=point.evaluation.problem.problem_id,
            created_utc=utc_timestamp(),
            spec_json=json.dumps(spec, sort_keys=True),
            metrics=metrics,
        )
    )
    return metrics


def require_scorable_context(point: StudyPoint) -> None:
    """Refuse an evaluation context nothing can currently honour.

    D20's rider gives `EvaluationSpec` its own `ComputeContext` so that
    contenders trained at different precisions are *scored* at one, and that
    context is signed into `MeasurementID`. Nothing, however, casts a
    synthesized controller onto it: the artifact holds the parameters it was
    trained with, so a float64 model scored on float32 batches dies inside the
    rollout with ``expected m1 and m2 to have the same dtype``. Measured for
    every family, learned and analytic alike.

    Until a controller can be rehosted onto a context (plan §9.6.8), the honest
    behaviour is to refuse the pairing here, where both contexts are in hand and
    the message can name them, rather than to produce that `RuntimeError` from
    six frames inside a rollout after the training has been paid for.

    Args:
        point: The materialised point about to be executed.

    Raises:
        SpecificationError: If the evaluation context differs from the training
            context.
    """
    training, evaluation = point.training.ctx, point.evaluation.ctx
    if training.get_signature() == evaluation.get_signature():
        return
    raise SpecificationError(
        f"contender {point.contender.resolved_label!r} is trained under "
        f"{training.get_signature()} but scored under "
        f"{evaluation.get_signature()}. An evaluation context is part of what a "
        "measurement means and is signed into its identifier, but nothing "
        "rehosts a synthesized controller onto one, so the rollout would fail on "
        "a dtype mismatch after the training was paid for. Declare "
        "[evaluation.compute] equal to [compute], or leave it out"
    )


def training_batch_spec(
    training: TrainingSpec,
    problem: ProblemSpec,
    bindings: SpecBindings,
    *,
    stream: int,
) -> Any:
    """The sampler the training distribution names, at the declared count.

    The composition F1 left to the producer: `DataSpec` says *what* is drawn and
    from which stream, `BatchPlan.effective_size` says *how many*, and the
    problem says at what dimensions. Built through the injected `_build`
    vocabulary, so a distribution written in a document and the same one
    composed in Python produce the same sampler.

    Args:
        training: The training declaration.
        problem: The problem being trained on — never the evaluation problem.
        bindings: The Tier-4 vocabulary.
        stream: This replicate's data stream, from
            `derive_replicate_streams`. Required and keyword-only rather than
            defaulted to `training.data.seed`: that default is correct at
            replicate 0 and silently wrong at every other one, which is the
            defect G-3 exists to close. Forgetting it is a `TypeError`.

    Returns:
        The `BatchSpec` the harness draws from.

    Raises:
        SpecificationError: If the distribution's parameters restate the batch
            size or the stream seed. Both have a home; a second one that
            silently lost would be the defect F1 was taken to prevent, and
            silently winning would make `training.batch.effective_size`
            unwritable by a tier.
    """
    restated = sorted(
        set(training.data.params) & set(COUNT_AND_STREAM_HAVE_THEIR_OWN_HOME)
    )
    if restated:
        raise SpecificationError(
            f"training.data declares {', '.join(restated)}, which it does not "
            "own: the count is training.batch.effective_size and the stream is "
            "training.data.seed. One quantity, one home -- the alternative is a "
            "guard that fires on every tier application"
        )
    return bindings.build(
        training.data.kind,
        {
            **training.data.params,
            "seed": stream,
            "batch_size": training.batch.effective_size,
        },
        {"state_dim": problem.state_dim, "horizon": problem.data.horizon},
    )


# -- what gets written down -------------------------------------------------


def _model_spec(point: StudyPoint, study: StudySpec) -> dict[str, Any]:
    """A model record's `spec.json`.

    The first five keys are `store.maintenance.MODEL_IDENTITY_KEYS` exactly, so
    `mbl store verify` can recompute the identifier the record is filed under
    rather than reporting it unverifiable. The next block is what
    `index._model_row_from_record` projects, so a `reindex` rebuilds the same
    rows the producer wrote — anything living only in the index row is lost on
    the first rebuild (trap §11.2).
    """
    return {
        "problem": str(point.problem.problem_id),
        "contender": point.contender.get_signature(),
        "training": point.training.get_signature(),
        "ctx": point.training.ctx.get_signature(),
        "seed": point.seed,
        "problem_id": str(point.problem.problem_id),
        "family": point.contender.family,
        "contender_id": point.contender.resolved_label,
        "state_dim": point.problem.state_dim,
        "control_dim": point.problem.control_dim,
        "horizon": point.problem.horizon,
        # Written HERE, and not only into the index row, because `reindex`
        # reads it from this file: a name living only in the index is lost on
        # the first rebuild (trap §11.2), and the store's own tree then falls
        # back to rendering the problem as a raw digest.
        "semantic_name": derive_semantic_name(point, study),
        "study": study.id,
        "study_id": str(study.study_id),
        # What the reuse path needs and cannot recompute without a synthesizer.
        "synthesizer": _synthesizer_signature(point),
    }


def _measurement_samples(arrays: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """A measurement record's `samples.parquet`: one row per trajectory.

    Annex 03 §A.3.1. `MeasurementStore`'s docstring said "per-trajectory detail
    was retained" while this function wrote one batch *mean* per batch, which
    left §A.4's paired procedures with no operand and made the within-seed
    aggregation unit the batch rather than the trajectory.

    `(batch_index, trajectory_index)` is the pairing key, and it is
    within-batch by construction: the evaluation batches are common random
    numbers shared by every contender (§A.2), so the same pair names the same
    realisation across contenders, which is exactly what makes a paired
    difference paired. A global running index would say the same thing for a
    fixed batch count and stop saying it the moment one changed.

    §A.3.1a adds `max_abs_control` and `control_saturation` on the same key,
    for a problem that declares a box. They are **columns and not a second
    frame** because they describe the same trajectory the cost describes, on
    the same unit, drawn in the same rollout — and because a reader asking
    "which trajectories were pinned to the bound, and what did they cost?"
    should not have to join two payloads to find out. Absent, never zero,
    where there is no box.
    """
    costs = arrays["eval_trajectory_costs"]
    n_batches, batch_size = costs.shape
    frame = pd.DataFrame(
        {
            "batch_index": np.repeat(np.arange(n_batches), batch_size),
            "trajectory_index": np.tile(np.arange(batch_size), n_batches),
            "trajectory_cost": costs.reshape(-1),
        }
    )
    for column, key in (
        ("max_abs_control", "eval_max_abs_control"),
        ("control_saturation", "eval_control_saturation"),
    ):
        if key in arrays:
            frame[column] = arrays[key].reshape(-1)
    return frame


def _measurement_trace(arrays: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """A measurement record's `evaluation.parquet`: one row per batch.

    Annex 02 §2.2. `eval_batch_costs` is computed on every evaluation and was
    reduced to one scalar and discarded -- so a measurement could report an
    expected cost and nothing about the spread the mean came from. Paired with
    the wall-clock of the batch that produced it, which is the only per-batch
    quantity this pass ever knew.

    `batch_index` is the same key `samples.parquet` pairs on (§A.4), so the two
    payloads join without either of them naming the other.
    """
    costs = arrays["eval_batch_costs"]
    return pd.DataFrame(
        {
            "batch_index": np.arange(len(costs)),
            "batch_cost": costs,
            "wall_time_s": arrays["eval_batch_wall_time_s"],
        }
    )


def _measurement_spec(
    point: StudyPoint, study: StudySpec, services: _Services, online: _Online
) -> dict[str, Any]:
    """A measurement record's `spec.json`.

    The first three keys are `MEASUREMENT_IDENTITY_KEYS` exactly. `provenance`
    carries the tier stamp, which is what `require_analysable_measurement`
    reads: D15's exemption lets `smoke` truncate a swept axis, and the guard
    against that reaching a figure has to be answerable from the store.
    """
    return {
        "model": str(point.model_id),
        "eval_problem": str(point.evaluation.problem.problem_id),
        "eval_protocol": point.evaluation.get_signature(),
        "study": study.id,
        "study_id": str(study.study_id),
        "metrics": sorted(point.evaluation.metrics),
        "provenance": {
            "created_utc": utc_timestamp(),
            "stamp": services.stamp,
            # The online half of Annex 04 §7's compute row. It lives in this
            # block rather than in a `provenance.json` of its own because the
            # tier stamp `require_analysable_measurement` reads is already
            # here, and 162 stored records would otherwise need a reader for
            # each home (Annex 02 §2.2's recorded deviation).
            "wall_time_s": online.wall_time_s,
            "peak_vram_mb": (
                None if online.peak_vram is None else online.peak_vram / 1e6
            ),
            **services.run_provenance,
        },
    }


def _model_provenance(synthesis: _Synthesis, services: _Services) -> dict[str, Any]:
    """Never signed: how this model came to exist, not what it is."""
    peak_vram = peak_gpu_memory_bytes()
    declared = getattr(synthesis.artifact, "provenance", None) or {}
    return {
        "created_utc": utc_timestamp(),
        "stamp": services.stamp,
        "wall_time_s": synthesis.wall_time_s,
        "peak_vram_mb": None if peak_vram is None else peak_vram / 1e6,
        "final_metrics": {
            key: float(value)
            for key, value in dict(declared.get("final_metrics", {})).items()
            if isinstance(value, (int, float))
        },
        **services.run_provenance,
    }


def _model_row(record: ModelRecord) -> ModelRow:
    """The index row a stored model projects to.

    Built from the record rather than from the point, so re-running a study
    writes the row it already had: the timestamps come out of the stored
    provenance, not off the clock, and the upsert is therefore idempotent.
    """
    spec, prov = record.spec, record.provenance
    return ModelRow(
        model_id=record.model_id,
        semantic_name=str(spec.get("semantic_name", "")),
        problem_id=spec["problem_id"],
        family=str(spec.get("family", "")),
        contender_id=str(spec.get("contender_id", "")),
        seed=int(spec.get("seed", 0)),
        state_dim=spec.get("state_dim"),
        control_dim=spec.get("control_dim"),
        horizon=spec.get("horizon"),
        created_utc=prov.get("created_utc"),
        stamp=prov.get("stamp"),
        wall_time_s=prov.get("wall_time_s"),
        peak_vram_mb=prov.get("peak_vram_mb"),
        spec_json=json.dumps(spec, sort_keys=True),
    )


# -- weights ----------------------------------------------------------------


def _weights_of(artifact: SynthesizedController) -> dict[str, torch.Tensor]:
    """A synthesized artifact's parameters, or nothing.

    The module is looked up on the *controller* rather than on the artifact:
    `TrainedControllerArtifact.as_module` exists unconditionally and delegates,
    so `hasattr(artifact, "as_module")` is `True` even for a NumPy-native
    analytic controller for which calling it is an `AttributeError`.
    """
    controller = getattr(artifact, "controller", artifact)
    module = getattr(controller, "as_module", None)
    if module is None:
        return {}
    return dict(module().state_dict())


def _load_weights(module: torch.nn.Module, record: ModelRecord) -> None:
    """Put a stored checkpoint back into a freshly built module, exactly.

    Not `strict=True`: `dense_state_dict` deliberately drops non-strided
    entries when publishing, because safetensors stores dense layouts only and
    the sparse entries in this tree are specification-derived buffers (the
    compiled constraint matrices of a cone layer), reconstructible from the
    spec and never trained weights. A strict load would therefore refuse every
    such family outright.

    Not lenient either. Every stored tensor must land, and every parameter left
    unfilled must be one publishing would have dropped — checked against the
    live module's own layouts rather than against a list of family names, so a
    new family cannot drift out of the rule.
    """
    report = module.load_state_dict(dict(record.weights), strict=False)
    live = dict(module.state_dict())
    unfilled = [key for key in report.missing_keys if live[key].layout == torch.strided]
    if report.unexpected_keys or unfilled:
        raise SpecificationError(
            f"the stored weights of {record.model_id} do not correspond to the "
            f"controller rebuilt for them: {sorted(report.unexpected_keys)} are "
            f"stored but unknown to it, {sorted(unfilled)} are its parameters "
            "and were not stored. Loading anyway would produce a plausible "
            "controller with untrained parameters, and file its measurement "
            "under an identifier claiming otherwise"
        )


def _require_no_orphan_weights(record: ModelRecord, controller: str) -> None:
    """A controller with no module must have been published with no weights."""
    if record.weights:
        raise SpecificationError(
            f"{record.model_id} was published with weights "
            f"{sorted(record.weights)}, but {controller} exposes no module to "
            "load them into; the record and the recipe disagree about whether "
            "this family learns anything"
        )


def _synthesizer_signature(point: StudyPoint) -> dict[str, Any]:
    """What produced this model, as specification.

    Taken from the contender rather than from the live synthesizer: it is the
    recipe's own signature either way, and reading it here keeps the publishing
    path from having to hold a `Synthesizer` any longer than the call itself.
    """
    return {"recipe": point.contender.get_signature()}


def _stored_synthesizer_signature(record: ModelRecord) -> dict[str, Any]:
    signature = record.spec.get("synthesizer", {})
    return dict(signature) if isinstance(signature, dict) else {}
