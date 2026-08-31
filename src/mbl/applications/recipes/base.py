"""`ModelRecipe` -- the per-family Strategy object (REFACTOR_PLAN v3, T3.a):
each controller family owns its problem-to-engine wiring (which controller,
which `TrainingStrategy`, which optimizer via its `TrainingPlan`, which
sampler backend, which extra callbacks), so the application layer composes
recipes as *data* and the ``if name == "neural": ... elif name == "cocp":``
routing ladders (M2) are dead.

Two base wirings exist, implemented exactly once each:

* `AnalyticRecipe` -- closed-form or frozen families, evaluated through
  `AnalyticalStrategy` for a single epoch.
* `TrainableRecipe` -- torch-trained families: `RolloutModel` wrapper +
  `GradientDescentStrategy` built **from the recipe's own `TrainingPlan`**
  through the `TrainableController.as_module()` seam (T2.b), which is also
  the systemic C4 fix -- the logged `TrainingConfig` derives from the same
  `OptimizerSpec` the live optimizer was built from.

Each recipe also produces the family's `Synthesizer` (T2.a): analytic
families return their direct solver-backed synthesizers; trainable families
return an `EngineTrainedSynthesizer` whose ``synthesize`` = build controller
-> wrap in `RolloutModel` -> train via the Runner from the `TrainingPlan` ->
freeze into a `TrainedControllerArtifact`; frozen torch families return a
`FrozenControllerSynthesizer` (construction is the entire offline phase).
"""

import dataclasses
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar, cast

import torch

from ..factories import BatchSpec
from ..rollout import RolloutModel
from ...core.optimal_control_problem import OptimalControlProblem
from ...core.system.state_space_system import ControlPolicy
from ...core.runtime import Backend, ComputeContext
from ...engine.callbacks import (
    Callback,
    ComputeContextAnnouncementCallback,
    ExperimentTrackingCallback,
    MetricsHistoryCallback,
    ModelCheckpointCallback,
    ProblemSignatureCallback,
    ProfilingCallback,
    TrainingStateCheckpointCallback,
    TrajectoryLoggingCallback,
)
from ...engine.config import TrainingConfig
from ...engine.engine import Engine
from ...engine.runner import Runner, TrainingPhase
from ...engine.strategy import AnalyticalStrategy, GradientDescentStrategy
from ...engine.training_plan import TrainingPlan
from ...models.base import Controller
from ...models.lifecycle import (
    SynthesizedController,
    Synthesizer,
    TrainableController,
)
from ...models.registry import ControllerRegistry
from ...persistence.null_tracker import NullExperimentTracker
from ...persistence.tracker import ExperimentTracker

logger = logging.getLogger(__name__)

#: The Tier-2 registry seam activated for recipes (T2.f): recipe families
#: register here by name, so orchestration layers (and, from Stage S4, the
#: `ContenderSpec` resolution of `Experiment`s) can select families as data
#: without importing concrete classes.
RECIPE_REGISTRY: ControllerRegistry = ControllerRegistry()


def register_recipe(name: str) -> Callable[[type["ModelRecipe"]], type["ModelRecipe"]]:
    """Class decorator: register the decorated `ModelRecipe` under `name` in
    `RECIPE_REGISTRY` as an import side effect -- new families are added by
    defining one new recipe module, never by editing a routing ladder.

    Args:
        name: The unique family name to register under.

    Returns:
        The class decorator.
    """

    def decorator(cls: type["ModelRecipe"]) -> type["ModelRecipe"]:
        RECIPE_REGISTRY.register(name, cls)
        return cls

    return decorator


def accept_enum_values(recipe: Any, **fields: type[StrEnum]) -> None:
    """Normalise string inputs into their enums, in place, at construction.

    The declarative surface hands a recipe whatever the document said, and a
    document can only say ``init_method = "cold"``. A `StrEnum` member compares
    equal to its value but is not identical to it, so an un-normalised string
    silently takes the *wrong branch* of every ``is`` test in this module and
    then fails on ``.value`` several frames later -- which is exactly how this
    was found, by loading NB04 from TOML.

    Accepting the enums' string values at the boundary and normalising once is
    the same ingress contract `core.runtime.ComputeContext` already documents.

    Raises:
        ValueError: If a value is not one of the enum's members, naming the
            field and the alternatives.
    """
    for name, enum in fields.items():
        value = getattr(recipe, name)
        if isinstance(value, enum):
            continue
        try:
            object.__setattr__(recipe, name, enum(value))
        except ValueError:
            raise ValueError(
                f"{type(recipe).__name__}.{name} is {value!r}; expected one of "
                f"{sorted(member.value for member in enum)}."
            ) from None


def build_default_recipe_registry() -> ControllerRegistry:
    """Import this project's built-in recipe modules -- triggering their
    `register_recipe` decorators -- and return the populated registry.

    Returns:
        The global `RECIPE_REGISTRY`, populated with every built-in family
        (``"riccati"``, ``"truncated_riccati"``, ``"neural"``,
        ``"unfolded"``, ``"cocp"``, ``"cocp_lower_bound"``).
    """
    from . import analytic, cocp, neural, unfolded

    del analytic, cocp, neural, unfolded  # imported for registration only
    return RECIPE_REGISTRY


@dataclass(frozen=True)
class EngineHarness:
    """Execution-time resources a recipe wires its engine from -- the
    per-run counterpart to the recipe's specification-time fields. Interim
    Stage-S3 shape: Stage S4's `run_experiment`/`EvaluationProtocol`
    supersedes it.

    Attributes:
        batch_spec: The case study's shared `BatchSpec` (a `GaussianBatchSpec`
            for every existing case study; NB07 additionally passes an
            `applications.uncertainty.noise.ExoticBatchSpec` or
            `applications.uncertainty.dataset.FiniteTrajectoryDatasetSpec`).
        tracker: The run's `ExperimentTracker`.
        ctx: The case study's `ComputeContext` (single dtype/device
            authority -- recipes contain no dtype/device literals).
        log_every: Metric-logging cadence forwarded to `TrainingConfig`.
        microbatch: Trajectories per forward/backward pass, or `None` for one
            pass over the whole effective batch (Annex 06 §4.3). It rides
            **here** and not on `TrainingPlan` because the plan is signed into
            `ModelID` and D20 excludes the microbatch from identity: it is a
            property of the machine that ran the job, not of the experiment.
            Nothing on this harness has a `get_signature`, which is what makes
            that structural rather than a convention to remember.
    """

    batch_spec: BatchSpec
    tracker: ExperimentTracker
    ctx: ComputeContext
    log_every: int = 1
    microbatch: int | None = None


def _make_rollout_fn(
    controller: Controller, problem: OptimalControlProblem
) -> Callable[[Any], Any]:
    """The shared post-training rollout closure `TrajectoryLoggingCallback`
    consumes -- previously duplicated verbatim inside both apps'
    `build_engine` bodies.

    Args:
        controller: The controller whose fresh policy drives the rollout.
        problem: The problem whose system is simulated.

    Returns:
        ``batch -> (states, observations, controls)``.
    """

    def rollout_fn(batch: Any) -> Any:
        initial_state, process_noise, measurement_noise = batch
        policy = controller.get_control_policy()
        with torch.no_grad():
            return problem.system.run(
                policy, initial_state, process_noise, measurement_noise
            )

    return rollout_fn


class ModelRecipe(ABC):
    """One controller family's complete recipe: how to construct the
    controller for a problem, and how to wire it into the engine.

    Concrete recipes are frozen dataclasses (signable specification-time
    data); per-case-study instances differ only in their field values, e.g.
    two `UnfoldedRecipe`s with different `kind`s.

    Attributes:
        family: The registry name of this family (class-level).
        SAMPLER_BACKEND: Which backend this family's batches are authored
            on (class-level; NumPy-native families override).
        label: The per-case-study instance name (tracker/run identity),
            declared as a field on each concrete recipe dataclass.
    """

    family: ClassVar[str]
    SAMPLER_BACKEND: ClassVar[Backend] = Backend.TORCH

    label: str

    @abstractmethod
    def build_controller(
        self, problem: OptimalControlProblem, ctx: ComputeContext
    ) -> Controller:
        """Construct this family's controller for `problem` on `ctx`."""
        raise NotImplementedError

    @abstractmethod
    def build_engine(
        self,
        controller: Controller,
        problem: OptimalControlProblem,
        harness: EngineHarness,
    ) -> Engine:
        """Wire `controller` into a fully-configured, tracked `Engine`."""
        raise NotImplementedError

    @abstractmethod
    def build_synthesizer(
        self, problem: OptimalControlProblem, harness: EngineHarness
    ) -> Synthesizer:
        """This family's two-phase `Synthesizer` (T2.a)."""
        raise NotImplementedError

    def batch_overrides(self) -> tuple[int | None, int | None]:
        """Per-family ``(batch_size, horizon)`` overrides of the shared
        `GaussianBatchSpec` (e.g. COCP's smaller training scale); ``None``
        keeps the spec's value.

        Returns:
            ``(batch_size_override, horizon_override)``.
        """
        return None, None

    def build_rehosted_controller(
        self,
        offline_problem: OptimalControlProblem,
        online_problem: OptimalControlProblem,
        ctx: ComputeContext,
    ) -> SynthesizedController:
        """The aware rebuild (Annex 01 §2.4.1): offline frozen, online live.

        The governing principle is the practical offline/online separation —
        no offline computation may run at the online stage, analytic or not.
        Everything a family computes offline stays derived from
        `offline_problem` (the plant the model was told); the plant it is
        handed, `online_problem`, enters only its per-step solver
        expressions. Trained parameters are NOT loaded here — the producer
        transplants them from the stored record afterwards, because the
        record is the store's and this tier has no store.

        The default covers every family whose construction *is* its online
        expression: the convex policy compiles its QP against the given
        matrices (its trained cost-to-go is overwritten by the transplant),
        and the recurrent baseline consumes no plant at all. Families with a
        genuine offline artifact inside their construction — a Riccati
        cost-to-go stack, a precomputed step — override this with the
        two-problem construction.
        """
        controller = self.build_controller(online_problem, ctx)
        return TrainedControllerArtifact(
            controller=controller,
            context=ctx,
            synthesizer_signature={
                "type": "RehostedRebuild",
                "recipe": self.get_signature(),
            },
        )

    def extra_callbacks(
        self,
        controller: Controller,
        problem: OptimalControlProblem,
        harness: EngineHarness,
    ) -> Sequence[Callback]:
        """Family-specific callbacks (e.g. convergence-replay parameter
        snapshots); may also log family-declared provenance params on
        ``harness.tracker`` (the sanctioned replacement for the M10
        attribute smuggling).

        Returns:
            The extra callbacks; empty by default.
        """
        return ()

    def get_signature(self) -> dict[str, Any]:
        """`Signable` member: this recipe's specification-time fields only
        (the C1 law) -- nested specs (e.g. `TrainingPlan`) contribute their
        own signatures.

        Returns:
            ``{"type":, "family":, **fields}``.
        """
        signature: dict[str, Any] = {
            "type": type(self).__name__,
            "family": self.family,
        }
        for spec_field in dataclasses.fields(self):  # type: ignore[arg-type]  # every concrete recipe is a dataclass
            value = getattr(self, spec_field.name)
            if hasattr(value, "get_signature"):
                value = value.get_signature()
            signature[spec_field.name] = value
        return signature

    # -- shared wiring ------------------------------------------------------

    def _make_sampler(self, harness: EngineHarness) -> tuple[Any, Any]:
        """Build this family's batch sampler + distributions from the shared
        spec, honoring `batch_overrides` and the family's declared backend."""
        batch_size, horizon = self.batch_overrides()
        return harness.batch_spec.build(
            self.SAMPLER_BACKEND,
            torch_dtype=harness.ctx.torch_dtype,
            torch_device=harness.ctx.torch_device,
            batch_size=batch_size,
            horizon=horizon,
        )

    def _effective_batch_size(self, harness: EngineHarness) -> int:
        batch_size, _ = self.batch_overrides()
        return batch_size or harness.batch_spec.batch_size

    def _standard_callbacks(self, harness: EngineHarness) -> list[Callback]:
        """The callbacks every run gets, regardless of family."""
        return [
            # Hardware transparency (Zero-Black-Boxes): every run announces
            # its resolved device/backend/precision before anything else.
            ComputeContextAnnouncementCallback(harness.ctx),
            ExperimentTrackingCallback(),
            ModelCheckpointCallback(),
            MetricsHistoryCallback(),
            # Always-on at the Runner scope: records this run's latency/memory
            # footprint into metadata.json (fine-grained block metrics remain
            # opt-in via SC_PROFILE).
            ProfilingCallback(),
        ]

    def _assemble_callbacks(
        self,
        controller: Controller,
        problem: OptimalControlProblem,
        harness: EngineHarness,
        distributions: Any,
    ) -> list[Callback]:
        """Standard + family callbacks + the signature/trajectory tail, in
        the exact order the pre-S3 apps assembled them.

        The trajectory-logging callback builds its OWN dedicated, deterministic
        evaluation sampler (see below) rather than sharing the caller's
        training sampler -- the training sampler is passed to the `Runner`
        separately, and is deliberately NOT threaded in here."""
        callbacks = self._standard_callbacks(harness)
        callbacks.extend(self.extra_callbacks(controller, problem, harness))
        callbacks.append(ProblemSignatureCallback(problem, controller, distributions))
        # The trajectory artifact is rolled out on a DEDICATED, deterministic
        # evaluation batch -- a FRESH sampler built from the same (seeded)
        # `batch_spec`, never the training `batch_sampler`. Reusing the
        # training sampler would draw the trajectory batch only AFTER every
        # training step has advanced that generator, so the persisted
        # trajectory would depend on the training LENGTH (e.g. the layer-wise
        # contender's `K*warmup + refinement`, which varies with depth K):
        # different contenders and different depths would each roll out on a
        # different (x0, w), and no external baseline (e.g. the notebook's
        # Riccati rollout) could reproduce any of them. A fresh sampler's
        # first draw is fixed and identical for every contender/depth, so
        # every `trajectory_controls` artifact is directly comparable and the
        # exact batch is reconstructible outside training.
        trajectory_sampler, _ = self._make_sampler(harness)
        callbacks.append(
            TrajectoryLoggingCallback(
                _make_rollout_fn(controller, problem), trajectory_sampler
            )
        )
        return callbacks


class AnalyticRecipe(ModelRecipe):
    """Base wiring for closed-form or frozen families: no optimizer, one
    `AnalyticalStrategy` epoch (simulate + score)."""

    def build_engine(
        self,
        controller: Controller,
        problem: OptimalControlProblem,
        harness: EngineHarness,
    ) -> Engine:
        """See `ModelRecipe.build_engine`.

        The `TrainingConfig` carries no learning rate: there is no optimizer
        to describe (and `ExperimentTrackingCallback` omits optimizer-only
        fields for non-trainable runs anyway).
        """
        batch_sampler, distributions = self._make_sampler(harness)
        callbacks = self._assemble_callbacks(
            controller, problem, harness, distributions
        )
        config = TrainingConfig(
            batch_size=self._effective_batch_size(harness),
            log_every=harness.log_every,
        )
        return Runner(
            model=controller,
            tracker=harness.tracker,
            config=config,
            batch_sampler=batch_sampler,
            phases=[
                TrainingPhase(AnalyticalStrategy(controller), epochs=1, name=self.label)
            ],
            callbacks=callbacks,
        )

    def build_synthesizer(
        self, problem: OptimalControlProblem, harness: EngineHarness
    ) -> Synthesizer:
        """Default for frozen (non-solving, non-training) torch families:
        constructing the controller IS the entire offline phase. Families
        with a real solve (Riccati, truncated Riccati) override this with
        their direct solver-backed synthesizers.
        """
        return FrozenControllerSynthesizer(recipe=self, ctx=harness.ctx)


class TrainableRecipe(ModelRecipe):
    """Base wiring for torch-trained families. Concrete recipes declare a
    ``plan: TrainingPlan`` field; the optimizer is built from it through the
    controller's `as_module` seam -- no per-family parameter-extraction
    ladders, and no second home for the learning rate (C4)."""

    plan: TrainingPlan

    def build_engine(
        self,
        controller: Controller,
        problem: OptimalControlProblem,
        harness: EngineHarness,
    ) -> Engine:
        """See `ModelRecipe.build_engine`.

        `RolloutModel` always runs its own rollout and computes a
        torch-native (differentiable) cost -- required for gradient
        training, since e.g. UnfoldedController.forward()'s own cost is
        numpy-only (fine for AnalyticalStrategy's no-grad evaluation, not
        for backprop).

        The `problem` handed in is the WORLD the rollout runs in (Annex 01
        §2.3.5): every caller before D25 passed the controller's own, so
        threading it here is a no-op for them, and the one caller that
        passes a different plant gets exactly the unweld the field exists
        for.
        """
        rollout = RolloutModel(controller, problem=problem)
        # Trainable families satisfy the T2.b seam; the base Controller
        # protocol doesn't declare it.
        module = controller.as_module()  # type: ignore[attr-defined]
        strategy = GradientDescentStrategy.from_plan(
            rollout, module, self.plan, microbatch=harness.microbatch
        )
        batch_sampler, distributions = self._make_sampler(harness)
        callbacks = self._assemble_callbacks(
            controller, problem, harness, distributions
        )
        # The resumability seam (T3.b): every trained run leaves a
        # TrainingState snapshot next to its model checkpoint.
        callbacks.append(TrainingStateCheckpointCallback(module, strategy.optimizer))
        # The systemic C4 fix: the logged config derives from the SAME plan
        # the strategy's optimizer was built from.
        config = self.plan.training_config(
            batch_size=self._effective_batch_size(harness),
            log_every=harness.log_every,
        )
        return Runner(
            model=rollout,
            tracker=harness.tracker,
            config=config,
            batch_sampler=batch_sampler,
            phases=[TrainingPhase(strategy, epochs=self.plan.epochs, name=self.label)],
            callbacks=callbacks,
        )

    def build_synthesizer(
        self, problem: OptimalControlProblem, harness: EngineHarness
    ) -> Synthesizer:
        """The engine-backed synthesizer (T3.a): offline training through
        the same wiring as `build_engine`."""
        return EngineTrainedSynthesizer(recipe=self, harness=harness)


@dataclass(frozen=True)
class TrainedControllerArtifact:
    """The frozen offline artifact of a trained (or frozen) torch family:
    the controller with its final parameters, plus synthesis provenance.
    Satisfies the `TrainableController` protocol (T2.b).

    Attributes:
        controller: The trained controller (policies/modules read its live,
            now-final parameters).
        context: The effective `ComputeContext` the artifact was synthesized
            under (residency law).
        synthesizer_signature: The producing synthesizer's specification-time
            signature.
        provenance: Declared synthesis facts (executed training plan, final
            training metrics, family-specific values such as an SDP lower
            bound) -- carried data, never signed (live values are not
            specification).
    """

    controller: Any
    context: ComputeContext
    synthesizer_signature: dict[str, Any]
    provenance: dict[str, Any] = field(default_factory=dict)

    def make_policy(self) -> ControlPolicy:
        """A fresh policy closure per call (freshness law).

        Returns:
            The controller's batched ``(t, y_t) -> u_t`` policy.
        """
        return cast(ControlPolicy, self.controller.get_control_policy())

    def as_module(self) -> torch.nn.Module:
        """The trainable-module seam of the underlying controller (T2.b)."""
        return cast(torch.nn.Module, self.controller.as_module())

    def get_signature(self) -> dict[str, Any]:
        """Synthesizer signature + synthesis provenance identity -- never
        ``problem.*``, never live parameter values (C1 law).

        Returns:
            ``{"type":, "synthesizer":, "compute_context":}``.
        """
        return {
            "type": type(self).__name__,
            "synthesizer": dict(self.synthesizer_signature),
            "compute_context": self.context.get_signature(),
        }


@dataclass(frozen=True)
class EngineTrainedSynthesizer:
    """Engine-backed `Synthesizer` for trainable families (T3.a):
    ``synthesize`` = build controller -> wrap in `RolloutModel` -> train via
    the Runner assembled from the recipe's `TrainingPlan` -> freeze into a
    `TrainedControllerArtifact`. Stateless (frozen): every call trains a
    fresh controller.

    Attributes:
        recipe: The owning `TrainableRecipe` (specification + wiring).
        harness: The execution resources; when its tracker should not
            receive this synthesis (e.g. standalone/offline use), construct
            with a `NullExperimentTracker`.
        data_problem: The WORLD the training rolls out in, when it is not
            the plant the controller is built from (Annex 01 §2.3.5, D25).
            `None` — the default, and every pre-D25 caller — trains in the
            controller's own plant.
    """

    recipe: TrainableRecipe
    harness: EngineHarness
    data_problem: OptimalControlProblem | None = None

    def synthesize(
        self, problem: OptimalControlProblem, ctx: ComputeContext | None = None
    ) -> TrainedControllerArtifact:
        """Train this family against `problem` and freeze the result.

        Offline phase only: the engine's ``train()`` runs; evaluation of the
        resulting policy is the caller's online concern.

        Args:
            problem: The `OptimalControlProblem` the controller is built
                from — the plant the contender is *told*.
            ctx: Optional context override; defaults to the harness context.

        Returns:
            The frozen `TrainedControllerArtifact`.
        """
        effective_ctx = ctx if ctx is not None else self.harness.ctx
        harness = (
            self.harness
            if effective_ctx == self.harness.ctx
            else dataclasses.replace(self.harness, ctx=effective_ctx)
        )
        logger.info("Synthesis (training) started for %r", self.recipe.label)
        controller = self.recipe.build_controller(problem, effective_ctx)
        world = self.data_problem if self.data_problem is not None else problem
        engine = self.recipe.build_engine(controller, world, harness)
        final_metrics = dict(engine.train())
        logger.info("Synthesis finished for %r: %s", self.recipe.label, final_metrics)
        provenance: dict[str, Any] = {
            "training": self.recipe.plan.get_signature(),
            "final_metrics": final_metrics,
        }
        if self.data_problem is not None:
            # The world's identity reaches ModelID through the spec; this is
            # the human-facing breadcrumb that a mismatch training happened.
            provenance["trained_in_declared_world"] = True
        return TrainedControllerArtifact(
            controller=controller,
            context=effective_ctx,
            synthesizer_signature=self.get_signature(),
            provenance=provenance,
        )

    def get_signature(self) -> dict[str, Any]:
        """Specification-time configuration only (C1 law): the recipe.

        Returns:
            ``{"type": "EngineTrainedSynthesizer", "recipe": {...}}``.
        """
        return {"type": type(self).__name__, "recipe": self.recipe.get_signature()}


@dataclass(frozen=True)
class FrozenControllerSynthesizer:
    """`Synthesizer` for frozen torch families (fixed unfolded, SDP-seeded
    COCP lower bound): the offline phase is the controller's construction --
    no training, no solve beyond what construction performs.

    Attributes:
        recipe: The owning recipe.
        ctx: The context the controller is authored on.
    """

    recipe: ModelRecipe
    ctx: ComputeContext

    def synthesize(
        self, problem: OptimalControlProblem, ctx: ComputeContext | None = None
    ) -> TrainedControllerArtifact:
        """Construct the frozen controller and wrap it as the artifact.

        Args:
            problem: The `OptimalControlProblem` the controller serves.
            ctx: Optional context override; defaults to the recipe's.

        Returns:
            The frozen `TrainedControllerArtifact` (empty training
            provenance -- nothing was trained).
        """
        effective_ctx = ctx if ctx is not None else self.ctx
        controller = self.recipe.build_controller(problem, effective_ctx)
        return TrainedControllerArtifact(
            controller=controller,
            context=effective_ctx,
            synthesizer_signature=self.get_signature(),
        )

    def get_signature(self) -> dict[str, Any]:
        """Specification-time configuration only (C1 law): the recipe.

        Returns:
            ``{"type": "FrozenControllerSynthesizer", "recipe": {...}}``.
        """
        return {"type": type(self).__name__, "recipe": self.recipe.get_signature()}


def null_harness(batch_spec: BatchSpec, ctx: ComputeContext) -> EngineHarness:
    """An `EngineHarness` for untracked synthesis (nothing persisted):
    convenience for standalone `Synthesizer` use and tests.

    Args:
        batch_spec: The training-batch specification.
        ctx: The compute context.

    Returns:
        A harness bound to a `NullExperimentTracker`.
    """
    return EngineHarness(
        batch_spec=batch_spec, tracker=NullExperimentTracker(), ctx=ctx
    )


# Re-exported for concrete recipe modules' typing convenience.
__all__ = [
    "RECIPE_REGISTRY",
    "register_recipe",
    "build_default_recipe_registry",
    "EngineHarness",
    "ModelRecipe",
    "AnalyticRecipe",
    "TrainableRecipe",
    "TrainedControllerArtifact",
    "EngineTrainedSynthesizer",
    "FrozenControllerSynthesizer",
    "null_harness",
    "TrainableController",
]
