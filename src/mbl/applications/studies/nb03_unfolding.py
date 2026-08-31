"""NB03 -- Deep-Unfolded LQR Benchmarking: the Tier-3 `Experiment`
declaration (docs/planning/03_studies/nb03_deep_unfolded_lqr/deep_unfolding_benchmark_blueprint_v2.md SS2). Four contenders on
one shared stochastic LQR instance, common-noise evaluated:

    Sim-Riccati        -- the exact finite-horizon Riccati optimum, scored
                          by Monte-Carlo rollout (family "riccati").
    Standard-GD        -- fixed step-size unrolled GD, true Riccati P,
                          nothing learned (family "unfolded_fixed").
    Unfolded-alpha     -- learned per-iteration step size, true Riccati P,
                          trained end to end (family "unfolded").
    Unfolded-alpha+P   -- learned per-iteration step size AND a learned
                          time-invariant matrix replacing Riccati P, trained
                          LAYER-WISE (family "unfolded_warmstart") -- the
                          contender that exercises the NB03 flagship build.

A new study module, not an edit to `applications.standard_lqr`: that case
study is a product artifact under golden-master test; this is a new research
declaration (the S3 law -- new declarations are new modules, never edits to
a benchmarked one).

Theo-Riccati (the closed-form analytical baseline) is deliberately NOT a
`ContenderSpec` here: the shared `EvaluationProtocol` only scores Monte-Carlo
rollouts, and Theo-Riccati is a closed-form NUMBER
(`models.analytic.riccati.compute_theoretical_expected_cost`). It enters the
notebook as a `reference_lines` baseline computed directly from the
experiment's own problem, never through `run_experiment`.
"""

from dataclasses import dataclass, field
from typing import Literal

import torch

from ..factories import GaussianBatchSpec, LQRProblemFactory
from ..recipes import UnfoldedKind
from ...core.runtime import Backend, ComputeContext, Precision
from ...engine.training_plan import LayerwiseTrainingPlan, OptimizerSpec, TrainingPlan
from ...experiments import ContenderSpec, EvaluationProtocol, Experiment
from ...models.iterative import ControlInitMethod

#: A learned unfolded model's training regime, a free per-model choice
#: (`UnfoldedModelConfig.training_mode`). ``"layerwise"`` trains greedily
#: layer-by-layer (family ``unfolded_warmstart``, driven by the model's
#: `schedule`); ``"end_to_end"`` trains the SAME learned parameters jointly
#: from scratch (family ``unfolded``, driven by the model's `plan``) -- the two
#: regimes differ only in HOW the identical parameter set is optimized, never
#: in what is learned. Both regimes work for both learned kinds: the layer-wise
#: machinery is kind-agnostic (`LayerwiseTrainingPlan.compile(has_matrix=...)`),
#: simply skipping the matrix-activation step for a model with no learnable
#: matrix (Unfolded-alpha).
TrainingMode = Literal["layerwise", "end_to_end"]

#: NB03's experiment name (the run/registry identity every contender's run
#: directory is prefixed with).
EXPERIMENT_NAME = "NB03_DeepUnfoldedLQR"


def _default_standard_plan() -> TrainingPlan:
    """Unfolded-alpha's flat, end-to-end `TrainingPlan` (contender #3)."""
    return TrainingPlan(optimizer=OptimizerSpec("adam", 1e-2), epochs=200)


def _default_warmstart_schedule() -> LayerwiseTrainingPlan:
    """Unfolded-alpha+P's layer-wise `LayerwiseTrainingPlan` (contender #4):
    one 25-epoch warm-up phase per unfolding iteration, then a 100-epoch
    end-to-end refinement phase that first activates the unified matrix --
    comparable total budget to `_default_standard_plan`'s 200 epochs."""
    return LayerwiseTrainingPlan(
        optimizer=OptimizerSpec("adam", 1e-2),
        warmup_epochs_per_layer=25,
        refinement_epochs=100,
        train_matrix_from="refinement",
    )


@dataclass(frozen=True)
class UnfoldedModelConfig:
    """The per-model configuration of ONE learned unfolded contender
    (Unfolded-alpha or Unfolded-alpha+P): which learned kind it is, how its
    ``u^(0)`` is seeded, which training regime it runs, and the plan/schedule
    each regime uses. Making the regime a per-model field (rather than a fixed
    property of the model) is what lets both learned models run either mode
    (see `TrainingMode`).

    Attributes:
        kind: The learned unfolded configuration -- ``LEARNED_STEP_SIZE``
            (Unfolded-alpha) or ``LEARNED_STEP_SIZE_AND_MATRIX``
            (Unfolded-alpha+P).
        init_method: How this model seeds ``u^(0)`` before refinement
            (`ControlInitMethod`).
        training_mode: ``"end_to_end"`` (flat joint training via `plan`) or
            ``"layerwise"`` (greedy warm-start via `schedule`).
        plan: The flat `TrainingPlan` used when
            ``training_mode == "end_to_end"``.
        schedule: The `LayerwiseTrainingPlan` used when
            ``training_mode == "layerwise"``.
    """

    kind: UnfoldedKind
    init_method: ControlInitMethod = ControlInitMethod.COLD
    training_mode: TrainingMode = "end_to_end"
    plan: TrainingPlan = field(default_factory=_default_standard_plan)
    schedule: LayerwiseTrainingPlan = field(default_factory=_default_warmstart_schedule)


def _default_unfolded_alpha() -> UnfoldedModelConfig:
    """Unfolded-alpha's default: learned step size, cold start, end to end."""
    return UnfoldedModelConfig(kind=UnfoldedKind.LEARNED_STEP_SIZE)


def _default_unfolded_alpha_p() -> UnfoldedModelConfig:
    """Unfolded-alpha+P's default: learned step size AND matrix, cold start,
    trained layer-wise (the notebook's flagship regime)."""
    return UnfoldedModelConfig(
        kind=UnfoldedKind.LEARNED_STEP_SIZE_AND_MATRIX, training_mode="layerwise"
    )


@dataclass(frozen=True)
class NB03Config:
    """NB03's frozen settings: the shared problem/evaluation scale plus each
    trainable family's own training specification.

    Attributes:
        state_dim: State dimension ``n``.
        control_dim: Control dimension ``m``.
        horizon: The finite horizon ``N``.
        num_unfolding_iterations: Unrolled depth ``J``, shared by every
            unfolded contender (and the layer count `warmstart_schedule`
            compiles into).
        step_size_init: Initial per-iteration step size.
        step_size_max: Step-size reparameterization upper bound.
        batch_size: Trajectories per evaluation batch.
        n_eval_batches: Evaluation batches drawn and averaged (``> 1``
            additionally reports ``eval_expected_cost_std``).
        seed: Problem/batch seed.
        process_noise_std: Process noise standard deviation.
        initial_state_std: Initial-state standard deviation of the shared
            evaluation/training batch (Stochastic Fairness doctrine, point 4).
        standard_gd_init_method: How Standard-GD (the fixed-step iterative
            contender) seeds ``u^(0)`` (`ControlInitMethod`).
        unfolded_alpha: Unfolded-alpha's per-model config -- learned step size,
            with its own init method and training regime
            (`UnfoldedModelConfig`).
        unfolded_alpha_p: Unfolded-alpha+P's per-model config -- learned step
            size AND matrix, with its own init method and training regime
            (`UnfoldedModelConfig`).
        random_init_std: Std of the Gaussian control-init draw, shared by
            every contender whose ``init_method`` is
            `ControlInitMethod.RANDOMIZED` (seeded from `seed` for
            reproducibility).
        log_first_epochs: ``a`` of the dynamic training-log narration schedule
            (NB03 Phase B, directive 2): the leading epochs narrated
            consecutively before the coarser cadence begins.
        log_every_epochs: ``b`` of the dynamic schedule: the narration cadence
            after the leading block (the final epoch is always narrated).
        dtype: The working torch precision.
        device: The torch device every contender's `ComputeContext` resolves
            to (e.g. ``"cpu"``, ``"cuda"``, ``"cuda:1"``). Signature-relevant
            by construction (`ComputeContext.get_signature` folds it into
            `Experiment.contender_content_digest`), so a run on ``"cuda"``
            and a run on ``"cpu"`` are distinct cache entries -- a CUDA
            profile is never silently served in place of a CPU one, or vice
            versa.
    """

    state_dim: int = 4
    control_dim: int = 2
    horizon: int = 30
    num_unfolding_iterations: int = 8
    step_size_init: float = 1e-2
    step_size_max: float = 5e-2
    batch_size: int = 512
    n_eval_batches: int = 8
    seed: int = 0
    process_noise_std: float = 0.1
    initial_state_std: float = 1.0
    standard_gd_init_method: ControlInitMethod = ControlInitMethod.COLD
    unfolded_alpha: UnfoldedModelConfig = field(default_factory=_default_unfolded_alpha)
    unfolded_alpha_p: UnfoldedModelConfig = field(
        default_factory=_default_unfolded_alpha_p
    )
    random_init_std: float = 1.0
    log_first_epochs: int = 10
    log_every_epochs: int = 20
    dtype: torch.dtype = torch.float64
    device: str = "cpu"


def build_nb03_config(  # noqa: PLR0913 -- a deliberate keyword-only FLAT
    # mirror of the `NB03Config` carrier (@dataclass is itself exempt from the
    # arg-count gate): the notebook assembles the carrier from flat,
    # individually-typed, discoverable top-level settings rather than authoring
    # an opaque `NB03Config(...)` literal (NB03 directive 1). A `**kwargs`
    # passthrough would satisfy the gate numerically but erase exactly that
    # per-parameter type/IDE discoverability -- the wrong trade here. The two
    # learned models' OWN settings are grouped into one `UnfoldedModelConfig`
    # literal each (directive 1 exempts a model's own natural grouping -- the
    # law is "no NB03Config(...) literal", not "no nested spec literals";
    # every other NB03Config field participating in `TrainingPlan`/
    # `LayerwiseTrainingPlan` already works the same way).
    *,
    state_dim: int,
    control_dim: int,
    horizon: int,
    num_unfolding_iterations: int,
    step_size_init: float,
    step_size_max: float,
    batch_size: int,
    n_eval_batches: int,
    seed: int,
    process_noise_std: float,
    unfolded_alpha: UnfoldedModelConfig,
    unfolded_alpha_p: UnfoldedModelConfig,
    initial_state_std: float = 1.0,
    standard_gd_init_method: ControlInitMethod = ControlInitMethod.COLD,
    random_init_std: float = 1.0,
    log_first_epochs: int = 10,
    log_every_epochs: int = 20,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
) -> NB03Config:
    """Assemble the internal `NB03Config` carrier from flat, standalone
    settings (NB03 directive 1): the notebook exposes every knob as a
    top-level tunable variable and calls this to build the carrier the
    factory/sweep pass around -- it never authors an `NB03Config(...)` literal
    itself, and no parameter is encapsulated away inside an opaque wrapper.

    `initial_state_std` is threaded through to the shared evaluation/training
    `GaussianBatchSpec` (Stochastic Fairness doctrine, point 4: every contender
    is scored on the identical seeded batch distribution).

    Args:
        See `NB03Config` for every field's meaning; this is a flat, keyword-
        only mirror of that carrier's constructor plus `initial_state_std`.

    Returns:
        The assembled `NB03Config`.
    """
    return NB03Config(
        state_dim=state_dim,
        control_dim=control_dim,
        horizon=horizon,
        num_unfolding_iterations=num_unfolding_iterations,
        step_size_init=step_size_init,
        step_size_max=step_size_max,
        batch_size=batch_size,
        n_eval_batches=n_eval_batches,
        seed=seed,
        process_noise_std=process_noise_std,
        initial_state_std=initial_state_std,
        standard_gd_init_method=standard_gd_init_method,
        unfolded_alpha=unfolded_alpha,
        unfolded_alpha_p=unfolded_alpha_p,
        random_init_std=random_init_std,
        log_first_epochs=log_first_epochs,
        log_every_epochs=log_every_epochs,
        dtype=dtype,
        device=device,
    )


def _learned_unfolded_contender(
    label: str,
    model: UnfoldedModelConfig,
    *,
    shape: dict[str, object],
    log_kwargs: dict[str, object],
    random_init_std: float,
    seed: int,
) -> ContenderSpec:
    """One learned unfolded contender's `ContenderSpec`, built from its own
    `UnfoldedModelConfig` -- shared by Unfolded-alpha and Unfolded-alpha+P, the
    two models that differ only in `model.kind` (and, in general, every other
    `UnfoldedModelConfig` field) yet are wired identically otherwise: EITHER
    regime works for EITHER learned kind (see `TrainingMode`).

    Args:
        label: The contender's instance label (``"unfolded_alpha"`` or
            ``"unfolded_alpha_p"``).
        model: This contender's own per-model configuration.
        shape: The shared problem-shape kwargs (iterations/step-size/horizon).
        log_kwargs: The shared dynamic training-log narration kwargs.
        random_init_std: Std of the Gaussian draw for
            `ControlInitMethod.RANDOMIZED`; ignored otherwise.
        seed: Seed for the reproducible, device-native generator of
            `ControlInitMethod.RANDOMIZED`; ignored otherwise.

    Returns:
        The `ContenderSpec`, routed to family ``"unfolded_warmstart"``
        (layer-wise) or ``"unfolded"`` (end-to-end) per
        ``model.training_mode`` -- same label, same learned kind; only the
        family/schedule differs, so its cache identity still distinguishes
        the two regimes (the plan/schedule participates in `get_signature`).
    """
    init_kwargs: dict[str, object] = {
        "init_method": model.init_method,
        "random_init_std": random_init_std,
        "random_init_seed": seed,
    }
    family: str
    plan_kwargs: dict[str, object]
    if model.training_mode == "layerwise":
        family, plan_kwargs = "unfolded_warmstart", {"schedule": model.schedule}
    else:
        family, plan_kwargs = "unfolded", {"plan": model.plan}
    return ContenderSpec(
        family=family,
        config={
            "kind": model.kind,
            **plan_kwargs,
            **shape,
            **log_kwargs,
            **init_kwargs,
        },
        label=label,
    )


def nb03_unfolding_experiment(cfg: NB03Config) -> Experiment:
    """Compose the NB03 declaration from `cfg`.

    Args:
        cfg: The frozen NB03 settings.

    Returns:
        The `Experiment` (problem factory + four contenders + shared
        common-noise evaluation), ready for ``run_experiment``.
    """
    ctx = ComputeContext(
        backend=Backend.TORCH,
        device=cfg.device,
        precision=Precision.from_torch_dtype(cfg.dtype),
    )
    shape: dict[str, object] = {
        "num_iterations": cfg.num_unfolding_iterations,
        "step_size_init": cfg.step_size_init,
        "step_size_max": cfg.step_size_max,
        "horizon": cfg.horizon,
    }
    # The dynamic training-log schedule (directive 2), forwarded only to the
    # learned families (`unfolded`/`unfolded_warmstart`) whose recipes attach a
    # StructuredTrainingLogCallback -- never to the analytic contenders, whose
    # recipes carry no such field.
    log_kwargs: dict[str, object] = {
        "log_first_epochs": cfg.log_first_epochs,
        "log_every_epochs": cfg.log_every_epochs,
    }
    return Experiment(
        name=EXPERIMENT_NAME,
        problem=LQRProblemFactory(
            state_dim=cfg.state_dim,
            control_dim=cfg.control_dim,
            horizon=cfg.horizon,
            seed=cfg.seed,
        ),
        contenders=(
            ContenderSpec(
                family="riccati",
                config={"horizon": cfg.horizon},
                label="sim_riccati",
            ),
            ContenderSpec(
                family="unfolded_fixed",
                config={
                    **shape,
                    "init_method": cfg.standard_gd_init_method,
                    "random_init_std": cfg.random_init_std,
                    "random_init_seed": cfg.seed,
                },
                label="standard_gd",
            ),
            _learned_unfolded_contender(
                "unfolded_alpha",
                cfg.unfolded_alpha,
                shape=shape,
                log_kwargs=log_kwargs,
                random_init_std=cfg.random_init_std,
                seed=cfg.seed,
            ),
            _learned_unfolded_contender(
                "unfolded_alpha_p",
                cfg.unfolded_alpha_p,
                shape=shape,
                log_kwargs=log_kwargs,
                random_init_std=cfg.random_init_std,
                seed=cfg.seed,
            ),
        ),
        evaluation=EvaluationProtocol(
            batch_spec=GaussianBatchSpec(
                state_dim=cfg.state_dim,
                horizon=cfg.horizon,
                batch_size=cfg.batch_size,
                seed=cfg.seed,
                process_noise_std=cfg.process_noise_std,
                initial_state_std=cfg.initial_state_std,
            ),
            n_batches=cfg.n_eval_batches,
        ),
        ctx=ctx,
    )
