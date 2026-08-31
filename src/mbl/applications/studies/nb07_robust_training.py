"""NB07 -- Robust Training under Model Mismatch and Uncertainty (Experiment
2): the Tier-3 declaration (docs/planning/03_studies/nb07_robust_training/robust_training_under_model_mismatch.md),
extending NB04's box-constrained instance with six uncertainty-training
axes. Six training-sweep contenders, `cocp` excluded from every sweep (plan
Sec 13 decision 2):

    Truncated-Riccati  -- unconstrained Riccati optimum, saturated; non-
                          learnable (family "truncated_riccati").
    Standard-PGD       -- fixed step-size projected gradient descent, true
                          Riccati P, nothing learned ("unfolded_fixed").
    Unfolded-alpha     -- learned per-iteration step size, true Riccati P
                          ("unfolded"/"unfolded_warmstart").
    Unfolded-alpha+P   -- learned step size AND a learned matrix replacing
                          Riccati P -- the flagship under NOMINAL-mode model
                          mismatch (Sec 4.2).
    GRU                -- recurrent neural policy, no internal model at all
                          -- the structural asymmetry every axis probes
                          ("neural").
    COCP-LB            -- the SDP-frozen QP policy; analytic (no training),
                          same cost class as an SDP floor solve
                          ("cocp_lower_bound").

`cocp` (the trainable QP) trains exactly ONCE, as the nominal-training
anchor `nb07_nominal_anchor_experiment` shares with NB04's own declaration
shape -- it never appears in `six_training_sweep_contenders` and is never
retrained under any uncertainty condition (plan Sec 13 decision 2, driven by
its measured ~21-1259s per training, NB04 run directories).

`UnfoldedModelConfig`/`TrainingMode` are REUSED VERBATIM from `.nb03_unfolding`
(imported, never redefined) -- the same S3-law reuse `nb04_box_constrained.py`
and `nb05_ltv_box_constrained.py` already apply.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import torch

from .nb03_unfolding import UnfoldedModelConfig
from ..factories import GaussianBatchSpec, LQRProblemFactory
from ..recipes import (
    COCPLowerBoundRecipe,
    FixedUnfoldedRecipe,
    NeuralRecipe,
    TruncatedRiccatiRecipe,
    UnfoldedKind,
    UnfoldedRecipe,
    WarmStartUnfoldedRecipe,
)
from ..recipes.base import EngineHarness, ModelRecipe
from ..uncertainty.dataset import FiniteTrajectoryDatasetSpec
from ..uncertainty.noise import ExoticBatchSpec, NoiseFamily
from ..uncertainty.perturbations import (
    PerturbationDistribution,
    PerturbationKind,
    PlantPerturbation,
)
from ...core.optimal_control_problem import OptimalControlProblem
from ...core.runtime import Backend, ComputeContext, Precision
from ...core.system.linear_system import LinearSystem
from ...core.utils import ensure_positive_integer
from ...engine.training_plan import OptimizerSpec, TrainingPlan
from ...experiments import ContenderSpec, EvaluationProtocol, Experiment
from ...experiments.experiment import EvaluationBatch
from ...models.guards import require_linear_quadratic
from ...models.iterative import ControlInitMethod
from ...persistence.null_tracker import NullExperimentTracker
from ...workbench.robust_training_sweep import TrainingCondition, UncertaintyAxis

#: NB07's experiment name (the nominal-anchor run's identity).
EXPERIMENT_NAME = "NB07_RobustTrainingUnderUncertainty"

#: The declared, uniform gradient-clip norm every trainable family's plan
#: uses (NB07 plan Sec 4.5) -- never enabled only where a family struggles.
DEFAULT_GRADIENT_CLIP_NORM = 5.0

#: Fixed seed roots (NB07 plan Sec 5.4's four seed roles), offset far apart
#: so `base_seed + seed_index` (training data / perturbation / init) can
#: never collide with the shared, level-invariant evaluation/test streams.
EVAL_SEED = 10_000
TEST_SEED = 30_000


def _default_unfolded_alpha() -> UnfoldedModelConfig:
    return UnfoldedModelConfig(kind=UnfoldedKind.LEARNED_STEP_SIZE)


def _default_unfolded_alpha_p() -> UnfoldedModelConfig:
    return UnfoldedModelConfig(
        kind=UnfoldedKind.LEARNED_STEP_SIZE_AND_MATRIX, training_mode="layerwise"
    )


def _default_gru_plan() -> TrainingPlan:
    return TrainingPlan(
        optimizer=OptimizerSpec("adam", 1e-2),
        epochs=100,
        gradient_clip_norm=DEFAULT_GRADIENT_CLIP_NORM,
    )


def _default_cocp_plan() -> TrainingPlan:
    return TrainingPlan(
        optimizer=OptimizerSpec("adam", 0.1),
        epochs=100,
        gradient_clip_norm=DEFAULT_GRADIENT_CLIP_NORM,
    )


@dataclass(frozen=True)
class NB07Config:
    """NB07's frozen settings: NB04's box-constrained problem scale plus the
    six uncertainty-sweep ranges and their statistical protocol.

    Attributes:
        state_dim: State dimension ``n``.
        control_dim: Control dimension ``m``.
        horizon: The NOMINAL finite horizon ``N``.
        u_max: The infinity-norm control bound.
        num_unfolding_iterations: Unrolled depth, shared by every unfolded
            contender.
        step_size_init: Initial per-iteration step size.
        step_size_max: Step-size reparameterization upper bound.
        batch_size: Trajectories per training/evaluation batch.
        n_eval_batches: Evaluation batches drawn and averaged for the
            nominal-anchor experiment.
        seed: The nominal problem's own seed.
        process_noise_std: Nominal process noise standard deviation.
        initial_state_std: Nominal initial-state standard deviation.
        standard_pgd_init_method: Standard-PGD's ``u^(0)`` seeding method.
        unfolded_alpha: Unfolded-alpha's per-model config (reused verbatim
            from `nb03_unfolding`); its own `plan`/`schedule` should be
            constructed with `gradient_clip_norm=DEFAULT_GRADIENT_CLIP_NORM`
            for the uniform-clipping law (Sec 4.5) to actually hold --
            `build_nb07_config`'s caller's responsibility, exactly as NB04
            documents for its own `cocp_plan`/`unfolded_alpha` parity.
        unfolded_alpha_p: Unfolded-alpha+P's per-model config, same note.
        gru_hidden_dim: The GRU backbone's hidden dimension (selected by the
            notebook's own pre-ablation, Sec 7 Table 6).
        gru_plan: The GRU's `TrainingPlan`.
        gru_init_seed: `NeuralRecipe.init_seed` -- the Sec 2.4 determinism
            fix; distinct from every training-data/perturbation seed.
        cocp_plan: COCP's `TrainingPlan` -- used ONLY by the nominal-anchor
            experiment (COCP is never part of a sweep, Sec 13 decision 2).
        cocp_solver_eps: COCP/COCP-LB solver convergence tolerance.
        cocp_solver_max_iters: COCP/COCP-LB solver iteration cap.
        cocp_batch_size: Optional COCP/COCP-LB batch override.
        cocp_horizon: Optional COCP horizon override (nominal anchor only).
        random_init_std: Std of the Gaussian control-init draw.
        noise_scale_multipliers: Axis S levels.
        noise_families: Axis X levels.
        plant_additive_epsilons: Axis D-add levels.
        plant_rotation_degrees: Axis D-rot levels.
        train_horizons: Axis H levels.
        sample_sizes: Axis (sample complexity) levels.
        n_training_seeds: Seeds per level, every axis (``>= 5``, the
            statistical-rigor mandate; enforced by
            `workbench.robust_training_sweep.RobustTrainingSpec`).
        log_first_epochs: ``a`` of the dynamic training-log narration schedule.
        log_every_epochs: ``b`` of the dynamic schedule.
        dtype: The working torch precision.
        device: The torch device every contender's `ComputeContext` resolves to.
    """

    state_dim: int = 4
    control_dim: int = 2
    horizon: int = 50
    u_max: float = 0.5
    num_unfolding_iterations: int = 8
    step_size_init: float = 0.05
    step_size_max: float = 1.0
    batch_size: int = 256
    n_eval_batches: int = 8
    seed: int = 0
    process_noise_std: float = 0.5
    initial_state_std: float = 1.0
    standard_pgd_init_method: ControlInitMethod = ControlInitMethod.COLD
    unfolded_alpha: UnfoldedModelConfig = field(default_factory=_default_unfolded_alpha)
    unfolded_alpha_p: UnfoldedModelConfig = field(
        default_factory=_default_unfolded_alpha_p
    )
    gru_hidden_dim: int = 64
    gru_plan: TrainingPlan = field(default_factory=_default_gru_plan)
    gru_init_seed: int = 0
    cocp_plan: TrainingPlan = field(default_factory=_default_cocp_plan)
    cocp_solver_eps: float = 1e-8
    cocp_solver_max_iters: int = 10000
    cocp_batch_size: int | None = None
    cocp_horizon: int | None = None
    random_init_std: float = 1.0
    noise_scale_multipliers: tuple[float, ...] = (0.1, 0.5, 1.0, 2.0, 10.0)
    noise_families: tuple[NoiseFamily, ...] = (
        NoiseFamily.GAUSSIAN,
        NoiseFamily.UNIFORM,
        NoiseFamily.LAPLACE,
        NoiseFamily.CAUCHY,
    )
    plant_additive_epsilons: tuple[float, ...] = (0.0, 0.1, 0.2, 0.4)
    plant_rotation_degrees: tuple[float, ...] = (0.0, 15.0, 30.0, 45.0)
    train_horizons: tuple[int, ...] = (25, 50, 100)
    sample_sizes: tuple[int, ...] = (16, 32, 64, 128, 256, 1024, 4096)
    n_training_seeds: int = 7
    log_first_epochs: int = 10
    log_every_epochs: int = 20
    dtype: torch.dtype = torch.float64
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.u_max <= 0:
            raise ValueError(f"u_max must be strictly positive, got {self.u_max}.")
        ensure_positive_integer(self.n_training_seeds, "n_training_seeds")


def build_nb07_config(  # noqa: PLR0913 -- a deliberate keyword-only FLAT
    # mirror of the NB07Config carrier, matching NB03/04/05's own directive:
    # the notebook assembles the carrier from flat, individually-typed,
    # discoverable top-level settings rather than an opaque NB07Config(...)
    # literal.
    *,
    state_dim: int,
    control_dim: int,
    horizon: int,
    u_max: float,
    num_unfolding_iterations: int,
    step_size_init: float,
    step_size_max: float,
    batch_size: int,
    n_eval_batches: int,
    seed: int,
    process_noise_std: float,
    unfolded_alpha: UnfoldedModelConfig,
    unfolded_alpha_p: UnfoldedModelConfig,
    gru_hidden_dim: int,
    gru_plan: TrainingPlan,
    cocp_plan: TrainingPlan,
    noise_scale_multipliers: tuple[float, ...],
    noise_families: tuple[NoiseFamily, ...],
    plant_additive_epsilons: tuple[float, ...],
    plant_rotation_degrees: tuple[float, ...],
    train_horizons: tuple[int, ...],
    sample_sizes: tuple[int, ...],
    n_training_seeds: int,
    initial_state_std: float = 1.0,
    standard_pgd_init_method: ControlInitMethod = ControlInitMethod.COLD,
    gru_init_seed: int = 0,
    cocp_solver_eps: float = 1e-8,
    cocp_solver_max_iters: int = 10000,
    cocp_batch_size: int | None = None,
    cocp_horizon: int | None = None,
    random_init_std: float = 1.0,
    log_first_epochs: int = 10,
    log_every_epochs: int = 20,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
) -> NB07Config:
    """Assemble the internal `NB07Config` carrier from flat, standalone
    settings. See `NB07Config` for every field's meaning."""
    return NB07Config(
        state_dim=state_dim,
        control_dim=control_dim,
        horizon=horizon,
        u_max=u_max,
        num_unfolding_iterations=num_unfolding_iterations,
        step_size_init=step_size_init,
        step_size_max=step_size_max,
        batch_size=batch_size,
        n_eval_batches=n_eval_batches,
        seed=seed,
        process_noise_std=process_noise_std,
        initial_state_std=initial_state_std,
        standard_pgd_init_method=standard_pgd_init_method,
        unfolded_alpha=unfolded_alpha,
        unfolded_alpha_p=unfolded_alpha_p,
        gru_hidden_dim=gru_hidden_dim,
        gru_plan=gru_plan,
        gru_init_seed=gru_init_seed,
        cocp_plan=cocp_plan,
        cocp_solver_eps=cocp_solver_eps,
        cocp_solver_max_iters=cocp_solver_max_iters,
        cocp_batch_size=cocp_batch_size,
        cocp_horizon=cocp_horizon,
        random_init_std=random_init_std,
        noise_scale_multipliers=noise_scale_multipliers,
        noise_families=noise_families,
        plant_additive_epsilons=plant_additive_epsilons,
        plant_rotation_degrees=plant_rotation_degrees,
        train_horizons=train_horizons,
        sample_sizes=sample_sizes,
        n_training_seeds=n_training_seeds,
        log_first_epochs=log_first_epochs,
        log_every_epochs=log_every_epochs,
        dtype=dtype,
        device=device,
    )


def nb07_compute_context(cfg: NB07Config) -> ComputeContext:
    """The shared `ComputeContext` every NB07 recipe builds/trains under."""
    return ComputeContext(
        backend=Backend.TORCH,
        device=cfg.device,
        precision=Precision.from_torch_dtype(cfg.dtype),
    )


def _learned_unfolded_recipe(
    label: str,
    model: UnfoldedModelConfig,
    *,
    horizon: int,
    cfg: NB07Config,
) -> ModelRecipe:
    """One learned unfolded contender, built DIRECTLY (not via `ContenderSpec`
    /registry -- the sweep engine calls `build_controller`/`build_engine`
    itself, see `workbench.robust_training_sweep`'s module docstring)."""
    init_kwargs: dict[str, Any] = {
        "init_method": model.init_method,
        "random_init_std": cfg.random_init_std,
        "random_init_seed": cfg.seed,
    }
    shape: dict[str, Any] = {
        "num_iterations": cfg.num_unfolding_iterations,
        "step_size_init": cfg.step_size_init,
        "step_size_max": cfg.step_size_max,
        "horizon": horizon,
    }
    if model.training_mode == "layerwise":
        return WarmStartUnfoldedRecipe(
            kind=model.kind,
            schedule=model.schedule,
            label=label,
            log_first_epochs=cfg.log_first_epochs,
            log_every_epochs=cfg.log_every_epochs,
            **shape,
            **init_kwargs,
        )
    return UnfoldedRecipe(
        kind=model.kind,
        plan=model.plan,
        label=label,
        log_first_epochs=cfg.log_first_epochs,
        log_every_epochs=cfg.log_every_epochs,
        **shape,
        **init_kwargs,
    )


def six_training_sweep_contenders(
    cfg: NB07Config, *, horizon: int | None = None
) -> dict[str, ModelRecipe]:
    """The six training-sweep contenders (`cocp` excluded, Sec 13 decision 2).

    Args:
        cfg: The NB07 settings.
        horizon: Overrides `cfg.horizon` for every horizon-bearing recipe --
            the `TRAIN_HORIZON` axis's own seam
            (`workbench.robust_training_sweep.run_robust_training_sweep`'s
            ``recipes_at`` parameter calls this per level).

    Returns:
        label -> `ModelRecipe`.
    """
    effective_horizon = horizon if horizon is not None else cfg.horizon
    return {
        "truncated_riccati": TruncatedRiccatiRecipe(horizon=effective_horizon),
        "standard_pgd": FixedUnfoldedRecipe(
            num_iterations=cfg.num_unfolding_iterations,
            step_size_init=cfg.step_size_init,
            step_size_max=cfg.step_size_max,
            horizon=effective_horizon,
            label="standard_pgd",
            init_method=cfg.standard_pgd_init_method,
            random_init_std=cfg.random_init_std,
            random_init_seed=cfg.seed,
        ),
        "unfolded_alpha": _learned_unfolded_recipe(
            "unfolded_alpha", cfg.unfolded_alpha, horizon=effective_horizon, cfg=cfg
        ),
        "unfolded_alpha_p": _learned_unfolded_recipe(
            "unfolded_alpha_p", cfg.unfolded_alpha_p, horizon=effective_horizon, cfg=cfg
        ),
        "cocp_lower_bound": COCPLowerBoundRecipe(
            process_noise_std=cfg.process_noise_std,
            solver_eps=cfg.cocp_solver_eps,
            solver_max_iters=cfg.cocp_solver_max_iters,
            batch_size=cfg.cocp_batch_size,
        ),
        "neural": NeuralRecipe(
            hidden_dim=cfg.gru_hidden_dim,
            plan=cfg.gru_plan,
            init_seed=cfg.gru_init_seed,
            label="neural",
        ),
    }


def nb07_nominal_anchor_experiment(cfg: NB07Config) -> Experiment:
    """The seven-contender nominal-training anchor (Sec 7): identical in
    shape to NB04's own declaration, plus `neural` -- the ONE run where
    `cocp` trains, and the reference every sweep axis's degeneracy point
    (level 0 / scale 1.0 / Gaussian) must reproduce.

    Args:
        cfg: The NB07 settings.

    Returns:
        The `Experiment`, ready for ``run_experiment``.
    """
    ctx = nb07_compute_context(cfg)
    six = six_training_sweep_contenders(cfg)
    contenders = tuple(ContenderSpec.from_recipe(recipe) for recipe in six.values()) + (
        ContenderSpec(
            family="cocp",
            config={
                "plan": cfg.cocp_plan,
                "solver_eps": cfg.cocp_solver_eps,
                "solver_max_iters": cfg.cocp_solver_max_iters,
                "batch_size": cfg.cocp_batch_size,
                "horizon": cfg.cocp_horizon,
                "log_first_epochs": cfg.log_first_epochs,
                "log_every_epochs": cfg.log_every_epochs,
            },
            label="cocp",
        ),
    )
    return Experiment(
        name=EXPERIMENT_NAME,
        problem=LQRProblemFactory(
            state_dim=cfg.state_dim,
            control_dim=cfg.control_dim,
            horizon=cfg.horizon,
            seed=cfg.seed,
            u_max=cfg.u_max,
        ),
        contenders=contenders,
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


def _nominal_system_matrices(cfg: NB07Config) -> tuple[np.ndarray, np.ndarray]:
    """The nominal `(A, B)` `LQRProblemFactory` draws for `cfg.seed` -- the
    base every plant-perturbation axis perturbs."""
    problem = LQRProblemFactory(
        state_dim=cfg.state_dim,
        control_dim=cfg.control_dim,
        horizon=cfg.horizon,
        seed=cfg.seed,
        u_max=cfg.u_max,
    ).build()
    system, _ = require_linear_quadratic(problem)
    return system.A_t[0], system.B_t[0]


def _draw_batches(
    spec: Any, backend: Backend, *, ctx: ComputeContext, n_batches: int
) -> tuple[EvaluationBatch, ...]:
    """`n_batches` sequential draws from `spec`'s own seeded stream (mirrors
    `EvaluationProtocol.build_batches`, generalized to any `BatchSpec`)."""
    sampler, _ = spec.build(
        backend, torch_dtype=ctx.torch_dtype, torch_device=ctx.torch_device
    )
    return tuple(sampler() for _ in range(n_batches))


def nominal_problem_and_batches(
    cfg: NB07Config, ctx: ComputeContext
) -> tuple[OptimalControlProblem, tuple[EvaluationBatch, ...]]:
    """The clean nominal problem and its fixed evaluation batches -- the
    price-of-robustness study's reference point, drawn under the dedicated,
    level-invariant `EVAL_SEED` (never a training seed)."""
    problem = LQRProblemFactory(
        state_dim=cfg.state_dim,
        control_dim=cfg.control_dim,
        horizon=cfg.horizon,
        seed=cfg.seed,
        u_max=cfg.u_max,
    ).build()
    eval_spec = GaussianBatchSpec(
        state_dim=cfg.state_dim,
        horizon=cfg.horizon,
        batch_size=cfg.batch_size,
        seed=EVAL_SEED,
        process_noise_std=cfg.process_noise_std,
        initial_state_std=cfg.initial_state_std,
    )
    batches = _draw_batches(
        eval_spec, Backend.TORCH, ctx=ctx, n_batches=cfg.n_eval_batches
    )
    return problem, batches


def _harness_for(batch_spec: Any, ctx: ComputeContext) -> Any:
    """Wrap `batch_spec` into an `EngineHarness` for one training run.

    A `NullExperimentTracker` is deliberate, not a placeholder: the sweep
    engine mints its OWN tracker per cache MISS
    (`TrackerBackedExperimentCache.open_run`, `workbench.robust_training_sweep
    ._train_and_score_point`) and writes exactly one meaningful payload per
    point there -- persisting every standard per-epoch callback's own
    metadata/checkpoints here as well would mean one near-empty run
    directory per (contender, level, seed), most of it redundant with the
    single cache payload. The per-epoch learning curve is still recovered
    (`MetricsHistoryCallback._history` is read directly off the live
    callback instance, never through the tracker)."""
    return EngineHarness(
        batch_spec=batch_spec, tracker=NullExperimentTracker(), ctx=ctx
    )


class ConditionBuilder(Protocol):
    """The closure each `<axis>_condition_builder` returns: one axis level
    plus one training seed to that point's `TrainingCondition`.

    Spelled as a `Protocol` rather than `Callable[[Any, int], ...]` because a
    bare `Callable` erases its parameters' names, so every
    `condition_at(level, seed=...)` call site -- which is how all of them
    read -- fails to type-check against it.
    """

    def __call__(self, level: Any, seed: int) -> TrainingCondition: ...


def noise_scale_condition_builder(
    cfg: NB07Config, ctx: ComputeContext
) -> ConditionBuilder:
    """Axis S (NB07 plan Sec 4.4): scale_multiplier applied jointly to
    process/initial-state noise; the plant itself is untouched."""
    problem = LQRProblemFactory(
        state_dim=cfg.state_dim,
        control_dim=cfg.control_dim,
        horizon=cfg.horizon,
        seed=cfg.seed,
        u_max=cfg.u_max,
    ).build()

    def condition_at(level: float, seed: int) -> TrainingCondition:
        train_spec = ExoticBatchSpec(
            state_dim=cfg.state_dim,
            horizon=cfg.horizon,
            batch_size=cfg.batch_size,
            seed=seed,
            process_noise_std=cfg.process_noise_std,
            initial_state_std=cfg.initial_state_std,
            family=NoiseFamily.GAUSSIAN,
            scale_multiplier=level,
        )
        eval_spec = ExoticBatchSpec(
            state_dim=cfg.state_dim,
            horizon=cfg.horizon,
            batch_size=cfg.batch_size,
            seed=EVAL_SEED,
            process_noise_std=cfg.process_noise_std,
            initial_state_std=cfg.initial_state_std,
            family=NoiseFamily.GAUSSIAN,
            scale_multiplier=level,
        )
        eval_batches = _draw_batches(
            eval_spec, Backend.TORCH, ctx=ctx, n_batches=cfg.n_eval_batches
        )
        return TrainingCondition(
            training_problem=problem,
            training_harness=_harness_for(train_spec, ctx),
            eval_problem=problem,
            eval_batches=eval_batches,
        )

    return condition_at


def noise_family_condition_builder(
    cfg: NB07Config, ctx: ComputeContext
) -> ConditionBuilder:
    """Axis X (NB07 plan Sec 4.4): the noise LAW changes; the plant does not."""
    problem = LQRProblemFactory(
        state_dim=cfg.state_dim,
        control_dim=cfg.control_dim,
        horizon=cfg.horizon,
        seed=cfg.seed,
        u_max=cfg.u_max,
    ).build()

    def condition_at(level: NoiseFamily, seed: int) -> TrainingCondition:
        train_spec = ExoticBatchSpec(
            state_dim=cfg.state_dim,
            horizon=cfg.horizon,
            batch_size=cfg.batch_size,
            seed=seed,
            process_noise_std=cfg.process_noise_std,
            initial_state_std=cfg.initial_state_std,
            family=level,
        )
        eval_spec = ExoticBatchSpec(
            state_dim=cfg.state_dim,
            horizon=cfg.horizon,
            batch_size=cfg.batch_size,
            seed=EVAL_SEED,
            process_noise_std=cfg.process_noise_std,
            initial_state_std=cfg.initial_state_std,
            family=level,
        )
        eval_batches = _draw_batches(
            eval_spec, Backend.TORCH, ctx=ctx, n_batches=cfg.n_eval_batches
        )
        return TrainingCondition(
            training_problem=problem,
            training_harness=_harness_for(train_spec, ctx),
            eval_problem=problem,
            eval_batches=eval_batches,
        )

    return condition_at


def _plant_condition_builder(
    cfg: NB07Config,
    ctx: ComputeContext,
    *,
    kind: PerturbationKind,
    resample_per_epoch: bool,
) -> ConditionBuilder:
    """Shared machinery for axes D-add/D-rot: the SAME nominal problem
    object is trained on (mutated in place by `DomainRandomizationCallback`,
    restored after); the evaluation instance is a SEPARATELY constructed,
    fixed perturbed problem (never the live, post-restoration training
    system -- see `workbench.robust_training_sweep.TrainingCondition`'s
    `eval_problem` docstring)."""
    problem = LQRProblemFactory(
        state_dim=cfg.state_dim,
        control_dim=cfg.control_dim,
        horizon=cfg.horizon,
        seed=cfg.seed,
        u_max=cfg.u_max,
    ).build()
    nominal_A, nominal_B = _nominal_system_matrices(cfg)
    system, cost = require_linear_quadratic(problem)
    W = (cfg.process_noise_std**2) * np.eye(cfg.state_dim)

    def condition_at(level: float, seed: int) -> TrainingCondition:
        perturbation = PlantPerturbation(kind=kind, level=level, seed=seed)
        distribution = PerturbationDistribution(
            perturbation=perturbation, resample_per_epoch=resample_per_epoch
        )
        eval_A, eval_B, _, _ = perturbation.draw(
            nominal_A, nominal_B, W, np.random.default_rng(seed)
        )
        eval_problem = OptimalControlProblem(
            system=LinearSystem.fully_observable(eval_A, eval_B),
            cost=cost,
            constraints=problem.constraints,
        )
        eval_spec = GaussianBatchSpec(
            state_dim=cfg.state_dim,
            horizon=cfg.horizon,
            batch_size=cfg.batch_size,
            seed=EVAL_SEED,
            process_noise_std=cfg.process_noise_std,
            initial_state_std=cfg.initial_state_std,
        )
        eval_batches = _draw_batches(
            eval_spec, Backend.TORCH, ctx=ctx, n_batches=cfg.n_eval_batches
        )
        train_spec = GaussianBatchSpec(
            state_dim=cfg.state_dim,
            horizon=cfg.horizon,
            batch_size=cfg.batch_size,
            seed=seed,
            process_noise_std=cfg.process_noise_std,
            initial_state_std=cfg.initial_state_std,
        )
        return TrainingCondition(
            training_problem=problem,
            training_harness=_harness_for(train_spec, ctx),
            eval_problem=eval_problem,
            eval_batches=eval_batches,
            dr_distribution=distribution,
        )

    return condition_at


def plant_additive_condition_builder(
    cfg: NB07Config, ctx: ComputeContext
) -> ConditionBuilder:
    """Axis D-add: fresh additive draw every epoch (the DR regime, NB07 plan
    Sec 4.4)."""
    return _plant_condition_builder(
        cfg, ctx, kind=PerturbationKind.ADDITIVE, resample_per_epoch=True
    )


def plant_rotation_condition_builder(
    cfg: NB07Config, ctx: ComputeContext
) -> ConditionBuilder:
    """Axis D-rot: one fixed rotated plant for the whole training run (NB07
    plan Sec 4.4) -- rotation is deterministic given ``(level, seed)``, so
    there is nothing left to resample epoch to epoch."""
    return _plant_condition_builder(
        cfg, ctx, kind=PerturbationKind.ROTATION, resample_per_epoch=False
    )


def train_horizon_condition_builder(
    cfg: NB07Config, ctx: ComputeContext
) -> ConditionBuilder:
    """Axis H: trains AT the perturbed horizon (no rehost -- that is NB06's
    zero-shot problem, not this one).

    `nominal_override` is set to this SAME (problem, eval_batches) pair: a
    contender's gain/refinement stacks have exactly `level` entries, so
    scoring it against the sweep's GLOBAL nominal problem (fixed at
    `cfg.horizon`, a different length whenever `level != cfg.horizon`)
    indexes past the end of those stacks (`workbench.robust_training_sweep
    .TrainingCondition.nominal_override`'s own docstring has the full
    story). `problem`/`eval_batches` are already built at nominal noise and
    plant, at `level`'s own horizon, so they already ARE this axis's correct
    "nominal, at the horizon actually trained for" reference -- nothing
    further to draw.
    """

    def condition_at(level: int, seed: int) -> TrainingCondition:
        problem = LQRProblemFactory(
            state_dim=cfg.state_dim,
            control_dim=cfg.control_dim,
            horizon=level,
            seed=cfg.seed,
            u_max=cfg.u_max,
        ).build()
        train_spec = GaussianBatchSpec(
            state_dim=cfg.state_dim,
            horizon=level,
            batch_size=cfg.batch_size,
            seed=seed,
            process_noise_std=cfg.process_noise_std,
            initial_state_std=cfg.initial_state_std,
        )
        eval_spec = GaussianBatchSpec(
            state_dim=cfg.state_dim,
            horizon=level,
            batch_size=cfg.batch_size,
            seed=EVAL_SEED,
            process_noise_std=cfg.process_noise_std,
            initial_state_std=cfg.initial_state_std,
        )
        eval_batches = _draw_batches(
            eval_spec, Backend.TORCH, ctx=ctx, n_batches=cfg.n_eval_batches
        )
        return TrainingCondition(
            training_problem=problem,
            training_harness=_harness_for(train_spec, ctx),
            eval_problem=problem,
            eval_batches=eval_batches,
            nominal_override=(problem, eval_batches),
        )

    return condition_at


def sample_size_condition_builder(
    cfg: NB07Config, ctx: ComputeContext
) -> ConditionBuilder:
    """The sample-complexity axis (NB07 plan Sec 4.3): a FINITE, replayed
    training set of size `level`; the test set is DISJOINT and FIXED across
    every rung of the ladder (the SAME `TEST_SEED` stream, never `level`- or
    `seed`-dependent), so every point's generalization gap is measured
    against one common yardstick."""
    problem = LQRProblemFactory(
        state_dim=cfg.state_dim,
        control_dim=cfg.control_dim,
        horizon=cfg.horizon,
        seed=cfg.seed,
        u_max=cfg.u_max,
    ).build()

    def condition_at(level: int, seed: int) -> TrainingCondition:
        source = ExoticBatchSpec(
            state_dim=cfg.state_dim,
            horizon=cfg.horizon,
            batch_size=cfg.batch_size,
            seed=seed,
            process_noise_std=cfg.process_noise_std,
            initial_state_std=cfg.initial_state_std,
            family=NoiseFamily.GAUSSIAN,
        )
        train_spec = FiniteTrajectoryDatasetSpec(
            source=source, n_trajectories=level, shuffle_seed=seed
        )
        eval_spec = GaussianBatchSpec(
            state_dim=cfg.state_dim,
            horizon=cfg.horizon,
            batch_size=cfg.batch_size,
            seed=TEST_SEED,
            process_noise_std=cfg.process_noise_std,
            initial_state_std=cfg.initial_state_std,
        )
        eval_batches = _draw_batches(
            eval_spec, Backend.TORCH, ctx=ctx, n_batches=cfg.n_eval_batches
        )
        return TrainingCondition(
            training_problem=problem,
            training_harness=_harness_for(train_spec, ctx),
            eval_problem=problem,
            eval_batches=eval_batches,
        )

    return condition_at


#: Axis -> its condition-builder factory, and the levels field it sweeps
#: (NB07Config attribute name) -- the notebook's single dispatch table.
AXIS_CONDITION_BUILDERS: Mapping[
    UncertaintyAxis,
    Callable[[NB07Config, ComputeContext], ConditionBuilder],
] = {
    UncertaintyAxis.NOISE_SCALE: noise_scale_condition_builder,
    UncertaintyAxis.NOISE_FAMILY: noise_family_condition_builder,
    UncertaintyAxis.PLANT_ADDITIVE: plant_additive_condition_builder,
    UncertaintyAxis.PLANT_ROTATION: plant_rotation_condition_builder,
    UncertaintyAxis.TRAIN_HORIZON: train_horizon_condition_builder,
    UncertaintyAxis.SAMPLE_SIZE: sample_size_condition_builder,
}

#: Axis -> the NB07Config field carrying its levels.
AXIS_LEVELS_FIELD: Mapping[UncertaintyAxis, str] = {
    UncertaintyAxis.NOISE_SCALE: "noise_scale_multipliers",
    UncertaintyAxis.NOISE_FAMILY: "noise_families",
    UncertaintyAxis.PLANT_ADDITIVE: "plant_additive_epsilons",
    UncertaintyAxis.PLANT_ROTATION: "plant_rotation_degrees",
    UncertaintyAxis.TRAIN_HORIZON: "train_horizons",
    UncertaintyAxis.SAMPLE_SIZE: "sample_sizes",
}


def recipes_at_for(
    cfg: NB07Config, axis: UncertaintyAxis
) -> Callable[[Any], dict[str, ModelRecipe]]:
    """label -> `ModelRecipe` for one `level` of `axis` -- level-invariant
    for every axis except `TRAIN_HORIZON` (whose recipes must be rebuilt
    with ``horizon=level``, a specification-time field)."""
    if axis is UncertaintyAxis.TRAIN_HORIZON:
        return lambda level: six_training_sweep_contenders(cfg, horizon=level)
    fixed = six_training_sweep_contenders(cfg)
    return lambda level: fixed
