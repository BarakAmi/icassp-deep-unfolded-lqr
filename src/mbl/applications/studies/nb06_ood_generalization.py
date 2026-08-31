"""NB06 -- Zero-Shot Out-of-Distribution Generalization: the Tier-3
`Experiment` declaration (docs/planning/03_studies/nb06_ood_generalization/zero_shot_ood_generalization_benchmark.md Sec
3/4). Six contenders, trained ONCE on one shared, box-constrained stochastic
LQR instance, common-noise evaluated -- the exact set NB06's plan names:

    Truncated-Riccati  -- the unconstrained Riccati optimum, saturated onto
                          the box at every step; non-learnable (family
                          "truncated_riccati"). K-independent baseline.
    Unfolded-alpha     -- learned per-iteration step size, true Riccati P,
                          projected each inner step (family "unfolded").
    Unfolded-alpha+P   -- learned per-iteration step size AND a learned
                          time-invariant matrix replacing Riccati P, trained
                          layer-wise (family "unfolded_warmstart") -- the
                          flagship contender under a box (mirrors NB04/05's
                          own framing).
    COCP               -- a per-step differentiable convex QP over a learned
                          cost-to-go (family "cocp"). Depth-invariant.
    GRU                -- a recurrent neural policy (family "neural"); the
                          hidden dimension is chosen once via a brief offline
                          ablation (NB06 plan Sec 4 Sec 6) and then becomes a
                          fixed Control-Panel value here, like every other
                          per-model hyperparameter.
    COCP-LB            -- the SAME QP policy as COCP, frozen at the box-aware
                          SDP's own cost-to-go (family "cocp_lower_bound").
                          Depth-invariant, attained UPPER bound on J*_C.

Unlike NB04, this module has NO ``standard_pgd``/``unfolded_fixed``
contender: NB06's own plan (Sec 4/7) names exactly these six, and the
zero-shot OOD question is about ARCHITECTURES that embed vs. learn
optimality structure, not about a fixed-step iterative baseline.

Mirrors `nb04_box_constrained.py`'s composition-only construction: turning on
the box constraint is the same one-line `LQRProblemFactory(...,
u_max=cfg.u_max)`. `workbench.ood_sweep` reuses
`nb04_box_constrained.compute_box_constrained_floors` VERBATIM for every
perturbation point's floor -- it is already a pure function of ANY problem's
own matrices plus `process_noise_std`/`u_max`, so this module need not
import or re-derive it at all.

`UnfoldedModelConfig` is reused VERBATIM from `.nb03_unfolding` (imported,
never redefined), identical to NB04's own reuse -- it carries no NB03-
specific coupling.
"""

from dataclasses import dataclass, field

import torch

from .nb03_unfolding import UnfoldedModelConfig
from ..factories import GaussianBatchSpec, LQRProblemFactory
from ..recipes import UnfoldedKind
from ...core.runtime import Backend, ComputeContext, Precision
from ...engine.training_plan import OptimizerSpec, TrainingPlan
from ...experiments import ContenderSpec, EvaluationProtocol, Experiment
from ...models.neural.nerual import SequenceModelType

#: NB06's experiment name (the run/registry identity every contender's run
#: directory is prefixed with).
EXPERIMENT_NAME = "NB06_ZeroShotOODGeneralization"

#: The six contender labels this module declares -- the exact set
#: `workbench.ood_sweep`'s rehost dispatch and figure legends key against.
CONTENDER_LABELS = (
    "truncated_riccati",
    "unfolded_alpha",
    "unfolded_alpha_p",
    "cocp",
    "neural",
    "cocp_lower_bound",
)


def _default_unfolded_alpha() -> UnfoldedModelConfig:
    """Unfolded-alpha's default: learned step size, cold start, end to end
    -- identical defaults to NB03/NB04's contender of the same name."""
    return UnfoldedModelConfig(kind=UnfoldedKind.LEARNED_STEP_SIZE)


def _default_unfolded_alpha_p() -> UnfoldedModelConfig:
    """Unfolded-alpha+P's default: learned step size AND matrix, trained
    layer-wise -- identical defaults to NB03/NB04's flagship contender."""
    return UnfoldedModelConfig(
        kind=UnfoldedKind.LEARNED_STEP_SIZE_AND_MATRIX, training_mode="layerwise"
    )


def _default_cocp_plan() -> TrainingPlan:
    """COCP's standalone default `TrainingPlan`; only a fallback for callers
    that never override `NB06Config.cocp_plan` -- the notebook always builds
    its own from the shared `OPTIMIZER_SPEC`/`N_EPOCHS` (strict parity, as
    NB04's COCP refinement plan established)."""
    return TrainingPlan(optimizer=OptimizerSpec("adam", 0.1), epochs=100)


def _default_neural_plan() -> TrainingPlan:
    """GRU's standalone default `TrainingPlan`; only a fallback -- the
    notebook builds its own after the hidden-dimension ablation (NB06 plan
    Sec 4 Sec 6) selects `NB06Config.neural_hidden_dim`."""
    return TrainingPlan(optimizer=OptimizerSpec("adam", 1e-3), epochs=200)


@dataclass(frozen=True)
class NB06Config:
    """NB06's frozen settings: the shared box-constrained problem/evaluation
    scale, at a SINGLE fixed unfolding depth (the OOD sweep varies
    perturbation level, not depth -- the separate over-thinking ablation,
    NB06 plan Sec 3.6, sweeps depth on its own smaller nominal-training
    ladder), plus each trainable family's own training specification.

    Attributes:
        state_dim: State dimension ``n``.
        control_dim: Control dimension ``m``.
        horizon: The finite (nominal) horizon ``N``.
        u_max: The infinity-norm control bound every contender is projected
            onto. Must genuinely bind on this instance (verify with
            `workbench.analysis.compute_box_binding_fraction`).
        num_unfolding_iterations: Unrolled depth ``K``, shared by both
            unfolded contenders, fixed for the main experiment.
        step_size_init: Initial per-iteration step size.
        step_size_max: Step-size reparameterization upper bound.
        batch_size: Trajectories per evaluation batch.
        n_eval_batches: Evaluation batches drawn and averaged.
        seed: Problem/batch seed.
        process_noise_std: Nominal process noise standard deviation.
        initial_state_std: Nominal initial-state standard deviation.
        unfolded_alpha: Unfolded-alpha's per-model config, reused verbatim
            from `nb03_unfolding.UnfoldedModelConfig`.
        unfolded_alpha_p: Unfolded-alpha+P's per-model config.
        cocp_plan: COCP's `TrainingPlan`.
        cocp_solver_eps: Convergence tolerance for COCP's resolved QP solver.
        cocp_solver_max_iters: Iteration budget for COCP's resolved QP solver.
        cocp_batch_size: Optional per-run override of COCP's training/
            evaluation batch size; ``None`` means strict parity.
        cocp_horizon: The same escape hatch for `horizon`; ``None`` means
            strict parity.
        neural_hidden_dim: The GRU's hidden dimension, chosen once via the
            offline ablation (Sec 4 Sec 6) and then fixed here.
        neural_plan: The GRU's `TrainingPlan`.
        neural_init_seed: Seed of the GRU's initial weights
            (`applications.recipes.neural.NeuralRecipe.init_seed`) --
            reproducibility for the multi-seed OOD sweep.
        random_init_std: Std of the Gaussian control-init draw, shared by
            every contender whose ``init_method`` is
            `ControlInitMethod.RANDOMIZED`.
        log_first_epochs: ``a`` of the dynamic training-log narration
            schedule.
        log_every_epochs: ``b`` of the dynamic schedule.
        dtype: The working torch precision.
        device: The torch device every contender's `ComputeContext`
            resolves to.
    """

    state_dim: int = 7
    control_dim: int = 3
    horizon: int = 50
    u_max: float = 0.1
    num_unfolding_iterations: int = 5
    step_size_init: float = 0.05
    step_size_max: float = 1.0
    batch_size: int = 256
    n_eval_batches: int = 8
    seed: int = 0
    process_noise_std: float = 0.5
    initial_state_std: float = 1.0
    unfolded_alpha: UnfoldedModelConfig = field(default_factory=_default_unfolded_alpha)
    unfolded_alpha_p: UnfoldedModelConfig = field(
        default_factory=_default_unfolded_alpha_p
    )
    cocp_plan: TrainingPlan = field(default_factory=_default_cocp_plan)
    cocp_solver_eps: float = 1e-8
    cocp_solver_max_iters: int = 10000
    cocp_batch_size: int | None = None
    cocp_horizon: int | None = None
    neural_hidden_dim: int = 64
    neural_plan: TrainingPlan = field(default_factory=_default_neural_plan)
    neural_init_seed: int = 0
    random_init_std: float = 1.0
    log_first_epochs: int = 10
    log_every_epochs: int = 20
    dtype: torch.dtype = torch.float64
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.u_max <= 0:
            raise ValueError(f"u_max must be strictly positive, got {self.u_max}.")


def build_nb06_config(  # noqa: PLR0913 -- a deliberate keyword-only FLAT
    # mirror of the `NB06Config` carrier (@dataclass is itself exempt from the
    # arg-count gate), matching `nb03_unfolding.build_nb03_config`/
    # `nb04_box_constrained.build_nb04_config`'s own directive: the notebook
    # assembles the carrier from flat, individually-typed, discoverable
    # top-level settings rather than authoring an opaque `NB06Config(...)`
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
    cocp_plan: TrainingPlan,
    neural_hidden_dim: int,
    neural_plan: TrainingPlan,
    initial_state_std: float = 1.0,
    cocp_solver_eps: float = 1e-8,
    cocp_solver_max_iters: int = 10000,
    cocp_batch_size: int | None = None,
    cocp_horizon: int | None = None,
    neural_init_seed: int = 0,
    random_init_std: float = 1.0,
    log_first_epochs: int = 10,
    log_every_epochs: int = 20,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
) -> NB06Config:
    """Assemble the internal `NB06Config` carrier from flat, standalone
    settings (see `NB06Config` for every field's meaning).

    Returns:
        The assembled `NB06Config`.
    """
    return NB06Config(
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
        unfolded_alpha=unfolded_alpha,
        unfolded_alpha_p=unfolded_alpha_p,
        cocp_plan=cocp_plan,
        cocp_solver_eps=cocp_solver_eps,
        cocp_solver_max_iters=cocp_solver_max_iters,
        cocp_batch_size=cocp_batch_size,
        cocp_horizon=cocp_horizon,
        neural_hidden_dim=neural_hidden_dim,
        neural_plan=neural_plan,
        neural_init_seed=neural_init_seed,
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
    """One learned unfolded contender's `ContenderSpec` -- the same per-model
    routing logic as `nb03_unfolding._learned_unfolded_contender`/
    `nb04_box_constrained._learned_unfolded_contender` (kept as NB06's own
    copy: each study module is its own standalone declaration per this
    package's docstring, never a cross-module import of a private symbol).

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
        ``model.training_mode``.
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


def nb06_ood_generalization_experiment(cfg: NB06Config) -> Experiment:
    """Compose the NB06 declaration from `cfg`.

    Args:
        cfg: The frozen NB06 settings.

    Returns:
        The `Experiment` (box-constrained problem factory + six contenders +
        shared common-noise evaluation), ready for ``run_experiment`` /
        `experiments.zero_shot.synthesize_nominal_contenders`.
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
            u_max=cfg.u_max,
        ),
        contenders=(
            ContenderSpec(
                family="truncated_riccati",
                config={"horizon": cfg.horizon},
                label="truncated_riccati",
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
            ContenderSpec(
                family="cocp",
                config={
                    "plan": cfg.cocp_plan,
                    "solver_eps": cfg.cocp_solver_eps,
                    "solver_max_iters": cfg.cocp_solver_max_iters,
                    "batch_size": cfg.cocp_batch_size,
                    "horizon": cfg.cocp_horizon,
                    **log_kwargs,
                },
                label="cocp",
            ),
            ContenderSpec(
                family="neural",
                config={
                    "hidden_dim": cfg.neural_hidden_dim,
                    "plan": cfg.neural_plan,
                    "model_type": SequenceModelType.GRU,
                    "init_seed": cfg.neural_init_seed,
                },
                label="neural",
            ),
            ContenderSpec(
                family="cocp_lower_bound",
                config={
                    "process_noise_std": cfg.process_noise_std,
                    "solver_eps": cfg.cocp_solver_eps,
                    "solver_max_iters": cfg.cocp_solver_max_iters,
                    "batch_size": cfg.cocp_batch_size,
                },
                label="cocp_lower_bound",
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


__all__ = [
    "EXPERIMENT_NAME",
    "CONTENDER_LABELS",
    "NB06Config",
    "build_nb06_config",
    "nb06_ood_generalization_experiment",
]
