"""NB04 -- Box-Constrained LQR Benchmark: the Tier-3 `Experiment` declaration
(docs/planning/03_studies/nb04_box_constrained/box_constrained_lqr_benchmark.md, extended by
docs/planning/03_studies/nb04_box_constrained/cocp_integration_and_convergence_visualization.md and
docs/planning/03_studies/nb04_box_constrained/reference_bounds_and_figure_refinement.md). Six
contenders on one shared, box-constrained (``|u_t| <= u_max``) stochastic
LQR instance, common-noise evaluated:

    Truncated-Riccati  -- the unconstrained Riccati optimum, saturated onto
                          the box at every step; non-learnable (family
                          "truncated_riccati").
    Standard-PGD       -- fixed step-size unrolled, PROJECTED gradient
                          descent, true Riccati P, nothing learned (family
                          "unfolded_fixed") -- NB03's "Standard-GD" under a
                          box: every inner iterate is now clamped.
    Unfolded-alpha     -- learned per-iteration step size, true Riccati P,
                          projected each inner step (family "unfolded" or
                          "unfolded_warmstart", per its own training_mode)
                          -- tied to the UNCONSTRAINED curvature P*, so
                          expected to be the weaker learned contender here.
    Unfolded-alpha+P   -- learned per-iteration step size AND a learned
                          time-invariant matrix replacing Riccati P,
                          projected each inner step -- the FLAGSHIP: under an
                          active box the DARE gain is no longer optimal, so
                          only this contender can reshape its cost-to-go for
                          the constraint (RESEARCH_PLAN.md Sec 3/5).
    COCP               -- a *fundamentally different kind* of learned policy
                          (family "cocp"): a per-step differentiable convex
                          QP over a learned cost-to-go, not a projected-
                          gradient unrolling. Depth-invariant (no unfolding
                          depth to sweep), so it trains ONCE and is treated
                          as a K-independent baseline (`NB04_BASELINE_LABELS`
                          in `workbench.depth_sweep`) exactly like Truncated-
                          Riccati. The practical, best-*attained* optimality
                          reference this notebook now benchmarks the unfolded
                          families against -- not a proven optimum (that
                          remains J_SDP's job); see the COCP
                          refinement plan Sec 1.3 for why this framing
                          matters. Reverses this module's earlier "COCP is
                          out of scope" decision.
    COCP-LB            -- the SAME QP policy as COCP, but FROZEN (never
                          trained) at the box-aware SDP's own cost-to-go
                          matrix (family "cocp_lower_bound"; recipe
                          `applications.recipes.cocp.COCPLowerBoundRecipe`).
                          Depth-invariant, like COCP and Truncated-Riccati.
                          Its attained Monte-Carlo cost is an UPPER bound on
                          the true constrained optimum J*_C -- it is a real,
                          feasible policy, not a relaxation -- so it tightens
                          the notebook's bracket from the other side of
                          J_SDP: ``J_LQR <= J_SDP <= J*_C <= J_COCP-LB``
                          (reference-bounds plan Sec 1.3). Un-defers this
                          module's own earlier "remains deferred" note.

Turning the constraint on is exactly ONE line relative to an unconstrained
declaration: `LQRProblemFactory(..., u_max=cfg.u_max)`. Every unfolded family
already auto-projects through `apply_constraints`
(`applications.recipes.unfolded.build_unfolded_controller` reads
``problem.constraints``); `TruncatedRiccatiRecipe` already reads
``problem.constraints[0]`` to saturate its gain; COCP's box is a hard
constraint of its own QP (`applications.recipes.cocp.COCPRecipe`). No
controller code changes for NB04 at all -- this module is composition, not
construction.

`UnfoldedModelConfig` is REUSED VERBATIM from `.nb03_unfolding` (imported,
never redefined): it carries no NB03-specific coupling -- one learned
model's own ``(kind, init_method, training_mode, plan, schedule)`` is a free
per-model choice regardless of which study builds it. Per the user's explicit
directive, NB04 hardcodes NO training regime: both learned models expose
`training_mode` ("end_to_end" vs "layerwise") as an independent Control-Panel
choice, exactly as NB03 does. This keeps `nb03_unfolding.py` byte-unchanged
(the S3 law -- new declarations are new modules, never edits to an existing
one) while never duplicating that dataclass.

The two SDP lower bounds (`compute_box_constrained_floors`) are deliberately
NOT `ContenderSpec`s, for the same reason NB03's closed-form Theo-Riccati
isn't one: `EvaluationProtocol` only scores Monte-Carlo rollouts, and both
floors are closed-form/convex-optimization numbers computed directly from the
experiment's own problem matrices. They bracket the truth:
``J_LQR <= J_SDP <= J*_C <= every contender's evaluated cost``
(NB04 plan Sec 2.2, RESEARCH_PLAN.md Sec 5.1). COCP is scored against this
same J_SDP floor like every other contender -- it is a strong
attained reference, not a substitute for the floor itself. Its frozen
sibling, COCP-LB (`cocp_lower_bound`, the same QP fixed at the SDP's own
cost-to-go), is now wired in as a sixth contender (reference-bounds plan
Sec 1.4) -- despite its family name, it is an attained UPPER bound on J*_C,
not a second lower bound; see the module-docstring entry above and the
reference-bounds plan Sec 1.3 for the full bracket argument.
"""

from dataclasses import dataclass, field
from typing import cast

import numpy as np
import torch
from scipy.linalg import solve_discrete_are

from .nb03_unfolding import UnfoldedModelConfig
from ..factories import GaussianBatchSpec, LQRProblemFactory
from ..recipes import UnfoldedKind
from ...core.kernels import time_invariant_slice
from ...core.optimal_control_problem import OptimalControlProblem
from ...core.runtime import Backend, ComputeContext, Precision
from ...core.utils import ensure_positive_integer
from ...engine.training_plan import OptimizerSpec, TrainingPlan
from ...experiments import ContenderSpec, EvaluationProtocol, Experiment
from ...models.constrained.lower_bound import solve_box_constrained_lower_bound
from ...models.guards import require_linear_quadratic
from ...models.iterative import ControlInitMethod

#: NB04's experiment name (the run/registry identity every contender's run
#: directory is prefixed with).
EXPERIMENT_NAME = "NB04_BoxConstrainedLQR"


def _default_unfolded_alpha() -> UnfoldedModelConfig:
    """Unfolded-alpha's default: learned step size, cold start, end to end
    -- identical defaults to NB03's contender of the same name (the training
    regime stays a free per-model Control-Panel choice; see module
    docstring)."""
    return UnfoldedModelConfig(kind=UnfoldedKind.LEARNED_STEP_SIZE)


def _default_unfolded_alpha_p() -> UnfoldedModelConfig:
    """Unfolded-alpha+P's default: learned step size AND matrix, cold start,
    trained layer-wise -- identical defaults to NB03's flagship contender."""
    return UnfoldedModelConfig(
        kind=UnfoldedKind.LEARNED_STEP_SIZE_AND_MATRIX, training_mode="layerwise"
    )


def _default_cocp_plan() -> TrainingPlan:
    """COCP's standalone default `TrainingPlan` (COCP refinement plan Sec
    3.2) -- lr=0.1 matches `box_constraint_lqr`'s own `cocp_learning_rate`
    default and was verified stable (no divergence, healthy gradient norms)
    at NB04's own problem scale; 100 epochs matches NB04's own `N_EPOCHS`
    Control-Panel convention. Only a fallback for callers that never
    override `NB04Config.cocp_plan` -- the notebook always builds its own
    from the shared `OPTIMIZER_SPEC`/`N_EPOCHS` (strict parity, Sec 1.4)."""
    return TrainingPlan(optimizer=OptimizerSpec("adam", 0.1), epochs=100)


@dataclass(frozen=True)
class NB04Config:
    """NB04's frozen settings: the shared box-constrained problem/evaluation
    scale plus each trainable family's own training specification. Mirrors
    `nb03_unfolding.NB03Config` field-for-field, with two changes: `u_max`
    is added (the ONE new knob that turns on the whole constraint pathway,
    see `nb04_box_constrained_experiment`), and `standard_gd_init_method` is
    renamed to `standard_pgd_init_method` (the fixed-step contender is now a
    PROJECTED gradient descent under the box, not plain GD).

    Attributes:
        state_dim: State dimension ``n``.
        control_dim: Control dimension ``m``.
        horizon: The finite horizon ``N``.
        u_max: The infinity-norm control bound ``|u_t| <= u_max`` every
            contender is projected onto (Truncated-Riccati by saturation,
            every unfolded family by its inner-loop projection). Must be
            chosen so the constraint genuinely BINDS on this problem
            instance -- verify with
            `workbench.analysis.compute_box_binding_fraction` before
            trusting a run (NB04 plan Sec 6/9).
        num_unfolding_iterations: Unrolled depth ``J``, shared by every
            unfolded contender.
        step_size_init: Initial per-iteration step size.
        step_size_max: Step-size reparameterization upper bound.
        batch_size: Trajectories per evaluation batch.
        n_eval_batches: Evaluation batches drawn and averaged (``> 1``
            additionally reports ``eval_expected_cost_std``).
        seed: Problem/batch seed.
        process_noise_std: Process noise standard deviation.
        initial_state_std: Initial-state standard deviation of the shared
            evaluation/training batch.
        standard_pgd_init_method: How Standard-PGD (the fixed-step,
            projected iterative contender) seeds ``u^(0)``
            (`ControlInitMethod`).
        unfolded_alpha: Unfolded-alpha's per-model config -- learned step
            size, with its own init method and training regime
            (`UnfoldedModelConfig`, reused verbatim from `nb03_unfolding`).
        unfolded_alpha_p: Unfolded-alpha+P's per-model config -- learned
            step size AND matrix, with its own init method and training
            regime.
        cocp_plan: COCP's `TrainingPlan` -- strict parity (COCP refinement
            plan Sec 1.4) means the notebook builds this from the SAME
            shared `OptimizerSpec`/epoch count every other learned
            contender uses, not a COCP-specific one.
        cocp_solver_eps: Convergence tolerance for COCP's resolved QP
            solver (translated into that solver's own vocabulary by
            `COCPSolverSpec.to_solver_args`).
        cocp_solver_max_iters: Iteration budget for COCP's resolved QP
            solver, translated the same way.
        cocp_batch_size: Optional per-run override of COCP's training/
            evaluation batch size. ``None`` (the default) means strict
            parity with `batch_size` -- the loud, explicit escape hatch
            (COCP refinement plan Sec 1.4) for when strict parity proves
            too slow; any non-``None`` value trades that parity away and
            must be treated as a documented deviation, never a silent one.
        cocp_horizon: The same escape hatch for `horizon`; ``None`` means
            strict parity. A non-``None`` value also breaks Section 8's
            trajectory-baseline switch (COCP refinement plan Sec 3.5): a
            different horizon gives COCP a different-shaped persisted
            trajectory than every unfolded contender's own.
        random_init_std: Std of the Gaussian control-init draw, shared by
            every contender whose ``init_method`` is
            `ControlInitMethod.RANDOMIZED`.
        log_first_epochs: ``a`` of the dynamic training-log narration
            schedule: the leading epochs narrated consecutively before the
            coarser cadence begins.
        log_every_epochs: ``b`` of the dynamic schedule: the narration
            cadence after the leading block.
        dtype: The working torch precision.
        device: The torch device every contender's `ComputeContext`
            resolves to.
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
    cocp_plan: TrainingPlan = field(default_factory=_default_cocp_plan)
    cocp_solver_eps: float = 1e-8
    cocp_solver_max_iters: int = 10000
    cocp_batch_size: int | None = None
    cocp_horizon: int | None = None
    random_init_std: float = 1.0
    log_first_epochs: int = 10
    log_every_epochs: int = 20
    dtype: torch.dtype = torch.float64
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.u_max <= 0:
            raise ValueError(f"u_max must be strictly positive, got {self.u_max}.")
        ensure_positive_integer(self.cocp_solver_max_iters, "cocp_solver_max_iters")
        if self.cocp_batch_size is not None:
            ensure_positive_integer(self.cocp_batch_size, "cocp_batch_size")
        if self.cocp_horizon is not None:
            ensure_positive_integer(self.cocp_horizon, "cocp_horizon")


def build_nb04_config(  # noqa: PLR0913 -- a deliberate keyword-only FLAT
    # mirror of the `NB04Config` carrier (@dataclass is itself exempt from the
    # arg-count gate), matching `nb03_unfolding.build_nb03_config`'s own
    # directive: the notebook assembles the carrier from flat, individually-
    # typed, discoverable top-level settings rather than authoring an opaque
    # `NB04Config(...)` literal. The two learned models' OWN settings stay
    # grouped into one `UnfoldedModelConfig` literal each (the same exemption
    # NB03 applies).
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
    initial_state_std: float = 1.0,
    standard_pgd_init_method: ControlInitMethod = ControlInitMethod.COLD,
    cocp_solver_eps: float = 1e-8,
    cocp_solver_max_iters: int = 10000,
    cocp_batch_size: int | None = None,
    cocp_horizon: int | None = None,
    random_init_std: float = 1.0,
    log_first_epochs: int = 10,
    log_every_epochs: int = 20,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
) -> NB04Config:
    """Assemble the internal `NB04Config` carrier from flat, standalone
    settings: the notebook exposes every knob -- including `u_max`, a
    top-level tunable Control-Panel variable like every other parameter -- and
    calls this to build the carrier the factory/sweep pass around.

    Args:
        See `NB04Config` for every field's meaning; this is a flat, keyword-
        only mirror of that carrier's constructor.

    Returns:
        The assembled `NB04Config`.
    """
    return NB04Config(
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
        cocp_plan=cocp_plan,
        cocp_solver_eps=cocp_solver_eps,
        cocp_solver_max_iters=cocp_solver_max_iters,
        cocp_batch_size=cocp_batch_size,
        cocp_horizon=cocp_horizon,
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
    routing logic as `nb03_unfolding._learned_unfolded_contender` (kept as
    NB04's own copy, ~15 lines of orchestration glue with no shared
    mathematics, rather than a cross-module import of a private symbol: each
    study module is its own standalone declaration per this package's
    docstring).

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


def nb04_box_constrained_experiment(cfg: NB04Config) -> Experiment:
    """Compose the NB04 declaration from `cfg`.

    Args:
        cfg: The frozen NB04 settings.

    Returns:
        The `Experiment` (box-constrained problem factory + five contenders
        + shared common-noise evaluation), ready for ``run_experiment``.
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
            ContenderSpec(
                family="unfolded_fixed",
                config={
                    **shape,
                    "init_method": cfg.standard_pgd_init_method,
                    "random_init_std": cfg.random_init_std,
                    "random_init_seed": cfg.seed,
                },
                label="standard_pgd",
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
                    # None/None is strict training parity (COCP refinement
                    # plan Sec 1.4) -- COCP trains at the SAME batch_size/
                    # horizon as every other contender unless explicitly
                    # overridden, which Sec 3.5's trajectory-baseline switch
                    # requires.
                    "batch_size": cfg.cocp_batch_size,
                    "horizon": cfg.cocp_horizon,
                    # The SAME dynamic narration cadence every learned
                    # contender gets -- previously omitted here, leaving
                    # COCP's own (often much longer) training run with no
                    # per-epoch progress visibility at all.
                    **log_kwargs,
                },
                label="cocp",
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


@dataclass(frozen=True)
class BoxConstrainedFloors:
    """The two SDP lower bounds bracketing a box-constrained LQR problem's
    achievable infinite-horizon average cost (NB04 plan Sec 2.2,
    RESEARCH_PLAN.md Sec 5.1):
    ``J_LQR <= J_SDP <= J*_C <= every contender's evaluated cost``.

    Attributes:
        j_lqr: The loose, UNCONSTRAINED LQR floor -- ``tr(P_dare @ W)``, the
            infinite-horizon steady-state cost the DARE solution attains
            when the box is dropped entirely. Provably a valid lower bound:
            dropping a constraint can only lower the optimal cost.
        j_sdp: The tighter, box-aware SDP floor (Boyd-style convex
            relaxation of the box-constrained problem;
            `models.constrained.lower_bound.solve_box_constrained_lower_bound`).
        p_lqr: The DARE solution achieving `j_lqr`, shape ``(n, n)``.
        p_sdp: The SDP-optimal matrix achieving `j_sdp`, shape ``(n, n)``.
    """

    j_lqr: float
    j_sdp: float
    p_lqr: np.ndarray
    p_sdp: np.ndarray


def compute_box_constrained_floors(
    problem: OptimalControlProblem,
    *,
    process_noise_std: float,
    u_max: float,
) -> BoxConstrainedFloors:
    """The two reference floors for a box-constrained LQR problem (NB04 plan
    Sec 2.2/3.3): the loose unconstrained-LQR bound and the tighter box-aware
    SDP bound, both closed-form/convex-optimization numbers computed directly
    from `problem`'s own matrices -- deliberately never a `ContenderSpec`
    (see module docstring).

    Args:
        problem: The box-constrained `OptimalControlProblem`. Its own
            ``constraints`` are NOT read here -- `u_max` is passed
            explicitly, since the SDP formulation needs the scalar bound
            directly, not a `BoxConstraint` object.
        process_noise_std: The problem's process noise standard deviation;
            ``W = process_noise_std**2 * I_n`` (matching
            `GaussianBatchSpec`'s own noise convention).
        u_max: The infinity-norm control bound.

    Returns:
        The `BoxConstrainedFloors`.

    Raises:
        RuntimeError: (from `solve_box_constrained_lower_bound`) if the SDP
            solver fails to find a feasible/optimal solution.
    """
    system, cost = require_linear_quadratic(problem)
    A = system.A_t[0]
    B = system.B_t[0]
    # `time_invariant_slice` is a dual-backend kernel (np.ndarray | Tensor);
    # this study's `LQRProblemFactory`-built problem is always NumPy-backed
    # (never routed through a torch ComputeContext), matching
    # `models.analytic.riccati.get_lqr_gradient_matrices`'s own cast idiom
    # for narrowing the same kernel's return type back to plain NumPy.
    Q = cast(np.ndarray, time_invariant_slice(cost.Q))
    R = cast(np.ndarray, time_invariant_slice(cost.R))
    n = system.dimensions.state_dim
    W = (process_noise_std**2) * np.eye(n)

    P_lqr = solve_discrete_are(A, B, Q, R)
    j_lqr = float(np.trace(P_lqr @ W))
    j_sdp, P_sdp = solve_box_constrained_lower_bound(A, B, Q, R, W, u_max)

    return BoxConstrainedFloors(
        j_lqr=j_lqr, j_sdp=float(j_sdp), p_lqr=P_lqr, p_sdp=P_sdp
    )
