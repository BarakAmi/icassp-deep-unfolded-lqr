"""NB05 -- LTV Box-Constrained LQR Benchmark: the Tier-3 `Experiment`
declaration (docs/planning/03_studies/nb05_ltv_box_constrained/ltv_box_constrained_lqr_benchmark.md). NB04 restated
over a TIME-VARYING linear system in one of three structured regimes
(periodic / block-constant / fully time-varying, unified by the distinct-
slice count ``D`` -- NB04 is this study's own ``D == 1`` endpoint). Four
contenders on one shared, LTV box-constrained (``|u_t| <= u_max``)
stochastic LQR instance, common-noise evaluated -- structurally IDENTICAL
wiring to NB04's contenders 1-4, plus two more added by the P-resolution
extension (docs/planning/03_studies/nb05_ltv_box_constrained/ltv_box_constrained_lqr_benchmark.md Sec 14):

    Truncated-Riccati  -- the unconstrained (per-step, LTV) Riccati optimum,
                          saturated onto the box at every step;
                          non-learnable (family "truncated_riccati").
    Standard-PGD       -- fixed step-size unrolled, PROJECTED gradient
                          descent, TRUE per-step Riccati P_t, nothing
                          learned (family "unfolded_fixed").
    Unfolded-alpha     -- learned per-iteration step size, TRUE per-step
                          Riccati P_t, projected each inner step (family
                          "unfolded" or "unfolded_warmstart", per its own
                          training_mode) -- carries the exact time-varying
                          cost-to-go "for free" (its representation SCALES
                          with D).
    Unfolded-alpha+P   -- learned per-iteration step size AND a learned
                          time-invariant matrix replacing Riccati P,
                          projected each inner step -- the ONLY contender
                          whose cost-to-go representation does NOT scale
                          with D (NB05 plan Sec 4.2). NB04's flagship;
                          here the object of study, not a foregone winner:
                          the headline question is whether/where its
                          box-constraint-awareness advantage is overtaken
                          by the true-P_t contenders' time-awareness as D
                          grows (plan Sec 0.3's thesis).
    Unfolded-alpha+P (scalar-modulated) -- R1.5 (plan Sec 14.9): learned
                          step size, a SHARED learned matrix (identical
                          construction to Unfolded-alpha+P's own), and a
                          learned per-iteration POSITIVE SCALAR rescaling
                          it: P^(j) = c_j * P. The parameter-matched
                          control for the next contender.
    Unfolded-alpha+P (per-iteration) -- R2 (plan Sec 14.1-14.10): learned
                          step size AND J fully independent learned
                          matrices, one per unfolding iteration --
                          disentangles whether Unfolded-alpha+P's single
                          learned matrix is limited by being LEARNED (vs.
                          exact) or by being ITERATION-invariant (vs.
                          varying), a confound the four-contender run left
                          untested (plan Sec 14.1).

COCP and COCP-LB are DROPPED (user decision, plan Sec 0.4/4.4): both are
structurally LTI -- a single `CvxpyLayer` baked with one fixed ``(A, B)``
cannot represent a genuinely time-varying plant, and COCP-LB's frozen seed
is an infinite-horizon LTI SDP that is not even a valid reference here. This
also removes NB04's dominant compute cost, so NB05 (four contenders, three
regimes) is cheaper overall than NB04 (six contenders, one regime) despite
the added regime axis.

Turning on time variation is exactly what `LTVLQRProblemFactory` is FOR
(`applications.ltv_factories`): `nb05_ltv_experiment` differs from
`nb04_box_constrained.nb04_box_constrained_experiment` only in which problem
factory it builds (`LTVLQRProblemFactory` vs `LQRProblemFactory`) and in
dropping the two COCP contenders -- every unfolded family already
auto-projects through `apply_constraints`, and the LTV densification fix
(`applications.recipes.unfolded.build_unfolded_controller`) already makes
every contender LTV-correct. No controller code changes for NB05 beyond
that one fix, made once, upstream of this module.

`UnfoldedModelConfig` is REUSED VERBATIM from `.nb03_unfolding` (imported,
never redefined), exactly as `nb04_box_constrained` already does: one
learned model's own ``(kind, init_method, training_mode, plan, schedule)``
is a free per-model Control-Panel choice, NB05 hardcodes no training regime.

This module also provides the LTV reference floors (`compute_ltv_floors`,
plan Sec 5/6.6) -- NB04's two floors
(`nb04_box_constrained.compute_box_constrained_floors`) are INFINITE-HORIZON
LTI objects (a DARE solve and a Boyd SDP) and must never be reused here:
called against an LTV problem they silently evaluate the plant frozen at
``t=0``, returning a number that is not a bound on the true LTV-constrained
cost at all. The replacements are both EXACT finite-horizon computations
over the problem's own per-step ``(A_t, B_t)`` -- no SDP, no infinite-
horizon approximation:

    Floor A (``j_lqr_fin``, mandatory) -- the exact finite-horizon
        UNCONSTRAINED optimum: a zero-terminal-cost backward Riccati sweep
        (Rule 5: the evaluated objective has NO terminal-state term, so the
        recursion computing its EXACT minimizer must seed ``P_N = 0``, not
        ``Q_N`` -- verified independently against the classical closed form
        ``tr(P_0 Sigma_0) + sum_t tr(P_{t+1} W)``) plus a forward covariance
        propagation of the resulting closed-loop gains, scored under the
        problem's OWN cost conventions (`compute_theoretical_expected_cost`,
        already LTV-native). Dropping a constraint can only lower the
        optimal cost, so this is a valid (if loose) lower bound on the
        box-constrained optimum, with NO Monte-Carlo error and NO horizon
        mismatch (unlike NB04's ``J_LQR``, an infinite-horizon proxy for a
        finite-horizon metric).
    Floor B (``j_box_fin``, phase-gated, on by default) -- the tighter,
        BOX-AWARE floor: a Lagrangian relaxation of the box constraint
        (``R_t -> R_t + diag(lambda_t)``, ``lambda_t >= 0``) maximized over
        the multipliers via L-BFGS-B, using the EXACT envelope-theorem
        gradient ``d g/d lambda_{t,i} = (1/N)(E[u_{t,i}^2] - u_max^2)`` --
        both the closed form and the gradient were verified against finite
        differences / direct Monte-Carlo simulation to machine precision
        while writing this module (not merely asserted). ``g(0) ==
        j_lqr_fin`` exactly, so this reproduces NB04's "price of the box"
        band ``[j_lqr_fin, j_box_fin]`` with the same semantics. Like NB04's
        SDP, this is a relaxation bound (a deterministic multiplier
        standing in for an almost-sure constraint), not the true optimum.

Both floors carry the SAME finite-horizon/Monte-Carlo-free caveats NB04's
own floors carry (documented there): "% above a lower bound" over-estimates
true suboptimality, and neither floor is an ATTAINED policy (unlike
Truncated-Riccati). Because NB05 drops COCP-LB, it has NO attained upper
bound on the true optimum other than the contenders themselves.
"""

from dataclasses import dataclass, field
from functools import partial

import numpy as np
import torch
from scipy.optimize import minimize

from .nb03_unfolding import UnfoldedModelConfig
from ..factories import GaussianBatchSpec
from ..ltv_factories import LTVLQRProblemFactory, LTVRegime
from ..recipes import UnfoldedKind
from ...core.cost.quadratic_cost import QuadraticCost
from ...core.optimal_control_problem import OptimalControlProblem
from ...core.runtime import Backend, ComputeContext, Precision
from ...core.system.linear_system import LinearSystem
from ...experiments import ContenderSpec, EvaluationProtocol, Experiment
from ...models.analytic.riccati import (
    compute_state_covariance_trajectory,
    compute_theoretical_expected_cost,
    finite_horizon_riccati,
)
from ...models.guards import require_linear_quadratic
from ...models.iterative import ControlInitMethod

#: NB05's experiment name (the run/registry identity every contender's run
#: directory is prefixed with).
EXPERIMENT_NAME = "NB05_LTVBoxConstrainedLQR"


def _zero_terminal_cost(cost: QuadraticCost) -> QuadraticCost:
    """Rule 5 (NB05 plan Sec 5.2): a copy of `cost` whose terminal slice
    ``Q[N]`` is zeroed -- the objective actually evaluated by
    `experiments.evaluation.evaluate_synthesized_controller` has NO
    terminal-state cost term (`QuadraticCost`'s ``include_terminal_cost``
    default is ``False``, and `applications.factories.LQRProblemFactory`/
    `applications.ltv_factories.LTVLQRProblemFactory` never override it), so
    the Riccati recursion computing the EXACT minimizer of THAT objective
    must seed ``P_N = 0``, not ``Q_N`` (`riccati_recursion`'s own default).
    Seeding ``P_N = Q_N`` instead optimizes a DIFFERENT (terminal-inclusive)
    objective, whose gains are not necessarily optimal for -- and whose
    closed-form cost can exceed the true minimum of -- the metric actually
    being evaluated, which would silently turn a "floor" into a value that
    is not one.

    Args:
        cost: The problem's own `QuadraticCost`; ``Q`` must already be
            genuinely time-stacked (`finite_horizon_riccati`'s own
            precondition, always true for a `LQRProblemFactory`/
            `LTVLQRProblemFactory`-built problem).

    Returns:
        A fresh `QuadraticCost` with ``Q[-1] = 0``, ``R`` and every other
        field unchanged.

    Raises:
        ValueError: If `cost.Q` is not genuinely time-stacked (2D).
    """
    if cost.Q.ndim != 3:
        raise ValueError(
            "_zero_terminal_cost requires a genuinely time-stacked Q "
            f"(ndim == 3), got ndim={cost.Q.ndim}."
        )
    Q = np.array(cost.Q, copy=True)
    Q[-1] = 0.0
    return QuadraticCost(
        Q=Q,
        R=cost.R,
        include_terminal_cost=cost.include_terminal_cost,
        is_time_averaged=cost.is_time_averaged,
    )


def _finite_horizon_value(
    system: LinearSystem,
    running_cost: QuadraticCost,
    score_cost: QuadraticCost,
    process_noise_cov: np.ndarray,
    initial_state_cov: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """The shared inner solve both Floor A and every Floor B candidate
    evaluation reduce to: the exact finite-horizon unconstrained optimum
    under `running_cost` (whose terminal slice is ALREADY zeroed by the
    caller -- see `_zero_terminal_cost`), scored under `score_cost`'s own
    conventions (Rule 5: the cost used to SOLVE for the optimal gains and
    the cost used to SCORE them may differ -- Floor B solves under a
    penalized ``R``, but always scores under the TRUE ``R``).

    Args:
        system: The problem's `LinearSystem` (per-step ``A_t``/``B_t``).
        running_cost: The (zero-terminal) cost the Riccati recursion solves
            under -- its ``R`` is what varies between Floor A (the true
            ``R``) and each Floor B candidate (``R + diag(lambda_t)``).
        score_cost: The cost the resulting gains are SCORED under
            (`compute_theoretical_expected_cost`'s own ``cost`` argument) --
            always the problem's TRUE, unmodified cost.
        process_noise_cov: ``W``, shape ``(n, n)``.
        initial_state_cov: ``Sigma_0``, shape ``(n, n)``.

    Returns:
        ``(value, K_arr, Sigma_arr)``: the scored finite-horizon time-
        averaged cost (a Python `float`), the achieving gains ``K_arr``
        (shape ``(N, m, n)``), and the closed-loop state covariance
        trajectory EXCLUDING the terminal step (shape ``(N, n, n)``, i.e.
        ``Sigma_0, ..., Sigma_{N-1}`` -- what `Floor B's gradient needs at
        every ``t``).
    """
    horizon = running_cost.R.shape[0]
    _, K_arr = finite_horizon_riccati(system, running_cost, horizon)
    j_curve = compute_theoretical_expected_cost(
        system, score_cost, K_arr, process_noise_cov, initial_state_cov
    )
    sigma_arr = compute_state_covariance_trajectory(
        system, K_arr, process_noise_cov, initial_state_cov
    )
    return float(j_curve[-1]), K_arr, sigma_arr[:-1]


@dataclass(frozen=True)
class _BoxFloorContext:
    """The fixed (non-optimized) data one Floor B candidate evaluation
    needs -- bundled into one object (mirroring `models.analytic.riccati
    .OneStepCostModel`/`LocalCostToGoModel`'s own pattern) so
    `_box_floor_value_and_grad` stays under the PLR0913 argument-count gate
    despite the genuinely many quantities `g(lambda)` depends on.

    Attributes:
        system: The problem's `LinearSystem`.
        Q: The problem's running state-cost stack, shape ``(N+1, n, n)``
            (terminal slice irrelevant -- the penalized cost is built
            zero-terminal internally).
        R: The problem's TRUE control-cost stack, shape ``(N, m, m)``.
        W: Process noise covariance, shape ``(n, n)``.
        Sigma0: Initial state covariance, shape ``(n, n)``.
        u_max: The infinity-norm control bound.
        horizon: The finite horizon ``N``.
    """

    system: LinearSystem
    Q: np.ndarray
    R: np.ndarray
    W: np.ndarray
    Sigma0: np.ndarray
    u_max: float
    horizon: int


def _box_floor_value_and_grad(
    lam: np.ndarray, ctx: _BoxFloorContext
) -> tuple[float, np.ndarray]:
    """One evaluation of ``-g(lambda)`` and its gradient (negated because
    `scipy.optimize.minimize` minimizes -- maximizing `g` is minimizing
    `-g`), at the candidate multipliers `lam` (NB05 plan Sec 5.3).

    ``g(lambda) = (1/N) [min over the penalized problem] - (u_max^2/N) *
    sum(lambda)``; the first term is `_finite_horizon_value` applied to the
    SAME ``Q``/system with ``R_t -> R_t + diag(lambda_t)``, scored under
    that SAME penalized cost (Rule 5 does not apply here -- unlike Floor A,
    Floor B's own definition scores under the penalized objective, by
    construction of the Lagrangian relaxation). The gradient is the exact
    envelope-theorem formula (verified against finite differences while
    writing this module): ``d g/d lambda_{t,i} = (1/N)(E[u_{t,i}^2] -
    u_max^2)``, with ``E[u_t u_t^T] = K_t Sigma_t K_t^T``.

    Args:
        lam: Candidate multipliers, shape ``(N, m)``, flattened by the
            caller for `scipy.optimize.minimize` and reshaped here.
        ctx: The fixed problem data (see `_BoxFloorContext`).

    Returns:
        ``(-g(lambda), -grad_g(lambda))``, the gradient flattened to match
        `lam`'s flattened shape.
    """
    horizon, u_max = ctx.horizon, ctx.u_max
    m = ctx.R.shape[-1]
    lam_2d = lam.reshape(horizon, m)
    Lambda = np.zeros((horizon, m, m))
    diag_idx = np.arange(m)
    Lambda[:, diag_idx, diag_idx] = lam_2d
    R_penalized = ctx.R + Lambda

    penalized_cost = QuadraticCost(
        Q=_running_padded(ctx.Q),
        R=R_penalized,
        include_terminal_cost=False,
        is_time_averaged=True,
    )
    value, K_arr, sigma_arr = _finite_horizon_value(
        ctx.system, penalized_cost, penalized_cost, ctx.W, ctx.Sigma0
    )

    grad = np.empty((horizon, m))
    for t in range(horizon):
        Euu = K_arr[t] @ sigma_arr[t] @ K_arr[t].T
        grad[t] = np.diag(Euu) / horizon

    g = value - (u_max**2 / horizon) * float(lam_2d.sum())
    grad_g = grad - (u_max**2 / horizon)
    return -g, -grad_g.reshape(-1)


def _running_padded(Q: np.ndarray) -> np.ndarray:
    """`Q` with its terminal slice zeroed -- the explicit, self-documenting
    zero-terminal-cost stack every penalized-R candidate in the Floor B
    search reuses (Rule 5), built once per call from the problem's own
    ``(N+1, n, n)`` running-cost stack.
    """
    zeroed = np.array(Q, copy=True)
    zeroed[-1] = 0.0
    return zeroed


@dataclass(frozen=True)
class LTVFloors:
    """The reference floor(s) bracketing an LTV box-constrained LQR
    problem's achievable finite-horizon average cost (NB05 plan Sec 5.4):
    ``j_lqr_fin <= j_box_fin <= J*_C <= every contender's evaluated cost``.

    Attributes:
        j_lqr_fin: The EXACT finite-horizon unconstrained floor (Sec 5.2) --
            mandatory, always computed.
        j_box_fin: The finite-horizon Lagrangian box floor (Sec 5.3) --
            ``None`` when `compute_ltv_floors` was called with
            ``compute_box_floor=False``. Always ``>= j_lqr_fin`` when
            present (``g(0) == j_lqr_fin`` and the optimizer only improves
            on the ``lambda=0`` starting point).
        k_arr: The gains achieving `j_lqr_fin` (the zero-terminal-cost
            Riccati solution under the TRUE ``R``), shape ``(N, m, n)``.
        lambda_star: The optimized box multipliers achieving `j_box_fin`,
            shape ``(N, m)``; ``None`` when `j_box_fin` is ``None``.
        box_gradient_norm: The final KKT stationarity residual (the
            PROJECTED gradient's infinity norm, not the raw gradient's --
            at an active bound ``lambda_i == 0`` a large raw gradient is
            expected and not a convergence failure; only a residual
            pointing back into the infeasible region counts) the box
            floor's optimizer converged to (Sec 5.3's convergence
            diagnostic -- report this alongside `j_box_fin`, never assume
            convergence silently); ``None`` when `j_box_fin` is ``None``.
    """

    j_lqr_fin: float
    j_box_fin: float | None
    k_arr: np.ndarray
    lambda_star: np.ndarray | None
    box_gradient_norm: float | None


def compute_ltv_floors(
    problem: OptimalControlProblem,
    *,
    process_noise_std: float,
    initial_state_std: float,
    u_max: float,
    compute_box_floor: bool = True,
) -> LTVFloors:
    """The reference floor(s) for an LTV box-constrained LQR problem (NB05
    plan Sec 5/6.6) -- deliberately NEVER
    `nb04_box_constrained.compute_box_constrained_floors` (see module
    docstring: that helper is an infinite-horizon LTI computation that
    silently evaluates only the ``t=0`` slice under LTV).

    Args:
        problem: The box-constrained `OptimalControlProblem`, built from
            `applications.ltv_factories.LTVLQRProblemFactory` (or any
            problem whose system/cost are LTV-shaped); `u_max` is passed
            explicitly rather than read from `problem.constraints` (Floor B
            needs the scalar bound directly, mirroring
            `nb04_box_constrained.compute_box_constrained_floors`'s own
            convention).
        process_noise_std: The problem's process noise standard deviation;
            ``W = process_noise_std**2 * I_n`` (matching
            `applications.factories.GaussianBatchSpec`'s own convention).
        initial_state_std: The problem's initial-state standard deviation;
            ``Sigma_0 = initial_state_std**2 * I_n``.
        u_max: The infinity-norm control bound.
        compute_box_floor: Whether to additionally solve for the tighter,
            box-aware Floor B (Sec 5.3's phase gate) -- ``True`` by default;
            set ``False`` to skip the L-BFGS-B solve (e.g. inside a large
            sweep where only the free Floor A is needed).

    Returns:
        The `LTVFloors`.
    """
    system, cost = require_linear_quadratic(problem)
    n = system.dimensions.state_dim
    m = system.dimensions.control_dim
    horizon = cost.R.shape[0]
    W = (process_noise_std**2) * np.eye(n)
    Sigma0 = (initial_state_std**2) * np.eye(n)

    zero_terminal = _zero_terminal_cost(cost)
    j_lqr_fin, k_arr, _ = _finite_horizon_value(system, zero_terminal, cost, W, Sigma0)

    if not compute_box_floor:
        return LTVFloors(
            j_lqr_fin=j_lqr_fin,
            j_box_fin=None,
            k_arr=k_arr,
            lambda_star=None,
            box_gradient_norm=None,
        )

    # `cost.R` is already required to be genuinely time-stacked, shape
    # (horizon, m, m) -- the same precondition `finite_horizon_riccati`
    # already enforced above via Floor A's own solve (`_finite_horizon_value`
    # -> `_check_riccati_preconditions`), so no 2D/3D branch is needed here.
    lam0 = np.zeros(horizon * m)
    box_ctx = _BoxFloorContext(
        system=system,
        Q=cost.Q,
        R=cost.R,
        W=W,
        Sigma0=Sigma0,
        u_max=u_max,
        horizon=horizon,
    )
    objective = partial(_box_floor_value_and_grad, ctx=box_ctx)
    result = minimize(
        objective,
        lam0,
        jac=True,
        method="L-BFGS-B",
        bounds=[(0.0, None)] * (horizon * m),
    )
    lambda_star = result.x.reshape(horizon, m)
    j_box_fin = -float(result.fun)
    # The KKT stationarity residual, NOT the raw ||grad||: at a converged
    # BOUND-constrained optimum, a coordinate pinned at lambda_i == 0 can
    # legitimately carry a large raw gradient (the bound, not a vanishing
    # gradient, is what stops it there) -- KKT only requires that gradient
    # to point away from the feasible region (>= 0 for this minimization of
    # -g); only a NEGATIVE component at an active bound is a genuine
    # violation. An interior coordinate (lambda_i > 0) still requires its
    # raw gradient to vanish. This is the standard projected-gradient
    # convergence check for box-constrained optimization.
    at_lower_bound = result.x <= 1e-12
    kkt_residual = np.where(at_lower_bound, np.minimum(result.jac, 0.0), result.jac)
    box_gradient_norm = float(np.max(np.abs(kkt_residual)))

    return LTVFloors(
        j_lqr_fin=j_lqr_fin,
        j_box_fin=j_box_fin,
        k_arr=k_arr,
        lambda_star=lambda_star,
        box_gradient_norm=box_gradient_norm,
    )


def _default_unfolded_alpha() -> UnfoldedModelConfig:
    """Unfolded-alpha's default: learned step size, cold start, end to end
    -- identical defaults to NB04's contender of the same name (NB05 plan
    Sec 4.1: structurally identical wiring to NB04's contenders 1-4; the
    training regime stays a free per-model Control-Panel choice)."""
    return UnfoldedModelConfig(kind=UnfoldedKind.LEARNED_STEP_SIZE)


def _default_unfolded_alpha_p() -> UnfoldedModelConfig:
    """Unfolded-alpha+P's default: learned step size AND matrix, cold
    start, trained layer-wise -- identical defaults to NB04's flagship
    contender."""
    return UnfoldedModelConfig(
        kind=UnfoldedKind.LEARNED_STEP_SIZE_AND_MATRIX, training_mode="layerwise"
    )


def _default_unfolded_alpha_p_scalarmod() -> UnfoldedModelConfig:
    """Unfolded-alpha+P (scalar-modulated)'s default: R1.5 (NB05 plan Sec
    14.9) -- learned step size, a shared learned matrix, and a learned
    per-iteration positive scalar rescaling it. Layer-wise, mirroring the
    flagship's own regime: layer-wise phase j trains step-size row j AND
    the modulation scalar c_j together, the SAME per-iteration pairing the
    flagship already applies to step size and (once "each"-gated) the
    matrix."""
    return UnfoldedModelConfig(
        kind=UnfoldedKind.LEARNED_STEP_SIZE_AND_SCALAR_MODULATED_MATRIX,
        training_mode="layerwise",
    )


def _default_unfolded_alpha_p_periter() -> UnfoldedModelConfig:
    """Unfolded-alpha+P (per-iteration)'s default: R2 (NB05 plan Sec
    14.1-14.10) -- learned step size AND J fully independent learned
    matrices, one per unfolding iteration. Layer-wise, mirroring the
    flagship's own regime: layer-wise phase j trains step-size row j AND
    P^(j) together."""
    return UnfoldedModelConfig(
        kind=UnfoldedKind.LEARNED_STEP_SIZE_AND_PER_ITERATION_MATRIX,
        training_mode="layerwise",
    )


@dataclass(frozen=True)
class NB05Config:
    """NB05's frozen settings: the shared LTV box-constrained problem/
    evaluation scale plus each trainable family's own training
    specification. Mirrors `nb04_box_constrained.NB04Config` field-for-field
    with two changes: the five ``cocp_*`` fields are DROPPED (Sec 0.4/4.4 --
    COCP is out of scope for NB05), and four new fields turn on time
    variation (`regime`, `period_or_block`, `variation_strength`,
    `perturb_B`) -- the LTV analogue of NB04Config's own single new
    `u_max` field relative to NB03Config.

    Attributes:
        state_dim: State dimension ``n``.
        control_dim: Control dimension ``m``.
        horizon: The finite horizon ``N``. Must be divisible by
            `period_or_block` for `LTVRegime.PERIODIC`/`BLOCK_CONSTANT`
            (`LTVLQRProblemFactory`'s own Rule 1) -- the default ``48`` is
            divisible by every period/block this plan's defaults suggest
            (2, 3, 4, 6, 8, 12, 16, 24).
        u_max: The infinity-norm control bound ``|u_t| <= u_max`` every
            contender is projected onto. Must be chosen so the constraint
            genuinely BINDS on this problem instance (NB05 plan Sec 9 Gate
            1) -- verify with
            `workbench.analysis.compute_box_binding_fraction` before
            trusting a run, exactly as NB04.
        regime: Which time-variation regime to draw
            (`applications.ltv_factories.LTVRegime`) -- the outer
            Control-Panel axis this study adds over NB04.
        period_or_block: The period ``p`` (`PERIODIC`) or block length ``c``
            (`BLOCK_CONSTANT`); ``None`` only valid for `FULLY_VARYING`
            (`LTVLQRProblemFactory`'s own validation).
        variation_strength: The perturbation strength ``epsilon`` (NB05 plan
            Sec 2.1/9 Gate 3) -- ``0.0`` degenerates to NB04's own LTI
            problem bit-for-bit; must be chosen so the realized time
            variation is genuinely non-negligible.
        perturb_B: Whether ``B`` is perturbed the same way ``A`` is; ``True``
            by default, with an ``A``-only diagnostic run available by
            setting this ``False``.
        num_unfolding_iterations: Unrolled depth ``J``, shared by every
            unfolded contender.
        step_size_init: Initial per-iteration step size.
        step_size_max: Step-size reparameterization upper bound.
        batch_size: Trajectories per evaluation batch.
        n_eval_batches: Evaluation batches drawn and averaged (``> 1``
            additionally reports ``eval_expected_cost_std`` -- needed for
            the crossover-with-uncertainty estimate, NB05 plan Sec 4.3).
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
            regime. Its single time-invariant ``P`` is deliberately kept
            (NB05 plan Sec 0.4 decision 2, Sec 4.2) -- the representational
            deficit under study, not engineered away.
        unfolded_alpha_p_scalarmod: The P-resolution extension's R1.5 (NB05
            plan Sec 14.9) -- learned step size, a SHARED learned matrix,
            and a learned per-iteration positive scalar rescaling it
            (``P^(j) = c_j * P``). The parameter-matched control for
            `unfolded_alpha_p_periter` below.
        unfolded_alpha_p_periter: The P-resolution extension's R2 (NB05 plan
            Sec 14.1-14.10) -- learned step size AND J fully independent
            learned matrices, one per unfolding iteration.
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
    horizon: int = 48
    u_max: float = 0.5
    regime: LTVRegime = LTVRegime.PERIODIC
    period_or_block: int | None = 4
    variation_strength: float = 0.4
    perturb_B: bool = True
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
    unfolded_alpha_p_scalarmod: UnfoldedModelConfig = field(
        default_factory=_default_unfolded_alpha_p_scalarmod
    )
    unfolded_alpha_p_periter: UnfoldedModelConfig = field(
        default_factory=_default_unfolded_alpha_p_periter
    )
    random_init_std: float = 1.0
    log_first_epochs: int = 10
    log_every_epochs: int = 20
    dtype: torch.dtype = torch.float64
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.u_max <= 0:
            raise ValueError(f"u_max must be strictly positive, got {self.u_max}.")


def build_nb05_config(  # noqa: PLR0913 -- a deliberate keyword-only FLAT
    # mirror of the `NB05Config` carrier (@dataclass is itself exempt from the
    # arg-count gate), matching `nb04_box_constrained.build_nb04_config`'s own
    # directive: the notebook assembles the carrier from flat, individually-
    # typed, discoverable top-level settings rather than authoring an opaque
    # `NB05Config(...)` literal. The two learned models' OWN settings stay
    # grouped into one `UnfoldedModelConfig` literal each (the same exemption
    # NB04 applies).
    *,
    state_dim: int,
    control_dim: int,
    horizon: int,
    u_max: float,
    regime: LTVRegime,
    variation_strength: float,
    num_unfolding_iterations: int,
    step_size_init: float,
    step_size_max: float,
    batch_size: int,
    n_eval_batches: int,
    seed: int,
    process_noise_std: float,
    unfolded_alpha: UnfoldedModelConfig,
    unfolded_alpha_p: UnfoldedModelConfig,
    unfolded_alpha_p_scalarmod: UnfoldedModelConfig,
    unfolded_alpha_p_periter: UnfoldedModelConfig,
    period_or_block: int | None = None,
    perturb_B: bool = True,
    initial_state_std: float = 1.0,
    standard_pgd_init_method: ControlInitMethod = ControlInitMethod.COLD,
    random_init_std: float = 1.0,
    log_first_epochs: int = 10,
    log_every_epochs: int = 20,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
) -> NB05Config:
    """Assemble the internal `NB05Config` carrier from flat, standalone
    settings: the notebook exposes every knob -- including `regime`/
    `period_or_block`/`variation_strength`/`perturb_B`, the LTV axis this
    study adds -- and calls this to build the carrier the factory/sweep pass
    around.

    Args:
        See `NB05Config` for every field's meaning; this is a flat, keyword-
        only mirror of that carrier's constructor.

    Returns:
        The assembled `NB05Config`.
    """
    return NB05Config(
        state_dim=state_dim,
        control_dim=control_dim,
        horizon=horizon,
        u_max=u_max,
        regime=regime,
        period_or_block=period_or_block,
        variation_strength=variation_strength,
        perturb_B=perturb_B,
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
        unfolded_alpha_p_scalarmod=unfolded_alpha_p_scalarmod,
        unfolded_alpha_p_periter=unfolded_alpha_p_periter,
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
    routing logic as `nb04_box_constrained._learned_unfolded_contender`
    (kept as NB05's own copy, ~15 lines of orchestration glue with no shared
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


def nb05_ltv_experiment(cfg: NB05Config) -> Experiment:
    """Compose the NB05 declaration from `cfg`.

    Differs from `nb04_box_constrained.nb04_box_constrained_experiment`
    ONLY in which problem factory it builds (`LTVLQRProblemFactory` instead
    of `LQRProblemFactory`) and in dropping the two COCP contenders (NB05
    plan Sec 0.4/4.4) -- every other line is structurally identical, by
    design (Sec 0.1: NB05 is NB04 restated over a time-varying plant).

    Args:
        cfg: The frozen NB05 settings.

    Returns:
        The `Experiment` (LTV box-constrained problem factory + four
        contenders + shared common-noise evaluation), ready for
        ``run_experiment``.
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
        problem=LTVLQRProblemFactory(
            state_dim=cfg.state_dim,
            control_dim=cfg.control_dim,
            horizon=cfg.horizon,
            seed=cfg.seed,
            regime=cfg.regime,
            period_or_block=cfg.period_or_block,
            variation_strength=cfg.variation_strength,
            perturb_B=cfg.perturb_B,
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
            _learned_unfolded_contender(
                "unfolded_alpha_p_scalarmod",
                cfg.unfolded_alpha_p_scalarmod,
                shape=shape,
                log_kwargs=log_kwargs,
                random_init_std=cfg.random_init_std,
                seed=cfg.seed,
            ),
            _learned_unfolded_contender(
                "unfolded_alpha_p_periter",
                cfg.unfolded_alpha_p_periter,
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
