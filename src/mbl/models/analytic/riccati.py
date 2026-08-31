from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch

from ...core.kernels.riccati import (
    compute_lqr_gradient_matrices,
    gradient_lipschitz_constant as kernel_gradient_lipschitz_constant,
    riccati_recursion,
)
from ...core.runtime import ComputeContext
from ...core.system.state_space_system import ControlPolicy
from ...core.system.linear_system import LinearSystem
from ...core.optimal_control_problem import OptimalControlProblem
from ...core.cost.quadratic_cost import QuadraticCost
from ..base import Config
from ..guards import (
    check_controllability,
    ensure_lqr_dimensional_coherence,
    ensure_riccati_horizon_coherence,
    require_linear_quadratic,
)
from ..lifecycle import SynthesizerBase
from ..registry import register_model


@register_model("analytic")
class RiccatiController:
    """Controller that implements the optimal feedback policy for a
    finite-horizon LQR problem, computed via backward Riccati recursion.

    LEGACY single-phase shape (solves in ``__init__``, serves policies from
    the same object): retained this stage because the application layer
    still constructs it directly; the Stage-S3 recipe work routes new
    orchestration through `RiccatiSynthesizer` instead, whose two-phase
    contract this class's behavior already matches (its P/K are identical
    bit-for-bit on the NumPy path — both run the same kernel recursion).

    Attributes:
        problem: The `OptimalControlProblem` this controller solves; `problem.system`
            must be a `LinearSystem` and `problem.cost` a `QuadraticCost`.
        horizon: The finite horizon length ``N``.
        config: Always ``None`` -- this controller has no learnable
            hyperparameters to log.
        P_arr: Cost-to-go matrices, shape ``(N+1, n, n)`` (see
            `finite_horizon_riccati`).
        K_arr: Feedback gains, shape ``(N, m, n)`` (see `finite_horizon_riccati`).
    """

    def __init__(self, problem: OptimalControlProblem, horizon: int) -> None:
        """
        Args:
            problem: The `OptimalControlProblem` to solve.
            horizon: The finite horizon length ``N``.

        Raises:
            ValueError: (from `finite_horizon_riccati`) if ``(A, B, Q, R)`` are
                dimensionally incoherent, or `problem.cost`'s ``Q``/``R`` are
                not genuinely time-stacked with at least ``N+1``/``N`` slices
                respectively (Phase 1A fail-fast guards).

        Warns:
            UserWarning: If ``(A, B)`` is not controllable (a diagnostic only
                -- finite-horizon Riccati remains well-posed; see
                `models.guards.check_controllability`).
        """
        self.problem = problem
        self.horizon = horizon
        self.config: Config | None = None
        system, cost = require_linear_quadratic(problem)
        # Diagnostic only (warns): finite-horizon Riccati is well-posed even
        # when (A, B) is uncontrollable -- see models/guards.py.
        check_controllability(system.A_t[0], system.B_t[0])
        self.P_arr, self.K_arr = finite_horizon_riccati(system, cost, horizon)

    def get_control_policy(self) -> ControlPolicy:
        """Return the optimal linear feedback policy ``u_t = -K_t x_t``.

        Returns:
            A `ControlPolicy` built from `self.K_arr` (see
            `get_riccati_control_policy`).
        """
        return get_riccati_control_policy(self.K_arr)

    def get_signature(self) -> dict:
        """`Signable` member: this controller's own type + horizon.

        The solved `problem`'s signature is intentionally omitted --
        `ProblemSignatureCallback` already logs it once at the root
        (``problem.*``); embedding it again here would just duplicate every
        key under ``controller.problem.*``.

        Returns:
            ``{"type": "RiccatiController", "horizon":}``.
        """
        return {
            "type": type(self).__name__,
            "horizon": self.horizon,
        }


@dataclass(frozen=True)
class SynthesizedRiccatiController:
    """The frozen offline artifact of `RiccatiSynthesizer` (T2.a): solved
    cost-to-go and gain stacks plus synthesis provenance. Cheap to hold,
    signable; performs no computation beyond building policy closures.

    Attributes:
        P_arr: Cost-to-go matrices, shape ``(N+1, n, n)``, resident on
            `context`'s substrate.
        K_arr: Feedback gains, shape ``(N, m, n)``, resident on `context`'s
            substrate.
        context: The effective `ComputeContext` the recursion executed
            under — the artifact's declared residency (T2.a v3 law).
        synthesizer_signature: The producing synthesizer's
            specification-time signature.
    """

    P_arr: np.ndarray | torch.Tensor
    K_arr: np.ndarray | torch.Tensor
    context: ComputeContext
    synthesizer_signature: dict[str, Any]

    def make_policy(self) -> ControlPolicy:
        """A fresh optimal linear feedback closure ``u_t = -K_t x_t``.

        Freshness law: every call builds a new closure over the frozen
        gains; no state can leak between rollouts (trivially satisfied by a
        stateless linear policy, still honored structurally).

        Returns:
            A batched `ControlPolicy` on this artifact's substrate.
        """
        return get_riccati_control_policy(self.K_arr)

    def get_signature(self) -> dict[str, Any]:
        """`Signable` member: synthesizer signature + synthesis provenance.

        Never embeds ``problem.*`` (C1 law) and never hashes the live
        ``P_arr``/``K_arr`` values — they are derived, not specification.

        Returns:
            ``{"type":, "synthesizer":, "compute_context":}``.
        """
        return {
            "type": type(self).__name__,
            "synthesizer": dict(self.synthesizer_signature),
            "compute_context": self.context.get_signature(),
        }


@dataclass(frozen=True)
class RiccatiSynthesizer(SynthesizerBase):
    """Two-phase (T2.a) synthesizer for the finite-horizon Riccati family:
    `synthesize` runs the backward recursion on the injected context's
    substrate — NumPy/CPU, torch/CPU, or torch/GPU — via the dual-backend
    kernel, and returns the frozen `SynthesizedRiccatiController`.

    Stateless by construction (frozen dataclass): synthesis can never leak
    state into this object, and one synthesizer may serve many problems.

    Attributes:
        horizon: The finite horizon length ``N`` (specification-time
            configuration — the only signature content besides the type).
    """

    horizon: int

    def synthesize(
        self, problem: OptimalControlProblem, ctx: ComputeContext | None = None
    ) -> SynthesizedRiccatiController:
        """Solve the finite-horizon LQR problem on `ctx`'s substrate.

        Args:
            problem: The `OptimalControlProblem` to solve; ``problem.system``
                must be a `LinearSystem` and ``problem.cost`` a
                `QuadraticCost` with genuinely time-stacked ``Q``/``R``.
            ctx: The injected compute context; ``None`` inherits the module
                default (NumPy/CPU/float64).

        Returns:
            The frozen `SynthesizedRiccatiController`, resident on the
            effective context's substrate.

        Raises:
            ValueError: If the Riccati preconditions fail (via
                `_check_riccati_preconditions`).
            UnsupportedBackendError: Never for this family — its envelope
                is the full backend set; documented for the contract.

        Warns:
            UserWarning: If ``(A, B)`` is not controllable (diagnostic only).
        """
        ctx = self._resolve_context(ctx)
        system, cost = require_linear_quadratic(problem)
        _check_riccati_preconditions(system, cost, self.horizon)
        check_controllability(system.A_t[0], system.B_t[0])

        # The single ingress conversion (T1.f): densify the per-step system
        # matrices (TimeSeriesMatrix stores unique slices + a modulo
        # schedule, so indexing — not `.array` — is the honest expansion)
        # and author every operand onto the context's substrate. Below this
        # point the kernel never converts or branches on backend.
        A = ctx.asarray(np.stack([system.A_t[k] for k in range(self.horizon)]))
        B = ctx.asarray(np.stack([system.B_t[k] for k in range(self.horizon)]))
        Q = ctx.asarray(cost.Q)
        R = ctx.asarray(cost.R)

        P_arr, K_arr = riccati_recursion(A, B, Q, R, self.horizon)
        return SynthesizedRiccatiController(
            P_arr=P_arr,
            K_arr=K_arr,
            context=ctx,
            synthesizer_signature=self.get_signature(),
        )

    def get_signature(self) -> dict[str, Any]:
        """Specification-time configuration only (C1 law): type + horizon.

        Returns:
            ``{"type": "RiccatiSynthesizer", "horizon":}``.
        """
        return super().get_signature() | {"horizon": self.horizon}


def finite_horizon_riccati(
    system: LinearSystem, cost: QuadraticCost, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """Solve finite-horizon LQR via backward Riccati recursion.

    Args:
        system: LinearSystem object that specifies the dynamics of the system.
        cost: QuadraticCost object that specifies the cost function.
        horizon: The time horizon for the LQR problem.

    Returns:
        P_arr: array of cost-to-go matrices P[k], shape (N+1, n, n).
        K_arr: array of feedback gains K[k], shape (N, m, n).

    Raises:
        ValueError: If ``(A, B, Q, R)`` are dimensionally incoherent (via
            `models.guards.ensure_lqr_dimensional_coherence`), or ``Q``/``R``
            are not genuinely time-stacked with at least ``horizon+1``/``horizon``
            slices respectively (via `models.guards.ensure_riccati_horizon_coherence`;
            Phase 1A fail-fast guards, checked before the recursion begins).
    """

    # Fail fast on the true finite-horizon preconditions, before the recursion:
    # dimensional coherence of (A, B, Q, R) and long-enough, genuinely
    # time-stacked cost matrices (the recursion indexes Q[horizon] and R[k]).
    _check_riccati_preconditions(system, cost, horizon)

    # The recursion mathematics lives once in the dual-backend kernel layer
    # (T1.e); on this NumPy path it executes the exact pre-S2 operation
    # sequence (golden-master bit-stability). A_t/B_t are passed as their
    # per-step-indexable TimeSeriesMatrix selves, never densified here.
    return cast(
        "tuple[np.ndarray, np.ndarray]",
        riccati_recursion(system.A_t, system.B_t, cost.Q, cost.R, horizon),
    )


def _check_riccati_preconditions(
    system: LinearSystem, cost: QuadraticCost, horizon: int
) -> None:
    """Shared fail-fast gate for every Riccati synthesis entry point.

    Args:
        system: The `LinearSystem` supplying ``A_t``/``B_t``.
        cost: The `QuadraticCost` supplying ``Q``/``R``.
        horizon: The finite horizon length ``N``.

    Raises:
        ValueError: If ``(A, B, Q, R)`` are dimensionally incoherent, or
            ``Q``/``R`` are not genuinely time-stacked with at least
            ``horizon+1``/``horizon`` slices respectively.
    """
    ensure_lqr_dimensional_coherence(
        A=system.A_t[0], B=system.B_t[0], Q=cost.Q, R=cost.R
    )
    ensure_riccati_horizon_coherence(horizon=horizon, Q=cost.Q, R=cost.R)


def get_lqr_gradient_matrices(
    P: np.ndarray, A: np.ndarray, B: np.ndarray, R: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the gradient matrices M_k = R + B^T P_{k+1} B and C_k = B^T P_{k+1} A for LQR gradient-based methods.

    Args:
        P: Array of cost-to-go matrices P[k], shape (N+1, n, n).
        A: Array of state transition matrices A[k], shape (N, n, n).
        B: Array of control input matrices B[k], shape (N, n, m).
        R: Array of control cost matrices R[k], shape (N, m, m).

    Returns:
        M_arr: Array of matrices M[k], shape (N, m, m).
        C_arr: Array of matrices C[k], shape (N, m, n).
    """
    return cast(
        "tuple[np.ndarray, np.ndarray]",
        compute_lqr_gradient_matrices(P[1:], A, B, R),
    )


def gradient_lipschitz_constant(
    P: np.ndarray, A: np.ndarray, B: np.ndarray, R: np.ndarray
) -> float:
    """The Lipschitz constant of the per-step LQR gradient over the horizon.

    ``L = max_t lambda_max(2 (R_t + B_t^T P_{t+1} B_t))`` -- the largest
    eigenvalue of exactly the ``M_stack`` tensor `build_unfolded_controller`
    hands the refinement, which is stored there as ``2 * M``. The factor of two
    belongs to the constant: the gradient this bounds is that of the true
    per-step objective, not of half of it.

    It lives beside the ``M`` it is the norm of, so the factor of two and the
    ``P[1:]`` offset are stated once. `1/L` is the step size projected gradient
    descent converges fastest at, and ``2/L`` is the step at which it stops
    converging at all -- measured on the frozen ICASSP plants, the literal
    ``0.05`` the studies declare exceeds ``2/L`` on four of seven.

    **The routine is `eigvalsh` by name, and that is a contract rather than a
    detail.** ``eigvals``, ``norm(., 2)``, ``svd`` and the closed 2x2 form each
    land one ULP away on the n=4 plant, and one ULP of a declared
    ``step_size_init`` moves the `ModelID`. Thread count and torch-vs-numpy
    were bit-identical, so the hazard is the routine, not the machine.

    Args:
        P: Cost-to-go matrices ``P[k]``, shape ``(N+1, n, n)`` -- the array
            `finite_horizon_riccati` returns, offset internally exactly as
            `get_lqr_gradient_matrices` offsets it.
        A: State transition matrices, shape ``(N, n, n)``.
        B: Control input matrices, shape ``(N, n, m)``.
        R: Control cost matrices, shape ``(N, m, m)``.

    Returns:
        ``L``, strictly positive because ``R`` is positive definite.
    """
    # Delegates to the single-home kernel (T1.e): the authoring tool, this
    # wrapper and the spec tier's `step_is_inverse_lipschitz` gate must be one
    # computation, or the gate's exact-equality comparison would depend on
    # which of two identical-looking paths authored the literal.
    return kernel_gradient_lipschitz_constant(P, A, B, R)


def problem_gradient_lipschitz_constant(problem: OptimalControlProblem) -> float:
    """`gradient_lipschitz_constant` for a whole problem: solve the Riccati
    recursion and take the constant off it.

    The entry point every caller should use, because the two lines it saves are
    the two that are easy to get wrong. ``A_t[k]`` is the only correct per-step
    access -- ``TimeSeriesMatrix.array`` is the *storage*, which for a
    time-invariant system is the un-broadcast ``(n, n)`` rather than the
    ``(N, n, n)`` the gradient kernel requires, and passing it raises inside a
    transpose two frames away. `build_unfolded_controller` carries the same
    warning over the same two lines.

    Args:
        problem: A linear-quadratic problem; refused otherwise.

    Returns:
        ``L`` over the problem's own horizon.
    """
    system, cost = require_linear_quadratic(problem)
    horizon = int(np.asarray(cost.R).shape[0])
    P, _ = finite_horizon_riccati(system, cost, horizon)
    A = np.stack([system.A_t[k] for k in range(horizon)])
    B = np.stack([system.B_t[k] for k in range(horizon)])
    return gradient_lipschitz_constant(P, A, B, np.asarray(cost.R))


@dataclass(frozen=True)
class LocalCostToGoModel:
    """The frozen one-timestep model slice `evaluate_local_cost_to_go` (and
    the landscape oracle built on it) evaluates against: the system/cost
    matrices active at the chosen timestep plus the Riccati continuation.

    Attributes:
        A: State transition matrix at this timestep, shape ``(n, n)``.
        B: Control input matrix at this timestep, shape ``(n, m)``.
        Q: Running state cost matrix at this timestep, shape ``(n, n)``.
        R: Control cost matrix at this timestep, shape ``(m, m)``.
        P_next: Riccati cost-to-go matrix ``P_{t+1}``, shape ``(n, n)``.
        process_noise_cov: ``Sigma_w``, shape ``(n, n)``.
    """

    A: np.ndarray
    B: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    P_next: np.ndarray
    process_noise_cov: np.ndarray


def evaluate_local_cost_to_go(
    x: np.ndarray,
    U: np.ndarray,
    model: LocalCostToGoModel,
) -> np.ndarray:
    """The exact one-step Bellman cost-to-go at a FROZEN state ``x`` (Eq. 6 of
    the GD-LQR signal-space notebook's derivation, ``Q_k(x, u)``), evaluated
    over a batch of candidate controls ``U``:

        V(x, u) = x^T Q x + u^T R u + E_w[(Ax+Bu+w)^T P_next (Ax+Bu+w)]
                = u^T M u + 2 u^T (C x) + [x^T (Q + A^T P_next A) x + tr(P_next Sigma_w)]

    where ``M = R + B^T P_next B`` and ``C = B^T P_next A`` are exactly
    `get_lqr_gradient_matrices`'/`build_riccati_gd_refinement`'s own gradient
    coefficients (reused verbatim via `compute_lqr_gradient_matrices`, never
    re-derived) -- so this quadratic's unique minimizer over ``u`` is
    provably ``u* = -M^{-1} (C x) = -K x`` (the Riccati gain applied to the
    SAME frozen ``x``), by construction and independent of any grid
    resolution. Evaluating a landscape with this function therefore can
    never exhibit a numerically "displaced" optimum: the true minimizer is
    a fixed point of the algebra, not a rollout-estimated location.

    Unlike replaying ``U`` open-loop through the full dynamics (a Monte
    Carlo full-horizon rollout, `viz.landscape.oracles
    .make_rollout_cost_oracle`), this never propagates past `x` -- there is
    no "downstream" control policy to get wrong, which is exactly why a
    rollout-based surrogate can only approximate this cost-to-go rather than
    reproduce it exactly (its downstream continuation, held open-loop at a
    stale baseline rather than re-optimized against the perturbed state,
    makes the surrogate's argmin drift from the true ``-Kx``).

    Args:
        x: the frozen reference state, shape ``(n,)``.
        U: batch of candidate controls, shape ``(P, m)``.
        model: the one-timestep model slice (system/cost matrices at this
            timestep, Riccati continuation, and noise covariance).

    Returns:
        ``V(x, u)`` for every row of `U`, shape ``(P,)``.
    """
    A, B, Q, R = model.A, model.B, model.Q, model.R
    P_next, process_noise_cov = model.P_next, model.process_noise_cov
    M, C = compute_lqr_gradient_matrices(P_next, A, B, R)
    Cx = C @ x  # (m,)
    quad_u = np.einsum("pi,ij,pj->p", U, M, U)
    linear = 2.0 * (U @ Cx)
    const = x @ (Q + A.T @ P_next @ A) @ x + np.trace(P_next @ process_noise_cov)
    return cast(np.ndarray, quad_u + linear + const)


def get_riccati_control_policy(K_arr: "np.ndarray | torch.Tensor") -> ControlPolicy:
    """Get a control policy function from Riccati feedback gains.

    Args:
        K_arr: Riccati feedback gains (N, m, n).

    Returns:
        ControlPolicy: A function that takes time step t and a *batched* state
        x (batch_size, n), and returns a batched control input (batch_size, m),
        matching ControlPolicy's batched contract used by System.run.
    """

    def policy(t: int, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        # K_arr's backend follows the synthesis ComputeContext. The one mixed
        # combination -- torch state over numpy gains -- must still run its
        # matmul in NUMPY (bit-identical to the S0 golden fixtures; torch's
        # BLAS rounds differently at the 1e-8 level), but spelled explicitly
        # instead of `tensor @ ndarray`, whose implicit interop goes through
        # numpy's deprecated __array_wrap__ hook. `.numpy()` intentionally
        # raises on a grad-requiring state, exactly as the implicit coercion
        # always did. Backend-matched pairs take the plain matmul unchanged.
        K_t = K_arr[t]
        if isinstance(x, torch.Tensor) and isinstance(K_t, np.ndarray):
            return torch.from_numpy((-x).numpy() @ K_t.T)
        return -x @ K_t.T

    return policy


def get_riccati_control_trajectory(X: np.ndarray, K_arr: np.ndarray) -> np.ndarray:
    """Get control inputs from Riccati feedback gains and state trajectory.

    Args:
        X: State trajectory (batch_size, N+1, n).
        K_arr: Riccati feedback gains (N, m, n).

    Returns:
        U: Control trajectory (N, m) or (batch_size, N, m).
    """
    return cast(np.ndarray, -np.einsum("tmi,bti->btm", K_arr, X[:, :-1]))


def compute_state_covariance_trajectory(
    system: LinearSystem,
    K_arr: np.ndarray,
    process_noise_cov: np.ndarray,
    initial_state_cov: np.ndarray,
) -> np.ndarray:
    """Propagate the state covariance Sigma_t forward under closed-loop
    dynamics x_{t+1} = (A_t - B_t K_t) x_t + w_t, w_t ~ (0, process_noise_cov)
    -- the second-moment counterpart to `finite_horizon_riccati`'s backward
    P_t recursion, needed to compute a linear feedback controller's
    *theoretical* (as opposed to Monte-Carlo empirical) expected cost in
    closed form (see `compute_theoretical_expected_cost`).

    Args:
        system: LinearSystem whose A_t/B_t define the open-loop dynamics.
        K_arr: Feedback gains (u_t = -K_t x_t), shape (horizon, m, n).
        process_noise_cov: Process noise covariance, shape (n, n).
        initial_state_cov: Initial state covariance Sigma_0, shape (n, n).

    Returns:
        Sigma: State covariance matrices, shape (horizon + 1, n, n).
    """
    horizon = K_arr.shape[0]
    covariances = [initial_state_cov]
    for k in range(horizon):
        closed_loop = system.A_t[k] - system.B_t[k] @ K_arr[k]
        covariances.append(
            closed_loop @ covariances[-1] @ closed_loop.T + process_noise_cov
        )
    return np.stack(covariances)


def compute_theoretical_expected_cost(
    system: LinearSystem,
    cost: QuadraticCost,
    K_arr: np.ndarray,
    process_noise_cov: np.ndarray,
    initial_state_cov: np.ndarray,
) -> np.ndarray:
    """Closed-form counterpart to `QuadraticCost.__call__`'s Monte-Carlo
    cumulative cost: analytically propagates the state covariance under the
    given feedback gains (`compute_state_covariance_trajectory`) and
    evaluates E[x_t^T Q_t x_t + u_t^T R_t u_t] = tr((Q_t + K_t^T R_t K_t) Sigma_t)
    at each step. Mirrors `QuadraticCost`'s own `include_terminal_cost`/
    `is_time_averaged` semantics exactly (reading both straight off `cost`,
    never as separate parameters, so the two curves cannot silently
    disagree), so the two curves are directly comparable (the empirical
    curve should converge to this one as the rollout batch size grows)
    regardless of which cost variant the problem uses.

    Args:
        system: LinearSystem whose A_t/B_t define the closed-loop dynamics.
        cost: The problem's QuadraticCost (supplies Q, R, and the
            include_terminal_cost/is_time_averaged configuration to mirror).
        K_arr: Feedback gains (u_t = -K_t x_t), shape (horizon, m, n).
        process_noise_cov: Process noise covariance, shape (n, n).
        initial_state_cov: Initial state covariance Sigma_0, shape (n, n).

    Returns:
        Cumulative expected cost, shape (horizon,) -- one value per step
        k=0,...,horizon-1, normalized by the elapsed step count
        (`cost.is_time_averaged` ``True``) or left as the raw cumulative sum
        (``False``), matching `QuadraticCost.__call__`'s own convention exactly.
    """
    horizon = K_arr.shape[0]
    Sigma = compute_state_covariance_trajectory(
        system, K_arr, process_noise_cov, initial_state_cov
    )

    def _stacked(mat: np.ndarray, length: int) -> np.ndarray:
        """Broadcast a (possibly time-invariant) 2D cost matrix to an
        explicit per-step stack, so the einsums below never need to
        special-case QuadraticCost's 2D-vs-3D Q/R convention."""
        return mat if mat.ndim == 3 else np.broadcast_to(mat, (length, *mat.shape))

    Q = _stacked(cost.Q, horizon + 1)  # Q[0..horizon]
    R = _stacked(cost.R, horizon)  # R[0..horizon-1]

    # trace(A @ B) == sum(A * B) elementwise whenever A and B are both
    # symmetric (true here: Q/R are validated PSD/PD by QuadraticCost, Sigma
    # is a covariance) -- the same identity the legacy costs.py relied on,
    # avoiding an explicit per-step matmul + np.trace loop.
    KtRK = np.einsum("tpq,tqr,trs->tps", K_arr.transpose(0, 2, 1), R, K_arr)
    stage_cost = np.einsum("tij,tij->t", Q[:-1] + KtRK, Sigma[:-1])
    J_cum = np.cumsum(stage_cost)

    if cost.include_terminal_cost:
        # x_{k+1}^T Q_{k+1} x_{k+1} folded in for k=0,...,horizon-2; the true
        # final state (k=horizon-1) always reuses Q[horizon] too -- exactly
        # QuadraticCost.__call__'s own branching (no separate terminal matrix).
        lookahead = np.einsum("tij,tij->t", Q[1:-1], Sigma[1:-1])
        lookahead = np.append(lookahead, np.trace(Q[horizon] @ Sigma[horizon]))
        J_cum += lookahead

    if cost.is_time_averaged:
        return J_cum / np.arange(1, horizon + 1)
    return J_cum
