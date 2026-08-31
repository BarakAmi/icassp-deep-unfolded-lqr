"""Report analysis (T3.g *analysis*): result payloads in, DataFrames,
plot-ready grids/curves, and validation verdicts out — no simulation of new
trajectories, no display. The `compute_*` helpers here assemble the exact
data the pure `viz` renderers consume (REFACTOR_PLAN v3, T4.b: math lives
below the presentation tier, never inside a plotting module).
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import solve_discrete_are, sqrtm

from ..core.constraint.constraint import Constraint
from ..core.cost.quadratic_cost import QuadraticCost
from ..core.kernels import time_invariant_slice
from ..core.optimal_control_problem import OptimalControlProblem
from ..core.system.linear_system import LinearSystem
from ..core.utils import to_numpy
from ..models.analytic.riccati import compute_theoretical_expected_cost
from ..models.constrained.solver_resolution import (
    COCPCanaryInstance,
    ResolvedCOCPSolver,
    resolve_cocp_solver,
)
from ..models.guards import require_linear_quadratic
from ..models.lqr_gradient import compute_lqr_gradient_matrices
from ..viz.plots.benchmarking import BenchmarkRecord
from ..viz.plots.convergence import ConvergenceRecord

#: Every (terminal-cost, time-averaged) convention this project's
#: `QuadraticCost` supports (Phase 1F: expanded from one axis to two), keyed
#: by the human-readable label `evaluate_cost_conventions`/
#: `summarize_cost_comparison` report it under.
COST_CONVENTIONS = {
    "Terminal, Time-Averaged": (True, True),
    "No Terminal, Time-Averaged": (False, True),
    "Terminal, Raw": (True, False),
    "No Terminal, Raw": (False, False),
}


@dataclass(frozen=True)
class NoiseCovariances:
    """The two covariances the theoretical (closed-form) expected-cost curve
    needs alongside the closed-loop gains.

    Attributes:
        process_noise: Process noise covariance ``Sigma_w``, shape ``(n, n)``.
        initial_state: Initial state covariance ``Sigma_0``, shape ``(n, n)``.
    """

    process_noise: np.ndarray
    initial_state: np.ndarray


def evaluate_cost_conventions(
    system: LinearSystem,
    base_cost: QuadraticCost,
    K_arr: np.ndarray,
    states: np.ndarray,
    controls: np.ndarray,
    *,
    covariances: NoiseCovariances,
) -> dict[str, dict[str, np.ndarray]]:
    """Empirical (Monte Carlo) and theoretical (closed-form) cost curves for
    every (terminal-cost, time-averaged) `QuadraticCost` convention in
    `COST_CONVENTIONS`, evaluated on the same already-simulated trajectories
    and closed-loop gains -- rescoring `states`/`controls` under a different
    cost convention needs no additional simulation.

    Args:
        system: LinearSystem whose A_t/B_t define the closed-loop dynamics.
        base_cost: The problem's `QuadraticCost`; only its ``Q``/``R``
            matrices are read (each convention constructs its own flag
            variant over them).
        K_arr: Feedback gains (u_t = -K_t x_t), shape (horizon, m, n).
        states: Simulated state trajectories, shape (batch, horizon+1, n).
        controls: Simulated control trajectories, shape (batch, horizon, m).
        covariances: Process-noise and initial-state covariances for the
            theoretical curve.

    Returns:
        {convention_label: {"empirical_cost": ..., "theoretical_cost": ...}}
        for each convention in `COST_CONVENTIONS`, each value a 1D array of
        length `horizon` (see `QuadraticCost.__call__` and
        `compute_theoretical_expected_cost`).
    """
    curves = {}
    for label, (include_terminal_cost, is_time_averaged) in COST_CONVENTIONS.items():
        cost = QuadraticCost(
            Q=base_cost.Q,
            R=base_cost.R,
            include_terminal_cost=include_terminal_cost,
            is_time_averaged=is_time_averaged,
        )
        curves[label] = {
            "empirical_cost": cost(states, controls),
            "theoretical_cost": compute_theoretical_expected_cost(
                system,
                cost,
                K_arr,
                covariances.process_noise,
                covariances.initial_state,
            ),
        }
    return curves


def summarize_cost_comparison(
    curves: Mapping[str, Mapping[str, np.ndarray]],
) -> pd.DataFrame:
    """Publication-ready summary table for the terminal-time (t=N) empirical
    vs. theoretical cost, one row per named curve (e.g. the output of
    `evaluate_cost_conventions`) -- replaces ad hoc printed strings with a
    single `pandas.DataFrame` Jupyter renders as an HTML table.

    Args:
        curves: label -> {"empirical_cost": ..., "theoretical_cost": ...},
            each a 1D cost array (time-averaged or raw cumulative, depending
            on the `QuadraticCost` convention that produced it); only the
            final entry (t=N) of each is reported.

    Returns:
        DataFrame indexed by label, with columns "Empirical", "Theoretical",
        "Absolute Error", and "Relative Error (%)".
    """
    rows = {}
    for label, curve in curves.items():
        empirical = float(curve["empirical_cost"][-1])
        theoretical = float(curve["theoretical_cost"][-1])
        absolute_error = abs(empirical - theoretical)
        rows[label] = {
            "Empirical": empirical,
            "Theoretical": theoretical,
            "Absolute Error": absolute_error,
            "Relative Error (%)": 100.0 * absolute_error / abs(theoretical),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def validate_convergence(
    records: Mapping[str, ConvergenceRecord], J_opt: float, tolerance: float
) -> None:
    """Every record's final expected cost must land within `tolerance`
    relative error of the Riccati optimum -- the shared acceptance check for
    both Global Convergence Diagnostics experiments (Phase: every starting
    signal/update ordering is expected to reach the same stationary point of
    Eq. 6, only the transient should differ).

    Raises:
        AssertionError: If any record's relative error to `J_opt` is not
            below `tolerance`.
    """
    for name, record in records.items():
        rel_err = abs(record.J_final - J_opt) / abs(J_opt)
        if rel_err >= tolerance:
            raise AssertionError(
                f"{name!r} did not reach the expected optimum "
                f"(relative error {rel_err:.2e} >= tolerance {tolerance:.1e})."
            )


def first_sustained_index(J_history: np.ndarray, threshold: float) -> int | None:
    """Smallest iteration index `i` such that `J_history[i:]` never again
    exceeds `threshold` -- "dropped and stayed below", not merely the first
    (possibly transient) dip under it. `None` if no such index exists (the
    margin is never durably reached).
    """
    above = np.flatnonzero(J_history > threshold)
    if above.size == 0:
        return 0
    last_above = int(above[-1])
    return last_above + 1 if last_above + 1 < len(J_history) else None


def build_convergence_margin_table(
    records: Mapping[str, ConvergenceRecord],
    J_opt: float,
    margin_percentages: Sequence[float],
) -> pd.DataFrame:
    """ "Convergence Margins" table: for each strategy in `records` and each
    requested margin `p` in `margin_percentages`, the first macro-iteration
    index at which the empirical batch-averaged expected cost
    $\\widehat{\\mathbb{E}}[J(U^{(i)})]$ drops and stays within `p`% of the
    theoretical Riccati optimum $J_{\\mathrm{opt}}$ -- `"not reached"` if the
    margin is never durably entered within the recorded iterations.

    Args:
        records: label -> `OptimizationResult`, e.g. one experiment's
            per-strategy solves.
        J_opt: The theoretical (Riccati) optimal expected cost.
        margin_percentages: The requested margins `p`, e.g. `[1.0, 5.0,
            10.0]` for 1%/5%/10% bands around `J_opt`.

    Returns:
        A DataFrame indexed by strategy label, one column per requested
        margin (e.g. `"5% margin"`), entries either the first sustained
        iteration index or `"not reached"`.
    """
    rows = {}
    for name, record in records.items():
        row = {}
        for p in margin_percentages:
            threshold = abs(J_opt) * (1.0 + p / 100.0)
            idx = first_sustained_index(record.J_history, threshold)
            row[f"{p:g}% margin"] = idx if idx is not None else "not reached"
        rows[name] = row
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "Strategy"
    return table


@dataclass(frozen=True)
class _DepthCurveRecord:
    """A `ConvergenceRecord`-shaped view of one contender's per-depth cost
    curve, letting `build_convergence_margin_table`'s macro-iteration margin
    logic apply to unfolding DEPTH K exactly as it applies to solver
    iterations (`build_unfolding_convergence_margin_table`'s sole use)."""

    J_history: np.ndarray
    J_final: float


def build_unfolding_convergence_margin_table(
    k_values: Sequence[int],
    curves: Mapping[str, Sequence[float]],
    J_opt: float,
    margin_percentages: Sequence[float],
) -> pd.DataFrame:
    """NB03's "Convergence Margins" table: the unfolding-DEPTH analogue of
    `build_convergence_margin_table` (which reports solver ITERATIONS). For
    each K-dependent (iterative) contender in `curves` and each requested
    margin `p`, the SMALLEST unfolding depth K at which that contender's
    evaluated cost drops and stays within `p`% of the theoretical Riccati
    optimum `J_opt` -- ``"not reached"`` if no swept depth durably enters the
    margin. Reuses `build_convergence_margin_table`'s own margin logic
    (`first_sustained_index`) verbatim by wrapping each curve in a
    `ConvergenceRecord`-shaped record (DRY: the "drops and stays within p%"
    rule is defined once), then remaps the returned array INDICES back to
    the actual depth values `k_values` names.

    Args:
        k_values: the unfolding depths swept, shared x-axis for every entry
            in `curves` (e.g. `workbench.depth_sweep.DepthSweepResult
            .k_values`).
        curves: label -> cost per depth (same length/order as `k_values`),
            one entry per K-DEPENDENT contender (e.g. `DepthSweepResult
            .curves` -- already excludes the depth-invariant baselines, so
            every entry here is a genuinely iterative contender, learned or
            not).
        J_opt: the theoretical (Riccati) optimal expected cost.
        margin_percentages: the requested margins `p`, e.g. ``[1.0, 5.0,
            10.0]`` for 1%/5%/10% bands around `J_opt`.

    Returns:
        A DataFrame indexed by contender label, one column per requested
        margin, entries either the first sustained unfolding depth K (an
        actual value from `k_values`, never a raw array index) or
        ``"not reached"``.

    Raises:
        ValueError: If `curves` is empty, or any curve's length disagrees
            with `k_values`'s.
    """
    if not curves:
        raise ValueError("curves must contain at least one named curve.")
    k_arr = np.asarray(k_values)
    records: dict[str, _DepthCurveRecord] = {}
    for label, values in curves.items():
        value_arr = np.asarray(values, dtype=float)
        if value_arr.shape != k_arr.shape:
            raise ValueError(
                f"Curve '{label}' has shape {value_arr.shape}, expected "
                f"{k_arr.shape} (one cost per entry in k_values)."
            )
        records[label] = _DepthCurveRecord(
            J_history=value_arr, J_final=float(value_arr[-1])
        )

    index_table = build_convergence_margin_table(records, J_opt, margin_percentages)
    return index_table.map(
        lambda cell: (
            int(k_arr[int(cell)]) if isinstance(cell, (int, np.integer)) else cell
        )
    )


def build_benchmark_table(records: Sequence[BenchmarkRecord]) -> pd.DataFrame:
    """The tabular face of a benchmark suite's `BenchmarkRecord`s: one row
    per method, phase-split wall-clock and stratified-memory columns --
    exactly the table the benchmarking notebook displays alongside Asset 7's
    barplot grid.

    Returns:
        DataFrame indexed by method label with "Offline (s)"/"Online (s)"/
        "Offline NumPy/CPU Memory (MB)"/"Offline PyTorch Memory (MB)"/
        "Online NumPy/CPU Memory (MB)"/"Online PyTorch Memory (MB)" columns.
    """
    return pd.DataFrame(
        {
            "Offline (s)": [r.offline_s for r in records],
            "Online (s)": [r.online_s for r in records],
            "Offline NumPy/CPU Memory (MB)": [
                r.offline_memory.numpy_mb for r in records
            ],
            "Offline PyTorch Memory (MB)": [r.offline_memory.torch_mb for r in records],
            "Online NumPy/CPU Memory (MB)": [r.online_memory.numpy_mb for r in records],
            "Online PyTorch Memory (MB)": [r.online_memory.torch_mb for r in records],
        },
        index=[r.label for r in records],
    )


def compute_cumulative_average_cost(
    states: np.ndarray,
    controls: np.ndarray,
    *,
    Q: np.ndarray | None = None,
    R: np.ndarray | None = None,
) -> np.ndarray:
    """Running average stage cost x_t^T Q x_t + u_t^T R u_t over time,
    averaged across the batch -- the data
    `viz.plots.cost_analysis.plot_cumulative_cost_over_time` expects. Q/R
    default to identity, matching this project's applications
    (StandardLQRApp/BoxConstraintLQRApp both use Q=R=I).

    Args:
        states: (batch, horizon+1, n) or (horizon+1, n).
        controls: (batch, horizon, m) or (horizon, m).

    Returns:
        1D array of length `horizon`.
    """
    states = to_numpy(states)
    controls = to_numpy(controls)
    if states.ndim == 2:
        states = states[None]
    if controls.ndim == 2:
        controls = controls[None]

    horizon = controls.shape[1]
    n, m = states.shape[-1], controls.shape[-1]
    Q = np.eye(n) if Q is None else Q
    R = np.eye(m) if R is None else R

    state_cost = np.einsum(
        "bti,ij,btj->bt", states[:, :horizon], Q, states[:, :horizon]
    )
    control_cost = np.einsum("bti,ij,btj->bt", controls, R, controls)
    stage_cost = (state_cost + control_cost).mean(axis=0)
    return np.cumsum(stage_cost) / np.arange(1, horizon + 1)


@dataclass(frozen=True)
class OneStepCostModel:
    """The four matrices defining the one-step control cost-to-go
    ``u^T R u + (Ax+Bu)^T P_next (Ax+Bu)`` that `compute_stage_cost_grid`
    evaluates and `compute_gradient_descent_path` descends.

    Attributes:
        A: State transition matrix, shape ``(n, n)``.
        B: Control input matrix, shape ``(n, m)``.
        R: Control cost matrix, shape ``(m, m)``.
        P_next: (True or learned) cost-to-go matrix, shape ``(n, n)``.
    """

    A: np.ndarray
    B: np.ndarray
    R: np.ndarray
    P_next: np.ndarray


def compute_stage_cost_grid(
    u1_range: np.ndarray,
    u2_range: np.ndarray,
    x: np.ndarray,
    *,
    model: OneStepCostModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the LQR stage cost u^T R u + (Ax+Bu)^T P_next (Ax+Bu) over a
    2D grid of control values, for a fixed state `x` -- the cost-to-go
    landscape a gradient-descent (unfolded) controller walks over `u`, in the
    exact shape `viz.plots.convergence.plot_control_convergence_contour_2d`
    consumes.

    Returns:
        (U1, U2, cost_grid), each of shape (len(u2_range), len(u1_range)).
    """
    A, B, R, P_next = model.A, model.B, model.R, model.P_next
    U1, U2 = np.meshgrid(u1_range, u2_range)
    cost_grid = np.empty_like(U1, dtype=float)
    for i in range(U1.shape[0]):
        for j in range(U1.shape[1]):
            u = np.array([U1[i, j], U2[i, j]])
            next_state = A @ x + B @ u
            cost_grid[i, j] = u @ R @ u + next_state @ P_next @ next_state
    return U1, U2, cost_grid


def compute_gradient_descent_path(
    x: np.ndarray,
    *,
    model: OneStepCostModel,
    step_sizes: np.ndarray,
    u0: np.ndarray | None = None,
    constraint: Constraint | None = None,
) -> np.ndarray:
    """Replay an unfolded controller's per-iteration control updates in plain
    numpy, given its learned step-size schedule and the (true or learned)
    cost-to-go matrix -- the same update rule as
    `models.unfolded.iterative_refinement.StepSizeRefinement`/`RiccatiRefinement`
    (`u <- constraint(u - alpha * (M @ u + C @ x))`), reused here via
    `compute_lqr_gradient_matrices` so the math can't silently drift from the
    real controller's.

    Args:
        x: The frozen reference state, shape ``(n,)``.
        model: The one-timestep cost-to-go model (see `OneStepCostModel`).
        step_sizes: The per-iteration step sizes to replay.
        u0: Optional initial guess; defaults to zero (cold start).
        constraint: Optional feasible-set projection, applied after EVERY
            iteration (never only the last) -- exactly
            `models.iterative.refinement.GradientDescentRefinement
            .refine_step`'s own per-step `apply_constraints` call. ``None``
            (default) replays the unconstrained update unchanged, so every
            existing (unconstrained) call site is unaffected; a
            box-constrained caller (NB04) should pass the problem's own
            `core.constraint.box_constraint.BoxConstraint` here so the
            replayed path can never wander somewhere the real controller's
            projected inner loop could never actually reach.

    Returns:
        Array of shape (len(step_sizes) + 1, m): the control iterate
        including the initial guess u0.
    """
    A, B, R, P_next = model.A, model.B, model.R, model.P_next
    M, C = compute_lqr_gradient_matrices(P_next, A, B, R)
    M, C = 2 * M, 2 * C
    m = B.shape[1]
    u = np.zeros(m) if u0 is None else np.array(u0, dtype=float)
    x_Ct = x @ C.T

    path = [u.copy()]
    for alpha in np.atleast_1d(step_sizes):
        grad = u @ M + x_Ct
        u = u - alpha * grad
        if constraint is not None:
            u = np.asarray(constraint(u))
        path.append(u.copy())
    return np.stack(path)


def compute_box_binding_fraction(controls: np.ndarray, u_max: float) -> float:
    """Fraction of entries in an UNCONSTRAINED control trajectory whose
    magnitude exceeds `u_max` -- the box-binding diagnostic
    (docs/planning/03_studies/nb04_box_constrained/box_constrained_lqr_benchmark.md Sec 6/9's acceptance gate):
    if this is near zero, `u_max` does not actually bind on this problem
    instance and a box-constrained benchmark run degenerates to an
    unconstrained one in all but name.

    Args:
        controls: An UNCONSTRAINED control trajectory (e.g. the plain
            Riccati policy's rollout -- never a controller that already
            projects, which would trivially read back as zero regardless of
            how tight `u_max` is), any shape.
        u_max: The infinity-norm control bound a box-constrained experiment
            is about to enforce.

    Returns:
        The fraction (in ``[0, 1]``) of entries with ``|controls| > u_max``.
    """
    return float(np.mean(np.abs(controls) > u_max))


def compute_box_binding_fraction_per_timestep(
    controls: np.ndarray, u_max: float
) -> np.ndarray:
    """Per-timestep fraction of entries in an UNCONSTRAINED control
    trajectory whose magnitude exceeds `u_max` -- the SAME diagnostic as
    `compute_box_binding_fraction`, resolved along the time axis rather than
    collapsed to one scalar (docs/planning/03_studies/nb05_ltv_box_constrained/ltv_box_constrained_lqr_benchmark.md
    Sec 6.7): under LTV the constraint may bind hard in some phases of the
    schedule and not at all in others, and the single aggregate fraction
    hides that shape entirely.

    Args:
        controls: An UNCONSTRAINED control trajectory, shape
            ``(batch, horizon, m)`` (`compute_box_binding_fraction`'s own
            "never a controller that already projects" caveat applies
            identically here). Unlike that function, this one assumes the
            batch-first, time-second layout every rollout in this project
            produces -- not "any shape".
        u_max: The infinity-norm control bound a box-constrained experiment
            is about to enforce.

    Returns:
        Array of shape ``(horizon,)``: the fraction (in ``[0, 1]``) of
        ``(batch, m)`` entries with ``|controls[:, t, :]| > u_max``, one
        value per time step ``t``.
    """
    exceeds = np.abs(controls) > u_max
    reduce_axes = tuple(axis for axis in range(exceeds.ndim) if axis != 1)
    return np.asarray(exceeds.mean(axis=reduce_axes))


@dataclass(frozen=True)
class CrossoverEstimate:
    """Where a signed "gap" curve (e.g. one contender's cost minus
    another's) crosses zero, if at all, with a heuristic uncertainty
    bracket from each point's own standard error
    (docs/planning/03_studies/nb05_ltv_box_constrained/ltv_box_constrained_lqr_benchmark.md Sec 4.3/7 -- the
    "locate the crossover, with its uncertainty" deliverable).

    NOT a calibrated confidence interval: `x_lower`/`x_upper` come from
    re-interpolating the SAME bracketing pair after shifting each gap value
    by its own +-1 standard error (a sensitivity bracket), never from a
    bootstrap or an analytic sampling distribution of the crossing point
    itself -- report it as exactly that, and only ever alongside
    `x_estimate`, never in its place. The perturbed crossing can legitimately
    fall OUTSIDE ``[x_values[bracket_index], x_values[bracket_index + 1]]``
    (a large standard error can push the perturbed line's own zero-crossing
    well past the sampled bracket) -- that is not a bug, it is the bracket
    correctly reporting "the estimate is not well pinned down here".

    Attributes:
        found: Whether a sign change was found anywhere along the swept
            points (`False` means the plan's Sec 4.3 falsifier 1 fired --
            no crossover was observed in the swept range at all).
        x_estimate: The linearly-interpolated zero-crossing of `gap`
            itself; ``None`` when `found` is ``False``.
        x_lower: The lesser of the two +-1-std-perturbed zero-crossings;
            ``None`` when `found` is ``False`` or no std was supplied.
        x_upper: The greater of the two +-1-std-perturbed zero-crossings;
            ``None`` under the same conditions as `x_lower`.
        bracket_index: The index ``i`` such that the sign change occurs
            between ``x_values[i]`` and ``x_values[i+1]``; ``None`` when
            `found` is ``False``.
    """

    found: bool
    x_estimate: float | None
    x_lower: float | None
    x_upper: float | None
    bracket_index: int | None


def _linear_zero_crossing(x0: float, y0: float, x1: float, y1: float) -> float:
    """The x-value where the line through ``(x0, y0)``-``(x1, y1)`` crosses
    ``y = 0``. Callers only invoke this once a sign change (or a
    perturbation of one) is already known, so ``y0 != y1`` always holds at
    every call site in this module.
    """
    return x0 + (0.0 - y0) * (x1 - x0) / (y1 - y0)


def estimate_crossover(
    x_values: Sequence[float],
    gap: Sequence[float],
    gap_std: Sequence[float] | None = None,
) -> CrossoverEstimate:
    """Locate the FIRST sign change of `gap` along `x_values` (assumed
    sorted ascending) and linearly interpolate its zero-crossing, with an
    optional heuristic uncertainty bracket from `gap_std`
    (docs/planning/03_studies/nb05_ltv_box_constrained/ltv_box_constrained_lqr_benchmark.md Sec 4.3/7 -- see
    `CrossoverEstimate`'s own caveat about what the bracket is NOT).

    Args:
        x_values: The swept axis (e.g. `variation_strength` or the distinct-
            slice count ``D``), ascending.
        gap: One signed value per `x_values` entry (e.g.
            ``cost[unfolded_alpha_p] - cost[unfolded_alpha]``): negative
            where the first contender wins, positive where the second does.
        gap_std: Optional standard error of `gap` at each point (e.g. the
            two contenders' own ``eval_expected_cost_std`` combined in
            quadrature by the caller); when given, `x_lower`/`x_upper` are
            also computed.

    Returns:
        The `CrossoverEstimate` for the first sign change found (a
        multiply-crossing gap reports only its earliest crossing; re-invoke
        on a truncated suffix of `x_values`/`gap`/`gap_std` for a later one).

    Raises:
        ValueError: If `x_values`/`gap` (and `gap_std`, if given) disagree
            in length, or either has fewer than 2 points.
    """
    if len(x_values) != len(gap):
        raise ValueError(
            f"x_values and gap must have the same length, got "
            f"{len(x_values)} and {len(gap)}."
        )
    if gap_std is not None and len(gap_std) != len(gap):
        raise ValueError(
            f"gap_std must match gap's length, got {len(gap_std)} and {len(gap)}."
        )
    if len(x_values) < 2:
        raise ValueError("Need at least 2 points to locate a crossing.")

    for i in range(len(gap) - 1):
        y0, y1 = gap[i], gap[i + 1]
        if (y0 <= 0.0) == (y1 <= 0.0):
            continue
        x0, x1 = x_values[i], x_values[i + 1]
        x_estimate = _linear_zero_crossing(x0, y0, x1, y1)
        x_lower = x_upper = None
        if gap_std is not None:
            s0, s1 = gap_std[i], gap_std[i + 1]
            # The two perturbed lines that widen (never narrow) the
            # bracket, regardless of the gap's sign/slope at this crossing.
            candidates = [
                _linear_zero_crossing(x0, y0 - s0, x1, y1 - s1),
                _linear_zero_crossing(x0, y0 + s0, x1, y1 + s1),
            ]
            x_lower, x_upper = min(candidates), max(candidates)
        return CrossoverEstimate(
            found=True,
            x_estimate=x_estimate,
            x_lower=x_lower,
            x_upper=x_upper,
            bracket_index=i,
        )
    return CrossoverEstimate(
        found=False, x_estimate=None, x_lower=None, x_upper=None, bracket_index=None
    )


def describe_cocp_solver_choice(
    problem: OptimalControlProblem,
    u_max: float,
    *,
    device: str,
    eps: float = 1e-8,
    max_iters: int = 10000,
) -> ResolvedCOCPSolver:
    """Which COCP solver this problem instance would resolve to, and why --
    a standalone diagnostic call (NB04 COCP refinement plan Sec 3.1's Tier-3
    narration helper), independent of actually training COCP, for a notebook
    to narrate before committing to a full run. Mirrors
    `compute_box_binding_fraction`'s own pattern: a fresh, self-contained
    check computed alongside the main pipeline, not read back from it.

    The probe point (`P_sqrt` seeded at the unconstrained DARE solution,
    `x=ones(n)`, `q=zeros(n)`) is the same deterministic, non-degenerate
    construction `COCPRecipe.build_controller` seeds a fresh controller
    with -- this reports the resolution that recipe would itself reach,
    without constructing a full controller to find out.

    Args:
        problem: The box-constrained `OptimalControlProblem` COCP would run
            against; only its linear-quadratic system/cost matrices are
            read (`models.guards.require_linear_quadratic`).
        u_max: The infinity-norm control bound COCP's QP would enforce.
        device: The torch device the resolved solver should run on --
            typically the owning `ComputeContext.device`.
        eps: Convergence tolerance forwarded to the candidate/reference specs.
        max_iters: Iteration budget forwarded to the candidate/reference specs.

    Returns:
        The `ResolvedCOCPSolver` (spec, human-readable reason, the canary's
        verdict if one ran).
    """
    system, cost = require_linear_quadratic(problem)
    A = system.A_t.array
    B = system.B_t.array
    Q_2d = np.asarray(time_invariant_slice(cost.Q))
    R_2d = np.asarray(time_invariant_slice(cost.R))
    P_sqrt_init = np.real(sqrtm(solve_discrete_are(A, B, Q_2d, R_2d)))
    n = A.shape[0]
    instance = COCPCanaryInstance(
        A=A, B=B, R=R_2d, u_max=u_max, x=np.ones(n), P_sqrt=P_sqrt_init, q=np.zeros(n)
    )
    return resolve_cocp_solver(instance, device=device, eps=eps, max_iters=max_iters)


def compute_loss_grid(
    loss_fn: Callable[[np.ndarray], float],
    param_1_range: np.ndarray,
    param_2_range: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a scalar `loss_fn(params)` (params = [p1, p2]) over every point
    of a 2D grid, for feeding into
    `viz.plots.loss_landscape.plot_loss_landscape_3d`.

    Returns:
        (param_1_grid, param_2_grid, loss_grid), each of shape
        (len(param_2_range), len(param_1_range)).
    """
    param_1_grid, param_2_grid = np.meshgrid(param_1_range, param_2_range)
    loss_grid = np.empty_like(param_1_grid, dtype=float)
    for i in range(param_1_grid.shape[0]):
        for j in range(param_1_grid.shape[1]):
            loss_grid[i, j] = loss_fn(
                np.array([param_1_grid[i, j], param_2_grid[i, j]])
            )
    return param_1_grid, param_2_grid, loss_grid
