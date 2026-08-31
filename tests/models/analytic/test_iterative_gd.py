"""Acceptance tests for the engine-free analytical iterative-GD LQR solver
(Phase 2.5): `AnalyticalIterativeGDController.solve()`, its pluggable
`SweepStrategy` topologies (Jacobi/Gauss-Seidel), and the shared
Riccati-derived local-gradient refinement they both drive. Written before
the implementation was accepted, per the project's verification-hook rule;
kept as the permanent regression suite.

Non-lenient parameters throughout: float64, a genuinely multi-state,
multi-control, marginally-stable instance -- never a toy 1D/1-step problem.
"""

import ast
import inspect

import numpy as np
import pytest
import torch

from mbl.applications.rollout import RolloutModel
from mbl.core.constraint.box_constraint import BoxConstraint
from mbl.core.cost.quadratic_cost import QuadraticCost
from mbl.core.optimal_control_problem import OptimalControlProblem
from mbl.core.system.linear_system import LinearSystem
from mbl.models.analytic.iterative_gd import (
    SolveSpec,
    AnalyticalIterativeGDController,
    build_riccati_gd_refinement,
)
from mbl.models.analytic.riccati import RiccatiController
from mbl.models.iterative.initializers import ConstantInitializer, SamplerInitializer
from mbl.models.iterative.step_size import StepSizeSchedule
from mbl.models.iterative.sweeps import GaussSeidelSweep, JacobiSweep
from mbl.models.samplers import (
    ConstantDistribution,
    NumpyGaussianDistribution,
    RandomSampler,
    ZeroDistribution,
    gaussian_sampler,
)

DTYPE = torch.float64
STATE_DIM, CONTROL_DIM, HORIZON = 4, 2, 30
ALPHA = 0.02
MAX_ITERS = 3000
TOLERANCE = 1e-12

_SWEEP_STRATEGIES = [JacobiSweep(), GaussSeidelSweep()]
_SWEEP_IDS = ["jacobi", "gauss_seidel"]


def _build_system_and_cost():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(STATE_DIM, STATE_DIM))
    A /= np.max(np.abs(np.linalg.eigvals(A))) * 1.2
    B = rng.normal(size=(STATE_DIM, CONTROL_DIM)) * 0.5
    system = LinearSystem.fully_observable(A, B)
    Q = np.repeat(np.eye(STATE_DIM)[None], HORIZON + 1, axis=0)
    R = np.repeat((0.5 * np.eye(CONTROL_DIM))[None], HORIZON, axis=0)
    cost = QuadraticCost(Q=Q, R=R)
    return system, cost


def _riccati_baseline(system, cost, x0_np):
    problem = OptimalControlProblem(system=system, cost=cost)
    riccati = RiccatiController(problem, HORIZON)
    x0 = torch.as_tensor(x0_np, dtype=DTYPE)
    w0 = torch.zeros(1, HORIZON, STATE_DIM, dtype=DTYPE)
    v0 = torch.zeros(1, HORIZON, STATE_DIM, dtype=DTYPE)
    _, _, U_opt = problem.system.run(riccati.get_control_policy(), x0, w0, v0)
    J_opt = float(RolloutModel(riccati)(x0, w0, v0)[3].mean())
    return U_opt.detach().numpy(), J_opt


def _make_step_size(num_iterations: int) -> StepSizeSchedule:
    return StepSizeSchedule(
        raw=torch.tensor(ALPHA, dtype=DTYPE),
        num_iterations=num_iterations,
        horizon=HORIZON,
        control_dim=CONTROL_DIM,
    )


def _solve(
    system,
    cost,
    x0_np,
    *,
    sweep_strategy=None,
    max_iters=MAX_ITERS,
    tolerance=TOLERANCE,
    control_initializer=None,
    constraints=None,
):
    controller = AnalyticalIterativeGDController(
        _make_step_size(max_iters), dtype=DTYPE
    )
    control_initializer = control_initializer or ConstantInitializer(
        0.0, CONTROL_DIM, DTYPE, torch.device("cpu")
    )
    result = controller.solve(
        OptimalControlProblem(system=system, cost=cost),
        SolveSpec(
            horizon=HORIZON,
            x0_sampler=ConstantDistribution(x0_np),
            noise_sampler=ZeroDistribution(),
            control_initializer=control_initializer,
            max_iters=max_iters,
            tolerance=tolerance,
            sweep_strategy=sweep_strategy,
            constraints=constraints,
        ),
    )
    return result, controller


@pytest.fixture
def problem_instance():
    system, cost = _build_system_and_cost()
    rng = np.random.default_rng(1)
    x0_np = rng.normal(size=(1, STATE_DIM))
    return system, cost, x0_np


# -- 1. Convergence to the Riccati optimum, both topologies -----------------


@pytest.mark.parametrize("sweep_strategy", _SWEEP_STRATEGIES, ids=_SWEEP_IDS)
def test_converges_to_riccati_optimum(problem_instance, sweep_strategy) -> None:
    system, cost, x0_np = problem_instance
    U_opt, J_opt = _riccati_baseline(system, cost, x0_np)

    result, _ = _solve(system, cost, x0_np, sweep_strategy=sweep_strategy)

    rel_err = abs(result.J_final - J_opt) / abs(J_opt)
    assert rel_err < 1e-6
    assert np.max(np.abs(result.U_final - U_opt)) < 1e-3


# -- 2. Exact history pairing: J_history[i] independently reproducible ------


@pytest.mark.parametrize("sweep_strategy", _SWEEP_STRATEGIES, ids=_SWEEP_IDS)
def test_history_pairing_is_exact(problem_instance, sweep_strategy) -> None:
    """J_history[i] must equal the INDEPENDENTLY re-scored cost of
    U_history[i] -- the load-bearing SweepResult postcondition (Gauss-Seidel
    self-consistency / Jacobi's replay), asserted per topology, never assumed."""
    system, cost, x0_np = problem_instance
    result, _ = _solve(
        system,
        cost,
        x0_np,
        sweep_strategy=sweep_strategy,
        max_iters=200,
        tolerance=None,
    )

    problem = OptimalControlProblem(system=system, cost=cost)
    x0 = torch.as_tensor(x0_np, dtype=DTYPE)
    w0 = torch.zeros(1, HORIZON, STATE_DIM, dtype=DTYPE)
    v0 = torch.zeros(1, HORIZON, STATE_DIM, dtype=DTYPE)
    Q2, R2 = cost.Q[0], cost.R[0]

    for i in (0, 1, 50, len(result.J_history) - 1):
        U_i = torch.as_tensor(result.U_history[i], dtype=DTYPE)

        def policy_i(t, y, U_i=U_i):
            return U_i[:, t]

        X_i, _, U_i_out = problem.system.run(policy_i, x0, w0, v0)
        cost_x = float(
            np.einsum("ti,ij,tj->", X_i[0, :-1].numpy(), Q2, X_i[0, :-1].numpy())
        )
        cost_u = float(
            np.einsum("ti,ij,tj->", U_i_out[0].numpy(), R2, U_i_out[0].numpy())
        )
        expected = (
            cost_x + cost_u
        ) / HORIZON  # is_time_averaged=True, no terminal (defaults)
        assert expected == pytest.approx(result.J_history[i], rel=1e-9)


# -- 3. Descent + tolerance semantics -----------------------------------------


def test_descent_and_tolerance_semantics(problem_instance) -> None:
    system, cost, x0_np = problem_instance

    result, _ = _solve(system, cost, x0_np, max_iters=200, tolerance=None)
    J = result.J_history
    assert all(J[i + 1] <= J[i] + 1e-9 for i in range(len(J) - 1))

    loose_result, _ = _solve(system, cost, x0_np, max_iters=MAX_ITERS, tolerance=1e-2)
    assert loose_result.converged
    assert loose_result.iterations_run < MAX_ITERS

    full_result, _ = _solve(system, cost, x0_np, max_iters=50, tolerance=None)
    assert full_result.iterations_run == 50
    assert not full_result.converged
    assert full_result.U_history.shape[0] == 51  # proposal row + 50 sweeps


# -- 4. Stochastic knobs: batch_size, noise, reproducibility -----------------


def test_stochastic_knobs_batch_and_reproducibility(problem_instance) -> None:
    system, cost, _ = problem_instance

    def run_once():
        controller = AnalyticalIterativeGDController(_make_step_size(50), dtype=DTYPE)
        return controller.solve(
            OptimalControlProblem(system=system, cost=cost),
            SolveSpec(
                horizon=HORIZON,
                x0_sampler=NumpyGaussianDistribution(std=1.0, seed=0),
                noise_sampler=NumpyGaussianDistribution(std=0.05, seed=1),
                control_initializer=ConstantInitializer(
                    0.0, CONTROL_DIM, DTYPE, torch.device("cpu")
                ),
                max_iters=50,
                batch_size=4,
            ),
        )

    result = run_once()
    assert result.U_history.shape == (51, 4, HORIZON, CONTROL_DIM)
    assert result.J_history.shape == (51,)
    assert result.X_final.shape == (4, HORIZON + 1, STATE_DIM)

    result2 = run_once()
    np.testing.assert_allclose(result.U_history, result2.U_history)


# -- 5. Initializer swap: changes the start, not the optimum ----------------


def test_initializer_swap_changes_start_but_not_the_optimum(problem_instance) -> None:
    system, cost, x0_np = problem_instance
    _, J_opt = _riccati_baseline(system, cost, x0_np)

    constant_result, _ = _solve(
        system,
        cost,
        x0_np,
        control_initializer=ConstantInitializer(
            0.0, CONTROL_DIM, DTYPE, torch.device("cpu")
        ),
    )

    sampler = RandomSampler("u0", gaussian_sampler)
    sampler.unify_sampler(device=torch.device("cpu"), dtype=DTYPE)
    sampler_init = SamplerInitializer(sampler, CONTROL_DIM, DTYPE, torch.device("cpu"))
    sampler_result, _ = _solve(system, cost, x0_np, control_initializer=sampler_init)

    assert not np.allclose(constant_result.U_history[0], sampler_result.U_history[0])
    assert abs(constant_result.J_final - J_opt) / abs(J_opt) < 1e-6
    assert abs(sampler_result.J_final - J_opt) / abs(J_opt) < 1e-6


# -- 6. Constraint projection, both topologies -------------------------------


@pytest.mark.parametrize("sweep_strategy", _SWEEP_STRATEGIES, ids=_SWEEP_IDS)
def test_constraint_projection_is_respected_every_iteration(
    problem_instance, sweep_strategy
) -> None:
    system, cost, x0_np = problem_instance
    u_max = 0.1
    result, _ = _solve(
        system,
        cost,
        x0_np,
        sweep_strategy=sweep_strategy,
        max_iters=500,
        tolerance=None,
        constraints=[BoxConstraint(u_max=u_max)],
    )
    assert np.all(np.abs(result.U_history) <= u_max + 1e-9)


# -- 7. Engine-freedom (architectural regression lock) -----------------------


def test_iterative_gd_module_imports_nothing_from_engine() -> None:
    """The analytical solver must never depend on the ML engine
    (Runner/TrainingConfig/TrainingStrategy) -- this test fails the suite the
    moment that decoupling silently regresses."""
    import mbl.models.analytic.iterative_gd as module

    source = inspect.getsource(module)
    tree = ast.parse(source)
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name)

    assert not any("engine" in name for name in imported_modules), imported_modules


# -- 8. Discriminating first step: Jacobi vs. Gauss-Seidel -------------------


def test_sweep_topologies_agree_at_t0_but_differ_downstream(problem_instance) -> None:
    """Jacobi's first macro-iterate is the exact simultaneous update against
    U^(0)'s own realized states X^(0); at t=0 (no earlier control to have
    been "freshly updated" yet) Gauss-Seidel's update coincides with it
    exactly, but the two topologies provably diverge from t=1 onward, since
    Gauss-Seidel's live rollout has already applied the updated u_0 -- the
    knob demonstrably does something, and does the RIGHT something."""
    system, cost, x0_np = problem_instance
    step_size = _make_step_size(1)
    control_initializer = ConstantInitializer(
        0.0, CONTROL_DIM, DTYPE, torch.device("cpu")
    )

    def run(sweep_strategy):
        controller = AnalyticalIterativeGDController(step_size, dtype=DTYPE)
        return controller.solve(
            OptimalControlProblem(system=system, cost=cost),
            SolveSpec(
                horizon=HORIZON,
                x0_sampler=ConstantDistribution(x0_np),
                noise_sampler=ZeroDistribution(),
                control_initializer=control_initializer,
                max_iters=1,
                tolerance=None,
                sweep_strategy=sweep_strategy,
            ),
        )

    jacobi_result = run(JacobiSweep())
    gs_result = run(GaussSeidelSweep())

    # Hand-compute the simultaneous update against U^(0)=0's own realized
    # states X^(0) (a pure zero-control rollout of x0).
    problem = OptimalControlProblem(system=system, cost=cost)
    refinement = build_riccati_gd_refinement(
        problem,
        step_size,
        SolveSpec(
            horizon=HORIZON,
            x0_sampler=ConstantDistribution(x0_np),
            noise_sampler=ZeroDistribution(),
            control_initializer=control_initializer,
            max_iters=1,
        ),
        dtype=DTYPE,
    )
    x0 = torch.as_tensor(x0_np, dtype=DTYPE)
    w0 = torch.zeros(1, HORIZON, STATE_DIM, dtype=DTYPE)
    v0 = torch.zeros(1, HORIZON, STATE_DIM, dtype=DTYPE)
    U0 = torch.zeros(1, HORIZON, CONTROL_DIM, dtype=DTYPE)
    X0, _, _ = problem.system.run(lambda t, y: U0[:, t], x0, w0, v0)

    expected = []
    for k in range(HORIZON):
        M, y_Ct = refinement.pre_iteration_hook(k, X0[:, k])
        expected.append(U0[:, k] - ALPHA * (U0[:, k] @ M + y_Ct))
    expected_simultaneous_U1 = torch.stack(expected, dim=1).numpy()

    jacobi_U1 = jacobi_result.U_history[1]
    gs_U1 = gs_result.U_history[1]

    # Jacobi IS the simultaneous formula (its "replay" policy ignores the
    # live state and returns the precomputed U_next verbatim).
    np.testing.assert_allclose(jacobi_U1, expected_simultaneous_U1, atol=1e-10)
    # Gauss-Seidel agrees at t=0 only (same x_0, nothing "fresh" yet there)...
    np.testing.assert_allclose(gs_U1[:, 0], expected_simultaneous_U1[:, 0], atol=1e-10)
    # ...and provably diverges once B != 0 lets an updated u_0 move x_1.
    assert not np.allclose(gs_U1, jacobi_U1, atol=1e-9)


# -- 9. Shared fixed point ----------------------------------------------------


def test_both_topologies_converge_to_the_same_fixed_point(problem_instance) -> None:
    system, cost, x0_np = problem_instance
    jacobi_result, _ = _solve(system, cost, x0_np, sweep_strategy=JacobiSweep())
    gs_result, _ = _solve(system, cost, x0_np, sweep_strategy=GaussSeidelSweep())

    assert np.max(np.abs(jacobi_result.U_final - gs_result.U_final)) < 1e-4


# -- 10. Compute symmetry: one rollout call per macro-iteration, both -------


@pytest.mark.parametrize("sweep_strategy_cls", [JacobiSweep, GaussSeidelSweep])
def test_one_rollout_call_per_macro_iteration(
    problem_instance, sweep_strategy_cls
) -> None:
    """Fairness property (Phase 2.5 addendum Part A2): both topologies must
    cost exactly one `system.run` call per macro-iteration, so notebook
    convergence comparisons are fair in wall-clock too, not just iteration
    count. Verified with a spy wrapping `LinearSystem.run`."""
    system, cost, x0_np = problem_instance
    call_count = 0
    original_run = LinearSystem.run

    def spy_run(self, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_run(self, *args, **kwargs)

    max_iters = 20
    LinearSystem.run = spy_run  # type: ignore[method-assign]  # deliberate spy patch
    try:
        _solve(
            system,
            cost,
            x0_np,
            sweep_strategy=sweep_strategy_cls(),
            max_iters=max_iters,
            tolerance=None,
        )
    finally:
        LinearSystem.run = original_run  # type: ignore[method-assign]  # restore the spy patch

    # +1 for the strategy-independent proposal sweep.
    assert call_count == max_iters + 1


# -- 11. Provenance: signature records the injected sweep strategy ----------


@pytest.mark.parametrize(
    "sweep_strategy,expected_type",
    list(zip(_SWEEP_STRATEGIES, ["JacobiSweep", "GaussSeidelSweep"])),
    ids=_SWEEP_IDS,
)
def test_signature_records_the_injected_sweep_strategy(
    problem_instance, sweep_strategy, expected_type
) -> None:
    system, cost, x0_np = problem_instance
    result, _ = _solve(system, cost, x0_np, sweep_strategy=sweep_strategy, max_iters=5)

    assert result.signature["sweep_strategy"] == {"type": expected_type}
    assert result.signature["control_initializer"]["type"] == "ConstantInitializer"
