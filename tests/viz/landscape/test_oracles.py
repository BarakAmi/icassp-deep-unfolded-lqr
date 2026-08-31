"""The load-bearing consistency proof for the visualization toolkit's
`CostOracle` seam: `make_rollout_cost_oracle` must (a) preserve the batch
axis (the "scalar collapse hazard" `RolloutModel.forward`'s own reduction
would cause if used directly on a grid batch), and (b) average to exactly
`RolloutModel`'s own scalar cost for an identical batch of reference
controls -- the property `grids.GridProjector`'s paraboloid recovery
(`test_grids.py`) and every landscape asset ultimately depend on.
"""

import numpy as np
import pytest
import torch

from mbl.applications.rollout import RolloutModel
from mbl.viz.landscape.oracles import (
    make_local_cost_to_go_oracle,
    make_rollout_cost_oracle,
)
from mbl.core.cost.quadratic_cost import QuadraticCost
from mbl.core.optimal_control_problem import OptimalControlProblem
from mbl.core.system.linear_system import LinearSystem
from mbl.models.analytic.riccati import (
    LocalCostToGoModel,
    evaluate_local_cost_to_go,
    finite_horizon_riccati,
)

DTYPE = torch.float64
STATE_DIM, CONTROL_DIM, HORIZON = 3, 2, 6


def _build_problem(
    *, include_terminal_cost=False, is_time_averaged=True
) -> OptimalControlProblem:
    rng = np.random.default_rng(0)
    A = rng.normal(size=(STATE_DIM, STATE_DIM))
    A /= np.max(np.abs(np.linalg.eigvals(A))) * 1.5
    B = rng.normal(size=(STATE_DIM, CONTROL_DIM)) * 0.5
    system = LinearSystem.fully_observable(A, B)
    Q = np.repeat(np.eye(STATE_DIM)[None], HORIZON + 1, axis=0)
    R = np.repeat((0.5 * np.eye(CONTROL_DIM))[None], HORIZON, axis=0)
    cost = QuadraticCost(
        Q=Q,
        R=R,
        include_terminal_cost=include_terminal_cost,
        is_time_averaged=is_time_averaged,
    )
    return OptimalControlProblem(system=system, cost=cost)


def _reference_batch():
    x0 = torch.tensor([[1.0, -0.5, 0.75]], dtype=DTYPE)
    w = torch.zeros(1, HORIZON, STATE_DIM, dtype=DTYPE)
    v = torch.zeros(1, HORIZON, STATE_DIM, dtype=DTYPE)
    return x0, w, v


def _random_controls(batch: int, generator: torch.Generator) -> torch.Tensor:
    return torch.randn(batch, HORIZON, CONTROL_DIM, generator=generator, dtype=DTYPE)


def test_oracle_output_preserves_the_batch_axis() -> None:
    """The scalar-collapse hazard: RolloutModel.forward's own cost reduces
    over both batch and time to ONE number. The oracle must not."""
    problem = _build_problem()
    oracle = make_rollout_cost_oracle(problem, _reference_batch(), dtype=DTYPE)
    generator = torch.Generator().manual_seed(0)
    U_batch = _random_controls(batch=7, generator=generator)

    costs = oracle(U_batch)

    assert costs.shape == (7,)


def test_oracle_gives_distinct_costs_for_distinct_controls() -> None:
    problem = _build_problem()
    oracle = make_rollout_cost_oracle(problem, _reference_batch(), dtype=DTYPE)
    generator = torch.Generator().manual_seed(1)
    U_batch = _random_controls(batch=5, generator=generator)

    costs = oracle(U_batch)

    assert len(torch.unique(costs)) == 5


def test_oracle_mean_matches_rollout_model_scalar_for_identical_batch() -> None:
    """The consistency proof (Phase 2B plan Part 1.2 / Part 8): with the
    default cost flags (include_terminal_cost=False, is_time_averaged=True),
    the oracle's per-sample output for a batch of IDENTICAL reference
    controls averages to exactly RolloutModel.forward's own scalar cost for
    that same control -- since a per-sample-then-batch-mean reduction over a
    rectangular grid of identical rows is mathematically the same
    computation RolloutModel performs directly (mean over batch AND time
    collapsed at once)."""
    problem = _build_problem()
    x0, w, v = _reference_batch()
    oracle = make_rollout_cost_oracle(problem, (x0, w, v), dtype=DTYPE)

    generator = torch.Generator().manual_seed(2)
    U_ref = torch.randn(1, HORIZON, CONTROL_DIM, generator=generator, dtype=DTYPE)
    U_batch = U_ref.repeat(9, 1, 1)

    oracle_costs = oracle(U_batch)
    assert torch.allclose(oracle_costs, oracle_costs[0].expand_as(oracle_costs))

    class _FixedControlController:
        def __init__(self, problem, U):
            self.problem = problem
            self.U = U

        def get_control_policy(self):
            return lambda t, y: self.U[:, t]

    rollout = RolloutModel(_FixedControlController(problem, U_ref))
    with torch.no_grad():
        _, _, _, rollout_cost = rollout(x0, w, v)

    assert torch.allclose(oracle_costs.mean(), rollout_cost, rtol=1e-10, atol=1e-12)


def test_oracle_runs_under_no_grad_even_if_input_requires_grad() -> None:
    problem = _build_problem()
    oracle = make_rollout_cost_oracle(problem, _reference_batch(), dtype=DTYPE)
    U_batch = torch.zeros(3, HORIZON, CONTROL_DIM, dtype=DTYPE, requires_grad=True)

    costs = oracle(U_batch)

    assert not costs.requires_grad


def test_oracle_broadcasts_a_batch1_reference_across_the_grid_batch() -> None:
    """The reference (x0, w, v) is always batch-1 (Phase 2A Decision 13.1);
    the oracle must broadcast it to whatever batch size the grid supplies,
    not require a matching batch size from the caller."""
    problem = _build_problem()
    oracle = make_rollout_cost_oracle(problem, _reference_batch(), dtype=DTYPE)

    for batch in (1, 4, 100):
        costs = oracle(torch.zeros(batch, HORIZON, CONTROL_DIM, dtype=DTYPE))
        assert costs.shape == (batch,)


def test_oracle_mirrors_include_terminal_cost_flag() -> None:
    """Enabling include_terminal_cost must strictly increase the reported
    cost relative to the same control scored without it (the added terminal
    term is a genuine positive quadratic form for a nonzero terminal state)."""
    x0, w, v = _reference_batch()
    U = torch.randn(
        1, HORIZON, CONTROL_DIM, generator=torch.Generator().manual_seed(3), dtype=DTYPE
    )

    problem_no_terminal = _build_problem(include_terminal_cost=False)
    problem_with_terminal = _build_problem(include_terminal_cost=True)
    oracle_no_terminal = make_rollout_cost_oracle(
        problem_no_terminal, (x0, w, v), dtype=DTYPE
    )
    oracle_with_terminal = make_rollout_cost_oracle(
        problem_with_terminal, (x0, w, v), dtype=DTYPE
    )

    cost_no_terminal = oracle_no_terminal(U).item()
    cost_with_terminal = oracle_with_terminal(U).item()

    assert cost_with_terminal > cost_no_terminal


def test_oracle_mirrors_is_time_averaged_flag() -> None:
    """is_time_averaged=False must scale the reported cost up by exactly
    HORIZON relative to the averaged version (same stage-cost sum, divided
    vs. not divided by the horizon)."""
    x0, w, v = _reference_batch()
    U = torch.randn(
        1, HORIZON, CONTROL_DIM, generator=torch.Generator().manual_seed(4), dtype=DTYPE
    )

    problem_averaged = _build_problem(is_time_averaged=True)
    problem_summed = _build_problem(is_time_averaged=False)
    oracle_averaged = make_rollout_cost_oracle(
        problem_averaged, (x0, w, v), dtype=DTYPE
    )
    oracle_summed = make_rollout_cost_oracle(problem_summed, (x0, w, v), dtype=DTYPE)

    cost_averaged = oracle_averaged(U).item()
    cost_summed = oracle_summed(U).item()

    assert cost_summed == pytest.approx(cost_averaged * HORIZON, rel=1e-9)


def _local_cost_to_go_setup(timestep: int = 2):
    """A concrete (system, cost, Riccati) instance plus the frozen state and
    per-timestep matrices `make_local_cost_to_go_oracle` needs."""
    problem = _build_problem()
    P_arr, _ = finite_horizon_riccati(problem.system, problem.cost, HORIZON)
    x_star = np.array([0.6, -0.3, 0.9])
    model = LocalCostToGoModel(
        A=problem.system.A_t[timestep],
        B=problem.system.B_t[timestep],
        Q=problem.cost.Q[0] if problem.cost.Q.ndim == 3 else problem.cost.Q,
        R=problem.cost.R[0] if problem.cost.R.ndim == 3 else problem.cost.R,
        P_next=P_arr[timestep + 1],
        process_noise_cov=0.2 * np.eye(STATE_DIM),
    )
    return x_star, model


def test_local_cost_to_go_oracle_matches_direct_evaluation() -> None:
    """The oracle wraps `evaluate_local_cost_to_go` verbatim: scoring a
    (batch, T, m) grid row must equal calling the closed-form function
    directly on that row's `timestep` slice."""
    timestep = 2
    x_star, model = _local_cost_to_go_setup(timestep)
    oracle = make_local_cost_to_go_oracle(x_star, model=model, timestep=timestep)

    generator = torch.Generator().manual_seed(0)
    U_batch = torch.randn(6, HORIZON, CONTROL_DIM, generator=generator, dtype=DTYPE)

    oracle_costs = oracle(U_batch).numpy()
    direct_costs = evaluate_local_cost_to_go(
        x_star, U_batch[:, timestep, :].numpy(), model
    )

    assert np.allclose(oracle_costs, direct_costs)


def test_local_cost_to_go_oracle_ignores_every_other_timestep_entry() -> None:
    """The property `grids.GridProjector` relies on for correctness here:
    the local Bellman cost-to-go depends only on the frozen state and the
    `timestep` slice, so perturbing every OTHER (t, j) entry of a candidate
    row must leave the reported cost unchanged."""
    timestep = 2
    x_star, model = _local_cost_to_go_setup(timestep)
    oracle = make_local_cost_to_go_oracle(x_star, model=model, timestep=timestep)

    generator = torch.Generator().manual_seed(1)
    U_batch = torch.randn(4, HORIZON, CONTROL_DIM, generator=generator, dtype=DTYPE)
    U_perturbed = U_batch.clone()
    perturbation = torch.randn(
        4, HORIZON, CONTROL_DIM, generator=generator, dtype=DTYPE
    )
    perturbation[:, timestep, :] = 0.0  # never touch the timestep that matters
    U_perturbed = U_perturbed + perturbation

    assert torch.allclose(oracle(U_batch), oracle(U_perturbed))


def test_local_cost_to_go_oracle_preserves_the_batch_axis() -> None:
    timestep = 2
    x_star, model = _local_cost_to_go_setup(timestep)
    oracle = make_local_cost_to_go_oracle(x_star, model=model, timestep=timestep)

    costs = oracle(torch.zeros(9, HORIZON, CONTROL_DIM, dtype=DTYPE))

    assert costs.shape == (9,)
