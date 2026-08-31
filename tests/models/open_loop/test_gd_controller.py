"""Load-bearing end-to-end proof for Phase 2A: driving `OpenLoopGDController`
through the Runner's own epoch loop (via `GradientDescentStrategy` +
`AnalyticalGradientDescent`) must converge the open-loop control sequence
``U`` -- and its cost -- to the closed-form finite-horizon Riccati optimum,
for a solvable, non-trivial LQR instance under a deterministic (batch-1,
zero-noise) rollout.
"""

import numpy as np
import pytest
import torch
from torch import nn

from mbl.applications.rollout import RolloutModel
from mbl.core.cost.quadratic_cost import QuadraticCost
from mbl.core.optimal_control_problem import OptimalControlProblem
from mbl.core.system.linear_system import LinearSystem
from mbl.core.utils import to_numpy
from mbl.engine.callbacks import (
    ControlSequenceHistoryCallback,
    EarlyStoppingCallback,
    MetricsHistoryCallback,
    ParameterSnapshotCallback,
)
from mbl.engine.config import TrainingConfig
from mbl.engine.runner import Runner, TrainingPhase
from mbl.engine.strategy import GradientDescentStrategy
from mbl.models.analytic.riccati import RiccatiController
from mbl.models.open_loop.gd_controller import OpenLoopGDConfig, OpenLoopGDController
from mbl.models.open_loop.differentiable.optimizer import AnalyticalGradientDescent
from mbl.models.iterative.step_size import StepSizeSchedule
from mbl.models.iterative.initializers import ConstantInitializer
from mbl.persistence.local_tracker import LocalExperimentTracker

DTYPE = torch.float64
STATE_DIM, CONTROL_DIM, HORIZON = 3, 2, 6


def _build_problem() -> OptimalControlProblem:
    """A small but non-trivial (multi-state, multi-control, multi-step),
    solvable, marginally-stable LQR instance -- fixed seed, so the whole test
    is deterministic."""
    rng = np.random.default_rng(0)
    A = rng.normal(size=(STATE_DIM, STATE_DIM))
    A /= np.max(np.abs(np.linalg.eigvals(A))) * 1.5
    B = rng.normal(size=(STATE_DIM, CONTROL_DIM)) * 0.5
    system = LinearSystem.fully_observable(A, B)
    # RiccatiController's finite_horizon_riccati indexes Q[horizon]/R[k], so
    # Q/R must be genuinely time-stacked even though the cost is time-invariant.
    Q = np.repeat(np.eye(STATE_DIM)[None], HORIZON + 1, axis=0)
    R = np.repeat((0.5 * np.eye(CONTROL_DIM))[None], HORIZON, axis=0)
    cost = QuadraticCost(Q=Q, R=R)
    return OptimalControlProblem(system=system, cost=cost)


def _deterministic_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A fixed initial state, zero process/measurement noise -- the
    Part-13.1-approved deterministic rollout: with no noise, the optimal
    open-loop control sequence for this x0 coincides exactly with the
    trajectory the optimal closed-loop (Riccati) policy realizes."""
    x0 = torch.tensor([[1.0, -0.5, 0.75]], dtype=DTYPE)
    w = torch.zeros(1, HORIZON, STATE_DIM, dtype=DTYPE)
    v = torch.zeros(1, HORIZON, STATE_DIM, dtype=DTYPE)
    return x0, w, v


def _riccati_optimal_cost(
    problem: OptimalControlProblem,
    x0: torch.Tensor,
    w: torch.Tensor,
    v: torch.Tensor,
) -> float:
    """The ground-truth optimal cost: the Riccati-controlled deterministic
    rollout's full-horizon (no-terminal, time-averaged) cost -- exactly
    `QuadraticCost.__call__`'s final (k=N) entry."""
    riccati = RiccatiController(problem, HORIZON)
    policy = riccati.get_control_policy()
    X, _, U = problem.system.run(policy, to_numpy(x0), to_numpy(w), to_numpy(v))
    return float(problem.cost(X, U)[-1])


def _build_gd_stack(
    num_iterations: int, alpha: float, *, stop_tolerance: float | None = None
) -> tuple[
    OptimalControlProblem, OpenLoopGDController, RolloutModel, GradientDescentStrategy
]:
    problem = _build_problem()
    schedule = StepSizeSchedule(
        raw=torch.tensor(alpha, dtype=DTYPE),
        num_iterations=num_iterations,
        horizon=HORIZON,
        control_dim=CONTROL_DIM,
    )
    config = OpenLoopGDConfig(
        step_size=schedule,
        control_initializer=ConstantInitializer(
            0.0, control_dim=CONTROL_DIM, dtype=DTYPE, device=torch.device("cpu")
        ),
        horizon=HORIZON,
        num_iterations=num_iterations,
        control_dim=CONTROL_DIM,
        batch_size=1,
        stop_tolerance=stop_tolerance,
    )
    controller = OpenLoopGDController(problem, config, dtype=DTYPE)
    rollout = RolloutModel(controller)
    optimizer = AnalyticalGradientDescent([controller.U], schedule)
    strategy = GradientDescentStrategy(rollout, optimizer)
    return problem, controller, rollout, strategy


def test_open_loop_gd_controller_is_not_an_nn_module() -> None:
    _, controller, _, _ = _build_gd_stack(num_iterations=1, alpha=0.01)
    assert not isinstance(controller, nn.Module)


def test_open_loop_gd_converges_to_the_riccati_optimal_cost(tmp_path) -> None:
    """The load-bearing convergence proof (Part 12.1): running the GD/epoch
    loop to completion drives J^(M) (the native Runner.evaluate() eval_loss)
    to the closed-form Riccati optimum, and the per-epoch loss curve is
    monotonically non-increasing throughout."""
    num_iterations = 3000
    alpha = 0.02
    problem, controller, rollout, strategy = _build_gd_stack(num_iterations, alpha)
    x0, w, v = _deterministic_batch()
    J_star = _riccati_optimal_cost(problem, x0, w, v)

    tracker = LocalExperimentTracker(tmp_path, "gd_convergence")
    metrics_cb = MetricsHistoryCallback()
    runner = Runner(
        model=rollout,
        tracker=tracker,
        config=TrainingConfig(batch_size=1, learning_rate=1.0),
        batch_sampler=lambda: (x0, w, v),
        phases=[TrainingPhase(strategy, epochs=num_iterations, name="gd")],
        callbacks=[metrics_cb, ParameterSnapshotCallback({"U_final": controller.U})],
    )

    result = runner.run()

    assert result["eval_loss"] == pytest.approx(J_star, rel=1e-4, abs=1e-5)
    assert result["final_loss"] == pytest.approx(J_star, rel=1e-4, abs=1e-5)

    losses = [row["loss"] for row in metrics_cb._history]
    assert len(losses) == num_iterations
    non_increasing_slack = 1e-9
    assert all(
        losses[i + 1] <= losses[i] + non_increasing_slack
        for i in range(len(losses) - 1)
    )

    with np.load(tracker.artifacts_dir / "parameter_U_final.npz") as payload:
        saved_U = payload["array"]
    assert saved_U.shape == (1, HORIZON, CONTROL_DIM)


def test_control_and_metrics_history_are_aligned_and_reproducible(tmp_path) -> None:
    """Part 12.2: control_history and metrics_history have equal length,
    control_history[0] is exactly the initializer's U^(0), and recomputing
    the cost from each snapshotted U^(i) reproduces that same epoch's logged
    J^(i) -- proving the two native-callback histories describe the same
    sequence of iterates, off-by-one-free."""
    num_iterations = 50
    alpha = 0.02
    problem, controller, rollout, strategy = _build_gd_stack(num_iterations, alpha)
    x0, w, v = _deterministic_batch()

    tracker = LocalExperimentTracker(tmp_path, "gd_history_alignment")
    metrics_cb = MetricsHistoryCallback()
    control_cb = ControlSequenceHistoryCallback(controller)
    runner = Runner(
        model=rollout,
        tracker=tracker,
        config=TrainingConfig(batch_size=1, learning_rate=1.0),
        batch_sampler=lambda: (x0, w, v),
        phases=[TrainingPhase(strategy, epochs=num_iterations, name="gd")],
        callbacks=[metrics_cb, control_cb],
    )
    runner.train()

    assert len(control_cb._history) == len(metrics_cb._history) == num_iterations
    assert np.allclose(control_cb._history[0], 0.0)  # ConstantInitializer(0.0)

    for i in (0, 1, num_iterations // 2, num_iterations - 1):
        U_i = torch.tensor(control_cb._history[i], dtype=DTYPE)

        def policy_i(t: int, y: torch.Tensor, U_i: torch.Tensor = U_i) -> torch.Tensor:
            return U_i[:, t]

        X_i, _, U_i_out = problem.system.run(policy_i, x0, w, v)
        recomputed_loss = rollout._differentiable_cost(X_i, U_i_out).item()
        assert recomputed_loss == pytest.approx(
            metrics_cb._history[i]["loss"], rel=1e-9
        )


def test_early_stopping_stops_the_run_before_all_iterations(tmp_path) -> None:
    """Part 12.3: a large stop_tolerance must trigger EarlyStoppingCallback
    well before the configured num_iterations, and both native histories must
    end up the same (shortened) length -- proving stop_requested propagates
    from the callback through Runner.train's break correctly."""
    num_iterations = 500
    alpha = 0.02
    stop_tolerance = 1e-2
    problem, controller, rollout, strategy = _build_gd_stack(
        num_iterations, alpha, stop_tolerance=stop_tolerance
    )
    x0, w, v = _deterministic_batch()

    tracker = LocalExperimentTracker(tmp_path, "gd_early_stop")
    metrics_cb = MetricsHistoryCallback()
    control_cb = ControlSequenceHistoryCallback(controller)
    early_stop_cb = EarlyStoppingCallback(tolerance=stop_tolerance)
    runner = Runner(
        model=rollout,
        tracker=tracker,
        config=TrainingConfig(batch_size=1, learning_rate=1.0),
        batch_sampler=lambda: (x0, w, v),
        phases=[TrainingPhase(strategy, epochs=num_iterations, name="gd")],
        callbacks=[metrics_cb, control_cb, early_stop_cb],
    )
    runner.train()

    epochs_run = len(metrics_cb._history)
    assert 0 < epochs_run < num_iterations
    assert len(control_cb._history) == epochs_run
