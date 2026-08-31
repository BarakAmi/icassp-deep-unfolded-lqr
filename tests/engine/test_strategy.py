from unittest.mock import MagicMock

import numpy as np
import torch

from mbl.core.cost.quadratic_cost import QuadraticCost
from mbl.core.optimal_control_problem import OptimalControlProblem
from mbl.core.system.linear_system import LinearSystem
from mbl.engine.config import TrainingConfig
from mbl.engine.context import RunContext
from mbl.engine.strategy import (
    AnalyticalStrategy,
    GradientDescentStrategy,
    TrainingStrategy,
)
from mbl.models.analytic.riccati import RiccatiController
from mbl.models.unfolded.base import UnfoldedController, UnfoldingConfig
from mbl.models.iterative.initializers import ConstantInitializer
from mbl.models.unfolded.iterative_refinement import StepSizeRefinement
from mbl.models.unfolded.parameters import StepSizeParameter, StepSizeParameterConfig


def _make_context() -> RunContext:
    return RunContext(
        model=MagicMock(),
        tracker=MagicMock(),
        config=TrainingConfig(batch_size=1, learning_rate=0.01),
    )


def _make_gd_model(cost_values: list[float]) -> MagicMock:
    def side_effect(initial_state, process_noise, measurement_noise):
        cost = torch.tensor(cost_values, requires_grad=True)
        return initial_state, process_noise, measurement_noise, cost

    return MagicMock(side_effect=side_effect)


def test_gradient_descent_strategy_satisfies_training_strategy_protocol() -> None:
    strategy = GradientDescentStrategy(_make_gd_model([1.0]), MagicMock())
    assert isinstance(strategy, TrainingStrategy)


def test_gradient_descent_strategy_step_runs_full_backprop_cycle() -> None:
    model = _make_gd_model([2.0, 4.0])
    optimizer = MagicMock()
    strategy = GradientDescentStrategy(model, optimizer)
    batch = (torch.zeros(1, 2), torch.zeros(1, 1, 2), torch.zeros(1, 1, 2))

    metrics = strategy.step(_make_context(), batch)

    optimizer.zero_grad.assert_called_once()
    optimizer.step.assert_called_once()
    assert metrics == {"loss": 3.0}


def test_gradient_descent_strategy_evaluate_step_does_not_touch_optimizer() -> None:
    model = _make_gd_model([1.0, 3.0])
    optimizer = MagicMock()
    strategy = GradientDescentStrategy(model, optimizer)
    batch = (torch.zeros(1, 2), torch.zeros(1, 1, 2), torch.zeros(1, 1, 2))

    metrics = strategy.evaluate_step(_make_context(), batch)

    optimizer.zero_grad.assert_not_called()
    optimizer.step.assert_not_called()
    assert metrics == {"loss": 2.0}


def test_gradient_descent_strategy_uses_custom_loss_reduction() -> None:
    model = _make_gd_model([1.0, 2.0, 3.0])
    strategy = GradientDescentStrategy(model, MagicMock(), loss_reduction=torch.sum)
    batch = (torch.zeros(1, 2), torch.zeros(1, 1, 2), torch.zeros(1, 1, 2))

    metrics = strategy.step(_make_context(), batch)
    assert metrics == {"loss": 6.0}


def _build_riccati_controller(horizon: int = 3) -> RiccatiController:
    system = LinearSystem.fully_observable(np.eye(2), np.eye(2))
    cost = QuadraticCost(
        Q=np.repeat(np.eye(2)[None], horizon + 1, axis=0),
        R=np.repeat(np.eye(2)[None], horizon, axis=0),
    )
    problem = OptimalControlProblem(system=system, cost=cost)
    return RiccatiController(problem, horizon)


def test_analytical_strategy_satisfies_training_strategy_protocol() -> None:
    strategy = AnalyticalStrategy(_build_riccati_controller())
    assert isinstance(strategy, TrainingStrategy)


def test_analytical_strategy_never_touches_an_optimizer() -> None:
    """The whole point of AnalyticalStrategy: no optimizer exists anywhere in
    its construction or step methods, proving non-learnable controllers are
    fully supported."""
    controller = _build_riccati_controller()
    strategy = AnalyticalStrategy(controller)
    assert not hasattr(strategy, "optimizer")

    horizon = controller.horizon
    batch = (
        np.zeros((1, 2)),
        np.zeros((1, horizon, 2)),
        np.zeros((1, horizon, 2)),
    )
    metrics = strategy.step(_make_context(), batch)
    assert "cost" in metrics
    assert (
        "loss" not in metrics
    )  # non-learnable strategies aren't forced to report a "loss"


def test_analytical_strategy_step_and_evaluate_step_are_deterministic_and_equal() -> (
    None
):
    """Nothing is learned, so repeated calls (or step vs evaluate_step) on the
    same batch must return the same cost."""
    controller = _build_riccati_controller()
    strategy = AnalyticalStrategy(controller)
    horizon = controller.horizon
    batch = (
        np.ones((1, 2)),
        np.zeros((1, horizon, 2)),
        np.zeros((1, horizon, 2)),
    )

    step_metrics = strategy.step(_make_context(), batch)
    eval_metrics = strategy.evaluate_step(_make_context(), batch)
    again_metrics = strategy.step(_make_context(), batch)

    assert step_metrics == eval_metrics == again_metrics


def _build_frozen_unfolded_controller(
    horizon: int = 3, state_dim: int = 2, control_dim: int = 1
) -> UnfoldedController:
    """A torch-native controller with a frozen (requires_grad=False) step-size
    parameter -- the "no learned parameters" configuration: gradient-descent
    refinement using true, fixed step sizes, never trained."""
    dtype = torch.float64
    system = LinearSystem.fully_observable(
        np.eye(state_dim), np.ones((state_dim, control_dim)) * 0.5
    )
    cost = QuadraticCost(Q=np.eye(state_dim), R=np.eye(control_dim))
    problem = OptimalControlProblem(system=system, cost=cost)

    step_size = StepSizeParameter(
        StepSizeParameterConfig(
            name="step_size",
            num_iterations=2,
            action_dim=control_dim,
            alpha_init=0.1,
            alpha_max=1.0,
            dtype=dtype,
        )
    )
    step_size.get_raw().requires_grad_(False)
    control_initializer = ConstantInitializer(
        constant=0.0, control_dim=control_dim, dtype=dtype, device=torch.device("cpu")
    )
    iterative_refinement = StepSizeRefinement(
        step_size=step_size,
        num_iterations=2,
        static_parameters={
            "M_stack": torch.full(
                (horizon, control_dim, control_dim), 2.0, dtype=dtype
            ),
            "C_stack": torch.full((horizon, control_dim, state_dim), 0.5, dtype=dtype),
        },
    )
    config = UnfoldingConfig(
        parameters={"step_size": step_size},
        control_initializer=control_initializer,
        iterative_refinement=iterative_refinement,
        horizon=horizon,
    )
    return UnfoldedController(problem, config)


def test_analytical_strategy_supports_a_torch_native_frozen_controller() -> None:
    """Regression test: AnalyticalStrategy used to call problem.cost(X, U)
    directly, which crashed for any torch-native Controller (e.g. a "no
    learned parameters" UnfoldedController) because QuadraticCost is
    numpy-only and torch tensors carrying a live autograd graph raise on
    implicit numpy conversion. to_numpy(X)/to_numpy(U) fixes this generically,
    the same way UnfoldedController.forward() already handles it."""
    horizon, batch, state_dim = 3, 4, 2
    controller = _build_frozen_unfolded_controller(horizon=horizon, state_dim=state_dim)
    strategy = AnalyticalStrategy(controller)

    initial_state = torch.ones(batch, state_dim, dtype=torch.float64)
    process_noise = torch.zeros(batch, horizon, state_dim, dtype=torch.float64)
    measurement_noise = torch.zeros(batch, horizon, state_dim, dtype=torch.float64)
    torch_batch = (initial_state, process_noise, measurement_noise)

    metrics = strategy.step(_make_context(), torch_batch)

    assert "cost" in metrics
    assert np.isfinite(metrics["cost"])


def test_gradient_descent_strategy_declares_trains_parameters_true() -> None:
    assert GradientDescentStrategy.trains_parameters is True


def test_analytical_strategy_declares_trains_parameters_false() -> None:
    assert AnalyticalStrategy.trains_parameters is False
