import numpy as np
import torch

from mbl.core.cost.quadratic_cost import QuadraticCost
from mbl.core.optimal_control_problem import OptimalControlProblem
from mbl.core.system.linear_system import LinearSystem
from mbl.models.base import Controller
from mbl.models.analytic.riccati import RiccatiController
from mbl.models.neural.nerual import NeuralConfig, NeuralPolicy
from mbl.models.unfolded.base import UnfoldedController, UnfoldingConfig
from mbl.models.iterative.initializers import ConstantInitializer
from mbl.models.iterative.step_size import StepSizeSchedule
from mbl.models.unfolded.iterative_refinement import StepSizeRefinement


def _build_problem() -> OptimalControlProblem:
    system = LinearSystem.fully_observable(np.eye(2), np.eye(2))
    cost = QuadraticCost(Q=np.eye(2), R=np.eye(2))
    return OptimalControlProblem(system=system, cost=cost)


def test_controller_protocol_is_runtime_checkable() -> None:
    assert getattr(Controller, "_is_runtime_protocol", False)


def test_riccati_controller_satisfies_controller_protocol() -> None:
    # finite_horizon_riccati requires time-stacked Q/R (it indexes Q[horizon]),
    # unlike QuadraticCost's general 2D-or-3D support used elsewhere.
    horizon = 3
    system = LinearSystem.fully_observable(np.eye(2), np.eye(2))
    cost = QuadraticCost(
        Q=np.repeat(np.eye(2)[None], horizon + 1, axis=0),
        R=np.repeat(np.eye(2)[None], horizon, axis=0),
    )
    problem = OptimalControlProblem(system=system, cost=cost)

    controller = RiccatiController(problem, horizon=horizon)
    assert isinstance(controller, Controller)


def test_neural_policy_satisfies_controller_protocol() -> None:
    problem = _build_problem()
    config = NeuralConfig(state_dim=2, control_dim=2)
    policy_net = NeuralPolicy(problem, config)
    assert isinstance(policy_net, Controller)


def test_unfolded_controller_satisfies_controller_protocol() -> None:
    problem = _build_problem()
    unfolding_config = UnfoldingConfig(
        parameters={},
        control_initializer=ConstantInitializer(
            constant=0.0, control_dim=2, dtype=torch.float32, device=torch.device("cpu")
        ),
        iterative_refinement=StepSizeRefinement(
            step_size=StepSizeSchedule(
                raw=torch.tensor(0.1, dtype=torch.float32),
                num_iterations=1,
                horizon=3,
                control_dim=2,
            ),
            num_iterations=1,
        ),
        horizon=3,
    )
    controller = UnfoldedController(problem, unfolding_config)
    assert isinstance(controller, Controller)


def test_object_missing_problem_or_config_does_not_satisfy_protocol() -> None:
    class _NotAController:
        def get_control_policy(self):
            return lambda t, y: y

    assert not isinstance(_NotAController(), Controller)


def test_object_missing_get_signature_does_not_satisfy_protocol() -> None:
    """Regression lock: get_signature is now a required Controller member,
    not an optional extra -- a Controller-shaped object missing it must no
    longer satisfy the protocol."""

    class _LegacyController:
        problem = None
        config = None

        def get_control_policy(self):
            return lambda t, y: y

    assert not isinstance(_LegacyController(), Controller)
