"""Regression tests proving UnfoldedController.forward() actually executes.

NOTE (discovered while writing these tests, out of scope to fix here):
StateSpaceSystem.run() (src/core/system/state_space_system.py) assembles its
return values with `np.stack(states/observations/controls, axis=1)`. When the
inputs are torch.Tensor (as they are for any torch-based controller, including
UnfoldedController), np.stack silently converts them to plain numpy arrays via
the tensor's __array__ protocol -- destroying the autograd graph. That means
X, Y, U (and therefore `cost`) returned by forward() are numpy arrays, not
tensors with gradients, which would block real backprop-based training of
UnfoldedController through GradientDescentStrategy. This is a materially
bigger issue than the horizon AttributeError these tests target (it lives in
the shared System.run() path, not just this one controller) and needs its own
follow-up: System.run() likely needs a framework-generic stack (numpy vs.
torch) chosen by input type, the same way compute_lqr_gradient_matrices was
made framework-generic via `.mT`/`@` instead of np.einsum.
"""

import numpy as np
import pytest
import torch

from mbl.core.cost.quadratic_cost import QuadraticCost
from mbl.core.optimal_control_problem import OptimalControlProblem
from mbl.core.system.linear_system import LinearSystem
from mbl.models.unfolded.base import UnfoldedController, UnfoldingConfig
from mbl.models.iterative.initializers import ConstantInitializer
from mbl.models.iterative.step_size import StepSizeSchedule
from mbl.models.unfolded.iterative_refinement import StepSizeRefinement
from mbl.models.unfolded.parameters import StepSizeParameter, StepSizeParameterConfig

# LinearSystem's matrices come from plain `np.eye`/`np.ones` (float64), and that
# dtype propagates through system.run's numpy/torch-mixed arithmetic -- so every
# torch tensor built in this test must also be float64, or torch's matmuls
# downstream (e.g. in StepSizeRefinement) raise a dtype-mismatch RuntimeError.
DTYPE = torch.float64


class _ConstantStepSizeParam:
    """Minimal stand-in for an UnfoldedParameter/StepSizeProvider (only needs
    .get()/.for_iteration()/.initialize()), matching the pattern already used
    in test_iterative_refinement.py. Deliberately not the real
    StepSizeParameter: keeping this test isolated to what it's actually
    regression-testing (the horizon fix), not StepSizeParameter's own
    behavior (covered by test_parameters.py/test_gradient_flow.py).
    """

    def __init__(self, value: torch.Tensor) -> None:
        self._value = value

    def get(self) -> torch.Tensor:
        return self._value

    def for_iteration(self, i: int) -> torch.Tensor:
        return self._value[i]

    def initialize(self) -> None:
        pass


def _build_controller(horizon: int = 3) -> tuple[UnfoldedController, int, int]:
    state_dim, control_dim = 2, 1
    system = LinearSystem.fully_observable(
        np.eye(state_dim), np.ones((state_dim, control_dim)) * 0.5
    )
    cost = QuadraticCost(Q=np.eye(state_dim), R=np.eye(control_dim))
    problem = OptimalControlProblem(system=system, cost=cost)

    step_size = _ConstantStepSizeParam(torch.full((2, control_dim), 0.1, dtype=DTYPE))
    control_initializer = ConstantInitializer(
        constant=0.0, control_dim=control_dim, dtype=DTYPE, device=torch.device("cpu")
    )
    iterative_refinement = StepSizeRefinement(
        step_size=step_size,
        num_iterations=2,
        static_parameters={
            "M_stack": torch.full(
                (horizon, control_dim, control_dim), 2.0, dtype=DTYPE
            ),
            "C_stack": torch.full((horizon, control_dim, state_dim), 0.5, dtype=DTYPE),
        },
    )
    config = UnfoldingConfig(
        parameters={"step_size": step_size},  # type: ignore[dict-item]  # deliberately minimal test double
        control_initializer=control_initializer,
        iterative_refinement=iterative_refinement,
        horizon=horizon,
    )

    controller = UnfoldedController(problem, config)
    controller.initialize()
    return controller, state_dim, control_dim


def test_unfolded_controller_forward_executes_with_default_noise() -> None:
    """Regression test for the AttributeError: UnfoldedController.forward used to
    read self.problem.horizon, but OptimalControlProblem has no such field --
    forward() could never execute. horizon now lives on UnfoldingConfig instead
    (mirroring RiccatiController's own explicit `horizon` constructor argument).

    This specifically exercises the *default*-noise path (process_noise=None,
    measurement_noise=None), which is the branch that needs `horizon` to build
    zero-filled noise tensors.
    """
    horizon, batch = 3, 2
    controller, state_dim, control_dim = _build_controller(horizon=horizon)

    initial_state = torch.zeros(batch, state_dim, dtype=DTYPE)

    X, Y, U, cost = controller(initial_state)

    assert X.shape == (batch, horizon + 1, state_dim)
    assert Y.shape == (
        batch,
        horizon,
        state_dim,
    )  # fully observable: obs_dim == state_dim
    assert U.shape == (batch, horizon, control_dim)
    assert cost.shape == (horizon,)
    # NOTE: cost currently comes back as a plain numpy array, not a torch.Tensor
    # with a live autograd graph -- StateSpaceSystem.run() assembles its return
    # values with np.stack(...), which silently converts torch tensors to numpy
    # (see the module docstring above). That's a separate, more significant bug
    # than the one this test targets; using np.isfinite (not torch.isfinite)
    # here reflects that current reality rather than papering over it.
    assert np.isfinite(cost).all()


def test_unfolded_controller_forward_executes_with_explicit_noise() -> None:
    horizon, batch = 3, 2
    controller, state_dim, control_dim = _build_controller(horizon=horizon)

    initial_state = torch.zeros(batch, state_dim, dtype=DTYPE)
    process_noise = torch.zeros(batch, horizon, state_dim, dtype=DTYPE)
    measurement_noise = torch.zeros(batch, horizon, state_dim, dtype=DTYPE)

    X, Y, U, cost = controller(initial_state, process_noise, measurement_noise)

    assert X.shape == (batch, horizon + 1, state_dim)
    assert U.shape == (batch, horizon, control_dim)
    assert np.isfinite(cost).all()  # see NOTE above re: numpy vs. torch.Tensor


def test_unfolding_config_rejects_non_positive_horizon() -> None:
    control_initializer = ConstantInitializer(
        constant=0.0, control_dim=1, dtype=DTYPE, device=torch.device("cpu")
    )
    iterative_refinement = StepSizeRefinement(
        step_size=StepSizeSchedule(
            raw=torch.tensor(0.1, dtype=DTYPE),
            num_iterations=1,
            horizon=1,
            control_dim=1,
        ),
        num_iterations=1,
    )

    with pytest.raises(ValueError, match="horizon"):
        UnfoldingConfig(
            parameters={},
            control_initializer=control_initializer,
            iterative_refinement=iterative_refinement,
            horizon=0,
        )


def test_unfolded_controller_get_signature_reports_type_horizon_parameters_and_problem() -> (
    None
):
    horizon, state_dim, control_dim = 3, 2, 1
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
            dtype=DTYPE,
        )
    )
    control_initializer = ConstantInitializer(
        constant=0.0, control_dim=control_dim, dtype=DTYPE, device=torch.device("cpu")
    )
    iterative_refinement = StepSizeRefinement(
        step_size=step_size,
        num_iterations=2,
        static_parameters={
            "M_stack": torch.full(
                (horizon, control_dim, control_dim), 2.0, dtype=DTYPE
            ),
            "C_stack": torch.full((horizon, control_dim, state_dim), 0.5, dtype=DTYPE),
        },
    )
    config = UnfoldingConfig(
        parameters={"step_size": step_size},
        control_initializer=control_initializer,
        iterative_refinement=iterative_refinement,
        horizon=horizon,
    )
    controller = UnfoldedController(problem, config)

    signature = controller.get_signature()

    assert signature["type"] == "UnfoldedController"
    assert signature["horizon"] == horizon
    assert signature["parameters"]["step_size"] == step_size.get_signature()
    assert signature["parameters"]["step_size"]["alpha_init"] == 0.1
    # `problem` is intentionally absent (Phase 1E): `ProblemSignatureCallback`
    # already logs it once at the root.
    assert "problem" not in signature
