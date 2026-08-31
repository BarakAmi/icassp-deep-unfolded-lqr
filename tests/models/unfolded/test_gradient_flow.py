"""Proves the autograd-severing bug is actually fixed: a genuinely learnable
(requires_grad=True) StepSizeParameter can flow gradients all the way through
LinearSystem's per-step maps and StateSpaceSystem.run()'s trajectory stacking,
back to a non-None .grad on the parameter -- which is exactly what
GradientDescentStrategy needs to make real end-to-end training of
UnfoldedController possible.

Uses a dummy cost computed directly from the rollout's control trajectory U
(not problem.cost(...)/QuadraticCost) since QuadraticCost is numpy-only -- a
separate, still-tracked issue -- and this test is specifically isolating
System.run()'s own autograd behavior from that unrelated limitation.
"""

import numpy as np
import torch

from mbl.core.cost.quadratic_cost import QuadraticCost
from mbl.core.optimal_control_problem import OptimalControlProblem
from mbl.core.system.linear_system import LinearSystem
from mbl.models.unfolded.base import UnfoldedController, UnfoldingConfig
from mbl.models.iterative.initializers import ConstantInitializer
from mbl.models.unfolded.iterative_refinement import StepSizeRefinement
from mbl.models.unfolded.parameters import StepSizeParameter, StepSizeParameterConfig

DTYPE = torch.float64


def _build_controller_with_real_step_size(
    horizon: int, state_dim: int, control_dim: int
):
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
    return controller, problem, step_size


def test_step_size_parameter_initializes_with_a_genuine_learnable_tensor() -> None:
    """Regression test for the torch.clip(float, float, float) TypeError: this
    used to make StepSizeParameter impossible to instantiate at all."""
    step_size = StepSizeParameter(
        StepSizeParameterConfig(
            name="step_size",
            num_iterations=3,
            action_dim=1,
            alpha_init=0.2,
            alpha_max=1.0,
            dtype=DTYPE,
        )
    )
    assert isinstance(step_size.rho, torch.nn.Parameter)
    assert step_size.rho.requires_grad
    assert step_size.rho.grad is None  # nothing has run backward() yet

    # sigmoid(rho) * alpha_max should reconstruct alpha_init (that's the point
    # of this initialization scheme).
    reconstructed_alpha = torch.sigmoid(step_size.rho) * step_size.config.alpha_max
    assert torch.allclose(
        reconstructed_alpha, torch.full_like(reconstructed_alpha, 0.2), atol=1e-6
    )


def test_system_run_preserves_autograd_graph_through_full_unfolded_rollout() -> None:
    """The critical proof: instantiate an UnfoldedController with a real,
    requires_grad=True learnable parameter, run a full rollout via
    System.run(), compute a dummy cost from the result, and successfully
    execute .backward() -- with the resulting gradient not None.
    """
    horizon, batch, state_dim, control_dim = 3, 2, 2, 1
    controller, problem, step_size = _build_controller_with_real_step_size(
        horizon, state_dim, control_dim
    )

    # Non-zero initial state: with an all-zero state/noise, the observation
    # stays y=0 throughout, so the LQR gradient term (u @ M + y_Ct) is
    # identically zero regardless of the step-size -- d(u_final)/d(alpha)
    # would be mathematically, correctly, zero, which would look identical to
    # (but isn't) the autograd graph being broken. Non-zero state avoids that
    # degenerate case.
    initial_state = torch.ones(batch, state_dim, dtype=DTYPE)
    process_noise = torch.zeros(batch, horizon, state_dim, dtype=DTYPE)
    measurement_noise = torch.zeros(batch, horizon, state_dim, dtype=DTYPE)

    control_policy = controller.get_control_policy()
    X, Y, U = problem.system.run(
        policy=control_policy,
        initial_state=initial_state,
        process_noises=process_noise,
        measurement_noises=measurement_noise,
    )

    # The graph must have survived System.run() end to end -- not silently
    # detached into numpy by np.stack, and not crashed by numpy/torch mixing
    # inside LinearSystem's per-step maps.
    assert isinstance(X, torch.Tensor)
    assert isinstance(Y, torch.Tensor)
    assert isinstance(U, torch.Tensor)
    assert U.requires_grad
    assert X.requires_grad  # states from t=1 onward depend on the learnable control

    dummy_cost = (U**2).sum()
    dummy_cost.backward()

    assert step_size.rho.grad is not None
    assert torch.isfinite(step_size.rho.grad).all()
    assert (step_size.rho.grad != 0).any()


def test_gradient_descent_step_actually_updates_the_learnable_parameter() -> None:
    """End-to-end confidence check: a real optimizer.step() on this real
    parameter, driven by a dummy loss computed from a real System.run()
    rollout, must change the parameter's value."""
    horizon, batch, state_dim, control_dim = 3, 2, 2, 1
    controller, problem, step_size = _build_controller_with_real_step_size(
        horizon, state_dim, control_dim
    )
    optimizer = torch.optim.SGD([step_size.rho], lr=0.1)

    # Non-zero initial state: with an all-zero state/noise, the observation
    # stays y=0 throughout, so the LQR gradient term (u @ M + y_Ct) is
    # identically zero regardless of the step-size -- d(u_final)/d(alpha)
    # would be mathematically, correctly, zero, which would look identical to
    # (but isn't) the autograd graph being broken. Non-zero state avoids that
    # degenerate case.
    initial_state = torch.ones(batch, state_dim, dtype=DTYPE)
    process_noise = torch.zeros(batch, horizon, state_dim, dtype=DTYPE)
    measurement_noise = torch.zeros(batch, horizon, state_dim, dtype=DTYPE)

    rho_before = step_size.rho.detach().clone()

    control_policy = controller.get_control_policy()
    _, _, U = problem.system.run(
        policy=control_policy,
        initial_state=initial_state,
        process_noises=process_noise,
        measurement_noises=measurement_noise,
    )
    optimizer.zero_grad()
    (U**2).sum().backward()
    optimizer.step()

    assert not torch.equal(step_size.rho.detach(), rho_before)
