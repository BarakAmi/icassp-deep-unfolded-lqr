"""Phase C acceptance for the exact convex policy: the family, and its isolation.

The unit's own gate is that nothing existing moves -- checked by
`tests/architecture/test_identity_stability.py`, whose golden gained two lines
for this family and changed no pre-existing one. What is checked here is that
the family exists, that its declaration reaches its identifier, and that it
computes the same policy as the cvxpylayers family it sits beside.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.linalg import solve_discrete_are, sqrtm

from mbl.applications.factories import GaussianBatchSpec, LQRProblemFactory
from mbl.applications.recipes import ExactCOCPRecipe
from mbl.applications.recipes.base import build_default_recipe_registry, null_harness
from mbl.core.runtime import Backend, ComputeContext
from mbl.engine.training_plan import OptimizerSpec, TrainingPlan
from mbl.applications.recipes.cocp import COCPRecipe
from mbl.models.constrained.cocp import COCPConfig, COCPController
from mbl.models.constrained.cocp_exact import ExactCOCPConfig, ExactCOCPController

U_MAX = 0.5


def _problem(state_dim: int = 3, control_dim: int = 2):
    return LQRProblemFactory(
        state_dim=state_dim, control_dim=control_dim, horizon=6, seed=0, u_max=U_MAX
    ).build()


def _plan(epochs: int = 1) -> TrainingPlan:
    return TrainingPlan(optimizer=OptimizerSpec("adam", 0.1), epochs=epochs)


def _ctx(device: str = "cpu") -> ComputeContext:
    return ComputeContext(backend=Backend.TORCH, device=device)


def test_the_family_is_registered_and_resolves_as_data() -> None:
    registry = build_default_recipe_registry()
    assert "cocp_exact" in registry.available()
    recipe = registry.create("cocp_exact", plan=_plan())
    assert isinstance(recipe, ExactCOCPRecipe)
    assert recipe.label == "cocp_exact"


def test_controls_are_feasible_and_certified() -> None:
    """The policy respects the box and its solve carries an optimality proof."""
    problem = _problem()
    controller = ExactCOCPRecipe(plan=_plan()).build_controller(problem, _ctx())
    states = torch.tensor(
        np.random.default_rng(0).normal(scale=0.5, size=(64, 3)), dtype=torch.float64
    )
    u = controller.get_control_policy()(0, states)
    assert float(u.detach().abs().max()) <= U_MAX + 1e-12
    certificate = controller.certificate_at(states)
    assert certificate.box_violation == 0.0
    assert certificate.worst < 1e-9


def test_the_linear_term_is_absent_unless_declared() -> None:
    """`q` is a declaration, not a constant the code picks."""
    problem = _problem()
    without = ExactCOCPRecipe(plan=_plan()).build_controller(problem, _ctx())
    with_term = ExactCOCPRecipe(plan=_plan(), use_linear_term=True).build_controller(
        problem, _ctx()
    )
    assert without.q is None
    assert with_term.q is not None
    assert "q" not in dict(without.named_parameters())
    assert "q" in dict(with_term.named_parameters())


def test_the_declaration_reaches_the_signature() -> None:
    """Two controllers differing only in the declaration are different models."""
    problem = _problem()
    without = ExactCOCPRecipe(plan=_plan()).build_controller(problem, _ctx())
    with_term = ExactCOCPRecipe(plan=_plan(), use_linear_term=True).build_controller(
        problem, _ctx()
    )
    assert without.get_signature()["use_linear_term"] is False
    assert with_term.get_signature()["use_linear_term"] is True
    assert without.get_signature() != with_term.get_signature()


def test_it_computes_the_same_policy_as_the_cvxpylayers_family() -> None:
    """Same program, different solver: the controls must agree.

    Seeded identically and compared at the same states, so any difference is
    the solver's. The cvxpylayers path is accurate at this size -- it is only
    from n = 20 upward that its reference stops being trustworthy.
    """
    problem = _problem()
    system, cost = problem.system, problem.cost
    A, B = system.A_t.array, system.B_t.array
    R = np.asarray(cost.R[0]) if np.asarray(cost.R).ndim == 3 else np.asarray(cost.R)
    Q = np.asarray(cost.Q[0]) if np.asarray(cost.Q).ndim == 3 else np.asarray(cost.Q)
    P_sqrt = np.real(sqrtm(solve_discrete_are(A, B, Q, R)))

    exact = ExactCOCPController(
        problem, problem.constraints[0], ExactCOCPConfig(P_sqrt_init=P_sqrt)
    )
    shipped = COCPController(
        problem, problem.constraints[0], COCPConfig(P_sqrt_init=P_sqrt)
    )
    states = torch.tensor(
        np.random.default_rng(1).normal(scale=0.5, size=(16, A.shape[0])),
        dtype=torch.float64,
    )
    torch.testing.assert_close(
        exact.get_control_policy()(0, states),
        shipped.get_control_policy()(0, states),
        rtol=1e-6,
        atol=1e-7,
    )


def test_the_family_declares_no_second_device_authority() -> None:
    """Unlike the cvxpylayers family, there is no `solver_device` to disagree."""
    recipe = ExactCOCPRecipe(plan=_plan())
    assert not hasattr(recipe, "solver_device")
    assert not hasattr(recipe, "solver_backend")


def test_a_smoke_training_runs_end_to_end() -> None:
    """The recipe builds an engine that trains, and the parameters move."""
    problem = _problem()
    recipe = ExactCOCPRecipe(plan=_plan(epochs=2))
    ctx = _ctx()
    controller = recipe.build_controller(problem, ctx)
    before = controller.P_sqrt.detach().clone()
    harness = null_harness(
        GaussianBatchSpec(
            state_dim=3, horizon=6, batch_size=16, seed=0, process_noise_std=0.5
        ),
        ctx,
    )
    metrics = recipe.build_engine(controller, problem, harness).train()
    assert metrics
    assert not torch.equal(before, controller.P_sqrt.detach())
    assert torch.isfinite(controller.P_sqrt).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
@pytest.mark.parametrize(
    "recipe_for",
    [
        lambda: ExactCOCPRecipe(plan=_plan()),
        lambda: _frozen_recipe(),
    ],
    ids=["trained", "frozen"],
)
def test_the_recipe_puts_the_whole_controller_on_the_declared_device(
    recipe_for,
) -> None:
    """No device hop, and no hand-holding: `build_controller` must be enough.

    **This test used to call `.to("cuda")` itself**, and so verified a condition
    it had created. The production path does not, and Figure 3 -- the campaign's
    only CUDA study -- died on its first matmul with the plant buffers on the
    CPU and the states on the card. A unit test that performs the step under
    test is not a test of that step.
    """
    problem = _problem()
    controller = recipe_for().build_controller(problem, _ctx("cuda"))

    assert controller.P_sqrt.device.type == "cuda"
    for name, buffer in controller.named_buffers():
        assert buffer.device.type == "cuda", f"buffer {name} stayed behind"

    states = torch.zeros((8, 3), dtype=controller.P_sqrt.dtype, device="cuda")
    assert controller.get_control_policy()(0, states).device.type == "cuda"


def _frozen_recipe():
    from mbl.applications.recipes import ExactCOCPLowerBoundRecipe

    return ExactCOCPLowerBoundRecipe(process_noise_std=0.5)


def test_training_tracks_the_cvxpylayers_family_step_for_step() -> None:
    """Phase E, at a scale the suite can afford.

    Both families start at the same seed, see the same batches in the same
    order, and take the same optimiser steps; the only difference is where the
    gradients come from. So the trained parameters must stay together, and a
    divergence would mean one of the two adjoints is wrong rather than that
    training is stochastic.

    Measured at the campaign's own scale (n = 4, 30 epochs, batch 2048) the
    trained costs land at 8.344001 and 8.344585 -- **+0.0070 %**, some two
    hundred times below the error bars those figures carry -- while the exact
    family trains **20x faster** in a twentieth of the memory. That comparison
    needs two minutes and lives in the campaign record, not here.
    """
    problem = _problem()
    ctx = _ctx()
    plan = _plan(epochs=3)
    batch = GaussianBatchSpec(
        state_dim=3, horizon=6, batch_size=16, seed=0, process_noise_std=0.5
    )

    trained = {}
    for name, recipe in (
        ("shipped", COCPRecipe(plan=plan)),
        # use_linear_term=True: the shipped family ALWAYS trains q, so this is
        # the comparison of two SOLVERS. Comparing formulations is a separate
        # question, and one already answered -- q moves the cost by 0.0002 %.
        ("exact", ExactCOCPRecipe(plan=plan, use_linear_term=True)),
    ):
        torch.manual_seed(0)
        controller = recipe.build_controller(problem, ctx)
        recipe.build_engine(controller, problem, null_harness(batch, ctx)).train()
        trained[name] = controller.P_sqrt.detach().clone()

    torch.testing.assert_close(
        trained["exact"], trained["shipped"], rtol=1e-4, atol=1e-6
    )
    # Anti-vacuity: the parameters must actually have moved, or the assertion
    # above compares two untouched seeds and holds for the wrong reason.
    seeded = (
        ExactCOCPRecipe(plan=plan, use_linear_term=True)
        .build_controller(problem, ctx)
        .P_sqrt.detach()
    )
    assert not torch.allclose(trained["exact"], seeded, rtol=1e-6, atol=1e-8)


# --- the frozen policy, and which side of the truth each number is on --------


def test_the_frozen_policy_is_frozen_and_seeded_by_the_bound() -> None:
    """Nothing trains, and the seed is the dual bound's cost-to-go."""
    from scipy.linalg import sqrtm as _sqrtm

    from mbl.applications.recipes import ExactCOCPLowerBoundRecipe
    from mbl.models.constrained.box_lagrangian import (
        BoxLQRData,
        infinite_horizon_box_bound,
    )

    problem = _problem()
    recipe = ExactCOCPLowerBoundRecipe(process_noise_std=0.5)
    controller = recipe.build_controller(problem, _ctx())

    assert not controller.P_sqrt.requires_grad
    assert controller.q is None

    system, cost = problem.system, problem.cost
    A = system.A_t.array
    bound = infinite_horizon_box_bound(
        BoxLQRData(
            A=A,
            B=system.B_t.array,
            Q=np.asarray(cost.Q[0] if np.asarray(cost.Q).ndim == 3 else cost.Q),
            R=np.asarray(cost.R[0] if np.asarray(cost.R).ndim == 3 else cost.R),
            W=0.25 * np.eye(A.shape[0]),
            u_max=U_MAX,
        )
    )
    assert bound.cost_to_go is not None
    np.testing.assert_allclose(
        controller.P_sqrt.detach().numpy(),
        np.real(_sqrtm(bound.cost_to_go)),
        rtol=1e-10,
        atol=1e-12,
    )


def test_the_bound_and_the_attainment_sit_either_side_of_the_truth() -> None:
    """The naming trap this family exists to avoid repeating.

    The dual value is a LOWER bound on the constrained optimum; this policy's
    attained cost is an UPPER one, because the policy is real, feasible and
    box-respecting. A caption that swaps them claims the bracket backwards, and
    the shipped family's name (`cocp_lower_bound`) names the seed rather than
    the side -- which is on the record as a misnomer.
    """
    from mbl.applications.recipes import ExactCOCPLowerBoundRecipe

    problem = _problem()
    recipe = ExactCOCPLowerBoundRecipe(process_noise_std=0.5)
    reported = recipe.provenance(problem)
    controller = recipe.build_controller(problem, _ctx())

    assert reported["dual_lower_bound_kkt_residual"] < 1e-7
    assert reported["dual_lower_bound_lmi_slack"] > -1e-8

    states = torch.tensor(
        np.random.default_rng(4).normal(scale=0.5, size=(256, 3)),
        dtype=torch.float64,
    )
    control = controller.get_control_policy()(0, states)
    assert float(control.detach().abs().max()) <= U_MAX + 1e-12
    assert controller.certificate_at(states).worst < 1e-9


def test_the_frozen_family_is_registered_and_declares_no_solver() -> None:
    from mbl.applications.recipes import ExactCOCPLowerBoundRecipe

    registry = build_default_recipe_registry()
    assert "cocp_exact_lower_bound" in registry.available()
    recipe = ExactCOCPLowerBoundRecipe(process_noise_std=0.5)
    assert not hasattr(recipe, "solver_backend")
    assert not hasattr(recipe, "solver_device")
