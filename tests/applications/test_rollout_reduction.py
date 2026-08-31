"""`loss_reduction` must govern the gradient it claims to govern.

`RolloutModel.forward` returned `total_quadratic_cost(..., reduction=BATCH_MEAN)`
-- a 0-dimensional tensor -- and `GradientDescentStrategy` then applied the
declared `loss_reduction` to it. `torch.mean` and `torch.sum` are both the
identity on a scalar, so **`sum` and `mean` trained identically** while
`loss_reduction` was signed into `TrainingPlan.get_signature()`: two studies
differing only in that word derived different `ModelID`s and produced the same
model. The inverse of the defect G-1 fixed, where a *presentation* field split
identity; here a *training* field split identity while changing nothing.

Annex 06 §4.3's correction of 2026-08-04 records it, the author settled it
("make it execute"), and these are the checks that had to fail first. They are
also the precondition for the microbatch executor: chunk weighting is a
consequence of the reduction's semantics, so the reduction has to mean
something before a chunk can be weighted by it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mbl.applications.rollout import RolloutModel
from mbl.core.cost.quadratic_cost import QuadraticCost
from mbl.core.kernels.quadratic import CostReduction, total_quadratic_cost
from mbl.core.optimal_control_problem import OptimalControlProblem
from mbl.core.system.linear_system import LinearSystem
from mbl.engine.strategy import GradientDescentStrategy
from mbl.engine.context import RunContext
from mbl.engine.config import TrainingConfig
from mbl.core.runtime.compute_context import ComputeContext
from mbl.experiments.bindings import DEFAULT_SPEC_BINDINGS

BATCH = 8
HORIZON = 4
STATE_DIM = 3
CONTROL_DIM = 2
DTYPE = torch.float64


def _problem() -> OptimalControlProblem:
    rng = np.random.default_rng(0)
    a = rng.normal(size=(STATE_DIM, STATE_DIM)) * 0.2 + np.eye(STATE_DIM)
    b = rng.normal(size=(STATE_DIM, CONTROL_DIM)) * 0.5
    # Time-stacked Q/R: the unfolded families resolve a Riccati reference and
    # require one slice per step (`models/guards.py::require_linear_quadratic`).
    return OptimalControlProblem(
        system=LinearSystem.fully_observable(a, b),
        cost=QuadraticCost(
            Q=np.repeat(np.eye(STATE_DIM)[None], HORIZON + 1, axis=0),
            R=np.repeat((np.eye(CONTROL_DIM) * 0.1)[None], HORIZON, axis=0),
        ),
    )


def _trainable_rollout() -> RolloutModel:
    """A rollout whose parameters actually receive a gradient.

    Built through the registry rather than by hand: that is the surface a
    study author edits, so a family whose construction changes breaks this
    loudly instead of leaving it testing a shape nothing produces.
    """
    problem = _problem()
    context = {"state_dim": STATE_DIM, "horizon": HORIZON}
    plan = DEFAULT_SPEC_BINDINGS.build(
        "end_to_end",
        {"optimizer": "adam", "learning_rate": 0.01, "epochs": 1},
        context,
    )
    recipe = DEFAULT_SPEC_BINDINGS.registry.create(
        "unfolded",
        kind="learned_step_size",
        plan=plan,
        num_iterations=2,
        step_size_init=0.05,
        step_size_max=1.0,
        horizon=HORIZON,
    )
    ctx = ComputeContext(backend="torch", device="cpu", precision="float64")
    return RolloutModel(recipe.build_controller(problem, ctx))


def _batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(11)
    return (
        torch.randn(BATCH, STATE_DIM, generator=g, dtype=DTYPE),
        torch.randn(BATCH, HORIZON, STATE_DIM, generator=g, dtype=DTYPE) * 0.3,
        torch.zeros(BATCH, HORIZON, STATE_DIM, dtype=DTYPE),
    )


def _context() -> RunContext:
    return RunContext(
        model=torch.nn.Module(),
        tracker=None,  # step() never touches it
        config=TrainingConfig(batch_size=BATCH, learning_rate=0.01),
    )


# -- 1. the shape the seam needs ---------------------------------------------


def test_the_rollout_returns_one_cost_per_trajectory() -> None:
    """The reduction is the caller's decision, so the rollout may not make it.

    A 0-d cost is a reduction already performed, and it is what made
    `loss_reduction` inert.
    """
    _, _, _, cost = _trainable_rollout()(*_batch())

    assert cost.shape == (BATCH,), (
        f"expected one cost per trajectory, got shape {tuple(cost.shape)}; "
        "a pre-reduced cost makes the declared loss_reduction a no-op"
    )


def test_the_per_trajectory_costs_are_all_distinct() -> None:
    """Guards the degenerate pass: a `(BATCH,)` of one repeated value would
    satisfy the shape check while carrying no per-trajectory information."""
    _, _, _, cost = _trainable_rollout()(*_batch())

    assert len(torch.unique(cost)) == BATCH


# -- 2. the bridge to the value the store was built on -----------------------


def test_the_mean_still_agrees_with_the_batch_mean_kernel() -> None:
    """`mean(PER_SAMPLE)` and `BATCH_MEAN` are documented as distinct float
    association orders, so this pins how far apart they are allowed to drift.

    Measured before the change at 1.34 ULP across batch sizes 8..8192 in both
    float32 and float64; asserted here at four, which fails on a real
    divergence and not on association order.
    """
    rollout = _trainable_rollout()
    initial_state, process_noise, measurement_noise = _batch()
    X, _, U, per_sample = rollout(initial_state, process_noise, measurement_noise)

    batch_mean = total_quadratic_cost(
        rollout._Q.to(dtype=X.dtype),
        rollout._R.to(dtype=U.dtype),
        X,
        U,
        conventions=rollout._conventions,
        reduction=CostReduction.BATCH_MEAN,
    )

    assert torch.allclose(
        per_sample.mean(), batch_mean, rtol=4 * torch.finfo(DTYPE).eps, atol=0.0
    )


# -- 3. the defect itself ----------------------------------------------------


def test_sum_and_mean_report_different_losses() -> None:
    """The declared word has to reach the number it names."""
    batch = _batch()
    mean_loss = GradientDescentStrategy(
        _trainable_rollout(), torch.optim.SGD([torch.zeros(1)], lr=0.0)
    ).evaluate_step(_context(), batch)["loss"]
    sum_loss = GradientDescentStrategy(
        _trainable_rollout(),
        torch.optim.SGD([torch.zeros(1)], lr=0.0),
        loss_reduction=torch.sum,
    ).evaluate_step(_context(), batch)["loss"]

    assert sum_loss == pytest.approx(mean_loss * BATCH, rel=1e-12)


def test_sum_and_mean_produce_different_gradients() -> None:
    """The load-bearing one, and the reason a loss comparison is not enough.

    Two implementations can agree on a reported loss and disagree on the
    gradient; identity is derived from the specification, so a `loss_reduction`
    that does not reach `p.grad` splits `ModelID` without splitting the model.
    """
    batch = _batch()
    grads: dict[str, torch.Tensor] = {}
    for name, reduction in (("mean", torch.mean), ("sum", torch.sum)):
        rollout = _trainable_rollout()
        module = rollout.controller.as_module()
        optimizer = torch.optim.SGD(module.parameters(), lr=0.0)
        GradientDescentStrategy(rollout, optimizer, loss_reduction=reduction).step(
            _context(), batch
        )
        grads[name] = torch.cat(
            [
                p.grad.reshape(-1).clone()
                for p in module.parameters()
                if p.grad is not None
            ]
        )

    assert grads["mean"].numel() > 0, (
        "the fixture trains nothing; the test proves nothing"
    )
    assert not torch.allclose(grads["mean"], grads["sum"]), (
        "sum and mean produced the same gradient: the declared reduction is "
        "inert, and two studies differing only in that word derive different "
        "ModelIDs for one model"
    )
    assert torch.allclose(grads["sum"], grads["mean"] * BATCH, rtol=1e-10)
