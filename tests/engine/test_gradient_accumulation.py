"""One effective batch, executed in chunks, must give the unchunked gradient.

Annex 06 §4.3 and D20: the effective batch size is scientific content and
participates in `ModelID`; the microbatch is a resource decision chosen from
available memory, recorded in provenance and **excluded** from identity. The
chunking itself was never built -- `BatchPlan.microbatch` was parsed, signed out
of identity, and refused by the producer, because a declaration that changes
nothing is reported in provenance as one that was honoured.

Everything here tests one statement, not two rules:

    For any declared reduction and any microbatch, the accumulated gradient
    equals the gradient of THE DECLARED LOSS over the whole effective batch.

Under `mean` that is `L = sum_c (n_c/N) * mean_c`; under `sum` it is
`sum_c sum_c` with no rescaling. The uniform `1/K` weighting §4.3's prose
suggests is right only when the microbatch divides the batch, and
`test_a_uniform_one_over_k_weighting_is_refused` exists to fail against it.

**Gradients are compared, never losses.** Two implementations can agree on a
reported loss and disagree on `p.grad`, and it is `p.grad` that trains a model.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mbl.applications.rollout import RolloutModel
from mbl.core.cost.quadratic_cost import QuadraticCost
from mbl.core.optimal_control_problem import OptimalControlProblem
from mbl.core.runtime.compute_context import ComputeContext
from mbl.core.system.linear_system import LinearSystem
from mbl.engine.config import TrainingConfig
from mbl.engine.context import RunContext
from mbl.engine.strategy import GradientDescentStrategy
from mbl.experiments.bindings import DEFAULT_SPEC_BINDINGS

HORIZON = 4
STATE_DIM = 3
CONTROL_DIM = 2
DTYPE = torch.float64


def _problem() -> OptimalControlProblem:
    rng = np.random.default_rng(0)
    a = rng.normal(size=(STATE_DIM, STATE_DIM)) * 0.2 + np.eye(STATE_DIM)
    b = rng.normal(size=(STATE_DIM, CONTROL_DIM)) * 0.5
    return OptimalControlProblem(
        system=LinearSystem.fully_observable(a, b),
        cost=QuadraticCost(
            Q=np.repeat(np.eye(STATE_DIM)[None], HORIZON + 1, axis=0),
            R=np.repeat((np.eye(CONTROL_DIM) * 0.1)[None], HORIZON, axis=0),
        ),
    )


def _rollout() -> RolloutModel:
    """Built through the registry -- the surface a study author edits."""
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
    return RolloutModel(recipe.build_controller(_problem(), ctx))


def _batch(size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(11)
    return (
        torch.randn(size, STATE_DIM, generator=g, dtype=DTYPE),
        torch.randn(size, HORIZON, STATE_DIM, generator=g, dtype=DTYPE) * 0.3,
        torch.zeros(size, HORIZON, STATE_DIM, dtype=DTYPE),
    )


def _context(size: int) -> RunContext:
    return RunContext(
        model=torch.nn.Module(),
        tracker=None,
        config=TrainingConfig(batch_size=size, learning_rate=0.01),
    )


def _gradient_after_one_step(
    size: int,
    *,
    microbatch: int | None,
    reduction=torch.mean,
    gradient_clip_norm: float | None = None,
) -> tuple[torch.Tensor, float]:
    """One `step` at `lr = 0`, returning the flattened gradient and the loss.

    `lr = 0` keeps the parameters fixed, so every configuration differentiates
    at the same point and the comparison is of gradients rather than of two
    diverging trajectories.
    """
    rollout = _rollout()
    module = rollout.controller.as_module()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.0)
    strategy = GradientDescentStrategy(
        rollout,
        optimizer,
        loss_reduction=reduction,
        gradient_clip_norm=gradient_clip_norm,
        microbatch=microbatch,
    )
    metrics = strategy.step(_context(size), _batch(size))
    grad = torch.cat(
        [p.grad.reshape(-1).clone() for p in module.parameters() if p.grad is not None]
    )
    assert grad.numel() > 0, "the fixture trains nothing; the test proves nothing"
    return grad, metrics["loss"]


# -- the invariant -----------------------------------------------------------


@pytest.mark.parametrize(
    ("size", "microbatch"),
    [
        (12, 12),  # one whole pass
        (12, 6),  # an exact divisor
        (12, 5),  # a NON-divisor: chunks 5/5/2
        (12, 1),  # one trajectory per pass
        (12, 100),  # larger than the batch -- still one pass
        (7, 3),  # prime batch, chunks 3/3/1
    ],
)
def test_the_accumulated_gradient_equals_the_unchunked_gradient(
    size: int, microbatch: int
) -> None:
    """The whole feature, stated once."""
    reference, _ = _gradient_after_one_step(size, microbatch=None)
    chunked, _ = _gradient_after_one_step(size, microbatch=microbatch)

    assert torch.allclose(chunked, reference, rtol=1e-11, atol=1e-13), (
        f"microbatch={microbatch} over an effective batch of {size} changed the "
        f"gradient by {float((chunked - reference).abs().max()):.3e}"
    )


def test_a_uniform_one_over_k_weighting_is_refused() -> None:
    """The defect §4.3's prose invites, in the case its own gate requires.

    "Each pass contributes its microbatch's share" reads as `1/K`. With
    `size = 10, microbatch = 3` the chunks are 3/3/3/1 and `1/K = 0.25`
    over-weights the last by 2.5x. This test computes what `1/K` would produce
    and requires the implementation NOT to match it -- so it fails against the
    natural wrong reading, not merely passes against the right one.
    """
    size, microbatch = 10, 3
    reference, _ = _gradient_after_one_step(size, microbatch=None)
    chunked, _ = _gradient_after_one_step(size, microbatch=microbatch)

    # What a 1/K implementation computes: every chunk mean weighted equally.
    rollout = _rollout()
    module = rollout.controller.as_module()
    initial_state, process_noise, measurement_noise = _batch(size)
    bounds = [(0, 3), (3, 6), (6, 9), (9, 10)]
    for low, high in bounds:
        cost = rollout(
            initial_state[low:high],
            process_noise[low:high],
            measurement_noise[low:high],
        )[3]
        (cost.mean() / len(bounds)).backward()
    one_over_k = torch.cat(
        [p.grad.reshape(-1).clone() for p in module.parameters() if p.grad is not None]
    )

    assert not torch.allclose(one_over_k, reference, rtol=1e-6), (
        "the fixture cannot distinguish 1/K from the correct weighting, so "
        "this test could not fail against the defect it exists for"
    )
    assert torch.allclose(chunked, reference, rtol=1e-11, atol=1e-13)


def test_sum_accumulates_without_rescaling() -> None:
    """The reduction-agnostic half: under `sum` there is nothing to rescale."""
    reference, _ = _gradient_after_one_step(12, microbatch=None, reduction=torch.sum)
    chunked, _ = _gradient_after_one_step(12, microbatch=5, reduction=torch.sum)

    assert torch.allclose(chunked, reference, rtol=1e-11, atol=1e-13)


def test_the_reported_loss_is_the_declared_loss() -> None:
    """A chunked step reports the loss over the effective batch, not a chunk's."""
    _, reference = _gradient_after_one_step(12, microbatch=None)
    _, chunked = _gradient_after_one_step(12, microbatch=5)

    assert chunked == pytest.approx(reference, rel=1e-11)


def test_clipping_applies_once_to_the_accumulated_gradient() -> None:
    """A per-chunk clip makes the threshold depend on the chunk size -- the
    machine-dependent quantity D20 excludes from identity."""
    clip = 1e-3
    unclipped, _ = _gradient_after_one_step(12, microbatch=None)
    reference, _ = _gradient_after_one_step(
        12, microbatch=None, gradient_clip_norm=clip
    )
    chunked, _ = _gradient_after_one_step(12, microbatch=5, gradient_clip_norm=clip)

    # Without this the test would compare two UNCLIPPED gradients and pass for
    # the wrong reason. Measured: 1.869e-01 unclipped against a 1e-3 threshold.
    assert float(unclipped.norm()) > 100 * clip, "the clip does not bind"
    # `rel=1e-4`, not tighter: `clip_grad_norm_` scales by
    # `max_norm / (total_norm + 1e-6)`, so the result lands just under the
    # threshold rather than exactly on it (measured 9.999947e-04).
    assert float(reference.norm()) == pytest.approx(clip, rel=1e-4)
    assert torch.allclose(chunked, reference, rtol=1e-9, atol=1e-12)


def test_a_microbatch_of_none_and_of_the_whole_batch_agree_exactly() -> None:
    """`None` and `N` are the same execution, so they must agree bit for bit --
    not merely within a tolerance, which would hide a stray rescaling."""
    reference, _ = _gradient_after_one_step(12, microbatch=None)
    whole, _ = _gradient_after_one_step(12, microbatch=12)

    assert torch.equal(whole, reference)


# -- the layer-wise sibling --------------------------------------------------


def _layerwise_gradient(
    size: int, *, microbatch: int | None
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """One masked phase step at `lr = 0`: the flattened gradient, and the
    frozen rows' gradients by name."""
    from mbl.engine.strategy import LayerwiseGradientDescentStrategy, StepExecution
    from mbl.engine.training_plan import OptimizerSpec
    from mbl.models.unfolded.layerwise import (
        LayerFreezeBuilder,
        ParameterActivation,
    )

    rollout = _rollout()
    controller = rollout.controller
    module = controller.as_module()
    freeze = LayerFreezeBuilder().build(
        ParameterActivation(step_size_rows=frozenset({0}), train_matrix=False),
        controller.config.parameters,
    )
    strategy = LayerwiseGradientDescentStrategy(
        rollout,
        module,
        # `OptimizerSpec` requires a positive rate; the gradient is computed
        # before the step, so a negligible one leaves both configurations
        # differentiating at the same point.
        OptimizerSpec("sgd", 1e-12),
        freeze,
        execution=StepExecution(microbatch=microbatch),
    )
    strategy.step(_context(size), _batch(size))
    grad = torch.cat(
        [p.grad.reshape(-1).clone() for p in module.parameters() if p.grad is not None]
    )
    masked = {
        name: p.grad.clone()
        for name, p in module.named_parameters()
        if p.grad is not None and freeze.grad_masks.get(name) is not None
    }
    return grad, masked


def test_the_layer_wise_sibling_chunks_to_the_same_gradient() -> None:
    """Scope is both strategies: which one trains a contender is the author's
    free choice and has nothing to do with memory, so an executor covering one
    would make the microbatch's availability depend on how the training was
    spelled (Annex 06 §4.3's correction)."""
    reference, _ = _layerwise_gradient(12, microbatch=None)
    chunked, _ = _layerwise_gradient(12, microbatch=5)

    assert torch.allclose(chunked, reference, rtol=1e-11, atol=1e-13)


def test_the_frozen_rows_stay_frozen_under_chunking() -> None:
    """The mask applies ONCE to the accumulated gradient. Masking per chunk is
    arithmetically the same only while the mask is constant across chunks, and
    relying on that would make a future per-chunk mask silently wrong."""
    _, reference_masked = _layerwise_gradient(12, microbatch=None)
    _, chunked_masked = _layerwise_gradient(12, microbatch=5)

    assert reference_masked, "no masked parameter: the freeze fixture is inert"
    for name, reference in reference_masked.items():
        assert torch.allclose(chunked_masked[name], reference, rtol=1e-11, atol=1e-13)
