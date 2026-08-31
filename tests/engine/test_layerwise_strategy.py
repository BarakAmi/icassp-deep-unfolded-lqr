"""NB03 acceptance tests for `LayerwiseGradientDescentStrategy` (blueprint v2
SS3.4/SS7.1, the Freeze Contract): frozen step-size rows are provably airtight
against BOTH weight decay (rejected outright) and residual Adam momentum
carryover (each phase's strategy owns a fresh optimizer), and the unified
matrix's whole-tensor `requires_grad` freeze is airtight too -- executed end
to end against a REAL `UnfoldedController` (kind
``LEARNED_STEP_SIZE_AND_MATRIX``, the RiccatiRefinement configuration), not a
mock, so the freeze is exercised through the actual autograd graph the
production recipe builds.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from mbl.applications.recipes.unfolded import (
    UnfoldedBuildSpec,
    UnfoldedKind,
    build_unfolded_controller,
)
from mbl.applications.rollout import RolloutModel
from mbl.core.cost.quadratic_cost import QuadraticCost
from mbl.core.optimal_control_problem import OptimalControlProblem
from mbl.core.runtime import Backend, ComputeContext
from mbl.core.system.linear_system import LinearSystem
from mbl.engine.config import TrainingConfig
from mbl.engine.context import RunContext
from mbl.engine.strategy import (
    LayerwiseGradientDescentStrategy,
    StepExecution,
    TrainingStrategy,
)
from mbl.engine.training_plan import OptimizerSpec
from mbl.models.unfolded.base import UnfoldedController
from mbl.models.unfolded.layerwise import (
    LayerFreeze,
    LayerFreezeBuilder,
    ParameterActivation,
)

DTYPE = torch.float64
STATE_DIM, CONTROL_DIM, HORIZON = 3, 2, 5
NUM_ITERATIONS = 4


def _problem() -> OptimalControlProblem:
    rng = np.random.default_rng(0)
    A = rng.normal(size=(STATE_DIM, STATE_DIM))
    A /= np.max(np.abs(np.linalg.eigvals(A)))
    B = rng.normal(size=(STATE_DIM, CONTROL_DIM))
    system = LinearSystem.fully_observable(A, B)
    Q = np.repeat(np.eye(STATE_DIM)[None], HORIZON + 1, axis=0)
    R = np.repeat(np.eye(CONTROL_DIM)[None], HORIZON, axis=0)
    return OptimalControlProblem(system=system, cost=QuadraticCost(Q=Q, R=R))


def _controller() -> UnfoldedController:
    ctx = ComputeContext(backend=Backend.TORCH, device="cpu")
    return build_unfolded_controller(
        _problem(),
        ctx,
        UnfoldedBuildSpec(
            kind=UnfoldedKind.LEARNED_STEP_SIZE_AND_MATRIX,
            num_iterations=NUM_ITERATIONS,
            step_size_init=0.05,
            step_size_max=0.5,
            horizon=HORIZON,
        ),
    )


def _batch(batch_size: int = 8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    x0 = torch.randn(batch_size, STATE_DIM, dtype=DTYPE)
    w = torch.zeros(batch_size, HORIZON, STATE_DIM, dtype=DTYPE)
    v = torch.zeros(batch_size, HORIZON, STATE_DIM, dtype=DTYPE)
    return x0, w, v


def _context() -> RunContext:
    return RunContext(
        model=MagicMock(),
        tracker=MagicMock(),
        config=TrainingConfig(batch_size=8, learning_rate=0.01),
    )


def _strategy(
    controller: UnfoldedController,
    activation: ParameterActivation,
    *,
    lr: float = 0.1,
) -> LayerwiseGradientDescentStrategy:
    module = controller.as_module()
    freeze = LayerFreezeBuilder().build(activation, controller.config.parameters)
    return LayerwiseGradientDescentStrategy(
        RolloutModel(controller), module, OptimizerSpec("adam", lr), freeze
    )


def _raw(controller: UnfoldedController, name: str) -> torch.Tensor:
    """The live, module-aliased raw parameter tensor by name -- typed as a
    plain `torch.Tensor` (unlike `controller.as_module().<name>`, which
    resolves through `nn.Module.__getattr__`'s `Tensor | Module` stub and
    cannot be chained/indexed under strict mypy). Reads the SAME object
    `as_module()` registers (the T2.b aliasing law, proven in
    `tests/applications/test_recipes.py`), so this is behaviorally identical
    to attribute access, not a copy or a parallel lookup.
    """
    return controller.config.parameters[name].get_raw()


class TestProtocolConformance:
    def test_satisfies_training_strategy_protocol(self) -> None:
        strategy = _strategy(
            _controller(), ParameterActivation(step_size_rows="all", train_matrix=True)
        )
        assert isinstance(strategy, TrainingStrategy)

    def test_declares_trains_parameters_true(self) -> None:
        assert LayerwiseGradientDescentStrategy.trains_parameters is True


class TestWeightDecayGuard:
    """Defense-in-depth: the strategy itself refuses weight_decay, even if a
    caller bypasses LayerwiseTrainingPlan and constructs it directly."""

    def test_nonzero_weight_decay_is_rejected_at_construction(self) -> None:
        controller = _controller()
        module = controller.as_module()
        freeze = LayerFreezeBuilder().build(
            ParameterActivation(step_size_rows=frozenset({0}), train_matrix=False),
            controller.config.parameters,
        )
        with pytest.raises(ValueError, match="weight_decay"):
            LayerwiseGradientDescentStrategy(
                RolloutModel(controller),
                module,
                OptimizerSpec("adamw", 0.1, {"weight_decay": 1e-3}),
                freeze,
            )


class TestRowFreezeCorrectness:
    """A-2: after a masked step, only the active row(s) of step_size move."""

    def test_only_the_active_row_changes(self) -> None:
        controller = _controller()
        strategy = _strategy(
            controller,
            ParameterActivation(step_size_rows=frozenset({1}), train_matrix=False),
        )
        before = _raw(controller, "step_size").detach().clone()

        strategy.step(_context(), _batch())

        after = _raw(controller, "step_size").detach().clone()
        assert torch.equal(before[0], after[0])
        assert torch.equal(before[2:], after[2:])
        assert not torch.equal(before[1], after[1])

    def test_frozen_rows_have_exactly_zero_gradient_after_step(self) -> None:
        """The mechanism, not just the outcome: masked rows' .grad is an
        EXACT zero after backward+mask, proving the freeze is not merely a
        near-zero learning-rate effect."""
        controller = _controller()
        strategy = _strategy(
            controller,
            ParameterActivation(step_size_rows=frozenset({1}), train_matrix=False),
        )
        strategy.step(_context(), _batch())
        grad = _raw(controller, "step_size").grad
        assert grad is not None
        assert torch.equal(grad[0], torch.zeros_like(grad[0]))
        assert torch.equal(grad[2], torch.zeros_like(grad[2]))
        assert grad[1].abs().sum() > 0  # the active row genuinely receives signal


class TestNoMomentumCarryover:
    """A-3a: the central v1 defect this build fixes. A row trained (with
    real Adam momentum accrued) in phase A, then frozen in phase B, must be
    BIT-IDENTICAL through every step of phase B -- proving a fresh,
    phase-owned optimizer eliminates the carryover a shared optimizer would
    exhibit (empirically demonstrated in the blueprint's SS7.1: 5/5 steps of
    drift under a shared optimizer, even at weight_decay=0)."""

    def test_a_frozen_row_is_bit_identical_across_the_entire_next_phase(self) -> None:
        controller = _controller()
        batch = _batch()

        # Phase A: row 0 trains for several steps -- real Adam momentum accrues.
        phase_a = _strategy(
            controller,
            ParameterActivation(step_size_rows=frozenset({0}), train_matrix=False),
        )
        for _ in range(5):
            phase_a.step(_context(), batch)
        row0_after_phase_a = _raw(controller, "step_size")[0].detach().clone()
        assert not torch.equal(
            row0_after_phase_a, torch.zeros_like(row0_after_phase_a)
        )  # sanity: it actually moved from its init

        # Phase B: a FRESH strategy/optimizer; row 0 is no longer active.
        phase_b = _strategy(
            controller,
            ParameterActivation(step_size_rows=frozenset({1}), train_matrix=False),
        )
        for _ in range(5):
            phase_b.step(_context(), batch)
            current_row0 = _raw(controller, "step_size")[0].detach()
            assert torch.equal(current_row0, row0_after_phase_a), (
                "row 0 drifted during phase B -- residual momentum leaked "
                "across the phase boundary"
            )


class TestMatrixFreeze:
    """A-3c: the whole-tensor requires_grad freeze for riccati_matrix."""

    def test_matrix_frozen_is_bit_identical_while_step_size_trains(self) -> None:
        controller = _controller()
        L_before = _raw(controller, "riccati_matrix").detach().clone()

        strategy = _strategy(
            controller,
            ParameterActivation(step_size_rows=frozenset({0}), train_matrix=False),
        )
        for _ in range(3):
            strategy.step(_context(), _batch())

        matrix = _raw(controller, "riccati_matrix")
        assert torch.equal(matrix.detach(), L_before)
        assert not matrix.requires_grad

    def test_matrix_active_actually_trains(self) -> None:
        controller = _controller()
        L_before = _raw(controller, "riccati_matrix").detach().clone()

        strategy = _strategy(
            controller,
            ParameterActivation(step_size_rows=frozenset({0}), train_matrix=True),
        )
        strategy.step(_context(), _batch())

        matrix = _raw(controller, "riccati_matrix")
        assert matrix.requires_grad
        assert not torch.equal(matrix.detach(), L_before)

    def test_reactivating_the_matrix_in_a_later_phase_resumes_training(self) -> None:
        """_apply_activation is idempotent and re-checked every step, so a
        matrix frozen in phase A can train again in phase B."""
        controller = _controller()

        frozen_phase = _strategy(
            controller,
            ParameterActivation(step_size_rows=frozenset({0}), train_matrix=False),
        )
        frozen_phase.step(_context(), _batch())
        assert not _raw(controller, "riccati_matrix").requires_grad

        active_phase = _strategy(
            controller,
            ParameterActivation(step_size_rows=frozenset({1}), train_matrix=True),
        )
        L_before_active = _raw(controller, "riccati_matrix").detach().clone()
        active_phase.step(_context(), _batch())
        matrix = _raw(controller, "riccati_matrix")
        assert matrix.requires_grad
        assert not torch.equal(matrix.detach(), L_before_active)


class TestGradientClipping:
    def test_clip_bounds_the_update_of_the_active_row(self) -> None:
        """Mirrors TrainingPlan's own clipping test: a pathologically steep
        loss on the active row, with clipping the step is bounded by
        lr * clip regardless of the frozen rows sharing the same tensor."""
        num_iterations, action_dim = 2, 1
        raw = torch.nn.Parameter(torch.zeros(num_iterations, action_dim, dtype=DTYPE))
        module = torch.nn.Module()
        module.register_parameter("step_size", raw)

        class _SteepModel(torch.nn.Module):
            def forward(self, x0: torch.Tensor, w: torch.Tensor, v: torch.Tensor):
                cost = (raw[0] - 1000.0).pow(2).sum()
                return None, None, None, cost

        # Built by hand (not via LayerFreezeBuilder): this synthetic model is
        # a bare torch.nn.Module, not an UnfoldedController, so there is no
        # UnfoldedParameter for the builder to introspect.
        mask = torch.zeros_like(raw)
        mask[0] = True
        freeze = LayerFreeze(
            grad_masks={"step_size": mask}, trainable={"step_size": True}
        )

        strategy = LayerwiseGradientDescentStrategy(
            _SteepModel(),
            module,
            OptimizerSpec("sgd", 0.01),
            freeze,
            execution=StepExecution(gradient_clip_norm=1.0),
        )
        strategy.step(_context(), (None, None, None))
        assert abs(float(raw[0].detach())) <= 0.01 + 1e-12
