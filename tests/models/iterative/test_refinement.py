"""Unification regression coverage for `models.iterative.refinement` (Phase
2.5-B): `GradientDescentRefinement.__call__`'s per-time-step loop must be
provably equivalent to manually chaining its own `refine_step` (the
single-iteration body an external whole-horizon driver -- the upcoming
analytical solver -- steps through one call at a time), and the explicit
`StepSizeProvider` seam must accept both a fixed `StepSizeSchedule` and a
learnable stand-in without behavior changing."""

import torch

from mbl.models.iterative.refinement import GradientDescentRefinement
from mbl.models.iterative.step_size import StepSizeSchedule

DTYPE = torch.float64


class _QuadraticRefinement(GradientDescentRefinement):
    """Minimal concrete GradientDescentRefinement: grad = M @ u (no C/y
    term), M supplied via static_parameters -- just enough to exercise the
    shared loop mechanics without any LQR-specific machinery."""

    def pre_iteration_hook(self, t, y):
        return (self.static_parameters["M"],)

    def get_gradient(self, u, M):
        return u @ M


def _build_refinement(num_iterations: int, alpha: float) -> _QuadraticRefinement:
    M = torch.tensor([[2.0]], dtype=DTYPE)
    step_size = StepSizeSchedule(
        raw=torch.tensor(alpha, dtype=DTYPE),
        num_iterations=num_iterations,
        horizon=1,
        control_dim=1,
    )
    return _QuadraticRefinement(
        step_size=step_size, num_iterations=num_iterations, static_parameters={"M": M}
    )


def test_call_equals_manually_chained_refine_step() -> None:
    """The defining unification property: __call__'s loop is EXACTLY
    refine_step invoked num_iterations times -- the factoring changed no
    behavior, it only exposed the loop body as a reusable single-step call."""
    refinement = _build_refinement(num_iterations=5, alpha=0.1)
    y = torch.zeros(1, 1, dtype=DTYPE)
    u0 = torch.full((1, 1), 3.0, dtype=DTYPE)

    u_via_call = refinement(t=0, y=y, u=u0.clone())

    u_manual = refinement.apply_constraints(u0.clone())
    args = refinement.pre_iteration_hook(0, y)
    for k in range(5):
        u_manual, _ = refinement.refine_step(k, u_manual, *args)

    assert torch.allclose(u_via_call, u_manual)


def test_refine_step_applies_one_update_and_returns_the_gradient() -> None:
    refinement = _build_refinement(num_iterations=1, alpha=0.5)
    u0 = torch.full((1, 1), 4.0, dtype=DTYPE)
    M = torch.tensor([[2.0]], dtype=DTYPE)

    u_next, grad = refinement.refine_step(0, u0, M)

    expected_grad = u0 @ M
    expected_u_next = u0 - 0.5 * expected_grad
    assert torch.allclose(grad, expected_grad)
    assert torch.allclose(u_next, expected_u_next)


def test_iteration_index_selects_the_right_step_size_row() -> None:
    """A 2D (per-iteration) schedule proves refine_step's iteration_index
    reaches the step-size provider correctly, mirroring the analogous
    AnalyticalGradientDescent counter-advance test."""
    raw = torch.tensor([[0.5], [0.1], [0.01]], dtype=DTYPE)
    step_size = StepSizeSchedule(raw=raw, num_iterations=3, horizon=1, control_dim=1)
    refinement = _QuadraticRefinement(
        step_size=step_size,
        num_iterations=3,
        static_parameters={"M": torch.tensor([[1.0]], dtype=DTYPE)},
    )
    u = torch.full((1, 1), 1.0, dtype=DTYPE)

    for k in range(3):
        u_next, grad = refinement.refine_step(k, u, torch.tensor([[1.0]], dtype=DTYPE))
        assert torch.allclose(u_next, u - raw[k] * grad)
        u = u_next


def test_get_signature_includes_step_size_shape_and_hash_for_a_schedule() -> None:
    refinement = _build_refinement(num_iterations=4, alpha=0.2)

    signature = refinement.get_signature()

    assert signature["type"] == "_QuadraticRefinement"
    assert signature["num_iterations"] == 4
    assert signature["step_size_shape"] == ()
    assert signature["step_size_hash"].startswith("sha256:")


def test_get_signature_omits_step_size_fields_for_a_non_schedule_provider() -> None:
    class _LearnableStub:
        def for_iteration(self, i):
            return torch.tensor([0.1], dtype=DTYPE)

    refinement = _QuadraticRefinement(
        step_size=_LearnableStub(),
        num_iterations=2,
        static_parameters={"M": torch.tensor([[1.0]], dtype=DTYPE)},
    )

    signature = refinement.get_signature()

    assert signature == {"type": "_QuadraticRefinement", "num_iterations": 2}
