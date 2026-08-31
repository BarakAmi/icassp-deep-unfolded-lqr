"""Notebook-facing dimension-robustness spot check (NB03 review follow-up):
the permanent CI gate `tests/applications/test_unfolded_dimension_robustness
.py` already covers the full ``(n, m)`` grid for every unfolded kind -- this
module gives a notebook a ONE-LINE way to re-run the same shape/broadcast
assertion against ITS OWN configured problem knobs (step sizes, horizon,
dtype, ...), instead of each unfolding notebook re-authoring the inline
build-and-forward loop from scratch.
"""

from dataclasses import dataclass, field

import torch

from ..applications.factories import LQRProblemFactory, ProblemFactory
from ..applications.ltv_factories import LTVLQRProblemFactory, LTVRegime
from ..applications.recipes import UnfoldedKind
from ..applications.recipes.unfolded import UnfoldedBuildSpec, build_unfolded_controller
from ..core.runtime import Backend, ComputeContext, Precision

_ALL_UNFOLDED_KINDS = tuple(UnfoldedKind)


@dataclass(frozen=True)
class DimensionRobustnessSpec:
    """The one experiment's own knobs to probe the unfolded backend with, at
    every control dimension in `control_dims` -- everything else (state
    dimension, horizon, step sizes, dtype, noise level) held fixed at the
    calling notebook's configured values, so a shape/broadcast regression in
    the ACTUAL configuration in use is what this catches.

    Attributes:
        state_dim: Fixed state dimension every probed problem uses.
        horizon: Fixed horizon every probed problem uses.
        num_iterations: Unfolding depth to build each probe controller at.
        step_size_init: Initial per-iteration step size.
        step_size_max: Per-iteration step size cap.
        seed: Problem-generation seed (shared across every probed `m`).
        dtype: Torch dtype for both the probe tensors and the `ComputeContext`.
        process_noise_std: Std of the synthetic process-noise probe batch.
        control_dims: Control dimensions `m` to probe, in order.
        batch_size: Synthetic probe-batch size (no training occurs; a small
            fixed batch is enough to exercise every shape/broadcast path).
        kinds: Which `UnfoldedKind` variants to probe; defaults to all of them.
        u_max: Optional box bound (docs/planning/03_studies/nb05_ltv_box_constrained/ltv_box_constrained_lqr_benchmark.md
            Sec 6.7). ``None`` (default) probes the UNCONSTRAINED forward
            path, exactly as before this field existed. When set, every
            probed control is additionally asserted to satisfy
            ``|u| <= u_max + eps`` -- exercising, and verifying, the
            PROJECTED refinement rather than merely its shape.
        regime: Optional `applications.ltv_factories.LTVRegime`. ``None``
            (default) probes an `LQRProblemFactory` (LTI) problem, exactly
            as before this field existed; when set, probes an
            `LTVLQRProblemFactory` problem instead -- the constrained, LTV
            forward path in one call.
        period_or_block: Forwarded to `LTVLQRProblemFactory` when `regime`
            is set; ignored otherwise (see that factory's own field).
        variation_strength: Forwarded to `LTVLQRProblemFactory` when
            `regime` is set; ignored otherwise.
    """

    state_dim: int
    horizon: int
    num_iterations: int
    step_size_init: float
    step_size_max: float
    seed: int
    dtype: torch.dtype
    process_noise_std: float
    control_dims: tuple[int, ...] = (1, 2, 3)
    batch_size: int = 4
    kinds: tuple[UnfoldedKind, ...] = field(default=_ALL_UNFOLDED_KINDS)
    u_max: float | None = None
    regime: LTVRegime | None = None
    period_or_block: int | None = None
    variation_strength: float = 0.0

    def _build_problem_factory(self, control_dim: int) -> ProblemFactory:
        """The probe problem for one control dimension `control_dim` --
        `LQRProblemFactory` (LTI) when `regime` is ``None``,
        `LTVLQRProblemFactory` otherwise. Both take `u_max` identically."""
        if self.regime is None:
            return LQRProblemFactory(
                state_dim=self.state_dim,
                control_dim=control_dim,
                horizon=self.horizon,
                seed=self.seed,
                u_max=self.u_max,
            )
        return LTVLQRProblemFactory(
            state_dim=self.state_dim,
            control_dim=control_dim,
            horizon=self.horizon,
            seed=self.seed,
            regime=self.regime,
            period_or_block=self.period_or_block,
            variation_strength=self.variation_strength,
            u_max=self.u_max,
        )


def assert_unfolded_dimension_robustness(spec: DimensionRobustnessSpec) -> str:
    """Build and forward every kind in `spec.kinds`, at every control
    dimension in `spec.control_dims`, and assert each produces a
    finite, correctly-shaped control tensor -- raising `AssertionError` on
    the first shape/broadcast mismatch, exactly as the CI gate does, but
    against `spec`'s own configuration rather than the CI gate's fixed grid.
    When `spec.u_max` is set, additionally asserts every probed control
    satisfies the box (the projected, not merely unconstrained, path).

    Args:
        spec: The problem/backend configuration to probe.

    Returns:
        A one-line human-readable confirmation, meant to be `print`-ed by
        the caller (this function performs no I/O of its own).

    Raises:
        AssertionError: If any probed `(kind, m)` combination forwards a
            control tensor of the wrong shape, or (when `spec.u_max` is set)
            a control exceeding the box.
    """
    probe_ctx = ComputeContext(
        backend=Backend.TORCH,
        device="cpu",
        precision=Precision.from_torch_dtype(spec.dtype),
    )
    for m in spec.control_dims:
        problem = spec._build_problem_factory(m).build()
        x0 = torch.randn(spec.batch_size, spec.state_dim, dtype=spec.dtype)
        w = spec.process_noise_std * torch.randn(
            spec.batch_size, spec.horizon, spec.state_dim, dtype=spec.dtype
        )
        v = torch.zeros(
            spec.batch_size,
            spec.horizon,
            problem.system.dimensions.observation_dim,
            dtype=spec.dtype,
        )
        for kind in spec.kinds:
            controller = build_unfolded_controller(
                problem,
                probe_ctx,
                UnfoldedBuildSpec(
                    kind=kind,
                    num_iterations=spec.num_iterations,
                    step_size_init=spec.step_size_init,
                    step_size_max=spec.step_size_max,
                    horizon=spec.horizon,
                ),
            )
            with torch.no_grad():
                _, _, u, _ = controller.forward(x0, w, v)
            assert u.shape == (spec.batch_size, spec.horizon, m), (
                kind,
                m,
                tuple(u.shape),
            )
            if spec.u_max is not None:
                assert bool((u.abs() <= spec.u_max + 1e-6).all()), (
                    kind,
                    m,
                    float(u.abs().max()),
                )
    regime_note = f", regime={spec.regime}" if spec.regime is not None else ""
    return (
        "Dimension-robustness check passed: every unfolded kind forwards "
        f"cleanly for m in {list(spec.control_dims)} (no shape/broadcast "
        f"mismatch{regime_note})."
    )
