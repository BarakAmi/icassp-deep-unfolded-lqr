"""Phase E acceptance tests for `DimensionRobustnessSpec`'s LTV/`u_max`
threading (docs/planning/03_studies/nb05_ltv_box_constrained/ltv_box_constrained_lqr_benchmark.md Sec 6.7): the
add-only fields must preserve `test_robustness.py`'s existing (unconstrained,
LTI) behavior exactly when omitted, and exercise the constrained, LTV
forward path -- with a genuine box-satisfaction assertion, not just a shape
check -- when set.
"""

import pytest
import torch

from mbl.applications.ltv_factories import LTVRegime
from mbl.workbench import DimensionRobustnessSpec, assert_unfolded_dimension_robustness


def _spec(**overrides: object) -> DimensionRobustnessSpec:
    defaults: dict[str, object] = dict(
        state_dim=3,
        horizon=6,
        num_iterations=3,
        step_size_init=0.02,
        step_size_max=0.3,
        seed=0,
        dtype=torch.float64,
        process_noise_std=0.1,
        control_dims=(1, 2, 3),
        batch_size=4,
    )
    defaults.update(overrides)
    return DimensionRobustnessSpec(**defaults)


class TestBackwardCompatibility:
    def test_default_fields_preserve_lti_unconstrained_behavior(self) -> None:
        message = assert_unfolded_dimension_robustness(_spec())
        assert "passed" in message
        assert "regime=" not in message


class TestUMaxThreading:
    def test_u_max_alone_probes_the_constrained_lti_path(self) -> None:
        message = assert_unfolded_dimension_robustness(_spec(u_max=0.2))
        assert "passed" in message

    def test_probed_controls_respect_u_max(self) -> None:
        """Not just 'it ran' -- every probed control must actually satisfy
        the box; a broken projection would still pass the shape check."""
        from mbl.applications.recipes import UnfoldedKind
        from mbl.applications.recipes.unfolded import (
            UnfoldedBuildSpec,
            build_unfolded_controller,
        )
        from mbl.applications.ltv_factories import LTVLQRProblemFactory
        from mbl.core.runtime import Backend, ComputeContext, Precision

        u_max = 0.05  # tight enough to force saturation
        problem = LTVLQRProblemFactory(
            state_dim=3,
            control_dim=2,
            horizon=6,
            seed=0,
            regime=LTVRegime.FULLY_VARYING,
            variation_strength=0.5,
            u_max=u_max,
        ).build()
        ctx = ComputeContext(
            backend=Backend.TORCH,
            device="cpu",
            precision=Precision.from_torch_dtype(torch.float64),
        )
        controller = build_unfolded_controller(
            problem,
            ctx,
            UnfoldedBuildSpec(
                kind=UnfoldedKind.LEARNED_STEP_SIZE_AND_MATRIX,
                num_iterations=3,
                step_size_init=0.05,
                step_size_max=1.0,
                horizon=6,
            ),
        )
        x0 = 50.0 * torch.ones(4, 3, dtype=torch.float64)
        w = torch.zeros(4, 6, 3, dtype=torch.float64)
        v = torch.zeros(4, 6, 3, dtype=torch.float64)
        with torch.no_grad():
            _, _, u, _ = controller.forward(x0, w, v)
        assert bool((u.abs() <= u_max + 1e-6).all())


class TestLtvRegimeThreading:
    @pytest.mark.parametrize(
        "regime,period_or_block",
        [
            (LTVRegime.PERIODIC, 3),
            (LTVRegime.BLOCK_CONSTANT, 3),
            (LTVRegime.FULLY_VARYING, None),
        ],
    )
    def test_every_regime_probes_cleanly_with_a_box(
        self, regime, period_or_block
    ) -> None:
        message = assert_unfolded_dimension_robustness(
            _spec(
                u_max=0.3,
                regime=regime,
                period_or_block=period_or_block,
                variation_strength=0.4,
            )
        )
        assert "passed" in message
        assert regime.value in message

    def test_regime_without_u_max_still_probes_the_unconstrained_ltv_path(self) -> None:
        message = assert_unfolded_dimension_robustness(
            _spec(regime=LTVRegime.PERIODIC, period_or_block=3, variation_strength=0.4)
        )
        assert "passed" in message
