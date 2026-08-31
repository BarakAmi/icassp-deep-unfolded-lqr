"""Acceptance tests for the NB05 P-resolution extension's two new
contenders (docs/planning/03_studies/nb05_ltv_box_constrained/ltv_box_constrained_lqr_benchmark.md Sec 14.9/14.10):

    R1.5 -- UnfoldedKind.LEARNED_STEP_SIZE_AND_SCALAR_MODULATED_MATRIX
    R2   -- UnfoldedKind.LEARNED_STEP_SIZE_AND_PER_ITERATION_MATRIX

Mirrors `test_ltv_unfolded_recipes.py`'s box-feasibility pattern for the
construction-and-forward check, plus the plan's own Sec 14.10 acceptance
criteria narrowed to what this extension actually built: J=1 degeneracy,
identity-at-init, and the R1 <-> R1.5/R2 nesting property.
"""

import numpy as np
import pytest
import torch

from mbl.applications.ltv_factories import LTVLQRProblemFactory, LTVRegime
from mbl.applications.factories import LQRProblemFactory
from mbl.applications.recipes.unfolded import (
    UnfoldedBuildSpec,
    UnfoldedKind,
    build_unfolded_controller,
)
from mbl.core.runtime import Backend, ComputeContext, Precision

_NEW_KINDS = (
    UnfoldedKind.LEARNED_STEP_SIZE_AND_SCALAR_MODULATED_MATRIX,
    UnfoldedKind.LEARNED_STEP_SIZE_AND_PER_ITERATION_MATRIX,
)
_LTV_REGIMES = [
    (LTVRegime.PERIODIC, 4),
    (LTVRegime.BLOCK_CONSTANT, 4),
    (LTVRegime.FULLY_VARYING, None),
]

_CTX = ComputeContext(
    backend=Backend.TORCH,
    device="cpu",
    precision=Precision.from_torch_dtype(torch.float64),
)


def _spec(kind: UnfoldedKind, num_iterations: int, horizon: int) -> UnfoldedBuildSpec:
    return UnfoldedBuildSpec(
        kind=kind,
        num_iterations=num_iterations,
        step_size_init=0.05,
        step_size_max=1.0,
        horizon=horizon,
    )


def _random_psd(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    L = rng.standard_normal((n, n))
    return L @ L.T + np.eye(n)


class TestConstructionAndForward:
    @pytest.mark.parametrize("regime,period_or_block", _LTV_REGIMES)
    @pytest.mark.parametrize("kind", _NEW_KINDS)
    def test_builds_and_forwards_with_box_respected(
        self, regime: LTVRegime, period_or_block: int | None, kind: UnfoldedKind
    ) -> None:
        horizon = 12
        u_max = 0.5
        problem = LTVLQRProblemFactory(
            state_dim=4,
            control_dim=2,
            horizon=horizon,
            seed=3,
            regime=regime,
            period_or_block=period_or_block,
            variation_strength=0.3,
            u_max=u_max,
        ).build()

        controller = build_unfolded_controller(problem, _CTX, _spec(kind, 3, horizon))
        x0 = torch.full((5, 4), 50.0, dtype=torch.float64)
        w = torch.zeros(5, horizon, 4, dtype=torch.float64)
        v = torch.zeros(5, horizon, 4, dtype=torch.float64)

        with torch.no_grad():
            _, _, u, _ = controller.forward(x0, w, v)

        assert torch.isfinite(u).all()
        assert u.shape == (5, horizon, 2)
        assert torch.all(u.abs() <= u_max + 1e-6)

    @pytest.mark.parametrize("kind", _NEW_KINDS)
    def test_builds_and_forwards_on_lti(self, kind: UnfoldedKind) -> None:
        horizon = 10
        problem = LQRProblemFactory(
            state_dim=4, control_dim=2, horizon=horizon, seed=0
        ).build()
        controller = build_unfolded_controller(problem, _CTX, _spec(kind, 4, horizon))
        x0 = torch.randn(6, 4, dtype=torch.float64)
        w = torch.zeros(6, horizon, 4, dtype=torch.float64)
        v = torch.zeros(6, horizon, 4, dtype=torch.float64)
        with torch.no_grad():
            _, _, u, _ = controller.forward(x0, w, v)
        assert torch.isfinite(u).all()


#: R2 and R1 are fed bit-identical parameters, but reach the same quantity
#: through different op sequences -- `P[j]` selected from a stack against a
#: single shared `P` -- so their outputs are two reduction orders of one sum.
#: That is a property of the BLAS, not of the mathematics, and no two machines
#: owe each other the same last bit; asserting `torch.equal` here made the
#: suite pass or fail according to which CPU the CI runner allocated. Matches
#: the tolerance the sibling R1.5 comparison below already uses, for the same
#: shapes at the same precision.
#:
#: The full fix -- assert the *constructed parameters* exactly (structural,
#: portable) and the forward outputs at a measured tolerance -- is item 4 of
#: `docs/methods/cross_machine_numerical_reproducibility.md` Sec 5, due when
#: Stage 6 rewrites this layer.
DEGENERACY_ATOL = DEGENERACY_RTOL = 1e-8


class TestDegeneracyAndIdentityAtInit:
    """NB05 plan Sec 14.10: at J=1 and at identity-initialization, the two
    new contenders must be indistinguishable from R1
    (LEARNED_STEP_SIZE_AND_MATRIX) -- untrained, every P^(j) starts at the
    identity, exactly reproducing R1's own default."""

    def _forward(self, problem, kind: UnfoldedKind, num_iterations: int, horizon: int):
        controller = build_unfolded_controller(
            problem, _CTX, _spec(kind, num_iterations, horizon)
        )
        x0 = torch.randn(
            7, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(1)
        )
        w = torch.randn(
            7,
            horizon,
            4,
            dtype=torch.float64,
            generator=torch.Generator().manual_seed(2),
        )
        v = torch.zeros(7, horizon, 4, dtype=torch.float64)
        with torch.no_grad():
            _, _, u, _ = controller.forward(x0, w, v)
        return u

    def test_j1_per_iteration_matrix_matches_r1(self) -> None:
        horizon = 8
        problem = LQRProblemFactory(
            state_dim=4, control_dim=2, horizon=horizon, seed=0
        ).build()
        u_r1 = self._forward(
            problem, UnfoldedKind.LEARNED_STEP_SIZE_AND_MATRIX, 1, horizon
        )
        u_r2 = self._forward(
            problem, UnfoldedKind.LEARNED_STEP_SIZE_AND_PER_ITERATION_MATRIX, 1, horizon
        )
        torch.testing.assert_close(
            u_r1, u_r2, atol=DEGENERACY_ATOL, rtol=DEGENERACY_RTOL
        )

    def test_identity_at_init_per_iteration_matrix_matches_r1_at_j4(self) -> None:
        horizon = 8
        problem = LQRProblemFactory(
            state_dim=4, control_dim=2, horizon=horizon, seed=0
        ).build()
        u_r1 = self._forward(
            problem, UnfoldedKind.LEARNED_STEP_SIZE_AND_MATRIX, 4, horizon
        )
        u_r2 = self._forward(
            problem, UnfoldedKind.LEARNED_STEP_SIZE_AND_PER_ITERATION_MATRIX, 4, horizon
        )
        torch.testing.assert_close(
            u_r1, u_r2, atol=DEGENERACY_ATOL, rtol=DEGENERACY_RTOL
        )

    def test_the_degeneracy_tolerance_still_rejects_a_real_difference(self) -> None:
        """Measure the detector, do not assume it.

        Relaxing `torch.equal` to a tolerance is only defensible if what
        remains can still fail. A degeneracy that genuinely broke -- R2 no
        longer collapsing to R1 -- would move the controls by a large fraction;
        1e-6 relative is four orders below that and four above the arithmetic
        noise `DEGENERACY_ATOL` admits, and must be caught.
        """
        horizon = 8
        problem = LQRProblemFactory(
            state_dim=4, control_dim=2, horizon=horizon, seed=0
        ).build()
        u_r1 = self._forward(
            problem, UnfoldedKind.LEARNED_STEP_SIZE_AND_MATRIX, 4, horizon
        )
        with pytest.raises(AssertionError):
            torch.testing.assert_close(
                u_r1,
                u_r1 * (1.0 + 1e-6),
                atol=DEGENERACY_ATOL,
                rtol=DEGENERACY_RTOL,
            )

    def test_identity_at_init_scalar_modulated_matches_r1_at_j4(self) -> None:
        """The sigmoid/log reparameterization roundtrip in
        `MatrixModulationParameter`'s init (c_j ~ 1.0) introduces
        floating-point error at the ~1e-15 level, so this is `allclose`,
        not `torch.equal` -- unlike R2's direct identity-tensor
        construction, which needs no such roundtrip."""
        horizon = 8
        problem = LQRProblemFactory(
            state_dim=4, control_dim=2, horizon=horizon, seed=0
        ).build()
        u_r1 = self._forward(
            problem, UnfoldedKind.LEARNED_STEP_SIZE_AND_MATRIX, 4, horizon
        )
        u_r1_5 = self._forward(
            problem,
            UnfoldedKind.LEARNED_STEP_SIZE_AND_SCALAR_MODULATED_MATRIX,
            4,
            horizon,
        )
        torch.testing.assert_close(u_r1, u_r1_5, atol=1e-8, rtol=1e-8)


class TestNesting:
    """NB05 plan Sec 14.2/14.10: R2 strictly contains R1 (loading R1's
    matrix into every R2 slice reproduces R1 exactly), and R1.5's c_j = 1
    reproduces R1 exactly -- the plumbing must carry this nesting, not just
    the algebra."""

    def _controller_and_inputs(
        self, kind: UnfoldedKind, num_iterations: int, horizon: int
    ):
        problem = LQRProblemFactory(
            state_dim=4, control_dim=2, horizon=horizon, seed=1
        ).build()
        controller = build_unfolded_controller(
            problem, _CTX, _spec(kind, num_iterations, horizon)
        )
        x0 = torch.randn(
            6, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(9)
        )
        w = torch.randn(
            6,
            horizon,
            4,
            dtype=torch.float64,
            generator=torch.Generator().manual_seed(10),
        )
        v = torch.zeros(6, horizon, 4, dtype=torch.float64)
        return controller, x0, w, v

    def test_per_iteration_matrix_loaded_with_r1s_matrix_reproduces_r1(self) -> None:
        horizon, num_iterations = 8, 4
        P_custom = _random_psd(4, seed=7)

        r1, x0, w, v = self._controller_and_inputs(
            UnfoldedKind.LEARNED_STEP_SIZE_AND_MATRIX, num_iterations, horizon
        )
        r1.config.parameters["riccati_matrix"].load(P_custom)
        with torch.no_grad():
            _, _, u_r1, _ = r1.forward(x0, w, v)

        r2, _, _, _ = self._controller_and_inputs(
            UnfoldedKind.LEARNED_STEP_SIZE_AND_PER_ITERATION_MATRIX,
            num_iterations,
            horizon,
        )
        r2.config.parameters["riccati_matrix"].load(
            np.stack([P_custom] * num_iterations)
        )
        with torch.no_grad():
            _, _, u_r2, _ = r2.forward(x0, w, v)

        torch.testing.assert_close(u_r1, u_r2, atol=1e-10, rtol=1e-10)

    def test_scalar_modulation_of_one_reproduces_r1(self) -> None:
        horizon, num_iterations = 8, 4
        P_custom = _random_psd(4, seed=8)

        r1, x0, w, v = self._controller_and_inputs(
            UnfoldedKind.LEARNED_STEP_SIZE_AND_MATRIX, num_iterations, horizon
        )
        r1.config.parameters["riccati_matrix"].load(P_custom)
        with torch.no_grad():
            _, _, u_r1, _ = r1.forward(x0, w, v)

        r1_5, _, _, _ = self._controller_and_inputs(
            UnfoldedKind.LEARNED_STEP_SIZE_AND_SCALAR_MODULATED_MATRIX,
            num_iterations,
            horizon,
        )
        r1_5.config.parameters["riccati_matrix"].load(P_custom)
        r1_5.config.parameters["matrix_modulation"].load(np.ones(num_iterations))
        with torch.no_grad():
            _, _, u_r1_5, _ = r1_5.forward(x0, w, v)

        torch.testing.assert_close(u_r1, u_r1_5, atol=1e-8, rtol=1e-8)
