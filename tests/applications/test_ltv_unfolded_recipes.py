"""Phase B acceptance tests for unblocking the unfolded-controller family on
LTV systems (docs/planning/03_studies/nb05_ltv_box_constrained/ltv_box_constrained_lqr_benchmark.md Sec 6.3/6.4/12):

1. The LTI bit-identity regression: the per-step-densified construction must
   reproduce, bit-for-bit, the OLD `.expand()`-based construction it
   replaces -- verified once here as a PERMANENT test (not just the ad hoc
   check done while writing the fix), so a future refactor of
   `build_unfolded_controller` cannot silently reintroduce a numeric drift
   NB03/NB04's golden masters would otherwise be the only thing to catch.
2. All three `UnfoldedKind`s build and forward cleanly on a periodic, a
   block-constant, and a fully-varying LTV problem, with every realized
   control satisfying the box (the projection guarantee, now proven under
   LTV specifically, not just LTI).
3. The snapshot/replay contract (Sec 6.4): `unfolded_convergence_parameters`
   carries the new "A_t"/"B_t" per-step stacks additively, and
   `unfolding_landscape_inputs` prefers them (sliced at `t_star`) when
   present, falling back to the single "A"/"B" matrix for artifacts that
   predate this change -- so no previously cached NB03/NB04 run loses the
   ability to build landscape inputs.
"""

import numpy as np
import pytest
import torch

from mbl.applications.factories import LQRProblemFactory
from mbl.applications.ltv_factories import LTVLQRProblemFactory, LTVRegime
from mbl.applications.recipes.unfolded import (
    UnfoldedBuildSpec,
    UnfoldedKind,
    build_unfolded_controller,
    unfolded_convergence_parameters,
)
from mbl.core.runtime import Backend, ComputeContext, Precision
from mbl.experiments.experiment import ContenderResult
from mbl.models.analytic.riccati import (
    finite_horizon_riccati,
    get_lqr_gradient_matrices,
)
from mbl.models.guards import require_linear_quadratic
from mbl.workbench.replay import unfolding_landscape_inputs

_ALL_KINDS = (
    UnfoldedKind.FIXED,
    UnfoldedKind.LEARNED_STEP_SIZE,
    UnfoldedKind.LEARNED_STEP_SIZE_AND_MATRIX,
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


def _spec(kind: UnfoldedKind, horizon: int) -> UnfoldedBuildSpec:
    return UnfoldedBuildSpec(
        kind=kind,
        num_iterations=3,
        step_size_init=0.05,
        step_size_max=1.0,
        horizon=horizon,
    )


class TestLtiBitIdentityRegression:
    """Sec 6.3's mandatory acceptance test: the fix must be numerically
    invisible on every LTI problem, at every intermediate stage -- not just
    "the final cost looks similar"."""

    @pytest.fixture
    def problem_and_system(self):
        problem = LQRProblemFactory(
            state_dim=4, control_dim=3, horizon=10, seed=0
        ).build()
        system, cost = require_linear_quadratic(problem)
        return problem, system, cost

    def test_m_and_c_stacks_match_the_old_array_based_construction(
        self, problem_and_system
    ) -> None:
        problem, system, cost = problem_and_system
        horizon = 10

        # OLD construction, inlined verbatim (never re-imported from git
        # history -- git operations are not used for this comparison, see
        # the plan's own Sec 6.3 caution about not touching git state for
        # verification purposes).
        A_old, B_old, R_old = system.A_t.array, system.B_t.array, cost.R
        P_arr, _ = finite_horizon_riccati(system, cost, horizon)
        M_old, C_old = get_lqr_gradient_matrices(P_arr, A_old, B_old, R_old)

        # NEW construction (the per-step densification now inside
        # build_unfolded_controller).
        A_new = np.stack([system.A_t[k] for k in range(horizon)])
        B_new = np.stack([system.B_t[k] for k in range(horizon)])
        M_new, C_new = get_lqr_gradient_matrices(P_arr, A_new, B_new, cost.R)

        np.testing.assert_array_equal(M_old, M_new)
        np.testing.assert_array_equal(C_old, C_new)

    def test_a_stack_expand_vs_densified_are_bit_identical(
        self, problem_and_system
    ) -> None:
        _, system, _ = problem_and_system
        horizon = 10
        n = system.dimensions.state_dim
        dtype = torch.float64

        A_old = system.A_t.array
        expanded = torch.tensor(A_old, dtype=dtype).unsqueeze(0).expand(horizon, n, n)
        densified = torch.tensor(
            np.stack([system.A_t[k] for k in range(horizon)]), dtype=dtype
        )
        assert torch.equal(expanded, densified)

    @pytest.mark.parametrize("kind", _ALL_KINDS)
    def test_full_forward_pass_matches_a_reference_seeded_run(
        self, problem_and_system, kind: UnfoldedKind
    ) -> None:
        """Not bit-identity against the pre-fix code path directly (the
        controller construction changed), but a determinism check: two
        independently built controllers from the identical problem/spec,
        under the identical seed, must forward to the identical output --
        i.e. the fix introduced no hidden nondeterminism (e.g. dict
        iteration order, uninitialized memory from a stride-0 view)."""
        problem, _, _ = problem_and_system
        x0 = torch.randn(
            6, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(1)
        )
        w = torch.randn(
            6, 10, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(2)
        )
        v = torch.zeros(6, 10, 4, dtype=torch.float64)

        outputs = []
        for _ in range(2):
            torch.manual_seed(42)
            controller = build_unfolded_controller(problem, _CTX, _spec(kind, 10))
            with torch.no_grad():
                _, _, u, _ = controller.forward(x0, w, v)
            outputs.append(u)

        assert torch.equal(outputs[0], outputs[1])


class TestLtvConstructionAndForward:
    """Sec 6.3/9: every UnfoldedKind builds and forwards on every regime,
    with the box respected (the projection guarantee under LTV)."""

    @pytest.mark.parametrize("regime,period_or_block", _LTV_REGIMES)
    @pytest.mark.parametrize("kind", _ALL_KINDS)
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

        controller = build_unfolded_controller(problem, _CTX, _spec(kind, horizon))
        # Deliberately large-magnitude state to force saturation (mirrors
        # test_nb04_box_constrained.py's TestBoxConstraintSatisfaction).
        x0 = torch.full((5, 4), 50.0, dtype=torch.float64)
        w = torch.zeros(5, horizon, 4, dtype=torch.float64)
        v = torch.zeros(5, horizon, 4, dtype=torch.float64)

        with torch.no_grad():
            _, _, u, _ = controller.forward(x0, w, v)

        assert torch.isfinite(u).all()
        assert u.shape == (5, horizon, 2)
        assert torch.all(u.abs() <= u_max + 1e-6)

    @pytest.mark.parametrize("regime,period_or_block", _LTV_REGIMES)
    def test_riccati_derived_gradient_coefficients_vary_across_time(
        self, regime: LTVRegime, period_or_block: int | None
    ) -> None:
        """A positive check that the fix isn't ACCIDENTALLY correct (e.g.
        still broadcasting a single slice everywhere): under real time
        variation the per-step M/C the true-P branch computes must actually
        differ across t, not just avoid crashing."""
        horizon = 12
        problem = LTVLQRProblemFactory(
            state_dim=4,
            control_dim=2,
            horizon=horizon,
            seed=3,
            regime=regime,
            period_or_block=period_or_block,
            variation_strength=0.5,
        ).build()
        system, cost = require_linear_quadratic(problem)
        A = np.stack([system.A_t[k] for k in range(horizon)])
        B = np.stack([system.B_t[k] for k in range(horizon)])
        P_arr, _ = finite_horizon_riccati(system, cost, horizon)
        M_stack, _ = get_lqr_gradient_matrices(P_arr, A, B, cost.R)
        assert not np.allclose(M_stack[0], M_stack[-1])


class TestSnapshotAndReplayContract:
    """Sec 6.4: "A_t"/"B_t" are added additively, and landscape replay
    prefers them (sliced at t_star), falling back to "A"/"B" for
    artifacts that predate this change."""

    def test_convergence_parameters_carries_full_per_step_stacks(self) -> None:
        from mbl.applications.recipes.unfolded import UnfoldedRecipe
        from mbl.engine.training_plan import OptimizerSpec, TrainingPlan

        horizon = 8
        problem = LTVLQRProblemFactory(
            state_dim=3,
            control_dim=2,
            horizon=horizon,
            seed=1,
            regime=LTVRegime.PERIODIC,
            period_or_block=4,
            variation_strength=0.4,
        ).build()
        recipe = UnfoldedRecipe(
            kind=UnfoldedKind.LEARNED_STEP_SIZE,
            plan=TrainingPlan(optimizer=OptimizerSpec("adam", 0.05), epochs=1),
            num_iterations=3,
            step_size_init=0.05,
            step_size_max=1.0,
            horizon=horizon,
        )
        controller = recipe.build_controller(problem, _CTX)
        parameters = unfolded_convergence_parameters(
            controller, problem, horizon=horizon
        )

        assert parameters["A_t"].shape == (horizon, 3, 3)
        assert parameters["B_t"].shape == (horizon, 3, 2)
        np.testing.assert_array_equal(parameters["A"], parameters["A_t"][0])
        np.testing.assert_array_equal(parameters["B"], parameters["B_t"][0])
        # Genuinely time-varying: not every slice of A_t is the same matrix.
        assert not np.allclose(parameters["A_t"][0], parameters["A_t"][1])

    def test_lti_snapshot_a_equals_a_t_every_slice(self) -> None:
        """For an LTI problem A_t must be `horizon` copies of the same 2D
        matrix, and "A" must equal every one of them -- the degenerate case
        the fallback logic below relies on being harmless."""
        from mbl.applications.recipes.unfolded import UnfoldedRecipe
        from mbl.engine.training_plan import OptimizerSpec, TrainingPlan

        horizon = 6
        problem = LQRProblemFactory(
            state_dim=3, control_dim=2, horizon=horizon, seed=0
        ).build()
        recipe = UnfoldedRecipe(
            kind=UnfoldedKind.LEARNED_STEP_SIZE,
            plan=TrainingPlan(optimizer=OptimizerSpec("adam", 0.05), epochs=1),
            num_iterations=3,
            step_size_init=0.05,
            step_size_max=1.0,
            horizon=horizon,
        )
        controller = recipe.build_controller(problem, _CTX)
        parameters = unfolded_convergence_parameters(
            controller, problem, horizon=horizon
        )

        for t in range(horizon):
            np.testing.assert_array_equal(parameters["A_t"][t], parameters["A"])

    def _result(self, arrays: dict) -> ContenderResult:
        return ContenderResult(
            label="unfolded_alpha_p",
            family="unfolded_warmstart",
            cache_key="k",
            content_digest="d",
            disposition="fresh",
            run_dir=None,
            metrics={},
            arrays=arrays,
        )

    def _base_artifacts(self, n: int, m: int, horizon: int) -> dict:
        return {
            "trajectory_states": np.zeros((5, horizon + 1, n)),
            "parameter_step_size": np.zeros((3, m)),
            "parameter_P": np.eye(n),
            "parameter_A": np.full((n, n), -1.0),
            "parameter_B": np.full((n, m), -1.0),
            "parameter_R": np.eye(m),
        }

    def test_prefers_a_t_sliced_at_t_star_when_present(self) -> None:
        n, m, horizon, t_star = 3, 2, 6, 4
        artifacts = self._base_artifacts(n, m, horizon)
        A_t = np.stack([np.full((n, n), float(t)) for t in range(horizon)])
        B_t = np.stack([np.full((n, m), float(t)) for t in range(horizon)])
        artifacts["parameter_A_t"] = A_t
        artifacts["parameter_B_t"] = B_t

        inputs = unfolding_landscape_inputs(self._result(artifacts), t_star=t_star)

        np.testing.assert_array_equal(inputs.A, A_t[t_star])
        np.testing.assert_array_equal(inputs.B, B_t[t_star])
        # Must NOT have fallen back to the (deliberately different) "A"/"B".
        assert not np.allclose(inputs.A, artifacts["parameter_A"])

    def test_falls_back_to_a_and_b_when_a_t_absent(self) -> None:
        """A pre-existing NB03/NB04 cached artifact (persisted before this
        change) has no "A_t"/"B_t" key -- landscape replay must still work,
        unchanged, off the single "A"/"B" matrix."""
        n, m, horizon, t_star = 3, 2, 6, 4
        artifacts = self._base_artifacts(n, m, horizon)

        inputs = unfolding_landscape_inputs(self._result(artifacts), t_star=t_star)

        np.testing.assert_array_equal(inputs.A, artifacts["parameter_A"])
        np.testing.assert_array_equal(inputs.B, artifacts["parameter_B"])

    def test_landscape_inputs_a_stays_2d_either_way(self) -> None:
        """No renderer change is needed downstream (Sec 6.4's promise) only
        if `.A`/`.B` are always plain 2D matrices regardless of which path
        was taken."""
        n, m, horizon, t_star = 3, 2, 6, 4
        artifacts = self._base_artifacts(n, m, horizon)
        artifacts["parameter_A_t"] = np.stack([np.eye(n) for _ in range(horizon)])
        artifacts["parameter_B_t"] = np.stack(
            [np.zeros((n, m)) for _ in range(horizon)]
        )

        with_stack = unfolding_landscape_inputs(self._result(artifacts), t_star=t_star)
        del artifacts["parameter_A_t"], artifacts["parameter_B_t"]
        without_stack = unfolding_landscape_inputs(
            self._result(artifacts), t_star=t_star
        )

        assert with_stack.A.ndim == 2
        assert without_stack.A.ndim == 2
