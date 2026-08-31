"""Phase C acceptance tests for the LTV reference floors
(docs/planning/03_studies/nb05_ltv_box_constrained/ltv_box_constrained_lqr_benchmark.md Sec 5/6.6/12):

- Floor A is cross-checked against a FULLY INDEPENDENT closed-form
  computation (a hand-rolled zero-terminal backward Riccati sweep plus the
  classical LQG formula ``tr(P_0 Sigma_0) + sum_t tr(P_{t+1} W)``), not
  merely re-derived from the same production helpers it is meant to verify.
- Floor B's exact envelope-theorem gradient is cross-checked against finite
  differences at a random feasible point.
- The bracket ``j_lqr_fin <= j_box_fin <= every contender's evaluated
  cost`` holds -- checked against a real, feasible SATURATED policy (the
  LTV analogue of Truncated-Riccati: the zero-terminal-cost gains, clipped
  onto the box), which is a real attained upper reference even though NB05
  drops COCP-LB (its own attained-upper-bound policy).
- ``g(0) == j_lqr_fin`` exactly (Sec 5.3).
- NB04's floors must never silently apply to an LTV problem: reusing
  `compute_box_constrained_floors` on the SAME LTV problem gives a
  DIFFERENT (and wrong -- t=0-frozen) answer, which this suite pins down
  concretely rather than merely asserting by prose.
"""

import numpy as np
import pytest
import torch

from mbl.applications.ltv_factories import LTVLQRProblemFactory, LTVRegime
from mbl.applications.studies.nb04_box_constrained import compute_box_constrained_floors
from mbl.applications.studies.nb05_ltv_box_constrained import (
    LTVFloors,
    _BoxFloorContext,
    _box_floor_value_and_grad,
    _zero_terminal_cost,
    compute_ltv_floors,
)
from mbl.models.guards import require_linear_quadratic

_REGIMES = [
    (LTVRegime.PERIODIC, 4),
    (LTVRegime.BLOCK_CONSTANT, 4),
    (LTVRegime.FULLY_VARYING, None),
]


def _ltv_problem(
    regime: LTVRegime,
    period_or_block: int | None,
    *,
    seed: int = 5,
    variation_strength: float = 0.4,
    u_max: float = 0.5,
    horizon: int = 12,
):
    return LTVLQRProblemFactory(
        state_dim=4,
        control_dim=2,
        horizon=horizon,
        seed=seed,
        regime=regime,
        period_or_block=period_or_block,
        variation_strength=variation_strength,
        u_max=u_max,
    ).build()


class TestFloorAIndependentCrossCheck:
    """Sec 5.2/12: Floor A must match a computation that shares NONE of
    `compute_ltv_floors`'s own code -- only the mathematical claim (the
    classical LQG closed form)."""

    @pytest.mark.parametrize("regime,period_or_block", _REGIMES)
    def test_matches_hand_rolled_zero_terminal_riccati_closed_form(
        self, regime: LTVRegime, period_or_block: int | None
    ) -> None:
        problem = _ltv_problem(regime, period_or_block)
        floors = compute_ltv_floors(
            problem,
            process_noise_std=0.6,
            initial_state_std=1.2,
            u_max=0.5,
            compute_box_floor=False,
        )

        system, _ = require_linear_quadratic(problem)
        n, m, horizon = 4, 2, 12
        A = [system.A_t[t] for t in range(horizon)]
        B = [system.B_t[t] for t in range(horizon)]
        Q, R = np.eye(n), np.eye(m)
        W = 0.6**2 * np.eye(n)
        Sigma0 = 1.2**2 * np.eye(n)

        P = np.zeros((n, n))
        P_list = [P]
        for t in reversed(range(horizon)):
            M = R + B[t].T @ P @ B[t]
            C = B[t].T @ P @ A[t]
            K = np.linalg.solve(M, C)
            P = Q + A[t].T @ P @ (A[t] - B[t] @ K)
            P_list.append(P)
        P_list.reverse()  # P_list[t] = P_t, t = 0..horizon

        closed_form = (
            np.trace(P_list[0] @ Sigma0)
            + sum(np.trace(P_list[t + 1] @ W) for t in range(horizon))
        ) / horizon

        assert floors.j_lqr_fin == pytest.approx(closed_form, rel=1e-9)

    def test_zero_terminal_cost_rejects_non_time_stacked_q(self) -> None:
        from mbl.core.cost.quadratic_cost import QuadraticCost

        cost_2d = QuadraticCost(Q=np.eye(3), R=np.eye(2))
        with pytest.raises(ValueError, match="time-stacked"):
            _zero_terminal_cost(cost_2d)


class TestFloorBGradientCrossCheck:
    """Sec 5.3/12: the envelope-theorem gradient must match finite
    differences at a random FEASIBLE point (not just at lambda=0)."""

    def test_gradient_matches_finite_differences(self) -> None:
        problem = _ltv_problem(
            LTVRegime.BLOCK_CONSTANT, 4, seed=11, variation_strength=0.5
        )
        system, cost = require_linear_quadratic(problem)
        n, m, horizon = 4, 2, 12
        W = 0.5**2 * np.eye(n)
        Sigma0 = np.eye(n)
        ctx = _BoxFloorContext(
            system=system,
            Q=cost.Q,
            R=cost.R,
            W=W,
            Sigma0=Sigma0,
            u_max=0.5,
            horizon=horizon,
        )

        rng = np.random.default_rng(0)
        lam0 = 0.3 * rng.random(horizon * m)
        _, analytic_grad = _box_floor_value_and_grad(lam0, ctx)

        eps = 1e-6
        fd_grad = np.empty_like(lam0)
        for i in range(lam0.size):
            lam_p, lam_m = lam0.copy(), lam0.copy()
            lam_p[i] += eps
            lam_m[i] -= eps
            f_p, _ = _box_floor_value_and_grad(lam_p, ctx)
            f_m, _ = _box_floor_value_and_grad(lam_m, ctx)
            fd_grad[i] = (f_p - f_m) / (2 * eps)

        np.testing.assert_allclose(analytic_grad, fd_grad, rtol=1e-4, atol=1e-6)

    def test_g_at_zero_equals_j_lqr_fin_exactly(self) -> None:
        problem = _ltv_problem(LTVRegime.PERIODIC, 4, seed=2)
        floors = compute_ltv_floors(
            problem,
            process_noise_std=0.5,
            initial_state_std=1.0,
            u_max=0.5,
            compute_box_floor=False,
        )
        system, cost = require_linear_quadratic(problem)
        n, m, horizon = 4, 2, 12
        W = 0.5**2 * np.eye(n)
        Sigma0 = np.eye(n)
        ctx = _BoxFloorContext(
            system=system,
            Q=cost.Q,
            R=cost.R,
            W=W,
            Sigma0=Sigma0,
            u_max=0.5,
            horizon=horizon,
        )
        neg_g_at_zero, _ = _box_floor_value_and_grad(np.zeros(horizon * m), ctx)
        assert -neg_g_at_zero == pytest.approx(floors.j_lqr_fin, rel=1e-9)


class TestFloorsBracket:
    """Sec 5.4/9/12: j_lqr_fin <= j_box_fin <= a real attained policy's cost,
    in every regime -- the defining property of a lower bound."""

    @pytest.mark.parametrize("regime,period_or_block", _REGIMES)
    def test_j_box_fin_is_at_least_j_lqr_fin(
        self, regime: LTVRegime, period_or_block: int | None
    ) -> None:
        problem = _ltv_problem(regime, period_or_block)
        floors = compute_ltv_floors(
            problem, process_noise_std=0.5, initial_state_std=1.0, u_max=0.5
        )
        assert floors.j_box_fin is not None
        assert floors.j_box_fin >= floors.j_lqr_fin - 1e-6

    @pytest.mark.parametrize("regime,period_or_block", _REGIMES)
    def test_floors_are_finite(
        self, regime: LTVRegime, period_or_block: int | None
    ) -> None:
        problem = _ltv_problem(regime, period_or_block)
        floors = compute_ltv_floors(
            problem, process_noise_std=0.5, initial_state_std=1.0, u_max=0.5
        )
        assert np.isfinite(floors.j_lqr_fin)
        assert floors.j_box_fin is not None and np.isfinite(floors.j_box_fin)

    @pytest.mark.parametrize("regime,period_or_block", _REGIMES)
    def test_both_floors_bracket_a_real_saturated_policys_attained_cost(
        self, regime: LTVRegime, period_or_block: int | None
    ) -> None:
        """The LTV analogue of Truncated-Riccati (Sec 5.4 notes NB05 has no
        attained upper reference like COCP-LB; this builds the cheapest
        possible one -- the zero-terminal-cost gains, saturated onto the
        box -- purely for this test's own bracket check, not as a
        recommendation for a new contender)."""
        u_max = 0.5
        problem = _ltv_problem(regime, period_or_block, u_max=u_max)
        floors = compute_ltv_floors(
            problem, process_noise_std=0.5, initial_state_std=1.0, u_max=u_max
        )

        system, cost = require_linear_quadratic(problem)
        horizon = 12
        K_arr = floors.k_arr

        batch = 4000
        rng = torch.Generator().manual_seed(0)
        x = torch.randn(batch, 4, dtype=torch.float64, generator=rng)
        total = torch.zeros(batch, dtype=torch.float64)
        for t in range(horizon):
            A_t = torch.as_tensor(system.A_t[t], dtype=torch.float64)
            B_t = torch.as_tensor(system.B_t[t], dtype=torch.float64)
            K_t = torch.as_tensor(K_arr[t], dtype=torch.float64)
            u = torch.clamp(-x @ K_t.T, min=-u_max, max=u_max)
            total = total + (x * x).sum(-1) + (u * u).sum(-1)
            w = 0.5 * torch.randn(batch, 4, dtype=torch.float64, generator=rng)
            x = x @ A_t.T + u @ B_t.T + w
        attained_cost = float((total / horizon).mean())

        assert floors.j_box_fin is not None
        assert floors.j_lqr_fin <= attained_cost + 1e-2
        assert floors.j_box_fin <= attained_cost + 1e-2


class TestNB04FloorsMustNotBeReusedOnLTV:
    """Sec 5.1/12: pins down CONCRETELY that NB04's (LTI, infinite-horizon)
    floors give a DIFFERENT -- and invalid -- answer on an LTV problem,
    rather than merely asserting this in prose. `compute_box_constrained_floors`
    only reads `A_t[0]`/`B_t[0]`, so it silently scores the plant FROZEN at
    t=0; this test proves that frozen-slice answer diverges from the true
    LTV floor whenever the system genuinely varies over time."""

    def test_nb04_floor_diverges_from_the_true_ltv_floor(self) -> None:
        problem = _ltv_problem(
            LTVRegime.FULLY_VARYING, None, seed=13, variation_strength=0.8
        )
        ltv_floors = compute_ltv_floors(
            problem,
            process_noise_std=0.5,
            initial_state_std=1.0,
            u_max=0.5,
            compute_box_floor=False,
        )
        nb04_floors = compute_box_constrained_floors(
            problem, process_noise_std=0.5, u_max=0.5
        )

        assert ltv_floors.j_lqr_fin != pytest.approx(nb04_floors.j_lqr, rel=1e-6)

    def test_nb04_floor_matches_at_zero_variation_strength(self) -> None:
        """The one case where reuse WOULD be harmless -- variation_strength
        == 0 collapses to the LTI degenerate problem, where "frozen at t=0"
        and "the true system" are the same system. Confirms the divergence
        above is really about time variation, not an unrelated numerical
        difference between the two code paths.

        Not exactly equal even here: Floor A is a FINITE-horizon
        time-averaged cost, NB04's j_lqr an INFINITE-horizon steady-state
        cost -- two different (if related) quantities that only converge
        as the horizon grows (verified empirically: relative gap shrinks
        0.21 -> 0.05 -> 0.02 -> 0.01 at horizon 12/50/100/300). A horizon of
        100 keeps this test fast while giving comfortable margin under a
        10% tolerance -- not a flaky near-boundary assertion.
        """
        problem = _ltv_problem(
            LTVRegime.FULLY_VARYING, None, seed=13, variation_strength=0.0, horizon=100
        )
        ltv_floors = compute_ltv_floors(
            problem,
            process_noise_std=0.5,
            initial_state_std=1.0,
            u_max=200.0,  # loose box: floors approach the pure LQR trace
            compute_box_floor=False,
        )
        nb04_floors = compute_box_constrained_floors(
            problem, process_noise_std=0.5, u_max=200.0
        )
        assert ltv_floors.j_lqr_fin == pytest.approx(nb04_floors.j_lqr, rel=0.1)


class TestComputeLtvFloorsSkipFlag:
    def test_compute_box_floor_false_skips_the_box_floor(self) -> None:
        problem = _ltv_problem(LTVRegime.PERIODIC, 4)
        floors = compute_ltv_floors(
            problem,
            process_noise_std=0.5,
            initial_state_std=1.0,
            u_max=0.5,
            compute_box_floor=False,
        )
        assert isinstance(floors, LTVFloors)
        assert floors.j_box_fin is None
        assert floors.lambda_star is None
        assert floors.box_gradient_norm is None
        assert np.isfinite(floors.j_lqr_fin)

    def test_k_arr_is_consistent_regardless_of_the_flag(self) -> None:
        problem = _ltv_problem(LTVRegime.PERIODIC, 4)
        floors_a_only = compute_ltv_floors(
            problem,
            process_noise_std=0.5,
            initial_state_std=1.0,
            u_max=0.5,
            compute_box_floor=False,
        )
        floors_both = compute_ltv_floors(
            problem, process_noise_std=0.5, initial_state_std=1.0, u_max=0.5
        )
        np.testing.assert_array_equal(floors_a_only.k_arr, floors_both.k_arr)
        assert floors_a_only.j_lqr_fin == floors_both.j_lqr_fin
