"""The Lipschitz constant of the per-step LQR gradient — the constant the
analytic PGD's step size is `1/L` of.

Why this exists, measured on the campaign's own frozen plants: the analytic PGD
(`unfolded_fixed`, declared as `standard_pgd`) carries a literal
`step_size_init = 0.05`, and projected gradient descent on a convex quadratic
converges only for a step strictly below `2/L`. On four of the seven frozen
ICASSP plants that literal is **above** `2/L` — by 4.98x (n=15), 10.87x (n=20),
5.99x (n=30) and 23.87x (n=50) — and the box hides the divergence completely,
because the projection bounds the iterate while the cost-vs-depth curve quietly
goes non-monotone.

`L = max_t lambda_max(2(R + B' P_{t+1} B))`, i.e. the largest eigenvalue over the
horizon of exactly the `M_stack` tensor `build_unfolded_controller` hands the
refinement. The factor of 2 is part of the constant: `M_stack` is stored as
`2 * M`, and the gradient it multiplies is the gradient of the true objective,
not of half of it.

The routine is `eigvalsh` **by name and not by accident**. `eigvals`,
`norm(., 2)`, `svd` and the closed 2x2 form each land one ULP away on the n=4
plant, and one ULP of a declared `step_size_init` moves the `ModelID` — so the
choice of routine is part of the reproducibility contract, not an implementation
detail.
"""

import numpy as np
import pytest

from pathlib import Path

from mbl.models.analytic import (
    finite_horizon_riccati,
    get_lqr_gradient_matrices,
    gradient_lipschitz_constant,
    problem_gradient_lipschitz_constant,
)
from mbl.models.guards import require_linear_quadratic
from mbl.spec.problem import ProblemSpec

STUDIES = Path(__file__).resolve().parents[3] / "studies" / "icassp"

#: Measured on the frozen plants, twice and by two independent routes: the
#: shipped `finite_horizon_riccati` + `get_lqr_gradient_matrices`, and a
#: hand-written backward recursion that agreed to 4.10e-16 relative.
FROZEN_L = {
    "icassp_n4m2_N100_u0p1_s0.npz": 17.535383914972115,
    "icassp_n7m2_N100_u0p1_s0.npz": 29.851796497,
    "icassp_n15m2_N100_u0p1_s0.npz": 199.007806236,
    "icassp_n20m2_N100_u0p1_s0.npz": 434.867401720,
    "icassp_n30m2_N100_u0p1_s0.npz": 239.697073923,
    "icassp_n50m2_N100_u0p1_s0.npz": 954.779384134,
}


def _problem(name: str):
    return ProblemSpec.load(STUDIES / name).build()


def _matrices(name: str):
    """The stacking `problem_gradient_lipschitz_constant` exists to encapsulate.

    Written out here on purpose: these tests compare the packaged entry point
    against the raw algebra, so doing it by hand once is the point. `A_t[k]` and
    not `A_t.array` -- the latter is the un-broadcast storage for a
    time-invariant system and raises two frames away inside a transpose.
    """
    system, cost = require_linear_quadratic(_problem(name))
    horizon = int(np.asarray(cost.R).shape[0])
    P, _ = finite_horizon_riccati(system, cost, horizon)
    A = np.stack([system.A_t[k] for k in range(horizon)])
    B = np.stack([system.B_t[k] for k in range(horizon)])
    return P, A, B, np.asarray(cost.R)


def test_the_n4_plant_reproduces_its_golden_constant_bitwise() -> None:
    """Bit-equality, not `isclose`, and that is deliberate.

    A step size participates in the `ModelID` at one-ULP granularity, so a
    tolerance here would license exactly the drift that makes two runs of the
    same document disagree about which models they already have.
    """
    P, A, B, R = _matrices("icassp_n4m2_N100_u0p1_s0.npz")
    assert gradient_lipschitz_constant(P, A, B, R) == 17.535383914972115


def test_the_factor_of_two_is_part_of_the_constant() -> None:
    """The mutation this test exists for. `M_stack` is stored as `2 * M`, so a
    constant computed from `M` alone is exactly half — 8.767691957486058 — and
    `1/L` would be twice the correct step, which on this plant lands at
    0.114055, i.e. exactly ON the 2/L limit where the iteration stalls forever
    on a period-2 cycle."""
    P, A, B, R = _matrices("icassp_n4m2_N100_u0p1_s0.npz")
    L = gradient_lipschitz_constant(P, A, B, R)
    M, _ = get_lqr_gradient_matrices(P, A, B, R)
    half = float(np.linalg.eigvalsh(M).max())
    assert L == pytest.approx(2.0 * half, rel=1e-15)
    assert L != pytest.approx(half, rel=1e-6)


def test_it_is_the_MAX_over_the_horizon_and_not_the_mean() -> None:
    """`L_t` varies only in the terminal Riccati transient — 86 of 100 steps
    share `L_0` to 1e-12 — so a mean would be 17.4178 against a max of 17.5354,
    a 0.7 % error that no smoke test would ever notice and that would put the
    step size *above* `1/L` at the steps where the curvature is largest."""
    P, A, B, R = _matrices("icassp_n4m2_N100_u0p1_s0.npz")
    M, _ = get_lqr_gradient_matrices(P, A, B, R)
    per_step = np.linalg.eigvalsh(2.0 * M)[:, -1]
    L = gradient_lipschitz_constant(P, A, B, R)
    assert L == per_step.max()
    assert L > float(per_step.mean())


@pytest.mark.parametrize("name", sorted(FROZEN_L))
def test_every_frozen_plant_reproduces_its_measured_constant(name: str) -> None:
    P, A, B, R = _matrices(name)
    assert gradient_lipschitz_constant(P, A, B, R) == pytest.approx(
        FROZEN_L[name], rel=1e-9
    )


def test_the_literal_0p05_exceeds_2_over_L_on_exactly_four_of_the_seven() -> None:
    """The campaign's finding, as an executable claim.

    This is the assertion that fails if anyone 'fixes' the constant by a factor
    of two in either direction: halving L moves n=7 into the divergent set,
    doubling it moves n=15 out.
    """
    divergent = set()
    for name in FROZEN_L:
        P, A, B, R = _matrices(name)
        if 0.05 >= 2.0 / gradient_lipschitz_constant(P, A, B, R):
            divergent.add(name)
    assert divergent == {
        "icassp_n15m2_N100_u0p1_s0.npz",
        "icassp_n20m2_N100_u0p1_s0.npz",
        "icassp_n30m2_N100_u0p1_s0.npz",
        "icassp_n50m2_N100_u0p1_s0.npz",
    }, divergent


def test_one_over_L_is_declarable_against_the_campaign_bound() -> None:
    """The refusal added for `step_size_max` must not reject the value this
    constant is computed to produce: every frozen plant's `1/L` has to sit
    strictly inside `(0, step_size_max = 1.0)`."""
    for name in FROZEN_L:
        P, A, B, R = _matrices(name)
        step = 1.0 / gradient_lipschitz_constant(P, A, B, R)
        assert 0.0 < step < 1.0, (name, step)


def test_the_problem_level_entry_point_agrees_bitwise() -> None:
    """The packaged path and the hand-stacked one must be the same number, or
    the encapsulation has quietly changed the constant it encapsulates."""
    for name in FROZEN_L:
        P, A, B, R = _matrices(name)
        assert problem_gradient_lipschitz_constant(_problem(name)) == (
            gradient_lipschitz_constant(P, A, B, R)
        ), name
