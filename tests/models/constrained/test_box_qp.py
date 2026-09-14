"""Phase A acceptance for the exact batched box-QP.

The gate is the problem's own optimality conditions, evaluated on the returned
point -- not agreement with another implementation. An independent interior-point
solve cross-checks a subset, so the certificate itself is corroborated rather
than merely trusted.

Plants are generated at the campaign's own shapes (n = 4, 7, 30, 50, 100 with its
m values, rho(A) = 0.999, u_max = 0.1) and the query point is the real one: the
DARE cost-to-go, which is what `COCPRecipe` seeds a controller with. `conftest`
deliberately isolates tests from the repository, so nothing is read from disk.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pytest
import torch
from scipy.linalg import solve_discrete_are

from mbl.models.constrained.box_qp import (
    box_qp,
    box_qp_tolerances,
    certify_box_qp,
    solve_box_qp,
)

U_MAX = 0.1
RHO = 0.999

#: The campaign's shapes. n = 4 and n = 7 are where Figure 1 and the dimension
#: grid live; n = 100, m = 30 is the instance the shipped path cannot reach.
SHAPES = [(4, 2), (7, 2), (30, 2), (50, 2), (100, 30)]


def _plant(n: int, m: int, seed: int = 0) -> tuple[np.ndarray, ...]:
    """A stable plant at the campaign's spectral radius, with its DARE cost-to-go."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    A *= RHO / np.max(np.abs(np.linalg.eigvals(A)))
    B = rng.normal(size=(n, m))
    Q, R = np.eye(n), np.eye(m)
    P = solve_discrete_are(A, B, Q, R)
    return A, B, Q, R, P


def _reduced(
    n: int, m: int, batch: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """`(H, c)` of the reduced program, over states spanning three regimes."""
    A, B, Q, R, P = _plant(n, m, seed)
    H = 2.0 * (R + B.T @ P @ B)
    M = 2.0 * (A.T @ P @ B)
    rng = np.random.default_rng(seed + 1)
    per = batch // 4
    states = np.concatenate(
        [
            rng.normal(scale=0.5, size=(batch - 2 * per, n)),  # in distribution
            rng.normal(scale=5.0, size=(per, n)),  # far out of distribution
            rng.normal(scale=0.01, size=(per, n)),  # near the origin
        ]
    )
    return H, states @ M


def _threshold(H: np.ndarray, c: np.ndarray, dtype: torch.dtype) -> float:
    """A residual threshold scaled by the problem, not fitted to the answer.

    A stationarity residual is a gradient, so it inherits the scale of ``H u``
    and ``c``; comparing it to a bare epsilon would be a different assertion at
    every plant.
    """
    eps = float(torch.finfo(dtype).eps)
    scale = float(np.abs(H).sum(axis=1).max()) * U_MAX + float(np.abs(c).max())
    return 100.0 * eps * max(scale, 1.0)


@pytest.mark.parametrize("n,m", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_certificate_holds_over_eight_thousand_states(
    n: int, m: int, dtype: torch.dtype
) -> None:
    """Every KKT residual is within the problem-scaled tolerance, everywhere."""
    H, c = _reduced(n, m, batch=8000)
    solution = solve_box_qp(
        torch.tensor(H, dtype=dtype), torch.tensor(c, dtype=dtype), U_MAX
    )
    assert solution.converged, f"active set never settled at n={n} ({dtype})"
    cert = solution.certificate
    assert cert.satisfies(_threshold(H, c, dtype)), (
        f"n={n} {dtype}: stationarity={cert.stationarity:.3e} "
        f"sign={cert.sign_violation:.3e} box={cert.box_violation:.3e}"
    )
    # The box is respected exactly, not to a tolerance -- the clamp guarantees it.
    assert cert.box_violation == 0.0


@pytest.mark.parametrize("n,m", SHAPES)
def test_matches_an_independent_interior_point_solve(n: int, m: int) -> None:
    """Corroborate the certificate against a solver from a different family."""
    H, c = _reduced(n, m, batch=64)
    u = solve_box_qp(torch.tensor(H), torch.tensor(c), U_MAX).u.numpy()
    rng = np.random.default_rng(7)
    for index in rng.choice(c.shape[0], size=4, replace=False):
        variable = cp.Variable(m)
        problem = cp.Problem(
            cp.Minimize(
                0.5 * cp.quad_form(variable, cp.psd_wrap(H)) + c[index] @ variable
            ),
            [cp.norm(variable, "inf") <= U_MAX],
        )
        problem.solve(solver=cp.CLARABEL, tol_gap_abs=1e-12, tol_gap_rel=1e-12)
        assert np.linalg.norm(u[index] - variable.value) < 1e-9


def test_shared_and_per_element_hessians_agree() -> None:
    """`H` may be shared or per-element; the two paths must not diverge."""
    H, c = _reduced(30, 2, batch=256)
    Ht, ct = torch.tensor(H), torch.tensor(c)
    shared = solve_box_qp(Ht, ct, U_MAX).u
    per_element = solve_box_qp(Ht.expand(256, 2, 2).clone(), ct, U_MAX).u
    torch.testing.assert_close(shared, per_element, rtol=0.0, atol=0.0)


def test_unconstrained_problem_returns_the_unconstrained_minimiser() -> None:
    """With a bound no solution reaches, the answer is `-H^-1 c` and nothing is active."""
    H, c = _reduced(7, 2, batch=128)
    solution = solve_box_qp(torch.tensor(H), torch.tensor(c), u_max=1e6)
    expected = np.linalg.solve(H, -c.T).T
    assert solution.certificate.active_count == 0
    torch.testing.assert_close(
        solution.u, torch.tensor(expected), rtol=1e-10, atol=1e-12
    )


# --- the acceptance must be capable of failing -------------------------------


def test_a_truncated_iteration_budget_fails_the_certificate() -> None:
    """The gate is not vacuous: one iteration is not enough at n = 100."""
    H, c = _reduced(100, 30, batch=512)
    starved = solve_box_qp(torch.tensor(H), torch.tensor(c), U_MAX, max_iter=1)
    assert not starved.converged
    assert not starved.certificate.satisfies(_threshold(H, c, torch.float64))


def test_certificate_rejects_a_perturbed_point() -> None:
    """`certify_box_qp` grades a point, so a wrong point must fail it."""
    H, c = _reduced(30, 2, batch=64)
    Ht, ct = torch.tensor(H), torch.tensor(c)
    u = solve_box_qp(Ht, ct, U_MAX).u
    assert certify_box_qp(Ht, ct, u, U_MAX).satisfies(_threshold(H, c, torch.float64))
    nudged = u.clone()
    nudged[0, 0] += 1e-3
    assert not certify_box_qp(Ht, ct, nudged, U_MAX).satisfies(
        _threshold(H, c, torch.float64)
    )


def test_tolerances_scale_with_the_working_precision() -> None:
    """A float32 tolerance that inherited a float64 literal would be unreachable."""
    f64, _ = box_qp_tolerances(torch.float64, U_MAX)
    f32, _ = box_qp_tolerances(torch.float32, U_MAX)
    assert f32 > f64 * 1e6


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"u_max": 0.0}, "u_max must be positive"),
        ({"u_max": -1.0}, "u_max must be positive"),
    ],
)
def test_refuses_a_non_positive_bound(kwargs: dict, message: str) -> None:
    H, c = _reduced(4, 2, batch=8)
    with pytest.raises(ValueError, match=message):
        solve_box_qp(torch.tensor(H), torch.tensor(c), **kwargs)


def test_refuses_a_hessian_of_the_wrong_shape() -> None:
    H, c = _reduced(4, 2, batch=8)
    with pytest.raises(ValueError, match="H must be"):
        solve_box_qp(torch.tensor(H)[:1], torch.tensor(c), U_MAX)


# --- Phase B: the analytic adjoint -------------------------------------------


def _clarabel(H: np.ndarray, c_row: np.ndarray, u_max: float) -> np.ndarray:
    """An interior-point solution, independent of anything in `box_qp`."""
    variable = cp.Variable(H.shape[0])
    problem = cp.Problem(
        cp.Minimize(0.5 * cp.quad_form(variable, cp.psd_wrap(H)) + c_row @ variable),
        [cp.norm(variable, "inf") <= u_max],
    )
    problem.solve(solver=cp.CLARABEL, tol_gap_abs=1e-12, tol_gap_rel=1e-12)
    return np.asarray(variable.value).ravel()


@pytest.mark.parametrize("n,m", [(4, 2), (100, 30)])
@pytest.mark.parametrize("dtype,tol", [(torch.float64, 1e-7), (torch.float32, 2e-3)])
def test_adjoint_matches_finite_differences_on_an_independent_solver(
    n: int, m: int, dtype: torch.dtype, tol: float
) -> None:
    """The gradient is checked against a method that shares no code with it.

    Central differences are taken on Clarabel's solution, so neither this
    module's forward pass nor its adjoint appears in the reference.
    """
    H, c = _reduced(n, m, batch=1)
    Ht = torch.tensor(H, dtype=dtype, requires_grad=True)
    ct = torch.tensor(c, dtype=dtype, requires_grad=True)
    box_qp(Ht, ct, U_MAX).sum().backward()
    assert Ht.grad is not None and ct.grad is not None

    rng = np.random.default_rng(3)
    step = 1e-6

    # PERTURB H SYMMETRICALLY. The quadratic form sees only (H + H^T)/2, so a
    # one-sided nudge of an off-diagonal entry is not a well-posed derivative
    # to compare against: any solver is free to symmetrise, and cvxpy does.
    # The well-posed statement pairs the symmetric perturbation with the sum of
    # the two partials, and is independent of that choice.
    def _nudge(base: np.ndarray, i: int, j: int, delta: float) -> np.ndarray:
        """Symmetric nudge -- and exactly once on the diagonal, where i == j."""
        out = base.copy()
        out[i, j] += delta
        if i != j:
            out[j, i] += delta
        return out

    pairs = [(int(i), int(j)) for i, j in rng.integers(0, m, size=(5, 2))]
    pairs.append((0, 0))  # the diagonal is a different case; do not leave it to chance
    for i, j in pairs:
        expected = float(Ht.grad[i, j]) + (float(Ht.grad[j, i]) if i != j else 0.0)
        plus = _clarabel(_nudge(H, i, j, step), c[0], U_MAX).sum()
        minus = _clarabel(_nudge(H, i, j, -step), c[0], U_MAX).sum()
        finite = (plus - minus) / (2 * step)
        assert abs(expected - finite) < tol, f"dH[{i},{j}]: {expected} vs {finite}"

    for k in rng.integers(0, m, size=6):
        shifted = c[0].copy()
        shifted[k] += step
        plus = _clarabel(H, shifted, U_MAX).sum()
        shifted[k] -= 2 * step
        minus = _clarabel(H, shifted, U_MAX).sum()
        finite = (plus - minus) / (2 * step)
        assert abs(float(ct.grad[0, k]) - finite) < tol, f"dc[{k}]"


def test_gradcheck_where_the_active_set_has_margin() -> None:
    """`torch.autograd.gradcheck`, at a point the solution map is smooth at.

    The map is piecewise affine, so gradcheck is only meaningful away from a
    face boundary; the margin is asserted rather than assumed.
    """
    H, c = _reduced(7, 2, batch=6)
    Ht = torch.tensor(H, requires_grad=True)
    ct = torch.tensor(c, requires_grad=True)
    solution = solve_box_qp(torch.tensor(H), torch.tensor(c), U_MAX)
    # Smoothness needs BOTH: free entries clear of a bound, and active ones held
    # by a gradient that is not itself near zero. Either alone permits a face
    # boundary the perturbation could cross.
    gradient = solution.u @ torch.tensor(H) + torch.tensor(c)
    free_margin = (U_MAX - solution.u.abs())[~solution.active]
    held_margin = gradient.abs()[solution.active]
    assert float(free_margin.min()) > 1e-4, "a free entry sits on a bound; retune"
    assert float(held_margin.min()) > 1e-4, "a held entry has a null multiplier; retune"
    assert torch.autograd.gradcheck(
        lambda h, cc: box_qp(h, cc, U_MAX), (Ht, ct), eps=1e-9, atol=1e-7
    )


def test_no_gradient_flows_to_a_saturated_control() -> None:
    """`lambda` vanishes on active coordinates -- the content of the adjoint.

    A saturated control does not respond to a perturbation of the problem, so
    its sensitivity is exactly zero, not merely small. An adjoint that forgot
    the mask would leak a finite value here.
    """
    H, c = _reduced(30, 2, batch=256)
    ct = torch.tensor(c, requires_grad=True)
    solution = solve_box_qp(torch.tensor(H), torch.tensor(c), U_MAX)
    assert bool(solution.active.any()), "no active bounds: the test proves nothing"
    box_qp(torch.tensor(H), ct, U_MAX).sum().backward()
    assert ct.grad is not None
    assert torch.all(ct.grad[solution.active] == 0.0)
    assert bool((ct.grad[~solution.active] != 0.0).any())


def _leaks_on_saturated(H: np.ndarray, c: np.ndarray, dtype: torch.dtype) -> float:
    """``max |dL/dc|`` over the coordinates that sit at a bound -- ideally zero."""
    Ht, ct = torch.tensor(H, dtype=dtype), torch.tensor(c, dtype=dtype)
    active = solve_box_qp(Ht, ct, U_MAX).active
    leaf = ct.clone().requires_grad_(True)
    box_qp(Ht, leaf, U_MAX).sum().backward()
    assert leaf.grad is not None
    return float(leaf.grad[active].abs().max())


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_the_saturated_zero_survives_a_perturbed_hessian(dtype: torch.dtype) -> None:
    """The zero must be structural, and one Hessian cannot show that it is.

    `test_no_gradient_flows_to_a_saturated_control` asks the question at a single
    ``H``, where the answer is luck: the masked row is ``e_i`` against a zero
    right-hand side, but LU with partial pivoting swaps that row away whenever
    ``|H_ji| > 1`` and recovers the entry by cancellation, so whether the
    cancellation is exact depends on the low bits of ``H``. Those bits come from
    LAPACK's DARE solve and therefore from the host's BLAS kernels -- which is
    how this passed here and failed on a CI runner. Sweeping the neighbourhood
    asks the question the code must actually answer.
    """
    H, c = _reduced(30, 2, batch=256)
    for step in range(-12, 13):
        perturbed = H.copy()
        toward = np.inf if step > 0 else -np.inf
        for _ in range(abs(step)):
            perturbed[1, 1] = np.nextafter(perturbed[1, 1], toward)
        leak = _leaks_on_saturated(perturbed, c, dtype)
        assert leak == 0.0, f"{step:+d} ULP on H[1,1] leaked {leak:.3e} ({dtype})"


def test_no_gradient_flows_to_a_saturated_control_at_the_campaign_shape() -> None:
    """The claim, asked in the precision and at the shape the campaign trains in.

    ``n = 100, m = 30`` in float32 is where `cocp_exact` actually runs, and it is
    where a leaked adjoint is largest -- measured at 7.7e-08 before the mask was
    applied on the way out, against free multipliers of 1.2e-02. At ``m = 2`` in
    float64 the same defect is ten orders of magnitude smaller, so the original
    test was posed where it was least able to see it.
    """
    H, c = _reduced(100, 30, batch=256)
    assert _leaks_on_saturated(H, c, torch.float32) == 0.0
    assert _leaks_on_saturated(H, c, torch.float64) == 0.0


def test_shared_and_per_element_hessian_gradients_agree() -> None:
    """A shared `H` accumulates over the batch; a per-element one does not."""
    H, c = _reduced(30, 2, batch=64)
    shared = torch.tensor(H, requires_grad=True)
    box_qp(shared, torch.tensor(c), U_MAX).sum().backward()
    per_element = torch.tensor(H).expand(64, 2, 2).clone().requires_grad_(True)
    box_qp(per_element, torch.tensor(c), U_MAX).sum().backward()
    assert shared.grad is not None and per_element.grad is not None
    torch.testing.assert_close(shared.grad, per_element.grad.sum(0))
