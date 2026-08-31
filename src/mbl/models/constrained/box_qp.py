"""The box-constrained QP COCP actually poses, solved as what it is.

Eliminating the next state from the convex policy's per-step program leaves

    min_u  0.5 u^T H u + c^T u    s.t.  |u_i| <= u_max

with ``H = 2(R + B^T P B)`` **shared by every element of the batch and every step
of the horizon**, and only ``c = 2 B^T P A x + B^T q`` varying across the batch.
``H`` is positive definite whenever ``R`` is, so the minimiser is unique and the
solution map is piecewise affine -- which is what makes both an exact solve and
an exact derivative available without a cone solver.

**Why a mask, and not a submatrix.** Different batch elements have different
active sets, which is what normally defeats vectorisation. Replacing the *rows*
of the shared ``H`` at active indices with ``e_i`` (and the corresponding
right-hand side with that bound) turns the per-element active-set subproblem back
into one batched dense solve of size ``m``. Permuting the active indices first,
the masked matrix is ``[[I, 0], [H_FA, H_FF]]``, whose determinant is
``det(H_FF) != 0`` because ``H`` is positive definite -- so the system is never
singular. The plausible alternative, zeroing active rows *and* columns, is.

**Nothing here trusts an implementation.** `certify_box_qp` evaluates the
problem's own optimality conditions on a candidate point, so a solution can be
graded without reference to any other solver -- including this one. That is the
gate; agreement with an established solver is a diagnostic beside it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor

#: Multiple of the working precision's epsilon used for every tolerance in this
#: module. Literals are forbidden here: a ``1e-13`` convergence test is
#: unreachable in float32 and silently exhausts the iteration budget instead of
#: converging, which is a failure that reports itself as success.
_EPS_SCALE = 100.0

#: Iterations after which the active-set loop gives up. The loop terminates when
#: the active set stops moving, which for a strictly convex box-QP happens
#: finitely; this bounds the pathological case rather than the expected one.
DEFAULT_MAX_ITER = 50


def box_qp_tolerances(dtype: torch.dtype, u_max: float) -> tuple[float, float]:
    """Bound-membership and step tolerances for `dtype`, derived from its epsilon.

    Args:
        dtype: The working precision the solve runs in.
        u_max: The scalar infinity-norm bound, which sets the problem's scale.

    Returns:
        ``(bound_tol, step_tol)`` -- how close to a bound counts as *at* it, and
        how small an iterate change counts as settled.
    """
    eps = float(torch.finfo(dtype).eps)
    scale = _EPS_SCALE * eps * max(u_max, 1.0)
    return scale, scale


@dataclass(frozen=True)
class BoxQPCertificate:
    """The KKT residuals of a candidate point, and hence its optimality.

    Independent of how the point was produced, so it grades this module's own
    solver and any other on the same footing. For a strictly convex box-QP these
    three residuals being zero *is* optimality -- there is no further appeal to
    an implementation.

    Attributes:
        stationarity: ``max |(Hu + c)_i|`` over indices strictly inside the box.
        sign_violation: How far a multiplier points the wrong way on an active
            bound -- ``max(0, g_i)`` at an upper bound, ``max(0, -g_i)`` at a
            lower one.
        box_violation: ``max(0, ||u||_inf - u_max)``.
        active_count: How many ``(element, index)`` pairs sit at a bound; a
            diagnostic, not a residual.
    """

    stationarity: float
    sign_violation: float
    box_violation: float
    active_count: int

    @property
    def worst(self) -> float:
        """The largest of the three residuals.

        Returns:
            ``max(stationarity, sign_violation, box_violation)``.
        """
        return max(self.stationarity, self.sign_violation, self.box_violation)

    def satisfies(self, tol: float) -> bool:
        """Whether every residual is within `tol`.

        Args:
            tol: The threshold, normally from `box_qp_tolerances`.

        Returns:
            ``True`` if `worst` does not exceed `tol`.
        """
        return self.worst <= tol


@dataclass(frozen=True)
class BoxQPSolution:
    """A solved batch, with the evidence that it is solved.

    Attributes:
        u: The minimiser, shape ``(batch, m)``.
        active: Which entries sit at a bound, shape ``(batch, m)``, dtype bool.
        iterations: Active-set iterations actually run.
        converged: Whether the active set settled before `DEFAULT_MAX_ITER`.
        certificate: The KKT residuals of `u`.
    """

    u: Tensor
    active: Tensor
    iterations: int
    converged: bool
    certificate: BoxQPCertificate


def _batched(H: Tensor, batch: int, m: int) -> Tensor:
    """Broadcast a shared ``(m, m)`` Hessian to ``(batch, m, m)`` without copying.

    Args:
        H: Either ``(m, m)`` (shared) or already ``(batch, m, m)``.
        batch: The batch size to expand to.
        m: The control dimension.

    Returns:
        A ``(batch, m, m)`` view or the input unchanged.

    Raises:
        ValueError: If `H` is neither 2- nor 3-dimensional, or disagrees with `m`.
    """
    if H.dim() == 2:
        if H.shape != (m, m):
            raise ValueError(f"H must be ({m}, {m}) when shared, got {tuple(H.shape)}.")
        return H.expand(batch, m, m)
    if H.dim() == 3:
        if H.shape != (batch, m, m):
            raise ValueError(
                f"H must be ({batch}, {m}, {m}) when per-element, got {tuple(H.shape)}."
            )
        return H
    raise ValueError(f"H must be 2- or 3-dimensional, got {H.dim()} dimensions.")


def _mask_rows(H: Tensor, active: Tensor) -> Tensor:
    """Replace the rows of `H` at active indices with the identity's.

    See the module docstring for why this is the row mask and not a symmetric
    one: the result stays non-singular precisely because the free block survives
    intact.

    Args:
        H: The batched Hessian, shape ``(batch, m, m)``.
        active: Which entries are at a bound, shape ``(batch, m)``.

    Returns:
        The masked system matrix, shape ``(batch, m, m)``.
    """
    batch, m = active.shape
    eye = torch.eye(m, dtype=H.dtype, device=H.device).expand(batch, m, m)
    return torch.where(active.unsqueeze(-1), eye, H)


def residual_tolerance(H: Tensor, c: Tensor, u_max: float) -> float:
    """The scale a KKT residual should be judged against on *this* problem.

    Stationarity is a gradient, so it inherits the scale of ``Hu`` and ``c``.
    Comparing it to a bare epsilon would silently be a different assertion at
    every plant -- strict at n = 4 and vacuous at n = 100.

    Args:
        H: Hessian, ``(m, m)`` or ``(batch, m, m)``.
        c: Linear term, shape ``(batch, m)``.
        u_max: The scalar infinity-norm bound.

    Returns:
        ``100 * eps * (max_row_sum|H| * u_max + max|c|)``, floored at
        ``100 * eps``.
    """
    eps = float(torch.finfo(c.dtype).eps)
    row_sum = float(H.abs().sum(dim=-1).max())
    scale = row_sum * u_max + float(c.abs().max())
    return _EPS_SCALE * eps * max(scale, 1.0)


def _bound_masks(u: Tensor, u_max: float, bound_tol: float) -> tuple[Tensor, Tensor]:
    """Which entries sit at the upper and lower bound, by position.

    Membership is decided by *where the point is*, never by which iteration
    produced it. A saturated control has zero local sensitivity whatever the
    solver believed on the way there, so this is also the mask the adjoint needs.

    Args:
        u: The point, shape ``(batch, m)``.
        u_max: The scalar infinity-norm bound.
        bound_tol: How close to a bound counts as at it.

    Returns:
        ``(at_upper, at_lower)``, both ``(batch, m)`` and boolean.
    """
    return u >= u_max - bound_tol, u <= -u_max + bound_tol


def certify_box_qp(
    H: Tensor,
    c: Tensor,
    u: Tensor,
    u_max: float,
    *,
    bound_tol: float | None = None,
) -> BoxQPCertificate:
    """Grade a candidate point against the problem's own optimality conditions.

    Args:
        H: Hessian, ``(m, m)`` shared or ``(batch, m, m)``; assumed symmetric
            positive definite.
        c: Linear term, shape ``(batch, m)``.
        u: The candidate minimiser, shape ``(batch, m)``.
        u_max: The scalar infinity-norm bound.
        bound_tol: How close to a bound counts as *at* it. Defaults to the
            epsilon-derived value, which is right for a point this module
            produced -- it lands on the bound exactly, by clamping. **A grader
            of somebody else's answer must widen it**: a solver that stops
            2e-07 outside the box leaves entries that are morally saturated
            classified as free, and their gradients then read as enormous
            stationarity residuals rather than as the small feasibility
            violation they are.

    Returns:
        The `BoxQPCertificate` for `u`.
    """
    batch, m = c.shape
    Hb = _batched(H, batch, m)
    if bound_tol is None:
        bound_tol, _ = box_qp_tolerances(u.dtype, u_max)

    gradient = torch.einsum("bij,bj->bi", Hb, u) + c
    at_upper, at_lower = _bound_masks(u, u_max, bound_tol)
    free = ~(at_upper | at_lower)

    zero = torch.zeros((), dtype=u.dtype, device=u.device)
    stationarity = torch.where(free, gradient, zero).abs().max()
    upper_violation = torch.where(at_upper, gradient, zero).clamp_min(0.0).max()
    lower_violation = torch.where(at_lower, -gradient, zero).clamp_min(0.0).max()
    box_violation = (u.abs().max() - u_max).clamp_min(0.0)

    # Detached on the way out: a certificate is a diagnostic about a point,
    # never a differentiable quantity, and converting a grad-tracking tensor to
    # a Python float is a warning this project treats as an error.
    return BoxQPCertificate(
        stationarity=float(stationarity.detach()),
        sign_violation=float(torch.maximum(upper_violation, lower_violation).detach()),
        box_violation=float(box_violation.detach()),
        active_count=int((at_upper | at_lower).sum()),
    )


def solve_box_qp(
    H: Tensor,
    c: Tensor,
    u_max: float,
    *,
    max_iter: int = DEFAULT_MAX_ITER,
) -> BoxQPSolution:
    """Minimise ``0.5 u'Hu + c'u`` over the box, for a whole batch at once.

    Projected Newton on the active set: identify which bounds are held by the
    gradient, solve the resulting equality-constrained subproblem as one batched
    dense system (see the module docstring for the masking argument), and repeat
    until the active set stops moving. Finite for a strictly convex box-QP.

    Args:
        H: Hessian, ``(m, m)`` shared across the batch or ``(batch, m, m)``;
            must be symmetric positive definite.
        c: Linear term, shape ``(batch, m)``.
        u_max: The scalar infinity-norm bound on every control entry.
        max_iter: Iteration cap; reaching it sets ``converged=False`` rather than
            raising, so a caller can inspect the certificate and decide.

    Returns:
        The `BoxQPSolution`, whose certificate is computed on the returned point.

    Raises:
        ValueError: If `u_max` is not positive, or `H`/`c` disagree on shape.
    """
    if u_max <= 0:
        raise ValueError(f"u_max must be positive, got {u_max}.")
    if c.dim() != 2:
        raise ValueError(f"c must be (batch, m), got {tuple(c.shape)}.")

    batch, m = c.shape
    Hb = _batched(H, batch, m)
    bound_tol, step_tol = box_qp_tolerances(c.dtype, u_max)

    # Warm start at the unconstrained minimiser, clamped into the box: the
    # correct answer wherever no bound binds, and a good guess where one does.
    u = torch.linalg.solve(Hb, -c.unsqueeze(-1)).squeeze(-1).clamp(-u_max, u_max)
    held = torch.zeros_like(u, dtype=torch.bool)  # the loop's own bookkeeping
    tol = residual_tolerance(Hb, c, u_max)

    # STOP ON THE CERTIFICATE, NOT ON THE BITMASK. A settled active set is
    # sufficient for optimality but not necessary, and it is not always
    # reachable: where an entry sits at a bound with a numerically zero
    # gradient, membership flips between iterations forever while the point
    # itself is already optimal. Measured at n = 100, m = 30 in float32 across
    # 8000 states -- the residuals were at 1e-3 and falling while the mask
    # oscillated. The residuals are what optimality means, so they decide.
    best_u = u
    best = certify_box_qp(Hb, c, u, u_max)
    converged = best.satisfies(tol)
    iterations = 1

    upper = c.new_full((), u_max)
    lower = c.new_full((), -u_max)
    while not converged and iterations < max_iter:
        iterations += 1
        gradient = torch.einsum("bij,bj->bi", Hb, u) + c
        # A bound is held only when the iterate is at it AND the gradient pushes
        # further out; a bound the gradient pulls away from is about to be left.
        at_upper = (u >= u_max - bound_tol) & (gradient < 0.0)
        at_lower = (u <= -u_max + bound_tol) & (gradient > 0.0)
        candidate = at_upper | at_lower

        rhs = torch.where(candidate, torch.where(at_upper, upper, lower), -c)
        step = torch.linalg.solve(_mask_rows(Hb, candidate), rhs.unsqueeze(-1))
        u_next = step.squeeze(-1).clamp(-u_max, u_max)
        settled = bool(torch.equal(candidate, held)) and bool(
            (u_next - u).abs().max() <= step_tol
        )
        u, held = u_next, candidate

        certificate = certify_box_qp(Hb, c, u, u_max)
        # Keep the best point seen, so an oscillating tail cannot return a worse
        # answer than one already in hand.
        if certificate.worst < best.worst:
            best, best_u = certificate, u
        if best.satisfies(tol) or settled:
            converged = best.satisfies(tol)
            break

    # DERIVED FROM THE RETURNED POINT, never from the loop's last candidate.
    # When the warm start is already optimal -- every control saturated, which
    # is the common case at a tight bound -- the loop never revises its mask,
    # and returning that stale all-free mask told the adjoint that saturated
    # controls were sensitive. Found by differentiating against an independent
    # solver, which is the only check that could have seen it.
    upper_mask, lower_mask = _bound_masks(best_u, u_max, bound_tol)
    return BoxQPSolution(
        u=best_u,
        active=upper_mask | lower_mask,
        iterations=iterations,
        converged=converged,
        certificate=best,
    )


class _BoxQPFunction(torch.autograd.Function):
    """The differentiable seam: an exact adjoint instead of a differentiated solver.

    With the active set identified the solution map is locally affine, so its
    derivative is closed form and reuses the *same* masked system the forward
    pass built. Only this gets a hand-written rule -- the chain that builds
    ``(H, c)`` from a controller's parameters stays ordinary autograd, because
    the part most likely to hide a sign error is the part nobody writes down.
    """

    @staticmethod
    def forward(ctx: object, H: Tensor, c: Tensor, u_max: float) -> Tensor:
        """Solve, and remember what the adjoint needs.

        Args:
            ctx: Autograd context.
            H: Hessian, ``(m, m)`` shared or ``(batch, m, m)``.
            c: Linear term, shape ``(batch, m)``.
            u_max: The scalar infinity-norm bound.

        Returns:
            The minimiser, shape ``(batch, m)``.
        """
        solution = solve_box_qp(H, c, u_max)
        ctx.save_for_backward(H, solution.u, solution.active)  # type: ignore[attr-defined]
        ctx.shared_hessian = H.dim() == 2  # type: ignore[attr-defined]
        return solution.u

    @staticmethod
    def backward(
        ctx: object, grad_u: Tensor
    ) -> tuple[Tensor | None, Tensor | None, None]:
        """``dL/dH = -lambda u^T`` and ``dL/dc = -lambda``, with ``H_FF lambda_F = ubar_F``.

        ``lambda`` vanishes on active coordinates, which is the whole content of
        the rule: a saturated control does not respond to a perturbation of the
        problem, so no gradient may flow to it.

        Args:
            ctx: Autograd context.
            grad_u: Upstream gradient, shape ``(batch, m)``.

        Returns:
            ``(dL/dH, dL/dc, None)`` -- `u_max` is not a differentiable input.
        """
        H, u, active = ctx.saved_tensors  # type: ignore[attr-defined]
        batch, m = u.shape
        Hb = _batched(H, batch, m)
        rhs = torch.where(active, torch.zeros_like(grad_u), grad_u)
        multiplier = torch.linalg.solve(
            _mask_rows(Hb, active), rhs.unsqueeze(-1)
        ).squeeze(-1)

        grad_H: Tensor | None = None
        grad_c: Tensor | None = None
        needs = ctx.needs_input_grad  # type: ignore[attr-defined]
        if needs[0]:
            grad_H = -multiplier.unsqueeze(-1) * u.unsqueeze(-2)
            if ctx.shared_hessian:  # type: ignore[attr-defined]
                grad_H = grad_H.sum(0)
        if needs[1]:
            grad_c = -multiplier
        return grad_H, grad_c, None


def box_qp(H: Tensor, c: Tensor, u_max: float) -> Tensor:
    """Solve the box-QP differentiably in ``H`` and ``c``.

    Args:
        H: Hessian, ``(m, m)`` shared across the batch or ``(batch, m, m)``;
            must be symmetric positive definite.
        c: Linear term, shape ``(batch, m)``.
        u_max: The scalar infinity-norm bound.

    Returns:
        The minimiser, shape ``(batch, m)``, differentiable w.r.t. `H` and `c`.
    """
    return cast(Tensor, _BoxQPFunction.apply(H, c, u_max))
