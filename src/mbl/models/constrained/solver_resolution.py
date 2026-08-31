"""COCP solver validation, and the diagnostic that used to select for you.

**D17, as ratified 2026-08-02: the backend is declared, and nothing resolves.**
A contender says which solver its gradients come from, that declaration is
signed into `ModelID`, and this module's canary **validates** it —
`require_validated_cocp_solver` raises where the controller is built if this
machine cannot honour the declaration. It never substitutes another solver,
because a substitution is exactly how a study came to hold two models whose
gradients were produced by different solvers under identifiers that said
nothing about it (`cocp` resolved to DIFFCP while `cocp_lower_bound` resolved
to MOREAU, since the two canary at different query points).

`resolve_cocp_solver` below is **not** that path any more. It survives as a
*diagnostic* — reachable through `workbench.describe_cocp_solver_choice`, which
an author runs to find out what this machine can honour **before** writing a
declaration. No recipe may reach it, and `tests/models/constrained/
test_declared_solver.py` asserts that structurally, because a behavioural test
cannot see a call that was simply never made.

The capability probe + correctness canary underneath is unchanged: it solves the
QP forward and backward under a candidate and under this project's DIFFCP/SCS
reference on the actual problem shape (never a toy instance) and compares.

MOREAU is a genuine capability, not a strict upgrade. Empirically (see
docs/planning/03_studies/nb04_box_constrained/cocp_integration_and_convergence_visualization.md Sec 1.0.6):
MOREAU-CPU's forward solution and gradients matched DIFFCP/SCS to ~1e-7
throughout -- fully trustworthy, and ~64x faster than DIFFCP/SCS at NB04's
own batch/shape. MOREAU-CUDA's *backward* pass disagreed with DIFFCP by up
to ~50% (relative), seed-dependent, even in the fully-unconstrained regime
where no active-constraint multivaluedness excuses it -- and it was not even
faster than MOREAU-CPU at this problem's scale. The two tolerances below are
calibrated against that measurement: loose enough to accept ordinary
solver-to-solver noise (~1e-7), tight enough to reject the CUDA failure mode
this canary exists to catch. CUDA is therefore never attempted automatically
(`resolve_cocp_solver` only tries MOREAU on the `device` it is explicitly
asked for -- an intentional caller choice, not a default).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

from .box_qp import BoxQPCertificate, box_qp, certify_box_qp
from .cocp import COCPController
from .solver_spec import COCPSolverSpec

logger = logging.getLogger(__name__)

#: Combined absolute+relative tolerance (`numpy.allclose` convention) the
#: canary requires between the candidate and reference solver's primal
#: solution and gradients. See module docstring for the calibration.
_CANARY_ATOL = 1e-4
_CANARY_RTOL = 1e-2

#: Feasibility slack: a control within this margin of the box counts as
#: feasible (matches this codebase's other box-constraint tolerance checks).
_FEASIBILITY_SLACK = 1e-4

#: The trusted backend every other one is validated against: this project's
#: long-standing DIFFCP/SCS path. Validating it against itself is skipped —
#: comparing the reference to the reference is not a check, and it costs two
#: QP solves and a backward pass to learn nothing.
REFERENCE_COCP_BACKEND = "DIFFCP"

#: The **measured** default (Annex 01 §5 rule 5 requires the default to be a
#: measurement, not a convenience). MOREAU on CPU reproduces DIFFCP's forward
#: solution and its gradients to ~1e-7 and is ~64x faster at NB04's shape;
#: MOREAU on CUDA agrees on the forward solution and disagrees on the backward
#: pass by up to ~50 % relative, seed-dependent. So the default names the
#: backend, and `solver_device` stays "cpu": CUDA is declarable, never default,
#: and the validator refuses it here today.
DEFAULT_COCP_BACKEND = "MOREAU"


class SolverValidationError(RuntimeError):
    """A declared solver backend this machine cannot honour.

    Raised rather than resolved past. The alternative — substituting whatever
    does work — produces a controller whose gradients came from a solver the
    specification does not name, and files its model under an identifier that
    says otherwise.
    """


@dataclass(frozen=True)
class COCPCanaryInstance:
    """One concrete QP a solver canary is evaluated against: a problem's raw
    system/cost matrices plus a representative query point. Deliberately raw
    arrays, not an `OptimalControlProblem`/`BoxConstraint` -- the same
    "callers that only have saved values" contract `build_qp_layer` already
    serves (e.g. a freshly-initialized COCPController's own seed point).

    Attributes:
        A: State transition matrix, shape ``(n, n)``.
        B: Control input matrix, shape ``(n, m)``.
        R: Control cost matrix, shape ``(m, m)``.
        u_max: The scalar infinity-norm control bound.
        x: The query state, shape ``(n,)``.
        P_sqrt: The query cost-to-go square root, shape ``(n, n)``.
        q: The query linear cost-to-go term, shape ``(n,)``.
    """

    A: np.ndarray
    B: np.ndarray
    R: np.ndarray
    u_max: float
    x: np.ndarray
    P_sqrt: np.ndarray
    q: np.ndarray


@dataclass(frozen=True)
class CanaryResult:
    """The canary's verdict: whether `candidate` reproduces `reference`'s
    primal solution and gradients on a `COCPCanaryInstance`, within
    tolerance, while itself remaining finite and feasible.

    Attributes:
        passed: ``finite and feasible and primal_close and grad_close``.
        finite: Whether the candidate's control is entirely finite.
        feasible: Whether the candidate's control satisfies the box bound.
        primal_close: Whether the candidate's control matches the
            reference's, within `_CANARY_ATOL`/`_CANARY_RTOL`.
        grad_close: Whether the candidate's `d(sum(u))/d(P_sqrt)` and
            `d(sum(u))/d(q)` match the reference's, within the same
            tolerance.
        detail: A human-readable summary (the Tier-3 narration payload).
    """

    passed: bool
    finite: bool
    feasible: bool
    primal_close: bool
    grad_close: bool
    detail: str


@dataclass(frozen=True)
class ResolvedCOCPSolver:
    """The outcome of `resolve_cocp_solver`: the spec to actually use, and
    the diagnostic trail a Tier-3 narration helper reports (never printed
    here -- this module returns data only, per the workbench contract its
    caller honors).

    Attributes:
        spec: The `COCPSolverSpec` to build the controller's QP layer with.
        reason: A one-line human-readable explanation of the outcome.
        attempted: Solver names tried, in order (e.g. ``("MOREAU",
            "DIFFCP")`` on a canary failure, or just ``("DIFFCP",)`` when
            MOREAU was never available to try).
        canary: The canary's verdict, or ``None`` when MOREAU was never
            attempted (import/device unavailable, so there was nothing to
            canary-test).
    """

    spec: COCPSolverSpec
    reason: str
    attempted: tuple[str, ...]
    canary: CanaryResult | None


def _probe_moreau(device: str) -> str | None:
    """Whether MOREAU can be imported and `device` is available to it.

    Args:
        device: The torch device string to check MOREAU's own availability
            for (``"cpu"`` is always available once MOREAU imports at all).

    Returns:
        ``None`` if MOREAU is usable on `device`; otherwise a human-readable
        reason it is not.
    """
    try:
        import moreau
    except ImportError as error:
        return f"import failed: {error}"
    device_kind = torch.device(device).type
    if device_kind == "cpu":
        return None
    if not moreau.device_available(device_kind):
        detail = moreau.device_error(device_kind)
        suffix = f" ({detail})" if detail else ""
        return f"device {device_kind!r} unavailable{suffix}"
    return None


def _solve_and_grad(
    instance: COCPCanaryInstance, spec: COCPSolverSpec
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve `instance`'s QP under `spec` and back-propagate a deterministic
    scalarization (``u.sum()``), returning the primal solution and both
    input gradients as plain NumPy arrays.

    A fixed scalarization (rather than a random direction) keeps the canary
    reproducible without needing a managed RNG seed; it is not a weaker
    check -- it was verified to still catch the ~50%-relative-error MOREAU-
    CUDA failure mode this canary exists to guard against.

    Returns:
        ``(u, d(sum(u))/d(P_sqrt), d(sum(u))/d(q))``.
    """
    n, m = instance.A.shape[0], instance.B.shape[1]
    layer = COCPController.build_qp_layer(
        instance.A, instance.B, instance.R, instance.u_max, spec
    )
    device = torch.device(spec.device)
    x = torch.tensor(
        np.asarray(instance.x, dtype=np.float64).reshape(1, n, 1), device=device
    )
    P_sqrt = torch.tensor(
        np.asarray(instance.P_sqrt, dtype=np.float64).reshape(1, n, n),
        device=device,
        requires_grad=True,
    )
    q = torch.tensor(
        np.asarray(instance.q, dtype=np.float64).reshape(1, n),
        device=device,
        requires_grad=True,
    )

    (u,) = layer(x, P_sqrt, q, solver_args=spec.to_solver_args())
    u.sum().backward()
    assert P_sqrt.grad is not None
    assert q.grad is not None
    return (
        u.detach().cpu().numpy().reshape(m),
        P_sqrt.grad.detach().cpu().numpy().reshape(n, n),
        q.grad.detach().cpu().numpy().reshape(n),
    )


def run_solver_canary(
    instance: COCPCanaryInstance,
    candidate: COCPSolverSpec,
    reference: COCPSolverSpec,
) -> CanaryResult:
    """Solve `instance` under both `candidate` and `reference`, forward and
    backward, and compare -- the correctness half of the preflight (the
    other half, availability, is `_probe_moreau`).

    Args:
        instance: The problem/query point both solvers are evaluated on.
        candidate: The solver spec under test (typically MOREAU).
        reference: The trusted solver spec to compare against (this
            project's DIFFCP/SCS default).

    Returns:
        The `CanaryResult`.
    """
    u_ref, grad_p_ref, grad_q_ref = _solve_and_grad(instance, reference)
    u_cand, grad_p_cand, grad_q_cand = _solve_and_grad(instance, candidate)

    finite = bool(np.isfinite(u_cand).all())
    feasible = bool(np.all(np.abs(u_cand) <= instance.u_max + _FEASIBILITY_SLACK))
    primal_close = bool(
        np.allclose(u_cand, u_ref, atol=_CANARY_ATOL, rtol=_CANARY_RTOL)
    )
    grad_close = bool(
        np.allclose(grad_p_cand, grad_p_ref, atol=_CANARY_ATOL, rtol=_CANARY_RTOL)
        and np.allclose(grad_q_cand, grad_q_ref, atol=_CANARY_ATOL, rtol=_CANARY_RTOL)
    )
    passed = finite and feasible and primal_close and grad_close
    detail = (
        f"finite={finite} feasible={feasible} primal_close={primal_close} "
        f"grad_close={grad_close} "
        f"||du||={np.linalg.norm(u_cand - u_ref):.2e} "
        f"||d(dP_sqrt)||={np.linalg.norm(grad_p_cand - grad_p_ref):.2e} "
        f"||d(dq)||={np.linalg.norm(grad_q_cand - grad_q_ref):.2e}"
    )
    return CanaryResult(
        passed=passed,
        finite=finite,
        feasible=feasible,
        primal_close=primal_close,
        grad_close=grad_close,
        detail=detail,
    )


def require_validated_cocp_solver(
    instance: COCPCanaryInstance, spec: COCPSolverSpec
) -> SolverVerdict:
    """Refuse a declared solver this machine cannot honour (D17).

    **The parameters are the specification of the rule.** There is one spec in
    and a verdict out — no candidate list, no fallback, and no `COCPSolverSpec`
    in the return type, so this function has nothing through which it could hand
    back a different solver. A function that could is a resolver whatever it is
    named, and resolving is what D17 forbids.

    **Validation is against the problem, not against another implementation**
    (Annex 01 §5 rule 5, corrected 2026-08-18). The reference is graded on the
    same footing as everything else and holds no privilege: it used to be exempt,
    and being exempt is what let it refuse correct models for years on the
    strength of its own error. Its distance from the certified point is still
    recorded — as a diagnostic that documents provenance and cannot refuse a run.

    Args:
        instance: The problem and query point to validate on — the caller's own
            problem shape, never a toy: a solver's disagreement is shape- and
            seed-dependent, which is how one study's two COCP contenders came to
            canary at different points and disagree.
        spec: The **declared** backend, device and tolerances.

    Returns:
        The `SolverVerdict`, always -- there is no backend this returns `None`
        for, because there is no backend that goes ungraded.

    Raises:
        SolverValidationError: If the declared backend cannot be imported, is
            unavailable on the declared device, fails while solving, or produces
            a solution or gradient outside tolerance on `instance`. The message
            names the declaration, because that is what the author has to change.
    """
    if spec.name != REFERENCE_COCP_BACKEND:
        unavailable = _probe_moreau(spec.device)
        if unavailable is not None:
            raise SolverValidationError(
                f"this study declares solver_backend={spec.name!r} on "
                f"solver_device={spec.device!r}, which this machine cannot provide "
                f"({unavailable}). Declaring {REFERENCE_COCP_BACKEND!r} would run "
                "here; it is a different model and gets a different identifier, "
                "which is the point"
            )

    try:
        verdict = certify_declared_solver(instance, spec)
    except SolverValidationError:
        raise
    except Exception as error:  # noqa: BLE001 -- a backend that dies is refused
        raise SolverValidationError(
            f"this study declares solver_backend={spec.name!r} on "
            f"solver_device={spec.device!r}, and it failed while solving this "
            f"problem ({type(error).__name__}: {error}). A backend that cannot "
            "produce an answer here cannot be honoured here"
        ) from error

    if verdict.accepted:
        logger.info(
            "Solver %s on %s certified against the problem: %s",
            spec.name,
            spec.device,
            verdict.detail,
        )
        return verdict
    raise SolverValidationError(
        f"this study declares solver_backend={spec.name!r} on "
        f"solver_device={spec.device!r}, and its answer does not satisfy this "
        f"problem's own optimality conditions: {verdict.detail}. The run is "
        "refused rather than quietly switched, because a model whose gradients "
        "came from a solver its specification does not name is filed under an "
        "identifier that says otherwise"
    )


def resolve_cocp_solver(
    instance: COCPCanaryInstance,
    *,
    device: str = "cpu",
    eps: float = 1e-8,
    max_iters: int = 10000,
) -> ResolvedCOCPSolver:
    """**A diagnostic, not a dispatch (D17).** Which solver this machine *would*
    accept for `instance` — what an author consults before writing a
    declaration, through `workbench.describe_cocp_solver_choice`. No recipe may
    call it; `require_validated_cocp_solver` is the pipeline's path.

    Pick and validate the COCP solver for one run: MOREAU on `device` if
    it is importable, available, and passes `run_solver_canary` against this
    project's DIFFCP/SCS default on `instance`; that DIFFCP default
    otherwise. Resolved once per run -- the caller (a `ModelRecipe.
    build_controller`) bakes the result into `COCPConfig` for the
    controller's whole lifetime, never re-resolved per QP solve.

    Args:
        instance: The problem/query point to canary-test MOREAU against.
        device: The torch device the resolved solver should run on --
            typically the owning `ComputeContext.device`. MOREAU-on-CUDA is
            attempted only when this is explicitly a CUDA device (an
            intentional caller choice, never an automatic default): measured
            CPU-vs-CUDA behavior differs (see module docstring).
        eps: Convergence tolerance forwarded to both the candidate and the
            reference spec.
        max_iters: Iteration budget forwarded to both specs.

    Returns:
        The `ResolvedCOCPSolver`.
    """
    reference = COCPSolverSpec(
        name="DIFFCP", device="cpu", eps=eps, max_iters=max_iters
    )
    unavailable = _probe_moreau(device)
    if unavailable is not None:
        return ResolvedCOCPSolver(
            spec=reference,
            reason=f"MOREAU unavailable ({unavailable}); using DIFFCP.",
            attempted=("DIFFCP",),
            canary=None,
        )

    candidate = COCPSolverSpec(
        name="MOREAU", device=device, eps=eps, max_iters=max_iters
    )
    canary = run_solver_canary(instance, candidate, reference)
    if canary.passed:
        return ResolvedCOCPSolver(
            spec=candidate,
            reason="MOREAU available and canary passed.",
            attempted=("MOREAU",),
            canary=canary,
        )
    return ResolvedCOCPSolver(
        spec=reference,
        reason=f"MOREAU canary failed ({canary.detail}); using DIFFCP.",
        attempted=("MOREAU", "DIFFCP"),
        canary=canary,
    )


# --- certificate-based validation (Annex 01 §5 rule 5, corrected 2026-08-18) --

#: How far a declared backend's SOLUTION may sit from optimality and still be
#: honoured. **Not machine epsilon**: this asks whether an answer is good enough
#: to build a model on, never whether it came from an exact method. Judged at
#: epsilon the reference itself is refused at n = 4, where it lands 2.6e-07
#: outside the box and is otherwise sound.
#:
#: Calibrated by measurement across n = 4 … 100 and both backends, as the rule
#: demands of every default. The gap it sits in is wide: the worst residual
#: among answers that should be honoured is **1.3e-05** (MOREAU-CPU at
#: n = 100, m = 30) and the best among those that should not is **4.1e-03**
#: (DIFFCP at n = 20, which returns an infeasible control). This threshold is
#: 8x above the first and 41x below the second.
PRIMAL_TOLERANCE = 1e-4

#: How far a declared backend's GRADIENT may sit from the closed-form
#: derivative at the certified active set. Separate from `PRIMAL_TOLERANCE`
#: because it measures a different quantity with its own scale, and forcing one
#: number on both would decide a real case by 20 %.
#:
#: Calibrated the same way. The worst deviation among answers that should be
#: honoured is **1.2e-04** (DIFFCP at n = 15, degrading but still sound) and
#: the best among those that should not is **1.9e-02** (MOREAU-CUDA at n = 4 --
#: the failure this gate exists for, whose SOLUTION is exact). This threshold
#: sits 8x above the first and 19x below the second.
GRADIENT_TOLERANCE = 1e-3


@dataclass(frozen=True)
class SolverVerdict:
    """What a declared backend's answer is worth on one concrete problem.

    Attributes:
        accepted: Whether both arms passed.
        certificate: The KKT residuals of the declared backend's *solution*.
        gradient_deviation: How far its *gradient* sits from the closed-form
            derivative at the certified active set, relative to that
            derivative's own scale.
        reference_deviation: How far the reference backend's solution sits from
            the same certified point. **A diagnostic, never a criterion** -- it
            records provenance, including where the established solvers stop
            agreeing, and nothing here may refuse a run on its account.
        detail: A human-readable summary, for the message and the record.
    """

    accepted: bool
    certificate: BoxQPCertificate
    gradient_deviation: float
    reference_deviation: float | None
    detail: str


def reduce_to_box_qp(instance: COCPCanaryInstance) -> tuple[np.ndarray, np.ndarray]:
    """The eliminated program ``(H, c)`` at this instance's query point.

    Args:
        instance: The problem and query point.

    Returns:
        ``(H, c)`` with shapes ``(m, m)`` and ``(m,)``.
    """
    P = instance.P_sqrt.T @ instance.P_sqrt
    H = 2.0 * (instance.R + instance.B.T @ P @ instance.B)
    c = 2.0 * (instance.B.T @ P @ instance.A @ instance.x) + instance.B.T @ instance.q
    return H, c


def exact_solution_and_gradient(
    instance: COCPCanaryInstance,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The certified solution and **the** derivative there, in closed form.

    Differentiates the same scalarisation `_solve_and_grad` uses, through the
    same chain the QP layer exposes -- ``P_sqrt`` and ``q`` -- so the two are
    directly comparable. This is not a second opinion: given the certified
    active set the solution map is affine, so its derivative is not estimated.

    Args:
        instance: The problem and query point.

    Returns:
        ``(u, d sum(u)/d P_sqrt, d sum(u)/d q)``.
    """
    P_sqrt = torch.tensor(instance.P_sqrt, dtype=torch.float64, requires_grad=True)
    q = torch.tensor(instance.q, dtype=torch.float64, requires_grad=True)
    A = torch.tensor(instance.A, dtype=torch.float64)
    B = torch.tensor(instance.B, dtype=torch.float64)
    R = torch.tensor(instance.R, dtype=torch.float64)
    x = torch.tensor(instance.x, dtype=torch.float64)

    P = P_sqrt.T @ P_sqrt
    H = 2.0 * (R + B.T @ P @ B)
    c = (2.0 * (B.T @ P @ A @ x) + B.T @ q).unsqueeze(0)
    u = box_qp(H, c, instance.u_max)
    u.sum().backward()
    assert P_sqrt.grad is not None and q.grad is not None
    return (
        u.detach().numpy().reshape(-1),
        P_sqrt.grad.numpy(),
        q.grad.numpy(),
    )


def certify_declared_solver(
    instance: COCPCanaryInstance, spec: COCPSolverSpec
) -> SolverVerdict:
    """Grade a declared backend on this instance, against the problem itself.

    Two arms, because the KKT conditions grade a *solution* and the failure
    this gate exists for is a *gradient*: MOREAU-on-CUDA's forward answer is
    accurate while its backward pass is ~183 % wrong, so a primal-only gate
    would honour the one declaration the old comparison correctly refused.

    Args:
        instance: The problem and query point -- the caller's own shape, never
            a toy, because a solver's disagreement is shape-dependent.
        spec: The declared backend, device and tolerances.

    Returns:
        The `SolverVerdict`.
    """
    u_declared, grad_P, grad_q = _solve_and_grad(instance, spec)
    H, c = reduce_to_box_qp(instance)
    certificate = certify_box_qp(
        torch.tensor(H, dtype=torch.float64),
        torch.tensor(c, dtype=torch.float64).unsqueeze(0),
        torch.tensor(u_declared, dtype=torch.float64).unsqueeze(0),
        instance.u_max,
        bound_tol=PRIMAL_TOLERANCE,
    )
    u_exact, exact_P, exact_q = exact_solution_and_gradient(instance)
    # RELATIVE WHERE THE GRADIENT IS LARGE, ABSOLUTE WHERE IT IS NOT. Dividing
    # by the true gradient's magnitude is meaningless exactly where this problem
    # class spends most of its time: when every control saturates the derivative
    # is numerically zero, and any difference at all reads as an infinite
    # relative error. Measured, a pure ratio put MOREAU-on-CPU at 1.3e+03 on a
    # plant where its gradient is right to 1e-09.
    scale = max(float(np.abs(exact_P).max()), float(np.abs(exact_q).max()), 1.0)
    gradient_deviation = (
        max(
            float(np.abs(grad_P - exact_P).max()),
            float(np.abs(grad_q - exact_q).max()),
        )
        / scale
    )

    reference_deviation: float | None = None
    if spec.name != REFERENCE_COCP_BACKEND:
        reference = COCPSolverSpec(
            name=REFERENCE_COCP_BACKEND,
            device="cpu",
            eps=spec.eps,
            max_iters=spec.max_iters,
        )
        u_reference, _, _ = _solve_and_grad(instance, reference)
        reference_deviation = float(np.linalg.norm(u_reference - u_exact))

    accepted = (
        bool(np.isfinite(u_declared).all())
        and certificate.satisfies(PRIMAL_TOLERANCE)
        and gradient_deviation <= GRADIENT_TOLERANCE
    )
    detail = (
        f"stationarity={certificate.stationarity:.2e} "
        f"sign={certificate.sign_violation:.2e} "
        f"box={certificate.box_violation:.2e} "
        f"gradient={gradient_deviation:.2e} "
        f"tolerances=({PRIMAL_TOLERANCE:.0e}, {GRADIENT_TOLERANCE:.0e})"
    )
    if reference_deviation is not None:
        detail += f" [diagnostic: {REFERENCE_COCP_BACKEND} sits {reference_deviation:.2e} from the certified point]"
    return SolverVerdict(
        accepted=accepted,
        certificate=certificate,
        gradient_deviation=gradient_deviation,
        reference_deviation=reference_deviation,
        detail=detail,
    )
