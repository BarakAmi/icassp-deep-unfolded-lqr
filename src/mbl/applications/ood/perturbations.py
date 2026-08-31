"""Dynamics-perturbation machinery for zero-shot OOD evaluation
(docs/planning/03_studies/nb06_ood_generalization/zero_shot_ood_generalization_benchmark.md Sec 2.3/3.1): a
`PerturbedLQRProblemFactory` that satisfies the existing `ProblemFactory`
Protocol (`applications.factories.ProblemFactory`) structurally, so it drops
into `Experiment.problem`/`EvaluationProtocol` exactly like
`LQRProblemFactory`/`LTVLQRProblemFactory` -- no edit to `factories.py`
(the S3 law).

Two perturbation families, both parameterized by one scalar `level` and one
`seed`:

* **Additive** -- ``A' = (A + level * G / ||G||_F)`` UNCONDITIONALLY
  renormalized to a fixed target spectral radius just under 1
  (`_TARGET_SPECTRAL_RADIUS`) for every `level > 0`, optionally the same
  direction-draw treatment on `B`. Renormalizing unconditionally --
  rather than only when the raw draw would reach/exceed 1 -- is a
  deliberate correction (NB06 plan Sec 2.3): the nominal `A` already sits at
  spectral radius essentially 1.0, so a *conditional* rescale only touches
  roughly half of all draws and systematically shrinks the perturbed plant
  relative to nominal -- making the perturbation axis measure "easier to
  control", the opposite of its intent. Renormalizing every draw to the SAME
  target holds the stability margin fixed across the whole level sweep, so
  `level` varies the perturbation's direction/eigenstructure alone.
* **Rotation** -- ``A' = R(level) A R(level)^T``, ``B' = R(level) B``, where
  `R` is built so that its canonical rotation angles all equal `level`
  degrees EXACTLY (`build_rotation_matrix`) -- the legacy notebook's Haar
  draw had no angle parameter and could not produce a genuine sweep.

Deliberate design law: for a FIXED `seed`, the underlying perturbation
direction (`G`, for additive) or generator (`S`, for rotation) is identical
across every `level` -- both are the first draw from a fresh
``np.random.default_rng(seed)`` inside `apply`, so a level sweep at one seed
traces one direction/plane at increasing magnitude/angle, and averaging over
several seeds (NB06 plan Sec 3.4) averages over *which* direction/plane was
perturbed. `level == 0` (or `kind == NONE`) short-circuits `apply` itself
before `_apply_additive`/`build_rotation_matrix` are ever reached (see
`DynamicsPerturbation.apply`), which is what makes the zero-level degeneracy
law exact regardless of how either branch renormalizes for `level > 0`.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
from scipy.linalg import expm

from ..factories import LQRProblemFactory
from ...core.constraint.box_constraint import BoxConstraint
from ...core.cost.quadratic_cost import QuadraticCost
from ...core.optimal_control_problem import OptimalControlProblem
from ...core.system.linear_system import LinearSystem
from ...models.guards import require_linear_quadratic

#: Denominator floor when normalizing a perturbation direction to unit
#: Frobenius norm (guards against a (measure-zero) all-zero draw).
_NORM_EPS = 1e-12

#: Fixed target spectral radius every ADDITIVE perturbation (level > 0) is
#: renormalized to, UNCONDITIONALLY -- not "only when it would reach/exceed
#: 1" (that conditional form is what NB06_OOD_GENERALIZATION_PLAN.md Sec 2.3
#: calls the v1 defect: since the nominal `A` already has spectral radius
#: essentially 1.0 by construction (`LQRProblemFactory.build`), a level>0
#: perturbation lands above the unit circle roughly half the time, and
#: conditionally rescaling ONLY those draws systematically shrinks the
#: perturbed plant -- making it MORE contractive, i.e. easier to control,
#: than nominal. Renormalizing every level>0 draw to the SAME fixed target
#: holds the stability margin constant across the whole level sweep, so
#: `level` varies the perturbation's DIRECTION/eigenstructure at fixed
#: margin -- the quantity the axis is actually supposed to sweep. Set just
#: under 1 (never exactly 1, which would be a measure-zero marginal case
#: indistinguishable from instability under floating point).
_TARGET_SPECTRAL_RADIUS = 0.999


class DynamicsPerturbationKind(StrEnum):
    """Which dynamics-perturbation family `DynamicsPerturbation.apply` builds."""

    NONE = "none"
    ADDITIVE = "additive"
    ROTATION = "rotation"


def build_rotation_matrix(
    *, state_dim: int, degrees: float, rng: np.random.Generator
) -> np.ndarray:
    """A rotation matrix whose canonical angle in EVERY invariant 2-plane
    equals `degrees` exactly.

    Construction: draw a Haar-random orthogonal `Q` (QR of a Gaussian
    matrix, sign-corrected so `Q` is genuinely uniform rather than biased
    by LAPACK's arbitrary QR sign convention), then conjugate the canonical
    skew-symmetric block generator ``J`` (unit-angle-per-plane blocks
    ``[[0,-1],[1,0]]``) by `Q`: ``S = Q @ J @ Q^T``. Similarity transforms
    preserve eigenvalues, so `S`'s eigenvalues are exactly `J`'s (``+-i``
    per plane, plus a zero mode if `state_dim` is odd) -- i.e. every one of
    `S`'s canonical angles is exactly 1. ``R = expm(radians(degrees) * S)``
    then has every canonical angle equal to `degrees` exactly, since
    ``expm`` of a block-diagonal skew-symmetric matrix is the corresponding
    block-diagonal rotation, conjugated by the same `Q`.

    Args:
        state_dim: The ambient dimension `n`.
        degrees: The requested rotation angle, applied identically to every
            invariant plane; may be negative (rotation in the opposite
            direction).
        rng: The generator drawing `Q` -- callers pass a FRESH generator
            seeded from the perturbation's own `seed`, so the same seed
            always yields the same `Q`/plane structure regardless of
            `degrees` (see module docstring).

    Returns:
        The orthogonal rotation matrix `R`, shape ``(n, n)``.
    """
    gaussian = rng.standard_normal((state_dim, state_dim))
    Q, upper = np.linalg.qr(gaussian)
    sign = np.sign(np.diagonal(upper))
    sign[sign == 0] = 1.0
    Q = Q @ np.diag(sign)

    generator = np.zeros((state_dim, state_dim))
    for i in range(0, state_dim - 1, 2):
        generator[i, i + 1] = -1.0
        generator[i + 1, i] = 1.0
    S = Q @ generator @ Q.T

    return np.asarray(expm(np.deg2rad(degrees) * S))


def _apply_additive(
    A: np.ndarray,
    B: np.ndarray,
    *,
    level: float,
    perturb_B: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    direction_A = rng.standard_normal(A.shape)
    direction_A = direction_A / (np.linalg.norm(direction_A, "fro") + _NORM_EPS)
    A_tilde = A + level * direction_A
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(A_tilde))))
    # UNCONDITIONAL: every level > 0 draw is renormalized to the same fixed
    # target, not only the draws that would otherwise cross 1 (see
    # _TARGET_SPECTRAL_RADIUS's docstring for why the conditional form is
    # wrong). `apply()`'s own level == 0.0 early return -- never reaching
    # this function at all -- is what keeps the degeneracy law exact.
    A_perturbed = A_tilde * (_TARGET_SPECTRAL_RADIUS / spectral_radius)

    if not perturb_B:
        return A_perturbed, np.array(B, copy=True)
    direction_B = rng.standard_normal(B.shape)
    direction_B = direction_B / (np.linalg.norm(direction_B, "fro") + _NORM_EPS)
    return A_perturbed, B + level * direction_B


@dataclass(frozen=True)
class DynamicsPerturbation:
    """One dynamics-perturbation specification (NB06 plan Sec 2.3).

    Attributes:
        kind: Which family to apply; `NONE` (the default) is the identity.
        level: The scalar perturbation magnitude -- an epsilon for
            `ADDITIVE`, degrees for `ROTATION`. Ignored for `NONE`.
        seed: Seeds the perturbation direction/generator (see module
            docstring for the fixed-seed-across-levels law).
        perturb_B: `ADDITIVE` only -- also perturb `B` (in its own
            direction, drawn from the same advancing stream as `A`'s).
    """

    kind: DynamicsPerturbationKind = DynamicsPerturbationKind.NONE
    level: float = 0.0
    seed: int = 0
    perturb_B: bool = False

    def apply(self, A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Perturb `(A, B)` per this specification.

        Args:
            A: State transition matrix, shape ``(n, n)``.
            B: Control input matrix, shape ``(n, m)``.

        Returns:
            ``(A', B')``, the perturbed pair -- exact copies of the inputs
            (never the same object) when `kind` is `NONE` or `level == 0`.
        """
        if self.kind is DynamicsPerturbationKind.NONE or self.level == 0.0:
            return np.array(A, copy=True), np.array(B, copy=True)
        rng = np.random.default_rng(self.seed)
        if self.kind is DynamicsPerturbationKind.ADDITIVE:
            return _apply_additive(
                A, B, level=self.level, perturb_B=self.perturb_B, rng=rng
            )
        R = build_rotation_matrix(state_dim=A.shape[0], degrees=self.level, rng=rng)
        return R @ A @ R.T, R @ B

    def get_signature(self) -> dict[str, Any]:
        """`Signable` member: type + every field.

        Returns:
            ``{"type": "DynamicsPerturbation", "kind":, "level":, "seed":,
            "perturb_B":}``.
        """
        return {
            "type": type(self).__name__,
            "kind": self.kind.value,
            "level": self.level,
            "seed": self.seed,
            "perturb_B": self.perturb_B,
        }


@dataclass(frozen=True)
class PerturbedLQRProblemFactory:
    """`ProblemFactory`-compatible wrapper: `base`'s own nominal `(A, B)`
    (drawn once via `base.build()`, never re-derived here -- the seam that
    keeps `factories.py` untouched, NB06 plan Sec 3.1), perturbed per
    `perturbation`, optionally at a different `horizon`. `Q`, `R`, and any
    `constraints` are carried through from `base` UNCHANGED (the
    Dynamics-OOD law, NB06 plan Sec 1.4: the controller's cost/constraint
    structure is not what is under test -- only the plant it is rolled out
    against is perturbed).

    Attributes:
        base: The nominal problem factory.
        perturbation: The dynamics perturbation to apply; the default
            (`DynamicsPerturbationKind.NONE`) leaves `(A, B)` untouched.
        horizon_override: Optional replacement horizon (Axis H); ``None``
            keeps `base`'s own horizon. A different horizon rebuilds `Q`/`R`
            at the new length (the same time-stacked-identity convention as
            `LQRProblemFactory.build`); `(A, B)` are unaffected by a pure
            horizon change.
        u_max_override: Optional replacement infinity-norm control bound
            (Axis C, NB06 plan Sec 2.6/3.9); ``None`` keeps `base`'s own
            bound untouched. Replaces `base`'s box constraint with a fresh
            one at the new bound -- `Q`/`R`/`(A, B)` are unaffected by a
            pure bound change (the Constraint-OOD law: only the feasible
            set moves, never the plant or cost).
    """

    base: LQRProblemFactory
    perturbation: DynamicsPerturbation = field(default_factory=DynamicsPerturbation)
    horizon_override: int | None = None
    u_max_override: float | None = None

    @property
    def state_dim(self) -> int:
        """See `applications.factories.ProblemFactory`."""
        return self.base.state_dim

    @property
    def control_dim(self) -> int:
        """See `applications.factories.ProblemFactory`."""
        return self.base.control_dim

    @property
    def horizon(self) -> int:
        """See `applications.factories.ProblemFactory`; `horizon_override`
        wins when set."""
        return (
            self.horizon_override
            if self.horizon_override is not None
            else self.base.horizon
        )

    def build(self) -> OptimalControlProblem:
        """Construct the perturbed problem.

        Returns:
            The `OptimalControlProblem`: `base`'s own `(A, B)` perturbed per
            `self.perturbation`, `Q`/`R` from `base` (re-stacked at
            `self.horizon` if `horizon_override` differs from `base`'s),
            and `base`'s own `constraints` passed through unchanged unless
            `u_max_override` is set, in which case a fresh box constraint at
            the new bound replaces it.

        Raises:
            ValueError: If `u_max_override` is set but `base`'s problem
                declares no box constraint to override.
        """
        nominal = self.base.build()
        system, cost = require_linear_quadratic(nominal)
        A = system.A_t[0]
        B = system.B_t[0]
        A_perturbed, B_perturbed = self.perturbation.apply(A, B)
        perturbed_system = LinearSystem.fully_observable(A_perturbed, B_perturbed)

        horizon = self.horizon
        if horizon == self.base.horizon:
            perturbed_cost = cost
        else:
            n, m = self.state_dim, self.control_dim
            perturbed_cost = QuadraticCost(
                Q=np.repeat(np.eye(n)[None], horizon + 1, axis=0),
                R=np.repeat(np.eye(m)[None], horizon, axis=0),
            )

        constraints = nominal.constraints
        if self.u_max_override is not None:
            if not constraints:
                raise ValueError(
                    "u_max_override requires the base problem to already "
                    "declare a box constraint (nominal.constraints[0])."
                )
            constraints = [BoxConstraint(u_max=self.u_max_override)]

        return OptimalControlProblem(
            system=perturbed_system,
            cost=perturbed_cost,
            constraints=constraints,
        )

    def get_signature(self) -> dict[str, Any]:
        """`Signable` member: the base factory + perturbation + horizon/
        bound overrides.

        Returns:
            ``{"type":, "base":, "perturbation":, "horizon_override":,
            "u_max_override":}``.
        """
        return {
            "type": type(self).__name__,
            "base": self.base.get_signature(),
            "perturbation": self.perturbation.get_signature(),
            "horizon_override": self.horizon_override,
            "u_max_override": self.u_max_override,
        }
