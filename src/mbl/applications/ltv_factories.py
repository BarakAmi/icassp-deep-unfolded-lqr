"""The LTV problem factory (NB05_LTV_BOX_CONSTRAINED_PLAN.md Sec 6.1): the
box-constrained benchmark problem restated over a time-varying linear system
in one of three structured regimes -- periodic, block-constant, or fully
time-varying -- unified by one distinct-slice count ``D`` (Sec 0.2) that
``TimeSeriesMatrix`` already stores natively as its period/run-length/
degenerate compression modes.

`LTVLQRProblemFactory` is `LQRProblemFactory`'s sibling, never its
replacement (`factories.py` stays byte-identical, the S3 law): the same
seeded marginally-stable draw, generalized to a perturbation bank around the
identical nominal system, so that ``variation_strength == 0.0`` reproduces
`LQRProblemFactory`'s own problem bit-for-bit (Sec 2.3's degeneracy law) --
the strongest possible regression anchor for NB05 against NB04.
"""

import dataclasses
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from .factories import LQRProblemFactory
from ..core.constraint.box_constraint import BoxConstraint
from ..core.cost.quadratic_cost import QuadraticCost
from ..core.optimal_control_problem import OptimalControlProblem
from ..core.system.linear_system import LinearSystem
from ..core.utils import ensure_positive_integer

#: Rescale thresholds for the horizon-product growth-rate accumulation
#: (Sec 2.2 Rule 4): re-normalize the running product whenever its Frobenius
#: norm leaves this band, so no intermediate product can overflow/underflow
#: float64 even over the largest horizons/variation strengths this study
#: contemplates -- checked every step (cheap at this project's problem
#: scale), never merely "periodically".
_RESCALE_HIGH = 1e10
_RESCALE_LOW = 1e-10


class LTVRegime(StrEnum):
    """Which of the three structured time-variation regimes to draw
    (NB05 plan Sec 1.1)."""

    PERIODIC = "periodic"
    BLOCK_CONSTANT = "block_constant"
    FULLY_VARYING = "fully_varying"


def _schedule(
    regime: LTVRegime, period_or_block: int | None, horizon: int
) -> tuple[np.ndarray, int]:
    """The schedule ``sigma: {0,...,horizon-1} -> {0,...,D-1}`` mapping each
    time step to a slice index into the ``D``-matrix bank, and ``D`` itself
    (NB05 plan Sec 1.1).

    Args:
        regime: Which regime's schedule to build.
        period_or_block: The period ``p`` (`PERIODIC`) or block length ``c``
            (`BLOCK_CONSTANT`); ``None`` for `FULLY_VARYING`.
        horizon: The rollout horizon ``N``.

    Returns:
        ``(sigma, D)``: `sigma` an ``int`` array of shape ``(horizon,)``,
        `D` the distinct-slice count.
    """
    if regime is LTVRegime.PERIODIC:
        assert period_or_block is not None
        return np.arange(horizon) % period_or_block, period_or_block
    if regime is LTVRegime.BLOCK_CONSTANT:
        assert period_or_block is not None
        return np.arange(horizon) // period_or_block, horizon // period_or_block
    return np.arange(horizon), horizon


def _horizon_product_growth_rate(A_stack: np.ndarray) -> float:
    """The per-step growth rate ``gamma = rho(Phi)^(1/N)`` of the horizon
    monodromy product ``Phi = A_{N-1} ... A_1 A_0`` (NB05 plan Sec 2.2 Rule
    4) -- the joint-stability generalization of NB04's per-slice spectral
    normalization, computed via a rescaled running product so that neither
    the product itself nor its eigenvalues ever overflow/underflow, and the
    whole computation stays in log-space until the final, always-moderate
    ``exp(log_rho_phi / horizon)``.

    Args:
        A_stack: Per-step state transition matrices, shape ``(N, n, n)``,
            UNNORMALIZED (this is what it solves for).

    Returns:
        `gamma`, such that dividing every slice of `A_stack` by `gamma`
        gives a horizon product with spectral radius exactly ``1``.

    Raises:
        ValueError: If the running product degenerates to a non-finite or
            (numerically) zero matrix at any step, or its final spectral
            radius is not strictly positive -- the drawn dynamics cannot be
            stability-normalized (a measure-zero event for continuous random
            draws, guarded defensively).
    """
    horizon, n, _ = A_stack.shape
    M = np.eye(n)
    log_scale = 0.0
    for t in range(horizon):
        M = A_stack[t] @ M
        norm = float(np.linalg.norm(M))
        if not math.isfinite(norm) or norm <= 0.0:
            raise ValueError(
                "The horizon product degenerated to a non-finite or zero "
                f"matrix at step {t}; this draw cannot be stability-normalized."
            )
        if norm > _RESCALE_HIGH or norm < _RESCALE_LOW:
            M = M / norm
            log_scale += math.log(norm)

    rho_M = float(np.max(np.abs(np.linalg.eigvals(M))))
    if not math.isfinite(rho_M) or rho_M <= 0.0:
        raise ValueError(
            "The horizon product's final spectral radius is not finite and "
            "positive; this draw cannot be stability-normalized."
        )
    log_rho_phi = log_scale + math.log(rho_M)
    return math.exp(log_rho_phi / horizon)


@dataclass(frozen=True)
class LTVLQRProblemFactory:
    """Builds NB05's shared box-constrained problem: a seeded, jointly
    stability-normalized linear system whose ``A_t``/``B_t`` follow one of
    three structured time-variation regimes around the same nominal draw
    `LQRProblemFactory` makes, plus the identical time-stacked identity
    quadratic cost.

    Attributes:
        state_dim: State dimension ``n``.
        control_dim: Control dimension ``m``.
        horizon: The finite horizon ``N`` the cost/dynamics are stacked for.
        seed: Seed of the nominal system draw and the perturbation bank.
        regime: Which time-variation regime to draw (`LTVRegime`).
        period_or_block: The period ``p`` (`PERIODIC`) or block length ``c``
            (`BLOCK_CONSTANT`); required (a positive divisor of `horizon`,
            NB05 plan Sec 1.3 Rule 1) for those two regimes, and must be
            ``None`` for `FULLY_VARYING` (a loud, explicit "this field is
            meaningless here" rather than a silently-ignored value).
        variation_strength: The perturbation strength ``epsilon`` in
            ``A^(d) = A_0 + epsilon * Delta^(d)`` (NB05 plan Sec 2.1).
            ``0.0`` (the default) collapses every slice to the nominal draw,
            reproducing `LQRProblemFactory`'s own problem bit-for-bit
            (Sec 2.3's degeneracy law) regardless of `regime`.
        perturb_B: Whether ``B`` is perturbed the same way `A` is (default
            ``True``); ``False`` isolates dynamics-only ("A-only")
            variation as a diagnostic (NB05 plan Sec 2.1).
        u_max: Optional box bound; ``None`` attaches no constraint.
    """

    state_dim: int
    control_dim: int
    horizon: int
    seed: int
    regime: LTVRegime
    period_or_block: int | None = None
    variation_strength: float = 0.0
    perturb_B: bool = True
    u_max: float | None = None

    def __post_init__(self) -> None:
        """Fail-fast guards (NB05 plan Sec 1.3 Rule 1, Sec 2's positivity
        requirements).

        Raises:
            ValueError: If `state_dim`/`control_dim`/`horizon` are not
                positive integers, `u_max` is not strictly positive when
                given, `variation_strength` is negative, `period_or_block`
                is missing/non-divisor for `PERIODIC`/`BLOCK_CONSTANT`, or
                `period_or_block` is given for `FULLY_VARYING`.
        """
        ensure_positive_integer(self.state_dim, "state_dim")
        ensure_positive_integer(self.control_dim, "control_dim")
        ensure_positive_integer(self.horizon, "horizon")
        if self.u_max is not None and self.u_max <= 0:
            raise ValueError(f"u_max must be strictly positive, got {self.u_max}.")
        if self.variation_strength < 0:
            raise ValueError(
                f"variation_strength must be >= 0, got {self.variation_strength}."
            )
        if self.regime is LTVRegime.FULLY_VARYING:
            if self.period_or_block is not None:
                raise ValueError(
                    "period_or_block must be None for LTVRegime.FULLY_VARYING "
                    f"(it has no period or block length), got {self.period_or_block}."
                )
            return
        if self.period_or_block is None or self.period_or_block <= 0:
            raise ValueError(
                f"period_or_block must be a positive integer for {self.regime!r}, "
                f"got {self.period_or_block}."
            )
        if self.horizon % self.period_or_block != 0:
            raise ValueError(
                f"horizon ({self.horizon}) must be divisible by period_or_block "
                f"({self.period_or_block}) for {self.regime!r} -- see NB05 plan "
                "Sec 1.3 Rule 1 (undivided periods/blocks silently mis-wrap at "
                "the horizon boundary)."
            )

    def distinct_slice_count(self) -> int:
        """The distinct-slice count ``D`` this factory's `regime`/
        `period_or_block`/`horizon` combination realizes (NB05 plan Sec 0.2).

        Returns:
            ``period_or_block`` for `PERIODIC`; ``horizon // period_or_block``
            for `BLOCK_CONSTANT`; `horizon` for `FULLY_VARYING`.
        """
        _, d = _schedule(self.regime, self.period_or_block, self.horizon)
        return d

    def schedule_indices(self) -> np.ndarray:
        """The per-step schedule ``sigma(t)``, shape ``(horizon,)`` (NB05
        plan Sec 1.1) -- `int` slice index into the ``D``-matrix bank at
        every time step.
        """
        sigma, _ = _schedule(self.regime, self.period_or_block, self.horizon)
        return sigma

    def build(self) -> OptimalControlProblem:
        """Construct the problem this factory specifies.

        Draws the nominal ``(A_0, B_0)`` from ``np.random.default_rng(seed)``
        in EXACTLY the order/shapes `LQRProblemFactory.build` does, so that
        at ``variation_strength == 0.0`` the two factories' RNG consumption
        for ``(A_0, B_0)`` -- and hence the returned problem -- agree
        bit-for-bit (Sec 2.3). The perturbation bank is drawn afterward (so
        it never perturbs the RNG state ``(A_0, B_0)`` were drawn from), and
        is skipped entirely at ``variation_strength == 0.0`` (nothing would
        consume it: multiplying by ``0.0`` discards it exactly, and every
        slice is `A_0`/`B_0`, so the degenerate path below builds the
        identical 2D, time-invariant system `LQRProblemFactory` would).

        Returns:
            The `OptimalControlProblem` (with a single `BoxConstraint` when
            `u_max` is set).
        """
        n, m, horizon = self.state_dim, self.control_dim, self.horizon
        rng = np.random.default_rng(self.seed)

        A0 = rng.normal(size=(n, n))
        A0 = A0 / np.max(np.abs(np.linalg.eigvals(A0)))
        B0 = rng.normal(size=(n, m))

        if self.variation_strength == 0.0:
            A_final, B_final = A0, B0
        else:
            sigma, d = _schedule(self.regime, self.period_or_block, horizon)
            delta = rng.normal(size=(d, n, n))
            A_bank = A0[None] + self.variation_strength * delta
            if self.perturb_B:
                xi = rng.normal(size=(d, n, m))
                B_bank = B0[None] + self.variation_strength * xi
            else:
                B_bank = np.broadcast_to(B0, (d, n, m)).copy()

            gamma = _horizon_product_growth_rate(A_bank[sigma])
            A_final = (A_bank / gamma)[sigma]
            B_final = B_bank[sigma]

        system = LinearSystem.fully_observable(A_final, B_final)

        # Identical time-stacked identity cost to LQRProblemFactory --
        # RiccatiController's finite_horizon_riccati indexes Q[horizon]/R[k].
        Q = np.repeat(np.eye(n)[None], horizon + 1, axis=0)
        R = np.repeat(np.eye(m)[None], horizon, axis=0)
        cost = QuadraticCost(Q=Q, R=R)

        constraints = [BoxConstraint(u_max=self.u_max)] if self.u_max else None
        if constraints is None:
            return OptimalControlProblem(system=system, cost=cost)
        return OptimalControlProblem(system=system, cost=cost, constraints=constraints)

    def get_signature(self) -> dict[str, Any]:
        """`Signable` member: the full problem specification. A distinct
        ``"type"`` from `LQRProblemFactory.get_signature` (never
        ``"LQRProblemFactory"``), so no NB05 cache key can ever collide with
        an NB04 one even at ``variation_strength == 0.0``.

        Returns:
            ``{"type": "LTVLQRProblemFactory", **fields}``.
        """
        return {"type": type(self).__name__, **dataclasses.asdict(self)}


def matching_lti_factory(ltv: LTVLQRProblemFactory) -> LQRProblemFactory:
    """The `LQRProblemFactory` a `LTVLQRProblemFactory` degenerates to at
    ``variation_strength == 0.0`` -- exposed so tests (and any notebook
    diagnostic) can assert the Sec 2.3 byte-identity law against the real
    NB04 factory, not merely against a plausible reimplementation of it.

    Args:
        ltv: The LTV factory to derive the matching LTI factory of; only
            `state_dim`/`control_dim`/`horizon`/`seed`/`u_max` are read --
            `regime`/`period_or_block`/`variation_strength`/`perturb_B` are
            irrelevant to the degenerate ``epsilon == 0`` problem.

    Returns:
        The equivalent `LQRProblemFactory`.
    """
    return LQRProblemFactory(
        state_dim=ltv.state_dim,
        control_dim=ltv.control_dim,
        horizon=ltv.horizon,
        seed=ltv.seed,
        u_max=ltv.u_max,
    )
