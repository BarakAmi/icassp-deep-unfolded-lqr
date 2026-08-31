"""Distribution-distance numerics for the exotic noise families
(docs/planning/03_studies/nb07_robust_training/robust_training_under_model_mismatch.md Sec 4.4/5.1): Hellinger, total
variation, and (where finite) Wasserstein-2 distance from each family's 1-D
marginal to the nominal zero-mean Gaussian AT THE SAME declared scale -- the
standardization `noise.py`'s families are built to honor, so
``distance_to_nominal_gaussian(GAUSSIAN, sigma=s)`` is the exact-zero anchor.
Pure numerics: no torch, no I/O.
"""

import math
from dataclasses import dataclass
from typing import Any

from scipy import integrate
from scipy.stats import cauchy, laplace, norm, uniform  # type: ignore[attr-defined]  # scipy.stats lazy-exports these; mypy's follow_untyped_imports can't resolve the dynamic __getattr__

from .noise import NoiseFamily, _cauchy_scale, _laplace_scale, _uniform_half_width

#: Integration half-width in multiples of the shared scale parameter -- wide
#: enough that every family's tail mass beyond it is negligible relative to
#: `scipy.integrate.quad`'s own default tolerance.
_INTEGRATION_HALF_WIDTH = 200.0
_QUAD_LIMIT = 400
#: Keeps the Wasserstein-2 quantile integral away from the exact endpoints
#: (where a Gaussian's `ppf` is formally +-inf), at a truncation cost that is
#: negligible for every finite-second-moment family this module evaluates.
_QUANTILE_EPSILON = 1e-9


def _frozen_distribution(family: NoiseFamily, scale: float) -> Any:
    """The `scipy.stats` frozen distribution for `family` at the declared
    scale, using the SAME standardization `noise.py` builds its samplers
    from (`_uniform_half_width`/`_laplace_scale`/`_cauchy_scale`) -- so a
    distance computed here is a distance between the distributions actually
    sampled, never a parallel description of them. Returns `Any`: scipy
    ships no usable stub for its frozen-distribution return type (see the
    lazy-export `type: ignore` above)."""
    if family == NoiseFamily.GAUSSIAN:
        return norm(loc=0.0, scale=scale)
    if family == NoiseFamily.UNIFORM:
        half_width = _uniform_half_width(scale)
        return uniform(loc=-half_width, scale=2 * half_width)
    if family == NoiseFamily.LAPLACE:
        return laplace(loc=0.0, scale=_laplace_scale(scale))
    if family == NoiseFamily.CAUCHY:
        return cauchy(loc=0.0, scale=_cauchy_scale(scale))
    raise ValueError(f"Unknown NoiseFamily {family!r}.")


def _hellinger_1d(p: Any, q: Any, scale: float) -> float:
    """Hellinger distance between two 1-D densities, via numerical
    quadrature of the Bhattacharyya coefficient ``BC = int sqrt(p q)``:
    ``H = sqrt(1 - BC)``.

    Args:
        p: The first frozen distribution.
        q: The second frozen distribution.
        scale: The shared scale parameter, used only to size the (finite)
            integration bound -- never to alter the distributions themselves.

    Returns:
        ``H(p, q) in [0, 1]``.
    """
    bound = _INTEGRATION_HALF_WIDTH * scale

    def integrand(x: float) -> float:
        return math.sqrt(p.pdf(x) * q.pdf(x))

    bhattacharyya, _ = integrate.quad(integrand, -bound, bound, limit=_QUAD_LIMIT)
    bhattacharyya = min(max(bhattacharyya, 0.0), 1.0)
    return float(math.sqrt(max(1.0 - bhattacharyya, 0.0)))


def _total_variation_1d(p: Any, q: Any, scale: float) -> float:
    """Total variation distance between two 1-D densities:
    ``TV = 1/2 int |p - q|``.

    Args:
        p: The first frozen distribution.
        q: The second frozen distribution.
        scale: See `_hellinger_1d`.

    Returns:
        ``TV(p, q) in [0, 1]``.
    """
    bound = _INTEGRATION_HALF_WIDTH * scale

    def integrand(x: float) -> float:
        return float(abs(p.pdf(x) - q.pdf(x)))

    total, _ = integrate.quad(integrand, -bound, bound, limit=_QUAD_LIMIT)
    return float(min(max(0.5 * total, 0.0), 1.0))


def _wasserstein_2(family: NoiseFamily, sigma: float) -> float:
    """Wasserstein-2 distance from `family` (at `sigma`) to the nominal
    Gaussian, via the 1-D quantile-difference identity
    ``W_2^2 = int_0^1 (Q_p(u) - Q_q(u))^2 du``.

    `NoiseFamily.CAUCHY` returns `math.inf` directly rather than attempting
    numerical integration: its quantile function grows too fast near the
    integration endpoints for the integral to converge (the analytic fact
    that a Cauchy has no finite second moment), and this is reported
    explicitly rather than left to a quadrature routine's unpredictable
    behavior on a divergent integral (NB07 plan Sec 4.5/13).

    Args:
        family: The non-nominal family.
        sigma: The shared scale parameter.

    Returns:
        The Wasserstein-2 distance, or `math.inf` for `CAUCHY`.
    """
    if family == NoiseFamily.CAUCHY:
        return math.inf
    if family == NoiseFamily.GAUSSIAN:
        return 0.0
    nominal = _frozen_distribution(NoiseFamily.GAUSSIAN, sigma)
    other = _frozen_distribution(family, sigma)

    def integrand(u: float) -> float:
        return float((nominal.ppf(u) - other.ppf(u)) ** 2)

    value, _ = integrate.quad(
        integrand, _QUANTILE_EPSILON, 1.0 - _QUANTILE_EPSILON, limit=_QUAD_LIMIT
    )
    return float(math.sqrt(max(value, 0.0)))


@dataclass(frozen=True)
class DistributionDistance:
    """The three distance statistics NB07's exotic-noise-family table
    reports (NB07 plan Sec 4.4/6.5).

    Attributes:
        hellinger: In ``[0, 1]``; the x-axis for every noise-family figure
            (finite for every family, including Cauchy).
        total_variation: In ``[0, 1]``; a secondary table column.
        wasserstein_2: Finite for finite-second-moment families; `math.inf`
            for `NoiseFamily.CAUCHY` (see `_wasserstein_2`).
    """

    hellinger: float
    total_variation: float
    wasserstein_2: float

    def get_signature(self) -> dict[str, Any]:
        """`Signable` member: the three computed statistics.

        Returns:
            ``{"type": "DistributionDistance", "hellinger":,
            "total_variation":, "wasserstein_2":}``.
        """
        return {
            "type": type(self).__name__,
            "hellinger": self.hellinger,
            "total_variation": self.total_variation,
            "wasserstein_2": self.wasserstein_2,
        }


def distance_to_nominal_gaussian(
    family: NoiseFamily, *, sigma: float
) -> DistributionDistance:
    """The three 1-D distances from `family` (at scale `sigma`, standardized
    exactly as `noise.py` samples it) to the nominal zero-mean Gaussian at
    the same `sigma`.

    Args:
        family: The family to measure.
        sigma: The shared scale parameter (`ExoticBatchSpec.process_noise_std`
            or `.initial_state_std`, before any `scale_multiplier`).

    Returns:
        The `DistributionDistance`; exactly zero in every field when
        `family == NoiseFamily.GAUSSIAN` (the degeneracy anchor).
    """
    if family == NoiseFamily.GAUSSIAN:
        return DistributionDistance(
            hellinger=0.0, total_variation=0.0, wasserstein_2=0.0
        )
    nominal = _frozen_distribution(NoiseFamily.GAUSSIAN, sigma)
    other = _frozen_distribution(family, sigma)
    return DistributionDistance(
        hellinger=_hellinger_1d(nominal, other, sigma),
        total_variation=_total_variation_1d(nominal, other, sigma),
        wasserstein_2=_wasserstein_2(family, sigma),
    )


def lift_hellinger_to_dimension(h1: float, n: int) -> float:
    """Lift a 1-D (per-coordinate) Hellinger distance to its `n`-dimensional
    i.i.d.-product value: ``H_n^2 = 1 - (1 - H_1^2)^n``.

    Args:
        h1: The 1-D Hellinger distance, in ``[0, 1]``.
        n: The number of i.i.d. coordinates.

    Returns:
        ``H_n in [0, 1]``.
    """
    h1_sq = h1**2
    h_n_sq = 1.0 - (1.0 - h1_sq) ** n
    return float(math.sqrt(max(h_n_sq, 0.0)))
