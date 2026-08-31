"""Statistical distance of an exotic noise family from the nominal Gaussian
(docs/planning/03_studies/nb06_ood_generalization/zero_shot_ood_generalization_benchmark.md Sec 2.4): Hellinger and total
variation are used as the x-axis because they are finite for EVERY family,
including Cauchy, which has no finite second moment and therefore an
INFINITE Wasserstein-2 distance to a Gaussian (`wasserstein_2` reports
`math.inf` for `NoiseFamily.CAUCHY` rather than a numerically-truncated
stand-in -- a documented fact, not an approximation).
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.integrate import quad
from scipy.stats import cauchy, laplace, norm, t, uniform  # type: ignore[attr-defined]  # scipy.stats lazy-exports these; mypy's follow_untyped_imports can't resolve the dynamic __getattr__

from .noise import NoiseFamily, family_scale_parameter

#: Half-width (in standard deviations of the INTEGRATION variable `z`, not
#: of the compared distributions) of the Gaussian-weighted quadrature used
#: by `_wasserstein_2_to_gaussian`'s change of variables -- 8 sigma leaves
#: residual Gaussian mass beyond it around 1e-15, negligible at float64
#: precision regardless of how heavy the other distribution's tail is.
_QUANTILE_INTEGRATION_HALF_WIDTH = 8.0

#: Clamp applied to the substituted CDF argument before evaluating the
#: other distribution's PPF, so `norm.cdf(+-half_width)` (which is not
#: exactly 0/1 in floating point, but can round to a value whose PPF is
#: +-inf) never produces a non-finite integrand value.
_PPF_ARGUMENT_EPS = 1e-14


@dataclass(frozen=True)
class DistributionDistance:
    """Three statistical distances of one distribution from the nominal
    Gaussian (NB06 plan Sec 2.4).

    Attributes:
        hellinger: ``H(p, q) in [0, 1]``; finite for every family.
        total_variation: ``TV(p, q) in [0, 1]``; finite for every family.
        wasserstein_2: ``W2(p, q) >= 0``; `math.inf` when the compared
            family has no finite second moment (`NoiseFamily.CAUCHY`).
    """

    hellinger: float
    total_variation: float
    wasserstein_2: float


def hellinger_total_variation(
    pdf_p: Callable[[np.ndarray], np.ndarray],
    pdf_q: Callable[[np.ndarray], np.ndarray],
) -> tuple[float, float]:
    """1-D Hellinger distance and total variation between two densities, by
    direct numerical quadrature over the whole real line -- well-defined for
    ANY pair of probability densities, independent of whether either has
    finite moments (unlike Wasserstein).

    Args:
        pdf_p: The first density, vectorized (``np.ndarray -> np.ndarray``).
        pdf_q: The second density, vectorized the same way.

    Returns:
        ``(hellinger, total_variation)``, ``H = sqrt(1 - BC)`` where
        ``BC = integral(sqrt(p*q))`` is the Bhattacharyya coefficient, and
        ``TV = 0.5 * integral(|p - q|)``.
    """
    bhattacharyya, _ = quad(
        lambda x: np.sqrt(pdf_p(x) * pdf_q(x)), -np.inf, np.inf, limit=200
    )
    absolute_difference, _ = quad(
        lambda x: np.abs(pdf_p(x) - pdf_q(x)), -np.inf, np.inf, limit=200
    )
    hellinger = math.sqrt(max(1.0 - bhattacharyya, 0.0))
    total_variation = 0.5 * absolute_difference
    return hellinger, total_variation


def _wasserstein_2_to_gaussian(
    ppf_other: Callable[[float], float], sigma: float
) -> float:
    """``W2`` of an arbitrary distribution from ``N(0, sigma^2)``, via the
    quantile-difference identity ``W2^2 = integral_0^1 (F^-1(u) - G^-1(u))^2
    du`` with the substitution ``u = Phi(z)`` (so ``G^-1(u) = sigma * z``
    exactly) -- this turns the integral into one over the WHOLE real line
    against the standard normal density, avoiding the boundary singularities
    `ppf(u)` has as ``u -> 0, 1`` that a direct integral over ``(0, 1)``
    would hit.

    Args:
        ppf_other: The other distribution's quantile function.
        sigma: The nominal Gaussian's standard deviation.

    Returns:
        ``W2 >= 0``. Callers must special-case a family with no finite
        second moment (`NoiseFamily.CAUCHY`) themselves -- this function
        does not detect divergence, it assumes the integral is finite.
    """

    def integrand(z: float) -> float:
        u = norm.cdf(z)
        u = min(max(u, _PPF_ARGUMENT_EPS), 1.0 - _PPF_ARGUMENT_EPS)
        diff = ppf_other(u) - sigma * z
        return float(diff**2 * norm.pdf(z))

    value, _ = quad(
        integrand,
        -_QUANTILE_INTEGRATION_HALF_WIDTH,
        _QUANTILE_INTEGRATION_HALF_WIDTH,
        limit=200,
    )
    return math.sqrt(max(value, 0.0))


def distance_to_nominal_gaussian(
    family: NoiseFamily, *, sigma: float, df: float = 3.0
) -> DistributionDistance:
    """`family`'s statistical distance from ``N(0, sigma^2)``, at the same
    variance-matched (or, for Cauchy, nominally-labeled) scale
    `family_scale_parameter` gives `ExoticBatchSpec` itself -- so the
    reported distance describes the EXACT distribution being sampled, not a
    parallel approximation of it.

    Args:
        family: Which family to measure.
        sigma: The nominal Gaussian's standard deviation.
        df: `NoiseFamily.STUDENT_T`'s degrees of freedom.

    Returns:
        The `DistributionDistance` (all zero for `NoiseFamily.GAUSSIAN`,
        since the nominal distribution is itself the reference).
    """
    if family is NoiseFamily.GAUSSIAN:
        return DistributionDistance(
            hellinger=0.0, total_variation=0.0, wasserstein_2=0.0
        )

    scale = family_scale_parameter(family, sigma, df)
    gaussian = norm(scale=sigma)
    other = _scipy_distribution(family, scale, df)

    hellinger, total_variation = hellinger_total_variation(gaussian.pdf, other.pdf)
    wasserstein_2 = (
        math.inf
        if family is NoiseFamily.CAUCHY
        else _wasserstein_2_to_gaussian(other.ppf, sigma)
    )
    return DistributionDistance(
        hellinger=hellinger,
        total_variation=total_variation,
        wasserstein_2=wasserstein_2,
    )


def _scipy_distribution(family: NoiseFamily, scale: float, df: float) -> Any:
    """The `scipy.stats` frozen distribution matching `family` at its own
    native `scale` (see `family_scale_parameter`)."""
    if family is NoiseFamily.UNIFORM:
        return uniform(loc=-scale, scale=2.0 * scale)
    if family is NoiseFamily.LAPLACE:
        return laplace(scale=scale)
    if family is NoiseFamily.STUDENT_T:
        return t(df=df, scale=scale)
    if family is NoiseFamily.CAUCHY:
        return cauchy(scale=scale)
    raise ValueError(f"No scipy.stats distribution wired for family {family!r}.")


def lift_hellinger_to_dimension(h1: float, n: int) -> float:
    """The ``n``-dimensional Hellinger distance of an i.i.d. product of `n`
    coordinates, each at the 1-D Hellinger distance `h1` from its Gaussian
    counterpart -- via the Bhattacharyya coefficient's multiplicativity
    under independence: ``BC_n = BC_1^n``, so
    ``H_n = sqrt(1 - (1 - H_1^2)^n)``.

    Args:
        h1: The per-coordinate (1-D) Hellinger distance, in ``[0, 1]``.
        n: The number of i.i.d. coordinates.

    Returns:
        The lifted ``n``-dimensional Hellinger distance, in ``[0, 1]``.
    """
    bhattacharyya_1d = 1.0 - h1**2
    bhattacharyya_n = bhattacharyya_1d**n
    return math.sqrt(max(1.0 - bhattacharyya_n, 0.0))
