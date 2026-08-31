"""Phase 1 acceptance tests for `applications.uncertainty.distances`
(docs/planning/03_studies/nb07_robust_training/robust_training_under_model_mismatch.md Sec 4.4/11): Hellinger against
the closed-form Gaussian-Gaussian value; `W2(Cauchy) == inf`; the dimension-
lift identity.
"""

import math

import pytest

from mbl.applications.uncertainty.distances import (
    _frozen_distribution,
    _hellinger_1d,
    distance_to_nominal_gaussian,
    lift_hellinger_to_dimension,
)
from mbl.applications.uncertainty.noise import NoiseFamily


def _closed_form_gaussian_hellinger(sigma1: float, sigma2: float) -> float:
    """Closed-form Hellinger distance between two zero-mean 1-D Gaussians:
    ``H^2 = 1 - sqrt(2 s1 s2 / (s1^2 + s2^2))``."""
    bhattacharyya = math.sqrt(2 * sigma1 * sigma2 / (sigma1**2 + sigma2**2))
    return math.sqrt(max(1.0 - bhattacharyya, 0.0))


class TestHellingerClosedForm:
    @pytest.mark.parametrize(
        "sigma1,sigma2", [(1.0, 1.0), (1.0, 2.0), (0.5, 3.0), (2.0, 2.5)]
    )
    def test_matches_closed_form_gaussian_gaussian(
        self, sigma1: float, sigma2: float
    ) -> None:
        p = _frozen_distribution(NoiseFamily.GAUSSIAN, sigma1)
        q = _frozen_distribution(NoiseFamily.GAUSSIAN, sigma2)
        numeric = _hellinger_1d(p, q, max(sigma1, sigma2))
        expected = _closed_form_gaussian_hellinger(sigma1, sigma2)
        assert numeric == pytest.approx(expected, abs=1e-4)


class TestDegeneracyAnchor:
    def test_gaussian_to_itself_is_exactly_zero(self) -> None:
        distance = distance_to_nominal_gaussian(NoiseFamily.GAUSSIAN, sigma=0.5)
        assert distance.hellinger == 0.0
        assert distance.total_variation == 0.0
        assert distance.wasserstein_2 == 0.0


class TestCauchyInfiniteWasserstein:
    def test_w2_is_exactly_inf(self) -> None:
        distance = distance_to_nominal_gaussian(NoiseFamily.CAUCHY, sigma=0.5)
        assert distance.wasserstein_2 == math.inf

    def test_hellinger_and_tv_are_finite(self) -> None:
        distance = distance_to_nominal_gaussian(NoiseFamily.CAUCHY, sigma=0.5)
        assert 0.0 <= distance.hellinger <= 1.0
        assert 0.0 <= distance.total_variation <= 1.0


class TestFiniteVarianceFamiliesHaveFiniteWasserstein:
    @pytest.mark.parametrize("family", [NoiseFamily.UNIFORM, NoiseFamily.LAPLACE])
    def test_w2_is_finite_and_positive(self, family: NoiseFamily) -> None:
        distance = distance_to_nominal_gaussian(family, sigma=0.5)
        assert math.isfinite(distance.wasserstein_2)
        assert distance.wasserstein_2 > 0.0
        assert 0.0 < distance.hellinger < 1.0


class TestDimensionLift:
    def test_zero_distance_lifts_to_zero(self) -> None:
        assert lift_hellinger_to_dimension(0.0, 7) == 0.0

    def test_lift_identity_matches_direct_formula(self) -> None:
        h1 = 0.3
        n = 5
        lifted = lift_hellinger_to_dimension(h1, n)
        expected = math.sqrt(1.0 - (1.0 - h1**2) ** n)
        assert lifted == pytest.approx(expected)

    def test_lift_is_monotone_increasing_in_n(self) -> None:
        h1 = 0.2
        values = [lift_hellinger_to_dimension(h1, n) for n in range(1, 10)]
        assert all(b >= a for a, b in zip(values, values[1:]))

    def test_lift_saturates_toward_one(self) -> None:
        assert lift_hellinger_to_dimension(0.5, 200) == pytest.approx(1.0, abs=1e-6)
