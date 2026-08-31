"""Phase 1 acceptance tests for `applications.ood.distances`
(docs/planning/03_studies/nb06_ood_generalization/zero_shot_ood_generalization_benchmark.md Sec 2.4/8): the numerical
Hellinger/total-variation quadrature checked against the closed-form
Gaussian-Gaussian Hellinger distance, `wasserstein_2 == inf` for Cauchy, and
the dimension-lift identity.
"""

import math

import pytest
from scipy.stats import norm

from mbl.applications.ood.distances import (
    distance_to_nominal_gaussian,
    hellinger_total_variation,
    lift_hellinger_to_dimension,
)
from mbl.applications.ood.noise import NoiseFamily


def _closed_form_gaussian_hellinger(sigma1: float, sigma2: float) -> float:
    """H(N(0,s1^2), N(0,s2^2)) has a known closed form (independent of the
    numerical quadrature under test): BC = sqrt(2*s1*s2/(s1^2+s2^2))."""
    bhattacharyya = math.sqrt(2.0 * sigma1 * sigma2 / (sigma1**2 + sigma2**2))
    return math.sqrt(1.0 - bhattacharyya)


class TestHellingerTotalVariation:
    @pytest.mark.parametrize("sigma1,sigma2", [(1.0, 1.0), (1.0, 2.0), (0.5, 3.0)])
    def test_matches_the_closed_form_gaussian_gaussian_value(
        self, sigma1: float, sigma2: float
    ) -> None:
        hellinger, _ = hellinger_total_variation(
            norm(scale=sigma1).pdf, norm(scale=sigma2).pdf
        )
        expected = _closed_form_gaussian_hellinger(sigma1, sigma2)
        assert hellinger == pytest.approx(expected, abs=1e-6)

    def test_identical_distributions_have_zero_distance(self) -> None:
        hellinger, total_variation = hellinger_total_variation(
            norm(scale=1.0).pdf, norm(scale=1.0).pdf
        )
        # Bounded by scipy.integrate.quad's own quadrature precision, not
        # exact float equality -- both are numerical integrals.
        assert hellinger == pytest.approx(0.0, abs=1e-6)
        assert total_variation == pytest.approx(0.0, abs=1e-6)

    def test_both_metrics_stay_within_unit_bounds(self) -> None:
        hellinger, total_variation = hellinger_total_variation(
            norm(scale=0.1).pdf, norm(scale=50.0).pdf
        )
        assert 0.0 <= hellinger <= 1.0
        assert 0.0 <= total_variation <= 1.0


class TestDistanceToNominalGaussian:
    def test_gaussian_family_is_exactly_zero(self) -> None:
        result = distance_to_nominal_gaussian(NoiseFamily.GAUSSIAN, sigma=0.5)
        assert result.hellinger == 0.0
        assert result.total_variation == 0.0
        assert result.wasserstein_2 == 0.0

    @pytest.mark.parametrize(
        "family",
        [
            NoiseFamily.UNIFORM,
            NoiseFamily.LAPLACE,
            NoiseFamily.STUDENT_T,
            NoiseFamily.CAUCHY,
        ],
    )
    def test_every_exotic_family_has_a_finite_positive_hellinger_and_tv(
        self, family: NoiseFamily
    ) -> None:
        result = distance_to_nominal_gaussian(family, sigma=0.5)
        assert 0.0 < result.hellinger <= 1.0
        assert 0.0 < result.total_variation <= 1.0
        assert math.isfinite(result.hellinger)
        assert math.isfinite(result.total_variation)

    def test_wasserstein_2_is_finite_for_light_tailed_families(self) -> None:
        for family in (NoiseFamily.UNIFORM, NoiseFamily.LAPLACE, NoiseFamily.STUDENT_T):
            result = distance_to_nominal_gaussian(family, sigma=0.5)
            assert math.isfinite(result.wasserstein_2)
            assert result.wasserstein_2 >= 0.0

    def test_wasserstein_2_is_infinite_for_cauchy(self) -> None:
        result = distance_to_nominal_gaussian(NoiseFamily.CAUCHY, sigma=0.5)
        assert result.wasserstein_2 == math.inf
        # Hellinger/TV remain finite even though W2 diverges -- the whole
        # point of using them as the x-axis.
        assert math.isfinite(result.hellinger)

    def test_cauchy_is_farther_than_laplace_on_both_metrics(self) -> None:
        # Cauchy (unbounded variance) is the most extreme family at matched
        # nominal scale; Laplace is the closest of the four exotic families
        # to a Gaussian on both metrics -- an ordering that holds regardless
        # of which of the two bounded metrics is read (unlike e.g.
        # Student-t vs Uniform, whose relative order flips between
        # Hellinger and TV because TV also penalizes Uniform's hard-edged,
        # compact support, not just tail weight).
        cauchy_distance = distance_to_nominal_gaussian(NoiseFamily.CAUCHY, sigma=0.5)
        laplace_distance = distance_to_nominal_gaussian(NoiseFamily.LAPLACE, sigma=0.5)
        assert cauchy_distance.hellinger > laplace_distance.hellinger
        assert cauchy_distance.total_variation > laplace_distance.total_variation


class TestDimensionLift:
    def test_zero_distance_stays_zero_at_any_dimension(self) -> None:
        assert lift_hellinger_to_dimension(0.0, 10) == pytest.approx(0.0)

    def test_dimension_one_is_the_identity(self) -> None:
        assert lift_hellinger_to_dimension(0.3, 1) == pytest.approx(0.3)

    def test_monotonically_increases_with_dimension(self) -> None:
        values = [lift_hellinger_to_dimension(0.1, n) for n in (1, 2, 4, 8, 16)]
        assert all(a < b for a, b in zip(values, values[1:]))

    def test_stays_within_unit_bound(self) -> None:
        assert lift_hellinger_to_dimension(0.5, 100) <= 1.0
