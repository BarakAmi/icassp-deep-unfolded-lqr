"""Phase E/F acceptance tests for the two NB05-driven `workbench.analysis`
additions (docs/planning/03_studies/nb05_ltv_box_constrained/ltv_box_constrained_lqr_benchmark.md Sec 6.7/7):
`compute_box_binding_fraction_per_timestep` (the per-t binding profile) and
`estimate_crossover` (the alignment-curve crossover, with its heuristic
uncertainty bracket -- cross-checked against known analytic answers, not
merely "runs without error").
"""

import numpy as np
import pytest

from mbl.workbench.analysis import (
    CrossoverEstimate,
    compute_box_binding_fraction,
    compute_box_binding_fraction_per_timestep,
    estimate_crossover,
)


class TestPerTimestepBindingFraction:
    def test_matches_the_scalar_version_when_averaged_over_time(self) -> None:
        rng = np.random.default_rng(0)
        controls = rng.normal(size=(20, 6, 3)) * 2.0
        u_max = 1.0

        per_t = compute_box_binding_fraction_per_timestep(controls, u_max)
        scalar = compute_box_binding_fraction(controls, u_max)

        assert per_t.shape == (6,)
        assert np.mean(per_t) == pytest.approx(scalar)

    def test_isolates_a_single_binding_timestep(self) -> None:
        controls = np.zeros((5, 4, 2))
        controls[:, 1, :] = 10.0
        result = compute_box_binding_fraction_per_timestep(controls, u_max=1.0)
        np.testing.assert_array_equal(result, [0.0, 1.0, 0.0, 0.0])

    def test_never_binding_is_all_zero(self) -> None:
        controls = np.full((5, 4, 2), 0.1)
        result = compute_box_binding_fraction_per_timestep(controls, u_max=1.0)
        np.testing.assert_array_equal(result, np.zeros(4))

    def test_always_binding_is_all_one(self) -> None:
        controls = np.full((5, 4, 2), 10.0)
        result = compute_box_binding_fraction_per_timestep(controls, u_max=1.0)
        np.testing.assert_array_equal(result, np.ones(4))


class TestEstimateCrossover:
    """Cross-checked against a KNOWN exact answer: gap(x) = x - 5 is exactly
    linear, so linear interpolation recovers the true root exactly, not
    approximately."""

    def test_recovers_the_exact_root_of_a_linear_gap(self) -> None:
        x = [0, 2, 4, 6, 8, 10]
        gap = [xx - 5 for xx in x]

        result = estimate_crossover(x, gap)

        assert isinstance(result, CrossoverEstimate)
        assert result.found
        assert result.x_estimate == pytest.approx(5.0)
        assert result.bracket_index == 2
        assert result.x_lower is None and result.x_upper is None

    def test_uncertainty_bracket_widens_symmetrically_for_a_symmetric_perturbation(
        self,
    ) -> None:
        x = [0, 2, 4, 6, 8, 10]
        gap = [xx - 5 for xx in x]
        result = estimate_crossover(x, gap, gap_std=[0.1] * 6)

        assert result.x_lower == pytest.approx(4.9)
        assert result.x_upper == pytest.approx(5.1)
        assert result.x_lower is not None
        assert result.x_upper is not None
        assert result.x_estimate is not None
        assert result.x_lower < result.x_estimate < result.x_upper

    def test_not_found_when_gap_never_changes_sign(self) -> None:
        result = estimate_crossover([0, 2, 4, 6, 8, 10], [-5, -4, -3, -2, -1, -0.5])

        assert not result.found
        assert result.x_estimate is None
        assert result.bracket_index is None

    def test_reports_only_the_first_crossing(self) -> None:
        """gap = -1, +1, -1, +1 crosses three times; the function reports
        only the first (index 0)."""
        result = estimate_crossover([0, 1, 2, 3], [-1, 1, -1, 1])
        assert result.found
        assert result.bracket_index == 0

    def test_mismatched_lengths_rejected(self) -> None:
        with pytest.raises(ValueError, match="length"):
            estimate_crossover([0, 1, 2], [0, 1])

    def test_mismatched_std_length_rejected(self) -> None:
        with pytest.raises(ValueError, match="gap_std"):
            estimate_crossover([0, 1, 2], [-1, 0, 1], gap_std=[0.1, 0.1])

    def test_too_few_points_rejected(self) -> None:
        with pytest.raises(ValueError, match="2 points"):
            estimate_crossover([0], [0])

    def test_larger_uncertainty_widens_the_bracket(self) -> None:
        x = [0, 2, 4, 6, 8, 10]
        gap = [xx - 5 for xx in x]
        tight = estimate_crossover(x, gap, gap_std=[0.05] * 6)
        loose = estimate_crossover(x, gap, gap_std=[0.5] * 6)

        assert tight.x_upper is not None and tight.x_lower is not None
        assert loose.x_upper is not None and loose.x_lower is not None
        assert (loose.x_upper - loose.x_lower) > (tight.x_upper - tight.x_lower)
