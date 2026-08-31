"""Acceptance tests for `workbench.ood_gates`
(docs/planning/03_studies/nb06_ood_generalization/zero_shot_ood_generalization_benchmark.md Sec 3.11): each gate returns
``passed=False`` (never raises) outside its window/threshold and
``passed=True`` inside it, with the v1 executed run's own measured numbers
used as concrete regression anchors -- that run's 77.6% binding fraction
must fail the box-binding gate, and a ~2.1% nominal cost spread must fail
the separation gate at a plausible Monte-Carlo stderr.
"""

import math

import pytest

from mbl.workbench.ood_gates import (
    check_box_binding_window,
    check_closed_loop_stationarity,
    check_nominal_separation,
)


class TestBoxBindingGate:
    def test_passes_inside_the_window(self) -> None:
        result = check_box_binding_window(0.30)
        assert result.passed
        assert result.fraction == 0.30

    def test_fails_above_the_window_the_v1_run_actually_hit(self) -> None:
        # The executed v1 run's own measured fraction (NB06 plan Sec 11 S1).
        result = check_box_binding_window(0.776)
        assert not result.passed
        assert "Raise U_MAX" in result.diagnosis()

    def test_fails_below_the_window(self) -> None:
        result = check_box_binding_window(0.001)
        assert not result.passed
        assert "Lower U_MAX" in result.diagnosis()

    def test_boundary_values_are_inclusive(self) -> None:
        low, high = 0.05, 0.60
        assert check_box_binding_window(low, window=(low, high)).passed
        assert check_box_binding_window(high, window=(low, high)).passed


class TestNominalSeparationGate:
    def test_passes_with_a_wide_spread(self) -> None:
        costs = {"a": 1.0, "b": 2.0, "c": 3.0}
        result = check_nominal_separation(costs, mc_stderr=0.1, threshold=3.0)
        assert result.passed
        assert result.spread == pytest.approx(2.0)
        assert result.ratio == pytest.approx(20.0)

    def test_fails_at_the_v1_runs_own_spread(self) -> None:
        # The executed v1 run's nominal costs spanned ~4.420-4.514 (NB06
        # plan Sec 11 S1/S2) -- a 2.1% spread. A plausible per-contender MC
        # stderr on that scale (comparable to the spread itself) fails the
        # 3x-separation gate.
        costs = {
            "truncated_riccati": 4.514399,
            "unfolded_alpha": 4.471191,
            "unfolded_alpha_p": 4.420432,
            "cocp": 4.434609,
            "cocp_lower_bound": 4.424049,
        }
        result = check_nominal_separation(costs, mc_stderr=0.05, threshold=3.0)
        assert not result.passed
        assert "indistinguishable" in result.diagnosis()

    def test_infinite_ratio_when_stderr_is_zero(self) -> None:
        result = check_nominal_separation({"a": 1.0, "b": 2.0}, mc_stderr=0.0)
        assert result.ratio == math.inf
        assert result.passed

    def test_empty_costs_raises(self) -> None:
        with pytest.raises(ValueError, match="nominal_costs"):
            check_nominal_separation({}, mc_stderr=1.0)


class TestStationarityGate:
    def test_passes_when_cost_has_converged(self) -> None:
        result = check_closed_loop_stationarity(4.20, 4.24, tolerance=0.05)
        assert result.passed
        assert result.relative_change == pytest.approx((4.24 - 4.20) / 4.20)

    def test_fails_at_the_v1_runs_own_horizon_doubling(self) -> None:
        # The executed v1 run's horizon axis: 4.217433 at N=15 -> 8.175407
        # at N=60 (NB06 plan Sec 11 S5) -- cost nearly doubled, nowhere near
        # converged.
        result = check_closed_loop_stationarity(4.217433, 8.175407)
        assert not result.passed
        assert "has not reached a stationary distribution" in result.diagnosis()

    def test_non_positive_cost_at_half_raises(self) -> None:
        with pytest.raises(ValueError, match="cost_at_half"):
            check_closed_loop_stationarity(0.0, 1.0)
