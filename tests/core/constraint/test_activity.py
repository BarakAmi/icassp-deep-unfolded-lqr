"""Acceptance tests for the constraint-activity statistic — Annex 03 §A.3.1a.

Written before the retention was wired into the evaluation pass. The statistic
is small and the temptation is to trust it by inspection; two things make that
unsafe, and both were found by measuring rather than by reading:

* **The tolerance is content, not housekeeping.** Across a nine-contender cast
  at one box, moving it from `0` to `1e-3` moved the exact convex policy from
  76.32 % to 87.82 % — from last place to third. A statistic whose tolerance is
  ignored, defaulted twice, or applied absolutely instead of relatively is
  wrong by more than the effects the figures discuss, so every one of those
  mistakes is driven from the failing side here.

* **The two quantities answer different questions.** `saturated` is the
  informative one and needs a tolerance; `max_abs` is the falsifiable one and
  must not have one. A suite that only checked the fraction would let
  `max_abs` silently acquire the tolerance and still pass.
"""

from __future__ import annotations

import pytest
import torch

from mbl.core.constraint.activity import (
    SATURATION_TOLERANCE,
    ControlActivity,
    control_activity,
)

U_MAX = 0.02


def _controls(values: list[list[list[float]]]) -> torch.Tensor:
    """`(batch, horizon, m)` — the layout `system.run` returns."""
    return torch.tensor(values, dtype=torch.float64)


class TestTheSaturatedFraction:
    def test_it_counts_entries_at_either_bound(self) -> None:
        """Two of four entries are at a bound, one at each — so a one-sided
        test would report a half of what this does."""
        activity = control_activity(
            _controls([[[U_MAX, -U_MAX], [0.0, 0.001]]]), u_max=U_MAX
        )
        assert activity.saturated.tolist() == pytest.approx([0.5])

    def test_the_fraction_is_per_trajectory_and_not_pooled(self) -> None:
        """§A.3 names the trajectory as the within-seed unit, so one value
        describes one trajectory. Pooling would give 0.5 for both rows here and
        lose the fact that one trajectory is pinned and the other is free."""
        activity = control_activity(
            _controls([[[U_MAX, U_MAX]], [[0.0, 0.0]]]), u_max=U_MAX
        )
        assert activity.saturated.tolist() == pytest.approx([1.0, 0.0])

    def test_the_tolerance_is_relative_and_not_absolute(self) -> None:
        """`0.0199` is 0.5 % inside a box of 0.02 — saturated at a *relative*
        `1e-2`, and untouched by an absolute one, which at this box would have
        to reach a hundredth of a unit to catch anything at all."""
        controls = _controls([[[0.0199, 0.0]]])
        assert control_activity(
            controls, u_max=U_MAX, tolerance=1e-2
        ).saturated.tolist() == pytest.approx([0.5])
        assert control_activity(
            controls, u_max=U_MAX, tolerance=1e-4
        ).saturated.tolist() == pytest.approx([0.0])

    def test_the_tolerance_is_read_and_not_defaulted(self) -> None:
        """The same controls, two tolerances, different fractions — so an
        implementation that ignored the argument cannot pass both."""
        controls = _controls([[[0.01998, 0.0]]])
        loose = control_activity(controls, u_max=U_MAX, tolerance=1e-3).saturated
        tight = control_activity(controls, u_max=U_MAX, tolerance=1e-9).saturated
        assert (float(loose[0]), float(tight[0])) == pytest.approx((0.5, 0.0))

    def test_tolerance_zero_asks_the_tolerance_free_question(self) -> None:
        """A meaningful reading in its own right — how many entries are at the
        *representable* bound — and the one the campaign's QP families answer
        differently from every other contender."""
        controls = _controls([[[U_MAX, 0.019999]]])
        assert control_activity(
            controls, u_max=U_MAX, tolerance=0.0
        ).saturated.tolist() == pytest.approx([0.5])

    def test_the_declared_default_is_the_one_that_is_used(self) -> None:
        """A default that disagreed with the constant would put one number in
        the annex and another in the store."""
        controls = _controls([[[0.019985, 0.0]]])
        assert control_activity(controls, u_max=U_MAX).saturated.tolist() == (
            control_activity(
                controls, u_max=U_MAX, tolerance=SATURATION_TOLERANCE
            ).saturated.tolist()
        )
        assert SATURATION_TOLERANCE == 1e-3

    def test_an_asymmetric_box_is_counted_on_the_side_each_entry_is_near(
        self,
    ) -> None:
        """`ProblemSpec` authors symmetric boxes, so this is the case with no
        user today — and the two-sided form costs one line, needs no
        enumeration, and reduces to `|u| >= (1 - tol) * u_max` exactly where
        the box is symmetric."""
        activity = control_activity(
            _controls([[[0.05, -0.01, 0.0]]]), u_max=0.05, u_min=-0.01
        )
        assert activity.saturated.tolist() == pytest.approx([2 / 3])

    def test_a_box_that_does_not_straddle_zero_is_counted_inward(self) -> None:
        """The case `(1 - tolerance) * bound` gets backwards.

        For a lower bound ABOVE zero, scaling it moves the threshold the wrong
        way: `(1 - 1e-2) * 0.1 = 0.099` counts an entry BELOW the bound as at
        it and misses the entry actually on the face. Measuring inward from the
        bound's own magnitude gets both right, and is the identical expression
        for every box this project authors.

        Three entries: one on the lower face, one a hair inside it, one in the
        interior. Two of the three are at a bound.
        """
        activity = control_activity(
            _controls([[[0.1, 0.1005, 0.3]]]), u_max=0.5, u_min=0.1, tolerance=1e-2
        )
        assert activity.saturated.tolist() == pytest.approx([2 / 3])

    def test_a_zero_bound_admits_only_exact_contact(self) -> None:
        """A bound of zero has no magnitude for a relative band to scale, so
        the tolerance contributes nothing and the test is exact.

        Written first with an entry BELOW the zero bound and expecting it not
        to count, which was wrong about this statistic rather than about the
        code — see the test below.
        """
        activity = control_activity(
            _controls([[[0.0, 0.2, 0.4]]]), u_max=0.5, u_min=0.0, tolerance=1e-2
        )
        assert activity.saturated.tolist() == pytest.approx([1 / 3])

    def test_an_entry_outside_the_box_counts_as_active(self) -> None:
        """Deliberate, and the reason `max_abs` is retained beside this. An
        entry past a face is not *less* active than one on it, so the
        comparisons are inclusive of violation — which means this fraction
        cannot be read as evidence of feasibility, and the audit reports the
        tolerance-free maximum for that.

        Both entries here violate; both count.
        """
        activity = control_activity(
            _controls([[[0.9, -0.9]]]), u_max=0.5, tolerance=0.0
        )
        assert activity.saturated.tolist() == pytest.approx([1.0])
        assert float(activity.max_abs[0]) == pytest.approx(0.9)

    def test_a_per_dimension_bound_broadcasts(self) -> None:
        """One bound per actuator: the first entry is at its own bound and the
        second is nowhere near a much larger one."""
        activity = control_activity(
            _controls([[[0.02, 0.02]]]), u_max=torch.tensor([0.02, 1.0])
        )
        assert activity.saturated.tolist() == pytest.approx([0.5])


class TestTheMaximum:
    def test_it_is_exact_and_carries_no_tolerance(self) -> None:
        """Feasibility is a claim about the worst entry and no tolerance may
        enter it; the same controls under two tolerances give one maximum."""
        controls = _controls([[[0.0199, -0.015]]])
        assert float(control_activity(controls, u_max=U_MAX, tolerance=0.0).max_abs[0])
        assert float(
            control_activity(controls, u_max=U_MAX, tolerance=0.0).max_abs[0]
        ) == float(control_activity(controls, u_max=U_MAX, tolerance=0.5).max_abs[0])

    def test_it_sees_a_violation_the_fraction_cannot(self) -> None:
        """The reason both quantities are retained. This trajectory is
        infeasible by 50 % and yet saturated on only one entry of four; the
        fraction alone would report it as an ordinary well-behaved policy."""
        activity = control_activity(_controls([[[0.03, 0.0], [0.0, 0.0]]]), u_max=U_MAX)
        assert float(activity.max_abs[0]) > U_MAX
        assert float(activity.saturated[0]) == pytest.approx(0.25)

    def test_it_is_the_maximum_and_not_a_mean(self) -> None:
        """A mean over the trajectory's entries would read 0.01 here and hide
        the one entry that decides feasibility."""
        activity = control_activity(
            _controls([[[U_MAX, 0.0], [0.0, 0.0]]]), u_max=U_MAX
        )
        assert float(activity.max_abs[0]) == pytest.approx(U_MAX)

    def test_it_is_per_trajectory(self) -> None:
        activity = control_activity(
            _controls([[[0.03, 0.0]], [[0.001, 0.0]]]), u_max=U_MAX
        )
        assert activity.max_abs.tolist() == pytest.approx([0.03, 0.001])


class TestWhatTravelsWithTheNumbers:
    def test_the_tolerance_is_carried_on_the_result(self) -> None:
        """A saturated fraction quoted without its tolerance says nothing, so
        the result carries it rather than leaving a caller to remember."""
        activity = control_activity(_controls([[[0.0, 0.0]]]), u_max=U_MAX)
        assert isinstance(activity, ControlActivity)
        assert activity.tolerance == SATURATION_TOLERANCE
        assert (
            control_activity(
                _controls([[[0.0, 0.0]]]), u_max=U_MAX, tolerance=1e-6
            ).tolerance
            == 1e-6
        )

    def test_both_quantities_describe_the_same_trajectories(self) -> None:
        """One row each, in the same order — the property that lets them be
        written as two columns on one pairing key."""
        activity = control_activity(
            _controls([[[0.02, 0.0]], [[0.0, 0.0]]]), u_max=U_MAX
        )
        assert activity.max_abs.shape == activity.saturated.shape == (2,)
