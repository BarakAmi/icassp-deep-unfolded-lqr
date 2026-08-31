"""Unit tests for `workbench.analysis.compute_box_binding_fraction` -- the
NB04 plan's (docs/planning/03_studies/nb04_box_constrained/box_constrained_lqr_benchmark.md Sec 6/9) u_max-binding
acceptance gate: if the box never binds on a problem instance, the
box-constrained benchmark degenerates to the unconstrained one.
"""

import numpy as np

from mbl.workbench.analysis import compute_box_binding_fraction


class TestComputeBoxBindingFraction:
    def test_all_within_bound_gives_zero(self) -> None:
        controls = np.array([[0.1, -0.2], [0.05, 0.3]])
        assert compute_box_binding_fraction(controls, u_max=0.5) == 0.0

    def test_all_exceeding_bound_gives_one(self) -> None:
        controls = np.array([[1.0, -2.0], [3.0, 4.0]])
        assert compute_box_binding_fraction(controls, u_max=0.5) == 1.0

    def test_partial_violation_gives_the_exact_fraction(self) -> None:
        controls = np.array([0.1, 0.9, -0.9, 0.2])  # 2 of 4 entries exceed 0.5
        assert compute_box_binding_fraction(controls, u_max=0.5) == 0.5

    def test_handles_batched_trajectory_shape(self) -> None:
        rng = np.random.default_rng(0)
        controls = rng.normal(scale=2.0, size=(8, 10, 3))  # (batch, horizon, m)
        fraction = compute_box_binding_fraction(controls, u_max=1.0)
        assert 0.0 < fraction < 1.0

    def test_boundary_value_does_not_count_as_exceeding(self) -> None:
        controls = np.array([0.5, -0.5])
        assert compute_box_binding_fraction(controls, u_max=0.5) == 0.0
