"""NB03's "Convergence Margins" table (NB03 overhaul Phase 4a): the
unfolding-DEPTH analogue of NB02's `build_convergence_margin_table` -- same
"drops and stays within p%" rule, applied to a depth-indexed cost curve
rather than a solver's per-iteration history, with results reported as
actual depth values, never raw array indices."""

import pytest

from mbl.workbench import build_unfolding_convergence_margin_table


def test_reports_the_actual_depth_not_an_array_index() -> None:
    """A curve that enters the 5% margin at its 3rd swept depth (array index
    2) must report the depth VALUE at that index (K=6), not the index (2)."""
    k_values = [2, 4, 6, 8, 10]
    j_opt = 1.0
    # Costs stay > 5% margin (1.05) at K=2,4; drop to and stay within it
    # from K=6 onward.
    curves = {"unfolded_alpha": [1.5, 1.2, 1.02, 1.01, 1.0]}

    table = build_unfolding_convergence_margin_table(k_values, curves, j_opt, [5.0])

    assert table.loc["unfolded_alpha", "5% margin"] == 6


def test_not_reached_when_margin_never_durably_entered() -> None:
    k_values = [1, 2, 3]
    curves = {"standard_gd": [2.0, 1.9, 1.8]}  # never within 5% of J_opt=1.0

    table = build_unfolding_convergence_margin_table(k_values, curves, 1.0, [5.0])

    assert table.loc["standard_gd", "5% margin"] == "not reached"


def test_one_row_per_contender_one_column_per_margin() -> None:
    k_values = [1, 2, 3]
    curves = {
        "standard_gd": [1.5, 1.1, 1.0],
        "unfolded_alpha": [1.2, 1.05, 1.0],
    }

    table = build_unfolding_convergence_margin_table(k_values, curves, 1.0, [1.0, 10.0])

    assert set(table.index) == {"standard_gd", "unfolded_alpha"}
    assert list(table.columns) == ["1% margin", "10% margin"]


def test_rejects_empty_curves() -> None:
    with pytest.raises(ValueError, match="at least one"):
        build_unfolding_convergence_margin_table([1, 2, 3], {}, 1.0, [5.0])


def test_rejects_a_curve_whose_length_disagrees_with_k_values() -> None:
    with pytest.raises(ValueError, match="shape"):
        build_unfolding_convergence_margin_table(
            [1, 2, 3], {"bad": [1.0, 2.0]}, 1.0, [5.0]
        )


def test_accepts_a_non_contiguous_k_values_set() -> None:
    """ITERATIONS_LIST need not be a contiguous 1..N range -- the remapped
    depth must come from k_values itself, never index+1."""
    k_values = [1, 5, 20]
    curves = {"unfolded_alpha_p": [3.0, 1.0, 1.0]}  # already at J_opt by K=5

    table = build_unfolding_convergence_margin_table(k_values, curves, 1.0, [1.0])

    assert table.loc["unfolded_alpha_p", "1% margin"] == 5


def test_reached_at_the_first_depth_reports_that_depth() -> None:
    k_values = [3, 6, 9]
    curves = {"c": [1.0, 1.0, 1.0]}  # already converged at the first swept depth

    table = build_unfolding_convergence_margin_table(k_values, curves, 1.0, [1.0])

    assert table.loc["c", "1% margin"] == 3
