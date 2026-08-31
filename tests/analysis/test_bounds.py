"""A bound is derived from the problem, declared by the author, and never typed.

`Role.BOUND` had no producer until this module, and every use of the role sat on
an attained policy -- the error the role exists to prevent, committed. What is
checked here is that a bound now enters as a bound: derived from the study's own
problem, carrying no seeds and no axis, refused when its kind is unknown, and
absent unless declared.
"""

from __future__ import annotations

import numpy as np
import pytest

from mbl.analysis.bounds import BOUND_KINDS, bound_provenance, bound_rows
from mbl.spec.errors import SpecificationError


def _study(u_max: float | None = 0.1, horizon: int = 20):
    """A minimal resolved study, built through the spec tier it is read from."""
    from mbl.applications.factories import GaussianBatchSpec
    from mbl.spec.problem import ProblemData, ProblemSpec

    rng = np.random.default_rng(0)
    n, m = 5, 2
    A = rng.normal(size=(n, n))
    A *= 0.99 / np.max(np.abs(np.linalg.eigvals(A)))
    data = ProblemData(
        system={"A": A, "B": rng.normal(size=(n, m))},
        cost={
            "Q": np.tile(np.eye(n), (horizon + 1, 1, 1)),
            "R": np.tile(np.eye(m), (horizon, 1, 1)),
        },
        horizon=horizon,
        control_bound=u_max,
    )

    class _Protocol:
        batch_spec = GaussianBatchSpec(
            state_dim=n,
            horizon=horizon,
            batch_size=8,
            seed=0,
            process_noise_std=0.5,
            initial_state_std=0.5,
        )

    class _Evaluation:
        protocol = _Protocol()

    class _Study:
        problem = ProblemSpec(data=data)
        evaluation = _Evaluation()

    return _Study()


def test_nothing_is_drawn_unless_it_is_declared() -> None:
    """A default bound would apply a claim about convention to every study."""
    assert bound_rows(_study(), {}) == []
    assert bound_provenance(_study(), {}) == {}


@pytest.mark.parametrize("kind", sorted(BOUND_KINDS))
def test_a_declared_bound_is_a_bound_and_not_a_measurement(kind: str) -> None:
    """It carries the role, and none of the things a measurement carries."""
    rows = bound_rows(_study(), {"bounds": [kind]})
    assert len(rows) == 1
    row = rows[0]
    assert row["contender"] == kind
    assert row["role"] == "bound"
    # No seeds, no trajectories, no interval, no spread: reporting any of them
    # would invent a dispersion for a quantity that has none.
    assert row["n_seeds"] == 0 and row["n_trajectories"] == 0
    assert row["seeds"] == [] and row["per_seed_aggregate"] == []
    assert row["interval_low"] == row["aggregate"] == row["interval_high"]
    assert row["within_seed_spread"] == 0.0 and row["across_seed_spread"] == 0.0
    # It does not vary with the swept axis, and says so as a flat contender does.
    assert np.isnan(row["axis_value"])


def test_the_value_is_derived_from_the_study_and_not_authored() -> None:
    """Changing the problem changes the bound; nothing here is a constant."""
    tight = bound_rows(_study(u_max=0.05), {"bounds": ["finite_horizon_box"]})[0]
    loose = bound_rows(_study(u_max=0.5), {"bounds": ["finite_horizon_box"]})[0]
    assert tight["aggregate"] > loose["aggregate"], (
        "a tighter box must not lower the floor on the constrained optimum"
    )


def test_the_two_conventions_are_different_bounds() -> None:
    """Which one a figure may draw is the author's claim, so both are offered.

    A figure of finite-horizon time-averages may draw only the finite-horizon
    one; the ICASSP campaign records omitting the other for exactly this reason.
    """
    rows = bound_rows(
        _study(), {"bounds": ["finite_horizon_box", "infinite_horizon_box"]}
    )
    assert len(rows) == 2
    finite, infinite = (r["aggregate"] for r in rows)
    assert finite != pytest.approx(infinite, rel=1e-3)


def test_the_residuals_travel_with_the_value() -> None:
    """A bound quoted without them is a number.

    This project has one on record at 56.043029, reported by a solver that
    placed it above the optimum it claimed to bound.
    """
    provenance = bound_provenance(_study(), {"bounds": ["infinite_horizon_box"]})
    entry = provenance["infinite_horizon_box"]
    assert entry["kkt_residual"] < 1e-6
    assert entry["lmi_slack"] > -1e-8
    assert (
        entry["value"]
        == bound_rows(_study(), {"bounds": ["infinite_horizon_box"]})[0]["aggregate"]
    )


def test_an_unknown_kind_is_refused_by_name() -> None:
    with pytest.raises(SpecificationError, match="unknown bound"):
        bound_rows(_study(), {"bounds": ["sdp_floor"]})


def test_a_problem_with_no_box_is_refused() -> None:
    """A box-aware bound needs a box to be aware of."""
    with pytest.raises(SpecificationError, match="control bound"):
        bound_rows(_study(u_max=None), {"bounds": ["finite_horizon_box"]})
