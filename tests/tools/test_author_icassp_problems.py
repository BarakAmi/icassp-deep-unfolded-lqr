"""Acceptance tests for Phase A of the ICASSP campaign — the frozen plants.

The four properties the campaign plan names, each asserted by something that
can fail:

* **The spectral radius is the one asked for**, and a miss is a refusal rather
  than a warning. This assertion exists here because the `stability` gate that
  would otherwise make it is inert — it is routed to `DECIDED_AT_PARSE` without
  ever looking at `A`.
* **A frozen problem round-trips**, identifier included. A `ProblemID` that
  moved across a save/load would detach every model trained on it.
* **The rotation rotates `A` and nothing else.** `B` bit-identical is the whole
  experiment: co-rotating it makes the transform a similarity, under which a
  controller handed the rotated matrices attains the nominal cost exactly.
* **An existing file is not silently replaced.**

The plan's traps are covered too: `ProblemSpec.save` returns a path that does
not exist when the name lacks `.npz`, and the factory's own normalisation lands
strictly above 1.0 on some seeds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mbl.applications.factories import LQRProblemFactory
from mbl.spec.problem import ProblemSpec

from tools.author_icassp_problems import (
    require_matching_nominal,
    DEFAULT_DEGREES,
    DEFAULT_RHO,
    RHO_TOLERANCE,
    AuthoringError,
    _require_spectral_radius,
    author_nominal,
    author_rotated_a,
    author_rotated_ab,
    main,
    nominal_filename,
    rotated_filename,
    spectral_radius,
)

#: The declared instance. Small horizon here: nothing under test depends on N
#: being 100, and a stacked Q at N=100 makes every fixture 8x larger.
INSTANCE = {
    "state_dim": 4,
    "control_dim": 2,
    "horizon": 12,
    "u_max": 0.1,
}


def _nominal(seed: int = 0, **overrides: object) -> ProblemSpec:
    return author_nominal(**{**INSTANCE, "seed": seed, **overrides})


# --------------------------------------------------------------------------
# The spectral radius, which nothing downstream checks
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(5))
def test_every_authored_plant_hits_the_target_spectral_radius(seed: int) -> None:
    A = _nominal(seed).data.system["A"]
    assert abs(spectral_radius(A) - DEFAULT_RHO) <= RHO_TOLERANCE


@pytest.mark.parametrize("seed", range(5))
def test_the_factory_alone_does_not_land_at_or_below_one(seed: int) -> None:
    """The negative control, and the measurement that motivated the target.

    If the factory already produced a spectral radius reliably at or below 1,
    the rescale above would be ceremony. It does not: its normalisation is
    exact only in exact arithmetic, and in float64 it lands strictly above 1.0
    on some seeds — which is what makes `rho <= 1`, read as an inequality, a
    claim no plant here satisfies.
    """
    problem = LQRProblemFactory(
        state_dim=INSTANCE["state_dim"],
        control_dim=INSTANCE["control_dim"],
        horizon=INSTANCE["horizon"],
        seed=seed,
        u_max=INSTANCE["u_max"],
    ).build()
    rho = spectral_radius(np.asarray(problem.system.A_t.array))
    assert rho == pytest.approx(1.0, abs=1e-12)
    assert rho != DEFAULT_RHO


def test_a_missed_target_is_refused() -> None:
    """The guard is unit-tested directly, and deliberately so.

    The first version of this test asserted that `author_nominal(rho=0)` is
    refused *by the spectral-radius check*. It is not: scaling by `0 / rho(A)`
    produces the zero matrix, whose spectral radius really is 0, so the check
    passed on a plant with no dynamics at all. The assertion was unfailable
    through that path. `rho <= 0` is now refused by name, above, and the check
    itself is exercised where it can actually miss.
    """
    with pytest.raises(AuthoringError, match="spectral radius"):
        _require_spectral_radius(2.0 * np.eye(3), DEFAULT_RHO, context="probe")
    _require_spectral_radius(DEFAULT_RHO * np.eye(3), DEFAULT_RHO, context="probe")


@pytest.mark.parametrize("rho", [0.0, -1.0, float("nan"), float("inf")])
def test_a_target_that_is_not_a_positive_radius_is_refused(rho: float) -> None:
    with pytest.raises(AuthoringError, match="rho must be"):
        _nominal(rho=rho)


def test_a_nonpositive_bound_is_refused() -> None:
    with pytest.raises(AuthoringError, match="u_max"):
        _nominal(u_max=0.0)


# --------------------------------------------------------------------------
# The frozen file
# --------------------------------------------------------------------------


def test_a_frozen_problem_round_trips_with_its_identifier(tmp_path: Path) -> None:
    spec = _nominal()
    target = tmp_path / nominal_filename(**INSTANCE, seed=0)
    spec.save(target)
    loaded = ProblemSpec.load(target)

    assert loaded.problem_id == spec.problem_id
    assert loaded.horizon == INSTANCE["horizon"]
    assert loaded.data.control_bound == INSTANCE["u_max"]
    for group, key in (("system", "A"), ("system", "B"), ("cost", "Q"), ("cost", "R")):
        np.testing.assert_array_equal(
            getattr(loaded.data, group)[key], getattr(spec.data, group)[key]
        )


def test_the_cost_is_the_declared_convention(tmp_path: Path) -> None:
    """No terminal term, averaged over time — both are the built problem's
    defaults, and the campaign depends on them being executed rather than
    declared. Asserted on the built object, not on the specification."""
    problem = _nominal().build()
    assert problem.cost.conventions.include_terminal_cost is False
    assert problem.cost.conventions.is_time_averaged is True


def test_two_seeds_are_two_different_problems() -> None:
    assert _nominal(0).problem_id != _nominal(1).problem_id


# --------------------------------------------------------------------------
# The rotation: A moves, B does not
# --------------------------------------------------------------------------


def test_the_rotation_leaves_b_bit_identical() -> None:
    """The whole of Experiment 1 rests on this. Co-rotating `B` makes the
    transform a similarity, and on this isotropic instance the re-solved
    optimum was measured to move by 0.000e+00 — the experiment would measure
    nothing at all."""
    nominal = _nominal()
    rotated = author_rotated_a(nominal, degrees=DEFAULT_DEGREES)

    np.testing.assert_array_equal(rotated.data.system["B"], nominal.data.system["B"])
    assert not np.array_equal(rotated.data.system["A"], nominal.data.system["A"])
    assert rotated.problem_id != nominal.problem_id


def test_the_rotation_preserves_the_spectral_radius() -> None:
    """`A -> R A R^T` is a similarity, so the eigenvalues are invariant and the
    rotated plant inherits the target stability. If this fails the rotation
    matrix was not orthogonal."""
    rotated = author_rotated_a(_nominal(), degrees=DEFAULT_DEGREES)
    assert abs(spectral_radius(rotated.data.system["A"]) - DEFAULT_RHO) <= 1e-12


def test_the_rotation_moves_a_by_an_amount_that_grows_with_the_angle() -> None:
    """A rotation that is applied but ignored would pass every test above.
    This one fails if the angle is not honoured."""
    nominal = _nominal()
    A = nominal.data.system["A"]
    distances = [
        float(np.linalg.norm(author_rotated_a(nominal, degrees=d).data.system["A"] - A))
        for d in (0.0, 5.0, 15.0, 30.0)
    ]
    assert distances[0] == pytest.approx(0.0, abs=1e-12)
    assert distances == sorted(distances)
    assert distances[-1] > 0.1


def test_the_rotation_records_what_it_rotated() -> None:
    rotated = author_rotated_a(_nominal(), degrees=DEFAULT_DEGREES)
    assert rotated.provenance is not None
    params = rotated.provenance.params
    assert params["degrees"] == DEFAULT_DEGREES
    assert params["rotated_from"] == str(_nominal().problem_id)
    assert "A only" in params["rotates"]


def test_provenance_does_not_reach_identity() -> None:
    """Two problems whose matrices agree must collide, however they were
    authored — otherwise the rotated file could never be reproduced."""
    left = _nominal()
    right = ProblemSpec(data=left.data, provenance=None)
    assert left.problem_id == right.problem_id


# --------------------------------------------------------------------------
# The co-rotated (similarity-control) companion — the author's ruling of
# 2026-08-08: `A' = R A R^T` AND `B' = R B` is the nominal plant in rotated
# coordinates, so its invariances are checks of the whole pipeline.
# --------------------------------------------------------------------------


def test_the_co_rotation_shares_the_a_rotation_bit_for_bit() -> None:
    """The only difference between the two shifted plants must be whether `B`
    co-rotates — same `R`, same `A'` — or comparing their columns compares two
    different rotations."""
    nominal = _nominal()
    a_only = author_rotated_a(nominal, degrees=DEFAULT_DEGREES)
    both = author_rotated_ab(nominal, degrees=DEFAULT_DEGREES)

    np.testing.assert_array_equal(both.data.system["A"], a_only.data.system["A"])


def test_the_co_rotation_moves_b_and_the_three_plants_are_distinct() -> None:
    nominal = _nominal()
    a_only = author_rotated_a(nominal, degrees=DEFAULT_DEGREES)
    both = author_rotated_ab(nominal, degrees=DEFAULT_DEGREES)

    assert not np.array_equal(both.data.system["B"], nominal.data.system["B"])
    assert len({str(s.problem_id) for s in (nominal, a_only, both)}) == 3


def test_the_co_rotation_preserves_the_spectral_radius() -> None:
    both = author_rotated_ab(_nominal(), degrees=DEFAULT_DEGREES)
    assert abs(spectral_radius(both.data.system["A"]) - DEFAULT_RHO) <= 1e-12


def test_the_co_rotation_preserves_the_gradient_lipschitz_constant() -> None:
    """The unitary-equivalence invariance the campaign leans on: `B'^T P' B' =
    B^T P B`, so `L` is the nominal plant's and the analytic PGD's nominal
    step literal is exactly right on this plant — no per-plant override. The
    A-only rotation is the anti-vacuity control: there `L` genuinely moves,
    or this test could pass on a constant function."""
    from mbl.models.analytic.riccati import problem_gradient_lipschitz_constant

    nominal = _nominal()
    both = author_rotated_ab(nominal, degrees=DEFAULT_DEGREES)
    a_only = author_rotated_a(nominal, degrees=DEFAULT_DEGREES)

    L_nominal = problem_gradient_lipschitz_constant(nominal.build())
    L_both = problem_gradient_lipschitz_constant(both.build())
    L_a_only = problem_gradient_lipschitz_constant(a_only.build())

    assert L_both == pytest.approx(L_nominal, rel=1e-9)
    assert abs(L_a_only - L_nominal) / L_nominal > 1e-4


def test_the_co_rotation_records_what_it_rotated() -> None:
    both = author_rotated_ab(_nominal(), degrees=DEFAULT_DEGREES)
    assert both.provenance is not None
    params = both.provenance.params
    assert params["degrees"] == DEFAULT_DEGREES
    assert params["rotated_from"] == str(_nominal().problem_id)
    assert "A and B" in params["rotates"]


def test_the_ab_flag_writes_the_third_file(tmp_path: Path) -> None:
    assert main([*_argv(tmp_path), "--ab"]) == 0
    names = sorted(path.name for path in tmp_path.glob("*.npz"))
    assert len(names) == 3
    assert any("rotAB" in name for name in names)


# --------------------------------------------------------------------------
# The command line
# --------------------------------------------------------------------------


def _argv(destination: Path, **overrides: object) -> list[str]:
    flags = {
        "--state-dim": INSTANCE["state_dim"],
        "--control-dim": INSTANCE["control_dim"],
        "--horizon": INSTANCE["horizon"],
        "--u-max": INSTANCE["u_max"],
        "--destination": destination,
        **overrides,
    }
    return [str(part) for pair in flags.items() for part in pair]


def test_the_command_writes_both_files(tmp_path: Path) -> None:
    assert main(_argv(tmp_path)) == 0
    written = sorted(path.name for path in tmp_path.glob("*.npz"))
    assert len(written) == 2
    assert any("rotA30" in name for name in written)


def test_every_written_file_exists_and_loads(tmp_path: Path) -> None:
    """The trap: `ProblemSpec.save` appends `.npz` and returns the path it was
    given, so a caller that trusts the return value can report a file that is
    not there."""
    assert main(_argv(tmp_path)) == 0
    for path in tmp_path.glob("*.npz"):
        assert path.exists()
        assert ProblemSpec.load(path).state_dim == INSTANCE["state_dim"]


def test_an_existing_file_is_refused_and_left_alone(tmp_path: Path) -> None:
    assert main(_argv(tmp_path)) == 0
    before = {path: path.read_bytes() for path in tmp_path.glob("*.npz")}

    assert main(_argv(tmp_path)) == 1
    assert {path: path.read_bytes() for path in tmp_path.glob("*.npz")} == before


def test_force_replaces_it(tmp_path: Path) -> None:
    assert main(_argv(tmp_path)) == 0
    assert main([*_argv(tmp_path), "--force"]) == 0


def test_degrees_zero_writes_only_the_nominal(tmp_path: Path) -> None:
    assert main(_argv(tmp_path, **{"--degrees": 0})) == 0
    assert len(list(tmp_path.glob("*.npz"))) == 1


def test_the_command_is_reproducible(tmp_path: Path) -> None:
    """Same seed, same matrices, same identifier — the property that lets a
    large plant be regenerated instead of committed."""
    left, right = tmp_path / "left", tmp_path / "right"
    assert main(_argv(left)) == 0
    assert main(_argv(right)) == 0
    for path in left.glob("*.npz"):
        assert (
            ProblemSpec.load(path).problem_id
            == ProblemSpec.load(right / path.name).problem_id
        )


def test_the_filename_carries_every_distinguishing_quantity() -> None:
    name = nominal_filename(state_dim=4, control_dim=2, horizon=100, u_max=0.1, seed=3)
    assert "n4m2" in name and "N100" in name and "s3" in name
    assert name.endswith(".npz")
    assert "." not in name.removesuffix(".npz"), "a dot would confuse the suffix check"
    assert rotated_filename(name, degrees=30).endswith("_rotA30.npz")


def test_a_companion_is_verified_against_the_frozen_nominal(tmp_path: Path) -> None:
    """`--rotations-only` exists so a new angle never rewrites the plant every
    stored model points at. What makes it safe is the converse check: the
    nominal it derives from must BE the frozen one, or the companion would be
    a rotation of a plant nobody else uses — and both files would look
    perfectly well formed."""
    nominal = _nominal()
    frozen = tmp_path / "nominal.npz"
    nominal.save(frozen)
    assert require_matching_nominal(nominal, frozen).problem_id == nominal.problem_id

    other = _nominal(seed=3)
    with pytest.raises(AuthoringError, match="differs from the authored one"):
        require_matching_nominal(other, frozen)

    with pytest.raises(AuthoringError, match="does not exist"):
        require_matching_nominal(nominal, tmp_path / "absent.npz")
