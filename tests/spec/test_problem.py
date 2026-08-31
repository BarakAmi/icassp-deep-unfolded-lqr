"""Acceptance tests for `ProblemSpec` — Stage 2 Phase A, decision D19.

Written before the implementation. The decision under test is that **problem
data is frozen, not generated**: a `ProblemID` is derived from the matrices
themselves, while the generator that produced them is provenance and never
touches identity.

Both halves have to hold, and each is worthless alone:

* Identity must be **blind to provenance**, or re-recording how a problem was
  authored would orphan every model trained on it.
* Identity must be **exhaustively sensitive to the data**, or freezing would
  have bought a stable identifier that no longer identifies anything. Every
  matrix is perturbed, not one representative -- a rule that covers N arrays is
  verified N times.

The portability claim is checked in a **subprocess**, because a `ProblemID`
recomputed inside the interpreter that built the object proves only that the
function is deterministic, not that the *stored* form reproduces it.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mbl.spec.problem import (
    COST_KEYS,
    SYSTEM_KEYS,
    GeneratorProvenance,
    ProblemData,
    ProblemSpec,
    SpecificationError,
)

N, M, HORIZON = 4, 2, 6


def _lti(seed: int = 0) -> ProblemData:
    rng = np.random.default_rng(seed)
    return ProblemData(
        system={"A": rng.normal(size=(N, N)), "B": rng.normal(size=(N, M))},
        cost={
            "Q": np.repeat(np.eye(N)[None], HORIZON + 1, axis=0),
            "R": np.repeat(np.eye(M)[None], HORIZON, axis=0),
        },
        horizon=HORIZON,
    )


def _ltv(seed: int = 0) -> ProblemData:
    rng = np.random.default_rng(seed)
    return ProblemData(
        system={
            "A": rng.normal(size=(HORIZON, N, N)),
            "B": rng.normal(size=(HORIZON, N, M)),
        },
        cost={
            "Q": np.repeat(np.eye(N)[None], HORIZON + 1, axis=0),
            "R": np.repeat(np.eye(M)[None], HORIZON, axis=0),
        },
        horizon=HORIZON,
    )


def _provenance(seed: int = 0) -> GeneratorProvenance:
    return GeneratorProvenance(
        generator="generate_marginally_stable_system",
        params={"state_dim": N, "control_dim": M, "seed": seed},
        package_version="0.1.0",
    )


# --------------------------------------------------------------------------
# Identity is derived from the data
# --------------------------------------------------------------------------


def test_identical_data_yields_one_identifier() -> None:
    """Determinism, without which nothing else here means anything."""
    assert ProblemSpec(_lti()).problem_id == ProblemSpec(_lti()).problem_id


def test_different_data_yields_different_identifiers() -> None:
    assert ProblemSpec(_lti(0)).problem_id != ProblemSpec(_lti(1)).problem_id


@pytest.mark.parametrize(
    ("group", "key"), [("system", "A"), ("system", "B"), ("cost", "Q"), ("cost", "R")]
)
def test_a_one_ulp_perturbation_of_any_matrix_changes_the_identifier(
    group: str, key: str
) -> None:
    """The sensitivity half of D19, over **every** stored matrix.

    Freezing the data buys a stable identifier; it must not buy one that has
    stopped discriminating. A single representative matrix would leave three
    others free to drift silently -- the omission that shipped when a gate was
    wired into two of three golden channels.
    """
    baseline = _lti()
    source = getattr(baseline, group)
    perturbed = {name: value.copy() for name, value in source.items()}
    flat = perturbed[key].ravel()
    flat[0] = np.nextafter(flat[0], np.inf)
    assert flat[0] != source[key].ravel()[0]

    changed = ProblemData(
        system=perturbed if group == "system" else baseline.system,
        cost=perturbed if group == "cost" else baseline.cost,
        horizon=baseline.horizon,
        control_bound=baseline.control_bound,
    )
    assert ProblemSpec(changed).problem_id != ProblemSpec(baseline).problem_id


def test_the_control_bound_participates_in_identity() -> None:
    """A box-constrained problem is a different problem."""
    free = ProblemData(**{**_fields(_lti()), "control_bound": None})
    bounded = ProblemData(**{**_fields(_lti()), "control_bound": 0.1})
    other = ProblemData(**{**_fields(_lti()), "control_bound": 0.2})
    ids = {ProblemSpec(d).problem_id for d in (free, bounded, other)}
    assert len(ids) == 3


def test_a_longer_horizon_yields_a_different_identifier() -> None:
    """Behavioural, and named for what it actually shows.

    It cannot isolate the `horizon` field, because validation forbids a horizon
    that disagrees with the cost shapes -- so the two always move together. The
    invariant that makes the field safe is pinned separately below.
    """
    short = _lti()
    long = ProblemData(
        system=short.system,
        cost={
            "Q": np.repeat(np.eye(N)[None], HORIZON + 2, axis=0),
            "R": np.repeat(np.eye(M)[None], HORIZON + 1, axis=0),
        },
        horizon=HORIZON + 1,
    )
    assert ProblemSpec(short).problem_id != ProblemSpec(long).problem_id


def test_the_horizon_is_implied_by_the_cost_shapes() -> None:
    """Why `horizon` cannot be tested in isolation, stated rather than left as
    a surviving mutant.

    `hash_array` covers an array's shape as well as its bytes, and validation
    requires `Q` to be `(N + 1, n, n)`. The horizon is therefore already inside
    the cost hashes, and blanking the `horizon` entry of the signature changes
    no identifier -- provably, not accidentally. The field stays because it
    makes the tree readable, and this test is what would fail first if
    validation were ever relaxed to let the two disagree.
    """
    data = _lti()
    assert data.cost["Q"].shape[0] == data.horizon + 1
    assert data.cost["R"].shape[0] == data.horizon

    with pytest.raises(SpecificationError, match="horizon"):
        ProblemData(system=data.system, cost=data.cost, horizon=data.horizon + 1)
    with pytest.raises(SpecificationError, match="horizon"):
        ProblemData(system=data.system, cost=data.cost, horizon=data.horizon - 1)


def _fields(data: ProblemData) -> dict:
    return {
        "system": data.system,
        "cost": data.cost,
        "horizon": data.horizon,
        "control_bound": data.control_bound,
    }


def test_key_insertion_order_does_not_change_the_identifier() -> None:
    """Two authors spelling the same problem must collide, not diverge. A test
    that builds both dictionaries in one order could never see this."""
    forward = _lti()
    reversed_order = ProblemData(
        system=dict(reversed(list(forward.system.items()))),
        cost=dict(reversed(list(forward.cost.items()))),
        horizon=forward.horizon,
    )
    assert ProblemSpec(reversed_order).problem_id == ProblemSpec(forward).problem_id


# --------------------------------------------------------------------------
# Identity is blind to provenance -- the other half of D19
# --------------------------------------------------------------------------


def test_provenance_does_not_participate_in_identity() -> None:
    """The decision itself. Recording *how* a problem was authored, or
    re-recording it differently, must not orphan the models trained on it."""
    data = _lti()
    without = ProblemSpec(data)
    with_one = ProblemSpec(data, provenance=_provenance(0))
    with_other = ProblemSpec(data, provenance=_provenance(999))
    assert without.problem_id == with_one.problem_id == with_other.problem_id


def test_the_signature_tree_contains_no_provenance_key() -> None:
    """Structural, not behavioural: equal identifiers could also come from a
    provenance field that happens to hash the same. The tree must not carry it
    at all."""
    spec = ProblemSpec(_lti(), provenance=_provenance())
    rendered = json.dumps(spec.get_signature(), sort_keys=True, default=str)
    assert "generate_marginally_stable_system" not in rendered
    assert "provenance" not in rendered
    assert "package_version" not in rendered


def test_provenance_survives_a_round_trip_even_though_it_is_unsigned() -> None:
    """Unsigned is not the same as discarded -- it is the record of how the
    matrices came to exist, and it is the reason freezing them is acceptable."""
    spec = ProblemSpec(_lti(), provenance=_provenance(7))
    assert spec.provenance is not None
    assert spec.provenance.params["seed"] == 7


# --------------------------------------------------------------------------
# Portability -- the point of freezing
# --------------------------------------------------------------------------


def test_a_stored_problem_reproduces_its_identifier_in_a_fresh_interpreter(
    tmp_path: Path,
) -> None:
    """The portability claim D19 exists to make.

    Recomputing the identifier in the process that built the object proves only
    determinism. Loading the *stored* form in a new interpreter is what shows
    the identifier travels -- which is the whole reason the matrices are frozen
    rather than regenerated per host.
    """
    path = tmp_path / "problem.npz"
    spec = ProblemSpec(_lti(), provenance=_provenance())
    spec.save(path)

    probe = (
        "from mbl.spec.problem import ProblemSpec;"
        f"print(ProblemSpec.load({str(path)!r}).problem_id)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == spec.problem_id


def test_a_round_trip_preserves_the_matrices_exactly(tmp_path: Path) -> None:
    """Bit-exact, not to a tolerance: a stored problem that drifts by one ULP
    is a different problem, and this is the write path where that could
    happen."""
    path = tmp_path / "problem.npz"
    original = ProblemSpec(_ltv(), provenance=_provenance())
    original.save(path)
    loaded = ProblemSpec.load(path)

    for group in ("system", "cost"):
        source, target = getattr(original.data, group), getattr(loaded.data, group)
        assert set(source) == set(target)
        for name in source:
            assert np.array_equal(source[name], target[name]), name
    assert loaded.data.horizon == original.data.horizon
    assert loaded.provenance == original.provenance


def test_an_archive_written_uncompressed_still_loads(tmp_path: Path) -> None:
    """`save` deflates, and every problem frozen before it did does not.

    A storage format that could only read what the current writer produces
    would strand `studies/box_lqr/box_lqr_n4m2_N50_u0.5_s0.npz` — the one
    problem that predates this — and with it every model whose identity points
    at those matrices. Written here in the older form deliberately, rather
    than trusting that `np.load` sniffs the container.
    """
    original = ProblemSpec(_lti(), provenance=_provenance())
    compressed, plain = tmp_path / "deflated.npz", tmp_path / "stored.npz"
    original.save(compressed)

    with np.load(compressed, allow_pickle=False) as archive:
        np.savez(plain, **{name: archive[name] for name in archive.files})

    assert plain.stat().st_size > compressed.stat().st_size
    assert ProblemSpec.load(plain).problem_id == original.problem_id
    assert ProblemSpec.load(compressed).problem_id == original.problem_id


def test_compression_does_not_reach_the_identifier(tmp_path: Path) -> None:
    """The bytes on disk are storage; the identifier is derived from the
    matrices. If these ever disagree, changing a storage detail would silently
    orphan every model trained on the problem."""
    original = ProblemSpec(_ltv(), provenance=_provenance())
    path = tmp_path / "problem.npz"
    original.save(path)
    assert ProblemSpec.load(path).problem_id == original.problem_id


def test_a_round_trip_preserves_an_absent_provenance(tmp_path: Path) -> None:
    """A hand-specified problem has no generator, and that is the native case
    under D19 rather than a special one."""
    path = tmp_path / "problem.npz"
    ProblemSpec(_lti()).save(path)
    assert ProblemSpec.load(path).provenance is None


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def test_float32_matrices_are_refused() -> None:
    """Problem data is float64 whatever the training precision. Storing it at
    the training dtype would make one problem two under D20, and holding a
    float32 quantity to a float64 contract is the defect that made the
    golden-master suite runner-dependent."""
    data = _lti()
    with pytest.raises(SpecificationError, match="float64"):
        ProblemData(
            system={k: v.astype(np.float32) for k, v in data.system.items()},
            cost=data.cost,
            horizon=data.horizon,
        )


def test_a_missing_matrix_is_refused() -> None:
    data = _lti()
    with pytest.raises(SpecificationError, match="B"):
        ProblemData(
            system={"A": data.system["A"]}, cost=data.cost, horizon=data.horizon
        )


def test_inconsistent_dimensions_are_refused() -> None:
    """`A` and `B` must agree on the state dimension, or the problem is not a
    problem and the failure surfaces much later, inside a solver."""
    data = _lti()
    with pytest.raises(SpecificationError, match="state dimension"):
        ProblemData(
            system={"A": data.system["A"], "B": np.zeros((N + 1, M))},
            cost=data.cost,
            horizon=data.horizon,
        )


def test_a_cost_horizon_mismatch_is_refused() -> None:
    data = _lti()
    with pytest.raises(SpecificationError, match="horizon"):
        ProblemData(system=data.system, cost=data.cost, horizon=HORIZON + 3)


def test_a_non_positive_control_bound_is_refused() -> None:
    with pytest.raises(SpecificationError, match="control_bound"):
        ProblemData(**{**_fields(_lti()), "control_bound": 0.0})


# --------------------------------------------------------------------------
# Dimensions and construction
# --------------------------------------------------------------------------


def test_dimensions_are_read_from_the_data() -> None:
    spec = ProblemSpec(_lti())
    assert (spec.state_dim, spec.control_dim, spec.horizon) == (N, M, HORIZON)


def test_a_time_varying_system_reports_the_same_dimensions() -> None:
    """LTI stores `A` as `(n, n)` and LTV as `(N, n, n)`; the dimensions a
    caller asks for must not depend on which."""
    spec = ProblemSpec(_ltv())
    assert (spec.state_dim, spec.control_dim, spec.horizon) == (N, M, HORIZON)


def test_a_time_varying_problem_has_its_own_identity() -> None:
    assert ProblemSpec(_ltv()).problem_id != ProblemSpec(_lti()).problem_id


def test_build_produces_a_problem_of_the_declared_shape() -> None:
    problem = ProblemSpec(_lti()).build()
    assert problem.system.dimensions.state_dim == N
    assert problem.system.dimensions.control_dim == M
    assert not problem.constraints


def test_build_attaches_a_box_constraint_only_when_bounded() -> None:
    bounded = ProblemSpec(ProblemData(**{**_fields(_lti()), "control_bound": 0.25}))
    assert len(bounded.build().constraints or []) == 1


def test_every_stored_matrix_is_covered_by_the_perturbation_test() -> None:
    """Negative control for the parametrisation above.

    The rule it enforces is "every matrix", so the list of cases must be the
    list of matrices. A matrix added to `ProblemData` and forgotten here would
    otherwise be free to drift out of identity silently -- and the parametrised
    test would still report green.
    """
    covered = {
        ("system", "A"),
        ("system", "B"),
        ("cost", "Q"),
        ("cost", "R"),
    }
    data = _lti()
    stored = {("system", name) for name in data.system} | {
        ("cost", name) for name in data.cost
    }
    assert stored == covered
    assert set(SYSTEM_KEYS) | set(COST_KEYS) == {key for _, key in covered}
