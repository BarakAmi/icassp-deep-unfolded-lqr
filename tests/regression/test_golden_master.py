"""Stage S0 golden-master regression harness (REFACTOR_PLAN v3 §7.1, §8 S0).

Loads the frozen fixtures produced by ``tests.regression.capture_golden`` on
the untouched pre-refactor tree and compares them against a FRESH execution
of the identical scenario code (``tests.regression.scenarios``). Every code
change in stages S1-S6 must keep this suite green; the ONLY sanctioned
exceptions are the four known structural defects (C1-C4), each documented in
``test_known_defect_sentinels.py`` with its own sentinel so that fixing it
produces an intentional, reviewed fixture/test update rather than a
mysterious failure here.

Identity contract:

* Every persisted array is compared elementwise with
  ``np.testing.assert_allclose`` (torch goldens via ``torch.allclose``) at the
  tolerance its **channel's working precision** earns -- 1e-12 for a float64
  pipeline, 1e-5 relative for a float32 one -- **except those whose measured
  conditioning shows they cannot meet it** (next bullet). The tolerance follows
  the channel rather than the storage dtype of an individual array, because
  ``J_history`` is persisted as float64 while holding float32 quantities, and a
  per-array rule would demand of it a precision it never had. Asking 1e-12 of a
  float32 pipeline is asking for bit-identity, which two hosts do not owe each
  other: it made this suite pass or fail according to which CPU the CI runner
  happened to allocate.
* **A declared precision describes the arithmetic, not the conditioning**, and
  the rule above needed both. The ``signal_space_gd`` solve is float32, takes
  100 gradient steps and deliberately does not converge, so each per-member
  iterate is a point part-way along a trajectory: perturb the input in its last
  bit and you land elsewhere along it. Measured, one float32 ULP moves
  ``random_jacobi__X_final_head`` by 7.3% of its own magnitude. Such arrays are
  listed in ``fixtures/ulp_movement.json``, excluded from the value comparison
  by ``unreproducible_arrays``, and still checked for dtype, shape and
  finiteness -- nothing is deleted and no fixture value is re-baselined. A
  tolerance wide enough to admit them could not fail on anything short of total
  breakage, which is worse than not asserting them.
  See ``docs/methods/cross_machine_numerical_reproducibility.md``.
* Full-resolution ensembles that are too large to freeze verbatim are
  additionally locked by SHA-256 content hashes (``hash_array``) -- a strict
  bit-for-bit sentinel, verified bit-identical across independent executions
  on the capture host. If these hash checks ever fail while the elementwise
  1e-12 checks pass, the cause is an environment change (BLAS/torch upgrade
  reordering reductions), not a code regression -- re-baseline deliberately
  via ``uv run python -m tests.regression.capture_golden --force``.
* Scalars, iteration counts and convergence flags are compared exactly.
* **Signature digests are bit-exact sentinels too, and the contract above did
  not say so.** A signature tree contains `hash_array` of the system matrices,
  and those matrices are produced by
  `workbench.setup.generate_marginally_stable_system`, whose final step is
  ``A /= np.max(np.abs(np.linalg.eigvals(A)))`` -- a LAPACK call. So a
  `signature_digest` inherits exactly the host-dependence the bullet above
  grants to content hashes, and the two rules contradicted each other. Digests
  are therefore gated on the same condition as hashes.

**How the capture host is detected.** Not by an environment variable or a
recorded fingerprint, but from the goldens themselves: every persisted array is
compared to its fresh regeneration bit-for-bit, and if any differs -- even
within the 1e-12 elementwise tolerance -- this host does not reproduce the
capture host's last bits, so its bit-exact sentinels carry no information here.
The elementwise checks, which are the actual regression detector, always run.
Off the capture host the sentinel test *skips with a stated reason*, so the
weakening is visible in the summary rather than silent.

Scenario execution is session-scoped: each channel runs once per pytest
session, then every test interrogates its slice of the outputs.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from safetensors.torch import load_file as load_safetensors

from tests.regression import scenarios

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@dataclass(frozen=True)
class Tolerance:
    """How closely two runs of one channel are required to agree."""

    atol: float
    rtol: float


#: The S0 comparison contract, keyed by the working precision a channel
#: declares. Not by the dtype of the array in hand: `J_history` is stored
#: float64 while holding float32 quantities, so a per-array rule would hold it
#: to a precision its computation never had.
TOLERANCES: dict[str, Tolerance] = {
    # A float64 pipeline agrees across machines to far inside this.
    "float64": Tolerance(atol=1e-12, rtol=1e-12),
    # float32 carries ~1.2e-7 of relative resolution, so two hosts whose BLAS
    # vectorises or contracts differently disagree in the last bits by
    # construction, and no baseline can be captured that satisfies both. The
    # worst disagreement observed between CI runners was 3.2e-7 relative;
    # 1e-5 leaves some thirty times that margin while staying two orders of
    # magnitude tighter than any real regression here, which moves these
    # values in their third significant digit.
    "float32": Tolerance(atol=1e-7, rtol=1e-5),
}


def tolerance_for(params: Mapping[str, Any]) -> Tolerance:
    """The tolerance a channel's declared working precision earns.

    A channel that names no dtype is a NumPy float64 pipeline, which is what
    `STANDARD_LQR_PARAMS` is.
    """
    return TOLERANCES[str(params.get("dtype", "float64"))]


#: Key under which the measurement table records the tolerances its ratios were
#: computed against, so editing `TOLERANCES` without re-running the tool cannot
#: leave a table describing a contract nobody asserts any more.
TOLERANCE_RECORD_KEY = "_tolerances"

#: Per frozen array, the worst ``|delta| / (atol + rtol*|golden|)`` observed
#: when EVERY entry of the scenario's process noise moves one ULP in
#: independently-drawn directions -- which is what two differently-built BLAS
#: libraries do to each other. One is exactly the contract; above one the array
#: cannot meet it. Regenerate with
#: ``uv run python tests/regression/measure_ulp_amplification.py --write``.
#:
#: A declared precision describes the *arithmetic*; it says nothing about
#: whether the computation amplifies a last-bit difference. The `signal_space
#: _gd` solve is float32, runs 100 gradient steps and deliberately does not
#: converge, so its per-member iterates are points part-way along a trajectory:
#: perturb the input in its last bit and you land elsewhere along it. Both facts
#: are needed, and keying the tolerance on the first alone is what made an
#: unconverged iterate look like an ordinary float32 quantity.
_MEASUREMENT: dict[str, dict[str, float]] = json.loads(
    (FIXTURES_DIR / "ulp_movement.json").read_text()
)
# `.get`, not `[...]`: the tool that writes this file imports this module, so a
# hard read here would make the table impossible to regenerate from scratch.
# Its presence and correctness are asserted by
# `test_the_measurement_was_taken_under_todays_tolerances` instead.
MEASURED_TOLERANCES: dict[str, float] = _MEASUREMENT.get(TOLERANCE_RECORD_KEY, {})
ULP_MOVEMENT: dict[str, dict[str, float]] = {
    channel: ratios
    for channel, ratios in _MEASUREMENT.items()
    if channel != TOLERANCE_RECORD_KEY
}

#: The probe samples three sign patterns out of two-to-the-409,600, so an
#: unluckier one is entirely possible; a factor of two covers that while leaving
#: every quantity REFACTOR_PLAN v3 section 7.1 actually specified comfortably
#: inside its contract (`J_history` at 0.019, the cost curves at 6e-4, the
#: Riccati stacks at 0).
#:
#: An earlier version of the probe moved a *single* entry and calibrated this at
#: ten. That understated the phenomenon by a factor of 46 -- two hosts disagree
#: in every entry at once -- and CI went red on the next pull request with
#: `cold_jacobi__X_final_meansq` at 2.5e-5. The measurement, not the margin, was
#: what needed fixing.
CROSS_HOST_MARGIN = 2.0


def unreproducible_arrays(channel: str) -> frozenset[str]:
    """Arrays this channel cannot promise, derived from measurement.

    An array that a one-ULP input perturbation already pushes past the
    channel's own contract cannot be held to that contract, and widening the
    tolerance to accommodate it is worse than not asserting it:
    `cold_jacobi__X_final_head` overshoots by a factor of 1,257, so any
    tolerance that admits it could not fail on anything short of total
    breakage. Such arrays stay in the fixture and keep their dtype, shape and
    finiteness checks; only the elementwise value comparison is dropped.

    Full rationale and the obligations this defers:
    `docs/methods/cross_machine_numerical_reproducibility.md`.

    Args:
        channel: The scenario name, keying `ULP_MOVEMENT`.

    Returns:
        The names whose measured violation ratio leaves less than
        `CROSS_HOST_MARGIN` of room inside the contract.
    """
    return frozenset(
        name
        for name, ratio in ULP_MOVEMENT[channel].items()
        if ratio * CROSS_HOST_MARGIN > 1.0
    )


def _require(path: Path) -> Path:
    if not path.exists():
        pytest.fail(
            f"Golden fixture missing: {path}. Generate the S0 baselines with "
            "`uv run python -m tests.regression.capture_golden` (on a tree "
            "whose numerics are known-good)."
        )
    return path


@pytest.fixture(scope="session")
def manifest() -> dict:
    return json.loads(_require(FIXTURES_DIR / "golden_manifest.json").read_text())


@pytest.fixture(scope="session")
def standard_lqr_golden() -> dict[str, np.ndarray]:
    with np.load(
        _require(FIXTURES_DIR / "standard_lqr_golden.npz"), allow_pickle=False
    ) as data:
        return dict(data)


@pytest.fixture(scope="session")
def gd_golden() -> dict[str, np.ndarray]:
    with np.load(
        _require(FIXTURES_DIR / "signal_space_gd_golden.npz"), allow_pickle=False
    ) as data:
        return dict(data)


@pytest.fixture(scope="session")
def gd_golden_tensors() -> dict[str, torch.Tensor]:
    return load_safetensors(
        _require(FIXTURES_DIR / "signal_space_gd_golden.safetensors")
    )


@pytest.fixture(scope="session")
def gd_3d_golden() -> dict[str, np.ndarray]:
    with np.load(
        _require(FIXTURES_DIR / "signal_space_gd_3d_golden.npz"), allow_pickle=False
    ) as data:
        return dict(data)


@pytest.fixture(scope="session")
def standard_lqr_fresh() -> tuple[dict[str, np.ndarray], dict]:
    return scenarios.run_standard_lqr_scenario()


@pytest.fixture(scope="session")
def gd_fresh() -> tuple[dict[str, np.ndarray], dict[str, torch.Tensor], dict]:
    return scenarios.run_signal_space_gd_scenario()


@pytest.fixture(scope="session")
def gd_3d_fresh() -> tuple[dict[str, np.ndarray], dict]:
    return scenarios.run_signal_space_gd_3d_scenario()


def _assert_array_sets_match(
    fresh: dict[str, np.ndarray],
    golden: dict[str, np.ndarray],
    *,
    tolerance: Tolerance,
    unreproducible: frozenset[str],
) -> None:
    """Same key set, then elementwise identity at the S0 tolerance contract
    (dtype must match exactly: a silent float64->float32 downgrade is a
    regression even when values agree).

    Args:
        fresh: Freshly computed arrays.
        golden: The frozen baseline.
        tolerance: The channel's contract. Deliberately has no default, so a
            channel that forgets to state its precision fails everywhere rather
            than silently inheriting float64's, which is the stricter value and
            would pass on whichever host happened to match.
        unreproducible: Names whose measured one-ULP movement exceeds what
            `tolerance` allows (`unreproducible_arrays`). Their dtype, shape
            and finiteness are still checked; only the value comparison is
            dropped, because no tolerance for them is both safe and capable of
            failing. Also has no default: a channel that forgets to state its
            unreproducible set would silently assert values it cannot promise,
            which is exactly the defect this parameter exists to close.
    """
    assert set(fresh) == set(golden), (
        f"Golden key set drifted. missing={sorted(set(golden) - set(fresh))} "
        f"unexpected={sorted(set(fresh) - set(golden))}"
    )
    for name in sorted(golden):
        assert fresh[name].dtype == golden[name].dtype, (
            f"{name}: dtype changed {golden[name].dtype} -> {fresh[name].dtype}"
        )
        assert fresh[name].shape == golden[name].shape, (
            f"{name}: shape changed {golden[name].shape} -> {fresh[name].shape}"
        )
        if name in unreproducible:
            # Structure only. The value is a point on an unconverged trajectory
            # and is not determined by the specification across machines; see
            # `unreproducible_arrays`.
            assert np.all(np.isfinite(fresh[name])), f"{name}: non-finite entry"
            continue
        np.testing.assert_allclose(
            fresh[name],
            golden[name],
            atol=tolerance.atol,
            rtol=tolerance.rtol,
            err_msg=f"Golden-master mismatch in array '{name}'",
            strict=True,
        )


def _assert_tensor_sets_match(
    fresh: dict[str, torch.Tensor],
    golden: dict[str, torch.Tensor],
    *,
    tolerance: Tolerance,
) -> None:
    """The torch counterpart of `_assert_array_sets_match`.

    Extracted rather than written inline because it is the third of three
    comparison paths, and the capture host cannot tell whether it received the
    right tolerance: torch reproduces bit-exactly there, so an inline path left
    on the old 1e-12 would pass locally and fail on every other machine. A
    shared helper with a defaulted-nothing `tolerance` makes that omission a
    `TypeError` instead of a CI surprise, and
    `test_every_comparison_call_states_a_tolerance` checks it structurally.
    """
    assert set(fresh) == set(golden)
    for name in sorted(golden):
        expected, actual = golden[name], fresh[name]
        assert actual.dtype == expected.dtype, f"{name}: torch dtype changed"
        assert actual.shape == expected.shape, f"{name}: torch shape changed"
        assert torch.allclose(
            actual, expected, atol=tolerance.atol, rtol=tolerance.rtol
        ), f"Golden-master mismatch in torch tensor '{name}'"


#: Scalars whose value is a bit-for-bit fingerprint of a float array rather
#: than a quantity. Meaningful only where the arrays themselves reproduce
#: bit-identically, i.e. on the capture host.
BIT_EXACT_SUFFIXES = ("_hash", "_digest")


def _is_capture_host(fresh_arrays: dict, golden_arrays: dict) -> bool:
    """Whether this host reproduces the goldens' last bits, not merely their value.

    Derived from the frozen arrays, so it needs no environment variable and no
    extra fixture field: if a regenerated array differs from its golden at all,
    every hash and digest taken over that computation will differ too, and
    asserting them here would report an environment difference as a regression.
    """
    shared = set(fresh_arrays) & set(golden_arrays)
    return all(
        fresh_arrays[name].dtype == golden_arrays[name].dtype
        and np.array_equal(fresh_arrays[name], golden_arrays[name])
        for name in shared
    )


def _assert_scalars_match(
    fresh: dict, golden: dict, *, bit_exact: bool, tolerance: Tolerance
) -> None:
    """Flags/counts exactly; floats at the S0 tolerance contract.

    Args:
        fresh: Freshly computed scalars.
        golden: The frozen baseline.
        bit_exact: Whether this host reproduces the capture host bit-for-bit.
            Deliberately has no default: a channel that forgets to pass it fails
            immediately and everywhere, rather than defaulting to exact
            comparison and passing on the single host where that is true.
            (`_is_capture_host`). When False, hash and digest sentinels are not
            compared -- they would be testing the CPU, not the code.
        tolerance: The channel's contract, as for `_assert_array_sets_match`.
    """
    assert set(fresh) == set(golden)
    for name in sorted(golden):
        if not bit_exact and name.endswith(BIT_EXACT_SUFFIXES):
            continue
        fresh_value, golden_value = fresh[name], golden[name]
        if isinstance(golden_value, float):
            np.testing.assert_allclose(
                fresh_value,
                golden_value,
                atol=tolerance.atol,
                rtol=tolerance.rtol,
                err_msg=f"Golden-master mismatch in scalar '{name}'",
            )
        else:
            assert fresh_value == golden_value, (
                f"Golden-master mismatch in '{name}': "
                f"{golden_value!r} -> {fresh_value!r}"
            )


class TestStandardLQRChannel:
    """Notebook 01's loop: Riccati recursion, seeded NumPy Monte-Carlo
    rollout, and all four cost-convention curves (empirical + closed-form)."""

    def test_arrays_reproduce_golden(self, standard_lqr_fresh, standard_lqr_golden):
        fresh_arrays, _ = standard_lqr_fresh
        _assert_array_sets_match(
            fresh_arrays,
            standard_lqr_golden,
            tolerance=tolerance_for(scenarios.STANDARD_LQR_PARAMS),
            unreproducible=unreproducible_arrays("standard_lqr"),
        )

    def test_scalars_and_digests_reproduce_golden(
        self, standard_lqr_fresh, standard_lqr_golden, manifest
    ):
        fresh_arrays, fresh_scalars = standard_lqr_fresh
        bit_exact = _is_capture_host(fresh_arrays, standard_lqr_golden)
        _assert_scalars_match(
            fresh_scalars,
            manifest["scenarios"]["standard_lqr"]["scalars"],
            bit_exact=bit_exact,
            tolerance=tolerance_for(scenarios.STANDARD_LQR_PARAMS),
        )
        if not bit_exact:
            pytest.skip(
                "portable scalars verified; bit-exact hash/digest sentinels are "
                "inactive because this host does not reproduce the capture host's "
                "last bits (see the module docstring). The elementwise 1e-12 "
                "checks, which are the regression detector, did run."
            )

    def test_riccati_shapes_lock_notebook_dimensions(self, standard_lqr_golden):
        p = scenarios.STANDARD_LQR_PARAMS
        n, m, horizon = p["state_dim"], p["control_dim"], p["horizon"]
        assert standard_lqr_golden["P_arr"].shape == (horizon + 1, n, n)
        assert standard_lqr_golden["K_arr"].shape == (horizon, m, n)

    def test_frozen_params_match_notebook_01(self, manifest):
        assert (
            manifest["scenarios"]["standard_lqr"]["params"]
            == scenarios.STANDARD_LQR_PARAMS
        )


class TestSignalSpaceGDChannel:
    """Notebook 02's primary (m=2) loop: Riccati baseline on the fixed torch
    batch, gradient-coefficient stacks, and one solve per initializer variant
    and per sweep topology (§7.1: one J_history per topology)."""

    def test_numpy_arrays_reproduce_golden(self, gd_fresh, gd_golden):
        fresh_arrays, _, _ = gd_fresh
        _assert_array_sets_match(
            fresh_arrays,
            gd_golden,
            tolerance=tolerance_for(scenarios.SIGNAL_SPACE_GD_PARAMS),
            unreproducible=unreproducible_arrays("signal_space_gd"),
        )

    def test_torch_tensors_reproduce_golden(self, gd_fresh, gd_golden_tensors):
        _, fresh_tensors, _ = gd_fresh
        _assert_tensor_sets_match(
            fresh_tensors,
            gd_golden_tensors,
            tolerance=tolerance_for(scenarios.SIGNAL_SPACE_GD_PARAMS),
        )

    def test_scalars_and_digests_reproduce_golden(self, gd_fresh, gd_golden, manifest):
        fresh_arrays, _, fresh_scalars = gd_fresh
        bit_exact = _is_capture_host(fresh_arrays, gd_golden)
        _assert_scalars_match(
            fresh_scalars,
            manifest["scenarios"]["signal_space_gd"]["scalars"],
            bit_exact=bit_exact,
            tolerance=tolerance_for(scenarios.SIGNAL_SPACE_GD_PARAMS),
        )
        if not bit_exact:
            pytest.skip(
                "portable scalars verified; bit-exact hash/digest sentinels are "
                "inactive because this host does not reproduce the capture host's "
                "last bits (see the module docstring). The elementwise 1e-12 "
                "checks, which are the regression detector, did run."
            )

    def test_every_variant_has_full_history_golden(self, gd_golden, manifest):
        """One J_history per initializer/topology variant, index-aligned with
        max_iters (no early stop was frozen: tolerance=None in the notebook)."""
        expected_len = scenarios.SIGNAL_SPACE_GD_PARAMS["max_iters"] + 1
        frozen_scalars = manifest["scenarios"]["signal_space_gd"]["scalars"]
        for variant in scenarios.GD_VARIANTS:
            assert gd_golden[f"{variant}__J_history"].shape == (expected_len,)
            assert frozen_scalars[f"{variant}__converged"] is False
            assert (
                frozen_scalars[f"{variant}__iterations_run"]
                == scenarios.SIGNAL_SPACE_GD_PARAMS["max_iters"]
            )

    def test_gd_descends_toward_riccati_optimum(self, gd_golden, manifest):
        """Structural sanity on the frozen baselines themselves: every
        variant's cost history is a descent whose final value sits near the
        Riccati optimum (the notebook's own 1e-3 relative acceptance)."""
        J_opt = manifest["scenarios"]["signal_space_gd"]["scalars"]["J_opt"]
        for variant in scenarios.GD_VARIANTS:
            J_history = gd_golden[f"{variant}__J_history"]
            assert J_history[-1] < J_history[0]
            assert abs(J_history[-1] - J_opt) / abs(J_opt) < 1e-3

    def test_frozen_params_match_notebook_02(self, manifest):
        assert (
            manifest["scenarios"]["signal_space_gd"]["params"]
            == scenarios.SIGNAL_SPACE_GD_PARAMS
        )


class TestSignalSpaceGD3DChannel:
    """Notebook 02's secondary (n=3, m=3, T=18) instance: Gauss-Seidel from a
    cold start, ensembles small enough to be frozen verbatim."""

    def test_arrays_reproduce_golden(self, gd_3d_fresh, gd_3d_golden):
        fresh_arrays, _ = gd_3d_fresh
        _assert_array_sets_match(
            fresh_arrays,
            gd_3d_golden,
            tolerance=tolerance_for(scenarios.SIGNAL_SPACE_GD_3D_PARAMS),
            unreproducible=unreproducible_arrays("signal_space_gd_3d"),
        )

    def test_scalars_and_digests_reproduce_golden(
        self, gd_3d_fresh, gd_3d_golden, manifest
    ):
        fresh_arrays, fresh_scalars = gd_3d_fresh
        bit_exact = _is_capture_host(fresh_arrays, gd_3d_golden)
        _assert_scalars_match(
            fresh_scalars,
            manifest["scenarios"]["signal_space_gd_3d"]["scalars"],
            bit_exact=bit_exact,
            tolerance=tolerance_for(scenarios.SIGNAL_SPACE_GD_3D_PARAMS),
        )
        if not bit_exact:
            pytest.skip(
                "portable scalars verified; bit-exact hash/digest sentinels are "
                "inactive because this host does not reproduce the capture host's "
                "last bits (see the module docstring). The elementwise 1e-12 "
                "checks, which are the regression detector, did run."
            )

    def test_frozen_params_match_notebook_02(self, manifest):
        assert (
            manifest["scenarios"]["signal_space_gd_3d"]["params"]
            == scenarios.SIGNAL_SPACE_GD_3D_PARAMS
        )


class TestToleranceContract:
    """The tolerances themselves, which are a judgement and therefore testable.

    Loosening a golden-master tolerance is the standard way to make a red suite
    green while destroying what it detected, so the separation the float32
    contract relies on is asserted rather than asserted-in-a-comment: it must
    sit **above** the last-bit disagreement two CPUs owe each other, and well
    **below** any change this code could make on purpose.
    """

    #: Worst relative disagreement observed between CI runners on the float32
    #: channels, from the run that exposed the problem.
    HARDWARE_NOISE = 3.2e-7

    #: A deliberately subtle regression: one part in ten thousand.
    SUBTLE_REGRESSION = 1e-4

    @staticmethod
    def _rejects(tolerance: Tolerance, relative: float) -> bool:
        golden = np.linspace(0.3, 0.9, 101)
        try:
            np.testing.assert_allclose(
                golden * (1 + relative),
                golden,
                atol=tolerance.atol,
                rtol=tolerance.rtol,
            )
        except AssertionError:
            return True
        return False

    def test_the_float64_contract_is_unchanged(self) -> None:
        """The precision that genuinely reproduces keeps the strict contract;
        this fix must not have relaxed it in passing."""
        assert TOLERANCES["float64"] == Tolerance(atol=1e-12, rtol=1e-12)

    def test_float32_accepts_cross_machine_last_bit_disagreement(self) -> None:
        assert not self._rejects(TOLERANCES["float32"], self.HARDWARE_NOISE)

    def test_float32_still_rejects_a_subtle_regression(self) -> None:
        """The half that matters. A contract that accepted this would be a
        green light rather than a test."""
        assert self._rejects(TOLERANCES["float32"], self.SUBTLE_REGRESSION)

    def test_the_two_thresholds_are_separated_by_orders_of_magnitude(self) -> None:
        """Noise and signal must not merely be ordered, but far apart: a margin
        of a few percent would make the suite flaky again on the next runner."""
        assert self.SUBTLE_REGRESSION / self.HARDWARE_NOISE > 100

    @pytest.mark.parametrize(
        ("channel", "params"),
        [
            ("standard_lqr", scenarios.STANDARD_LQR_PARAMS),
            ("signal_space_gd", scenarios.SIGNAL_SPACE_GD_PARAMS),
            ("signal_space_gd_3d", scenarios.SIGNAL_SPACE_GD_3D_PARAMS),
        ],
    )
    def test_every_channel_resolves_to_a_declared_contract(
        self, channel: str, params: dict
    ) -> None:
        """Every channel, not one: a channel whose dtype key was misspelled
        would otherwise fall through to float64 and fail on the next runner
        exactly as before."""
        assert tolerance_for(params) in TOLERANCES.values(), channel

    @pytest.mark.parametrize(
        ("channel", "runner_key"),
        [
            ("standard_lqr", "standard_lqr_golden"),
            ("signal_space_gd", "gd_golden"),
            ("signal_space_gd_3d", "gd_3d_golden"),
        ],
    )
    def test_every_frozen_array_has_been_measured(
        self, channel: str, runner_key: str, request: pytest.FixtureRequest
    ) -> None:
        """The structural control on the exclusion mechanism.

        `unreproducible_arrays` can only exclude what `ULP_MOVEMENT` mentions,
        so an array added to a channel *after* the measurement would be
        compared elementwise at a tolerance nobody checked it can meet -- which
        is precisely how the present situation arose. Asserting the table's key
        set *is* the channel's key set makes that omission impossible rather
        than merely discouraged: adding a golden array without re-running
        `measure_ulp_amplification.py --write` fails here.
        """
        golden = request.getfixturevalue(runner_key)
        measured = set(ULP_MOVEMENT[channel])
        assert measured == set(golden), (
            f"{channel}: the ULP-movement table and the frozen arrays disagree. "
            f"unmeasured={sorted(set(golden) - measured)} "
            f"stale={sorted(measured - set(golden))}. Re-run "
            "`uv run python tests/regression/measure_ulp_amplification.py --write`"
        )

    #: Restated here, deliberately NOT imported from `CROSS_HOST_MARGIN`. This
    #: is the property the containment must deliver; the implementation's
    #: constant is how it delivers it. Reading the constant back would make the
    #: assertion tautological -- it would agree with any margin, including one
    #: that lets the original defect back in.
    REQUIRED_HEADROOM = 2.0

    @pytest.mark.parametrize(
        "channel", ["standard_lqr", "signal_space_gd", "signal_space_gd_3d"]
    )
    def test_every_retained_array_has_headroom(self, channel: str) -> None:
        """Whatever is still compared must have room to be compared.

        This is the containment's actual promise: an array still asserted must
        sit a factor of `REQUIRED_HEADROOM` inside its own contract under a
        whole-array one-ULP perturbation, because the probe samples three sign
        patterns out of astronomically many and an unluckier one is possible.

        Behaviour cannot check this on the capture host -- every array
        reproduces exactly here, so *any* exclusion rule, including excluding
        nothing at all, passes locally and fails elsewhere. The property is
        asserted against the measurement directly instead.
        """
        excluded = unreproducible_arrays(channel)
        cramped = {
            name: ratio
            for name, ratio in ULP_MOVEMENT[channel].items()
            if name not in excluded and ratio * self.REQUIRED_HEADROOM > 1.0
        }
        assert not cramped, (
            f"{channel}: arrays still compared with less than "
            f"{self.REQUIRED_HEADROOM:g}x of room inside their own contract: "
            f"{ {k: f'{v:.2e}' for k, v in cramped.items()} }"
        )

    def test_the_measurement_was_taken_under_todays_tolerances(self) -> None:
        """A violation ratio is only meaningful under the tolerance it was
        computed against, so editing `TOLERANCES` invalidates the whole table.
        The tool records what it used; this asserts it is still current, which
        makes that staleness impossible rather than merely unlikely.
        """
        expected = {
            f"{name}.{field}": getattr(tolerance, field)
            for name, tolerance in TOLERANCES.items()
            for field in ("atol", "rtol")
        }
        assert MEASURED_TOLERANCES == expected, (
            "the measurement was taken under different tolerances than the "
            "suite now asserts; re-run `uv run python "
            "tests/regression/measure_ulp_amplification.py --write`"
        )

    @pytest.mark.parametrize(
        ("channel", "name"),
        [
            # Both arrays that actually reddened CI. Each was reached only
            # because it sorts early: `assert_allclose` raises on the first
            # mismatch, so everything after it was never even compared.
            ("signal_space_gd", "cold_jacobi__U_final_head"),
            ("signal_space_gd", "cold_jacobi__X_final_meansq"),
            # The worst offender, reached by neither failure.
            ("signal_space_gd", "cold_jacobi__X_final_head"),
            ("signal_space_gd", "cold_jacobi__U_history_member0"),
            # The other channel, which would have been next.
            ("signal_space_gd_3d", "X_final"),
        ],
    )
    def test_the_measured_offenders_are_excluded(self, channel: str, name: str) -> None:
        """Named anchors, so the rule cannot quietly stop covering the arrays it
        was built for. Asserted through the measurement rather than against
        pinned numbers, so re-running the probe does not mean editing a list of
        magic constants -- what must hold is that these stay refused.
        """
        assert ULP_MOVEMENT[channel][name] > 0.0
        assert name in unreproducible_arrays(channel)

    def test_structure_is_still_checked_for_excluded_arrays(self) -> None:
        """Excluding the value comparison must not exclude everything.

        An excluded array keeps its dtype, shape and finiteness checks -- and
        those are the *only* assertions left on it, so if they were dropped the
        array would be frozen in name alone. Nothing in the scenarios produces
        a NaN or a shape change today, so this is asserted directly rather than
        waited for.
        """
        excluded = frozenset({"chaotic"})
        golden = {"chaotic": np.ones((2, 3), dtype=np.float32)}
        float32 = TOLERANCES["float32"]

        with pytest.raises(AssertionError, match="non-finite"):
            _assert_array_sets_match(
                {"chaotic": np.full((2, 3), np.nan, dtype=np.float32)},
                golden,
                tolerance=float32,
                unreproducible=excluded,
            )
        with pytest.raises(AssertionError, match="shape changed"):
            _assert_array_sets_match(
                {"chaotic": np.ones((3, 2), dtype=np.float32)},
                golden,
                tolerance=float32,
                unreproducible=excluded,
            )
        with pytest.raises(AssertionError, match="dtype changed"):
            _assert_array_sets_match(
                {"chaotic": np.ones((2, 3), dtype=np.float64)},
                golden,
                tolerance=float32,
                unreproducible=excluded,
            )
        # ... and a merely different *value* is what it is allowed to tolerate.
        _assert_array_sets_match(
            {"chaotic": np.full((2, 3), 99.0, dtype=np.float32)},
            golden,
            tolerance=float32,
            unreproducible=excluded,
        )

    def test_a_float64_channel_excludes_nothing(self) -> None:
        """The classification must discriminate, not blanket-exempt.

        `standard_lqr` is a float64 Riccati pipeline with no iterative solve.
        Its worst measured violation ratio is 6.3e-4 -- some 1,600x of room
        inside its own contract -- so if it ever appears in an exclusion set,
        the rule has stopped measuring conditioning and started excusing
        everything.
        """
        assert unreproducible_arrays("standard_lqr") == frozenset()
        assert max(ULP_MOVEMENT["standard_lqr"].values()) < 1e-2

    @pytest.mark.parametrize("channel", ["signal_space_gd", "signal_space_gd_3d"])
    def test_the_unconverged_gd_channels_retain_their_objective(
        self, channel: str
    ) -> None:
        """What survives has to be the part that carries the science.

        The objective is batch-averaged and well conditioned (ratio 0.019 at
        worst); the per-member iterates are points on an unconverged
        trajectory. If an exclusion set ever swallowed `J_history`, the channel
        would assert nothing about whether the solver still descends.
        """
        excluded = unreproducible_arrays(channel)
        histories = {name for name in ULP_MOVEMENT[channel] if "J_history" in name}
        assert histories, f"{channel} froze no cost history"
        assert histories.isdisjoint(excluded)

    def test_the_retained_arrays_still_reject_a_regression(self) -> None:
        """The detector must survive the containment, and be measured doing it.

        Excluding an array is only defensible if what remains can still fail.
        A subtle regression is injected into every retained array of the worst
        channel -- one part in ten thousand, three orders below the change a
        real defect makes and two above the hardware noise the tolerance
        admits -- and each must be caught.
        """
        tolerance = TOLERANCES["float32"]
        excluded = unreproducible_arrays("signal_space_gd")
        retained = sorted(set(ULP_MOVEMENT["signal_space_gd"]) - excluded)
        assert retained, "nothing retained; the containment excluded the channel"

        undetected = []
        for name in retained:
            golden = {name: np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)}
            fresh = {name: golden[name] * np.float32(1.0001)}
            try:
                _assert_array_sets_match(
                    fresh, golden, tolerance=tolerance, unreproducible=excluded
                )
            except AssertionError:
                continue
            undetected.append(name)
        assert not undetected, (
            f"retained arrays that no longer detect a 1e-4 regression: {undetected}"
        )

    def test_every_comparison_call_states_its_unreproducible_set(self) -> None:
        """The sibling of the tolerance rule, for the same reason.

        A channel that omits `unreproducible` would assert values it cannot
        promise, and would do so invisibly on whichever host happened to
        reproduce them -- the capture-host trap once more. The parameter has no
        default, so omission is already a `TypeError`; this asserts it
        structurally too, because a default could be added back in one edit.
        """
        module = ast.parse(Path(__file__).read_text(encoding="utf-8"))
        calls = [
            node
            for node in ast.walk(module)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_assert_array_sets_match"
        ]
        silent = [
            node.lineno
            for node in calls
            if "unreproducible" not in {kw.arg for kw in node.keywords}
        ]
        assert not silent, f"array comparisons with no stated exclusion set: {silent}"
        assert len(calls) >= 3, f"only {len(calls)} array comparisons found"

    def test_every_comparison_call_states_a_tolerance(self) -> None:
        """Structural, because behaviour cannot check this here.

        The capture host reproduces the goldens bit-for-bit, so a comparison
        left on the old 1e-12 passes on this machine and fails on every other
        one -- the same shape as the defect where a gate was wired into two of
        three channels and the third was only ever exercised on the host where
        it could not fail. Parsing the module removes the host from the
        question: all three comparison paths must name their tolerance.
        """
        module = ast.parse(Path(__file__).read_text(encoding="utf-8"))
        helpers = {
            "_assert_array_sets_match",
            "_assert_tensor_sets_match",
            "_assert_scalars_match",
        }
        calls = [
            node
            for node in ast.walk(module)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in helpers
        ]
        silent = [
            (node.func.id, node.lineno)
            for node in calls
            if isinstance(node.func, ast.Name)
            and "tolerance" not in {kw.arg for kw in node.keywords}
        ]
        assert not silent, f"comparison calls with no stated tolerance: {silent}"
        assert len(calls) >= 7, f"only {len(calls)} comparison calls found"
        assert {
            node.func.id for node in calls if isinstance(node.func, ast.Name)
        } == helpers

    def test_no_comparison_hard_codes_a_tolerance(self) -> None:
        """The other half: a helper may accept a tolerance and then ignore it.

        Checked on the call rather than in the text, so that the `TOLERANCES`
        table -- which is where a numeric literal belongs -- does not read as a
        violation of the rule it defines.
        """
        module = ast.parse(Path(__file__).read_text(encoding="utf-8"))
        literal = []
        for node in ast.walk(module):
            if not isinstance(node, ast.Call):
                continue
            name = ast.unparse(node.func)
            if name not in {"torch.allclose", "np.testing.assert_allclose"}:
                continue
            for keyword in node.keywords:
                if keyword.arg in {"atol", "rtol"} and isinstance(
                    keyword.value, ast.Constant
                ):
                    literal.append((name, keyword.arg, node.lineno))
        assert not literal, (
            f"comparisons hard-code a tolerance instead of taking their "
            f"channel's contract: {literal}"
        )

    def test_a_channel_that_declares_float32_gets_the_float32_contract(self) -> None:
        """Guards the mapping itself. Both signal-space channels declare
        float32, and reading that wrongly is what the original defect was."""
        assert tolerance_for(scenarios.SIGNAL_SPACE_GD_PARAMS) == TOLERANCES["float32"]
        assert (
            tolerance_for(scenarios.SIGNAL_SPACE_GD_3D_PARAMS) == TOLERANCES["float32"]
        )
        assert tolerance_for(scenarios.STANDARD_LQR_PARAMS) == TOLERANCES["float64"]
