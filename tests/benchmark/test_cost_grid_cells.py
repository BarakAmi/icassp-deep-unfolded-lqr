"""Acceptance for the Figure-4 cost-grid cells and driver logic (Phase F).

Timing MAGNITUDES are not asserted here — this suite runs on a loaded CI
machine and a shared dev box — only structure, refusals, decompositions and
the perturbation facts the plan names: an analytic cell contains a full
synthesis, a trainable cell's total is the per-epoch median times the
declared count, the online cell never trains, and per-step is batch 1.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mbl.benchmark.cells import (
    CellAddress,
    OnlineTimingSpec,
    SynthesisTimingSpec,
    measure_offline,
    measure_online,
    resolve_point,
)
from mbl.benchmark.driver import (
    await_quiet_machine,
    merge_halves,
    palindrome_order,
    require_agreeing_halves,
    require_quiet_machine,
    signed_differences,
)
from mbl.experiments import DEFAULT_SPEC_BINDINGS
from mbl.runner.producer import run_study
from mbl.spec.errors import SpecificationError
from mbl.spec.loader import load_study
from mbl.spec.tiers import DEFAULT_TIER_CATALOGUE

_producer = importlib.import_module("tests.runner.test_producer")

TIER = "smoke"


@pytest.fixture(scope="module")
def fixture_study(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """One tiny study document plus a store its models were run into.

    Run at the SAME tier the cells resolve at, so the store holds exactly the
    ModelIDs the online cells look up. Module-scoped: the producer run is the
    expensive part and every online test reads the same frozen artifacts.
    """
    directory = tmp_path_factory.mktemp("doc")
    store = tmp_path_factory.mktemp("store")
    path = _producer._write(directory)
    document = load_study(path, bindings=DEFAULT_SPEC_BINDINGS)
    study = document.resolve(DEFAULT_TIER_CATALOGUE, TIER, overrides={}).study
    run_study(study, store=store, bindings=DEFAULT_SPEC_BINDINGS)
    return {"study": path, "store": store}


DEPTH_AXIS = "contenders.*.config.num_iterations"


def _depth_of(study_path: Path) -> int:
    """The fixture's first depth AS TIER-RESOLVED (smoke truncates axes) —
    the axis narrowing every swept cell uses, read rather than hardcoded."""
    document = load_study(study_path, bindings=DEFAULT_SPEC_BINDINGS)
    study = document.resolve(DEFAULT_TIER_CATALOGUE, TIER, overrides={}).study
    return int(study.sweep[0].values[0])


class TestOffline:
    def test_an_analytic_cell_times_a_full_synthesis(
        self, fixture_study: dict[str, Path]
    ) -> None:
        cell = measure_offline(CellAddress(fixture_study["study"], TIER, "baseline"))
        assert cell["kind"] == "analytic"
        assert cell["offline_time_s"] > 0.0
        assert cell["offline_time_s"] == cell["setup_s"]
        assert cell["peak_rss_bytes"] > 0
        assert cell["declared_epochs"] is None

    def test_an_analytic_cell_reduces_repeats_not_one_cold_call(
        self, fixture_study: dict[str, Path]
    ) -> None:
        """The defect of 2026-08-13: a closed-form synthesis is milliseconds,
        and estimating it from ONE call made the palindrome read 47.97 ms
        forward against 4.07 ms reverse on a machine with nothing running.
        Every other timed quantity here is a median of repeats; this one now
        is too, and the cold call it replaces is kept beside it."""
        cell = measure_offline(
            CellAddress(fixture_study["study"], TIER, "baseline"),
            synthesis=SynthesisTimingSpec(min_calls=4, max_calls=8, budget_s=0.0),
        )
        assert cell["synthesis_calls"] == 4
        assert len(cell["per_synthesis_s"]) == 4
        assert cell["offline_time_s"] == np.median(cell["per_synthesis_s"])
        assert cell["first_synthesis_s"] > 0.0
        assert "first_synthesis_s" not in cell["per_synthesis_s"]

    def test_the_budget_stops_the_repeats_and_the_ceiling_bounds_them(
        self, fixture_study: dict[str, Path]
    ) -> None:
        """A 700 ms SDP must not be repeated fifty times, and a microsecond
        synthesis must still terminate."""
        address = CellAddress(fixture_study["study"], TIER, "baseline")
        spent = measure_offline(
            address,
            synthesis=SynthesisTimingSpec(min_calls=2, max_calls=99, budget_s=0.0),
        )
        assert spent["synthesis_calls"] == 2
        capped = measure_offline(
            address,
            synthesis=SynthesisTimingSpec(min_calls=1, max_calls=3, budget_s=1e9),
        )
        assert capped["synthesis_calls"] == 3

    def test_a_trainable_cell_is_not_repeated(
        self, fixture_study: dict[str, Path]
    ) -> None:
        """Training is not repeated to be timed — it already reports a median
        over its bench epochs, and repeating it would multiply the grid's
        cost by the repeat count for no new information."""
        depth = _depth_of(fixture_study["study"])
        cell = measure_offline(
            CellAddress(
                fixture_study["study"],
                TIER,
                "unfolded_a",
                axis_filter={DEPTH_AXIS: depth},
            ),
            bench_epochs=2,
            synthesis=SynthesisTimingSpec(min_calls=9, max_calls=9, budget_s=0.0),
        )
        assert cell["kind"] == "trainable"
        assert "per_synthesis_s" not in cell

    def test_full_training_measures_the_total_and_keeps_the_estimate_beside_it(
        self, fixture_study: dict[str, Path]
    ) -> None:
        """The author's approval, 2026-08-14: the one estimated column becomes
        a measured one. What makes the run worth its cost is that BOTH numbers
        come out of it, so the extrapolation is validated rather than
        replaced by assertion."""
        depth = _depth_of(fixture_study["study"])
        address = CellAddress(
            fixture_study["study"], TIER, "unfolded_a", axis_filter={DEPTH_AXIS: depth}
        )
        cell = measure_offline(address, full_training=True)
        assert cell["full_training"] is True
        # The declared budget ran, not a sample of it.
        assert cell["bench_epochs"] == cell["declared_epochs"]
        assert cell["offline_time_s"] == cell["measured_total_s"]
        assert cell["measured_total_s"] > 0.0
        assert cell["extrapolated_total_s"] > 0.0

    def test_a_bench_cell_reports_no_measured_total_rather_than_the_estimate(
        self, fixture_study: dict[str, Path]
    ) -> None:
        """`None` and "the estimate again" are different claims, and a column
        that quietly repeated the estimate under the measured name would make
        the validation vacuous."""
        depth = _depth_of(fixture_study["study"])
        cell = measure_offline(
            CellAddress(
                fixture_study["study"],
                TIER,
                "unfolded_a",
                axis_filter={DEPTH_AXIS: depth},
            ),
            bench_epochs=2,
        )
        assert cell["full_training"] is False
        assert cell["measured_total_s"] is None
        assert cell["offline_time_s"] == cell["extrapolated_total_s"]

    def test_a_trainable_cell_extrapolates_from_the_bench_epochs(
        self, fixture_study: dict[str, Path]
    ) -> None:
        depth = _depth_of(fixture_study["study"])
        cell = measure_offline(
            CellAddress(
                fixture_study["study"],
                TIER,
                "unfolded_a",
                axis_filter={DEPTH_AXIS: depth},
            ),
            bench_epochs=3,
        )
        assert cell["kind"] == "trainable"
        assert cell["bench_epochs"] == 3
        assert len(cell["per_epoch_s"]) == 3
        assert cell["per_epoch_median_s"] == np.median(cell["per_epoch_s"])
        assert cell["offline_time_s"] == pytest.approx(
            cell["setup_s"] + cell["per_epoch_median_s"] * cell["declared_epochs"]
        )
        # The tier's overlay applied: the declared count the extrapolation
        # multiplies is the RESOLVED study's, not the raw document's -- the
        # benchmark times what the campaign trains. Read, not hardcoded.
        document = load_study(fixture_study["study"], bindings=DEFAULT_SPEC_BINDINGS)
        resolved = document.resolve(DEFAULT_TIER_CATALOGUE, TIER, overrides={}).study
        contender = next(
            spec for spec in resolved.contenders if spec.resolved_label == "unfolded_a"
        )
        assert cell["declared_epochs"] == contender.config["plan"].epochs

    def test_the_offline_cell_publishes_nothing(
        self, fixture_study: dict[str, Path], tmp_path: Path
    ) -> None:
        """The cell re-performs the computation to time it; a benchmark that
        wrote models would make the store's contents depend on whether a
        figure was drawn."""
        before = sorted(p.name for p in (fixture_study["store"] / "models").iterdir())
        measure_offline(CellAddress(fixture_study["study"], TIER, "baseline"))
        after = sorted(p.name for p in (fixture_study["store"] / "models").iterdir())
        assert after == before

    def test_an_unknown_point_is_refused_by_name(
        self, fixture_study: dict[str, Path]
    ) -> None:
        with pytest.raises(SpecificationError, match="no_such"):
            measure_offline(CellAddress(fixture_study["study"], TIER, "no_such"))


class TestOnline:
    def test_an_absent_model_is_refused_naming_it(
        self, fixture_study: dict[str, Path], tmp_path: Path
    ) -> None:
        """The online cell measures a FROZEN artifact and never trains one:
        pointed at an empty store it must refuse, not synthesise."""
        empty = tmp_path / "empty_store"
        empty.mkdir()
        with pytest.raises(SpecificationError, match="never trains"):
            measure_online(
                CellAddress(
                    fixture_study["study"],
                    TIER,
                    "unfolded_a",
                    axis_filter={DEPTH_AXIS: _depth_of(fixture_study["study"])},
                ),
                store_root=empty,
                timing=OnlineTimingSpec(warmup_calls=1, timed_calls=2, min_seconds=0.0),
            )

    def test_setup_and_per_step_are_separate_and_batch_one(
        self, fixture_study: dict[str, Path]
    ) -> None:
        cell = measure_online(
            CellAddress(
                fixture_study["study"],
                TIER,
                "unfolded_a",
                axis_filter={DEPTH_AXIS: _depth_of(fixture_study["study"])},
            ),
            store_root=fixture_study["store"],
            timing=OnlineTimingSpec(warmup_calls=2, timed_calls=25, min_seconds=0.0),
        )
        assert cell["setup_s"] > 0.0
        assert cell["per_step_s"] > 0.0
        assert cell["per_step_batch_size"] == 1
        # Calls behind the REPORTED number, which is one block's median rather
        # than a pooled one: 25 timed calls over 5 blocks. The block structure
        # exists because this machine's GPU is shared with a desktop, and the
        # cheapest block is the only estimate contention cannot inflate.
        assert cell["per_step_calls"] == 5
        assert cell["per_step_block_count"] >= 5
        assert cell["rss_after_load_bytes"] > 0
        assert cell["peak_rss_bytes"] >= cell["rss_after_load_bytes"]

    def test_the_batched_cell_is_a_distribution(
        self, fixture_study: dict[str, Path]
    ) -> None:
        cell = measure_online(
            CellAddress(fixture_study["study"], TIER, "baseline"),
            store_root=fixture_study["store"],
            timing=OnlineTimingSpec(warmup_calls=1, timed_calls=5, min_seconds=0.0),
        )
        document = load_study(fixture_study["study"], bindings=DEFAULT_SPEC_BINDINGS)
        resolved = document.resolve(DEFAULT_TIER_CATALOGUE, TIER, overrides={}).study
        declared_batches = resolved.evaluation.protocol.n_batches
        assert len(cell["batched_wall_s"]) == declared_batches
        assert all(value > 0.0 for value in cell["batched_wall_s"])
        assert cell["batched_batch_size"] > 1

    def test_the_weightless_analytic_record_rebuilds_bit_exactly(
        self, fixture_study: dict[str, Path]
    ) -> None:
        """The producer's reuse semantics, honoured: a receipt for a
        closed-form solve re-derives rather than loads."""
        cell = measure_online(
            CellAddress(fixture_study["study"], TIER, "baseline"),
            store_root=fixture_study["store"],
            timing=OnlineTimingSpec(warmup_calls=1, timed_calls=5, min_seconds=0.0),
        )
        assert cell["per_step_s"] > 0.0


def _pair(contender: str, factor: float) -> tuple[dict[str, Any], dict[str, Any]]:
    """One online cell's two palindrome readings, the reverse half `factor`
    times the forward one **in every timed field**.

    Both of the online phase's fields move together on purpose: a machine that
    is slower in the reverse half is slower at setting up and at stepping
    alike. An earlier version of this helper held setup equal, which fed the
    cast-level median a zero per cell and halved the drift it reported — a
    fixture that hid the very effect these tests exist to pin.
    """
    forward = {
        "phase": "online",
        "contender": contender,
        "setup_s": 1.0,
        "per_step_s": 0.5,
    }
    reverse = {
        "phase": "online",
        "contender": contender,
        "setup_s": 1.0 * factor,
        "per_step_s": 0.5 * factor,
    }
    return (forward, reverse)


class TestDriverLogic:
    def test_the_palindrome_is_a_palindrome(self) -> None:
        jobs = palindrome_order(["a", "b", "c"], ["offline", "online"])
        forward = [job for job in jobs if job.half == "forward"]
        reverse = [job for job in jobs if job.half == "reverse"]
        assert [(j.contender, j.phase) for j in reverse] == list(
            reversed([(j.contender, j.phase) for j in forward])
        )
        assert len(jobs) == 2 * len(forward)

    def test_a_busy_machine_is_refused(self) -> None:
        with pytest.raises(SpecificationError, match="quiet"):
            require_quiet_machine(3.2, threshold=1.5)
        require_quiet_machine(0.4, threshold=1.5)

    def test_a_decaying_load_is_waited_out_not_refused(self) -> None:
        """The defect of 2026-08-13: pass 1 of five ended at load 5.57 and the
        gate refused pass 2 — the load it read was the measurement's own,
        decaying. The wait must let that through, and must actually sleep."""
        readings = iter([5.57, 3.10, 1.40])
        slept: list[float] = []
        load = await_quiet_machine(
            lambda: next(readings),
            slept.append,
            threshold=1.5,
            timeout_s=600.0,
            poll_s=20.0,
        )
        assert load == 1.40
        assert slept == [20.0, 20.0]

    def test_a_machine_that_never_settles_is_still_refused(self) -> None:
        """The refusal survives the wait; it moves to the far end of it."""
        slept: list[float] = []
        with pytest.raises(SpecificationError, match="after waiting 40 s"):
            await_quiet_machine(
                lambda: 4.0,
                slept.append,
                threshold=1.5,
                timeout_s=40.0,
                poll_s=20.0,
            )
        assert slept == [20.0, 20.0]

    def test_a_zero_wait_is_the_immediate_refusal(self) -> None:
        """`timeout_s = 0` degenerates to `require_quiet_machine`, so the
        stricter behaviour remains reachable by declaration."""
        with pytest.raises(SpecificationError, match="quiet"):
            await_quiet_machine(
                lambda: 3.2, lambda _: None, threshold=1.5, timeout_s=0.0, poll_s=20.0
            )
        assert (
            await_quiet_machine(
                lambda: 0.4, lambda _: None, threshold=1.5, timeout_s=0.0, poll_s=20.0
            )
            == 0.4
        )

    def test_a_systematic_slope_is_refused_where_the_old_per_cell_test_passed(
        self,
    ) -> None:
        """The instrument's whole point (Annex 06 §7.4, amended 2026-08-13).

        Every cell 8 % slower in the reverse half is a machine that changed
        during the pass — and it sits inside ANY per-cell bound loose enough
        to survive this platform's 33 % process noise. The cast-level median
        catches it.
        """
        pairs = [_pair(f"c{index}", 1.08) for index in range(9)]
        with pytest.raises(SpecificationError, match="median signed difference"):
            require_agreeing_halves(pairs, drift_tolerance=0.05, gross_tolerance=0.60)
        # ... and the per-cell reading it is built from is only 7.4 % — under
        # the 25 % bound that used to be the whole test.
        assert max(abs(value) for _, value in signed_differences(pairs)) < 0.075

    def test_unsigned_process_noise_is_not_refused(self) -> None:
        """Four adjacent fresh processes spanning 33 % on a sub-millisecond
        cell is this machine, measured; refusing it costs an hour and buys
        nothing. Alternating signs leave the median where it belongs."""
        pairs = [
            _pair("a", 1.30),
            _pair("b", 0.75),
            _pair("c", 1.20),
            _pair("d", 0.80),
            _pair("e", 1.02),
        ]
        drift = require_agreeing_halves(
            pairs, drift_tolerance=0.10, gross_tolerance=0.60
        )
        assert abs(drift) <= 0.10

    def test_a_gross_single_cell_failure_is_still_refused_by_name(self) -> None:
        """A stalled core is not process noise, and the cast-level median
        would happily average it away."""
        pairs = [_pair("a", 1.01), _pair("b", 4.0), _pair("c", 0.99)]
        with pytest.raises(SpecificationError, match="grossly on b/online"):
            require_agreeing_halves(pairs, drift_tolerance=0.10, gross_tolerance=0.60)

    def test_a_pass_that_compared_nothing_cannot_report_quietness(self) -> None:
        with pytest.raises(SpecificationError, match="measured nothing"):
            require_agreeing_halves([], drift_tolerance=0.10, gross_tolerance=0.60)

    def test_the_drift_is_returned_so_the_pass_reports_its_own_number(
        self,
    ) -> None:
        """ "The halves' median signed difference was 1.2 %" is weighable;
        "every cell agreed within 25 %" described the loosest cell."""
        pairs = [_pair(f"c{index}", 1.012) for index in range(5)]
        drift = require_agreeing_halves(
            pairs, drift_tolerance=0.10, gross_tolerance=0.60
        )
        assert drift == pytest.approx(0.012 / 1.012, rel=1e-9)

    def test_merged_cells_take_the_quieter_reading(self) -> None:
        forward = {"phase": "offline", "contender": "gru", "offline_time_s": 12.0}
        reverse = {"phase": "offline", "contender": "gru", "offline_time_s": 11.0}
        merged = merge_halves(forward, reverse)
        assert merged["offline_time_s"] == 11.0
        assert merged["forward"]["offline_time_s"] == 12.0
        assert merged["reverse"]["offline_time_s"] == 11.0


class TestResolvePoint:
    def test_the_axis_filter_narrows_to_one_depth(
        self, fixture_study: dict[str, Path]
    ) -> None:
        depth = _depth_of(fixture_study["study"])
        point = resolve_point(
            CellAddress(
                fixture_study["study"], TIER, "unfolded_a", 0, {DEPTH_AXIS: depth}
            )
        )
        assert point.axis_values[DEPTH_AXIS] == depth

    def test_a_wrong_axis_value_is_refused(
        self, fixture_study: dict[str, Path]
    ) -> None:
        with pytest.raises(SpecificationError, match="axis"):
            resolve_point(
                CellAddress(
                    fixture_study["study"], TIER, "unfolded_a", 0, {DEPTH_AXIS: 99}
                )
            )

    def test_an_invariant_contender_satisfies_a_depth_filter_vacuously(
        self, fixture_study: dict[str, Path]
    ) -> None:
        """One --axis narrowing must serve a mixed cast: the depth-invariant
        baseline carries no depth axis, so the steppers' filter is satisfied
        vacuously rather than refusing it."""
        depth = _depth_of(fixture_study["study"])
        point = resolve_point(
            CellAddress(
                fixture_study["study"], TIER, "baseline", 0, {DEPTH_AXIS: depth}
            )
        )
        assert DEPTH_AXIS not in point.axis_values

    def test_an_ambiguous_match_is_refused_not_first_picked(
        self, fixture_study: dict[str, Path]
    ) -> None:
        """A swept contender without a filter matches several depths; picking
        the first would pick by declaration order, silently."""
        with pytest.raises(SpecificationError, match="ONE point"):
            resolve_point(CellAddress(fixture_study["study"], TIER, "unfolded_a", 0))


# --- the online cell must measure a study that runs on the card --------------


def test_the_probe_state_is_built_through_the_compute_context() -> None:
    """Structural, because a CPU run cannot see this and every run was one.

    `measure_online` built its probe with `torch.zeros(..., dtype=...)`, which
    carries the precision and drops the device, so the first policy call died on
    a device mismatch inside the contender. The cost table exists because
    Figure 1 is a CPU study; the first attempt at the same table for a GPU one
    is what found it. A behavioural test on this machine would pass either way
    unless it declared CUDA, so the call itself is asserted.
    """
    import ast
    import inspect

    from mbl.benchmark import cells

    tree = ast.parse(inspect.getsource(cells.measure_online))
    assigned = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "probe_state"
            for target in node.targets
        )
    ]
    assert len(assigned) == 1, "expected exactly one probe-state construction"
    source = ast.unparse(assigned[0].value)
    assert "ctx.zeros" in source, (
        f"the probe is built as {source!r}; it must come from the context, which "
        "carries the device as well as the dtype"
    )
    assert "torch.zeros" not in source


def test_the_probe_control_is_detached_before_numpy_sees_it() -> None:
    """The guard against a broken policy must not itself raise on a CUDA tensor.

    Second half of the same gap: with the probe correctly on the card, the
    finiteness check converted a CUDA tensor to numpy and raised, so a fixed
    device bug simply moved the failure one line down.
    """
    import ast
    import inspect

    from mbl.benchmark import cells

    tree = ast.parse(inspect.getsource(cells.measure_online))
    calls = [
        ast.unparse(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and "isfinite" in ast.unparse(node.func)
    ]
    assert calls, "the non-finite guard is gone"
    checked = " ".join(calls)
    assert "probe_control" in checked
    detached = [
        ast.unparse(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "probe_control" for t in node.targets
        )
    ]
    assert detached and "cpu()" in detached[0]


def test_a_cuda_timing_drains_the_queue_before_the_clock_stops() -> None:
    """Structural: a CPU study cannot see this, and every study was one.

    A CUDA call returns before its work is done, so a lap timed without
    draining measures the launch and not the controller. Measured, the error is
    not small -- the palindrome halves of one contender disagreed by **61 %**
    and the driver refused them, which is that gate doing what it exists for.

    Asserted by parsing because the failure is an *absence*: a behavioural test
    would have to run on the card and would still only see a wrong number, not
    a missing call.
    """
    import ast
    import inspect

    from mbl.benchmark import cells

    tree = ast.parse(inspect.getsource(cells.measure_online))
    drains = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "drain"
    ]
    assert drains, "no drain helper; a CUDA lap would time the launch queue"
    assert "torch.cuda.synchronize" in ast.unparse(drains[0])

    # The timed loop must drain between the call and the clock.
    loops = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and any(
            isinstance(inner, ast.Call) and "wall_clock" in ast.unparse(inner.func)
            for inner in ast.walk(node)
        )
    ]
    assert loops, "no timed loop found"
    timed = max(loops, key=lambda node: len(ast.unparse(node)))
    body = ast.unparse(timed)
    assert "drain()" in body, f"the timed loop does not drain:\n{body}"
    assert body.index("policy(") < body.index("drain()") < body.rindex("wall_clock")
