"""Acceptance for Figure 4's table (the dispersion plan, Phase 2).

The plan's acceptance is a HAND RECOMPUTATION: the rendered median and
quartiles must equal quantities computed here from the same cells by a
different route, not by calling the code under test. Everything else in this
suite is a refusal that must remain reachable — a table whose cells rest on
different numbers of samples, or on one sample, states an `n` it does not
have.
"""

from __future__ import annotations

from typing import Any

import pytest

from mbl.benchmark.table import (
    TABLE_COLUMNS,
    build_rows,
    compilation_costs,
    extrapolation_routes,
    render_latex,
    render_markdown,
    require_one_measurement,
    summarise,
)
from mbl.spec.errors import SpecificationError

DISPLAYS = {"cocp": "COCP", "cocp_lower_bound": "SDP-frozen policy"}
MIB = 1 << 20


def _offline(
    contender: str, total: float, *, peak_mib: float, epochs: int = 100
) -> dict[str, Any]:
    return {
        "phase": "offline",
        "contender": contender,
        "kind": "trainable",
        "declared_epochs": epochs,
        "bench_epochs": 5,
        "per_epoch_s": [total / epochs] * 4 + [total / epochs + 0.1],
        "per_epoch_median_s": total / epochs,
        "setup_s": 0.5,
        "offline_time_s": total,
        "baseline_rss_bytes": 700 * MIB,
        "peak_rss_bytes": int((700 + peak_mib) * MIB),
    }


def _online(contender: str, per_step: float, *, peak_mib: float) -> dict[str, Any]:
    return {
        "phase": "online",
        "contender": contender,
        "setup_s": 0.6,
        "per_step_s": per_step,
        "per_step_p10_s": per_step * 0.8,
        "per_step_p90_s": per_step * 1.4,
        "per_step_calls": 2000,
        "per_step_batch_size": 1,
        "baseline_rss_bytes": 700 * MIB,
        "peak_rss_bytes": int((700 + peak_mib) * MIB),
    }


#: Four process samples per cell — two palindrome halves over two passes.
PER_STEP = [0.00040, 0.00042, 0.00050, 0.00060]
TOTALS = [1200.0, 1210.0, 1230.0, 1260.0]


@pytest.fixture
def cells() -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = []
    for total, per_step in zip(TOTALS, PER_STEP, strict=True):
        grid.append(_offline("cocp", total, peak_mib=640.0))
        grid.append(_online("cocp", per_step, peak_mib=190.0))
    return grid


class TestTheHandRecomputation:
    def test_the_per_step_cell_equals_quartiles_computed_here(
        self, cells: list[dict[str, Any]]
    ) -> None:
        """Median and quartiles of [0.40, 0.42, 0.50, 0.60] ms, by hand.

        The INCLUSIVE convention — the one `numpy.percentile` uses — puts the
        p-quantile of m sorted points at index p·(m − 1), interpolating
        linearly. For m = 4 that is index 0.75 for Q1, 1.5 for the median and
        2.25 for Q3:

            Q1  = 0.40 + 0.75·(0.42 − 0.40) = 0.415
            med = 0.42 + 0.50·(0.50 − 0.42) = 0.46
            Q3  = 0.50 + 0.25·(0.60 − 0.50) = 0.525

        (Tukey's hinges would give 0.41 and 0.55 instead; the table states
        which convention it prints, and this is it.)
        """
        row = build_rows(cells, ["cocp"], DISPLAYS)[0]
        summary = row.summaries["per_step"]
        assert summary.n == 4
        assert summary.median == pytest.approx(0.46, abs=1e-12)
        assert summary.q1 == pytest.approx(0.415, abs=1e-12)
        assert summary.q3 == pytest.approx(0.525, abs=1e-12)

    def test_memory_is_a_delta_from_the_cell_s_own_baseline(
        self, cells: list[dict[str, Any]]
    ) -> None:
        """700 MiB of interpreter is charged to nobody: 640 and 190 stand."""
        row = build_rows(cells, ["cocp"], DISPLAYS)[0]
        assert row.summaries["offline_rss"].median == pytest.approx(640.0)
        assert row.summaries["online_rss"].median == pytest.approx(190.0)

    def test_the_within_process_spread_is_kept_apart_from_the_quartiles(
        self, cells: list[dict[str, Any]]
    ) -> None:
        """p10–p90 of 2000 calls is a different quantity from the spread over
        processes, and pooling them would state a dispersion nothing has."""
        row = build_rows(cells, ["cocp"], DISPLAYS)[0]
        # `None` only where the online phase is not claimed; it is here.
        assert row.within_process_p10_ms is not None
        assert row.within_process_p10_ms == pytest.approx(0.46 * 0.8, rel=1e-9)
        assert row.within_process_p90_ms == pytest.approx(0.46 * 1.4, rel=1e-9)
        assert row.within_process_p10_ms < row.summaries["per_step"].q1

    def test_the_display_name_comes_through_the_rename(
        self, cells: list[dict[str, Any]]
    ) -> None:
        rendered = render_markdown(
            build_rows(cells, ["cocp"], {"cocp": "SDP-frozen policy"}),
            [],
            depth=7,
            samples=4,
        )
        assert "SDP-frozen policy" in rendered
        assert "| cocp |" not in rendered


class TestTheExtrapolationRoutes:
    def test_both_routes_are_reported_for_a_trained_family(
        self, cells: list[dict[str, Any]]
    ) -> None:
        """The plan requires both, because their disagreement is a finding."""
        route = extrapolation_routes(cells, ["cocp"], DISPLAYS)[0]
        assert route.total_s == pytest.approx(1220.0)
        # Within process: four equal epochs and one 0.1 s longer. The mean sits
        # 0.02 above the four, so the squared deviations are 4·0.02² + 0.08² =
        # 0.008, the SAMPLE variance is 0.008/4, and the route multiplies its
        # root by the 100 declared epochs.
        assert route.within_process_route_s == pytest.approx(
            (0.008 / 4) ** 0.5 * 100, rel=1e-9
        )
        assert route.across_process_route_s > 0.0

    def test_an_analytic_family_is_omitted_rather_than_faked(self) -> None:
        """A closed-form synthesis is timed whole; it extrapolates nothing."""
        analytic = []
        for total in (0.0038, 0.0041):
            cell = _offline("riccati", total, peak_mib=12.0)
            cell["kind"] = "analytic"
            cell["declared_epochs"] = None
            cell["per_epoch_s"] = []
            analytic.append(cell)
        assert extrapolation_routes(analytic, ["riccati"], {}) == []


class TestTheClosedFormOfflineCost:
    """The trap of 2026-08-13: repeating a synthesis and taking the median is
    right for a numerical solve and wrong for one that compiles. The SDP-frozen
    policy's first synthesis cost 749 ms against a 79.6 ms steady state, and a
    deployment synthesises once."""

    def _analytic(self, cold: float, warm: float) -> dict[str, Any]:
        return {
            "phase": "offline",
            "contender": "sdp",
            "kind": "analytic",
            "declared_epochs": None,
            "per_epoch_s": [],
            "first_synthesis_s": cold,
            "per_synthesis_s": [warm, warm, warm],
            "synthesis_calls": 3,
            "setup_s": warm,
            "offline_time_s": warm,
            "baseline_rss_bytes": 700 * MIB,
            "peak_rss_bytes": 870 * MIB,
        }

    def _cells(self) -> list[dict[str, Any]]:
        grid: list[dict[str, Any]] = []
        for cold, warm in ((0.749, 0.0796), (0.731, 0.0774), (0.790, 0.0828)):
            grid.append(self._analytic(cold, warm))
            grid.append(_online("sdp", 0.00045, peak_mib=600.0))
        return grid

    def test_the_offline_column_charges_the_cold_synthesis(self) -> None:
        row = build_rows(self._cells(), ["sdp"], {})[0]
        assert row.summaries["offline_time"].median == pytest.approx(0.749)

    def test_the_steady_state_is_reported_beside_it_with_the_ratio(self) -> None:
        costs = compilation_costs(self._cells(), ["sdp"], {"sdp": "SDP-frozen policy"})
        assert costs[0].cold.median == pytest.approx(749.0)
        assert costs[0].warm.median == pytest.approx(79.6)
        rendered = render_markdown(
            build_rows(self._cells(), ["sdp"], {}), [], costs, depth=3, samples=3
        )
        assert "9.41x" in rendered
        assert "one-time cost against steady state" in rendered

    def test_a_trained_family_still_uses_its_extrapolated_total(
        self, cells: list[dict[str, Any]]
    ) -> None:
        """One rule, two families: the change must not touch the column that
        was already right."""
        row = build_rows(cells, ["cocp"], DISPLAYS)[0]
        assert row.summaries["offline_time"].median == pytest.approx(1220.0)


class TestTheProvenance:
    """The author's catch, 2026-08-13: the table was being measured at J = 7
    while the paper operates at J = 3, and nothing in a cell said which."""

    def _stamp(self, cell: dict[str, Any], depth: int) -> dict[str, Any]:
        return cell | {
            "study": "studies/icassp/fig1_depth.toml",
            "tier": "publication_b16k",
            "axis_filter": {"contenders.*.config.num_iterations": depth},
        }

    def test_the_depth_is_read_from_the_cells(
        self, cells: list[dict[str, Any]]
    ) -> None:
        stamped = [self._stamp(cell, 3) for cell in cells]
        provenance = require_one_measurement(stamped)
        assert provenance["axis_filter"]["contenders.*.config.num_iterations"] == 3
        assert provenance["tier"] == "publication_b16k"

    def test_two_depths_in_one_directory_are_refused(
        self, cells: list[dict[str, Any]]
    ) -> None:
        """The exact accident this exists for: a re-run at another operating
        point overwrites some files and leaves others behind."""
        mixed = [self._stamp(cells[0], 7), *(self._stamp(c, 3) for c in cells[1:])]
        with pytest.raises(SpecificationError, match="disagree about axis_filter"):
            require_one_measurement(mixed)

    def test_a_cell_without_provenance_is_refused_not_assumed(
        self, cells: list[dict[str, Any]]
    ) -> None:
        """ "Unknown" and "the same" are not the same thing."""
        with pytest.raises(SpecificationError, match="predates the provenance"):
            require_one_measurement([self._stamp(cells[0], 3), cells[1]])


class TestTheRefusals:
    def test_a_single_sample_cell_is_refused(self) -> None:
        with pytest.raises(SpecificationError, match="at least two"):
            summarise([0.004])

    def test_uneven_sample_depth_is_refused_naming_the_cell(
        self, cells: list[dict[str, Any]]
    ) -> None:
        """A pass that half-completed would otherwise print rows resting on
        four samples beside rows resting on three, under one stated `n`."""
        maimed = [*cells, _online("cocp", 0.00041, peak_mib=190.0)]
        with pytest.raises(SpecificationError, match="cocp/online=5"):
            build_rows(maimed, ["cocp"], DISPLAYS)

    def test_a_contender_with_no_cells_is_refused(
        self, cells: list[dict[str, Any]]
    ) -> None:
        with pytest.raises(SpecificationError, match="neural"):
            build_rows(cells, ["cocp", "neural"], DISPLAYS)


class TestTheRendering:
    def test_the_markdown_states_its_estimator_and_its_sample_count(
        self, cells: list[dict[str, Any]]
    ) -> None:
        rows = build_rows(cells, ["cocp"], DISPLAYS)
        rendered = render_markdown(
            rows, extrapolation_routes(cells, ["cocp"], DISPLAYS), depth=7, samples=4
        )
        assert "median (Q1–Q3)" in rendered
        assert "4 independent" in rendered
        assert "J = 7" in rendered
        for column in TABLE_COLUMNS:
            assert column.header in rendered

    def test_the_latex_escapes_the_quartile_dash(
        self, cells: list[dict[str, Any]]
    ) -> None:
        """An en dash inside a tabular is not a LaTeX range, and pdflatex
        would emit it as a byte the paper's font may not carry."""
        rendered = render_latex(
            build_rows(cells, ["cocp"], DISPLAYS), depth=7, samples=4
        )
        assert "–" not in rendered
        assert "0.46 (0.415--0.525)" in rendered
        assert rendered.count("\\\\") == 2  # header plus one row


# --- a table may claim fewer phases than the grid holds ----------------------


def _offline_cell(contender: str, **extra: object) -> dict:
    """One offline cell, the shape the driver writes."""
    cell = {
        "phase": "offline",
        "contender": contender,
        "seed": 0,
        "study": "s.toml",
        "tier": "publication",
        "axis_filter": {},
        "kind": "analytic",
        "peak_rss_bytes": 2**20,
        "baseline_rss_bytes": 0,
        "first_synthesis_s": 0.5,
        "per_synthesis_s": 0.05,
        "synthesis_calls": 10,
    }
    cell.update(extra)
    return cell


def test_a_table_may_declare_offline_only() -> None:
    """Some substrates cannot yield an online number worth printing.

    On the campaign's GPU the batch-1 per-step latency degrades sixfold *during*
    the measurement and recovers on rest, so the driver's palindrome gate
    refuses it — twice, at −61 % and +86 %. The honest table is the one without
    that column, not one holding a number a gate rejected.
    """
    cells = [_offline_cell("a"), _offline_cell("a")]
    rows = build_rows(cells, ["a"], {"a": "A"}, ["offline"])
    assert len(rows) == 1
    assert rows[0].within_process_p10_ms is None
    markdown = render_markdown(rows, depth=3, samples=2, routes=[], costs=[])
    assert "per-step p10–p90" not in markdown, "an unclaimed column was printed"
    assert "online" not in markdown.split("| controller")[1].split("\n")[0]


def test_an_undeclared_phase_is_still_refused() -> None:
    """The point is *declared* absence, not absence.

    A builder that dropped a column whenever its cells were missing would hide
    the difference between a phase nobody asked for and one that was measured
    and thrown away.
    """
    cells = [_offline_cell("a"), _offline_cell("a")]
    with pytest.raises(SpecificationError, match="no 'online' cells"):
        build_rows(cells, ["a"], {"a": "A"})
