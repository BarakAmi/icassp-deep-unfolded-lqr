"""Figure 4's table: the cost grid's cells reduced to median (Q1–Q3).

The bars this table replaces drew one number per cell and said nothing about
how firm it was. Every quantity here carries an empirical dispersion instead,
and it costs no extra computation: each cell is measured in its own fresh
process, twice per palindrome pass, so K passes leave **2K independent
process samples** of every scalar — setup, synthesis, per-step median and both
resident-memory readings alike.

The samples are read from the raw cell JSONs rather than from the driver's
tidy frame on purpose. The frame's rows are *merged* halves: timed fields take
the minimum of the two readings and the memory fields take the forward half
outright, so the reverse half's memory would be lost to a table built from it.
The cells are the measurement; the frame is its summary.

Two dispersions are kept apart and never pooled (they answer different
questions): *within-process* spread — the p10–p90 of 2000 timed calls inside
one process — and *across-process* spread, which is what the quartiles here
report.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from statistics import median, quantiles, stdev
from typing import Any

from ..spec.errors import SpecificationError

__all__ = [
    "Column",
    "Summary",
    "Row",
    "Extrapolation",
    "Compilation",
    "TABLE_COLUMNS",
    "require_one_measurement",
    "summarise",
    "build_rows",
    "extrapolation_routes",
    "compilation_costs",
    "render_markdown",
    "render_latex",
]

#: Scale from the recorded SI unit to the unit a column prints in.
UNIT_SCALES: dict[str, float] = {"s": 1.0, "ms": 1e3, "MiB": 1.0 / (1 << 20)}


@dataclass(frozen=True)
class Summary:
    """One cell's across-process distribution."""

    n: int
    median: float
    q1: float
    q3: float


@dataclass(frozen=True)
class Column:
    """One measured quantity, in the phase that measures it."""

    key: str
    header: str
    phase: str
    unit: str
    read: Callable[[Mapping[str, Any]], float]


def _rss_delta(cell: Mapping[str, Any]) -> float:
    """Peak resident memory ABOVE the process's own baseline.

    The absolute peak folds in the interpreter, torch's import and the
    allocator's arenas — some 700 MiB before a single controller exists — so
    the absolute number compares Python builds, not controllers.
    """
    return float(cell["peak_rss_bytes"]) - float(cell["baseline_rss_bytes"])


def _offline_cost(cell: Mapping[str, Any]) -> float:
    """What producing the artifact costs ONCE, from a cold process.

    For a trained family that is the extrapolated total, whose per-epoch
    median already discards its own first, slower epoch — a rounding error on
    a 500 s total. For a closed-form family it is the **cold** synthesis, not
    the warm median of the repeats: measured 2026-08-13, the SDP-frozen
    policy's first synthesis costs 749 ms against a 79.6 ms steady state, a
    9.4× ratio that is one-time solver compilation. A deployment synthesises
    once and pays it. (The purely numerical families show 1.5–1.8×, so the
    same rule costs them little and keeps one definition for the column.)
    """
    if str(cell["kind"]) == "analytic":
        return float(cell["first_synthesis_s"])
    return float(cell["offline_time_s"])


TABLE_COLUMNS: tuple[Column, ...] = (
    Column("offline_time", "offline time", "offline", "s", _offline_cost),
    Column(
        "online_setup",
        "online setup",
        "online",
        "ms",
        lambda cell: float(cell["setup_s"]),
    ),
    Column(
        "per_step",
        "per step",
        "online",
        "ms",
        lambda cell: float(cell["per_step_s"]),
    ),
    Column("offline_rss", "offline peak RSS", "offline", "MiB", _rss_delta),
    Column("online_rss", "online peak RSS", "online", "MiB", _rss_delta),
)


@dataclass(frozen=True)
class Row:
    """One contender's line: its display name and one summary per column."""

    label: str
    display: str
    summaries: dict[str, Summary]
    # `None` where the online phase was not measured. NOT a formatting
    # convenience: on a substrate where a per-step latency cannot be measured
    # to the driver's own tolerance, a blank is the honest cell and a number
    # would be whatever the machine was doing.
    within_process_p10_ms: float | None
    within_process_p90_ms: float | None


@dataclass(frozen=True)
class Extrapolation:
    """A trained family's offline total, and the two routes to its spread.

    The total is never timed whole — it is a per-epoch median times a declared
    epoch count — so its dispersion can be reached two ways, and the plan
    (`docs/planning/05_icassp_paper/figure_four_table/dispersion_table.md`
    §4.3) requires both to be reported: their DISAGREEMENT would itself be a
    finding about the extrapolation.
    """

    label: str
    display: str
    total_s: float
    within_process_route_s: float
    across_process_route_s: float


def require_one_measurement(cells: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Every cell must agree about WHAT it measured.

    The cell filenames carry contender, phase, half and pass — never the
    depth, the tier or the document. So a grid re-run at another operating
    point overwrites some files and leaves others, and a table built from the
    directory would pool two experiments into one row with no way to see it.
    Asked for after the author caught the table being measured at J = 7 while
    the paper operates at J = 3 (2026-08-13).

    Returns:
        The common provenance: ``study``, ``tier`` and ``axis_filter``.

    Raises:
        SpecificationError: If the cells disagree, naming the values found; or
            if any cell predates the provenance fields, since "unknown" and
            "the same" are not the same thing.
    """
    seen: dict[str, set[str]] = {"study": set(), "tier": set(), "axis_filter": set()}
    for cell in cells:
        for key in seen:
            if key not in cell:
                raise SpecificationError(
                    f"a cost-grid cell for {cell.get('contender')!r} carries no "
                    f"{key!r}: it predates the provenance fields and cannot be "
                    "shown to measure the same thing as its neighbours. "
                    "Re-measure rather than assume"
                )
            seen[key].add(json.dumps(cell[key], sort_keys=True))
    for key, values in seen.items():
        if len(values) > 1:
            raise SpecificationError(
                f"the cost grid's cells disagree about {key}: "
                f"{sorted(values)}. Two measurements are mixed in one "
                "directory; separate them rather than averaging over them"
            )
    return {key: json.loads(next(iter(values))) for key, values in seen.items()}


def summarise(values: Sequence[float]) -> Summary:
    """Median and quartiles of one cell's process samples.

    Raises:
        SpecificationError: On fewer than two samples — a single reading has
            no dispersion, and printing a bare number in a table whose whole
            claim is dispersion would be the defect this table exists to fix.
    """
    if len(values) < 2:
        raise SpecificationError(
            f"a table cell needs at least two process samples to carry a "
            f"dispersion; got {len(values)}"
        )
    lower, _, upper = quantiles(values, n=4, method="inclusive")
    return Summary(len(values), median(values), lower, upper)


def _grouped(
    cells: Iterable[Mapping[str, Any]],
) -> dict[tuple[str, str], list[Mapping[str, Any]]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for cell in cells:
        key = (str(cell["contender"]), str(cell["phase"]))
        groups.setdefault(key, []).append(cell)
    return groups


def _require_uniform_depth(groups: Mapping[tuple[str, str], list[Any]]) -> int:
    """Every cell must rest on the same number of process samples.

    A table whose rows silently mix ten samples with three states one `n` in
    its caption and means another in half its cells.
    """
    depths = {key: len(value) for key, value in groups.items()}
    distinct = sorted(set(depths.values()))
    if len(distinct) > 1:
        usual = Counter(depths.values()).most_common(1)[0][0]
        odd = sorted(
            f"{contender}/{phase}={count}"
            for (contender, phase), count in depths.items()
            if count != usual
        )
        raise SpecificationError(
            f"the cost grid's cells do not all carry the same number of "
            f"process samples ({distinct}); the odd ones are {odd}. A pass "
            "failed or a cell was measured twice — fix the input rather than "
            "averaging over it"
        )
    return distinct[0]


def build_rows(
    cells: Iterable[Mapping[str, Any]],
    order: Sequence[str],
    displays: Mapping[str, str],
    phases: Sequence[str] = ("offline", "online"),
) -> list[Row]:
    """The table's rows, in the declared contender order.

    Args:
        cells: Every cell JSON, all phases, all halves, all passes.
        order: The contenders to print, in the order to print them.
        displays: Label to display name; a label absent from it prints raw.
        phases: Which phases the table claims to carry. **Declared, not
            inferred**: the absence of a phase is refused unless the caller
            says the table is not making that claim. A table that quietly
            dropped a column whenever its cells were missing would hide the
            difference between "not measured" and "measured and refused".

    Raises:
        SpecificationError: If a contender has no cells for a **declared**
            phase, or if the cells do not all rest on the same number of
            process samples.
    """
    groups = _grouped(cells)
    _require_uniform_depth(groups)
    wanted = tuple(phases)
    rows: list[Row] = []
    for label in order:
        summaries: dict[str, Summary] = {}
        for column in TABLE_COLUMNS:
            if column.phase not in wanted:
                continue
            group = groups.get((label, column.phase))
            if not group:
                raise SpecificationError(
                    f"no {column.phase!r} cells for contender {label!r}; the "
                    "table would print a blank where a measurement was asked "
                    "for"
                )
            scale = UNIT_SCALES[column.unit]
            summaries[column.key] = summarise(
                [column.read(cell) * scale for cell in group]
            )
        online = groups.get((label, "online"), []) if "online" in wanted else []
        rows.append(
            Row(
                label=label,
                display=displays.get(label, label),
                summaries=summaries,
                within_process_p10_ms=None
                if not online
                else median([float(cell["per_step_p10_s"]) * 1e3 for cell in online]),
                within_process_p90_ms=None
                if not online
                else median([float(cell["per_step_p90_s"]) * 1e3 for cell in online]),
            )
        )
    return rows


@dataclass(frozen=True)
class Compilation:
    """A closed-form family's one-time cost, beside its steady state."""

    label: str
    display: str
    cold: Summary
    warm: Summary


def compilation_costs(
    cells: Iterable[Mapping[str, Any]],
    order: Sequence[str],
    displays: Mapping[str, str],
) -> list[Compilation]:
    """Cold synthesis against warm re-synthesis, per closed-form family.

    The ratio is the readable quantity: near 1.5 it is interpreter and cache
    warm-up, and near 10 it is a solver compiling its program — a real cost
    that the table's offline column charges and this section explains.
    """
    groups = _grouped(cells)
    costs: list[Compilation] = []
    for label in order:
        group = [
            cell
            for cell in groups.get((label, "offline"), [])
            if str(cell["kind"]) == "analytic"
        ]
        if not group:
            continue
        costs.append(
            Compilation(
                label=label,
                display=displays.get(label, label),
                cold=summarise([float(c["first_synthesis_s"]) * 1e3 for c in group]),
                warm=summarise([float(c["offline_time_s"]) * 1e3 for c in group]),
            )
        )
    return costs


def extrapolation_routes(
    cells: Iterable[Mapping[str, Any]],
    order: Sequence[str],
    displays: Mapping[str, str],
) -> list[Extrapolation]:
    """The two dispersions of every trained family's extrapolated offline total.

    The within-process route takes each process's own per-epoch spread times
    the declared epoch count; the across-process route takes the spread of the
    totals themselves. Analytic families are omitted: they time a synthesis
    whole and extrapolate nothing.
    """
    groups = _grouped(cells)
    routes: list[Extrapolation] = []
    for label in order:
        group = [
            cell
            for cell in groups.get((label, "offline"), [])
            if str(cell["kind"]) == "trainable"
        ]
        if not group:
            continue
        totals = [float(cell["offline_time_s"]) for cell in group]
        within = [
            stdev([float(value) for value in cell["per_epoch_s"]])
            * float(cell["declared_epochs"])
            for cell in group
        ]
        routes.append(
            Extrapolation(
                label=label,
                display=displays.get(label, label),
                total_s=median(totals),
                within_process_route_s=median(within),
                across_process_route_s=stdev(totals),
            )
        )
    return routes


def _number(value: float) -> str:
    return f"{value:.3g}"


def _cell_text(summary: Summary) -> str:
    return f"{_number(summary.median)} ({_number(summary.q1)}–{_number(summary.q3)})"


def _claimed(rows: Sequence[Row]) -> tuple[Column, ...]:
    """The columns these rows actually carry, in declaration order.

    A table may be declared for a subset of phases -- on a substrate where a
    per-step latency cannot be measured to the driver's own tolerance, the
    honest table is the one without that column rather than one holding a
    number the gate refused. Derived from the rows so the header, the body and
    the LaTeX cannot disagree about what the table contains.

    Args:
        rows: The built rows.

    Returns:
        The subset of `TABLE_COLUMNS` present in every row.
    """
    if not rows:
        return TABLE_COLUMNS
    return tuple(column for column in TABLE_COLUMNS if column.key in rows[0].summaries)


def _within_process_text(row: Row) -> str:
    """The p10–p90 cell, or a blank where the online phase was not claimed.

    Args:
        row: The row to render.

    Returns:
        The formatted range, or ``"--"``.
    """
    low, high = row.within_process_p10_ms, row.within_process_p90_ms
    if low is None or high is None:
        return "--"
    return f"{_number(low)}–{_number(high)}"


def render_markdown(
    rows: Sequence[Row],
    routes: Sequence[Extrapolation],
    costs: Sequence[Compilation] = (),
    *,
    depth: int,
    samples: int,
) -> str:
    """The table as GitHub-flavoured markdown, dispersion stated in the head."""
    claimed = _claimed(rows)
    header = ["controller"] + [f"{column.header} [{column.unit}]" for column in claimed]
    # The within-process spread belongs to the online phase. A column of dashes
    # states nothing the absent columns beside it have not already said.
    spread_claimed = bool(rows) and rows[0].within_process_p10_ms is not None
    if spread_claimed:
        header.append("per-step p10–p90 [ms]")
    lines = [
        f"# Figure 4 — the cost grid at depth J = {depth}",
        "",
        f"Each entry is the **median (Q1–Q3)** of {samples} independent "
        "fresh-process samples: two palindrome halves per pass, "
        f"{samples // 2} passes. Resident memory is the peak **above the "
        "measuring process's own baseline**. The last column is a "
        "*within-process* spread — the p10–p90 of 2000 timed calls — and is "
        "not comparable with the quartiles beside it.",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|",
    ]
    for row in rows:
        # Only the columns the rows actually claim: a table declared
        # offline-only has no online summaries, and indexing for them would
        # turn a declared absence into a KeyError.
        cells = [_cell_text(row.summaries[column.key]) for column in claimed]
        if spread_claimed:
            cells.append(_within_process_text(row))
        lines.append("| " + " | ".join([row.display] + cells) + " |")
    if costs:
        lines += [
            "",
            "## The closed-form families: one-time cost against steady state",
            "",
            "The offline column above charges the **cold** synthesis, which "
            "is what a deployment pays once. A ratio near 1.5 is interpreter "
            "and cache warm-up; a ratio near 10 is a solver compiling its "
            "program, and that cost is real.",
            "",
            "| controller | cold [ms] | warm re-synthesis [ms] | ratio |",
            "|---|---:|---:|---:|",
        ]
        for cost in costs:
            lines.append(
                f"| {cost.display} | {_cell_text(cost.cold)} | "
                f"{_cell_text(cost.warm)} | "
                f"{cost.cold.median / cost.warm.median:.2f}x |"
            )
    if routes:
        lines += [
            "",
            "## The extrapolated offline total, two routes to its spread",
            "",
            "The total is a per-epoch median times a declared epoch count, "
            "never a timed whole. Both routes are reported because their "
            "disagreement would be a finding about the extrapolation.",
            "",
            "| controller | total [s] | within-process route [s] | "
            "across-process route [s] |",
            "|---|---:|---:|---:|",
        ]
        for route in routes:
            lines.append(
                f"| {route.display} | {_number(route.total_s)} | "
                f"{_number(route.within_process_route_s)} | "
                f"{_number(route.across_process_route_s)} |"
            )
    return "\n".join(lines) + "\n"


def render_latex(rows: Sequence[Row], *, depth: int, samples: int) -> str:
    """The same table as a LaTeX tabular, for the paper.

    Display names arrive already in math mode where they carry it, so the
    cells are emitted verbatim; the en dash of a quartile range becomes `--`.
    """
    claimed = _claimed(rows)
    columns = "l" + "r" * len(claimed)
    lines = [
        "% Figure 4's cost grid. Generated by tools/emit_cost_grid_table.py;",
        "% edit the tool, never this file.",
        "\\begin{tabular}{" + columns + "}",
        "\\toprule",
        " & ".join(
            ["controller"]
            + [
                f"{column.header} [{column.unit}]".replace("Δ", "$\\Delta$")
                for column in claimed
            ]
        )
        + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        cells = [
            _cell_text(row.summaries[column.key]).replace("–", "--")
            for column in claimed
        ]
        lines.append(" & ".join([row.display] + cells) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        f"% depth J = {depth}; median (Q1--Q3) over {samples} fresh-process samples.",
    ]
    return "\n".join(lines) + "\n"
