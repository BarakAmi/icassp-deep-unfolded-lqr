"""Render Figure 4's two candidate layouts from the measured cost grid.

The author's instruction (campaign plan Phase F): both layouts are rendered
so the author can choose —

* **F-1** — four separate figures: offline time, online time, offline
  memory, online memory; one bar per controller.
* **F-2** — two figures: time and memory, with offline and online as two
  adjacent bars under each controller — the same shape as Figure 2.

Everything renders through the production `grouped_bars` renderer at the
`ieee-2col` profile and is written by the production artifact writer (the
four-artifact format, PDF fonts embedded TrueType). Quantities:

* time, offline: the extrapolated dedicated-machine training/synthesis cost;
* time, online: the per-step latency at batch 1 — the claim-bearing number
  (§A.5 forbids dividing batched throughput down);
* memory, both phases: peak RSS minus the process's own post-import baseline
  — the phase's working set, not the interpreter's.

Usage::

    uv run python tools/render_cost_grid_figures.py \
        --grid store/benchmarks/fig4_cost_grid/cost_grid.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mbl.analysis.cost_by_category import AXIS_KIND_COLUMN, CATEGORICAL  # noqa: E402
from mbl.present.artifacts import write_figure_artifacts  # noqa: E402
from mbl.present.grouped_bars import grouped_bars  # noqa: E402
from mbl.present.profiles import resolve_profile  # noqa: E402
from mbl.present.registry import FigureContext  # noqa: E402
from mbl.spec.contender import Role  # noqa: E402

#: Figure 1's declaration order and display names, so Figure 4 names its
#: controllers exactly as every other figure in the paper does.
ORDER = (
    "riccati_unconstrained",
    "truncated_riccati",
    "standard_pgd",
    "unfolded_alpha",
    "unfolded_alpha_p",
    "unfolded_alpha_pj",
    "neural",
    "cocp",
    "cocp_lower_bound",
)
DISPLAY = {
    "riccati_unconstrained": "Riccati (unc.)",
    "truncated_riccati": "Trunc-Riccati",
    "standard_pgd": "Standard-PGD",
    "unfolded_alpha": r"UF-$\alpha$",
    "unfolded_alpha_p": r"UF-$\alpha$P",
    "unfolded_alpha_pj": r"UF-$\alpha$P$^{(j)}$",
    "neural": "GRU",
    "cocp": "COCP",
    "cocp_lower_bound": "COCP (SDP-frozen)",
}
MB = 2.0**20


def _cells(grid: pd.DataFrame) -> dict[str, dict[str, float]]:
    """contender -> the four cells, memory as working-set deltas in MB."""
    out: dict[str, dict[str, float]] = {}
    for contender in ORDER:
        offline = grid[(grid.contender == contender) & (grid.phase == "offline")]
        online = grid[(grid.contender == contender) & (grid.phase == "online")]
        if offline.empty or online.empty:
            raise SystemExit(f"grid is missing a phase for {contender!r}")
        off, on = offline.iloc[0], online.iloc[0]
        cells = {
            "time_offline_s": float(off["offline_time_s"]),
            "time_online_s": float(on["per_step_s"]),
            "memory_offline_mb": float(
                (off["peak_rss_bytes"] - off["baseline_rss_bytes"]) / MB
            ),
            "memory_online_mb": float(
                (on["peak_rss_bytes"] - on["baseline_rss_bytes"]) / MB
            ),
        }
        for name, value in cells.items():
            if value <= 0.0:
                raise SystemExit(
                    f"{contender!r} {name} is {value!r}; a log bar axis cannot "
                    "hold it, and a non-positive working set means the cell "
                    "was measured wrong"
                )
        out[contender] = cells
    return out


def _row(contender: str, category: str, position: int, value: float) -> dict:
    return {
        "contender": contender,
        "role": Role.CONTENDER.value,
        "axis_path": "benchmark.phase",
        "axis_value": float(position),
        "axis_label": category,
        AXIS_KIND_COLUMN: CATEGORICAL,
        "aggregate": value,
        "interval_low": value,
        "interval_high": value,
        "across_seed_spread": 0.0,
        "n_seeds": 1,
    }


def _frame(
    cells: dict[str, dict[str, float]], quantities: dict[str, str]
) -> pd.DataFrame:
    rows = []
    for contender in ORDER:
        for position, (key, category) in enumerate(quantities.items()):
            rows.append(_row(contender, category, position, cells[contender][key]))
    return pd.DataFrame(rows)


def _render(name: str, table: pd.DataFrame, config: dict, out: Path) -> None:
    profile = resolve_profile("ieee-2col")
    figure = grouped_bars(
        FigureContext(
            figure_id=name,
            table=table,
            config=config,
            profile=profile,
            series_order=list(ORDER),
            roles={label: Role.CONTENDER for label in ORDER},
            display_names=dict(DISPLAY),
        )
    )
    write_figure_artifacts(
        figure,
        directory=out,
        figure_id=name,
        table=table,
        spec={"figure_id": name, "kind": "grouped_bars", "config": config},
        profile=profile,
    )
    print(f"  {name} -> {out / (name + '.pdf')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid", default="store/benchmarks/fig4_cost_grid/cost_grid.parquet"
    )
    parser.add_argument("--out", default=None, help="defaults beside the grid")
    arguments = parser.parse_args()
    grid_path = Path(arguments.grid)
    out = Path(arguments.out) if arguments.out else grid_path.parent / "figures"
    cells = _cells(pd.read_parquet(grid_path))

    print("F-1: four separate figures")
    for name, key, title, ylabel in (
        ("fig4a_offline_time", "time_offline_s", "Offline compute", "wall time (s)"),
        (
            "fig4b_online_time",
            "time_online_s",
            "Online per-step latency (batch 1)",
            "seconds per control",
        ),
        (
            "fig4c_offline_memory",
            "memory_offline_mb",
            "Offline memory",
            "peak working set (MB)",
        ),
        (
            "fig4d_online_memory",
            "memory_online_mb",
            "Online memory",
            "peak working set (MB)",
        ),
    ):
        _render(
            name,
            _frame(cells, {key: title}),
            {"ylabel": ylabel, "title": title, "yscale": "log"},
            out / "layout_f1",
        )

    print("F-2: two figures, offline and online adjacent")
    _render(
        "fig4_time",
        _frame(
            cells,
            {
                "time_offline_s": "offline (total)",
                "time_online_s": "online (per step, batch 1)",
            },
        ),
        {
            "ylabel": "seconds",
            "title": "Compute cost, offline and online",
            "yscale": "log",
        },
        out / "layout_f2",
    )
    _render(
        "fig4_memory",
        _frame(
            cells,
            {"memory_offline_mb": "offline", "memory_online_mb": "online"},
        ),
        {
            "ylabel": "peak working set (MB)",
            "title": "Memory cost, offline and online",
            "yscale": "log",
        },
        out / "layout_f2",
    )


if __name__ == "__main__":
    main()
