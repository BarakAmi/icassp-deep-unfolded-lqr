#!/usr/bin/env python
"""Render Figure 4's table from a measured cost grid.

Usage::

    uv run python tools/emit_cost_grid_table.py \
        --grid store/benchmarks/fig4_cost_grid \
        --study studies/icassp/fig1_depth.toml --tier publication_b16k \
        --depth 7

Reads every `*_pass*.json` cell the driver left behind — two palindrome halves
per pass, each its own fresh process — and writes `cost_grid_table.md` and
`cost_grid_table.tex` beside them. Display names come from the study document,
so a contender renamed in the figure is renamed here by the same edit.

Cells without a pass index are the pre-`--repeats` measurement; they are
reported and skipped rather than silently pooled with a later grid's.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mbl.benchmark.table import (  # noqa: E402
    build_rows,
    compilation_costs,
    extrapolation_routes,
    render_latex,
    render_markdown,
    require_one_measurement,
)
from mbl.experiments import DEFAULT_SPEC_BINDINGS  # noqa: E402
from mbl.spec.loader import load_study  # noqa: E402
from mbl.spec.tiers import DEFAULT_TIER_CATALOGUE  # noqa: E402

PASS_CELL = re.compile(r"_pass\d+\.json$")


def _displays(study_path: str, tier: str) -> dict[str, str]:
    """Label to display name, read from the document the grid measured."""
    document = load_study(Path(study_path), bindings=DEFAULT_SPEC_BINDINGS)
    study = document.resolve(DEFAULT_TIER_CATALOGUE, tier, overrides={}).study
    return {
        contender.resolved_label: contender.resolved_display
        for contender in study.contenders
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", required=True, help="the driver's --out directory")
    parser.add_argument("--study", required=True)
    parser.add_argument("--tier", required=True)
    parser.add_argument("--contenders", nargs="+", required=True)
    parser.add_argument(
        "--phases",
        nargs="+",
        default=["offline", "online"],
        choices=["offline", "online"],
        help=(
            "which phases this table CLAIMS to carry. Declared rather than\n"
            "            inferred: a table that silently dropped a column whenever its\n"
            "            cells were missing would hide the difference between a phase\n"
            "            that was not measured and one that was measured and refused"
        ),
    )
    arguments = parser.parse_args()

    grid = Path(arguments.grid)
    everything = sorted((grid / "cells").glob("*.json"))
    kept = [path for path in everything if PASS_CELL.search(path.name)]
    skipped = len(everything) - len(kept)
    if skipped:
        print(
            f"skipped {skipped} cell(s) with no pass index: they predate "
            "--repeats and belong to a different measurement"
        )
    cells = [json.loads(path.read_text()) for path in kept]

    # The depth is READ from the cells, never typed: the run that prompted
    # this measured J = 7 while the paper operates at J = 3, and a --depth
    # flag would have printed whatever I asserted.
    provenance = require_one_measurement(cells)
    depth = int(provenance["axis_filter"]["contenders.*.config.num_iterations"])
    displays = _displays(arguments.study, arguments.tier)
    rows = build_rows(cells, arguments.contenders, displays, arguments.phases)
    routes = extrapolation_routes(cells, arguments.contenders, displays)
    costs = compilation_costs(cells, arguments.contenders, displays)
    # ANY claimed column reports the same sample depth -- `_require_uniform_depth`
    # refuses a grid where they differ -- so read it from one the table
    # actually carries rather than from a phase it may not claim.
    samples = next(iter(rows[0].summaries.values())).n

    markdown = render_markdown(rows, routes, costs, depth=depth, samples=samples)
    latex = render_latex(rows, depth=depth, samples=samples)
    (grid / "cost_grid_table.md").write_text(markdown)
    (grid / "cost_grid_table.tex").write_text(latex)
    print(markdown)
    print(f"-> {grid / 'cost_grid_table.md'} and .tex ({samples} samples per cell)")


if __name__ == "__main__":
    main()
