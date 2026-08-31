#!/usr/bin/env python
"""Render the new Figure 2: the three mismatch conditions against severity.

Three documents measure the campaign's three conditions over one shared angle
axis — `fig2_angle_blind`, `fig2_angle_told` and `fig2_angle_world` — and each
stores a plain one-axis analysis. This tool assembles their three tables into
one figure of three panels sharing an x axis, through the **production**
presentation tier: the same series encodings, the same `ieee-2col` profile and
the same four-artifact writer, so the greyscale and Type-3 gates apply exactly
as they do to a study's own figures.

The precedent is `tools/render_cost_grid_figures.py`, and so is the constraint
(the plan's §5.1): **this tool computes nothing.** Every drawn point, every
error bar and every axis position comes from a stored analysis table. A figure
that recomputed even one number would become a second place where results are
made, and the two places would eventually disagree.

Panels run left to right in order of what the controller is given — nothing,
the matrices, the data — because that ladder is the figure's argument.

Usage::

    uv run python tools/render_severity_figure.py --style ieee-2col
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mbl.present.artifacts import write_figure_artifacts  # noqa: E402
from mbl.present.profiles import profile_context, resolve_profile  # noqa: E402
from mbl.present.registry import FigureContext, resolve_figure  # noqa: E402
from mbl.spec.contender import Role  # noqa: E402
from mbl.store.ids import composite_study_id  # noqa: E402

#: The panels, left to right, and the analysis each one reads — named in the
#: campaign's own vocabulary (the author's ruling, 2026-08-14) rather than in
#: the harness's.
#:
#: MATCHED is not a panel: it is theta = 0, where solver and simulator agree in
#: training and in inference, and every panel shares it as its left endpoint.
#:
#: TRAIN-TEST MISMATCH is one setup measured twice. Training is matched; at
#: inference the simulator's system is replaced while every learned parameter
#: stays frozen. *Informed* hands the controller the new system, so solver and
#: simulator agree again and only the frozen parameters are stale.
#: *Uninformed* does not, and it exists because the difference between the two
#: IS the value of the information — a quantity no single column can show.
#:
#: MODEL MISMATCH gives the solver one system and the simulator another, in the
#: offline phase and the online phase alike: the controller precomputes or
#: trains believing its own system throughout, while every trajectory it ever
#: sees comes from a different one.
PANELS: tuple[tuple[str, str], ...] = (
    ("cost_by_angle_blind", "train–test mismatch\n(uninformed)"),
    ("cost_by_angle_told", "train–test mismatch\n(informed)"),
    ("cost_by_angle_world", "model mismatch"),
)

#: Figure 1's declaration order, so a contender is encoded identically in
#: every figure of the paper (§B.3.1 — an encoding derived from what a figure
#: happens to draw repaints the survivors whenever one is dropped).
ORDER: tuple[str, ...] = (
    "truncated_riccati",
    "standard_pgd",
    "unfolded_alpha",
    "unfolded_alpha_p",
    "unfolded_alpha_pj",
    "neural",
    "cocp",
)
DISPLAY = {
    "truncated_riccati": "Clipped-LQR",
    "standard_pgd": "PGD",
    "unfolded_alpha": r"UF-$\alpha$",
    "unfolded_alpha_p": r"UF-$\alpha$P",
    "unfolded_alpha_pj": r"UF-$\alpha$P$^{(j)}$",
    "neural": "GRU",
    "cocp": "COCP",
}
ROLES = {
    "truncated_riccati": Role.BASELINE,
    "standard_pgd": Role.BASELINE,
}


def _source_study(analysis: Path) -> str:
    """The study a stored analysis belongs to, read from where it is filed.

    `store/studies/<study id>/analyses/<name>.parquet` — the id is the
    directory, because that is what the store's own layout means by it.
    """
    return analysis.parent.parent.name


def _find(analysis_id: str, store: Path) -> Path:
    """The one stored table for `analysis_id`, refusing ambiguity.

    Raises:
        SystemExit: If no table exists — the condition was never analysed, and
            an empty panel would say "measured, and flat" — or if several do,
            since silently taking the newest would draw one condition from a
            run the others do not share.
    """
    found = sorted((store / "studies").glob(f"*/analyses/{analysis_id}.parquet"))
    if not found:
        raise SystemExit(
            f"no stored analysis {analysis_id!r} under {store}; run the "
            "condition's document before drawing it, rather than drawing a "
            "panel with nothing in it"
        )
    if len(found) > 1:
        raise SystemExit(
            f"{len(found)} stored analyses named {analysis_id!r}: "
            f"{[str(p) for p in found]}. Two runs of one condition are in the "
            "store and this figure cannot choose between them"
        )
    return found[0]


def _series(
    table: pd.DataFrame, contender: str
) -> tuple[list[float], list[float], list[float]]:
    """One contender's (angle, cost, spread), sorted by angle.

    The angle is READ from the axis label the document declared, never
    recomputed from the plant: the label is what the study says the position
    is, and a second derivation would be a second answer.
    """
    rows = table[table["contender"] == contender]
    points = sorted(
        (
            float(row["axis_label"]),
            float(row["aggregate"]),
            float(row["across_seed_spread"]),
        )
        for _, row in rows.iterrows()
    )
    return ([p[0] for p in points], [p[1] for p in points], [p[2] for p in points])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=REPO / "store")
    parser.add_argument("--style", default="ieee-2col")
    parser.add_argument("--figure-id", default="fig2_mismatch_severity")
    parser.add_argument(
        "--rename",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help=(
            "rename one contender in the series order and its display map, "
            "repeatable. The parallel campaign substitutes the convex policy, "
            "and the order is FIXED on purpose (§B.3.1: an encoding derived "
            "from what a figure happens to draw repaints the survivors when "
            "one is dropped) -- so the substitution is declared here rather "
            "than inferred from the tables"
        ),
    )
    parser.add_argument(
        "--panel-suffix",
        default="",
        help=(
            "suffix on each panel's analysis id, so a parallel campaign's "
            "tables can be assembled by the same tool. `_find` refuses an "
            "ambiguous id, and two campaigns storing one analysis name is "
            "exactly that ambiguity -- which is why the second campaign's "
            "analyses carry their own names rather than this tool guessing"
        ),
    )
    parser.add_argument(
        "--draw-spread",
        action="store_true",
        help="draw across-seed error bars. Off by default: only the recurrent "
        "baseline's is visible at this scale, and the numbers live in the "
        "notebook's table instead",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="directory for the four artifacts (default: beside the store)",
    )
    arguments = parser.parse_args()

    renames = dict(pair.split("=", 1) for pair in arguments.rename)
    unknown = sorted(set(renames) - set(ORDER))
    if unknown:
        parser.error(f"--rename names {unknown}, which are not in the series order")
    order = tuple(renames.get(label, label) for label in ORDER)
    display = {renames.get(k, k): v for k, v in DISPLAY.items()}
    roles = {renames.get(k, k): v for k, v in ROLES.items()}

    panels = tuple(
        (f"{analysis}{arguments.panel_suffix}", title) for analysis, title in PANELS
    )
    sources = {analysis: _find(analysis, arguments.store) for analysis, _ in panels}
    tables = {analysis: pd.read_parquet(path) for analysis, path in sources.items()}
    profile = resolve_profile(arguments.style)

    # A composed figure is filed like everything else: under
    # `store/studies/<id>/figures/`. Its id is derived from the studies it
    # composes, so re-running the same three conditions writes to the same
    # place and changing any one of them moves it -- the same rule every other
    # identity in this store obeys. A bespoke `store/figures/` would have been
    # an unexplained asymmetry, and the author's objection to it (2026-08-14)
    # is exactly that: nothing about this figure justifies a different home.
    composed = sorted(_source_study(path) for path in sources.values())
    composite = composite_study_id(composed, kind=arguments.figure_id)

    rows = pd.concat(
        [table.assign(panel=analysis) for analysis, table in tables.items()],
        ignore_index=True,
    )
    spec = {
        "kind": "severity_panels",
        "config": {
            "panels": [[analysis, title] for analysis, title in panels],
            "xlabel": r"rotation of $A$ [deg]",
            "ylabel": "expected cost",
            "draw_spread": bool(arguments.draw_spread),
        },
        "series_order": list(order),
        "roles": {label: role.value for label, role in roles.items()},
        "display_names": dict(display),
        "composes": composed,
    }

    with profile_context(profile):
        # Drawn through the REGISTRY, which is the same call
        # `figure_from_artifacts` makes when someone reopens the stored spec.
        # Drawing it any other way is how a figure ends up in the store that
        # cannot be rebuilt from it -- found by the author, 2026-08-14.
        figure = resolve_figure(str(spec["kind"]))(
            FigureContext(
                figure_id=arguments.figure_id,
                table=rows,
                config=dict(spec["config"]),
                profile=profile,
                series_order=tuple(order),
                roles=dict(roles),
                display_names=dict(display),
            )
        )
        directory = arguments.out or (
            arguments.store / "studies" / str(composite) / "figures"
        )
        artifacts = write_figure_artifacts(
            figure,
            directory=directory,
            figure_id=arguments.figure_id,
            table=rows,
            spec=spec,
            profile=profile,
        )
    plt.close(figure)
    print(f"-> {artifacts.pdf}")


if __name__ == "__main__":
    main()
