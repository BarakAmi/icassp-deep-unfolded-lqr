#!/usr/bin/env python
"""The feasibility and saturation audit, read out of the store.

Usage::

    uv run python tools/emit_feasibility_audit.py \
        studies/icassp_exact_convex/fig5_stress_depth.toml \
        --tier publication --depth 10

At a box that binds on nine control entries in ten, ``max|u| - u_max`` stops
being a formality and the saturated fraction stops being decoration. This emits
both, per contender, from Annex 03 §A.3.1a's retained statistic -- **nothing is
re-rolled**, so the numbers are the ones the study's own measurements carry and
not a second scoring of the same models.

Three things the table is required to say, and each can fail:

* every contender that claims to respect the box **does**, to the precision it
  was measured in;
* the unconstrained reference **violates** it, which is what makes it a
  reference and not a competitor;
* the saturated fraction is reported per contender, because saturation is a
  property of a policy's own trajectory and not of the plant -- four contenders
  on one plant have been measured at 89.65 %, 87.62 %, 87.82 % and 94.25 %.

Writes `feasibility_audit.md` beside the study's figures, and prints the same
table.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mbl.replay import load_study  # noqa: E402
from mbl.store.content_store import MeasurementStore  # noqa: E402
from mbl.store.location import default_store  # noqa: E402

#: The metrics §A.3.1a retains. A measurement written before it carries none,
#: and is reported as such rather than skipped -- a row silently missing from
#: an audit reads as a contender that passed it.
MAX_ABS = "eval_max_abs_control"
FRACTION = "eval_saturation_fraction"
TOLERANCE = "eval_saturation_tolerance"

#: Roles whose feasibility is a claim. A `reference` is the unconstrained
#: optimum and is expected to violate; a `bound` attains no trajectory.
FEASIBLE_ROLES = frozenset({"contender", "baseline"})


def _depth_of(point: Any) -> int | None:
    value = next(
        (v for k, v in point.axis_values.items() if k.endswith("num_iterations")),
        None,
    )
    return None if value is None else int(value)


def rows(loaded: Any, measurements: MeasurementStore, depth: int) -> list[dict]:
    """One row per contender: every seed at `depth`, reduced.

    `max_abs` is reduced by the **maximum** across seeds and `saturated` by the
    mean, because the two are different claims: feasibility is about the worst
    trajectory any replicate produced, and activity is about the typical one.
    A single reduction applied to both would report a policy feasible on the
    strength of its quiet seeds.
    """
    gathered: dict[str, dict[str, Any]] = {}
    for point in loaded.study.materialise():
        point_depth = _depth_of(point)
        if point_depth is not None and point_depth != depth:
            continue
        measurement_id = str(point.measurement_id)
        if not measurements.exists(measurement_id):
            continue
        metrics = measurements.metrics(measurement_id)
        label = point.contender.resolved_label
        row = gathered.setdefault(
            label,
            {
                "label": label,
                "display": point.contender.resolved_display,
                "role": point.contender.role.value,
                "depth": point_depth,
                "seeds": 0,
                "max_abs": None,
                "fractions": [],
                "tolerance": metrics.get(TOLERANCE),
                "cost": [],
            },
        )
        row["seeds"] += 1
        row["cost"].append(float(metrics["eval_expected_cost"]))
        if MAX_ABS in metrics:
            value = float(metrics[MAX_ABS])
            row["max_abs"] = (
                value if row["max_abs"] is None else max(row["max_abs"], value)
            )
        if FRACTION in metrics:
            row["fractions"].append(float(metrics[FRACTION]))
    return sorted(gathered.values(), key=lambda r: -(r["max_abs"] or 0.0))


def render(rows_: list[dict], u_max: float, depth: int) -> str:
    """The audit, as the paper would print it."""
    tolerances = {r["tolerance"] for r in rows_ if r["tolerance"] is not None}
    if not tolerances:
        tolerance = "not recorded — no row carries the statistic"
    elif len(tolerances) == 1:
        tolerance = f"{next(iter(tolerances)):.0e}"
    else:
        # Two tolerances in one table is not a formatting problem: measured on
        # a nine-contender cast, the tolerance reordered the field.
        tolerance = "**MIXED, and these rows are not comparable**"
    lines = [
        f"# Feasibility and saturation audit — `u_max = {u_max:g}`, J = {depth}",
        "",
        f"Saturation counted at a relative tolerance of **{tolerance}** "
        "(Annex 01 §4.1). Read out of the study's own stored measurements; "
        "nothing was re-rolled.",
        "",
        "| contender | role | seeds | cost | max\\|u\\| | max\\|u\\| − u_max | feasible | saturated |",
        "|---|---|---:|---:|---:|---:|:--:|---:|",
    ]
    for row in rows_:
        max_abs = row["max_abs"]
        if max_abs is None:
            lines.append(
                f"| {row['display']} | {row['role']} | {row['seeds']} | "
                f"{sum(row['cost']) / len(row['cost']):.4f} | — | — | "
                "**no statistic** | — |"
            )
            continue
        violation = max_abs - u_max
        expected = row["role"] in FEASIBLE_ROLES
        ok = violation <= 0.0
        mark = "yes" if ok else "**NO**"
        if not expected:
            mark = "violates (expected)" if not ok else "**feasible?**"
        fraction = (
            "—"
            if not row["fractions"]
            else f"{sum(row['fractions']) / len(row['fractions']):.2%}"
        )
        lines.append(
            f"| {row['display']} | {row['role']} | {row['seeds']} | "
            f"{sum(row['cost']) / len(row['cost']):.4f} | {max_abs:.9f} | "
            f"{violation:+.3e} | {mark} | {fraction} |"
        )

    audited = [r for r in rows_ if r["max_abs"] is not None]
    infeasible = [
        r for r in audited if r["role"] in FEASIBLE_ROLES and r["max_abs"] - u_max > 0.0
    ]
    references = [r for r in rows_ if r["role"] == "reference"]
    lines.append("")
    if not audited:
        # NEVER a clean bill of health on an audit nobody performed. A summary
        # counting rows rather than checks would read "9 of 9 as declared" on a
        # store carrying no statistic at all -- which is the shape of failure
        # the `constraint_binds` gate spent thirty documents demonstrating.
        lines.append(
            f"**NOT AUDITED: none of the {len(rows_)} rows carries the "
            "statistic.** Every measurement here predates Annex 03 §A.3.1a. "
            "Re-evaluate the study to retain it -- the models are reused and "
            "no identifier moves."
        )
        return "\n".join(lines) + "\n"
    lines.append(
        f"**{len(audited) - len(infeasible)} of {len(audited)} audited rows are "
        f"as declared**"
        + (
            "."
            if len(audited) == len(rows_)
            else f", and {len(rows_) - len(audited)} carry no statistic."
        )
    )
    if infeasible:
        lines.append(
            "**"
            + ", ".join(r["display"] for r in infeasible)
            + " violate the box they claim to respect.**"
        )
    for reference in references:
        if reference["max_abs"] is None:
            continue
        lines.append(
            f"The reference `{reference['display']}` reaches "
            f"`max|u| = {reference['max_abs']:.6f}`, "
            f"**{reference['max_abs'] / u_max:.1f}×** the box — which is what "
            "makes it a reference and not a competitor."
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("document", type=Path, help="a study `.toml`, or its id")
    parser.add_argument("--tier", default="publication")
    parser.add_argument("--store", type=Path, default=None)
    parser.add_argument("--depth", type=int, default=10, help="J, where swept")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    root = default_store() if args.store is None else args.store
    loaded = load_study(args.document, tier=args.tier)
    u_max = loaded.study.evaluation.problem.data.control_bound
    if u_max is None:
        print("this study's evaluation problem declares no box; nothing to audit")
        return 1

    gathered = rows(loaded, MeasurementStore(root), args.depth)
    if not gathered:
        print(f"no measurements at J = {args.depth} in {root}")
        return 1
    rendered = render(gathered, float(u_max), args.depth)
    print(rendered)

    destination = args.out or (root / "feasibility_audit.md")
    destination.write_text(rendered, encoding="utf-8")
    print(f"# written to {destination}")
    return 0


if __name__ == "__main__":  # pragma: no cover - a command line
    raise SystemExit(main())
