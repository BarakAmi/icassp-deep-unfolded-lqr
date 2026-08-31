"""Reload any stored figure from its two raw files, and look at it.

Every figure this project writes leaves four artifacts beside each other:

    <id>.pdf            what goes in the paper
    <id>.png            what goes in a notebook
    <id>.data.parquet   the exact values plotted
    <id>.spec.json      the declaration that plotted them

The last two are the figure. This script turns them back into a rendered figure
without a study document, an analysis, a model, or the store's index — which is
D6's argument for storing a data-plus-spec pair rather than a pickled `Figure`:
a pickle silently fails to load across matplotlib versions, and it can never be
re-styled.

    # what is in the store, and where
    uv run python tools/show_figure.py --list

    # re-render one, at the profile it was stored at
    uv run python tools/show_figure.py store/studies/<id>/figures/<fig>.spec.json

    # ... at another venue's geometry, written somewhere harmless
    uv run python tools/show_figure.py <fig>.spec.json --style ieee-1col --out /tmp/f.png

    # ... and what the figure is made of
    uv run python tools/show_figure.py <fig>.spec.json --describe

**It never writes into the store.** `--out` defaults to a temporary file, so
looking at a figure cannot change the artifact of record; `mbl figure rebuild`
is the command that deliberately does rewrite it in place.

From a notebook or a REPL, the same thing in one line — this is the surface, and
the script is a wrapper over it:

    from mbl.present import figure_from_artifacts
    figure = figure_from_artifacts("store/studies/<id>/figures/<fig>.spec.json")
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from matplotlib import pyplot as plt

from mbl.present import figure_from_artifacts
from mbl.present.artifacts import read_figure_data
from mbl.present.profiles import PROFILES

REPO = Path(__file__).resolve().parents[1]
DEFAULT_STORE = REPO / "store"


def stored_specs(store: Path) -> list[Path]:
    """Every figure specification under `store`, sorted."""
    return sorted(store.glob("studies/*/figures/*.spec.json"))


def describe(spec_path: Path) -> str:
    """What the figure is made of, without rendering it."""
    spec = json.loads(spec_path.read_text())
    figure_id = spec_path.name[: -len(".spec.json")]
    table = read_figure_data(spec_path.parent, figure_id)
    roles = dict(spec.get("roles", {}))
    names = dict(spec.get("display_names", {}))
    lines = [
        f"{figure_id}   ({spec.get('kind')})",
        f"  study      {spec.get('study')}  [{spec.get('study_id')}]",
        f"  analysis   {spec.get('source')}",
        f"  profile    {spec.get('rendered_at_profile')}",
        f"  rows       {len(table)}",
        "  series:",
    ]
    for label in spec.get("series_order", sorted(set(table["contender"]))):
        rows = table[table["contender"] == label]
        if rows.empty:
            continue
        swept = rows["axis_value"].notna().any()
        shape = f"{len(rows)} points" if swept else "one level"
        lines.append(
            f"    {names.get(label, label):38s} {roles.get(label, 'contender'):10s} {shape}"
        )
    config = spec.get("config", {})
    if config:
        lines.append("  declared:")
        lines.extend(f"    {key} = {value!r}" for key, value in sorted(config.items()))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "spec",
        nargs="?",
        type=Path,
        help="path to a figure's <id>.spec.json (its <id>.data.parquet is read "
        "from beside it)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list every figure in the store with the path to reload it",
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help="print what the figure is made of instead of rendering it",
    )
    parser.add_argument(
        "--style",
        default=None,
        help=f"re-render at another profile ({', '.join(sorted(PROFILES))}); "
        "the default is the profile the figure was stored at",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="where to write the PNG; the default is a temporary file, so "
        "looking at a figure never rewrites the stored artifact",
    )
    parser.add_argument(
        "--store", type=Path, default=DEFAULT_STORE, help="store root, for --list"
    )
    args = parser.parse_args(argv)

    if args.list:
        found = stored_specs(args.store)
        if not found:
            print(f"no figures under {args.store}")
            return 1
        for path in found:
            print(path.relative_to(REPO) if path.is_relative_to(REPO) else path)
        return 0

    if args.spec is None:
        parser.error("give a <id>.spec.json path, or --list to see them")

    if args.describe:
        print(describe(args.spec))
        return 0

    figure = figure_from_artifacts(args.spec, style=args.style)
    try:
        out = args.out or Path(tempfile.gettempdir()) / f"{args.spec.stem}.png"
        # `bbox_inches="tight"` for the reason the artifact writer uses it:
        # §B.5's margin labels are drawn OUTSIDE the axes, and a plain save
        # crops them to their first letter.
        figure.savefig(out, dpi=200, bbox_inches="tight")
    finally:
        plt.close(figure)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
