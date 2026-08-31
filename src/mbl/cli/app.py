"""The `mbl` entry point: argument parsing and dispatch (Annex 02 §5).

Structure follows the annex's grouping -- `models`, `measurements`, `store` --
with one handler per subcommand. Handlers do no formatting of their own: they
query, and hand the rows to `render`, which is pure and separately tested.

Three rules hold throughout:

* **Read-only commands are safe by construction.** They open the index and
  nothing else, so there is no path by which a query mutates the store.
* **Destructive commands require confirmation or `--yes`,** and print what they
  would remove before asking.
* **Nothing on the read path imports a tensor library.** The store's exports are
  lazy for this reason, and `tests/store/test_import_cost.py` fails if that
  regresses.

`mbl run` is the one command that *writes*, and the one that imports Tier 5. It
is deliberately thin — Annex 06's job DAG, resource classes and persistent queue
are deferred with reasons recorded in the Stages 3–5 slice plan, and a command
that grew a scheduler would be the thing that plan exists to prevent. `mbl queue`
remains absent for that reason.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..store.location import DEFAULT_STORE, STORE_ENV, default_store
from ..store.index import (
    AmbiguousPrefixError,
    MeasurementRow,
    ModelRow,
    StoreIndex,
    UnknownPrefixError,
)
from ..store.maintenance import (
    GarbagePolicy,
    NoLiveStudiesError,
    apply_gc,
    plan_gc,
    problem_sizes,
    store_stat,
    verify_store,
)
from .render import (
    DASH,
    abbreviate,
    flatten_spec,
    format_size,
    group_models,
    render_diff,
    render_fields,
    render_table,
    render_tree,
)


#: Exit codes. `REFUSED` covers both a failed verification and a command that
#: declined to act, so a script can tell "did nothing" from "worked".
OK, REFUSED, USAGE = 0, 1, 2

#: Annex 01 §2.3's default tier. Held here rather than beside the runner so
#: that building the parser -- which every invocation does -- costs no import.
DEFAULT_TIER = "standard"

#: How `--sort` keys map onto model row attributes.
SORT_KEYS = {
    "name": "semantic_name",
    "created": "created_utc",
    "seed": "seed",
    "family": "family",
    "contender": "contender_id",
    "problem": "problem_id",
    "wall": "wall_time_s",
    "vram": "peak_vram_mb",
}

_DURATION = re.compile(r"(?P<count>\d+)(?P<unit>[smhdw])")
_DURATION_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86_400, "w": 604_800}


def parse_duration(text: str) -> float:
    """`7d`, `36h`, `90m` as seconds.

    Raises:
        argparse.ArgumentTypeError: If `text` is not a duration. A bare number
            is rejected rather than assumed to be seconds, because `--older-than
            7` meaning seconds when the user meant days would delete a week of
            recoverable work.
    """
    match = _DURATION.fullmatch(text.strip())
    if match is None:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not a duration; use a count and a unit, e.g. 7d, 36h, 90m"
        )
    return int(match["count"]) * _DURATION_SECONDS[match["unit"]]


def _root(args: argparse.Namespace) -> Path:
    if args.store is not None:
        return Path(args.store)
    return default_store()


def _open(args: argparse.Namespace) -> tuple[Path, StoreIndex]:
    root = _root(args)
    return root, StoreIndex(root)


def _resolve(index: StoreIndex, reference: str) -> str | None:
    """A model reference, or `None` after printing why it did not resolve."""
    try:
        return index.resolve_model_reference(reference)
    except (UnknownPrefixError, AmbiguousPrefixError) as error:
        print(error.args[0] if error.args else error)
        return None


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------


def _filtered_models(index: StoreIndex, args: argparse.Namespace) -> list[ModelRow]:
    rows = index.models(
        problem_id=args.problem,
        family=args.family,
        contender_id=args.contender,
        seed=args.seed,
    )
    since = getattr(args, "since", None)
    if since:
        rows = [row for row in rows if (row.created_utc or "") >= since]
    key = SORT_KEYS[getattr(args, "sort", "name") or "name"]
    return sorted(rows, key=lambda row: (getattr(row, key) is None, getattr(row, key)))


def models_list(args: argparse.Namespace) -> int:
    _, index = _open(args)
    rows = _filtered_models(index, args)
    short = abbreviate(row.model_id for row in rows)
    print(
        render_table(
            ("identifier", "seed", "created", "name"),
            [
                [
                    short[row.model_id],
                    str(row.seed),
                    (row.created_utc or DASH)[:19],
                    row.semantic_name or DASH,
                ]
                for row in rows
            ],
        )
    )
    return OK


def models_tree(args: argparse.Namespace) -> int:
    root, index = _open(args)
    # Every measurement is handed over unfiltered: `group_models` attributes one
    # only to a contender it can see, so a narrowed tree cannot pick up the
    # evaluations of a model the filter excluded. Filtering here as well would
    # be a second copy of that rule, and the copy nothing tests is the one that
    # drifts.
    print(
        render_tree(
            group_models(
                _filtered_models(index, args),
                index.measurements(),
                problem_sizes(root, index),
            )
        )
    )
    return OK


def models_show(args: argparse.Namespace) -> int:
    _, index = _open(args)
    model_id = _resolve(index, args.reference)
    if model_id is None:
        return REFUSED
    row = index.model(model_id)
    measurements = index.measurements(model_id=model_id)
    spec = json.loads(row.spec_json or "{}")

    if args.json:
        print(json.dumps({"model": asdict(row), "spec": spec}, indent=2, default=str))
        return OK

    print(row.semantic_name or row.model_id)
    print()
    print(render_fields(_show_fields(row, measurements)))
    if spec:
        print("\nspecification")
        print(render_fields(sorted(flatten_spec(spec).items())))
    return OK


def _show_fields(
    row: ModelRow, measurements: Sequence[MeasurementRow]
) -> list[tuple[str, str]]:
    shifted = sum(1 for m in measurements if m.is_shifted)
    dimensions = " ".join(
        f"{label}={value}"
        for label, value in (
            ("n", row.state_dim),
            ("m", row.control_dim),
            ("N", row.horizon),
        )
        if value is not None
    )
    return [
        ("model", row.model_id),
        ("problem", row.problem_id),
        ("family", row.family or DASH),
        ("contender", row.contender_id or DASH),
        ("seed", str(row.seed)),
        ("dimensions", dimensions or DASH),
        ("created", row.created_utc or DASH),
        ("stamp", row.stamp or DASH),
        ("wall time", f"{row.wall_time_s:.1f} s" if row.wall_time_s else DASH),
        ("peak VRAM", f"{row.peak_vram_mb:.0f} MB" if row.peak_vram_mb else DASH),
        (
            "measurements",
            f"{len(measurements) - shifted} nominal, {shifted} shifted",
        ),
    ]


def models_diff(args: argparse.Namespace) -> int:
    _, index = _open(args)
    left, right = (_resolve(index, ref) for ref in (args.left, args.right))
    if left is None or right is None:
        return REFUSED
    rows = [index.model(left), index.model(right)]
    print("comparing")
    for label, row in zip(("a", "b"), rows):
        print(f"  {label}  {row.semantic_name or row.model_id}")
    print()
    short = abbreviate((left, right))
    print(
        render_diff(
            short[left],
            short[right],
            json.loads(rows[0].spec_json or "{}"),
            json.loads(rows[1].spec_json or "{}"),
        )
    )
    return OK


# --------------------------------------------------------------------------
# measurements
# --------------------------------------------------------------------------


def measurements_list(args: argparse.Namespace) -> int:
    _, index = _open(args)
    model_id: str | None = None
    if args.model is not None:
        model_id = _resolve(index, args.model)
        if model_id is None:
            return REFUSED
    rows = index.measurements(model_id=model_id, shifted=args.shifted)
    keys = sorted({key for row in rows for key in row.metrics})
    # Each column is abbreviated against its own set: a measurement identifier
    # needing sixteen characters says nothing about how many the model column
    # needs, and widening both would waste the terminal.
    short = abbreviate(row.measurement_id for row in rows)
    models = abbreviate(row.model_id for row in rows)
    problems = abbreviate(row.eval_problem_id for row in rows)
    print(
        render_table(
            ("identifier", "model", "eval problem", "shift", *keys),
            [
                [
                    short[row.measurement_id],
                    models[row.model_id],
                    problems[row.eval_problem_id],
                    "shifted" if row.is_shifted else "nominal",
                    *(_metric(row.metrics.get(key)) for key in keys),
                ]
                for row in rows
            ],
        )
    )
    return OK


def _metric(value: float | None) -> str:
    return DASH if value is None else f"{value:.6g}"


# --------------------------------------------------------------------------
# store
# --------------------------------------------------------------------------


def store_stat_command(args: argparse.Namespace) -> int:
    root, index = _open(args)
    stat = store_stat(root, index)
    print(
        render_table(
            ("class", "count", "size"),
            [[c.kind, str(c.count), format_size(c.size_bytes)] for c in stat.classes],
        )
    )
    print(f"\ntotal  {format_size(stat.total_bytes)}  ({root})")
    if stat.growth:
        print("\ngrowth")
        print(
            render_table(
                ("month", "models", "measurements"),
                [[b.month, str(b.models), str(b.measurements)] for b in stat.growth],
            )
        )
    return OK


def store_verify(args: argparse.Namespace) -> int:
    root, index = _open(args)
    report = verify_store(root, index)
    for line in report.lines():
        print(line)
    return OK if report.ok else REFUSED


def store_reindex(args: argparse.Namespace) -> int:
    from ..store.content_store import MeasurementStore, ModelStore

    root, index = _open(args)
    index.reindex(ModelStore(root), MeasurementStore(root))
    print(
        f"reindexed {len(index.models())} models and "
        f"{len(index.measurements())} measurements; the queue was preserved"
    )
    return OK


def _confirm(prompt: str) -> bool:
    try:
        answer = input(f"{prompt} [y/N] ")
    except EOFError:
        return False
    return answer.strip().lower() in {"y", "yes"}


def store_gc(args: argparse.Namespace) -> int:
    if args.partials and args.models:
        print("--models and --partials collect different things; run them separately")
        return USAGE
    root, index = _open(args)
    try:
        plan = plan_gc(
            root,
            index,
            GarbagePolicy(
                keep_studies=args.keep_studies,
                measurements=not args.partials,
                include_models=args.models,
                partials=args.partials,
                older_than_seconds=args.older_than,
            ),
        )
    except NoLiveStudiesError as error:
        print(error.args[0])
        return REFUSED

    if plan.empty:
        print("nothing to collect")
        return OK
    for label, count in (
        ("measurements", len(plan.measurements)),
        ("models", len(plan.models)),
        ("resume artifacts", len(plan.partials)),
    ):
        if count:
            print(f"  {count} {label}")
    print(f"would free {format_size(plan.bytes_freed)}")
    if args.dry_run:
        return OK
    if not args.yes and not _confirm("proceed?"):
        print("cancelled; nothing was removed")
        return REFUSED
    apply_gc(root, index, plan)
    print(f"collected; freed {format_size(plan.bytes_freed)}")
    return OK


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------


def _add_model_filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--problem", help="restrict to one problem identifier")
    parser.add_argument("--family", help="restrict to one contender family")
    parser.add_argument("--contender", help="restrict to one contender")
    parser.add_argument("--seed", type=int, help="restrict to one training seed")


def _build_models(subparsers: Any) -> None:
    models = subparsers.add_parser("models", help="what models the store holds")
    commands = models.add_subparsers(dest="command", required=True)

    listing = commands.add_parser("list", help="one line per model")
    _add_model_filters(listing)
    listing.add_argument("--since", help="only models created on or after this date")
    listing.add_argument("--sort", choices=sorted(SORT_KEYS), default="name")
    listing.set_defaults(handler=models_list)

    tree = commands.add_parser(
        "tree", help="models grouped by problem, then contender, then seed"
    )
    _add_model_filters(tree)
    tree.set_defaults(handler=models_tree, sort="name", since=None)

    show = commands.add_parser("show", help="everything known about one model")
    show.add_argument(
        "reference", help="identifier, alias, or a unique prefix of either"
    )
    show.add_argument("--json", action="store_true", help="machine-readable output")
    show.set_defaults(handler=models_show)

    diff = commands.add_parser("diff", help="which specification fields differ")
    diff.add_argument("left")
    diff.add_argument("right")
    diff.set_defaults(handler=models_diff)


def _build_measurements(subparsers: Any) -> None:
    measurements = subparsers.add_parser(
        "measurements", help="what evaluations the store holds"
    )
    commands = measurements.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="one line per measurement")
    listing.add_argument("--model", help="restrict to one model")
    shift = listing.add_mutually_exclusive_group()
    shift.add_argument(
        "--shifted",
        action="store_const",
        const=True,
        dest="shifted",
        help="only measurements evaluated away from the training problem",
    )
    shift.add_argument(
        "--nominal",
        action="store_const",
        const=False,
        dest="shifted",
        help=argparse.SUPPRESS,
    )
    listing.set_defaults(handler=measurements_list, shifted=None)


def _build_store(subparsers: Any) -> None:
    store = subparsers.add_parser("store", help="maintenance")
    commands = store.add_subparsers(dest="command", required=True)
    commands.add_parser("stat", help="size by class, growth over time").set_defaults(
        handler=store_stat_command
    )
    commands.add_parser(
        "verify", help="integrity, identifier consistency, orphan detection"
    ).set_defaults(handler=store_verify)
    commands.add_parser(
        "reindex", help="rebuild the index from the trees (the queue is preserved)"
    ).set_defaults(handler=store_reindex)

    collect = commands.add_parser("gc", help="drop what no live study references")
    collect.add_argument(
        "--keep-studies",
        nargs="+",
        metavar="STUDY",
        help="treat only these studies as live",
    )
    collect.add_argument(
        "--models",
        action="store_true",
        help="also collect unreferenced models; off by default because a model "
        "is the one expensive object in the store",
    )
    collect.add_argument(
        "--partials",
        action="store_true",
        help="collect abandoned resume artifacts instead of measurements",
    )
    collect.add_argument("--older-than", type=parse_duration, metavar="AGE")
    collect.add_argument("--dry-run", action="store_true", help="report and stop")
    collect.add_argument("--yes", action="store_true", help="skip the confirmation")
    collect.set_defaults(handler=store_gc)


def run_study_command(args: argparse.Namespace) -> int:
    """`mbl run`. The one handler that writes, and the one that needs Tier 5.

    The runner is imported **inside** the handler, not at module scope. Every
    other command on this surface reads the store and must not pay for a tensor
    library to do it (`tests/store/test_import_cost.py`), and `from .run import
    …` at the top of this module pulled numpy and torch into `mbl models list`
    — as does importing even `mbl.spec.errors`, since that executes the
    grammar's package `__init__`. So the decision is made in `run.py` and
    returned as a value; this handler only prints it.
    """
    from .run import preview_or_execute

    ok, message = preview_or_execute(
        args.study,
        store=_root(args),
        tier=args.tier,
        catalogue=args.catalogue,
        assignments=args.set or (),
        previewed=args.dry_run,
    )
    print(message)
    return OK if ok else REFUSED


def analyse_command(args: argparse.Namespace) -> int:
    """`mbl analyse`. Reads the store; writes only into the study's own tree.

    Deliberately **not** `creates_store=True`: this command reads the store,
    so a mistyped `--store` must be refused by `main()` rather than silently
    producing an empty one. Only `mbl run` creates a store.

    The import is inside the handler for the reason `run_study_command`'s is:
    `mbl.analysis` pulls pandas and scipy, and every reading command on this
    surface must stay able to answer in milliseconds.
    """
    from .analyse import analyse

    ok, message = analyse(
        args.study,
        store=_root(args),
        tier=args.tier,
        catalogue=args.catalogue,
        overrides=_overrides(args),
        only=args.only or None,
    )
    print(message)
    return OK if ok else REFUSED


def _overrides(args: argparse.Namespace) -> dict[str, Any]:
    """`--set` assignments, parsed by the same function `mbl run` uses.

    Shared rather than re-parsed: a table computed for one set of overrides and
    a run executed for another would be a table about a different study, and
    two little parsers would eventually disagree about `--set training.seeds=2`.
    """
    from .run import parse_overrides

    return parse_overrides(args.set or ())


def figure_command(args: argparse.Namespace) -> int:
    """`mbl figure`. Reads the store; writes only into the study's own tree.

    Like `mbl analyse` and unlike `mbl run`, `creates_store` is not set: this
    command reads a store, so a mistyped `--store` must be refused rather than
    silently producing an empty one.
    """
    from .figure import rebuild, render

    if args.command == "rebuild":
        ok, message = rebuild(args.figure, store=_root(args), style=args.style)
    else:
        ok, message = render(
            args.study,
            store=_root(args),
            tier=args.tier,
            catalogue=args.catalogue,
            overrides=_overrides(args),
            only=args.only or None,
            style=args.style,
        )
    print(message)
    return OK if ok else REFUSED


def _build_figure(subparsers: Any) -> None:
    """`mbl figure render` and `mbl figure rebuild` -- Annex 03 §B.1."""
    figure = subparsers.add_parser(
        "figure", help="render a study's declared figures from its tables"
    )
    commands = figure.add_subparsers(dest="command", required=True)

    draw = commands.add_parser("render", help="draw every figure a study declares")
    draw.add_argument("study", help="the study .toml")
    draw.add_argument(
        "--tier",
        default=DEFAULT_TIER,
        metavar="NAME",
        help=f"execution tier the study was run at (default: {DEFAULT_TIER})",
    )
    draw.add_argument(
        "--catalogue",
        metavar="PATH",
        help="tier catalogue .toml (default: the shipped catalogue)",
    )
    draw.add_argument(
        "--set",
        action="append",
        metavar="PATH=VALUE",
        help="per-invocation override; VALUE is TOML, so a string needs quoting",
    )
    draw.add_argument(
        "--only", action="append", metavar="ID", help="render only this figure"
    )
    draw.add_argument(
        "--style",
        metavar="PROFILE",
        help="style profile (thesis, ieee-1col, ieee-2col, neurips, talk)",
    )
    draw.set_defaults(handler=figure_command)

    again = commands.add_parser(
        "rebuild",
        help="re-render one stored figure at another style, from its own "
        "spec and data alone",
    )
    again.add_argument("figure", help="the figure id")
    again.add_argument(
        "--style",
        metavar="PROFILE",
        help="style profile (thesis, ieee-1col, ieee-2col, neurips, talk)",
    )
    again.set_defaults(handler=figure_command)


def _build_analyse(subparsers: Any) -> None:
    """`mbl analyse` -- a study's declared tables, computed from the store."""
    analyse = subparsers.add_parser(
        "analyse", help="compute a study's declared analyses from the store"
    )
    analyse.add_argument("study", help="the study .toml")
    analyse.add_argument(
        "--tier",
        default=DEFAULT_TIER,
        metavar="NAME",
        help=f"execution tier the study was run at (default: {DEFAULT_TIER})",
    )
    analyse.add_argument(
        "--catalogue",
        metavar="PATH",
        help="tier catalogue .toml (default: the shipped catalogue)",
    )
    analyse.add_argument(
        "--set",
        action="append",
        metavar="PATH=VALUE",
        help="per-invocation override; VALUE is TOML, so a string needs quoting",
    )
    analyse.add_argument(
        "--only",
        action="append",
        metavar="ID",
        help="compute only this analysis; repeatable",
    )
    analyse.set_defaults(handler=analyse_command)


def _build_run(subparsers: Any) -> None:
    """`mbl run` -- the one command that writes, and the one that needs Tier 5."""
    run = subparsers.add_parser("run", help="execute a study document into the store")
    run.add_argument("study", help="the study .toml")
    run.add_argument(
        "--tier",
        default=DEFAULT_TIER,
        metavar="NAME",
        help=f"execution tier (default: {DEFAULT_TIER})",
    )
    run.add_argument(
        "--catalogue",
        metavar="PATH",
        help="tier catalogue .toml (default: the shipped catalogue)",
    )
    run.add_argument(
        "--set",
        action="append",
        metavar="PATH=VALUE",
        help="per-invocation override; VALUE is TOML, so a string needs quoting",
    )
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="report how many models would be trained versus reused, and stop",
    )
    # `creates_store` exempts this command from main()'s existing-store check.
    # A flag rather than a name, so a second writing command cannot forget.
    run.set_defaults(handler=run_study_command, creates_store=True)


def build_parser() -> argparse.ArgumentParser:
    """The whole command line."""
    parser = argparse.ArgumentParser(
        prog="mbl", description="Inspect and maintain the result store."
    )
    parser.add_argument(
        "--store",
        metavar="PATH",
        help=f"store root (default: ${STORE_ENV}, else ./{DEFAULT_STORE})",
    )
    subparsers = parser.add_subparsers(dest="group", required=True)
    _build_models(subparsers)
    _build_measurements(subparsers)
    _build_store(subparsers)
    _build_run(subparsers)
    _build_analyse(subparsers)
    _build_figure(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command. Returns the process exit code."""
    args = build_parser().parse_args(argv)
    root = _root(args)
    if not root.is_dir() and not getattr(args, "creates_store", False):
        # Checked once, here, rather than in each handler: opening the index
        # creates the directory, so a mistyped `--store` would otherwise
        # silently produce a new empty store and report it as an empty one.
        # Every *reading* subcommand needs an existing store; `mbl run` is the
        # one that creates it, and is exempted by its own flag rather than by
        # name, so a second writing command cannot forget to be.
        print(
            f"no store at {root}; pass --store, or set ${STORE_ENV} to the "
            "store root (Annex 05 §2.3)"
        )
        return REFUSED
    handler: Any = args.handler
    return int(handler(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
