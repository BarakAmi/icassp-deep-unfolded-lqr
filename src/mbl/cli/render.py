"""Formatting for the `mbl` command line (Annex 02 §5).

Pure functions over index rows: nothing here touches the filesystem, opens a
database or reads a clock, so every layout decision is testable against a
literal expected string. That is what lets the annex's `mbl models tree` block
serve as the acceptance test verbatim.

The tree's four columns are each a *lifted* part of a semantic name. A model
named

    boxlqr-n7m3-N100-u0.1-s0/unfolded-aP-J10/adam-lr1e-2-ep200-seed0#3f9a1c

contributes its problem segment to the header, `unfolded-aP` and `J=10` to the
first two columns, `adam-lr1e-2-ep200` to the third, and its seed to the fourth,
where it is merged with its siblings into a range. Two consequences worth
naming:

* Only the **seed** is lifted out of the training segment, never the batch size.
  Dropping a field that varies between models would render two genuinely
  different models as one indistinguishable row.
* A training segment holding *only* a seed (`pgd-fixed`, which has no fitted
  parameters but is still seeded) renders an empty training column and a
  populated seed column, while a segment that is empty outright
  (`riccati-trunc`, analytic) renders both as `—`. That distinction is what the
  annex's third and fourth rows show.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..store.naming import DIGEST_SEPARATOR, EMPTY_SEGMENT, SEGMENT_SEPARATOR
from ..store.index import MeasurementRow, ModelRow

#: Stands in for a column with nothing to show. One display column wide, so it
#: pads like any other content.
DASH = "—"

#: Completeness markers for the `k/n` column.
COMPLETE = "✓"
INCOMPLETE = "✗"

#: Minimum widths of the tree's four columns -- contender, depth, training,
#: seeds -- taken from the annex's block. They are minima, not maxima: content
#: wider than this widens the column for every row, because letting one row
#: overflow would push its remaining columns out of alignment with the rest.
TREE_COLUMNS = (19, 7, 20, 11)

#: Width the problem name is padded to before the header's summary.
HEADER_NAME_WIDTH = 60

#: Indent of a contender's children.
CHILD_INDENT = 4

_BRANCH, _LAST_BRANCH = "├── ", "└── "
_TRUNK, _NO_TRUNK = "│" + " " * (CHILD_INDENT - 1), " " * CHILD_INDENT

#: A trailing `J<depth>` in a contender segment, and a trailing `seed<n>` in a
#: training segment. Matched as whole-segment suffixes rather than by splitting
#: on `-`, because `lr1e-2` contains a hyphen of its own.
_DEPTH_SUFFIX = re.compile(r"(?:(?P<head>.+)-)?J(?P<value>\d+)")
_SEED_SUFFIX = re.compile(r"(?:(?P<head>.+)-)?seed(?P<value>\d+)")

#: Characters of a problem identifier shown when two problems render the same
#: name.
_DISAMBIGUATION_WIDTH = 8


@dataclass(frozen=True)
class ContenderGroup:
    """Every model of one contender, at one depth, under one training config.

    Attributes:
        contender: The contender segment with any depth lifted out.
        depth: Unfolding depth, or `None` where the contender has none.
        training: The training segment with the seed lifted out, or `None` where
            nothing remains (an analytic or seed-only contender).
        seeds: The distinct training seeds present, ascending. Empty when the
            contender has no seed dimension at all.
        has_seed: Whether a training seed is meaningful here.
        model_count: Models in the group, which exceeds `len(seeds)` only if two
            models share a seed while differing in something the name omits.
        nominal: Measurements evaluated on the training problem.
        shifted: Measurements evaluated on a different problem.
    """

    contender: str
    depth: int | None
    training: str | None
    seeds: tuple[int, ...]
    has_seed: bool
    model_count: int
    nominal: int = 0
    shifted: int = 0

    @property
    def present(self) -> int:
        """Distinct seeds held, or the model count where seeds are meaningless."""
        return len(self.seeds) if self.has_seed else self.model_count

    @property
    def expected(self) -> int:
        """How many the seed range implies.

        The span rather than the count, so that seeds `0,1,2,4` report `4/5`:
        a hole in a seed sequence is a crashed or never-launched run, and it is
        precisely what a glance at the tree should surface.
        """
        if not self.has_seed or not self.seeds:
            return self.model_count
        return self.seeds[-1] - self.seeds[0] + 1

    @property
    def complete(self) -> bool:
        return self.present == self.expected


@dataclass(frozen=True)
class ProblemGroup:
    """Every contender trained on one problem.

    Attributes:
        problem_id: The problem's identifier, which is what grouping keys on.
        display: The header label -- the problem's name segment, suffixed with a
            short identifier only when another problem renders the same name.
        contenders: Ordered as the tree prints them.
        model_count: Models across every contender.
        size_bytes: Bytes this problem occupies, or `None` when not measured.
    """

    problem_id: str
    display: str
    contenders: tuple[ContenderGroup, ...]
    model_count: int
    size_bytes: int | None = None

    @property
    def contender_count(self) -> int:
        """Distinct contenders, which is not the number of rows.

        A contender swept over several depths, or fitted under two training
        configurations, is **one** contender and several rows. Before Stage 2
        Phase F3 the two could not disagree: with no semantic name to parse,
        `_decompose` fell back to the indexed columns and every depth of a
        contender collapsed into a single row. With real names NB04 at `smoke`
        is six contenders across nine rows, and the header said nine.

        Annex 02 §3's own example is a case where the two coincide — four
        contenders at one depth each — which is why the verbatim block this
        module is pinned against cannot catch it.
        """
        return len({group.contender for group in self.contenders})


def format_size(size_bytes: int) -> str:
    """Bytes as a compact decimal quantity, e.g. `61 MB`."""
    value = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1000 or unit == "TB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}" if value < 10 else f"{value:.0f} {unit}"
        value /= 1000
    raise AssertionError("unreachable")  # pragma: no cover


def format_seeds(seeds: Sequence[int]) -> str:
    """Seeds as compacted runs, e.g. `seeds 0-2,4`.

    The runs matter: knowing that seed 3 is the one missing is what says which
    run to relaunch, and a bare `4/5` does not.
    """
    if not seeds:
        return DASH
    runs: list[tuple[int, int]] = []
    for seed in seeds:
        if runs and seed == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], seed)
        else:
            runs.append((seed, seed))
    return "seeds " + ",".join(
        str(low) if low == high else f"{low}-{high}" for low, high in runs
    )


def _split_suffix(
    segment: str, pattern: re.Pattern[str]
) -> tuple[str | None, int | None]:
    """Lift a trailing numeric field out of a name segment."""
    match = pattern.fullmatch(segment)
    if match is None:
        return segment, None
    return match["head"], int(match["value"])


def _decompose(
    row: ModelRow,
) -> tuple[str, str, tuple[str, int | None, str | None, bool]]:
    """`(problem_id, problem_display, contender key)` for one model.

    A row whose semantic name is absent or malformed is decomposed from its
    indexed columns instead. `reindex` writes exactly such rows -- projecting a
    name needs the grammar, which the store deliberately does not depend on --
    and dropping them would make the tree under-report what is stored.
    """
    try:
        problem, contender_segment, training_segment, _ = _split_name(row.semantic_name)
    except ValueError:
        fallback = row.contender_id or row.family or DASH
        return row.problem_id, row.problem_id, (fallback, None, None, True)

    contender, depth = _split_suffix(contender_segment, _DEPTH_SUFFIX)
    if training_segment == EMPTY_SEGMENT:
        training, has_seed = None, False
    else:
        training, seed = _split_suffix(training_segment, _SEED_SUFFIX)
        has_seed = seed is not None
    return row.problem_id, problem, (contender or DASH, depth, training, has_seed)


def _split_name(name: str) -> tuple[str, str, str, str]:
    head, _, digest = name.rpartition(DIGEST_SEPARATOR)
    segments = head.split(SEGMENT_SEPARATOR)
    if not digest or len(segments) != 3:
        raise ValueError(name)
    return segments[0], segments[1], segments[2], digest


def group_models(
    models: Iterable[ModelRow],
    measurements: Iterable[MeasurementRow] = (),
    sizes: Mapping[str, int] | None = None,
) -> tuple[ProblemGroup, ...]:
    """Arrange index rows into the tree's problem/contender/seed hierarchy.

    Args:
        models: The models to show.
        measurements: Measurements to attribute to their contender. Any whose
            model is not in `models` is ignored, so a filtered tree never
            reports another problem's evaluations as its own.
        sizes: Bytes per `problem_id`. A problem absent here reports `None`
            rather than zero, because an unmeasured size is not an empty one.

    Returns:
        Problems by name, each holding its contenders most-populated first.
    """
    Key = tuple[str, int | None, str | None, bool]
    buckets: dict[str, dict[Key, list[ModelRow]]] = defaultdict(
        lambda: defaultdict(list)
    )
    displays: dict[str, str] = {}
    located: dict[str, tuple[str, Key]] = {}

    for row in models:
        problem_id, display, key = _decompose(row)
        buckets[problem_id][key].append(row)
        displays.setdefault(problem_id, display)
        located[row.model_id] = (problem_id, key)

    counts: dict[tuple[str, Key], list[int]] = defaultdict(lambda: [0, 0])
    for measurement in measurements:
        where = located.get(measurement.model_id)
        if where is not None:
            counts[where][bool(measurement.is_shifted)] += 1

    labels = _disambiguate(displays)
    sizes = sizes or {}
    groups = [
        ProblemGroup(
            problem_id=problem_id,
            display=labels[problem_id],
            contenders=_contenders(problem_id, keyed, counts),
            model_count=sum(len(rows) for rows in keyed.values()),
            size_bytes=sizes.get(problem_id),
        )
        for problem_id, keyed in buckets.items()
    ]
    return tuple(sorted(groups, key=lambda g: (g.display, g.problem_id)))


def _disambiguate(displays: Mapping[str, str]) -> dict[str, str]:
    """Append a short identifier to any name two problems share.

    Distinct problems can render identical descriptors -- two box-LQR instances
    of the same shape whose matrices differ. Merging them under one header would
    report one problem's models as another's, so the collision is broken rather
    than tolerated; a name held by one problem is left untouched.
    """
    shared = {
        name for name in displays.values() if list(displays.values()).count(name) > 1
    }
    return {
        problem_id: (
            f"{name} #{problem_id[:_DISAMBIGUATION_WIDTH]}" if name in shared else name
        )
        for problem_id, name in displays.items()
    }


def _contenders(
    problem_id: str,
    keyed: Mapping[tuple[str, int | None, str | None, bool], list[ModelRow]],
    counts: Mapping[tuple[str, tuple[str, int | None, str | None, bool]], list[int]],
) -> tuple[ContenderGroup, ...]:
    groups = []
    for key, rows in keyed.items():
        contender, depth, training, has_seed = key
        nominal, shifted = counts.get((problem_id, key), [0, 0])
        groups.append(
            ContenderGroup(
                contender=contender,
                depth=depth,
                training=training,
                seeds=tuple(sorted({row.seed for row in rows})) if has_seed else (),
                has_seed=has_seed,
                model_count=len(rows),
                nominal=nominal,
                shifted=shifted,
            )
        )
    # Most-populated contender first; within a tie the descending name, which
    # puts the elaborate variant ahead of the plain one (`unfolded-aP` before
    # `unfolded-a`) and the learned families ahead of the analytic baselines.
    # This is the order the annex's example prints.
    return tuple(
        sorted(groups, key=lambda g: (-g.model_count, _Descending(g.contender)))
    )


@dataclass(frozen=True)
class _Descending:
    """Sort key inverting a string, so it can be mixed with ascending keys."""

    value: str

    def __lt__(self, other: _Descending) -> bool:
        return self.value > other.value


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def render_tree(groups: Sequence[ProblemGroup]) -> str:
    """The `mbl models tree` view (Annex 02 §5)."""
    if not groups:
        return "the store holds no models"
    widths = _column_widths(groups)
    lines: list[str] = []
    for group in groups:
        lines.append(_header(group))
        last = len(group.contenders) - 1
        for position, contender in enumerate(group.contenders):
            lines.append(_row(contender, widths, final=position == last))
            if contender.nominal or contender.shifted:
                trunk = _NO_TRUNK if position == last else _TRUNK
                lines.append(
                    f"{trunk}{_LAST_BRANCH}measurements: "
                    f"{contender.nominal} nominal, {contender.shifted} shifted"
                )
    return "\n".join(lines)


def _header(group: ProblemGroup) -> str:
    parts = [
        _plural(group.contender_count, "contender"),
        _plural(group.model_count, "model"),
    ]
    if group.size_bytes is not None:
        parts.append(format_size(group.size_bytes))
    summary = f"({', '.join(parts)})"
    name = group.display
    padded = (
        name.ljust(HEADER_NAME_WIDTH) if len(name) < HEADER_NAME_WIDTH else name + " "
    )
    return padded + summary


def _cells(contender: ContenderGroup) -> tuple[str, str, str, str]:
    return (
        contender.contender,
        f"J={contender.depth}" if contender.depth is not None else DASH,
        contender.training if contender.training is not None else DASH,
        format_seeds(contender.seeds),
    )


def _column_widths(groups: Sequence[ProblemGroup]) -> tuple[int, ...]:
    widest = [0] * len(TREE_COLUMNS)
    for group in groups:
        for contender in group.contenders:
            for column, cell in enumerate(_cells(contender)):
                widest[column] = max(widest[column], len(cell))
    return tuple(
        max(minimum, longest + 1) for minimum, longest in zip(TREE_COLUMNS, widest)
    )


def _row(contender: ContenderGroup, widths: Sequence[int], *, final: bool) -> str:
    cells = "".join(cell.ljust(width) for cell, width in zip(_cells(contender), widths))
    mark = COMPLETE if contender.complete else INCOMPLETE
    branch = _LAST_BRANCH if final else _BRANCH
    return f"{branch}{cells}{mark} {contender.present}/{contender.expected}"


# --------------------------------------------------------------------------
# Tables, field lists and specification diffs.
# --------------------------------------------------------------------------

#: Gap between table columns.
COLUMN_GAP = 2

#: Shortest identifier prefix ever shown. Long enough to recognise and to type.
ABBREVIATION_MINIMUM = 8


def abbreviate(
    identifiers: Iterable[str], minimum: int = ABBREVIATION_MINIMUM
) -> dict[str, str]:
    """Map each identifier to the shortest prefix that is unique among them.

    A fixed truncation cannot promise uniqueness, and the identifier column is
    precisely what a user copies into the next command: two records printed as
    the same eight characters give a reference that resolves to neither, while
    looking like a duplicate row. One width is chosen for the whole set, so the
    column stays aligned.
    """
    unique = sorted(set(identifiers))
    if not unique:
        return {}
    longest = max(len(identifier) for identifier in unique)
    width = minimum
    while width < longest and len({i[:width] for i in unique}) < len(unique):
        width += 1
    return {identifier: identifier[:width] for identifier in unique}


def render_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """A left-aligned table, sized to its contents.

    Trailing padding is stripped from every line, so output piped into a diff or
    a fixture compares on content rather than on invisible whitespace.
    """
    if not rows:
        return "nothing to show"
    widths = [
        max(len(str(cell)) for cell in column)
        for column in zip(headers, *rows, strict=True)
    ]
    gap = " " * COLUMN_GAP
    lines = [gap.join(h.ljust(w) for h, w in zip(headers, widths))]
    lines.append(gap.join("-" * w for w in widths))
    lines.extend(
        gap.join(str(cell).ljust(w) for cell, w in zip(row, widths)) for row in rows
    )
    return "\n".join(line.rstrip() for line in lines)


def render_fields(pairs: Sequence[tuple[str, str]]) -> str:
    """An aligned `label  value` block, for `mbl models show`."""
    if not pairs:
        return ""
    width = max(len(label) for label, _ in pairs)
    return "\n".join(
        f"  {label.ljust(width)}  {value}".rstrip() for label, value in pairs
    )


def flatten_spec(tree: Mapping[str, Any], prefix: str = "") -> dict[str, str]:
    """A specification tree as dotted paths to rendered leaf values.

    Nested mappings are walked so that a diff points at
    `training.plan.epochs` rather than reporting the whole `training` sub-tree
    as one opaque change. Lists are left whole: their order is meaningful, and
    an element-wise diff of a metric list is noise rather than signal.
    """
    flat: dict[str, str] = {}
    for key, value in sorted(tree.items()):
        path = f"{prefix}{key}"
        if isinstance(value, Mapping) and value:
            flat.update(flatten_spec(value, f"{path}."))
        elif isinstance(value, (Mapping, list, tuple)):
            flat[path] = json.dumps(value, sort_keys=True, default=str)
        else:
            flat[path] = str(value)
    return flat


def render_diff(
    left_label: str,
    right_label: str,
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> str:
    """Which specification fields differ between two records.

    A field present on one side only shows `—` on the other, rather than being
    omitted: "this model has no `u_max`" is exactly the kind of difference the
    command exists to surface.
    """
    a, b = flatten_spec(left), flatten_spec(right)
    differing = sorted(key for key in set(a) | set(b) if a.get(key) != b.get(key))
    if not differing:
        return "the two specifications are identical"
    return render_table(
        ("field", left_label, right_label),
        [[key, a.get(key, DASH), b.get(key, DASH)] for key in differing],
    )
