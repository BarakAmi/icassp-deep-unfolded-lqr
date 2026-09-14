"""Annex 04's generated sections: §1 Problem, §3 Protocol, §4 Gates, §5, §7.

Marked **generated** in the annex, and the reason is P5: prose that is written
by hand drifts from the study it describes, and no review catches it because
both halves are plausible. These functions render from the specification and
the store, so the only way for §3 to be wrong is for the study to be wrong.

They return markdown **strings** and never print. `cli/app.py` is the one
module on this surface allowed to print, and a notebook displays a value
anyway — returning text also makes both sections testable against what they
claim rather than against captured output.
"""

from __future__ import annotations

import base64
from collections.abc import Iterable, Mapping, Sequence
from enum import Enum
from typing import Any

import numpy as np

from ..analysis.gates import GateStatus
from ..models.guards import require_linear_quadratic
from ..spec.study import StudyPoint
from .errors import UnknownArtifactError
from .loading import LoadedStudy
from .resolution import Resolution

#: How many significant figures a rendered matrix entry carries. Four, because
#: §1 exists so a reader can recognise the instance, not so they can reproduce
#: it bit-for-bit — that is the frozen `.npz`'s job (D19), and the `ProblemID`
#: in the same table is what ties the two together.
ENTRY_FORMAT = "{:.4f}"

#: Above how many entries a matrix stops being rendered entry by entry, per
#: Annex 04 §1.2 as corrected 2026-08-21. A hundred is a 10x10 block, which is
#: the largest a reader can still find a given entry in; past it the section
#: stops stating the instance and starts burying the eight rows above it (the
#: campaign's n = 100 plant renders 222,806 characters, 219,000 of them A and
#: B). The threshold is a property of reading and so lives here, not in any
#: study's declaration.
MAX_RENDERED_ENTRIES = 100

#: The artifact §5 embeds. The `.pdf` is the publication deliverable and does
#: not render in a browser; the `.png` is the same figure and does.
DISPLAY_SUFFIX = "png"

#: The figure's own tidy table, which is where §5's `n` comes from. Annex 03
#: §B.1.1: an analysis *replaces in place* and a rendered figure does not, so
#: reading `n` off the analysis would caption an image with the numbers it
#: would have if it were re-rendered today.
FIGURE_DATA_SUFFIX = "data.parquet"


def _rows(pairs: Iterable[tuple[str, object]]) -> list[str]:
    return [f"| {name} | {value} |" for name, value in pairs]


def _seeds(seeds: Sequence[int]) -> str:
    """The seed count *and* the values.

    `", ".join(seeds)` renders a single-seed study as the bare `0`, and a table
    row reading `training seeds | 0` says *zero seeds* to anyone who has not
    read the code. Found by rendering the real NB04 study, where the count is
    also the thing a reader is checking — one seed means no across-seed
    interval anywhere on the page (Annex 03 §A.3).
    """
    plural = "" if len(seeds) == 1 else "s"
    return f"{len(seeds)} seed{plural} ({', '.join(str(s) for s in seeds)})"


#: How long a `str(value)` may be before an axis stops printing it. An axis
#: value is a table cell; past this it is a paragraph, and a multi-line one is
#: not a cell at all.
MAX_AXIS_VALUE = 40


def _axis_key(value: Any) -> tuple[str, Any]:
    """A hashable stand-in for one declared axis value.

    A spec-valued axis carries whole `ProblemSpec`s, which hold numpy arrays:
    unhashable, and with an `__eq__` that is not a boolean. Its identifier is
    derived from exactly the content that distinguishes two of them, which is
    what makes it the right key rather than a convenience.
    """
    identifier = getattr(value, "problem_id", None)
    if identifier is not None:
        return ("problem", str(identifier))
    try:
        hash(value)
    except TypeError:
        return ("repr", repr(value))
    return ("value", value)


class _AxisNames:
    """What a reader is shown for each swept value, joined by position.

    `SweepAxis.labels` exists because "`str(ProblemSpec)` is a repr and a
    filename would name a category `..._rotA30` in a paper" (Annex 01 §2.5.1),
    and the join to a value is **position**. Until 2026-08-21 neither §3 nor §7
    performed that join: both interpolated the value, so every Figure-2
    document rendered a quarter of a megabyte of array repr into what was
    meant to be a table cell — 224,778 characters for §3 alone, with embedded
    newlines that end the table at its first row. Found by rendering it.
    """

    def __init__(self, study: Any) -> None:
        self._named: dict[str, dict[tuple[str, Any], str]] = {}
        self._paths = [axis.path for axis in study.sweep]
        for axis in study.sweep:
            if not axis.labels:
                continue
            self._named[axis.path] = {
                _axis_key(value): label
                for value, label in zip(axis.values, axis.labels, strict=True)
            }

    def title(self, path: str) -> str:
        """The shortest suffix of an axis path that is unique among them.

        A leaf alone reads best and is what §7 has always printed — until a
        study sweeps two problems, where `evaluation.problem` and
        `training.problem` both leaf to `problem` and one column silently
        interleaves two axes. The shortest *unique* suffix keeps the common
        case short and separates the case that needs separating, without a
        rule that has to be revisited when a third axis appears.
        """
        parts = path.split(".")
        for depth in range(1, len(parts) + 1):
            suffix = ".".join(parts[-depth:])
            if sum(other.endswith(suffix) for other in self._paths) == 1:
                return suffix
        return path

    def declares(self, path: str) -> bool:
        """Whether this axis names its values for a reader at all.

        An unlabelled spec-valued axis has no reader-facing name, and inventing
        one from the identifier would put a hash where a table promises a
        category. §1's instance table asks this before quoting a label rather
        than after, so the identifier column is not restated as a name.
        """
        return path in self._named

    def of(self, path: str, value: Any) -> str:
        """The declared label, or the most specific readable stand-in."""
        if value is None:
            # A real declared value: the sweep's "absent" position, which the
            # world axis uses for "no world". Rendering it as `None` states a
            # Python object where the study states an absence.
            return "—"
        declared = self._named.get(path, {}).get(_axis_key(value))
        if declared is not None:
            return declared
        identifier = getattr(value, "problem_id", None)
        if identifier is not None:
            # Unlabelled and spec-valued. The identifier is unique and short;
            # the repr is neither. Annex 03 §A.6.1 refuses this axis in an
            # analysis, and a section that dumped the repr instead of saying
            # so would be the worse failure.
            return f"`{identifier}`"
        text = str(value)
        if "\n" in text or len(text) > MAX_AXIS_VALUE:
            return f"`{type(value).__name__}`"
        return text


def _axis_label(point: StudyPoint, names: _AxisNames) -> str:
    """The axis values that placed this point, as a reader would say them."""
    return (
        ", ".join(
            f"{names.title(path)}={names.of(path, value)}"
            for path, value in point.axis_values.items()
        )
        or "—"
    )


def _contender_table(points: Sequence[StudyPoint], names: _AxisNames) -> list[str]:
    """One row per contender, with the axis values it was expanded over."""
    seen: dict[str, tuple[str, dict[str, list[str]]]] = {}
    for point in points:
        label = point.contender.resolved_label
        family, by_axis = seen.setdefault(label, (point.contender.family, {}))
        for path, value in point.axis_values.items():
            shown = names.of(path, value)
            values = by_axis.setdefault(path, [])
            if shown not in values:
                values.append(shown)
    lines = ["| contender | family | role | axis values |", "|---|---|---|---|"]
    for label, (family, by_axis) in seen.items():
        role = next(
            p.contender.role.value
            for p in points
            if p.contender.resolved_label == label
        )
        # Grouped by axis, and named. A flat list interleaves two axes into one
        # unreadable run as soon as a study sweeps more than one — measured on
        # the world-trained document, whose cell read `0, —, 5, <id>, 10, <id>`.
        spread = (
            "; ".join(
                f"{names.title(path)} = {', '.join(values)}"
                for path, values in by_axis.items()
            )
            or "—"
        )
        lines.append(f"| `{label}` | `{family}` | {role} | {spread} |")
    return lines


def _identity_scale(slice_: Any) -> float | None:
    """The scalar `c` when the matrix is exactly `c * I`, else `None`.

    Exact equality, not `allclose`: this branch states a *closed form* rather
    than an approximation, and a matrix within a tolerance of `c * I` is not
    `c * I`. Every cost matrix in this project's frozen plants passes exactly,
    at every dimension, which is what makes the closed form worth having.
    """
    rows, columns = slice_.shape[-2], slice_.shape[-1]
    if rows != columns:
        return None
    scale = float(slice_[0, 0])
    return scale if np.array_equal(slice_, scale * np.eye(rows)) else None


def _summarised(label: str, slice_: Any) -> list[str]:
    """A matrix too large to read, stated by what identifies it.

    Annex 04 §1.2 as corrected 2026-08-21: the concrete instance is what a
    reader can *obtain*, and the frozen file named by the `ProblemID` in the
    same table is that instance exactly. What belongs here is therefore the
    handful of invariants a reader can check the file against -- the spectral
    radius above all, since it is the one quantity the authoring step asserts
    and the one a reader would recompute first.
    """
    rows, columns = slice_.shape[-2], slice_.shape[-1]
    invariants = [
        f"largest entry ${np.abs(slice_).max():.4f}$ in magnitude",
        f"Frobenius norm ${np.linalg.norm(slice_):.4f}$",
    ]
    if rows == columns:
        radius = float(np.max(np.abs(np.linalg.eigvals(slice_))))
        invariants.insert(0, f"spectral radius ${radius:.12g}$")
    return [
        f"${label}$ is ${rows} \\times {columns}$ and is stated by the frozen "
        f"instance the identifier above names, not entry by entry: "
        f"{', '.join(invariants)}.",
        "",
    ]


#: How a plan's own type names the schedule it runs, in the words Annex 04 §3
#: asks for. Two entries because the project has two plans; an unmapped one
#: renders its declared type rather than guessing, since "end-to-end" asserted
#: of a schedule nobody checked is exactly the drift §3 is generated to prevent.
SCHEDULES = {
    "TrainingPlan": "end-to-end",
    "LayerwiseTrainingPlan": "layer-wise",
}

#: What a row says where there is nothing to say.
ABSENT = "—"


def _declared(value: Any) -> str:
    """One declared value, written the way the document declares it.

    `repr` for numbers, because a step size is identity-bearing: the campaign's
    `1/(2L)` literal differs from a rounded copy of itself in the `ModelID` it
    produces, and a protocol section that prints `0.000361` has stated a
    different experiment from the one that ran.
    """
    if value is None:
        return "none"
    if isinstance(value, bool):
        # `true`/`false`, which is how the document declares it. `repr` would
        # print Python's spelling of a value the reader never wrote.
        return f"`{str(value).lower()}`"
    if isinstance(value, Enum):
        return f"`{value.value}`"
    if isinstance(value, str):
        return f"`{value}`"
    if hasattr(value, "get_signature"):
        # A nested specification: name it rather than dumping it. Its own
        # fields reach identity through the signature, not through this table.
        return f"`{type(value).__name__}`"
    return f"`{value!r}`"


def _keyed(items: Iterable[tuple[str, Any]]) -> str:
    """A `key = value` list, keys in the reader's words."""
    rendered = ", ".join(
        f"{key.replace('_', ' ')} = {_declared(value)}" for key, value in items
    )
    return rendered or ABSENT


def _plan_columns(plan: Any) -> tuple[str, str, str]:
    """One contender's fitting, as `(schedule, optimiser, budget)`.

    Read off the plan's **signature** rather than its fields, so a plan this
    module has never seen still renders every quantity that reaches identity —
    which is the only defensible definition of "the training was stated".
    """
    if plan is None or not hasattr(plan, "get_signature"):
        return "*not fitted*", ABSENT, ABSENT
    signature = dict(plan.get_signature())
    kind = str(signature.pop("type", ""))
    optimizer = dict(signature.pop("optimizer", {}) or {})
    optimizer.pop("type", None)
    name = optimizer.pop("name", None)
    hyperparameters = dict(optimizer.pop("hyperparameters", {}) or {})
    rate = optimizer.pop("learning_rate", None)
    clauses = [_declared(name)] if name is not None else []
    if rate is not None:
        clauses.append(f"lr = {_declared(rate)}")
    clauses += [
        f"{key.replace('_', ' ')} = {_declared(value)}"
        for key, value in sorted({**optimizer, **hyperparameters}.items())
    ]
    return (
        SCHEDULES.get(kind, f"`{kind}`"),
        ", ".join(clauses) or ABSENT,
        _keyed(sorted(signature.items())),
    )


def _fitting_table(points: Sequence[StudyPoint]) -> list[str]:
    """One row per contender: how it is fitted, and how it is constructed.

    Annex 04 §1.2 makes §3 "contenders, **training**, evaluation, statistics,
    seeds", and until this table the section named the contenders and stated no
    training at all — no schedule, no optimiser, no budget, no initialisation.
    Two of the campaign's own figures differ in exactly those quantities and in
    nothing else the section printed, so §3 rendered them identically.

    The plan is read from the **materialised point**, never from the study's
    default: a contender declaring its own keeps it, so the default governs
    only whoever did not, and it is the effective plan that signs the
    `ModelID`. The swept keys are left out because the contender table above
    already carries them on its own axis, and stating an axis twice invites the
    two statements to disagree.
    """
    swept = {path.rsplit(".", 1)[-1] for point in points for path in point.axis_values}
    lines = [
        "| contender | fitting | optimiser | budget | construction |",
        "|---|---|---|---|---|",
    ]
    seen: set[str] = set()
    for point in points:
        label = point.contender.resolved_label
        if label in seen:
            continue
        seen.add(label)
        config = dict(point.contender.config)
        schedule, optimiser, budget = _plan_columns(config.pop("plan", None))
        construction = _keyed(
            (key, value) for key, value in sorted(config.items()) if key not in swept
        )
        lines.append(
            f"| `{label}` | {schedule} | {optimiser} | {budget} | {construction} |"
        )
    return lines


def _matrix(name: str, matrix: Any) -> list[str]:
    """One matrix in the most specific form that is still readable.

    A time-stacked matrix renders its **first slice** and says so in the label,
    because rendering fifty of them answers no question a reader has; whether
    the remaining slices agree with it is reported separately, as a measurement.

    Three forms, per Annex 04 §1.2 as corrected 2026-08-21. A scalar multiple
    of the identity is stated in closed form, which is exact and complete at
    every size and is what every frozen plant's `Q` and `R` are. A matrix small
    enough to read is stated entry by entry. Anything larger is stated by its
    shape and its invariants, because at `MAX_RENDERED_ENTRIES` the entries stop
    being a specification a reader meets and become one that buries the table.
    """
    array = np.asarray(matrix)
    slice_ = array if array.ndim == 2 else array[0]
    label = name if array.ndim == 2 else f"{name}_0"

    scale = _identity_scale(slice_)
    if scale is not None:
        order = slice_.shape[-1]
        factor = "" if scale == 1.0 else f"{scale:g} \\, "
        return [f"$${label} = {factor}I_{{{order}}}$$", ""]

    if slice_.size > MAX_RENDERED_ENTRIES:
        return _summarised(label, slice_)

    rows = r" \\ ".join(
        " & ".join(ENTRY_FORMAT.format(float(value)) for value in row) for row in slice_
    )
    return [f"$${label} = \\begin{{bmatrix}} {rows} \\end{{bmatrix}}$$", ""]


def _varies_over_time(matrix: Any, *, ignore_last: bool = False) -> bool:
    """Whether a time-stacked matrix actually differs across its slices.

    Measured rather than inferred from the shape: `ProblemData` requires `Q`
    and `R` time-stacked whatever the study means, so a shape of `(N, m, m)`
    says nothing at all about whether the cost varies. `ignore_last` drops the
    terminal slice, which is a different quantity rather than a later value of
    the same one.
    """
    array = np.asarray(matrix)
    if array.ndim == 2 or array.shape[0] < 2:
        return False
    body = array[:-1] if ignore_last else array
    return bool(body.shape[0] > 1 and not np.allclose(body, body[0]))


def render_problem(loaded: LoadedStudy) -> str:
    """Annex 04 §1's parameter table — the concrete instance, generated.

    §1.2 requires "the named family *and* the concrete instance, per the
    author's requirement that a name alone is not a specification", so the
    matrices are here and not only their shapes. §1.3 requires that every
    convention be "stated explicitly rather than inherited silently", so the
    terminal and averaging conventions are **read off the built problem**. That
    distinction is not academic for this study: `ProblemData` requires `Q` over
    `N + 1` slices and `build` currently constructs a cost that does not score
    the terminal one, so `Q_N` is stored and takes no part in any number the
    study reports. A hand-written §1 could say either and be believed.

    Args:
        loaded: The resolved study.

    Returns:
        The section, generated.
    """
    spec = loaded.study.problem
    data = spec.data
    # The project's own shared narrowing, rather than a local `isinstance`:
    # a problem this grammar can express is linear-quadratic by construction,
    # and the one guard that says so is the one every controller already uses.
    _, cost = require_linear_quadratic(spec.build())
    conventions = cost.conventions
    bound = data.control_bound

    lines = [
        # A **sub**heading, and the only generated section that carries one.
        # §1 is the one part Annex 04 §1.2 splits — "prose is authored; the
        # parameter table is generated" — so the section heading belongs to the
        # authored half and this is the generated half announcing itself.
        "### The instance",
        "",
        f"*Generated from `{loaded.document.name}`, and from the frozen "
        "matrices it names.*",
        "",
        "| | |",
        "|---|---|",
        *_rows(
            [
                ("ProblemID", f"`{spec.problem_id}`"),
                ("state dimension $n$", data.state_dim),
                ("control dimension $m$", data.control_dim),
                ("horizon $N$", data.horizon),
                (
                    "constraint set",
                    "unconstrained"
                    if bound is None
                    else f"$\\lVert u_t \\rVert_\\infty \\le {bound:g}$",
                ),
                (
                    "dynamics",
                    "time-varying"
                    if _varies_over_time(data.system["A"])
                    or _varies_over_time(data.system["B"])
                    else "time-invariant",
                ),
                (
                    "running cost",
                    "time-varying"
                    if _varies_over_time(data.cost["Q"], ignore_last=True)
                    or _varies_over_time(data.cost["R"])
                    else "time-invariant",
                ),
                (
                    "terminal cost $x_N^\\top Q_N x_N$",
                    "scored"
                    if conventions.include_terminal_cost
                    else "stored, **not scored**",
                ),
                (
                    "objective",
                    "time-averaged over the horizon"
                    if conventions.is_time_averaged
                    else "cumulative over the horizon",
                ),
            ]
        ),
        "",
    ]
    for group in ("system", "cost"):
        for name in sorted(getattr(data, group)):
            lines += _matrix(name, getattr(data, group)[name])
    if spec.provenance is not None:
        parameters = ", ".join(
            f"{k} = {v}" for k, v in sorted(spec.provenance.params.items())
        )
        lines += [
            f"*Authored by `{spec.provenance.generator}` "
            f"({parameters}); the record takes no part in the identifier above.*"
        ]
    return "\n".join(lines + _swept_instances(loaded))


#: The two axes along which a study may bind a problem other than its declared
#: one, with what that makes the instance. Named rather than matched on a
#: suffix: `evaluation.problem` and `training.problem` are the grammar's two
#: paths, and a third would be a grammar change rather than a new spelling.
PROBLEM_AXES = {
    "training.problem": "trained on",
    "evaluation.problem": "scored on",
}


def _swept_instances(loaded: LoadedStudy) -> list[str]:
    """Every *other* instance the study binds, and how far it is from this one.

    A study that sweeps a problem has more than one concrete instance, and §1
    stating only the declared one states the plant the mismatch experiments are
    a departure *from* — never the plants they measure. The two distances are
    the point: the campaign's rotation moves `A` and leaves `B` alone, which is
    what separates a genuine mismatch from a similarity transform, and this is
    that claim as a measurement rather than as prose.

    Returns:
        The section's lines, or nothing at all when the study binds one
        instance — where a table of one row would imply a sweep that is not
        there.
    """
    study = loaded.study
    names = _AxisNames(study)
    reference = study.problem
    nominal_a = np.asarray(reference.data.system["A"])
    nominal_b = np.asarray(reference.data.system["B"])
    found: dict[str, dict[str, Any]] = {}
    for point in study.materialise():
        # The training WORLD, which is the training specification's problem and
        # not the point's own: the world axis moves the plant the trajectories
        # come from while the contender's declared model stays nominal, so
        # reading the point's problem would report the nominal plant as every
        # world the study ever trained in. Found by rendering the world-trained
        # document, whose first row then listed six labels at once.
        world = getattr(point.training, "problem", None)
        for path, used_as in PROBLEM_AXES.items():
            spec = point.evaluation.problem if path == "evaluation.problem" else world
            if spec is None:
                spec = point.problem
            entry = found.setdefault(
                str(spec.problem_id), {"spec": spec, "used": set(), "labels": []}
            )
            entry["used"].add(used_as)
            if path in point.axis_values and names.declares(path):
                shown = names.of(path, point.axis_values[path])
                if shown and shown not in entry["labels"]:
                    entry["labels"].append(shown)
    if len(found) < 2:
        return []

    lines = [
        "",
        "### The instances it runs on",
        "",
        "*The instance above is the one the study declares; these are the ones "
        "its own axes bind. Both distances are measured against it.*",
        "",
        "| axis value | used as | ProblemID | $\\rho(A)$ | "
        "$\\lVert A - A_0 \\rVert_F$ | $\\lVert B - B_0 \\rVert_F$ |",
        "|---|---|---|---|---|---|",
    ]
    for identifier, entry in found.items():
        data = entry["spec"].data
        a = np.asarray(data.system["A"])
        b = np.asarray(data.system["B"])
        radius = float(np.max(np.abs(np.linalg.eigvals(a if a.ndim == 2 else a[0]))))
        # Declaration order, not the order the points happened to arrive in:
        # two rows reading "trained on, scored on" and "scored on, trained on"
        # say the same thing and invite a reader to look for a difference.
        used = ", ".join(
            role for role in PROBLEM_AXES.values() if role in entry["used"]
        )
        lines.append(
            f"| {', '.join(entry['labels']) or '—'} | {used} "
            f"| `{identifier}` | {radius:.12g} "
            f"| {np.linalg.norm(a - nominal_a):.4f} "
            f"| {np.linalg.norm(b - nominal_b):.4f} |"
        )
    return lines


def _span(column: Any) -> str:
    """One count when every row agrees, and the range when they do not.

    `max()` alone would summarise a table in which one contender was scored on
    fewer seeds than the others as though it had been scored on all of them —
    the caption's whole job is to say what is behind the picture, and hiding an
    inhomogeneity is the one way it can be actively misleading.
    """
    low, high = int(column.min()), int(column.max())
    return str(low) if low == high else f"{low}–{high}"


def _figure_data(resolution: Resolution, figure_id: str) -> Any:
    """The tidy table the *rendered* figure carries. See `FIGURE_DATA_SUFFIX`."""
    import pandas as pd

    return pd.read_parquet(resolution.figures[figure_id][FIGURE_DATA_SUFFIX])


def render_figure(
    resolution: Resolution, figure_id: str, *, claim: str | None = None
) -> str:
    """Annex 04 §5 — one stored figure, with a caption reporting $n$.

    Deliberately **not** a registry and not keyed by analysis kind. Annex 04
    §1.5's block catalogue is Phase C, and its whole premise is that it cannot
    be designed from one notebook; this is the un-named call it will generalise.

    The artifact is embedded rather than linked, so the executed notebook is
    one file that carries its own figures — a relative path from a notebook to
    a store is a path that breaks the first time either moves.

    Args:
        resolution: A verified study.
        figure_id: The figure's declared id.
        claim: The scientific claim this figure supports, authored. §1.2 asks a
            caption for the claim *and* for `n`; only the second is derivable.

    Returns:
        The section, generated.

    Raises:
        UnknownArtifactError: If the study declares no such figure.
    """
    if figure_id not in resolution.figures:
        declared = ", ".join(sorted(resolution.figures)) or "none"
        raise UnknownArtifactError(
            f"study {resolution.loaded.study.id!r} declares no figure "
            f"{figure_id!r}; it declares {declared}"
        )
    spec = next(s for s in resolution.loaded.study.figures if s.id == figure_id)
    table = _figure_data(resolution, figure_id)
    payload = base64.b64encode(
        resolution.figures[figure_id][DISPLAY_SUFFIX].read_bytes()
    ).decode("ascii")

    seeds = _span(table["n_seeds"])
    trajectories = _span(table["n_trajectories"])
    points = len(table)
    title = str(spec.config.get("title", figure_id))

    caption = [
        f"**Figure — {title}.**",
        *([claim] if claim else []),
        f"{points} plotted point(s) over {len(set(table['contender']))} "
        f"contender(s); {seeds} training seed(s) and {trajectories} evaluation "
        "trajectories behind each plotted point.",
        # The store's path is deliberately absent. It is a fact about the
        # machine that rendered the page, it differs on every machine, and §7
        # already carries what identifies the results — a caption is part of a
        # publication artifact and should carry nothing that cannot travel.
        f"Rendered from the *{spec.source}* table.",
    ]
    return "\n".join(
        [
            f"![{title}](data:image/png;base64,{payload})",
            "",
            " ".join(caption),
        ]
    )


def render_composed_figure(
    parts: Sequence[Resolution],
    figure_id: str,
    *,
    claim: str | None = None,
    title: str | None = None,
) -> str:
    """One stored figure that no single study declares, with its caption.

    A figure assembled from several studies belongs to none of them, so
    `render_figure` cannot reach it: it looks the figure up among the ones its
    study declares, and a composed figure is declared by nothing. The exact-
    convex paper's Figure 2 is of this kind -- three mismatch conditions, one
    panelled figure -- and leaving it unrenderable would have meant a paper
    figure a reviewer cannot see.

    **Addressed by what composes it, never by an identifier typed out.** The
    composite id is derived here from the parts the caller resolved, which is
    the same derivation the renderer used when it wrote the figure. So the
    figure is found only if it was built from exactly these studies at exactly
    these tiers; change any one of them and the address moves and this refuses,
    rather than displaying a picture of something else.

    Args:
        parts: The resolved studies the figure is composed from, in any order.
        figure_id: The composed figure's id.
        claim: The scientific claim it supports, authored.
        title: What to call it in the caption. Authored, because a composed
            figure has no study to carry one -- a declared figure takes its
            title from its own specification, and there is no such document
            here. Defaults to `figure_id`, which reads as an identifier and is
            the honest fallback rather than a good caption.

    Returns:
        The section, generated, with the image embedded rather than linked.

    Raises:
        UnknownArtifactError: If no such figure is stored for those parts,
            naming what it looked for.
    """
    from ..store.ids import composite_study_id
    from ..store.location import default_store
    from ..store.study_artifacts import StudyArtifactStore
    from .resolution import FIGURE_KEYS, _figure_artifacts

    if not parts:
        raise UnknownArtifactError(
            f"figure {figure_id!r} is composed of nothing; pass the resolved "
            "studies it is drawn from"
        )
    # Sorted for the REFUSAL's sake, not for the address: the composite
    # identity sorts its own sources by contract, so this changes no id and
    # survives its own mutant. What it buys is an error message that lists the
    # parts in a stable order instead of in whatever order a caller passed.
    composed = sorted(str(part.loaded.study.study_id) for part in parts)
    study_id = str(composite_study_id(composed, kind=figure_id))
    root = StudyArtifactStore(default_store()).figures_root(study_id)
    artifacts = _figure_artifacts(root, figure_id)
    if set(artifacts) != FIGURE_KEYS:
        raise UnknownArtifactError(
            f"no composed figure {figure_id!r} is stored for the studies "
            f"{', '.join(composed)}. It would be filed under {study_id}, and "
            "that is derived from the parts -- so a figure built from a "
            "different set, or at a different tier, is a different figure and "
            "not this one."
        )
    payload = base64.b64encode(artifacts[DISPLAY_SUFFIX].read_bytes()).decode("ascii")

    import pandas as pd

    table = pd.read_parquet(artifacts["data.parquet"])
    seeds = _span(table["n_seeds"])
    trajectories = _span(table["n_trajectories"])
    caption = [
        f"**Figure — {title or figure_id}.**",
        *([claim] if claim else []),
        f"{len(table)} plotted point(s) over "
        f"{len(set(table['contender']))} contender(s); {seeds} training "
        f"seed(s) and {trajectories} evaluation trajectories behind each "
        "plotted point.",
        f"Composed from {len(composed)} studies.",
    ]
    return "\n".join(
        [
            f"![{title or figure_id}](data:image/png;base64,{payload})",
            "",
            " ".join(caption),
        ]
    )


#: Signature keys a distribution shares with the rest of §3's table, or that
#: describe the shape rather than the law. Excluded so the row states what only
#: it can state; `state_dim` and `horizon` in particular are the *problem's*,
#: and §1 has already reported them from the frozen matrices.
SHAPE_KEYS = frozenset({"type", "state_dim", "horizon", "batch_size"})


def _distribution(kind: str, params: Mapping[str, Any]) -> str:
    """A sampler as its family **and** its parameters.

    A row reading `gaussian` names a family and specifies nothing — the exact
    shape Annex 04 §1.2 rules out for the problem ("a name alone is not a
    specification") and no weaker here: two studies differing only in their
    process-noise scale are two different experiments, and a protocol section
    that renders them identically is prose drift with extra steps.
    """
    if not params:
        return f"`{kind}`"
    spelled = ", ".join(f"{name} = {params[name]:g}" for name in sorted(params))
    return f"`{kind}` ({spelled})"


def _batch_distribution(batch: Any) -> str:
    """The evaluation sampler, read off its signature.

    Its parameters are attributes of a built object rather than an open
    mapping, and the signature is the one description of it that is guaranteed
    complete — it is what `MeasurementID` is derived from, so a parameter this
    row could omit is one that could change a number without changing the row.
    """
    if batch is None:
        return "—"
    signature = batch.get_signature()
    kind = str(signature.get("type", "—"))
    return _distribution(
        kind,
        {
            name: value
            for name, value in signature.items()
            if name not in SHAPE_KEYS and isinstance(value, int | float)
        },
    )


def render_protocol(loaded: LoadedStudy) -> str:
    """Annex 04 §3 — contenders, training, evaluation, seeds — as markdown.

    Two tables and a summary. The first names the cast and the axis each
    member was expanded over; the second states how each is **fitted** and how
    it is **constructed**, which §1.2 requires of this section and which it did
    not carry until 2026-08-21; the third states the distributions, the effort
    and the substrate.

    Args:
        loaded: The resolved study.

    Returns:
        The section, generated. Deterministic for a given study: two calls
        return the same string, and two studies differing in any declared
        quantity return different ones.
    """
    study = loaded.study
    points = study.materialise()
    training, evaluation = study.training, study.evaluation
    protocol = evaluation.protocol
    batch = getattr(protocol, "batch_spec", None)

    lines = [
        "## Protocol",
        "",
        f"*Generated from `{loaded.document.name}` at tier `{loaded.tier}`.*",
        "",
        *_contender_table(points, _AxisNames(study)),
        "",
        *_fitting_table(points),
        "",
        "| | |",
        "|---|---|",
        *_rows(
            [
                ("study", f"`{study.id}`"),
                ("tier", f"`{loaded.tier}`"),
                ("points", len(points)),
                ("training seeds", _seeds(training.seeds)),
                (
                    "training distribution",
                    _distribution(training.data.kind, training.data.params),
                ),
                ("effective batch", training.batch.effective_size),
                (
                    "compute",
                    f"{training.ctx.backend} / {training.ctx.device} / {training.ctx.precision}",
                ),
                ("evaluation batches", getattr(protocol, "n_batches", "—")),
                ("trajectories per batch", getattr(batch, "batch_size", "—")),
                ("evaluation distribution", _batch_distribution(batch)),
                # What the controller is *told* when the scored plant is not
                # the fitted one. It is the whole of the campaign's blind /
                # informed distinction, it signs into every MeasurementID off
                # its default, and §3 stated it nowhere until 2026-08-21.
                ("controller rehosting", f"`{evaluation.rehost}`"),
                ("metrics", ", ".join(sorted(evaluation.metrics))),
            ]
        ),
    ]
    if loaded.overrides:
        lines += [
            "",
            "Per-invocation overrides:",
            "",
            *(f"- `{path}` = `{value}`" for path, value in loaded.overrides.items()),
        ]
    return "\n".join(lines)


def render_provenance(resolution: Resolution) -> str:
    """Annex 04 §7 — every identifier behind every number on the page.

    This section exists so that a figure in the thesis can be traced to the
    exact models that produced it, years later, without archaeology. It
    therefore lists **every** model and measurement the study resolves to, not
    a representative sample: a provenance section that summarised would be one
    a reader could not use for the one thing it is for.

    Args:
        resolution: A verified study.

    Returns:
        The section, generated.
    """
    loaded = resolution.loaded
    study = loaded.study
    points = study.materialise()
    names = _AxisNames(study)
    stamp = _stored_stamp(resolution)

    lines = [
        "## Provenance",
        "",
        f"*Generated from the store at tier `{loaded.tier}`.*",
        "",
        "| | |",
        "|---|---|",
        *_rows(
            [
                ("study", f"`{study.id}`"),
                ("StudyID", f"`{study.study_id}`"),
                ("ProblemID", f"`{study.problem.problem_id}`"),
                ("tier", f"`{loaded.tier}`"),
                ("points", len(points)),
                ("stamp", f"`{stamp}`"),
                ("offline compute", _offline(resolution)),
                # Named and stated absent rather than omitted, for as long as
                # it *is* absent — a row that simply were not here would be
                # read as "measured and negligible", the same distinction
                # `GateStatus` keeps four ways rather than three.
                ("online compute", _online(resolution)),
                ("command", f"`{loaded.command('run')}`"),
            ]
        ),
        "",
        # The axis column is load-bearing rather than decorative: without it
        # the eight depth points of one contender render as eight identical
        # rows, and §7's whole purpose -- tracing a figure back to the models
        # that produced it -- is defeated by its own table.
        "| contender | axis | seed | ModelID | MeasurementID |",
        "|---|---|---|---|---|",
    ]
    for point in points:
        lines.append(
            f"| `{point.contender.resolved_label}` "
            f"| {_axis_label(point, names)} "
            f"| {point.seed} | `{point.model_id}` | `{point.measurement_id}` |"
        )
    if resolution.analyses:
        lines += ["", "Analyses: " + ", ".join(f"`{n}`" for n in resolution.analyses)]
    if resolution.figures:
        lines += ["Figures: " + ", ".join(f"`{n}`" for n in resolution.figures)]
    return "\n".join(lines)


def _offline(resolution: Resolution) -> str:
    """What the study cost to synthesise, summed off the model records.

    Annex 04 §7 asks for "total offline and online compute" and Phase A
    reported neither; this is the half that is recorded. Read from each
    **record's own** provenance rather than from the store index, because
    `mbl store reindex` exists precisely because the index can be behind the
    store, and a provenance section that could disagree with what it describes
    is the one section that must not.
    """
    from ..store.content_store import ModelStore

    models = ModelStore(resolution.store)
    seconds, vram = 0.0, 0.0
    for point in resolution.loaded.study.materialise():
        provenance = models.get(point.model_id).provenance
        seconds += float(provenance.get("wall_time_s") or 0.0)
        vram = max(vram, float(provenance.get("peak_vram_mb") or 0.0))
    return f"{seconds:.1f} s over {resolution.completeness.points} point(s), peak {vram:.0f} MiB VRAM"


def _online(resolution: Resolution) -> str:
    """What the study cost to score, summed off the measurement records.

    The other half of Annex 04 §7's compute row, and the one this section could
    not answer until the evaluation pass was timed (Annex 02 §2.2).

    **Three states, never two.** Every record on this path predates the timer or
    postdates it, and a study is extended as often as it is re-run — so a store
    holding some of each is the normal case, not a defensive one. Reporting a
    partial sum as if it covered everything would understate the cost of the
    study by exactly the part nobody measured, silently; reporting it as
    unrecorded would throw away what was measured.
    """
    from ..store.content_store import MeasurementStore

    measurements = MeasurementStore(resolution.store)
    points = resolution.loaded.study.materialise()
    seconds, vram, timed = 0.0, 0.0, 0
    for point in points:
        provenance = measurements.get(point.measurement_id).spec.get("provenance", {})
        if not isinstance(provenance, Mapping) or "wall_time_s" not in provenance:
            continue
        timed += 1
        seconds += float(provenance.get("wall_time_s") or 0.0)
        vram = max(vram, float(provenance.get("peak_vram_mb") or 0.0))
    if not timed:
        return "*not recorded* (§8.6)"
    covered = (
        f"{len(points)} point(s)"
        if timed == len(points)
        else f"{timed} of {len(points)} point(s)"
    )
    return f"{seconds:.1f} s over {covered}, peak {vram:.0f} MiB VRAM"


def _stored_stamp(resolution: Resolution) -> str:
    """The producer's provenance stamp, read off a stored measurement.

    Off the **store** rather than off the running package: §7 says what
    produced these numbers, which may be an older version of this code than the
    one rendering the notebook — and a stamp taken from the interpreter would
    silently claim otherwise.
    """
    from ..store.content_store import MeasurementStore

    points = resolution.loaded.study.materialise()
    if not points:
        return ""
    spec = MeasurementStore(resolution.store).get(points[0].measurement_id).spec
    provenance = spec.get("provenance", {})
    return str(provenance.get("stamp", "")) if isinstance(provenance, Mapping) else ""


def render_gates(resolution: Resolution) -> str:
    """Annex 04 §4 — the declared gates, with their measured values.

    Every declared gate appears, including the ones this tier cannot decide.
    That is the section's whole point: a reader must be able to tell "checked
    and held" from "checked somewhere else" from "not checkable yet", and a
    table showing only the green ones is how a check nobody made becomes a
    check everybody believes.

    Nothing here raises. A gate that *failed* never reaches this function,
    because `resolve` refuses first — Annex 04 §4 requires a failed gate to
    stop the notebook, and stopping at the render step would mean the section
    is only as protective as the author's remembering to display it.

    Args:
        resolution: A verified study.

    Returns:
        The section, generated.
    """
    lines = [
        "## Pre-flight gates",
        "",
        f"*Generated from `{resolution.loaded.document.name}`. "
        "Every declared gate appears, decided or not.*",
        "",
        "| gate | when | verdict | measured | required |",
        "|---|---|---|---|---|",
    ]
    for outcome in resolution.gates:
        measured = "—" if outcome.measured is None else f"{outcome.measured:.4g}"
        required = "—" if outcome.threshold is None else f"{outcome.threshold:.4g}"
        lines.append(
            f"| `{outcome.kind.value}` | {outcome.stage.value} "
            f"| **{outcome.status.value}** | {measured} | {required} |"
        )
    if not resolution.gates:
        lines.append("| — | — | *this study declares none* | — | — |")
    undecided = [o for o in resolution.gates if o.status is GateStatus.NOT_EVALUABLE]
    if undecided:
        lines += ["", "Not evaluable here, and why:", ""]
        lines += [f"- `{o.kind.value}` — {o.detail}" for o in undecided]
    return "\n".join(lines)
