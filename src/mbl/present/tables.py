"""The presented table — a figure's values, typeset (Annex 03 §B.1.2).

"A curve answers *which*; it does not answer *by how much*. Every numerical
claim this project makes is eventually checked by someone reading a number off
it, and a reader who has to measure a figure with a ruler has been handed a
picture instead of a result."

Two objects now share the word "table". §A.6's **tidy table** is the data —
`<id>.data.parquet`, machine-read. The **presented table** is what a reader
meets, and it is a *rendering* of the tidy table in exactly the sense the PDF
is: same input, second output channel, **no computation of its own**. That is
the module's whole contract, and it is stated negatively on purpose — "a
presented table that computed anything would be a third statement about one
study, and the two that already exist disagree often enough".

So nothing here reduces, converts, re-aggregates or re-derives. A cell is a
frame value put through `format`. Where a number the table must show is not in
the frame, the fix belongs in the *analysis*, where the sidecar records it —
which is why §A.3.2 made `seeds` and `per_seed_aggregate` frame columns rather
than something the emitter reaches for.

**Two provenances, two escaping rules, and getting them the same way round is
the trap.** A column header comes from the data (`across_seed_spread`) and its
underscores are LaTeX specials that break compilation. A display name is
author-written LaTeX (`Unfolded-$\\alpha$+P`, and Annex 04 §1.3 is why) —
escaping it would print the markup instead of rendering it. Headers are
escaped; display names never are.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..spec.errors import SpecificationError

#: What a cell shows when the frame has no value there. §A.3.2: "the
#: distinction is carried by the table, which prints `—` and `0.000` and means
#: two different things" — an absent across-seed quantity (one seed) against a
#: measured zero (an analytic contender under common random numbers).
ABSENT_CELL = "—"

#: Columns a reader never meets unless the author asks for them by name.
#: `contender` is not here: it is shown, but as the DISPLAY name (clause 4).
#:
#: `role` and `axis_path` are plumbing the figure already expresses visually;
#: `axis_label` is `axis_value` a second time as a string, and printing both
#: would give one quantity two columns.
MACHINERY_COLUMNS = ("role", "axis_path", "axis_label")

#: Decimals a numeric cell is shown at when the declaration is silent. §B.1.2
#: clause 3 makes the displayed precision "a presentation decision and it is
#: declared"; this is the value a declaration that says nothing has declared.
DEFAULT_PRECISION = 3

#: The join key's column, and the one clause 4 renames.
LABEL_COLUMN = "contender"


def render_tables(
    frame: pd.DataFrame,
    *,
    declaration: Mapping[str, Any],
    display_names: Mapping[str, str],
    declarations: Mapping[str, Any],
) -> dict[str, str]:
    """One rendered table per registered format.

    Args:
        frame: `<id>.data.parquet` — the exact values plotted, and the only
            input (clause 1). Never an analysis table read a second time.
        declaration: The figure's `table` block: `columns`, `precision`,
            `caption`.
        display_names: `label -> what a reader is shown` (clause 4).
        declarations: §A.4's statistical declarations, frozen into
            `<id>.spec.json` at render time so a rebuild with no analysis in
            reach can still state them.

    Returns:
        `format name -> text`, keyed by `TABLE_FORMATS`.

    Raises:
        SpecificationError: If the declaration names a column the frame does
            not carry. Refused rather than skipped, for the same reason a
            figure selecting an absent series is: a table quietly missing the
            column its caption is about is the plausible-and-wrong output.
    """
    columns = _columns(frame, declaration)
    precision = _precision(declaration, columns)
    headers = [_header(column) for column in columns]
    body = [
        [
            _cell(row[column], column, precision[column], display_names)
            for column in columns
        ]
        for _, row in frame.iterrows()
    ]
    note = _note(declarations, declaration)
    return {
        name: formatter(headers, body, note)
        for name, formatter in TABLE_FORMATS.items()
    }


def _columns(frame: pd.DataFrame, declaration: Mapping[str, Any]) -> list[str]:
    declared = declaration.get("columns")
    if declared is None:
        return [column for column in frame.columns if column not in MACHINERY_COLUMNS]
    wanted = [str(name) for name in declared]
    missing = [name for name in wanted if name not in frame.columns]
    if missing:
        raise SpecificationError(
            f"the table asks for {', '.join(missing)}, which "
            f"`<id>.data.parquet` does not carry; available: "
            f"{', '.join(map(str, frame.columns))}. Anything the table must "
            "show has to be in the frame — obtaining it here would reach past "
            "the artifact the figure was drawn from (§B.1.2 clause 1)"
        )
    return wanted


def _precision(
    declaration: Mapping[str, Any], columns: Sequence[str]
) -> dict[str, int]:
    """Decimals per column, from `precision` as a number or as a table.

    One number for a whole table is the wrong granularity, and the tracked
    study shows why: at `precision = 4` an unrolling depth of J = 1 prints as
    `1.0000`. The axis is integer-valued and the cost is not, and both are
    "the frame's value formatted" — so the declaration is allowed to say so
    per column, with `default` covering the rest.

    This is not a knob standing in for a rule. §B.1.2 clause 3 already makes
    the displayed precision *a declared presentation decision*; a single
    number simply cannot express a decision the data has two of.

    Raises:
        SpecificationError: On a negative precision, or on a per-column entry
            naming a column the table does not show — a precision declared
            against a column nobody prints is a declaration that reaches
            nothing, which is the failure mode this project keeps finding.
    """
    declared = declaration.get("precision", DEFAULT_PRECISION)
    if not isinstance(declared, Mapping):
        return {column: _decimals(declared) for column in columns}
    default = _decimals(declared.get("default", DEFAULT_PRECISION))
    unknown = sorted(set(declared) - set(columns) - {"default"})
    if unknown:
        raise SpecificationError(
            f"the table declares a precision for {', '.join(unknown)}, which "
            f"it does not show; shown columns: {', '.join(columns)}. A "
            "precision against a column nobody prints takes no effect and "
            "reads like it did"
        )
    return {
        column: _decimals(declared[column]) if column in declared else default
        for column in columns
    }


def _decimals(value: Any) -> int:
    decimals = int(value)
    if decimals < 0:
        raise SpecificationError(
            f"the table declares precision {decimals}, which is not a number "
            "of decimals; §B.1.2 makes the displayed precision a declared "
            "presentation decision, and a negative one declares nothing"
        )
    return decimals


def _header(column: str) -> str:
    """A column's heading. Derived from the data, so it is escaped downstream."""
    return column.replace("_", " ")


def _cell(
    value: Any, column: str, precision: int, display_names: Mapping[str, str]
) -> str:
    """One frame value, formatted. Never recomputed (clause 3)."""
    if column == LABEL_COLUMN:
        # Clause 4: the join key is machinery and is never printed, so that a
        # reader moving between figure, table and caption never meets two
        # names for one contender.
        return str(display_names.get(str(value), value))
    if isinstance(value, (list, tuple, np.ndarray)):
        # A per-seed vector is one cell, not a reason to reshape the table:
        # §A.3.2 puts `per_seed_aggregate` in the frame precisely so the
        # reduction behind an error bar is reportable beside it.
        return ", ".join(_scalar(item, precision) for item in value)
    return _scalar(value, precision)


def _scalar(value: Any, precision: int) -> str:
    if value is None:
        return ABSENT_CELL
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not np.isfinite(number):
            # An absent value and a zero are different facts (§A.3.2), and
            # `nan` printed as text would read as a third thing entirely.
            return ABSENT_CELL
        return f"{number:.{precision}f}"
    return str(value)


def _note(declarations: Mapping[str, Any], declaration: Mapping[str, Any]) -> str:
    """§A.4's declarations, in one line under the table.

    "Emitted automatically from the specification, so they cannot be omitted or
    misstated" — so this is assembled from what the analysis declared, never
    from anything an author typed beside the figure.
    """
    parts: list[str] = []
    caption = declaration.get("caption")
    if caption:
        parts.append(str(caption))
    aggregate = declarations.get("aggregate")
    if aggregate:
        parts.append(f"aggregate: {aggregate}")
    interval = dict(declarations.get("interval") or {})
    if interval.get("kind"):
        level = interval.get("level")
        percent = "" if level is None else f" at {float(level) * 100:g}%"
        parts.append(f"interval: {interval['kind']}{percent} over {interval['over']}")
    spread_kind = declarations.get("across_seed_spread_kind")
    if spread_kind:
        parts.append(
            f"error bars: {spread_kind} over "
            f"{declarations.get('across_seed_spread_over', 'training seeds')}"
        )
    seeds = declarations.get("n_training_seeds")
    if seeds:
        parts.append(f"training seeds: {', '.join(str(count) for count in seeds)}")
    trajectories = declarations.get("n_evaluation_trajectories")
    if trajectories:
        # Labelled a TOTAL because that is what the sidecar holds — the sum
        # over every row of the table, not the count behind any one number.
        # Printing it bare invites a reader to take it for the n of the row
        # they are looking at, which on the tracked study is 4 423 680 against
        # a per-point 163 840.
        parts.append(f"evaluation trajectories (total): {trajectories}")
    return "; ".join(parts)


# -- the formats ------------------------------------------------------------


def _escape_latex(text: str) -> str:
    """A DERIVED string made safe for LaTeX.

    Applied to headers and never to display names: a header is data and its
    underscores are specials; a display name is author-written LaTeX and
    escaping it would print `$\\alpha$` rather than render it.
    """
    for character, replacement in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ):
        text = text.replace(character, replacement)
    return text


def _as_latex(headers: Sequence[str], body: Sequence[Sequence[str]], note: str) -> str:
    """A booktabs `tabular`, ready for `\\input` inside a float.

    A fragment rather than a `table` environment: §B.1.1's companion
    `figures.tex` is what places the float and writes the caption, so a float
    here would nest one inside another.
    """
    alignment = "l" + "r" * (len(headers) - 1) if headers else "l"
    lines = [
        "% generated by `mbl figure` — Annex 03 §B.1.2. Do not edit.",
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        " & ".join(_escape_latex(header) for header in headers) + r" \\",
        r"\midrule",
    ]
    lines.extend(" & ".join(row) + r" \\" for row in body)
    lines.append(r"\bottomrule")
    if note:
        lines.append(
            rf"\multicolumn{{{max(len(headers), 1)}}}{{l}}{{\footnotesize "
            rf"{_escape_latex(note)}}} \\"
        )
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def _as_markdown(
    headers: Sequence[str], body: Sequence[Sequence[str]], note: str
) -> str:
    """A GitHub-flavoured pipe table, for review and for notebook prose."""
    alignment = ["---"] + ["---:"] * (len(headers) - 1)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(alignment) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    if note:
        lines.extend(["", note])
    return "\n".join(lines) + "\n"


#: `format name -> formatter`. The suffix is `.table.<name>`, and §B.1 counts
#: "one file per registered table format" — so adding a format here adds an
#: artifact everywhere without another edit.
TABLE_FORMATS: dict[
    str, Callable[[Sequence[str], Sequence[Sequence[str]], str], str]
] = {
    "tex": _as_latex,
    "md": _as_markdown,
}
