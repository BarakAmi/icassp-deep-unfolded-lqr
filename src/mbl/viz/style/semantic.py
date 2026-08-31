"""Semantic color roles and the neutral series-style registry (REFACTOR_PLAN
v3, T4.c + N6).

Two responsibilities, both about *meaning* rather than geometry:

* `SemanticColors` — the BASELINE/OPTIMUM/CANDIDATE/AGGREGATE roles defined
  ONCE by reference into `theme.CATEGORICAL_PALETTE`, killing the previous
  "designated red" convention that was re-derived by literal palette index in
  three packages at once (`cost_analysis`, `convergence`, and the landscape
  toolkit's `palette.py`).
* The series-style registry — a neutral label -> color/linestyle resolver.
  The generic layer knows nothing about "analytic" or "unfolded" model
  families (M7): applications register their own family -> style mapping
  (`src/applications/styles.py`) at composition time, and unregistered
  labels fall back to deterministic palette cycling, so plotting an
  unanticipated series name never raises.
"""

from typing import Any

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .theme import BASELINE_LINESTYLES, CATEGORICAL_PALETTE


@dataclass(frozen=True)
class SemanticColors:
    """The role-keyed colors every renderer in `viz` shares: whichever figure
    a reader looks at, "the closed-form ground truth" is always the same red
    and "the candidate under study" the same aqua."""

    baseline: str
    optimum: str
    candidate: str
    aggregate: str
    masked_shade: str
    cocp: str


#: The single definition of the semantic roles (by palette reference, never
#: by re-derived literal index).
SEMANTIC_COLORS = SemanticColors(
    baseline=CATEGORICAL_PALETTE[5],
    optimum=CATEGORICAL_PALETTE[5],
    candidate=CATEGORICAL_PALETTE[1],
    aggregate=CATEGORICAL_PALETTE[6],
    masked_shade="0.85",
    # NOT a CATEGORICAL_PALETTE reference (unlike the roles above): the NB04
    # COCP/viz refinement plan (Sec 1.0.2) settled on this exact violet,
    # `#7b5cd6`, chosen for maximal distinction from OPTIMUM's red star, the
    # landscape toolkit's `box_region.BOX_REGION_COLOR` orangered outline,
    # and CANDIDATE's aqua path -- a fixed marker color for one specific
    # controller (COCP), not a cycled series role, so it doesn't need a
    # palette slot the way the roles above (each keyed by re-derived index)
    # do.
    cocp="#7b5cd6",
)

#: Legacy-named aliases (the landscape toolkit's renderers grew up on these
#: names); all resolve through the one `SEMANTIC_COLORS` instance.
BASELINE_COLOR = SEMANTIC_COLORS.baseline
OPTIMUM_COLOR = SEMANTIC_COLORS.optimum
CANDIDATE_COLOR = SEMANTIC_COLORS.candidate
AGGREGATE_COLOR = SEMANTIC_COLORS.aggregate
MASKED_SHADE = SEMANTIC_COLORS.masked_shade
COCP_COLOR = SEMANTIC_COLORS.cocp

#: Face alpha for the COCP diamond marker (NB04 reference-bounds plan Sec
#: 5.1) -- applied to the FACE only (`matplotlib.colors.to_rgba(COCP_COLOR,
#: COCP_MARKER_ALPHA)`), never via a bare `scatter(alpha=...)` kwarg, which
#: would fade the marker's black edge along with its face and cost the
#: crisp outline that makes it findable against a viridis/plasma backdrop.
#: Lets the unfolding path read through the marker instead of being
#: occluded by it; tune this one constant, never a per-call-site literal.
COCP_MARKER_ALPHA = 0.55

#: Relative-error denominator floor, guarding a baseline control that
#: legitimately passes through ~0 (see `landscape.signals
#: .compute_relative_error`). A numeric tolerance policy, re-exported here
#: (N6) so renderers never define their own epsilon.
DEFAULT_EPSILON = 1e-8

#: Asset 5's (3D scatter) cost colormap -- deliberately distinct from Assets
#: 3/4's "viridis" contour/surface colormap, so a sparse scatter cloud never
#: reads as "the same kind of plot" as a filled contour/surface at a glance.
SCATTER_CMAP = "plasma"


# --- The neutral series-style registry (T4.c) --------------------------------

#: Registered series-family name -> fixed color. Populated by applications
#: (e.g. `src/applications/styles.py`), never by this generic layer itself.
_SERIES_COLORS: dict[str, str] = {}
#: Registered family names rendered dashed (iterative/gradient-trained
#: models); everything else renders solid.
_DASHED_SERIES: set[str] = set()


def register_series_styles(
    colors: Mapping[str, str], *, dashed: Iterable[str] = ()
) -> None:
    """Register (or override) an application's series-family -> color mapping
    and, optionally, which of those families render dashed. Idempotent:
    re-registering the same mapping is a no-op, so import-time registration
    is safe under repeated imports."""
    _SERIES_COLORS.update(colors)
    _DASHED_SERIES.update(dashed)


def registered_series_styles() -> dict[str, str]:
    """A copy of the currently registered series-family -> color mapping."""
    return dict(_SERIES_COLORS)


def resolve_series_family(label: str) -> str | None:
    """Best-effort match of a registered series-family name inside an
    arbitrary plot label, e.g. "run_..._analytic :: trajectory_states" ->
    "analytic" -- dashboard labels often aren't bare family names. None if no
    registered family name appears in `label`."""
    lowered = label.lower()
    return next((name for name in _SERIES_COLORS if name in lowered), None)


def series_color(label: str, *, fallback_index: int = 0) -> str:
    """Fixed color for a registered series family; falls back to cycling
    through `CATEGORICAL_PALETTE` (keyed by `fallback_index`) for anything
    unregistered, so an unanticipated series name still gets a legible,
    palette-consistent color instead of raising."""
    if label in _SERIES_COLORS:
        return _SERIES_COLORS[label]
    return CATEGORICAL_PALETTE[fallback_index % len(CATEGORICAL_PALETTE)]


def series_linestyle(label: str) -> str:
    """Dashed for families registered as iterative/dashed, solid otherwise
    (including any label that resolves to no registered family) -- a second,
    redundant channel so the closed-form/iterative distinction survives
    grayscale printing too."""
    return "--" if resolve_series_family(label) in _DASHED_SERIES else "-"


@dataclass(frozen=True)
class ReferenceLineStyle:
    """An explicit, per-label override for one `draw_reference_lines` entry
    (NB04 reference-bounds plan Sec 2.2) -- the fix for that function's own
    index-coupled styling: cycling color/linestyle by ENUMERATION POSITION
    means removing one entry from a `reference_lines` mapping silently
    reassigns every subsequent entry's color and dash pattern, breaking
    cross-figure comparability the instant a caller toggles a line on/off.
    Both fields default to ``None``, reproducing today's index-cycled
    behavior exactly -- this is purely additive.

    Attributes:
        color: fixed color for this line, or ``None`` to fall back to
            `series_color`'s own registry/index-cycled resolution.
        linestyle: fixed linestyle for this line, or ``None`` to fall back
            to `theme.BASELINE_LINESTYLES`' own index-cycled resolution.
    """

    color: str | None = None
    linestyle: str | None = None


def draw_reference_lines(
    ax: Any,
    reference_lines: Mapping[str, float] | None,
    *,
    styles: Mapping[str, "ReferenceLineStyle"] | None = None,
) -> None:
    """Draw each named constant (e.g. a non-iterative baseline's final cost)
    as a flat horizontal line, cycling `theme.BASELINE_LINESTYLES` and colored
    by `series_color`, with the value folded into the legend label.

    Args:
        ax: the axes to draw on.
        reference_lines: label -> constant value.
        styles: optional label -> `ReferenceLineStyle` overrides. A label
            absent from this mapping (or `styles=None` entirely) keeps
            today's index-cycled color/linestyle -- so a caller that
            removes an ENTRY from `reference_lines` (e.g. a notebook toggle,
            NB04 reference-bounds plan Sec 2.3) does not also reassign every
            remaining line's color/linestyle out from under it, PROVIDED the
            remaining lines' styles were fixed via this parameter rather
            than left to the index cycle.
    """
    for i, (label, value) in enumerate((reference_lines or {}).items()):
        style = (styles or {}).get(label)
        color = (style.color if style else None) or series_color(
            label, fallback_index=i
        )
        linestyle = (style.linestyle if style else None) or BASELINE_LINESTYLES[
            i % len(BASELINE_LINESTYLES)
        ]
        ax.axhline(
            value,
            linestyle=linestyle,
            linewidth=1.6,
            color=color,
            label=f"{label}: {value:.4g}",
        )
