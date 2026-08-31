"""Shared plot styling (viz layer L0), applied per-figure (never mutates
global rcParams) so concurrent callers -- the test suite, a Streamlit rerun
rendering several figures per script run -- never leak style state into each
other. Imports nothing project-internal (REFACTOR_PLAN v3, T4.a)."""

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

from typing import Any, cast

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backend_bases import RendererBase
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.text import Text

PLOT_STYLE = {
    # Serif body text + Computer Modern-style math (matplotlib's built-in
    # "cm" mathtext font), so labels/titles read like a typeset LaTeX
    # document without requiring an actual system LaTeX install (`text.
    # usetex` would, and could break notebook execution on a machine
    # without one).
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 13,
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "lines.linewidth": 2,
    "grid.linewidth": 0.8,
    # A "subtle grid" (publication style): thin, low-alpha, dotted, so it
    # orients the reader without competing with data ink.
    "grid.alpha": 0.3,
    "grid.linestyle": ":",
    "axes.grid": True,
    "axes.grid.which": "both",
    "axes.grid.axis": "both",
    "figure.figsize": (10, 6),
    "figure.dpi": 120,
}

#: Fixed categorical palette (colorblind-safe, validated adjacent-pair
#: contrast) assigned by model-family identity -- never re-cycled per plot --
#: so the same controller reads as the same color across every figure in the
#: dashboard. Order matters: it's the CVD-safety mechanism, not cosmetic.
CATEGORICAL_PALETTE = (
    "#2a78d6",  # blue
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
    "#e87ba4",  # magenta
    "#eb6834",  # orange
)

#: Cycled (not keyed by identity) across whichever reference lines a single
#: plot draws, so multiple flat baselines stay visually distinct from each
#: other even in black-and-white print -- not just by color. Deliberately
#: excludes solid ``"-"``: several renderers that overlay `BASELINE_LINESTYLES`
#: reference lines on real data curves (e.g. `cost_analysis._LINESTYLES`,
#: `TrainingCurveStyle`'s default) use solid for the genuine (non-reference)
#: curves, so a solid baseline would read as "real data" rather than "floor/
#: heuristic" (NB04 COCP refinement plan Sec 3.3: the 4th entry, added for
#: COCP's learned-reference line, is a dash-dot-dot pattern for exactly this
#: reason).
BASELINE_LINESTYLES = ("--", ":", "-.", (0, (3, 1, 1, 1)))


#: Default `rect` for `place_legend_outside`: reserves the right-hand 22% of
#: the figure for the legend so `tight_layout` never clips it -- narrower
#: than a first cut at 25% (NB03 overhaul Phase 4: the reserved margin was
#: eating into the axes' own proportion more than the legend content
#: actually needed), giving the axes a visibly larger share of the figure.
_DEFAULT_EXTERNAL_LEGEND_RECT = (0.0, 0.0, 0.78, 1.0)


#: Matches matplotlib's own default `axes.ymargin` -- used so a caller that
#: bypasses autoscale (`autoscaled_ylim_from_data`, e.g. `TrainingCurveStyle
#: .autoscale_to_curves`) still gets the same breathing room above/below the
#: data that ordinary autoscaling would have given it, rather than clamping
#: the extreme points exactly onto the axes' edge.
_DEFAULT_Y_MARGIN_FRACTION = 0.05


def autoscaled_ylim_from_data(
    ax: Axes, *, margin: float = _DEFAULT_Y_MARGIN_FRACTION
) -> tuple[float, float]:
    """The y-limits `ax`'s own plotted DATA would autoscale to right now --
    i.e. `ax.dataLim` (which only grows as artists are added) padded by
    `margin`, computed independently of whatever the axes' CURRENT view
    limits happen to be (NB04 reference-bounds plan Sec 2.4: callers use
    this to snapshot "the curves' own range" before drawing reference lines
    that would otherwise stretch `ax.dataLim`/autoscale far past it).

    Args:
        ax: the axes to read `dataLim` from.
        margin: fractional padding added above/below the raw data span
            (matplotlib's own `axes.ymargin` default, so the result reads
            identically to ordinary autoscaling).

    Returns:
        ``(low, high)``, ready for `ax.set_ylim`.
    """
    low, high = ax.dataLim.intervaly
    pad = margin * (high - low)
    return (low - pad, high + pad)


@dataclass(frozen=True)
class LegendStyle:
    """How a legend is placed and painted (Annex 03 §B.5).

    Attributes:
        loc: matplotlib location. ``"best"`` minimises overlap and is what a
            caller with no opinion gets; §B.5 asks for ``"upper center"`` on a
            figure whose swept axis has interesting ENDS, because that is the
            placement whose overlap falls in the middle of the axis rather
            than on the two points carrying the claim.
        framealpha: Opacity of the legend's own box, or ``None`` for
            matplotlib's default. §B.5 draws an inside legend translucent so a
            mark behind it is DIMMED rather than deleted -- a reader has to be
            able to see that a curve passes behind the key.
        ncol: Column count. ``1`` reads top-to-bottom; several columns read
            left-to-right per row, which for many or long entries reads as
            cramped and out of proportion.

    `loc` and `framealpha` apply to an inside legend only -- an outside one has
    a placement of its own. `ncol` applies to both.
    """

    loc: str = "best"
    framealpha: float | None = None
    ncol: int = 1


def place_legend_outside(
    fig: Figure,
    ax: Axes,
    *,
    rect: tuple[float, float, float, float] = _DEFAULT_EXTERNAL_LEGEND_RECT,
    inside: bool = False,
    only: Sequence[str] | None = None,
    style: LegendStyle | None = None,
) -> None:
    """Place `ax`'s legend, either OUTSIDE the axes (to the right, the
    default -- so it can never obscure plotted data, reserving matching
    margin via `fig.tight_layout`'s `rect` so the legend itself isn't
    clipped) or INSIDE (matplotlib's own best-corner placement) when a
    caller/notebook explicitly prefers a self-contained figure over the
    external-legend convention -- both are legitimate choices (NB03
    overhaul Phase 4), so this is the one accessible toggle every plot
    routes through rather than each renderer reinventing it.

    Args:
        fig: the legend's owning Figure.
        ax: the Axes whose handles/labels populate the legend.
        rect: `fig.tight_layout`'s reserved-margin rect, used only when
            `inside` is ``False``.
        inside: draw the legend inside the axes instead of outside.
        only: If given, the ONLY labels that take legend rows -- Annex 03
            §B.5's 2026-08-07 rule, under which a flat series is labelled at
            the right margin instead. Filtering here rather than by clearing
            the artists' labels is deliberate: the greyscale gate reads artist
            labels, so a series whose label were dropped to suppress its
            legend row would leave the gate inspecting nothing.
        style: How the legend is drawn -- location, box opacity and column
            count (Annex 03 §B.5). Bundled rather than passed as three more
            parameters: this signature is at the permanent ``PLR0913=6`` gate,
            and the standing rule is to bundle rather than raise a gate to fit
            a feature.
    """
    entries: dict[str, Any] = {}
    if only is not None:
        handles, labels = ax.get_legend_handles_labels()
        chosen = [
            (h, s) for h, s in zip(handles, labels, strict=True) if s in set(only)
        ]
        entries = {
            "handles": [h for h, _ in chosen],
            "labels": [s for _, s in chosen],
        }
    settings = style or LegendStyle()
    if inside:
        if settings.framealpha is not None:
            entries["framealpha"] = settings.framealpha
        ax.legend(loc=settings.loc, ncol=settings.ncol, **entries)
        fig.tight_layout()
        return
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        ncol=settings.ncol,
        **entries,
    )
    fig.tight_layout(rect=rect)


#: `place_measured_legend`'s default legend placement -- the same external,
#: axes-anchored convention every landscape asset already uses; a caller
#: whose content needs a different starting anchor (e.g. a colorbar-bearing
#: 3D static renderer) overrides via `legend_kwargs`.
_DEFAULT_MEASURED_LEGEND_KWARGS: dict[str, Any] = {
    "loc": "center left",
    "bbox_to_anchor": (1.02, 0.5),
    "borderaxespad": 0.0,
}

#: Internal tuning constants for `place_measured_legend`'s measure-
#: reposition-adjust loop -- not caller parameters (PLR0913=6, and no call
#: site in this project needs a different value): the vertical gap
#: (axes-fraction) between the legend's measured bottom edge and
#: `status_text`'s top edge; extra pixels reserved beyond the measured
#: content width, absorbing sub-pixel/font-hinting variance across
#: environments; and the fixed-point iteration cap (see the function
#: docstring's `max_passes` note).
_MEASURED_LEGEND_TEXT_GAP = 0.03
_MEASURED_LEGEND_SAFETY_PAD_PX = 15.0
_MEASURED_LEGEND_MAX_PASSES = 5


def _force_draw(fig: Figure) -> RendererBase:
    """Force a real render pass and return its renderer -- required before
    any `get_window_extent` call can measure actual (not estimated)
    extents. This project always renders through an Agg-compatible canvas
    (the test suite forces ``Agg`` in `tests/conftest.py`; notebook
    execution's ``%matplotlib inline`` is Agg-based for static output), so
    the cast is a description of this project's one supported rendering
    path, not a narrowing of matplotlib's own public API."""
    fig.canvas.draw()
    return cast(RendererBase, cast(FigureCanvasAgg, fig.canvas).get_renderer())


def place_measured_legend(
    fig: Figure,
    ax: Axes,
    *,
    status_text: Text | None = None,
    legend_kwargs: Mapping[str, Any] | None = None,
) -> Legend | None:
    """Place `ax`'s legend outside the axes, and -- if `status_text` is
    given -- pin the text directly beneath the legend's ACTUALLY RENDERED
    position, both kept fully clear of every axes already on `fig` (the
    landscape assets' colorbar, when one exists) by reserving whatever right
    margin the real content measures out to (NB04 COCP/viz refinement plan
    Sec 2.3) -- never a hand-tuned anchor/margin literal.

    Replaces every per-call-site legend/status-text placement in
    `viz.landscape.landscapes`/`viz.landscape.animators`
    (`_place_wide_legend_outside`, the 3D renderers' own trailing
    `ax.legend(...)`, and each animator's `ax.legend(...)` +
    `ax.text(..., transform=ax.transAxes)`/`fig.subplots_adjust(...)` pair)
    with one measurement-based mechanism, fixing a real bug along the way:
    the previous per-call-site text position was a hand-picked axes-fraction
    literal (one animator's was `x=0.5` -- the middle of the plot, not past
    its right edge) that merely happened not to collide with the legend at
    the sizes it was tuned against; deriving the position from the legend's
    OWN measured extent instead makes that class of bug structurally
    impossible.

    Uses `fig.subplots_adjust` exclusively for the margin reservation
    (never `fig.tight_layout`), regardless of whether the caller already
    ran `tight_layout` once for its own base layout (e.g. the surface
    renderer's shared `viz.plots.loss_landscape.plot_loss_landscape_3d`) --
    `subplots_adjust` only sets margins directly, with no auto-layout
    re-solving to fight a prior pass over, so the documented "calling
    `tight_layout` twice collapses a 3D axes to a sliver" failure mode
    (`viz.landscape.landscapes.SurfaceLandscapeRenderer`'s own docstring)
    cannot recur through this helper.

    Args:
        fig: the legend's (and, if given, `status_text`'s) owning Figure.
        ax: the Axes whose handles/labels populate the legend, and whose
            current position anchors both the legend and `status_text`
            (`transform=ax.transAxes` either way -- see the blit note
            below).
        status_text: an existing (already-constructed, typically still
            empty) `Text` artist to reposition directly beneath the legend
            -- MUST already be an axes child (`ax.text`/`ax.text2D`, never
            `fig.text`): `FuncAnimation`'s `blit=True` mode drops any artist
            outside every axes' own bbox from *saved* GIF frames, so a
            figure-level text would render blank in the saved animation
            despite holding the correct content in memory
            (`viz.landscape.animators.LandscapeAnimator`'s own docstring).
            This function re-homes it onto `ax.transAxes` defensively
            (`clip_on=False`) regardless of blit, so every caller gets the
            same safe mechanism whether or not blitting is actually active.
            ``None`` (default) places the legend alone.
        legend_kwargs: overrides/extends the default external anchor
            (`loc="center left", bbox_to_anchor=(1.02, 0.5)`) -- e.g. a
            colorbar-bearing 3D static renderer may prefer a figure-fraction
            anchor (`bbox_transform=fig.transFigure`) as its OWN starting
            point; this function's margin reservation still measures
            whatever actually rendered, so it corrects for either choice.

    Returns:
        The constructed `Legend`, or ``None`` when nothing on `ax` is
        currently labeled (an animation ablation with no optimum/COCP/box
        marker at all, say) -- `ax.legend()` is never called in that case,
        since matplotlib's own "No artists with labels found" `UserWarning`
        is a hard error under this project's zero-warning pytest policy.
        `status_text`, if given, is still positioned even then (directly at
        its configured anchor point, since there is no legend bbox to sit
        beneath).

    Note:
        Internally, this is a measure-reposition-adjust FIXED POINT
        (`_MEASURED_LEGEND_MAX_PASSES` passes), not a closed-form solve:
        shrinking the right margin also shrinks a colorbar sharing the same
        `Figure`, which shifts exactly where "already-rightmost content"
        ends, which can in turn call for a further reposition/shrink. A
        handful of passes converges for every asset this project renders
        (an unusually deep legend/colorbar combination would just stop
        improving, never loop forever or raise).
    """
    kwargs: dict[str, Any] = dict(_DEFAULT_MEASURED_LEGEND_KWARGS)
    if legend_kwargs:
        kwargs.update(legend_kwargs)

    _, existing_labels = ax.get_legend_handles_labels()
    if not existing_labels:
        # Nothing to legend -- ax.legend() would otherwise raise
        # matplotlib's own "No artists with labels found" UserWarning (a
        # hard error here, see the Returns note). status_text still needs
        # positioning: every call site that also passes status_text anchors
        # in axes-fraction (never a figure-fraction override -- that's only
        # ever used by legend-only, colorbar-bearing static renderers), so
        # its own configured anchor point is a sensible position for the
        # text with no legend above it.
        if status_text is not None:
            status_text.set_transform(ax.transAxes)
            status_text.set_clip_on(False)
            status_text.set_horizontalalignment("left")
            status_text.set_verticalalignment("top")
            status_text.set_position(kwargs["bbox_to_anchor"])
            _force_draw(fig)
        return None

    legend = ax.legend(**kwargs)
    if status_text is not None:
        # Axes-child, never a figure-level artist -- see the blit note above.
        status_text.set_transform(ax.transAxes)
        status_text.set_clip_on(False)
        status_text.set_horizontalalignment("left")
        status_text.set_verticalalignment("top")

    current_right_frac = fig.subplotpars.right
    for _ in range(_MEASURED_LEGEND_MAX_PASSES):
        renderer = _force_draw(fig)
        # Every axes already on the figure (the data axes AND, when
        # present, a colorbar's own separate Axes) -- so a colorbar-bearing
        # renderer's reference point clears the colorbar too, not just the
        # data axes it sits beside.
        reference_px = max(a.get_window_extent(renderer).x1 for a in fig.axes)
        fig_width_px = fig.get_window_extent(renderer).width

        legend_bbox = legend.get_window_extent(renderer)
        if legend_bbox.x0 < reference_px:
            # The requested anchor (whatever `legend_kwargs` chose, or the
            # axes-fraction default) does not clear existing content -- e.g.
            # a colorbar sitting between `ax` and where an axes-fraction
            # anchor lands. Force a figure-fraction anchor computed from
            # the actual measurement, never a caller-guessed constant.
            new_left_frac = (
                reference_px + _MEASURED_LEGEND_SAFETY_PAD_PX
            ) / fig_width_px
            legend.set_bbox_to_anchor((new_left_frac, 0.5), transform=fig.transFigure)
            renderer = _force_draw(fig)
            legend_bbox = legend.get_window_extent(renderer)

        if status_text is not None:
            left_frac, bottom_frac = ax.transAxes.inverted().transform(
                (legend_bbox.x0, legend_bbox.y0)
            )
            status_text.set_position(
                (left_frac, bottom_frac - _MEASURED_LEGEND_TEXT_GAP)
            )
            renderer = _force_draw(fig)

        rightmost_content_px = legend.get_window_extent(renderer).x1
        if status_text is not None:
            rightmost_content_px = max(
                rightmost_content_px, status_text.get_window_extent(renderer).x1
            )
        # The DIRECT goal-check (does the content already fit the canvas),
        # not a proxy relative to the axes/colorbar edge: shrinking the
        # margin also rescales a colorbar sharing this figure, so "how many
        # pixels did reference_px move" is not a simple 1:1 function of the
        # margin change -- re-measuring against the canvas width directly is
        # what makes this converge regardless of that internal scaling.
        overflow_px = rightmost_content_px - fig_width_px
        if overflow_px <= 0:
            break  # already fits, with the safety pad still to spare below
        new_right_frac = (
            current_right_frac
            - (overflow_px + _MEASURED_LEGEND_SAFETY_PAD_PX) / fig_width_px
        )
        # Never shrink the axes past a minimal usable width, and never let
        # `right` reach (let alone cross) `left` -- an unusually wide
        # legend degrades to "as much room as can be spared", not a
        # `ValueError` from matplotlib's own `left cannot be >= right` guard.
        min_right_frac = fig.subplotpars.left + 0.15
        new_right_frac = max(min_right_frac, min(current_right_frac, new_right_frac))
        if new_right_frac >= current_right_frac - 1e-4:
            break  # at the floor already -- can't reserve any more room
        current_right_frac = new_right_frac
        fig.subplots_adjust(right=current_right_frac)

    _force_draw(fig)  # settle final positions for whatever the caller does next
    return legend


@contextmanager
def styled_figure() -> Iterator[None]:
    """Apply `PLOT_STYLE` for the duration of the `with` block only."""
    with plt.rc_context(PLOT_STYLE):
        yield


#: Default 3D-axes treatment (`style_3d_axes`): generous label padding (a
#: 3D projection's tick-number scale otherwise overlaps the LaTeX axis
#: labels), a vertically-STRETCHED box aspect (z > x, y), and a tilted camera
#: -- together read a quadratic bowl as the deep basin it mathematically is
#: rather than a flat, foreshortened sheet (Micro-Prompt 4c: a z aspect below
#: 1.0 compresses the vertical extent and is what produced that flat-sheet
#: look in the first place).
DEFAULT_3D_LABELPAD = 15.0
DEFAULT_3D_BOX_ASPECT = (1.0, 1.0, 1.2)
# Micro-Prompt 4e: `elev=40, azim=-50` looked from BEHIND the GD path's own
# descent -- `compute_bowl_ranges` sizes both axes' span off whichever
# single component deviates farthest from the optimum, so a path that
# barely moves along the other component sits deep inside a comparatively
# oversized square domain, and from that original angle the near wall of
# the surface swings around in front of it, hiding the whole cascade behind
# the mesh (confirmed by sweeping azimuth on the same asymmetric-deviation
# geometry this asset always produces). `azim=-100` looks from roughly the
# opposite side, so the surface's near wall falls away instead of rising in
# front of the path, leaving it riding the interior wall in full view.
DEFAULT_3D_ELEV = 30.0
DEFAULT_3D_AZIM = -45.0


def style_3d_axes(
    ax: Any,
    *,
    labelpad: float = DEFAULT_3D_LABELPAD,
    box_aspect: tuple[float, float, float] = DEFAULT_3D_BOX_ASPECT,
    elev: float = DEFAULT_3D_ELEV,
    azim: float = DEFAULT_3D_AZIM,
) -> None:
    """Apply the standard 3D-axes treatment shared by every static and
    animated 3D surface renderer in this project (Assets 4/4b): label
    padding, box aspect, and camera elevation/azimuth -- one place to tune
    the "deep bowl" perspective so every caller renders it identically.

    Args:
        ax: an ``Axes3D`` (``projection="3d"``).
        labelpad: padding (points) between each axis's tick labels and its
            own axis label -- `Axes3D`'s default is too small, letting the
            numeric tick scale overlap/obscure the LaTeX label next to it.
        box_aspect: relative ``(x, y, z)`` box edge lengths (`Axes3D
            .set_box_aspect`); reshape this (e.g. grow the ``z`` entry
            further past the x/y entries) whenever a surface still reads as
            too flat.
        elev: camera elevation in degrees (`Axes3D.view_init`).
        azim: camera azimuth in degrees (`Axes3D.view_init`).
    """
    ax.xaxis.labelpad = labelpad
    ax.yaxis.labelpad = labelpad
    ax.zaxis.labelpad = labelpad
    ax.set_box_aspect(box_aspect)
    ax.view_init(elev=elev, azim=azim)
