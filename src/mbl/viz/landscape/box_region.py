"""The feasible-box overlay for the cost-landscape renderers (NB04 plan Sec
3.4, restyled to a dashed outline by the NB04 COCP/viz refinement plan Sec
2.1): a dimension-adaptive "highlighted region on the control-component
domain" showing where the box constraint ``|u_j| <= u_max`` (or an
anisotropic per-component bound) sits relative to the plotted cost surface.
One small, focused, independently-testable drawing helper per landscape
kind (line/contour/surface/scatter), so `landscapes.py`'s four renderers
each call exactly one function rather than repeating patch/collection
bookkeeping four times.

Every helper is a pure "draw onto `ax`" side effect (matching this
package's `overlay.TrajectoryOverlay.draw` convention) and is a no-op the
caller simply never invokes when no bound is configured -- there is no
internal `None`-check here; `landscapes.py`'s renderers own that branch, so
an unconstrained (NB03) call site never even imports the feasible-set
concept into its control flow.
"""

from collections.abc import Sequence
from typing import cast

import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D

#: The feasible-region color -- deliberately distinct from every semantic
#: role in `viz.style.semantic` (BASELINE/OPTIMUM/CANDIDATE/AGGREGATE/COCP)
#: and from `SCATTER_CMAP`, so "this marks the constraint" reads
#: unambiguously and never collides with an existing series/marker color.
#: A bold, high-salience color at full opacity (NB04 COCP/viz refinement
#: plan Sec 2.1) -- the previous `seagreen`/``alpha=0.12`` fill read as too
#: subtle to notice, which was the complaint driving this restyle.
BOX_REGION_COLOR = "orangered"
BOX_REGION_LINEWIDTH = 1.8
BOX_REGION_LINESTYLE = "--"


def format_box_region_label(*bounds: tuple[float, float]) -> str:
    """The feasible-box legend label, carrying the numeric bound(s) (NB04
    COCP/viz refinement plan Sec 2.1) -- called internally by every
    `draw_box_region_*` function below, so no caller in `landscapes.py`/
    `animators.py` needs to build this string itself.

    Args:
        *bounds: one ``(low, high)`` pair per varied component (1-3 of
            them), exactly as already resolved by `resolve_component_bounds`.

    Returns:
        ``"Feasible box ($|u|_\\infty \\leq <bound>$)"`` when every pair is
        identical and symmetric about zero (the common isotropic case, e.g.
        a scalar ``u_max`` broadcast to every component); a per-component
        listing otherwise (anisotropic bounds, or a pair not symmetric about
        zero, where a single infinity-norm bound can't describe the region).
    """
    first_low, first_high = bounds[0]
    isotropic_symmetric = first_low == -first_high and all(
        (lo, hi) == (first_low, first_high) for lo, hi in bounds
    )
    if isotropic_symmetric:
        return rf"Feasible box ($|u|_\infty \leq {first_high:.2f}$)"
    per_component = ", ".join(f"[{lo:.2f}, {hi:.2f}]" for lo, hi in bounds)
    return f"Feasible box ({per_component})"


def resolve_component_bounds(
    overlay_bounds: tuple[float, float] | Sequence[tuple[float, float]] | None,
    num_components: int,
) -> tuple[tuple[float, float], ...] | None:
    """Resolve `overlay.TrajectoryOverlay.box_bounds` into exactly
    `num_components` ``(low, high)`` pairs, in `SliceSpec.components` order
    -- the single normalization site every renderer in `landscapes.py`
    calls, so the "single shared pair vs. one pair per component" ambiguity
    is handled once.

    Args:
        overlay_bounds: `TrajectoryOverlay.box_bounds` as given -- `None`,
            a single ``(low, high)`` pair (broadcast to every component,
            the common ISOTROPIC case), or a sequence of `num_components`
            pairs (anisotropic).
        num_components: how many varied components this landscape slice
            has (``len(field.spec.components)``, i.e. 1/2/3).

    Returns:
        A tuple of exactly `num_components` ``(low, high)`` pairs, or
        `None` when `overlay_bounds` is `None`.

    Raises:
        ValueError: If `overlay_bounds` is an explicit per-component
            sequence whose length disagrees with `num_components`.
    """
    if overlay_bounds is None:
        return None
    # Discriminate on overlay_bounds[0] ALONE -- never assume a minimum
    # length of 2 by also probing [1], which would raise IndexError on a
    # genuine (but short, e.g. length-1) per-component sequence before its
    # own length-mismatch check below gets a chance to run.
    first = overlay_bounds[0]
    if isinstance(first, int | float):
        # overlay_bounds IS the bare (low, high) pair -- broadcast it.
        low, high = cast("tuple[float, float]", overlay_bounds)
        return tuple((float(low), float(high)) for _ in range(num_components))
    # Not a bare pair -- a per-component sequence instead.
    bounds_seq = cast("Sequence[tuple[float, float]]", overlay_bounds)
    if len(bounds_seq) != num_components:
        raise ValueError(
            f"box_bounds gives {len(bounds_seq)} per-component pair(s), "
            f"expected {num_components} (one per varied component)."
        )
    return tuple((float(lo), float(hi)) for lo, hi in bounds_seq)


def draw_box_region_1d(ax: Axes, bounds: tuple[float, float]) -> None:
    """Outline the feasible interval on a 1D (m=1) line landscape: two
    dashed vertical lines at ``u = low`` and ``u = high`` -- the natural
    1D "frame" (NB04 COCP/viz refinement plan Sec 2.1; previously a filled
    translucent band).

    Args:
        ax: the line landscape's Axes.
        bounds: ``(low, high)`` of the single varied control component.
    """
    low, high = bounds
    label = format_box_region_label(bounds)
    ax.axvline(
        low,
        color=BOX_REGION_COLOR,
        linestyle=BOX_REGION_LINESTYLE,
        linewidth=BOX_REGION_LINEWIDTH,
        zorder=0,
        label=label,
    )
    ax.axvline(
        high,
        color=BOX_REGION_COLOR,
        linestyle=BOX_REGION_LINESTYLE,
        linewidth=BOX_REGION_LINEWIDTH,
        zorder=0,
    )


def draw_box_region_2d(
    ax: Axes, bounds_a: tuple[float, float], bounds_b: tuple[float, float]
) -> None:
    """Outline the feasible square/rectangle on a 2D (m=2) contour
    landscape: an unfilled, dashed `Rectangle` patch (NB04 COCP/viz
    refinement plan Sec 2.1; previously filled).

    Args:
        ax: the contour landscape's Axes.
        bounds_a: ``(low, high)`` of the first varied control component
            (the plot's x-axis).
        bounds_b: ``(low, high)`` of the second varied control component
            (the plot's y-axis).
    """
    lo_a, hi_a = bounds_a
    lo_b, hi_b = bounds_b
    label = format_box_region_label(bounds_a, bounds_b)
    ax.add_patch(
        Rectangle(
            (lo_a, lo_b),
            hi_a - lo_a,
            hi_b - lo_b,
            facecolor="none",
            edgecolor=BOX_REGION_COLOR,
            linestyle=BOX_REGION_LINESTYLE,
            linewidth=BOX_REGION_LINEWIDTH,
            zorder=0,
            label=label,
        )
    )


def draw_box_region_3d_shadow(
    ax: Axes3D,
    bounds_a: tuple[float, float],
    bounds_b: tuple[float, float],
    *,
    z: float,
) -> None:
    """Outline the feasible square onto a 3D (m=2) surface landscape's
    floor: a closed, dashed polyline at a fixed height `z` (the surface's
    own minimum cost, so it reads as a footprint beneath the bowl rather
    than a slice through it) -- NB04 COCP/viz refinement plan Sec 2.1;
    previously a filled `Poly3DCollection` patch.

    Args:
        ax: the surface landscape's 3D Axes.
        bounds_a: ``(low, high)`` of the first varied control component.
        bounds_b: ``(low, high)`` of the second varied control component.
        z: the constant height the outline is drawn at (typically
            ``field.cost_grid.min()``).
    """
    lo_a, hi_a = bounds_a
    lo_b, hi_b = bounds_b
    label = format_box_region_label(bounds_a, bounds_b)
    # Closes the loop (5th point repeats the 1st) -- the same "plain
    # ax.plot polyline" idiom draw_box_region_3d_wireframe already uses per
    # edge, just one closed quadrilateral instead of 12 cube edges.
    xs = [lo_a, hi_a, hi_a, lo_a, lo_a]
    ys = [lo_b, lo_b, hi_b, hi_b, lo_b]
    zs = [z] * 5
    ax.plot(
        xs,
        ys,
        zs,
        color=BOX_REGION_COLOR,
        linestyle=BOX_REGION_LINESTYLE,
        linewidth=BOX_REGION_LINEWIDTH,
        zorder=1,
        label=label,
    )


#: The 8 unit-cube corner sign combinations (each row picks "low" (-1) or
#: "high" (+1) per axis) and the 12 edges connecting corners that differ in
#: EXACTLY one coordinate -- a plain, dependency-free wireframe-cube
#: topology (no `itertools` needed: the combinations are few and fixed).
_CUBE_CORNER_SIGNS = np.array(
    [[sa, sb, sc] for sa in (0, 1) for sb in (0, 1) for sc in (0, 1)]
)
_CUBE_EDGES = [
    (i, j)
    for i in range(8)
    for j in range(i + 1, 8)
    if int(np.sum(_CUBE_CORNER_SIGNS[i] != _CUBE_CORNER_SIGNS[j])) == 1
]


def draw_box_region_3d_wireframe(
    ax: Axes3D,
    bounds_a: tuple[float, float],
    bounds_b: tuple[float, float],
    bounds_c: tuple[float, float],
) -> None:
    """Outline the feasible cube on a 3D (m=3) scatter landscape: the 12
    edges of the box ``u_a in bounds_a, u_b in bounds_b, u_c in bounds_c``,
    drawn as a bold dashed wireframe (NB04 COCP/viz refinement plan Sec
    2.1: restyled to match the other landscape kinds' outline; previously a
    translucent wireframe at a distinct alpha).

    Args:
        ax: the scatter landscape's 3D Axes.
        bounds_a: ``(low, high)`` of the first varied control component.
        bounds_b: ``(low, high)`` of the second varied control component.
        bounds_c: ``(low, high)`` of the third varied control component.
    """
    axis_bounds = (bounds_a, bounds_b, bounds_c)
    label = format_box_region_label(*axis_bounds)
    corners = np.array(
        [
            [axis_bounds[axis][sign] for axis, sign in enumerate(signs)]
            for signs in _CUBE_CORNER_SIGNS
        ]
    )
    for edge_index, (i, j) in enumerate(_CUBE_EDGES):
        ax.plot(
            [corners[i, 0], corners[j, 0]],
            [corners[i, 1], corners[j, 1]],
            [corners[i, 2], corners[j, 2]],
            color=BOX_REGION_COLOR,
            linestyle=BOX_REGION_LINESTYLE,
            linewidth=BOX_REGION_LINEWIDTH,
            zorder=1,
            # Only the first edge carries the (shared) legend label -- 12
            # identical entries would otherwise flood the legend.
            label=label if edge_index == 0 else None,
        )
