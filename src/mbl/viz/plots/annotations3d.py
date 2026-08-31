"""Collision-avoiding 3D text annotation placement, shared by every 3D
scatter of (point, z-like-value, text) triples in `viz` -- the loss-surface
trajectory labels (`plots.loss_landscape.plot_loss_landscape_3d`) and Asset
5's genuinely 3-spatial-dimension scatter
(`landscape.landscapes.ScatterLandscapeRenderer`) alike. Extracted into
its own public module (REFACTOR_PLAN v3, T4.a) so the landscape renderers no
longer reach into another module's privates for the Z-offset convention; it
lives at the plots layer (not `landscape/`) because `landscape.landscapes`
legitimately delegates down to `plots.loss_landscape`, and this shared
utility must sit at or below the lowest of its consumers to keep the
layering acyclic.
"""

from collections.abc import Sequence

from typing import Any

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import proj3d

#: How far above its own z-value (as a fraction of the surface's total z
#: span) a trajectory point's text annotation's SEARCH ORIGIN floats, before
#: `place_trajectory_annotations`'s greedy search displaces it further --
#: enough of a head start that an isolated, non-colliding point still reads
#: as sitting just above (rather than exactly on) its own marker.
ANNOTATION_Z_OFFSET_FRACTION = 0.04

#: `place_trajectory_annotations`'s greedy collision search (Micro-Prompt
#: 4c): candidate radii (AXES-fraction, i.e. screen-space, not data units --
#: see that function's docstring for why) tried in increasing order, each
#: paired with every compass direction below, until one clears
#: `_MIN_LABEL_SEPARATION` from every already-placed label.
_ANNOTATION_SEARCH_RADII = (0.0, 0.035, 0.06, 0.09, 0.12, 0.16, 0.20, 0.26, 0.32)
_ANNOTATION_SEARCH_DIRECTIONS = (
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
)
#: Minimum center-to-center axes-fraction distance between two labels --
#: comfortably clears this toolkit's own ``"i=<n>, J=<cost>"`` bbox (~7pt
#: font) at typical figure sizes.
_MIN_LABEL_SEPARATION = 0.085


def place_trajectory_annotations(
    ax: Any,
    fig: Figure,
    points: np.ndarray,
    z_values: np.ndarray,
    texts: Sequence[str],
    z_offset: float,
) -> None:
    """Greedy, collision-avoiding placement of each trajectory point's text
    label (Micro-Prompt 4c): projects the point's own on-screen position
    under the axes' FINAL camera (`style_3d_axes` -- box aspect, elevation,
    azimuth -- must already be applied to `ax` when this is called), then
    walks outward through a ring of candidate offsets (`_ANNOTATION_SEARCH_
    RADII` x `_ANNOTATION_SEARCH_DIRECTIONS`) until it lands one whose
    axes-fraction distance from every already-placed label clears
    `_MIN_LABEL_SEPARATION`.

    Reused verbatim by any 3D scatter of (point, z-like-value, text) triples,
    not just a loss surface's own trajectory -- e.g.
    `landscape.landscapes.ScatterLandscapeRenderer` (Asset 5), where
    `points`/`z_values` are a genuinely 3-spatial-dimension scatter's own
    (x, y) and z rather than (param_1, param_2) and a surface height.

    This searches in SCREEN space rather than nudging each point's own
    (x, y, z) data coordinates by some fixed or growing fraction: a
    data-space offset has no visibility into where other labels already
    landed, so late, near-coincident gradient-descent iterates could still
    collide, or a large enough push to separate them could just as easily
    drift into an unrelated, otherwise-isolated marker's territory -- both
    observed while tuning this function's data-space predecessor. Searching
    against the real projected positions is what actually guarantees every
    label is isolated, independent of camera angle or how tightly a given
    trajectory happens to cluster.

    Placed via `ax.text2D` (screen-space, axes-fraction) rather than
    `ax.text` (3D data-space): once the greedy search has picked a
    non-colliding screen position, that choice must survive verbatim -- a
    3D-data-space text is re-projected from scratch on every redraw and
    would drift the moment the point of view changes, undoing the search.

    Args:
        ax: the target `Axes3D`, with `style_3d_axes` already applied.
        fig: `ax`'s owning Figure -- `fig.canvas.draw()` is called once to
            finalize axis limits/projection before they're read.
        points: per-trajectory-point ``(param_1, param_2)``, shape
            ``(n, 2)``.
        z_values: per-trajectory-point loss/cost, shape ``(n,)``.
        texts: per-trajectory-point label string, length ``n``.
        z_offset: constant z clearance (data units) added to each point
            before projecting, so an isolated label's search origin already
            reads as "just above" its marker.
    """
    fig.canvas.draw()
    proj = ax.get_proj()
    to_axes_fraction = ax.transAxes.inverted()

    placed: list[np.ndarray] = []
    for (x, y), z, text in zip(points, z_values, texts):
        x2, y2, _ = proj3d.proj_transform(x, y, z + z_offset, proj)
        base = np.asarray(to_axes_fraction.transform(ax.transData.transform((x2, y2))))

        chosen = base
        displaced = False
        for radius in _ANNOTATION_SEARCH_RADII:
            candidates = (
                [base]
                if radius == 0.0
                else [
                    base + radius * np.array([dx, dy])
                    for dx, dy in _ANNOTATION_SEARCH_DIRECTIONS
                ]
            )
            free = [
                c
                for c in candidates
                if all(np.linalg.norm(c - p) >= _MIN_LABEL_SEPARATION for p in placed)
            ]
            if free:
                chosen = free[0]
                displaced = radius > 0.0
                break
        placed.append(chosen)
        if displaced:
            # A thin leader line back to the marker's own screen position --
            # without it, a label the search had to push away to avoid a
            # collision would float with no visible link to which point it
            # actually annotates. Built as a plain `Line2D` + `add_line`
            # (NOT `ax.plot`, which `Axes3D` unconditionally reinterprets as
            # 3D DATA coordinates via `art3d.line_2d_to_3d`, silently
            # discarding any `transform=ax.transAxes` and re-projecting the
            # line to the wrong place).
            ax.add_line(
                Line2D(
                    [base[0], chosen[0]],
                    [base[1], chosen[1]],
                    transform=ax.transAxes,
                    color="dimgray",
                    linewidth=0.6,
                    alpha=0.6,
                    zorder=9,
                    # `Axes3D.add_line` (inherited unchanged from `Axes`)
                    # defaults to clipping against `self.patch`, which for a
                    # 3D axes is NOT the plain 2D bounding rectangle this
                    # axes-fraction line is drawn in -- clips it away
                    # entirely. `Axes3D.text`/`text2D` sidestep the exact
                    # same trap by forcing `clip_on=False`; mirrored here.
                    clip_on=False,
                )
            )
        ax.text2D(
            chosen[0],
            chosen[1],
            text,
            transform=ax.transAxes,
            fontsize=7,
            zorder=10,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )
