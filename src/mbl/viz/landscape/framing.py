"""Grid-sizing (framing) policies for landscape projections (REFACTOR_PLAN
v3, T4.d): how wide a landscape's coordinate ranges should be, relative to
the trajectory and optimum they must frame. Presentation geometry, not
experiment logic -- relocated here from the notebook adapter layer.
"""

import numpy as np


def span_control_range(
    values: np.ndarray, num_points: int, *, pad: float = 0.25
) -> np.ndarray:
    """A padded, evenly-spaced coordinate range covering `values`' full span
    -- the shared "how wide should this landscape's grid be" policy for
    every control-component axis a landscape is projected onto: wide enough
    to frame both the historical GD path and the optimal point with visual
    breathing room, never hardcoded independently per call site.

    Args:
        values: the points (e.g. a path's samples plus the optimal point)
            the range must cover.
        num_points: number of points in the returned range.
        pad: fractional padding added on each side of `values`' own span
            (or, if that span is exactly zero, of a unit span instead).

    Returns:
        ``num_points`` evenly-spaced values from ``min(values) - pad*span``
        to ``max(values) + pad*span``.
    """
    lo, hi = float(values.min()), float(values.max())
    span = (hi - lo) or 1.0
    return np.linspace(lo - pad * span, hi + pad * span, num_points)


def compute_bowl_ranges(
    path_u: np.ndarray,
    u_star: np.ndarray,
    num_points: int,
    *,
    radius_scale: float = 1.4,
) -> tuple[np.ndarray, ...]:
    """A symmetric, per-component coordinate range centered EXACTLY at the
    analytical optimum `u_star` -- the "deep bowl" meshgrid policy
    (Micro-Prompt 4d) the local-cost-landscape adapters use for their 2D
    contour and 3D surface/scatter grids. `span_control_range` remains the
    policy for ranges that only need to *cover* a set of values.

    THE ROOT CAUSE: `span_control_range(np.append(path, u_star_component),
    ...)` pads the span from the GD path to `u_star` -- but gradient descent
    moves roughly monotonically toward `u_star`, so that padded interval
    barely extends past `u_star` on its far side. A 3D surface sliced from
    such a grid renders as a one-way slope ("hillside"), because the domain
    never captures the quadratic's curvature on `u_star`'s far side, and
    `u_star` itself sits near one edge of the frame rather than its center.

    THE FIX: center every component's range at `u_star` and radiate outward
    by the SAME scalar `radius` on every component (rather than an
    independently padded span per component), so `u_star` sits at the exact
    geometric center of a square domain -- matching this asset's equal x/y
    `box_aspect` (`viz.style.theme.style_3d_axes`) -- with identical
    curvature enclosed on every side of the minimum.

    Args:
        path_u: the GD candidate's full history at the frozen timestep,
            shape ``(I, m)`` -- `radius` is sized to enclose whichever
            historical iterate sits FARTHEST from `u_star` (not just the
            initial one), so the rendered path never runs off the edge of
            its own backdrop even if a component overshoots before
            converging.
        u_star: the analytical Riccati optimum at the frozen timestep,
            shape ``(m,)``.
        num_points: number of points in each returned range.
        radius_scale: safety margin multiplying the raw path-to-optimum
            deviation (Micro-Prompt 4d mandates ``1.4``).

    Returns:
        One ``num_points``-length, evenly-spaced ``np.ndarray`` per
        component of `u_star`, each spanning ``[u_star[j] - radius, u_star[j]
        + radius]`` with the SAME `radius` shared across every component.
    """
    deviation = float(np.abs(path_u - u_star[None, :]).max()) or 1.0
    radius = deviation * radius_scale
    return tuple(
        np.linspace(u_star[j] - radius, u_star[j] + radius, num_points)
        for j in range(u_star.shape[0])
    )
