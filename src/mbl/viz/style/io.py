"""Figure/animation file encoding (viz layer L0): `save_vector_figure` for
publication-ready static vector art, and its raster/video analogue
`save_animation` -- persists a `matplotlib.animation.Animation` as a GIF
(zero-dependency, via `PillowWriter`) or MP4 (via `FFMpegWriter`, only if a
system ``ffmpeg`` binary is available; otherwise falls back to GIF with one
warning). Each refuses the other's suffixes: a frame sequence cannot be
encoded as a single PDF/SVG document, and a manuscript figure must never be
silently degraded to raster.
"""

import warnings
from pathlib import Path
from typing import Any

from matplotlib.animation import Animation, FFMpegWriter, PillowWriter
from matplotlib.figure import Figure

#: Suffixes `save_vector_figure` accepts -- resolution-independent formats a
#: LaTeX manuscript can embed directly. Raster formats (.png, ...) are
#: rejected rather than silently degraded; `save_animation` rejects these
#: with a pointer back, since a frame sequence cannot be a single vector
#: document.
_VECTOR_SUFFIXES = frozenset({".pdf", ".svg"})


def save_vector_figure(fig: Figure, path: Path | str) -> Path:
    """Persist `fig` as publication-ready vector art, creating parent
    directories on demand.

    Written with a tight bounding box (so externally anchored legends -- see
    `theme.place_legend_outside` -- are never clipped) on an opaque white
    background (so the export renders with full contrast in any PDF viewer
    or manuscript, rather than inheriting a transparent canvas).

    Args:
        fig: The Figure to persist.
        path: Destination file; its suffix selects the matplotlib backend
            and must be one of `_VECTOR_SUFFIXES`.

    Returns:
        The resolved absolute path the figure was written to.

    Raises:
        ValueError: if `path`'s suffix is not a vector format.
    """
    path = Path(path)
    if path.suffix.lower() not in _VECTOR_SUFFIXES:
        raise ValueError(
            f"save_vector_figure only writes vector formats "
            f"{sorted(_VECTOR_SUFFIXES)}; got suffix {path.suffix!r}."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white", edgecolor="none")
    return path.resolve()


#: Suffixes `save_raster_figure` accepts -- the raster preview face beside a
#: figure's publication vector art.
_RASTER_FIGURE_SUFFIXES = frozenset({".png"})

#: The raster preview's resolution (dots per inch). High enough to read
#: crisply inline / in a slide deck, without bloating the artifact directory.
_DEFAULT_RASTER_DPI = 150


def save_raster_figure(
    fig: Figure, path: Path | str, *, dpi: int = _DEFAULT_RASTER_DPI
) -> Path:
    """Persist `fig` as a raster ``.png`` preview beside its publication vector
    art, creating parent directories on demand. Two roles: a directly
    embeddable face for slides / web / markdown (where a ``.pdf`` cannot go),
    and the load-from-store display asset the `FigureSink` reuses on a cache
    hit instead of re-rendering.

    Written with the same tight bounding box and opaque white background as
    `save_vector_figure`, so the raster face matches the vector one exactly.

    Args:
        fig: The Figure to persist.
        path: Destination file; its suffix must be one of
            `_RASTER_FIGURE_SUFFIXES`.
        dpi: The raster resolution.

    Returns:
        The resolved absolute path the preview was written to.

    Raises:
        ValueError: if `path`'s suffix is not a supported raster format.
    """
    path = Path(path)
    if path.suffix.lower() not in _RASTER_FIGURE_SUFFIXES:
        raise ValueError(
            f"save_raster_figure only writes raster figure formats "
            f"{sorted(_RASTER_FIGURE_SUFFIXES)}; got suffix {path.suffix!r}."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none")
    return path.resolve()


#: Suffix -> Writer class this function knows how to encode.
_RASTER_WRITERS = {
    ".gif": PillowWriter,
    ".mp4": FFMpegWriter,
}


def save_animation(
    anim: Animation, path: Path | str, *, fps: int = 10, writer: Any = None
) -> Path:
    """Persist `anim` to `path`, creating parent directories on demand.

    Args:
        anim: the animation to persist (e.g. from
            `animators.FrameAnimator.build`).
        path: destination file; ``".gif"`` (`PillowWriter`, always
            available -- the toolkit's default, per Phase 2B Decision D6)
            or ``".mp4"`` (`FFMpegWriter`, requires a system ``ffmpeg``
            binary -- falls back to ``".gif"`` with one warning if
            unavailable).
        fps: frames per second.
        writer: optional explicit matplotlib `Writer` instance, overriding
            the suffix-based default.

    Returns:
        The resolved absolute path the animation was actually written to
        (differs from `path` on an ``.mp4`` -> ``.gif`` fallback).

    Raises:
        ValueError: If `path`'s suffix is a vector format (``.pdf``/
            ``.svg``) or otherwise not one of `_RASTER_WRITERS`.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in _VECTOR_SUFFIXES:
        raise ValueError(
            f"save_animation only writes raster/video formats "
            f"{sorted(_RASTER_WRITERS)}; got vector suffix {path.suffix!r} "
            "(use save_vector_figure for a single static frame instead)."
        )
    if suffix not in _RASTER_WRITERS:
        raise ValueError(
            f"save_animation only writes {sorted(_RASTER_WRITERS)}; got "
            f"suffix {path.suffix!r}."
        )

    if writer is None:
        if suffix == ".mp4" and not FFMpegWriter.isAvailable():
            warnings.warn(
                "FFMpegWriter is unavailable (no system ffmpeg binary "
                "found) -- falling back to a .gif via PillowWriter instead "
                "of the requested .mp4.",
                stacklevel=2,
            )
            path = path.with_suffix(".gif")
            suffix = ".gif"
        writer = _RASTER_WRITERS[suffix](fps=fps)

    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(path), writer=writer)
    return path.resolve()
