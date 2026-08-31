"""Tier 7 — presentation: tidy tables to rendered artifacts (Annex 03 Part B).

A figure is a **pure function from an analysis table plus a specification to
rendered artifacts**. It never computes and it never touches the store's
models, which is what makes `mbl figure rebuild <id> --style ieee-2col` a
re-render with no re-execution — and the reason D6 rejected the matplotlib
pickle in favour of a data-plus-spec pair.

This tier may import `core`, `store`, `spec`, `analysis` and `viz`. It may not
be imported by any of them.
"""

from __future__ import annotations

from .axis_scaling import axis_scaling
from .grouped_bars import grouped_bars
from .severity_panels import severity_panels
from .artifacts import (
    FIGURE_SUFFIXES,
    RASTER_DPI,
    FigureArtifacts,
    read_figure_data,
    read_figure_spec,
    write_figure_artifacts,
)
from .greyscale import (
    LUMINANCE_FLOOR,
    EncodingDrift,
    GreyscaleError,
    SeriesDriftError,
    SeriesEncoding,
    UnreadableSeriesError,
    encoding_drifts,
    encodings_by_axes,
    encodings_of,
    figure_conflicts,
    greyscale_conflicts,
    printed_color,
    relative_luminance,
    require_greyscale_separable,
)
from .encodings import (
    BAR_FILL_COLORS,
    BAR_HATCHES,
    EncodingCapacityError,
    FillStyle,
    SeriesStyle,
    encode_fills,
    encode_series,
)
from .profiles import (
    DEFAULT_PROFILE,
    PROFILES,
    StyleProfile,
    profile_context,
    resolve_profile,
)
from .registry import (
    FigureContext,
    FigureKind,
    register_figure,
    registered_figures,
    resolve_figure,
)
from .runner import FigureOutcome, figure_from_artifacts, render_figures

__all__ = [
    "FigureContext",
    "FigureKind",
    "FigureOutcome",
    "SeriesStyle",
    "axis_scaling",
    "grouped_bars",
    "severity_panels",
    "BAR_FILL_COLORS",
    "BAR_HATCHES",
    "EncodingCapacityError",
    "FillStyle",
    "encode_fills",
    "encode_series",
    "register_figure",
    "registered_figures",
    "figure_from_artifacts",
    "render_figures",
    "resolve_figure",
    "DEFAULT_PROFILE",
    "FIGURE_SUFFIXES",
    "LUMINANCE_FLOOR",
    "PROFILES",
    "RASTER_DPI",
    "FigureArtifacts",
    "GreyscaleError",
    "SeriesEncoding",
    "StyleProfile",
    "encodings_of",
    "encoding_drifts",
    "encodings_by_axes",
    "figure_conflicts",
    "greyscale_conflicts",
    "printed_color",
    "profile_context",
    "read_figure_data",
    "read_figure_spec",
    "relative_luminance",
    "require_greyscale_separable",
    "EncodingDrift",
    "SeriesDriftError",
    "UnreadableSeriesError",
    "resolve_profile",
    "write_figure_artifacts",
]
