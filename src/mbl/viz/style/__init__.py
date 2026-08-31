"""viz layer L0 -- theme, semantic color roles, and file encoding. Imports
nothing project-internal (REFACTOR_PLAN v3, T4.a): the one layer every other
`viz` layer may depend on.
"""

from .io import save_animation, save_raster_figure, save_vector_figure
from .semantic import (
    AGGREGATE_COLOR,
    BASELINE_COLOR,
    CANDIDATE_COLOR,
    COCP_COLOR,
    COCP_MARKER_ALPHA,
    DEFAULT_EPSILON,
    MASKED_SHADE,
    OPTIMUM_COLOR,
    SCATTER_CMAP,
    SEMANTIC_COLORS,
    ReferenceLineStyle,
    SemanticColors,
    draw_reference_lines,
    register_series_styles,
    registered_series_styles,
    resolve_series_family,
    series_color,
    series_linestyle,
)
from .theme import (
    BASELINE_LINESTYLES,
    CATEGORICAL_PALETTE,
    PLOT_STYLE,
    autoscaled_ylim_from_data,
    place_legend_outside,
    place_measured_legend,
    style_3d_axes,
    styled_figure,
)

__all__ = [
    "PLOT_STYLE",
    "CATEGORICAL_PALETTE",
    "BASELINE_LINESTYLES",
    "autoscaled_ylim_from_data",
    "place_legend_outside",
    "place_measured_legend",
    "style_3d_axes",
    "styled_figure",
    "SemanticColors",
    "SEMANTIC_COLORS",
    "ReferenceLineStyle",
    "BASELINE_COLOR",
    "OPTIMUM_COLOR",
    "CANDIDATE_COLOR",
    "AGGREGATE_COLOR",
    "COCP_COLOR",
    "COCP_MARKER_ALPHA",
    "MASKED_SHADE",
    "SCATTER_CMAP",
    "DEFAULT_EPSILON",
    "draw_reference_lines",
    "register_series_styles",
    "registered_series_styles",
    "resolve_series_family",
    "series_color",
    "series_linestyle",
    "save_vector_figure",
    "save_raster_figure",
    "save_animation",
]
