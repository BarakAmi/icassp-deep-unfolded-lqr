"""viz layer L3 -- the adapters: the ONLY layer allowed to display or
persist (REFACTOR_PLAN v3, T4.a), and the only one importing `IPython`/
`streamlit`-facing machinery. Home of the dual-format `FigureSink` and the
"No Figure Unbacked" law (T4.e): every persisted figure is a PDF + raw-data
sidecar pair, and `re_render` reproduces any figure from its sidecar alone.

`viz.adapters.notebook` (the report adapters) and `viz.adapters.dashboard`
(the Agg backend selection) are imported explicitly by their consumers, not
re-exported here, so plain `FigureSink` use never drags in IPython.
"""

from .sink import (
    SIDECAR_SCHEMA_VERSION,
    FigureSidecar,
    FigureSink,
    SavedFigure,
    content_keyed_figures_dir,
    load_sidecar,
    re_render,
    register_kwarg_renderer,
    register_sidecar_renderer,
    write_sidecar,
)

__all__ = [
    "SIDECAR_SCHEMA_VERSION",
    "FigureSidecar",
    "FigureSink",
    "SavedFigure",
    "content_keyed_figures_dir",
    "load_sidecar",
    "re_render",
    "register_kwarg_renderer",
    "register_sidecar_renderer",
    "write_sidecar",
]
