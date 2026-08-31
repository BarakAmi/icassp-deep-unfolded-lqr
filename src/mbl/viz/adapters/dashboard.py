"""Streamlit-facing presentation adapter (viz layer L3).

Owns the non-interactive ``Agg`` backend selection (REFACTOR_PLAN v3, N9):
the dashboard renders figures headlessly on a server with no display, where
a GUI backend would hang or error. Selecting the backend is an entry-point
decision, so it lives here (and in ``tests/conftest.py`` for the test
suite) -- importing `src.viz` itself has NO backend side effect. Import this
module before rendering any figure in a Streamlit script:

    import mbl.viz.adapters.dashboard  # noqa: F401  (forces Agg)
"""

import matplotlib

matplotlib.use("Agg")
