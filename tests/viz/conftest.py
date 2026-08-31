import matplotlib.pyplot as plt
import pytest

import mbl.applications.styles  # noqa: F401  (T4.c: the model-family color
# assertions in this package require the applications' series-style
# registration, which production consumers perform at composition time)


@pytest.fixture(autouse=True)
def _close_figures_after_each_test():
    """Prevent matplotlib's "too many open figures" warning from accumulating
    across a test session that creates many Figures."""
    yield
    plt.close("all")
