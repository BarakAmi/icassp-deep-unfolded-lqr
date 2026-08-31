import matplotlib.pyplot as plt
import pytest


@pytest.fixture(autouse=True)
def _close_figures_after_each_test():
    """Prevent matplotlib's "too many open figures" warning from accumulating
    across a test session that creates many Figures/Animations -- the same
    fixture `tests/visualizations/conftest.py` uses."""
    yield
    plt.close("all")
