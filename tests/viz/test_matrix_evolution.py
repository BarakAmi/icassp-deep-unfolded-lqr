import numpy as np
import pytest
import torch
from matplotlib.figure import Figure

from mbl.viz.plots.matrix_evolution import plot_matrix_evolution


def _make_matrix_series(T: int, m: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(T, m, n))


def test_plot_matrix_evolution_overlay_returns_single_axis_with_every_component() -> (
    None
):
    T, m, n = 10, 2, 3
    matrices = _make_matrix_series(T, m, n, seed=0)

    fig = plot_matrix_evolution(matrices, layout="overlay")

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1
    assert len(fig.axes[0].get_lines()) == m * n


def test_plot_matrix_evolution_rows_returns_one_subplot_per_row() -> None:
    T, m, n = 8, 4, 2
    matrices = _make_matrix_series(T, m, n, seed=1)

    fig = plot_matrix_evolution(matrices, layout="rows")

    assert len(fig.axes) == m
    for ax in fig.axes:
        assert len(ax.get_lines()) == n


def test_plot_matrix_evolution_cols_returns_one_subplot_per_column() -> None:
    T, m, n = 8, 2, 4
    matrices = _make_matrix_series(T, m, n, seed=2)

    fig = plot_matrix_evolution(matrices, layout="cols")

    assert len(fig.axes) == n
    for ax in fig.axes:
        assert len(ax.get_lines()) == m


def test_plot_matrix_evolution_uses_matrix_symbol_in_component_labels() -> None:
    matrices = _make_matrix_series(5, 1, 1, seed=3)

    fig = plot_matrix_evolution(matrices, matrix_symbol="P", layout="overlay")

    line = fig.axes[0].get_lines()[0]
    assert line.get_label() == "$P_{11}(t)$"


def test_plot_matrix_evolution_uses_custom_title_when_given() -> None:
    matrices = _make_matrix_series(5, 1, 1, seed=4)
    fig = plot_matrix_evolution(matrices, title="My Custom Title")
    assert fig.get_suptitle() == "My Custom Title"


def test_plot_matrix_evolution_default_title_mentions_matrix_symbol() -> None:
    matrices = _make_matrix_series(5, 1, 1, seed=5)
    fig = plot_matrix_evolution(matrices, matrix_symbol="K")
    assert "K" in fig.get_suptitle()


def test_plot_matrix_evolution_accepts_torch_tensor() -> None:
    matrices = torch.randn(6, 2, 2)
    fig = plot_matrix_evolution(matrices, layout="rows")
    assert isinstance(fig, Figure)


def test_plot_matrix_evolution_rejects_non_3d_input() -> None:
    with pytest.raises(ValueError, match="3"):
        plot_matrix_evolution(np.zeros((5, 5)))


def test_plot_matrix_evolution_rejects_unknown_layout() -> None:
    matrices = _make_matrix_series(5, 1, 1, seed=6)
    with pytest.raises(ValueError, match="layout"):
        plot_matrix_evolution(matrices, layout="diagonal")
