"""Time-varying matrix visualizations -- e.g. a Riccati cost-to-go P_t or
feedback gain K_t sequence -- generalizing the legacy
`riccati_basic_plots.plot_matrix_rows_over_time` (rows-only) into three
selectable layouts.
"""

from itertools import product

import matplotlib.pyplot as plt
from typing import Any

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from ...core.utils import ensure_ndim, to_numpy
from ..style import CATEGORICAL_PALETTE, styled_figure

type MatrixEvolutionLayout = str
_LAYOUTS = ("overlay", "rows", "cols")

#: Right-hand band every layout reserves (via the single, final
#: `tight_layout(rect=...)` call) for its one external legend, plus a top
#: margin for the suptitle -- so the legend is never clipped and never
#: overlaps the axes regardless of layout.
_LEGEND_RECT = (0.0, 0.0, 0.82, 0.94)
#: Figure-coordinate anchor of that external legend, sitting inside the band
#: `_LEGEND_RECT` keeps clear of the axes.
_LEGEND_ANCHOR = (0.83, 0.5)
#: Cap on legend rows before spilling into an extra column, so a large
#: overlay of many components stays readable instead of running off the
#: figure vertically.
_LEGEND_MAX_ROWS = 8


def _component_label(matrix_symbol: str, row: int, col: int) -> str:
    return rf"${matrix_symbol}_{{{row + 1}{col + 1}}}(t)$"


def _line_colors(n_lines: int) -> list:
    """High-contrast color per line. Reuses the package's curated, CVD-safe
    `CATEGORICAL_PALETTE` while it has enough entries (the common m, n <= 8
    case); falls back to an evenly sampled `turbo` colormap for a large
    overlay of many components, where maximizing contrast between adjacent
    lines matters more than staying inside the fixed palette."""
    if n_lines <= len(CATEGORICAL_PALETTE):
        return list(CATEGORICAL_PALETTE[:n_lines])
    cmap = plt.get_cmap("turbo")
    # Trim the near-black/near-white colormap extremes so every line reads
    # clearly against the white publication background.
    return [cmap(x) for x in np.linspace(0.05, 0.95, n_lines)]


def _legend_ncols(n_entries: int) -> int:
    return max(1, -(-n_entries // _LEGEND_MAX_ROWS))  # ceil division


def _place_figure_legend(
    fig: Figure, handles: Any, labels: Any, *, title: str | None
) -> None:
    """The single external legend shared by every layout: anchored in the
    right-hand band `_LEGEND_RECT` reserves, so it can neither obscure the
    plotted data nor be clipped by the later `tight_layout` call."""
    fig.legend(
        handles,
        labels,
        title=title,
        loc="center left",
        bbox_to_anchor=_LEGEND_ANCHOR,
        borderaxespad=0.0,
        ncols=_legend_ncols(len(labels)),
    )


def _plot_overlay(
    fig: Figure, matrices: np.ndarray, matrix_symbol: str, xlabel: str
) -> None:
    """Every M_ij(t) component on one shared axis, each a distinct
    high-contrast color, behind a single external figure legend -- m*n
    components can otherwise outnumber an inline legend's readable capacity
    and swamp the default color cycle. Margin for that legend is reserved by
    the caller's own final `tight_layout` call."""
    _, m, n = matrices.shape
    t = np.arange(matrices.shape[0])
    ax = fig.add_subplot(1, 1, 1)
    colors = _line_colors(m * n)
    for idx, (row, col) in enumerate(product(range(m), range(n))):
        ax.plot(
            t,
            matrices[:, row, col],
            color=colors[idx],
            label=_component_label(matrix_symbol, row, col),
        )
    ax.set_xlabel(xlabel)
    _place_figure_legend(fig, *ax.get_legend_handles_labels(), title=None)


def _plot_split(
    fig: Figure,
    matrices: np.ndarray,
    matrix_symbol: str,
    xlabel: str,
    *,
    split_axis: int,
) -> None:
    """One subplot per row (split_axis=1) or per column (split_axis=2), each
    overlaying that row's/column's remaining components -- e.g. "rows" shows
    each state row's coupling to every column in its own panel.

    Line color encodes the within-panel component index (shared across every
    panel), so a *single* external figure legend -- keyed by that index --
    identifies every curve together with its panel title, replacing the
    per-axes legends that otherwise overlap and obscure the subplots."""
    _, m, n = matrices.shape
    t = np.arange(matrices.shape[0])
    num_panels, num_components = (m, n) if split_axis == 1 else (n, m)
    panel_kind = "Row" if split_axis == 1 else "Column"
    component_kind = "Column" if split_axis == 1 else "Row"
    colors = _line_colors(num_components)

    axes = fig.subplots(nrows=num_panels, ncols=1, sharex=True, squeeze=False)
    for panel in range(num_panels):
        ax = axes[panel, 0]
        for component in range(num_components):
            row, col = (panel, component) if split_axis == 1 else (component, panel)
            ax.plot(t, matrices[:, row, col], color=colors[component])
        ax.set_title(f"{panel_kind} {panel + 1}")
    axes[-1, 0].set_xlabel(xlabel)

    handles = [Line2D([], [], color=colors[c]) for c in range(num_components)]
    labels = [str(c + 1) for c in range(num_components)]
    _place_figure_legend(fig, handles, labels, title=component_kind)


def plot_matrix_evolution(
    matrices: np.ndarray,
    *,
    matrix_symbol: str = "M",
    layout: MatrixEvolutionLayout = "overlay",
    title: str | None = None,
    xlabel: str = "Time step",
) -> Figure:
    """Plot a (T, m, n) sequence of matrices (e.g. a Riccati cost-to-go P_t
    or feedback gain K_t) over time.

    Args:
        matrices: matrix time series, shape (T, m, n).
        matrix_symbol: LaTeX-friendly symbol used in component labels, e.g.
            "P" -> "$P_{11}(t)$".
        layout: one of:
            - "overlay": every component M_ij(t) on a single shared axis.
            - "rows": one subplot per matrix row, each showing that row's
              n components.
            - "cols": one subplot per matrix column, each showing that
              column's m components.
        title: figure title; defaults to a description of `matrix_symbol`.

    Returns:
        The Figure.
    """
    matrices = to_numpy(matrices)
    ensure_ndim("matrices", matrices, 3)
    if layout not in _LAYOUTS:
        raise ValueError(f"Unknown layout {layout!r}; expected one of {_LAYOUTS}.")

    default_title = rf"Evolution of $\mathbf{{{matrix_symbol}}}(t)$"

    with styled_figure():
        fig = plt.figure()
        if layout == "overlay":
            _plot_overlay(fig, matrices, matrix_symbol, xlabel)
        elif layout == "rows":
            _plot_split(fig, matrices, matrix_symbol, xlabel, split_axis=1)
        else:
            _plot_split(fig, matrices, matrix_symbol, xlabel, split_axis=2)

        fig.suptitle(title or default_title)
        fig.tight_layout(rect=_LEGEND_RECT)

    return fig
