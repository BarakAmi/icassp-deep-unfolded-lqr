"""Multi-window time-series comparisons for state/control/observation
trajectories -- accepts plain arrays (or parsed artifacts), never an
`LQRProblem` or other legacy model object."""

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
from typing import cast

import numpy as np
from matplotlib.figure import Figure

from ...core.utils import ensure_same_axis_size, to_numpy
from ..style import series_color, series_linestyle, resolve_series_family, styled_figure


def _as_unbatched_2d(name: str, array: np.ndarray, *, sample_index: int) -> np.ndarray:
    """Normalize a (horizon, dim) or (batch, horizon, dim) array down to
    (horizon, dim) by selecting one sample out of a batched trajectory."""
    array = to_numpy(array)
    if array.ndim == 2:
        return array
    if array.ndim == 3:
        return cast(np.ndarray, array[sample_index])
    raise ValueError(
        f"{name} must be a (horizon, dim) or (batch, horizon, dim) array, "
        f"got shape {array.shape}."
    )


def plot_trajectory_comparison(
    series: Mapping[str, np.ndarray],
    *,
    variable_name: str = "x",
    dim_labels: Sequence[str] | None = None,
    sample_index: int = 0,
    xlabel: str = "Time step",
    title: str | None = None,
) -> Figure:
    """Overlay named trajectories, one subplot per state/control/observation
    dimension -- the generalization of comparing e.g. a learned controller
    against a Riccati baseline to an arbitrary number of named series.

    Args:
        series: label -> array of shape (horizon, dim) or (batch, horizon, dim).
            Batched arrays are reduced to a single sample via `sample_index`.
        variable_name: symbol used for the y-axis labels, e.g. "x", "u", "y".
        dim_labels: optional per-dimension labels; defaults to
            "{variable_name}_{i}".
        sample_index: which batch element to plot when a series is batched.
        title: figure title; defaults to a description of `variable_name`.

    Returns:
        The Figure, with one row of subplots per dimension.
    """
    if not series:
        raise ValueError("series must contain at least one named trajectory.")

    normalized = {
        label: _as_unbatched_2d(label, array, sample_index=sample_index)
        for label, array in series.items()
    }
    ensure_same_axis_size(
        "All trajectories being compared must share the same dimension.",
        axis=-1,
        **normalized,
    )

    dim = next(iter(normalized.values())).shape[-1]
    if dim_labels is not None and len(dim_labels) != dim:
        raise ValueError(
            f"dim_labels must have length {dim} (the trajectory dimension), "
            f"got {len(dim_labels)}."
        )

    with styled_figure():
        fig, axes = plt.subplots(
            nrows=dim,
            ncols=1,
            sharex=True,
            squeeze=False,
            figsize=(10, max(3.0, 2.5 * dim)),
        )
        for i in range(dim):
            ax = axes[i, 0]
            for j, (label, array) in enumerate(normalized.items()):
                family = resolve_series_family(label)
                ax.plot(
                    np.arange(array.shape[0]),
                    array[:, i],
                    label=label,
                    color=series_color(family or label, fallback_index=j),
                    linestyle=series_linestyle(label),
                )
            component_label = dim_labels[i] if dim_labels else f"{variable_name}_{i}"
            ax.set_ylabel(component_label)

        # One shared legend (outside the axes, to the right) rather than a
        # repeated per-subplot legend -- every subplot plots the same set of
        # named series, so repeating it dim times would only add clutter.
        handles, series_labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            series_labels,
            loc="center left",
            bbox_to_anchor=(0.99, 0.5),
            borderaxespad=0.0,
        )

        axes[-1, 0].set_xlabel(xlabel)
        fig.suptitle(title or f"Comparison of ${variable_name}(t)$ Trajectories")
        fig.tight_layout(rect=(0.0, 0.0, 0.8, 0.96))

    return fig
