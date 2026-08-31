"""Asset 1 (control signals) and Asset 2 (per-timestep relative error)
static renderers: generic, per-control-component comparisons of a baseline
vs. either an automatic sparse selection of candidate iterations (one shared
alpha-faded hue) or a small, caller-named set of iterations, each in its own
color (`plot_sparse_signals_and_errors`).

Structurally mirrors `viz.plots.trajectories.plot_trajectory_comparison`'s
stacked, ``sharex`` subplot-per-dimension idiom -- that function's own
named-series signature doesn't fit the per-iteration alpha-fade/color-ramp
this needs, so the layout idiom is followed rather than the function called
directly.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from .overlay import select_iterations
from ..style import (
    AGGREGATE_COLOR,
    BASELINE_COLOR,
    CANDIDATE_COLOR,
    DEFAULT_EPSILON,
    MASKED_SHADE,
)
from .types import IterationSelection, normalize_control, normalize_history
from ..style import styled_figure

_DEFAULT_SPARSE_COUNT = 6


@dataclass(frozen=True)
class SignalStyle:
    """The presentation knobs shared by every signal-panel renderer and
    animator in this module (and by their Tier-4 adapters): which batch
    element to slice, how to label components/axes, and the figure title.

    Attributes:
        sample_index: Batch element to select if an input is batched.
        dim_labels: Optional per-component labels; ``None`` derives
            ``"$u_{j}$"``-style defaults.
        xlabel: Shared x-axis label.
        title: Figure title; ``None`` selects each renderer's default.
    """

    sample_index: int = 0
    dim_labels: Sequence[str] | None = None
    xlabel: str = "Time step"
    title: str | None = None


def _prepare(
    baseline_U: np.ndarray,
    candidate_history: np.ndarray,
    *,
    sample_index: int,
    dim_labels: Sequence[str] | None,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Normalize `baseline_U`/`candidate_history` to their unbatched
    ``(T, m)``/``(I, T, m)`` forms and fail-fast validate their shapes agree
    with each other and with `dim_labels` -- shared by every renderer in
    this module.

    Returns:
        ``(baseline, history, T, m)``.

    Raises:
        ValueError: If `candidate_history`'s ``(T, m)`` disagrees with
            `baseline_U`'s, or `dim_labels`'s length disagrees with ``m``.
    """
    baseline = normalize_control("baseline_U", baseline_U, sample_index=sample_index)
    history = normalize_history(
        "candidate_history", candidate_history, sample_index=sample_index
    )
    if history.shape[1:] != baseline.shape:
        raise ValueError(
            f"candidate_history's per-iteration (T, m) = {history.shape[1:]} "
            f"must match baseline_U's (T, m) = {baseline.shape}."
        )
    horizon, control_dim = baseline.shape
    if dim_labels is not None and len(dim_labels) != control_dim:
        raise ValueError(
            f"dim_labels must have length {control_dim}, got {len(dim_labels)}."
        )
    return baseline, history, horizon, control_dim


def _resolve_selection(
    selection: IterationSelection | None, num_iterations: int
) -> IterationSelection:
    """Default to a log-spaced sparse subset (Part 5.2: static plots must
    heavily subsample) when `selection` isn't given explicitly."""
    if selection is not None:
        return selection
    return select_iterations(
        num_iterations, k=min(_DEFAULT_SPARSE_COUNT, num_iterations)
    )


def _candidate_alphas(indices: np.ndarray) -> np.ndarray:
    """Alpha ramp over the selected iteration indices: oldest faintest,
    most recent boldest -- so the eye reads iteration recency directly from
    the figure without needing a colorbar/legend per curve."""
    if len(indices) == 1:
        return np.array([1.0])
    span = indices[-1] - indices[0]
    return cast(np.ndarray, 0.25 + 0.65 * (indices - indices[0]) / span)


def compute_relative_error(
    baseline: np.ndarray, candidate: np.ndarray, *, epsilon: float = DEFAULT_EPSILON
) -> np.ndarray:
    """Per-component relative percentage error
    ``|u_cand - u_base| / max(|u_base|, epsilon) * 100``.

    Args:
        baseline: shape ``(T, m)``.
        candidate: shape ``(T, m)``.
        epsilon: denominator floor.

    Returns:
        Shape ``(T, m)``, in percent. Callers that need to distinguish a
        genuinely small error from a numerically meaningless one (baseline
        ~ 0) should additionally check ``np.abs(baseline) < epsilon``
        (`RelativeErrorRenderer` masks these rather than plotting a
        spurious spike).
    """
    denom = np.maximum(np.abs(baseline), epsilon)
    return cast(np.ndarray, 100.0 * np.abs(candidate - baseline) / denom)


def compute_aggregate_relative_error(
    baseline: np.ndarray, candidate: np.ndarray, *, epsilon: float = DEFAULT_EPSILON
) -> np.ndarray:
    """Aggregate (per-timestep L2-norm-ratio) relative percentage error
    ``||u_cand_t - u_base_t||_2 / max(||u_base_t||_2, epsilon) * 100``.

    Args:
        baseline: shape ``(T, m)``.
        candidate: shape ``(T, m)``.
        epsilon: denominator floor.

    Returns:
        Shape ``(T,)``, in percent.
    """
    denom = np.maximum(np.linalg.norm(baseline, axis=-1), epsilon)
    return cast(
        np.ndarray, 100.0 * np.linalg.norm(candidate - baseline, axis=-1) / denom
    )


class ControlSignalRenderer:
    """Asset 1 (static): per-component control signal ``u_t`` over time,
    baseline vs. a sparse selection of candidate iterations."""

    def render(
        self,
        baseline_U: np.ndarray,
        candidate_history: np.ndarray,
        *,
        selection: IterationSelection | None = None,
        style: SignalStyle = SignalStyle(),
    ) -> Figure:
        """
        Args:
            baseline_U: reference control, shape ``(T, m)`` or ``(B, T, m)``.
            candidate_history: per-iteration control iterates, shape
                ``(I, T, m)`` or ``(I, B, T, m)``.
            selection: which iterations to overlay; defaults to a
                log-spaced sparse subset.
            style: slicing/labeling/title knobs (see `SignalStyle`).

        Returns:
            The Figure, one subplot per control component.
        """
        sample_index, dim_labels = style.sample_index, style.dim_labels
        xlabel, title = style.xlabel, style.title
        baseline, history, horizon, control_dim = _prepare(
            baseline_U,
            candidate_history,
            sample_index=sample_index,
            dim_labels=dim_labels,
        )
        selection = _resolve_selection(selection, history.shape[0])
        indices, alphas = selection.indices, _candidate_alphas(selection.indices)
        t = np.arange(horizon)

        with styled_figure():
            fig, axes = plt.subplots(
                nrows=control_dim,
                ncols=1,
                sharex=True,
                squeeze=False,
                figsize=(10, max(3.0, 2.5 * control_dim)),
            )
            for j in range(control_dim):
                ax = axes[j, 0]
                self._draw_component(
                    ax, t, baseline[:, j], history[:, :, j], indices, alphas
                )
                ax.set_ylabel(dim_labels[j] if dim_labels else f"$u_{{{j}}}$")

            self._finish_figure(
                fig,
                axes,
                xlabel,
                title or "Control Signals: Baseline vs. Candidate Iterations",
            )
        return fig

    @staticmethod
    def _draw_component(
        ax: Any,
        t: np.ndarray,
        baseline_j: np.ndarray,
        history_j: np.ndarray,
        indices: np.ndarray,
        alphas: np.ndarray,
    ) -> None:
        for iteration, alpha in zip(indices, alphas):
            is_last = iteration == indices[-1]
            ax.plot(
                t,
                history_j[iteration],
                color=CANDIDATE_COLOR,
                alpha=float(alpha),
                linewidth=2.2 if is_last else 1.2,
                label=f"Candidate (iter {iteration})" if is_last else None,
                zorder=5 + int(is_last),
            )
        ax.plot(
            t,
            baseline_j,
            color=BASELINE_COLOR,
            linestyle="--",
            linewidth=2.0,
            label="Baseline",
            zorder=10,
        )

    @staticmethod
    def _finish_figure(fig: Any, axes: Any, xlabel: str, title: str | None) -> None:
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(0.99, 0.5),
            borderaxespad=0.0,
        )
        axes[-1, 0].set_xlabel(xlabel)
        fig.suptitle(title)
        fig.tight_layout(rect=(0.0, 0.0, 0.82, 0.96))


class RelativeErrorRenderer:
    """Asset 2 (static): per-timestep relative percentage error between a
    sparse selection of candidate iterations and the baseline -- the
    fine-grained precision instrument for when Asset 1's raw signals
    visually overlap.

    Timesteps where ``|u_base| < epsilon`` are shaded gray (masked) rather
    than plotted as a spurious spike -- a baseline control legitimately
    passing through ~0 makes the relative error numerically meaningless
    there, not informative.
    """

    def __init__(self, *, epsilon: float = DEFAULT_EPSILON) -> None:
        """
        Args:
            epsilon: relative-error denominator floor.

        Raises:
            ValueError: If `epsilon` is not positive.
        """
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}.")
        self.epsilon = epsilon

    def render(
        self,
        baseline_U: np.ndarray,
        candidate_history: np.ndarray,
        *,
        selection: IterationSelection | None = None,
        show_aggregate: bool = True,
        style: SignalStyle = SignalStyle(),
    ) -> Figure:
        """
        Args:
            baseline_U: reference control, shape ``(T, m)`` or ``(B, T, m)``.
            candidate_history: per-iteration control iterates, shape
                ``(I, T, m)`` or ``(I, B, T, m)``.
            selection: which iterations to overlay; defaults to a
                log-spaced sparse subset.
            show_aggregate: overlay the aggregate (L2-norm-ratio) error as a
                heavy dash-dot line on every subplot.
            style: slicing/labeling/title knobs (see `SignalStyle`).

        Returns:
            The Figure, one log-scale subplot per control component.
        """
        sample_index, dim_labels = style.sample_index, style.dim_labels
        xlabel, title = style.xlabel, style.title
        baseline, history, horizon, control_dim = _prepare(
            baseline_U,
            candidate_history,
            sample_index=sample_index,
            dim_labels=dim_labels,
        )
        selection = _resolve_selection(selection, history.shape[0])
        indices, alphas = selection.indices, _candidate_alphas(selection.indices)
        t = np.arange(horizon)
        masked = np.abs(baseline) < self.epsilon  # (T, m)

        with styled_figure():
            fig, axes = plt.subplots(
                nrows=control_dim,
                ncols=1,
                sharex=True,
                squeeze=False,
                figsize=(10, max(3.0, 2.5 * control_dim)),
            )
            any_masked = False
            for j in range(control_dim):
                ax = axes[j, 0]
                for iteration, alpha in zip(indices, alphas):
                    is_last = iteration == indices[-1]
                    error = compute_relative_error(
                        baseline[:, j], history[iteration, :, j], epsilon=self.epsilon
                    )
                    error = np.where(masked[:, j], np.nan, error)
                    ax.plot(
                        t,
                        error,
                        color=CANDIDATE_COLOR,
                        alpha=float(alpha),
                        linewidth=2.2 if is_last else 1.2,
                        label=f"Candidate (iter {iteration})" if is_last else None,
                    )
                if show_aggregate:
                    aggregate = compute_aggregate_relative_error(
                        baseline, history[indices[-1]], epsilon=self.epsilon
                    )
                    ax.plot(
                        t,
                        aggregate,
                        color=AGGREGATE_COLOR,
                        linewidth=2.4,
                        linestyle="-.",
                        label="Aggregate (L2 ratio)",
                    )
                if masked[:, j].any():
                    any_masked = True
                    for t_idx in np.flatnonzero(masked[:, j]):
                        ax.axvspan(
                            t_idx - 0.5, t_idx + 0.5, color=MASKED_SHADE, zorder=0
                        )
                ax.set_yscale("log")
                ax.set_ylabel(
                    f"Rel. error {dim_labels[j] if dim_labels else f'$u_{{{j}}}$'} (%)"
                )

            handles, labels = axes[0, 0].get_legend_handles_labels()
            if any_masked:
                handles.append(Patch(color=MASKED_SHADE, label="baseline ≈ 0 (masked)"))
                labels.append("baseline ≈ 0 (masked)")
            fig.legend(
                handles,
                labels,
                loc="center left",
                bbox_to_anchor=(0.99, 0.5),
                borderaxespad=0.0,
            )
            axes[-1, 0].set_xlabel(xlabel)
            fig.suptitle(title or "Relative Error: Candidate vs. Baseline")
            fig.tight_layout(rect=(0.0, 0.0, 0.82, 0.96))
        return fig


#: Ordinal colormap for "position in the optimization" (iteration recency):
#: one hue family in monotone light -> dark steps, so the reader sees
#: iteration order in the color itself rather than needing a colorbar --
#: matplotlib's built-in Blue -> Purple sequential map. Sampled away from its
#: near-white low end (illegible against a white figure background) through
#: to its darkest step.
_ITERATION_CMAP = "BuPu"
_ITERATION_CMAP_RANGE = (0.35, 0.95)


def _resolve_explicit_indices(
    selected_iterations: Sequence[int], num_iterations: int
) -> np.ndarray:
    """Sorted, de-duplicated, bounds-checked array of caller-chosen iteration
    indices -- the ``"explicit"`` `IterationSelection.mode`, used wherever the
    caller (not a sparse log-spaced default) picks exactly which iterations
    to overlay.

    Raises:
        ValueError: If `selected_iterations` is empty, or any entry falls
            outside ``[0, num_iterations)``.
    """
    indices = np.array(sorted({int(i) for i in selected_iterations}))
    if indices.size == 0:
        raise ValueError("selected_iterations must be non-empty.")
    if indices[0] < 0 or indices[-1] >= num_iterations:
        raise ValueError(
            f"selected_iterations must all be in [0, {num_iterations}), "
            f"got {indices.tolist()}."
        )
    return indices


def _iteration_colors(indices: np.ndarray) -> list:
    """One color per entry of `indices`, light blue (earliest) to dark purple
    (latest) along `_ITERATION_CMAP` -- the same color identifies a given
    iteration across every subplot (each control component and the shared
    error panel alike), so the reader never has to color-match across
    figures, only within one."""
    cmap = plt.get_cmap(_ITERATION_CMAP)
    if len(indices) == 1:
        return [cmap(_ITERATION_CMAP_RANGE[1])]
    lo, hi = _ITERATION_CMAP_RANGE
    return [cmap(t) for t in np.linspace(lo, hi, len(indices))]


def plot_sparse_signals_and_errors(
    baseline_U: np.ndarray,
    candidate_history: np.ndarray,
    selected_iterations: Sequence[int],
    *,
    epsilon: float = DEFAULT_EPSILON,
    style: SignalStyle = SignalStyle(),
) -> Figure:
    """The ``m + 1``-subplot diagnostic for a small, explicitly chosen set of
    iterations: one subplot per control component (a line per entry of
    `selected_iterations`, colored by an ordinal light-blue -> dark-purple
    ramp keyed to iteration recency, plus the Riccati baseline as a bold
    black dashed reference), stacked above one shared aggregate relative-
    error subplot using the identical per-iteration colors.

    Distinct from `plot_linked_signals_and_error` (which auto-subsamples a
    dense log-spaced selection, every candidate drawn in one alpha-faded
    hue): here the caller names the exact iterations, each rendered in its
    own color, so a handful of iterates (e.g. ``[0, 5, 20, 100]``) read as
    distinct curves rather than overlapping strokes of one color.

    Args:
        baseline_U: reference (Riccati-optimal) control, shape ``(T, m)`` or
            ``(B, T, m)``.
        candidate_history: per-iteration control iterates, shape ``(I, T,
            m)`` or ``(I, B, T, m)``.
        selected_iterations: the exact iteration indices to overlay, e.g.
            ``[0, 5, 20, 100]``; order-independent and de-duplicated, but
            every entry must lie in ``[0, I)``.
        epsilon: relative-error denominator floor.
        style: slicing/labeling/title knobs (see `SignalStyle`).

    Returns:
        The Figure: ``control_dim`` signal rows + one shared aggregate
        relative-error row (``control_dim + 1`` total).

    Raises:
        ValueError: If `selected_iterations` is empty or any entry falls
            outside ``[0, I)``, or `candidate_history`/`dim_labels` disagree
            with `baseline_U` (see `_prepare`).
    """
    sample_index, dim_labels = style.sample_index, style.dim_labels
    xlabel, title = style.xlabel, style.title
    baseline, history, horizon, control_dim = _prepare(
        baseline_U, candidate_history, sample_index=sample_index, dim_labels=dim_labels
    )
    indices = _resolve_explicit_indices(selected_iterations, history.shape[0])
    colors = _iteration_colors(indices)
    t = np.arange(horizon)

    with styled_figure():
        fig, axes = plt.subplots(
            nrows=control_dim + 1,
            ncols=1,
            sharex=True,
            squeeze=False,
            figsize=(10, max(4.0, 2.2 * (control_dim + 1))),
        )
        for j in range(control_dim):
            ax = axes[j, 0]
            ax.plot(
                t,
                baseline[:, j],
                color="black",
                linestyle="-",
                linewidth=0.5,
                label="Optimal (Riccati)",
                zorder=10,
            )
            for iteration, color in zip(indices, colors):
                ax.plot(
                    t,
                    history[iteration, :, j],
                    color=color,
                    linewidth=2.0,
                    linestyle="--",
                    label=f"Gradient Descent Controller at Iteration {iteration}",
                )
            ax.set_ylabel(dim_labels[j] if dim_labels else f"$u_{{t,{j}}}$")

        error_ax = axes[-1, 0]
        for iteration, color in zip(indices, colors):
            error = compute_aggregate_relative_error(
                baseline, history[iteration], epsilon=epsilon
            )
            error_ax.plot(t, error, color=color, linewidth=1.0)
        error_ax.set_yscale("log")
        error_ax.set_ylabel("Relative Error (%)")
        error_ax.set_xlabel(xlabel)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(0.99, 0.5),
            borderaxespad=0.0,
        )
        fig.suptitle(title or "Recovered Control Signal vs. Optimal Feedback Law")
        fig.tight_layout(rect=(0.0, 0.0, 0.82, 0.96))
    return fig


def plot_linked_signals_and_error(
    baseline_U: np.ndarray,
    candidate_history: np.ndarray,
    *,
    selection: IterationSelection | None = None,
    epsilon: float = DEFAULT_EPSILON,
    style: SignalStyle = SignalStyle(),
) -> Figure:
    """Asset 1 + Asset 2, linked: control signals on top (one row per
    component), the aggregate (L2-norm-ratio) relative error beneath them
    (one shared row) -- so overlap in the signal panel above is explained by
    the precision panel below it, mirroring the stacked cost/relative-error
    idiom in `viz.plots.cost_analysis.plot_empirical_vs_theoretical_cost`.

    Args:
        baseline_U: reference control, shape ``(T, m)`` or ``(B, T, m)``.
        candidate_history: per-iteration control iterates, shape
            ``(I, T, m)`` or ``(I, B, T, m)``.
        selection: which iterations to overlay; defaults to a log-spaced
            sparse subset.
        epsilon: relative-error denominator floor.
        style: slicing/labeling/title knobs (see `SignalStyle`).

    Returns:
        The Figure: ``control_dim`` signal rows + one shared error row.
    """
    sample_index, dim_labels = style.sample_index, style.dim_labels
    xlabel, title = style.xlabel, style.title
    baseline, history, horizon, control_dim = _prepare(
        baseline_U, candidate_history, sample_index=sample_index, dim_labels=dim_labels
    )
    selection = _resolve_selection(selection, history.shape[0])
    indices, alphas = selection.indices, _candidate_alphas(selection.indices)
    t = np.arange(horizon)

    with styled_figure():
        fig, axes = plt.subplots(
            nrows=control_dim + 1,
            ncols=1,
            sharex=True,
            squeeze=False,
            figsize=(10, max(4.0, 2.2 * (control_dim + 1))),
        )
        for j in range(control_dim):
            ax = axes[j, 0]
            ControlSignalRenderer._draw_component(
                ax, t, baseline[:, j], history[:, :, j], indices, alphas
            )
            ax.set_ylabel(dim_labels[j] if dim_labels else f"$u_{{{j}}}$")

        error_ax = axes[-1, 0]
        for iteration, alpha in zip(indices, alphas):
            is_last = iteration == indices[-1]
            aggregate = compute_aggregate_relative_error(
                baseline, history[iteration], epsilon=epsilon
            )
            error_ax.plot(
                t,
                aggregate,
                color=AGGREGATE_COLOR,
                alpha=float(alpha),
                linewidth=2.2 if is_last else 1.2,
            )
        error_ax.set_yscale("log")
        error_ax.set_ylabel("Rel. error (%)")
        error_ax.set_xlabel(xlabel)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(0.99, 0.5),
            borderaxespad=0.0,
        )
        fig.suptitle(title or "Control Signals with Linked Relative Error")
        fig.tight_layout(rect=(0.0, 0.0, 0.82, 0.96))
    return fig
