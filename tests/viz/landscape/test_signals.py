"""Asset 1 (control signals) and Asset 2 (relative error) static renderers:
per-component figure structure, the hand-checked relative-error math, the
epsilon-floor/masking behavior guarding a baseline crossing ~0, and the
linked figure's shared x-axis."""

import numpy as np
import pytest

from mbl.viz.landscape.signals import (
    SignalStyle,
    ControlSignalRenderer,
    RelativeErrorRenderer,
    compute_aggregate_relative_error,
    compute_relative_error,
    plot_linked_signals_and_error,
    plot_sparse_signals_and_errors,
)

HORIZON, CONTROL_DIM, NUM_ITERATIONS = 6, 2, 10


def _baseline() -> np.ndarray:
    return np.stack(
        [np.linspace(1.0, 2.0, HORIZON), np.linspace(-1.0, 0.5, HORIZON)], axis=1
    )


def _history() -> np.ndarray:
    baseline = _baseline()
    # Each iteration halves the distance to the baseline -- a clean,
    # monotonically-converging synthetic history.
    return np.stack(
        [baseline + (baseline + 3.0) * (0.5**i) for i in range(NUM_ITERATIONS)], axis=0
    )


def test_compute_relative_error_hand_checked() -> None:
    baseline = np.array([[2.0, 4.0]])
    candidate = np.array([[2.2, 3.6]])
    error = compute_relative_error(baseline, candidate)
    assert np.allclose(error, [[10.0, 10.0]])


def test_compute_relative_error_applies_epsilon_floor() -> None:
    baseline = np.array([[0.0]])
    candidate = np.array([[0.001]])
    error = compute_relative_error(baseline, candidate, epsilon=1e-2)
    # denom floored to 1e-2 -> 100 * 0.001 / 1e-2 = 10.0, not inf/nan.
    assert np.isfinite(error).all()
    assert np.allclose(error, [[10.0]])


def test_compute_aggregate_relative_error_hand_checked() -> None:
    baseline = np.array([[3.0, 4.0]])  # norm 5
    candidate = np.array([[3.0, 4.4]])  # diff norm 0.4
    error = compute_aggregate_relative_error(baseline, candidate)
    assert error == pytest.approx([100.0 * 0.4 / 5.0])


def test_control_signal_renderer_has_one_subplot_per_component() -> None:
    fig = ControlSignalRenderer().render(_baseline(), _history())
    assert len(fig.axes) == CONTROL_DIM


def test_control_signal_renderer_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="candidate_history"):
        ControlSignalRenderer().render(_baseline(), _history()[:, :, :1])


def test_control_signal_renderer_rejects_bad_dim_labels_length() -> None:
    with pytest.raises(ValueError, match="dim_labels"):
        ControlSignalRenderer().render(
            _baseline(), _history(), style=SignalStyle(dim_labels=["only_one"])
        )


def test_control_signal_renderer_accepts_batched_inputs() -> None:
    baseline_batched = _baseline()[None]  # (1, T, m)
    history_batched = _history()[:, None]  # (I, 1, T, m)
    fig = ControlSignalRenderer().render(baseline_batched, history_batched)
    assert len(fig.axes) == CONTROL_DIM


def test_relative_error_renderer_has_one_subplot_per_component() -> None:
    fig = RelativeErrorRenderer().render(_baseline(), _history())
    assert len(fig.axes) == CONTROL_DIM


def test_relative_error_renderer_uses_log_scale_y_axis() -> None:
    fig = RelativeErrorRenderer().render(_baseline(), _history())
    for ax in fig.axes:
        assert ax.get_yscale() == "log"


def test_relative_error_renderer_masks_baseline_near_zero_timesteps() -> None:
    """A baseline crossing ~0 must be shaded (masked), not plotted as a
    spurious spike -- the line data at that timestep must be NaN."""
    baseline = _baseline().copy()
    baseline[3, 0] = 0.0  # exactly zero at t=3, component 0
    history = _history().copy()
    history[:, 3, 0] = 0.5  # a real, nonzero candidate value there

    renderer = RelativeErrorRenderer(epsilon=1e-6)
    fig = renderer.render(baseline, history)

    component_0_ax = fig.axes[0]
    # At least one plotted line (a non-aggregate candidate curve) must have
    # a NaN at the masked timestep (t=3), not a finite spurious spike.
    masked_found = False
    for line in component_0_ax.get_lines():
        y = np.asarray(line.get_ydata())
        if len(y) > 3 and np.isnan(y[3]):
            masked_found = True
    assert masked_found
    # And a shaded axvspan patch must be present for the masked region.
    assert len(component_0_ax.patches) > 0


def test_relative_error_renderer_rejects_non_positive_epsilon() -> None:
    with pytest.raises(ValueError, match="epsilon"):
        RelativeErrorRenderer(epsilon=0.0)


def test_relative_error_renderer_show_aggregate_toggle() -> None:
    fig_with = RelativeErrorRenderer().render(
        _baseline(), _history(), show_aggregate=True
    )
    fig_without = RelativeErrorRenderer().render(
        _baseline(), _history(), show_aggregate=False
    )
    lines_with = len(fig_with.axes[0].get_lines())
    lines_without = len(fig_without.axes[0].get_lines())
    assert lines_with > lines_without


def test_linked_signals_and_error_has_control_dim_plus_one_rows() -> None:
    fig = plot_linked_signals_and_error(_baseline(), _history())
    assert len(fig.axes) == CONTROL_DIM + 1


def test_linked_signals_and_error_shares_x_axis() -> None:
    fig = plot_linked_signals_and_error(_baseline(), _history())
    axes = fig.axes
    # sharex=True: every axis's shared-x group contains every other axis.
    for ax in axes:
        assert ax in axes[0].get_shared_x_axes().get_siblings(axes[0])


def test_sparse_signals_and_errors_has_control_dim_plus_one_rows() -> None:
    fig = plot_sparse_signals_and_errors(_baseline(), _history(), [0, 3, 9])
    assert len(fig.axes) == CONTROL_DIM + 1


def test_sparse_signals_and_errors_uses_log_scale_on_error_subplot() -> None:
    fig = plot_sparse_signals_and_errors(_baseline(), _history(), [0, 3, 9])
    assert fig.axes[-1].get_yscale() == "log"


def test_sparse_signals_and_errors_draws_one_line_per_iteration_plus_baseline() -> None:
    selected = [0, 3, 9]
    fig = plot_sparse_signals_and_errors(_baseline(), _history(), selected)
    signal_ax = fig.axes[0]
    # One line per selected iteration, plus the Riccati baseline.
    assert len(signal_ax.get_lines()) == len(selected) + 1


def test_sparse_signals_and_errors_error_subplot_has_one_line_per_iteration() -> None:
    selected = [0, 3, 9]
    fig = plot_sparse_signals_and_errors(_baseline(), _history(), selected)
    error_ax = fig.axes[-1]
    assert len(error_ax.get_lines()) == len(selected)


def test_sparse_signals_and_errors_colors_match_across_subplots() -> None:
    """The same iteration must be drawn in the identical color in every
    component subplot and the shared error subplot -- the whole point of the
    ordinal color ramp is that a curve's color alone identifies which
    iteration it is, consistently across the figure."""
    selected = [0, 3, 9]
    fig = plot_sparse_signals_and_errors(_baseline(), _history(), selected)
    # The baseline is drawn first on each signal subplot, so its candidate
    # iterates start at index 1.
    signal_lines = fig.axes[0].get_lines()[1:]
    error_lines = fig.axes[-1].get_lines()
    for k in range(len(selected)):
        assert signal_lines[k].get_color() == error_lines[k].get_color()


def test_sparse_signals_and_errors_baseline_is_black_solid() -> None:
    selected = [0, 3, 9]
    fig = plot_sparse_signals_and_errors(_baseline(), _history(), selected)
    baseline_line = fig.axes[0].get_lines()[0]  # drawn first, beneath every iterate
    assert baseline_line.get_color() == "black"
    assert baseline_line.get_linestyle() == "-"


def test_sparse_signals_and_errors_deduplicates_and_sorts_indices() -> None:
    fig_dupes = plot_sparse_signals_and_errors(_baseline(), _history(), [9, 0, 3, 3, 0])
    fig_clean = plot_sparse_signals_and_errors(_baseline(), _history(), [0, 3, 9])
    assert len(fig_dupes.axes[0].get_lines()) == len(fig_clean.axes[0].get_lines())


def test_sparse_signals_and_errors_rejects_empty_selection() -> None:
    with pytest.raises(ValueError, match="selected_iterations"):
        plot_sparse_signals_and_errors(_baseline(), _history(), [])


def test_sparse_signals_and_errors_rejects_out_of_range_iteration() -> None:
    with pytest.raises(ValueError, match="selected_iterations"):
        plot_sparse_signals_and_errors(_baseline(), _history(), [0, NUM_ITERATIONS])
