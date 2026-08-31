import numpy as np
import pytest
from matplotlib.figure import Figure

from mbl.workbench.analysis import compute_cumulative_average_cost  # noqa: F401
from mbl.viz.style import ReferenceLineStyle
from mbl.viz.plots.cost_analysis import (
    CostComparisonStyle,
    CostVsDepthStyle,
    plot_cost_vs_unfolding_depth,
    plot_cumulative_cost_over_time,
    plot_empirical_vs_theoretical_cost,
)


def test_compute_cumulative_average_cost_matches_direct_computation() -> None:
    rng = np.random.default_rng(0)
    batch, horizon, n, m = 5, 8, 3, 2
    states = rng.normal(size=(batch, horizon + 1, n))
    controls = rng.normal(size=(batch, horizon, m))

    result = compute_cumulative_average_cost(states, controls)

    stage_cost = (states[:, :horizon] ** 2).sum(axis=-1) + (controls**2).sum(axis=-1)
    expected = np.cumsum(stage_cost.mean(axis=0)) / np.arange(1, horizon + 1)
    assert result.shape == (horizon,)
    assert np.allclose(result, expected)


def test_compute_cumulative_average_cost_supports_unbatched_input() -> None:
    horizon, n, m = 6, 2, 1
    states = np.ones((horizon + 1, n))
    controls = np.ones((horizon, m))

    result = compute_cumulative_average_cost(states, controls)

    # Every stage cost is identical (n + m), so the running average is flat.
    assert np.allclose(result, n + m)


def test_compute_cumulative_average_cost_respects_custom_q_and_r() -> None:
    horizon, n, m = 4, 2, 1
    states = np.ones((1, horizon + 1, n))
    controls = np.ones((1, horizon, m))
    Q = np.diag([2.0, 3.0])
    R = np.array([[4.0]])

    result = compute_cumulative_average_cost(states, controls, Q=Q, R=R)

    assert np.allclose(result, 2.0 + 3.0 + 4.0)


def test_plot_cumulative_cost_over_time_returns_figure_with_multiple_curves() -> None:
    horizon = 60
    t = np.arange(1, horizon + 1)
    riccati_cost = 2.0 + 3.0 / t
    unfolded_cost = 2.2 + 4.0 / t

    fig = plot_cumulative_cost_over_time(
        {"analytic": riccati_cost, "unfolded_learned_step_size": unfolded_cost},
        reference_lines={"cocp_lower_bound": 1.8},
    )

    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert len(ax.get_lines()) == 3  # 2 curves + 1 reference line
    assert ax.get_xlabel() == "Time step"


def test_plot_cumulative_cost_over_time_supports_log_scale() -> None:
    fig = plot_cumulative_cost_over_time({"a": np.linspace(10, 1, 20)}, log_scale=True)
    assert fig.axes[0].get_yscale() == "log"


def test_plot_cumulative_cost_over_time_places_legend_outside_the_axes() -> None:
    fig = plot_cumulative_cost_over_time({"a": np.linspace(10, 1, 20)})
    assert fig.subplotpars.right < 0.85


def test_plot_cumulative_cost_over_time_rejects_empty_costs() -> None:
    with pytest.raises(ValueError, match="at least one"):
        plot_cumulative_cost_over_time({})


def test_plot_cumulative_cost_over_time_rejects_non_1d_curve() -> None:
    with pytest.raises(ValueError, match="must be 1D"):
        plot_cumulative_cost_over_time({"bad": np.zeros((5, 2))})


def test_known_model_names_get_their_fixed_color() -> None:
    from mbl.applications.styles import MODEL_FAMILY_COLORS

    fig = plot_cumulative_cost_over_time({"cocp": np.linspace(5, 1, 10)})
    line = fig.axes[0].get_lines()[0]
    assert line.get_color() == MODEL_FAMILY_COLORS["cocp"]


def test_plot_cost_vs_unfolding_depth_returns_figure_with_baselines() -> None:
    k_values = list(range(1, 11))
    step_size_cost = [2.5 - 0.05 * k for k in k_values]
    step_size_and_matrix_cost = [2.4 - 0.04 * k for k in k_values]

    fig = plot_cost_vs_unfolding_depth(
        k_values,
        {
            "unfolded_learned_step_size": step_size_cost,
            "unfolded_learned_step_size_and_matrix": step_size_and_matrix_cost,
        },
        reference_lines={
            "truncated_riccati": 2.6,
            "cocp": 2.1,
            "neural": 2.8,
            "cocp_lower_bound": 1.9,
        },
    )

    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    # 2 K-dependent curves + 4 flat baselines = 6 lines total.
    assert len(ax.get_lines()) == 6
    assert ax.get_xlabel() == "Unfolding depth K"


def test_plot_cost_vs_unfolding_depth_uses_distinct_markers_per_curve() -> None:
    k_values = [1, 2, 3]
    fig = plot_cost_vs_unfolding_depth(
        k_values,
        {"a": [3.0, 2.0, 1.5], "b": [3.2, 2.2, 1.7]},
    )
    markers = {line.get_marker() for line in fig.axes[0].get_lines()}
    assert len(markers) == 2


def test_plot_cost_vs_unfolding_depth_reference_line_styles_overrides_color() -> None:
    fig = plot_cost_vs_unfolding_depth(
        [1, 2, 3],
        {"a": [3.0, 2.0, 1.5]},
        reference_lines={"J_SDP": 2.0},
        reference_line_styles={"J_SDP": ReferenceLineStyle(color="#abcdef")},
    )
    ref_line = next(
        line for line in fig.axes[0].get_lines() if line.get_label().startswith("J_SDP")
    )
    assert ref_line.get_color() == "#abcdef"


def test_plot_cost_vs_unfolding_depth_explicit_ylim_is_honored() -> None:
    fig = plot_cost_vs_unfolding_depth(
        [1, 2, 3],
        {"a": [3.0, 2.0, 1.5]},
        reference_lines={"Far floor": -1000.0},
        style=CostVsDepthStyle(ylim=(0.0, 4.0)),
    )
    assert fig.axes[0].get_ylim() == pytest.approx((0.0, 4.0))


def test_plot_cost_vs_unfolding_depth_autoscale_to_curves_ignores_distant_reference() -> (
    None
):
    """The curve data spans [1.5, 3.0]; a reference line at -1000 must not
    stretch the visible y-axis down to it (NB04 reference-bounds plan Sec
    2.4) -- exactly the Section 6 clutter problem this knob exists for."""
    fig = plot_cost_vs_unfolding_depth(
        [1, 2, 3],
        {"a": [3.0, 2.0, 1.5]},
        reference_lines={"Far floor": -1000.0},
        style=CostVsDepthStyle(autoscale_to_curves=True),
    )
    low, high = fig.axes[0].get_ylim()
    assert low > -100.0
    assert high < 100.0


def test_plot_cost_vs_unfolding_depth_uses_publication_style_markers() -> None:
    """Triangles/squares/diamonds, per the publication-style requirement,
    rather than generic circles."""
    fig = plot_cost_vs_unfolding_depth(
        [1, 2, 3], {"a": [3.0, 2.0, 1.5], "b": [3.2, 2.2, 1.7]}
    )
    markers = [line.get_marker() for line in fig.axes[0].get_lines()]
    assert markers == ["^", "s"]


def test_plot_cost_vs_unfolding_depth_places_legend_outside_the_axes() -> None:
    fig = plot_cost_vs_unfolding_depth([1, 2, 3], {"a": [3.0, 2.0, 1.0]})
    assert fig.subplotpars.right < 0.85


def test_plot_cost_vs_unfolding_depth_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="shape"):
        plot_cost_vs_unfolding_depth([1, 2, 3], {"a": [1.0, 2.0]})


def test_plot_cost_vs_unfolding_depth_rejects_empty_curves() -> None:
    with pytest.raises(ValueError, match="at least one"):
        plot_cost_vs_unfolding_depth([1, 2, 3], {})


def test_plot_empirical_vs_theoretical_cost_returns_figure_with_two_curves() -> None:
    horizon = 10
    empirical = np.linspace(3.0, 2.0, horizon)
    theoretical = np.linspace(2.9, 1.9, horizon)

    fig = plot_empirical_vs_theoretical_cost(empirical, theoretical)

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2
    cost_ax, error_ax = fig.axes
    assert len(cost_ax.get_lines()) == 2
    assert error_ax.get_xlabel() == "Time step"


def test_plot_empirical_vs_theoretical_cost_uses_custom_labels_in_legend() -> None:
    fig = plot_empirical_vs_theoretical_cost(
        np.linspace(3.0, 2.0, 5),
        np.linspace(2.9, 1.9, 5),
        style=CostComparisonStyle(
            empirical_label="MC (batch=1000)", theoretical_label="Analytic Riccati"
        ),
    )
    legend_labels = {t.get_text() for t in fig.axes[0].get_legend().get_texts()}
    assert legend_labels == {"MC (batch=1000)", "Analytic Riccati"}


def test_plot_empirical_vs_theoretical_cost_theoretical_curve_is_dashed() -> None:
    fig = plot_empirical_vs_theoretical_cost(
        np.linspace(3.0, 2.0, 5), np.linspace(2.9, 1.9, 5)
    )
    empirical_line, theoretical_line = fig.axes[0].get_lines()
    assert empirical_line.get_linestyle() == "-"
    assert theoretical_line.get_linestyle() == "--"


def test_plot_empirical_vs_theoretical_cost_supports_log_scale() -> None:
    fig = plot_empirical_vs_theoretical_cost(
        np.linspace(10, 1, 20),
        np.linspace(9, 0.9, 20),
        style=CostComparisonStyle(log_scale=True),
    )
    assert fig.axes[0].get_yscale() == "log"


def test_plot_empirical_vs_theoretical_cost_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="shape"):
        plot_empirical_vs_theoretical_cost(np.zeros(5), np.zeros(6))


def test_plot_empirical_vs_theoretical_cost_relative_error_panel_is_log_scale() -> None:
    fig = plot_empirical_vs_theoretical_cost(
        np.linspace(3.0, 2.0, 10), np.linspace(2.9, 1.9, 10)
    )
    error_ax = fig.axes[1]
    assert error_ax.get_yscale() == "log"
    assert error_ax.get_ylabel() == "Relative error (%)"
    assert len(error_ax.get_lines()) == 1


def test_plot_empirical_vs_theoretical_cost_relative_error_matches_direct_computation() -> (
    None
):
    empirical = np.array([3.0, 2.5, 2.2])
    theoretical = np.array([2.9, 2.4, 2.1])

    fig = plot_empirical_vs_theoretical_cost(empirical, theoretical)

    expected = 100.0 * np.abs(empirical - theoretical) / np.abs(theoretical)
    (line,) = fig.axes[1].get_lines()
    assert np.allclose(line.get_ydata(), expected)
