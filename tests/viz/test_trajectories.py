import numpy as np
import pytest
import torch
from matplotlib.figure import Figure

from mbl.viz.plots.trajectories import plot_trajectory_comparison


def _make_state_trajectories(batch: int, horizon: int, dim: int, seed: int):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(batch, horizon + 1, dim))


def test_plot_trajectory_comparison_with_batched_series_returns_figure() -> None:
    batch, horizon, dim = 50, 200, 4
    learned = _make_state_trajectories(batch, horizon, dim, seed=0)
    riccati = _make_state_trajectories(batch, horizon, dim, seed=1)

    fig = plot_trajectory_comparison(
        {"Learned Controller": learned, "Riccati Optimal": riccati},
        variable_name="x",
        sample_index=3,
    )

    assert isinstance(fig, Figure)
    assert len(fig.axes) == dim
    for ax in fig.axes:
        assert len(ax.get_lines()) == 2


def test_plot_trajectory_comparison_with_unbatched_series_returns_figure() -> None:
    horizon, dim = 200, 2
    rng = np.random.default_rng(0)
    control = rng.normal(size=(horizon, dim))

    fig = plot_trajectory_comparison({"Control Input": control}, variable_name="u")

    assert isinstance(fig, Figure)
    assert len(fig.axes) == dim


def test_plot_trajectory_comparison_accepts_torch_tensors() -> None:
    horizon, dim = 100, 3
    observations = torch.randn(horizon, dim)

    fig = plot_trajectory_comparison({"Observed": observations}, variable_name="y")  # type: ignore[dict-item]  # tensor inputs are accepted (to_numpy ingress)

    assert isinstance(fig, Figure)
    assert len(fig.axes) == dim


def test_plot_trajectory_comparison_uses_custom_dim_labels() -> None:
    horizon, dim = 50, 2
    rng = np.random.default_rng(0)
    series = {"Sample": rng.normal(size=(horizon, dim))}

    fig = plot_trajectory_comparison(
        series, dim_labels=["position", "velocity"], title="State Trajectory"
    )

    ylabels = [ax.get_ylabel() for ax in fig.axes]
    assert ylabels == ["position", "velocity"]


def test_plot_trajectory_comparison_rejects_empty_series() -> None:
    with pytest.raises(ValueError, match="at least one"):
        plot_trajectory_comparison({})


def test_plot_trajectory_comparison_rejects_mismatched_dimensions() -> None:
    rng = np.random.default_rng(0)
    series = {
        "a": rng.normal(size=(200, 4)),
        "b": rng.normal(size=(200, 3)),
    }
    with pytest.raises(ValueError, match="same dimension"):
        plot_trajectory_comparison(series)


def test_plot_trajectory_comparison_rejects_wrong_length_dim_labels() -> None:
    rng = np.random.default_rng(0)
    series = {"a": rng.normal(size=(200, 4))}
    with pytest.raises(ValueError, match="dim_labels"):
        plot_trajectory_comparison(series, dim_labels=["only_one"])


def test_plot_trajectory_comparison_rejects_bad_ndim() -> None:
    with pytest.raises(ValueError, match="horizon, dim"):
        plot_trajectory_comparison({"a": np.zeros(10)})


def test_plot_trajectory_comparison_colors_recognized_model_families() -> None:
    """A label embedding a known model family (as the dashboard's
    `option_label` composite labels do) should get that family's fixed
    color, not an arbitrary cycled one."""
    from mbl.applications.styles import MODEL_FAMILY_COLORS

    rng = np.random.default_rng(0)
    series = {
        "run_20260101_000000_analytic :: trajectory_states": rng.normal(size=(50, 4)),
        "run_20260101_000000_neural :: trajectory_states": rng.normal(size=(50, 4)),
    }

    fig = plot_trajectory_comparison(series)

    ax = fig.axes[0]
    colors = {line.get_label(): line.get_color() for line in ax.get_lines()}
    assert (
        colors["run_20260101_000000_analytic :: trajectory_states"]
        == (MODEL_FAMILY_COLORS["analytic"])
    )
    assert (
        colors["run_20260101_000000_neural :: trajectory_states"]
        == (MODEL_FAMILY_COLORS["neural"])
    )


def test_plot_trajectory_comparison_uses_solid_for_analytic_dashed_for_unfolded() -> (
    None
):
    rng = np.random.default_rng(0)
    series = {
        "run_..._analytic :: trajectory_states": rng.normal(size=(50, 2)),
        "run_..._unfolded_learned_step_size :: trajectory_states": rng.normal(
            size=(50, 2)
        ),
    }

    fig = plot_trajectory_comparison(series)

    ax = fig.axes[0]
    styles = {line.get_label(): line.get_linestyle() for line in ax.get_lines()}
    assert styles["run_..._analytic :: trajectory_states"] == "-"
    assert styles["run_..._unfolded_learned_step_size :: trajectory_states"] == "--"


def test_plot_trajectory_comparison_uses_one_shared_legend_not_one_per_subplot() -> (
    None
):
    rng = np.random.default_rng(0)
    series = {
        "a": rng.normal(size=(50, 3)),
        "b": rng.normal(size=(50, 3)),
    }

    fig = plot_trajectory_comparison(series)

    assert all(ax.get_legend() is None for ax in fig.axes)
    assert fig.legends
    assert {t.get_text() for t in fig.legends[0].get_texts()} == {"a", "b"}
