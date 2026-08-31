import numpy as np
import pytest
import torch
from matplotlib.figure import Figure

from mbl.viz.style import ReferenceLineStyle
from mbl.viz.plots.training_curves import TrainingCurveStyle, plot_training_curves


def test_plot_training_curves_returns_figure_with_multiple_series_and_references() -> (
    None
):
    n_epochs = 300
    train_loss = np.geomspace(10.0, 0.05, n_epochs)
    eval_loss = np.geomspace(12.0, 0.08, n_epochs)

    fig = plot_training_curves(
        {"Train Loss": train_loss, "Eval Loss": eval_loss},
        reference_lines={"Theoretical Optimal": 0.02},
        style=TrainingCurveStyle(log_scale=True),
    )

    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_yscale() == "log"
    # 2 curves + 1 reference line = 3 plotted lines.
    assert len(ax.get_lines()) == 3
    legend_labels = {t.get_text() for t in ax.get_legend().get_texts()}
    # Each curve's own final (last-epoch) value is folded into its label.
    assert legend_labels == {
        f"Train Loss (final={train_loss[-1]:.4g})",
        f"Eval Loss (final={eval_loss[-1]:.4g})",
        "Theoretical Optimal: 0.02",
    }


def test_plot_training_curves_accepts_torch_tensor_curves() -> None:
    losses = torch.linspace(5.0, 0.1, 50)

    fig = plot_training_curves({"Loss": losses})  # type: ignore[dict-item]  # tensor inputs are accepted (to_numpy ingress)

    assert isinstance(fig, Figure)
    assert len(fig.axes[0].get_lines()) == 1


def test_plot_training_curves_rejects_empty_curves() -> None:
    with pytest.raises(ValueError, match="at least one"):
        plot_training_curves({})


def test_plot_training_curves_rejects_non_1d_curve() -> None:
    with pytest.raises(ValueError, match="must be 1D"):
        plot_training_curves({"bad": np.zeros((10, 2))})


def test_plot_training_curves_colors_known_model_families() -> None:
    from mbl.applications.styles import MODEL_FAMILY_COLORS

    fig = plot_training_curves({"cocp": np.linspace(5, 1, 20)})
    line = fig.axes[0].get_lines()[0]
    assert line.get_color() == MODEL_FAMILY_COLORS["cocp"]


def test_plot_training_curves_sparsifies_markers_for_long_runs() -> None:
    """A dense marker on every one of 300 epochs would be an unreadable
    blob -- markevery should thin them out."""
    fig = plot_training_curves({"Loss": np.geomspace(10.0, 0.1, 300)})
    line = fig.axes[0].get_lines()[0]
    assert line.get_markevery() not in (None, 1)


def test_plot_training_curves_reference_line_styles_overrides_color() -> None:
    fig = plot_training_curves(
        {"Loss": np.linspace(5.0, 1.0, 10)},
        reference_lines={"J_SDP": 2.0},
        reference_line_styles={"J_SDP": ReferenceLineStyle(color="#abcdef")},
    )
    ref_line = next(
        line for line in fig.axes[0].get_lines() if line.get_label().startswith("J_SDP")
    )
    assert ref_line.get_color() == "#abcdef"


def test_plot_training_curves_explicit_ylim_is_honored() -> None:
    fig = plot_training_curves(
        {"Loss": np.linspace(5.0, 1.0, 10)},
        reference_lines={"Far floor": -1000.0},
        style=TrainingCurveStyle(ylim=(0.0, 6.0)),
    )
    assert fig.axes[0].get_ylim() == pytest.approx((0.0, 6.0))


def test_plot_training_curves_autoscale_to_curves_ignores_a_distant_reference_line() -> (
    None
):
    """The curve data spans [1, 5]; a reference line at -1000 must not
    stretch the visible y-axis down to it when `autoscale_to_curves=True`
    (NB04 reference-bounds plan Sec 2.4)."""
    fig = plot_training_curves(
        {"Loss": np.linspace(5.0, 1.0, 10)},
        reference_lines={"Far floor": -1000.0},
        style=TrainingCurveStyle(autoscale_to_curves=True),
    )
    low, high = fig.axes[0].get_ylim()
    assert low > -100.0
    assert high < 100.0


def test_plot_training_curves_ylim_wins_over_autoscale_to_curves() -> None:
    fig = plot_training_curves(
        {"Loss": np.linspace(5.0, 1.0, 10)},
        style=TrainingCurveStyle(ylim=(-3.0, 3.0), autoscale_to_curves=True),
    )
    assert fig.axes[0].get_ylim() == pytest.approx((-3.0, 3.0))


def test_plot_training_curves_places_legend_outside_the_axes() -> None:
    fig = plot_training_curves({"Loss": np.linspace(5.0, 0.1, 50)})
    assert fig.subplotpars.right < 0.85


def test_plot_training_curves_legend_reports_the_final_epoch_value() -> None:
    values = np.linspace(5.0, 0.123, 50)
    fig = plot_training_curves({"$J=4$": values})
    (label,) = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert label == f"$J=4$ (final={values[-1]:.4g})"


def test_plot_training_curves_show_markers_false_disables_markers() -> None:
    fig = plot_training_curves(
        {"Loss": np.linspace(5.0, 0.1, 50)},
        style=TrainingCurveStyle(show_markers=False),
    )
    line = fig.axes[0].get_lines()[0]
    assert line.get_marker() in (None, "None", "none")


def test_plot_training_curves_honors_linestyle_and_linewidth() -> None:
    fig = plot_training_curves(
        {"Loss": np.linspace(5.0, 0.1, 50)},
        style=TrainingCurveStyle(linestyle="--", linewidth=3.5),
    )
    line = fig.axes[0].get_lines()[0]
    assert line.get_linestyle() == "--"
    assert line.get_linewidth() == 3.5


def test_plot_training_curves_legend_inside_draws_within_the_axes() -> None:
    fig = plot_training_curves(
        {"Loss": np.linspace(5.0, 0.1, 50)},
        style=TrainingCurveStyle(legend_inside=True),
    )
    assert fig.axes[0].get_legend() is not None
    assert fig.subplotpars.right > 0.85
