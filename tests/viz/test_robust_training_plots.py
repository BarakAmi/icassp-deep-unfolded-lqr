"""Phase 5 acceptance tests for `viz.plots.robust_training`
(docs/planning/03_studies/nb07_robust_training/robust_training_under_model_mismatch.md Sec 6/11): figures render
headless with bands; empty input raises; the paired cost/constraint figure
produces two axes.
"""

import pytest
from matplotlib.figure import Figure

from mbl.applications.uncertainty.noise import NoiseFamily
from mbl.viz.plots.robust_training import (
    BandCurveStyle,
    LearningDynamicsStyle,
    plot_band_curves,
    plot_learning_dynamics_with_constraint,
)
from mbl.workbench.robust_training_sweep import CurveWithBand


def _band(x: tuple, median: tuple, q25: tuple, q75: tuple) -> CurveWithBand:
    return CurveWithBand(x=x, median=median, q25=q25, q75=q75)


class TestPlotBandCurves:
    def test_renders_headless_with_bands(self) -> None:
        bands = {
            "unfolded_alpha": _band(
                (0.1, 1.0, 10.0), (1.0, 1.5, 3.0), (0.8, 1.3, 2.5), (1.2, 1.8, 3.5)
            ),
            "neural": _band(
                (0.1, 1.0, 10.0), (1.2, 2.0, 5.0), (1.0, 1.7, 4.0), (1.4, 2.3, 6.0)
            ),
        }
        fig = plot_band_curves(bands, style=BandCurveStyle(title="Test"))
        assert isinstance(fig, Figure)

    def test_with_reference_lines(self) -> None:
        bands = {"unfolded_alpha_p": _band((0, 1), (1.0, 1.1), (0.9, 1.0), (1.1, 1.2))}
        fig = plot_band_curves(bands, reference_lines={"nominal": 1.0})
        assert isinstance(fig, Figure)

    def test_log_x_and_log_y(self) -> None:
        bands = {
            "a": _band(
                (16, 256, 4096), (2.0, 1.5, 1.1), (1.8, 1.3, 1.0), (2.2, 1.7, 1.2)
            )
        }
        fig = plot_band_curves(bands, style=BandCurveStyle(log_x=True, log_y=True))
        assert isinstance(fig, Figure)

    def test_empty_bands_raises(self) -> None:
        with pytest.raises(ValueError, match="bands"):
            plot_band_curves({})

    def test_skips_empty_curve_without_raising(self) -> None:
        bands = {
            "no_learned_p": _band((), (), (), ()),
            "has_data": _band((1.0,), (2.0,), (1.5,), (2.5,)),
        }
        fig = plot_band_curves(bands)
        assert isinstance(fig, Figure)

    def test_categorical_axis_renders_with_family_tick_labels(self) -> None:
        # The noise-family (X) axis's levels are `NoiseFamily` members, not
        # numbers -- must not raise trying to `float()` them.
        bands = {
            "unfolded_alpha": _band(
                (
                    NoiseFamily.GAUSSIAN,
                    NoiseFamily.UNIFORM,
                    NoiseFamily.LAPLACE,
                    NoiseFamily.CAUCHY,
                ),
                (1.0, 1.1, 1.2, 1.3),
                (0.9, 1.0, 1.1, 1.2),
                (1.1, 1.2, 1.3, 1.4),
            ),
        }
        fig = plot_band_curves(bands)
        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        labels = [tick.get_text() for tick in ax.get_xticklabels()]
        assert labels == ["gaussian", "uniform", "laplace", "cauchy"]


class TestPlotLearningDynamicsWithConstraint:
    def test_renders_two_panels(self) -> None:
        cost_bands = {
            "unfolded_alpha_p": _band(
                (1, 2, 3), (5.0, 3.0, 2.0), (4.5, 2.5, 1.8), (5.5, 3.5, 2.2)
            ),
        }
        constraint_bands = {
            "unfolded_alpha_p": _band(
                (1, 2, 3), (0.1, 0.2, 0.3), (0.05, 0.15, 0.25), (0.15, 0.25, 0.35)
            ),
        }
        fig = plot_learning_dynamics_with_constraint(
            cost_bands, constraint_bands, style=LearningDynamicsStyle(title="Test")
        )
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 2

    def test_empty_cost_bands_raises(self) -> None:
        with pytest.raises(ValueError, match="cost_bands"):
            plot_learning_dynamics_with_constraint({}, {})
