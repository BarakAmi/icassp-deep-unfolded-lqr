"""Phase 5 acceptance tests for `viz.plots.ood`
(docs/planning/03_studies/nb06_ood_generalization/zero_shot_ood_generalization_benchmark.md Sec 5): figures render
headless with bands; empty input raises; the paired cost/constraint figure
produces two axes; the depth-ablation figure produces one panel per
contender; the phase portrait produces two axes and accepts trajectories of
differing lengths/dimension.
"""

import numpy as np
import pytest
from matplotlib.figure import Figure

from mbl.applications.styles import NB06_CONTENDER_COLORS, NB06_DASHED_CONTENDERS
from mbl.viz.plots.ood import (
    OODCostConstraintStyle,
    plot_ood_cost_and_constraint,
    plot_ood_cost_vs_compute,
    plot_ood_depth_ablation,
    plot_ood_interaction_heatmap,
    plot_ood_mandate_scatter,
    plot_ood_phase_portrait,
)
from mbl.viz.style import series_color, series_linestyle
from mbl.workbench.ood_sweep import CurveWithBand


def _band(x: tuple, median: tuple, q25: tuple, q75: tuple) -> CurveWithBand:
    return CurveWithBand(x=x, median=median, q25=q25, q75=q75)


class TestNB06ContenderStyleResolution:
    """Regression anchor for NB06 plan Sec 5.0/11: an NB06 instance label
    (e.g. "unfolded_alpha_p") must resolve to its OWN registered color, not
    a palette-cycled fallback keyed by its position in whatever bands
    mapping happens to be plotted -- the bug that drew the flagship
    contender in "unfolded_fixed" (the untrained baseline)'s color."""

    @pytest.mark.parametrize("label", list(NB06_CONTENDER_COLORS))
    def test_every_nb06_label_resolves_to_its_registered_color(
        self, label: str
    ) -> None:
        # A deliberately WRONG fallback_index: if the label were falling
        # through to positional cycling, this would change the result.
        assert series_color(label, fallback_index=99) == NB06_CONTENDER_COLORS[label]

    def test_removing_one_label_does_not_change_another_labels_color(self) -> None:
        full = dict(NB06_CONTENDER_COLORS)
        colors_full = {
            label: series_color(label, fallback_index=i) for i, label in enumerate(full)
        }
        reduced = {k: v for k, v in full.items() if k != "unfolded_alpha"}
        colors_reduced = {
            label: series_color(label, fallback_index=i)
            for i, label in enumerate(reduced)
        }
        for label in reduced:
            assert colors_reduced[label] == colors_full[label]

    @pytest.mark.parametrize("label", list(NB06_DASHED_CONTENDERS))
    def test_iterative_contenders_render_dashed(self, label: str) -> None:
        assert series_linestyle(label) == "--"

    def test_closed_form_contenders_render_solid(self) -> None:
        for label in ("truncated_riccati", "cocp", "cocp_lower_bound"):
            assert series_linestyle(label) == "-"


class TestPlotOodCostAndConstraint:
    def test_renders_two_panels_with_bands(self) -> None:
        cost_bands = {
            "unfolded_alpha": _band(
                (0.1, 1.0, 10.0), (0.1, 0.3, 0.8), (0.08, 0.25, 0.7), (0.12, 0.35, 0.9)
            ),
            "neural": _band(
                (0.1, 1.0, 10.0), (0.2, 0.5, 1.5), (0.15, 0.4, 1.2), (0.25, 0.6, 1.8)
            ),
        }
        saturation_bands = {
            "unfolded_alpha": _band(
                (0.1, 1.0, 10.0), (0.1, 0.4, 0.9), (0.05, 0.3, 0.8), (0.15, 0.5, 1.0)
            ),
            "neural": _band(
                (0.1, 1.0, 10.0), (0.05, 0.2, 0.6), (0.0, 0.1, 0.5), (0.1, 0.3, 0.7)
            ),
        }
        fig = plot_ood_cost_and_constraint(
            cost_bands, saturation_bands, style=OODCostConstraintStyle(title="Test")
        )
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 2

    def test_with_reference_lines(self) -> None:
        cost_bands = {
            "truncated_riccati": _band((0, 1), (1.0, 1.1), (0.9, 1.0), (1.1, 1.2))
        }
        saturation_bands = {
            "truncated_riccati": _band((0, 1), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
        }
        fig = plot_ood_cost_and_constraint(
            cost_bands, saturation_bands, reference_lines={"nominal": 1.0}
        )
        assert isinstance(fig, Figure)

    def test_log_x_and_log_y(self) -> None:
        cost_bands = {
            "a": _band(
                (16, 256, 4096), (2.0, 1.5, 1.1), (1.8, 1.3, 1.0), (2.2, 1.7, 1.2)
            )
        }
        saturation_bands = {
            "a": _band(
                (16, 256, 4096), (0.1, 0.2, 0.3), (0.05, 0.15, 0.25), (0.15, 0.25, 0.35)
            )
        }
        fig = plot_ood_cost_and_constraint(
            cost_bands,
            saturation_bands,
            style=OODCostConstraintStyle(log_x=True, log_y_cost=True),
        )
        assert isinstance(fig, Figure)

    def test_empty_cost_bands_raises(self) -> None:
        with pytest.raises(ValueError, match="cost_bands"):
            plot_ood_cost_and_constraint({}, {})

    def test_skips_empty_curve_without_raising(self) -> None:
        cost_bands = {
            "no_data": _band((), (), (), ()),
            "has_data": _band((1.0,), (2.0,), (1.5,), (2.5,)),
        }
        saturation_bands = {"has_data": _band((1.0,), (0.1,), (0.05,), (0.15,))}
        fig = plot_ood_cost_and_constraint(cost_bands, saturation_bands)
        assert isinstance(fig, Figure)


class TestPlotOodDepthAblation:
    def test_renders_one_panel_per_contender(self) -> None:
        bands_by_contender = {
            "unfolded_alpha": {
                3: _band(
                    (0.5, 1.0, 2.0),
                    (0.1, 0.2, 0.5),
                    (0.08, 0.18, 0.4),
                    (0.12, 0.22, 0.6),
                ),
                10: _band(
                    (0.5, 1.0, 2.0),
                    (0.05, 0.3, 0.9),
                    (0.03, 0.25, 0.8),
                    (0.08, 0.35, 1.0),
                ),
            },
            "unfolded_alpha_p": {
                3: _band(
                    (0.5, 1.0, 2.0),
                    (0.05, 0.1, 0.3),
                    (0.04, 0.08, 0.25),
                    (0.06, 0.12, 0.35),
                ),
                10: _band(
                    (0.5, 1.0, 2.0),
                    (0.02, 0.08, 0.4),
                    (0.01, 0.06, 0.3),
                    (0.03, 0.1, 0.5),
                ),
            },
        }
        fig = plot_ood_depth_ablation(bands_by_contender)
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 2

    def test_single_contender_still_produces_a_figure(self) -> None:
        bands_by_contender = {
            "unfolded_alpha": {
                5: _band((1.0, 2.0), (0.1, 0.2), (0.08, 0.18), (0.12, 0.22)),
            },
        }
        fig = plot_ood_depth_ablation(bands_by_contender)
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 1

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError, match="bands_by_contender"):
            plot_ood_depth_ablation({})


class TestPlotOodPhasePortrait:
    def test_renders_two_panels(self) -> None:
        rng = np.random.default_rng(0)
        trajectories = {
            "unfolded_alpha_p": rng.normal(size=(20, 4)).cumsum(axis=0) * 0.1,
            "neural": rng.normal(size=(20, 4)).cumsum(axis=0) * 5.0,
        }
        fig = plot_ood_phase_portrait(trajectories)
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 2

    def test_trajectories_of_different_lengths_are_accepted(self) -> None:
        rng = np.random.default_rng(1)
        trajectories = {
            "short": rng.normal(size=(10, 3)),
            "long": rng.normal(size=(30, 3)),
        }
        fig = plot_ood_phase_portrait(trajectories)
        assert isinstance(fig, Figure)

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError, match="trajectories"):
            plot_ood_phase_portrait({})


class TestPlotOodCostVsCompute:
    def test_renders_one_scatter_point_per_contender(self) -> None:
        fig = plot_ood_cost_vs_compute(
            {"unfolded_alpha": 120.0, "neural": 340.0},
            {"unfolded_alpha": 0.05, "neural": 0.20},
        )
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 1

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError, match="latency_us_by_label"):
            plot_ood_cost_vs_compute({}, {})


class TestPlotOodMandateScatter:
    def test_renders_one_scatter_point_per_contender(self) -> None:
        fig = plot_ood_mandate_scatter(
            {"truncated_riccati": 0.85, "neural": 0.60},
            {"truncated_riccati": 0.90, "neural": 0.30},
        )
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 1

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError, match="saturation_by_label"):
            plot_ood_mandate_scatter({}, {})


class TestPlotOodInteractionHeatmap:
    def test_renders_a_heatmap(self) -> None:
        grid = np.array([[1.0, 2.0], [3.0, 4.0]])
        fig = plot_ood_interaction_heatmap((1.0, 2.0), (0.5, 1.0), grid)
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 2  # the heatmap axes + the colorbar axes

    def test_nan_cells_are_accepted(self) -> None:
        grid = np.array([[1.0, np.nan], [3.0, 4.0]])
        fig = plot_ood_interaction_heatmap((1.0, 2.0), (0.5, 1.0), grid)
        assert isinstance(fig, Figure)

    def test_mismatched_shape_raises(self) -> None:
        grid = np.zeros((3, 2))
        with pytest.raises(ValueError, match="grid.shape"):
            plot_ood_interaction_heatmap((1.0, 2.0), (0.5, 1.0), grid)
