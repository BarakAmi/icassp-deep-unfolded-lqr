"""Phase 6 smoke tests for the Tier-4 NB07 adapters (`render_band_curves`,
`render_learning_dynamics_with_constraint`): each persists a PDF + PNG + JSON
sidecar triple, displays without raising, and `re_render` reproduces the
figure from the sidecar alone -- including the categorical (noise-family)
x-axis case, which exercises both this session's `plot_band_curves` fix and
the sidecar round-trip together.
"""

from matplotlib.figure import Figure

from mbl.applications.uncertainty.noise import NoiseFamily
from mbl.viz.adapters.notebook import (
    render_band_curves,
    render_learning_dynamics_with_constraint,
)
from mbl.viz.adapters.sink import FigureSink, load_sidecar, re_render
from mbl.workbench.robust_training_sweep import CurveWithBand


def _band(x: tuple, median: tuple, q25: tuple, q75: tuple) -> CurveWithBand:
    return CurveWithBand(x=x, median=median, q25=q25, q75=q75)


class TestRenderBandCurves:
    def test_persists_pdf_png_json_and_displays_without_raising(self, tmp_path) -> None:
        bands = {
            "unfolded_alpha": _band(
                (0.0, 0.2, 0.4), (2.04, 2.01, 2.06), (2.0, 1.98, 2.02), (2.1, 2.05, 2.1)
            ),
            "neural": _band(
                (0.0, 0.2, 0.4), (2.05, 2.03, 2.11), (2.0, 2.0, 2.05), (2.1, 2.08, 2.2)
            ),
        }
        sink = FigureSink(tmp_path, show=False)

        render_band_curves(
            bands, sink=sink, name="band_test", reference_lines={"nominal": 2.04}
        )

        assert (tmp_path / "band_test.pdf").exists()
        assert (tmp_path / "band_test.png").exists()
        assert (tmp_path / "band_test.json").exists()

    def test_re_render_reproduces_the_figure_from_the_sidecar_alone(
        self, tmp_path
    ) -> None:
        bands = {
            "unfolded_alpha_p": _band((1.0, 2.0), (1.0, 1.1), (0.9, 1.0), (1.1, 1.2))
        }
        sink = FigureSink(tmp_path, show=False)
        render_band_curves(bands, sink=sink, name="band_rr")

        sidecar = load_sidecar(tmp_path / "band_rr.json")
        assert sidecar.renderer == "robust_band_curves"
        fig = re_render(tmp_path / "band_rr.json")
        assert isinstance(fig, Figure)

    def test_categorical_noise_family_axis_round_trips(self, tmp_path) -> None:
        bands = {
            "unfolded_alpha": _band(
                (
                    NoiseFamily.GAUSSIAN,
                    NoiseFamily.UNIFORM,
                    NoiseFamily.LAPLACE,
                    NoiseFamily.CAUCHY,
                ),
                (1.0, 1.1, 1.2, 5.0),
                (0.9, 1.0, 1.1, 3.0),
                (1.1, 1.2, 1.3, 8.0),
            ),
        }
        sink = FigureSink(tmp_path, show=False)
        render_band_curves(bands, sink=sink, name="band_family")

        fig = re_render(tmp_path / "band_family.json")
        assert isinstance(fig, Figure)
        labels = [tick.get_text() for tick in fig.axes[0].get_xticklabels()]
        assert labels == ["gaussian", "uniform", "laplace", "cauchy"]


class TestRenderLearningDynamicsWithConstraint:
    def test_persists_pdf_png_json_and_displays_without_raising(self, tmp_path) -> None:
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
        sink = FigureSink(tmp_path, show=False)

        render_learning_dynamics_with_constraint(
            cost_bands, constraint_bands, sink=sink, name="dyn_test"
        )

        assert (tmp_path / "dyn_test.pdf").exists()
        assert (tmp_path / "dyn_test.png").exists()
        assert (tmp_path / "dyn_test.json").exists()

    def test_re_render_reproduces_two_panels_from_the_sidecar_alone(
        self, tmp_path
    ) -> None:
        cost_bands = {"neural": _band((1, 2), (3.0, 2.5), (2.8, 2.3), (3.2, 2.7))}
        constraint_bands = {
            "neural": _band((1, 2), (0.1, 0.1), (0.05, 0.05), (0.15, 0.15))
        }
        sink = FigureSink(tmp_path, show=False)
        render_learning_dynamics_with_constraint(
            cost_bands, constraint_bands, sink=sink, name="dyn_rr"
        )

        sidecar = load_sidecar(tmp_path / "dyn_rr.json")
        assert sidecar.renderer == "robust_learning_dynamics"
        fig = re_render(tmp_path / "dyn_rr.json")
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 2
