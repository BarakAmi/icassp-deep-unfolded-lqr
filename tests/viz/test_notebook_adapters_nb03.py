"""NB03 sidecar round-trip tests for the Tier-4 adapters
(`render_learning_curves`, `render_cost_vs_iterations`,
`render_trajectory_comparison`, and Phase 3's
`render_unfolding_landscape_animations`/`render_unfolding_learning_curves`/
`render_control_trajectories_vs_riccati`): every persisted figure is a PDF +
raw-data sidecar pair, and `re_render` reproduces the original figure's data
exactly from the sidecar alone -- matching the "No Figure Unbacked" law
every other adapter in this module already honors (see
`tests/viz/test_sink.py`).
"""

import logging

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

import mbl.viz.adapters.notebook  # noqa: F401  registers the standard sidecar renderers
from mbl.core.cost.quadratic_cost import QuadraticCost
from mbl.core.optimal_control_problem import OptimalControlProblem
from mbl.core.system.linear_system import LinearSystem
from mbl.experiments import ContenderResult, ExperimentReport
from mbl.models.analytic.riccati import RiccatiController
from mbl.viz.adapters.notebook import (
    DepthTrajectoryReportSpec,
    UnfoldingLandscapeSpec,
    render_cartesian_performance_tradeoff,
    render_control_trajectories_vs_riccati,
    render_cost_vs_iterations,
    render_depth_learning_curves_by_contender,
    render_learning_curves,
    render_trajectory_comparison,
    render_unfolding_landscape_animations,
    render_unfolding_learning_curves,
)
from mbl.viz.adapters.sink import FigureSink, load_sidecar, re_render
from mbl.viz.plots import BenchmarkRecord, MemoryBreakdown
from mbl.workbench.depth_sweep import DepthSweepResult
from mbl.workbench.replay import UnfoldingLandscapeInputs
from mbl.workbench.setup import generate_marginally_stable_system


def _histories() -> dict[str, np.ndarray]:
    epochs = np.arange(1, 21, dtype=float)
    return {
        "unfolded_alpha": 5.0 / epochs,
        "unfolded_alpha_p": 5.0 / (epochs**1.2),
    }


def _sweep() -> DepthSweepResult:
    return DepthSweepResult(
        k_values=(2, 4, 8),
        curves={
            "standard_gd": (2.5, 2.4, 2.35),
            "unfolded_alpha": (2.1, 1.6, 1.2),
            "unfolded_alpha_p": (1.9, 1.3, 1.05),
        },
        reference_lines={"sim_riccati": 1.0},
        reports={},
        dispositions={},
    )


def _benchmark_records() -> list[BenchmarkRecord]:
    return [
        BenchmarkRecord(
            label="sim_riccati",
            offline_s=0.0,
            online_s=8e-7,
            offline_memory=MemoryBreakdown(numpy_mb=0.0, torch_mb=0.0),
            online_memory=MemoryBreakdown(numpy_mb=0.002, torch_mb=0.0),
        ),
        BenchmarkRecord(
            label="unfolded_alpha_p",
            offline_s=25.4,
            online_s=9e-6,
            offline_memory=MemoryBreakdown(numpy_mb=1575.0, torch_mb=72.6),
            online_memory=MemoryBreakdown(numpy_mb=0.0, torch_mb=32.8),
        ),
    ]


class TestRenderCartesianPerformanceTradeoff:
    def test_persists_the_pdf_and_sidecar_pair(self, tmp_path) -> None:
        sink = FigureSink(tmp_path, show=False)
        render_cartesian_performance_tradeoff(
            _benchmark_records(), sink=sink, name="ct"
        )
        assert (tmp_path / "ct.pdf").exists()
        assert (tmp_path / "ct.json").exists()

    def test_re_render_reproduces_the_figure_from_the_sidecar_alone(
        self, tmp_path
    ) -> None:
        sink = FigureSink(tmp_path, show=False)
        render_cartesian_performance_tradeoff(
            _benchmark_records(), sink=sink, name="ct"
        )
        sidecar = load_sidecar(tmp_path / "ct.json")
        assert sidecar.renderer == "cartesian_performance_tradeoff"
        fig = re_render(tmp_path / "ct.json")
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 4


class TestRenderLearningCurves:
    def test_persists_the_pdf_and_sidecar_pair(self, tmp_path) -> None:
        sink = FigureSink(tmp_path, show=False)

        # Phase C directive 1: the adapter returns None (the sink displays
        # once); a returned Figure would double-render as the cell's Out[].
        result = render_learning_curves(  # type: ignore[func-returns-value]  # asserting the None contract at runtime
            _histories(), baselines={"Theo-Riccati": 0.9}, sink=sink
        )
        assert result is None
        pdf_path = tmp_path / "learning_curves.pdf"
        sidecar_path = tmp_path / "learning_curves.json"
        assert pdf_path.is_file()
        assert pdf_path.read_bytes().startswith(b"%PDF-")
        assert sidecar_path.is_file()

    def test_sidecar_carries_the_exact_curves_and_baselines(self, tmp_path) -> None:
        sink = FigureSink(tmp_path, show=False)
        histories = _histories()
        render_learning_curves(
            histories, baselines={"Theo-Riccati": 0.9}, sink=sink, name="lc"
        )

        sidecar = load_sidecar(tmp_path / "lc.json")
        assert sidecar.renderer == "training_curves"
        np.testing.assert_allclose(
            sidecar.data["curves"]["unfolded_alpha"], histories["unfolded_alpha"]
        )
        assert sidecar.data["reference_lines"]["Theo-Riccati"] == 0.9

    def test_re_render_reproduces_a_figure_from_the_sidecar_alone(
        self, tmp_path
    ) -> None:
        sink = FigureSink(tmp_path, show=False)
        render_learning_curves(
            _histories(), baselines={"Theo-Riccati": 0.9}, sink=sink, name="lc"
        )

        fig = re_render(tmp_path / "lc.json")
        assert fig is not None
        assert len(fig.axes) == 1

    def test_no_baselines_omits_reference_lines_from_the_sidecar(
        self, tmp_path
    ) -> None:
        sink = FigureSink(tmp_path, show=False)
        render_learning_curves(_histories(), sink=sink, name="lc_no_baseline")

        sidecar = load_sidecar(tmp_path / "lc_no_baseline.json")
        assert "reference_lines" not in sidecar.data
        # still re-renders cleanly with no baselines.
        assert re_render(tmp_path / "lc_no_baseline.json") is not None


class TestRenderCostVsIterations:
    def test_persists_the_pdf_and_sidecar_pair(self, tmp_path) -> None:
        sink = FigureSink(tmp_path, show=False)

        render_cost_vs_iterations(_sweep(), sink=sink)

        pdf_path = tmp_path / "cost_vs_unfolding_depth.pdf"
        sidecar_path = tmp_path / "cost_vs_unfolding_depth.json"
        assert pdf_path.is_file()
        assert pdf_path.read_bytes().startswith(b"%PDF-")
        assert sidecar_path.is_file()

    def test_sidecar_carries_k_values_curves_and_reference_lines(
        self, tmp_path
    ) -> None:
        sink = FigureSink(tmp_path, show=False)
        sweep = _sweep()

        render_cost_vs_iterations(sweep, sink=sink, name="depth")

        sidecar = load_sidecar(tmp_path / "depth.json")
        assert sidecar.renderer == "cost_vs_unfolding_depth"
        assert list(sidecar.data["k_values"]) == [2, 4, 8]
        np.testing.assert_allclose(
            sidecar.data["curves"]["unfolded_alpha_p"], [1.9, 1.3, 1.05]
        )
        assert sidecar.data["reference_lines"]["sim_riccati"] == 1.0

    def test_extra_reference_lines_are_merged_in(self, tmp_path) -> None:
        sink = FigureSink(tmp_path, show=False)

        render_cost_vs_iterations(
            _sweep(),
            sink=sink,
            name="depth_extra",
            extra_reference_lines={"Theo-Riccati": 0.85},
        )

        sidecar = load_sidecar(tmp_path / "depth_extra.json")
        assert sidecar.data["reference_lines"] == {
            "sim_riccati": 1.0,
            "Theo-Riccati": 0.85,
        }

    def test_re_render_reproduces_a_figure_from_the_sidecar_alone(
        self, tmp_path
    ) -> None:
        sink = FigureSink(tmp_path, show=False)
        render_cost_vs_iterations(_sweep(), sink=sink, name="depth")

        fig = re_render(tmp_path / "depth.json")
        assert fig is not None
        assert len(fig.axes) == 1


def _control_trajectories() -> dict[str, np.ndarray]:
    t = np.linspace(0, 1, 10)
    return {
        "sim_riccati": np.stack([-t, t], axis=-1),
        "unfolded_alpha_p": np.stack([-t * 0.9, t * 1.1], axis=-1),
    }


class TestRenderTrajectoryComparison:
    def test_persists_the_pdf_and_sidecar_pair(self, tmp_path) -> None:
        sink = FigureSink(tmp_path, show=False)

        render_trajectory_comparison(_control_trajectories(), sink=sink)

        pdf_path = tmp_path / "control_trajectory_comparison.pdf"
        sidecar_path = tmp_path / "control_trajectory_comparison.json"
        assert pdf_path.is_file()
        assert pdf_path.read_bytes().startswith(b"%PDF-")
        assert sidecar_path.is_file()

    def test_sidecar_carries_the_exact_trajectories(self, tmp_path) -> None:
        sink = FigureSink(tmp_path, show=False)
        series = _control_trajectories()

        render_trajectory_comparison(series, sink=sink, name="traj")

        sidecar = load_sidecar(tmp_path / "traj.json")
        assert sidecar.renderer == "trajectory_comparison"
        np.testing.assert_allclose(
            sidecar.data["series"]["sim_riccati"], series["sim_riccati"]
        )
        assert sidecar.config["variable_name"] == "u"

    def test_re_render_reproduces_a_figure_from_the_sidecar_alone(
        self, tmp_path
    ) -> None:
        sink = FigureSink(tmp_path, show=False)
        render_trajectory_comparison(_control_trajectories(), sink=sink, name="traj")

        fig = re_render(tmp_path / "traj.json")
        assert fig is not None
        assert len(fig.axes) == 2  # one subplot per control dimension


# --- render_unfolding_landscape_animations (Phase 3, requirement 1) ---------


def _tiny_lqr(
    n: int, m: int, horizon: int = 4
) -> tuple[OptimalControlProblem, "np.ndarray", "np.ndarray"]:
    A, B = generate_marginally_stable_system(n, m, seed=0)
    system = LinearSystem.fully_observable(A, B)
    Q = np.repeat(np.eye(n)[None], horizon + 1, axis=0)
    R = np.repeat(np.eye(m)[None] * 0.1, horizon, axis=0)
    cost = QuadraticCost(Q=Q, R=R, include_terminal_cost=True)
    problem = OptimalControlProblem(system=system, cost=cost)
    return problem, A, B


def _unfolding_inputs(
    n: int, m: int, riccati: RiccatiController
) -> UnfoldingLandscapeInputs:
    _, A, B = _tiny_lqr(n, m)
    rng = np.random.default_rng(0)
    return UnfoldingLandscapeInputs(
        x_star=rng.normal(size=n) * 0.5,
        step_size=np.full((5, m), 0.05),
        P_own=riccati.P_arr[1],
        A=riccati.problem.system.A_t.array,
        B=riccati.problem.system.B_t.array,
        R=riccati.problem.cost.R[0],
    )


def _cocp_result(riccati: RiccatiController) -> ContenderResult:
    """A COCP contender's persisted `ParameterSnapshotCallback` artifacts
    (`applications.recipes.cocp.cocp_convergence_parameters`'s naming),
    constructed directly (never a full `run_experiment`/training pass --
    `cocp_reference_point`'s own contract is exercised end to end,
    real-recipe-trained, in `tests/workbench/test_replay.py`) so this
    adapter-level test stays focused on the WIRING: does `cocp_result`
    reach the overlay and survive the sidecar round-trip."""
    n = riccati.problem.system.A_t.array.shape[0]
    return ContenderResult(
        label="cocp",
        family="cocp",
        cache_key="k",
        content_digest="d",
        disposition="fresh",
        run_dir=None,
        metrics={},
        arrays={
            "parameter_P_sqrt": np.eye(n),
            "parameter_q": np.zeros(n),
            "parameter_A": riccati.problem.system.A_t.array,
            "parameter_B": riccati.problem.system.B_t.array,
            "parameter_R": riccati.problem.cost.R[0],
            "parameter_u_max": np.asarray(0.5),
        },
    )


@pytest.mark.parametrize(
    "m, expected_static_kinds",
    [
        (1, {"line"}),
        (2, {"contour", "surface"}),
        (3, {"scatter"}),
    ],
)
def test_render_unfolding_landscape_animations_dispatches_on_control_dim(
    m, expected_static_kinds, tmp_path
) -> None:
    """No caller special-cases `m` anywhere -- the router alone decides
    which asset kind(s) to render, purely from `inputs.B.shape[1]`."""
    n = 2 if m < 2 else m + 1
    problem, A, B = _tiny_lqr(n, m)
    riccati = RiccatiController(problem, horizon=4)
    inputs = _unfolding_inputs(n, m, riccati)
    sink = FigureSink(tmp_path, show=False)

    render_unfolding_landscape_animations(
        riccati,
        inputs,
        sink=sink,
        spec=UnfoldingLandscapeSpec(t_star=1, process_noise_std=0.1, grid_resolution=6),
    )

    pdfs = {
        p.stem.rsplit("_", 1)[-1] for p in tmp_path.glob("unfolding_landscape_*.pdf")
    }
    assert pdfs == expected_static_kinds
    gifs = {
        p.stem.removesuffix("_animated").rsplit("_", 1)[-1]
        for p in tmp_path.glob("unfolding_landscape_*.gif")
    }
    assert gifs == expected_static_kinds


@pytest.mark.parametrize("m", [1, 2, 3])
def test_render_unfolding_landscape_animations_with_box_bounds_runs_and_still_renders(
    m, tmp_path
) -> None:
    """NB04 plan Sec 3.4: `UnfoldingLandscapeSpec.box_bounds` threads
    through to the STATIC figure(s) for every supported control dimension
    without error, and every expected artifact still gets written."""
    n = 2 if m < 2 else m + 1
    problem, A, B = _tiny_lqr(n, m)
    riccati = RiccatiController(problem, horizon=4)
    inputs = _unfolding_inputs(n, m, riccati)
    sink = FigureSink(tmp_path, show=False)

    render_unfolding_landscape_animations(
        riccati,
        inputs,
        sink=sink,
        spec=UnfoldingLandscapeSpec(
            t_star=1, process_noise_std=0.1, grid_resolution=6, box_bounds=(-1.0, 1.0)
        ),
    )

    assert list(tmp_path.glob("unfolding_landscape_*.pdf"))
    assert list(tmp_path.glob("unfolding_landscape_*.gif"))


def test_render_unfolding_landscape_animations_skips_and_logs_above_m3(
    tmp_path, caplog
) -> None:
    n, m = 5, 4
    problem, A, B = _tiny_lqr(n, m)
    riccati = RiccatiController(problem, horizon=4)
    inputs = _unfolding_inputs(n, m, riccati)
    sink = FigureSink(tmp_path, show=False)

    with caplog.at_level(logging.INFO):
        render_unfolding_landscape_animations(
            riccati,
            inputs,
            sink=sink,
            spec=UnfoldingLandscapeSpec(t_star=1, process_noise_std=0.1),
        )

    assert list(tmp_path.glob("*.pdf")) == []
    assert list(tmp_path.glob("*.gif")) == []
    assert any("control_dim=4" in record.message for record in caplog.records)


def test_render_unfolding_landscape_animations_re_renders_from_sidecar(
    tmp_path,
) -> None:
    n, m = 4, 2
    problem, A, B = _tiny_lqr(n, m)
    riccati = RiccatiController(problem, horizon=4)
    inputs = _unfolding_inputs(n, m, riccati)
    sink = FigureSink(tmp_path, show=False)

    render_unfolding_landscape_animations(
        riccati,
        inputs,
        sink=sink,
        spec=UnfoldingLandscapeSpec(
            t_star=1, process_noise_std=0.1, grid_resolution=6, name_prefix="lscape"
        ),
    )

    sidecar = load_sidecar(tmp_path / "lscape_contour.json")
    assert sidecar.renderer == "local_cost_landscape"
    fig = re_render(tmp_path / "lscape_contour.json")
    assert isinstance(fig, Figure)


def test_render_unfolding_landscape_animations_box_bounds_survives_sidecar_round_trip(
    tmp_path,
) -> None:
    """NB04 plan Sec 3.4: `UnfoldingLandscapeSpec.box_bounds` must still
    shade the feasible region after a cache-hit `re_render` from the
    sidecar alone -- it is carried on `TrajectoryOverlay.box_bounds`, which
    must be persisted into the landscape sidecar's `overlay` payload, not
    silently dropped."""
    from mbl.viz.landscape.box_region import format_box_region_label

    n, m = 4, 2
    problem, A, B = _tiny_lqr(n, m)
    riccati = RiccatiController(problem, horizon=4)
    inputs = _unfolding_inputs(n, m, riccati)
    sink = FigureSink(tmp_path, show=False)

    render_unfolding_landscape_animations(
        riccati,
        inputs,
        sink=sink,
        spec=UnfoldingLandscapeSpec(
            t_star=1,
            process_noise_std=0.1,
            grid_resolution=6,
            name_prefix="lscape_boxed",
            box_bounds=(-1.0, 1.0),
        ),
    )

    fig = re_render(tmp_path / "lscape_boxed_contour.json")
    expected_label = format_box_region_label((-1.0, 1.0), (-1.0, 1.0))
    labeled = [p for p in fig.axes[0].patches if p.get_label() == expected_label]
    assert len(labeled) == 1


def test_render_unfolding_landscape_animations_with_cocp_result_draws_second_marker(
    tmp_path,
) -> None:
    """NB04 COCP/viz refinement plan Sec 2.2/3.4: `UnfoldingLandscapeSpec
    .cocp_result` resolves to a real QP solve at the frozen x_star, scored
    on the same Riccati bowl, and reaches the static figure as a second,
    labeled marker."""
    n, m = 4, 2
    problem, A, B = _tiny_lqr(n, m)
    riccati = RiccatiController(problem, horizon=4)
    inputs = _unfolding_inputs(n, m, riccati)
    sink = FigureSink(tmp_path, show=False)

    render_unfolding_landscape_animations(
        riccati,
        inputs,
        sink=sink,
        spec=UnfoldingLandscapeSpec(
            t_star=1,
            process_noise_std=0.1,
            grid_resolution=6,
            name_prefix="lscape_cocp",
            cocp_result=_cocp_result(riccati),
        ),
    )

    fig = re_render(tmp_path / "lscape_cocp_contour.json")
    labels = [text.get_text() for text in fig.axes[0].get_legend().get_texts()]
    assert any(label.startswith("COCP control") for label in labels)


def test_render_unfolding_landscape_animations_cocp_marker_survives_sidecar_round_trip(
    tmp_path,
) -> None:
    """Same round-trip guarantee `box_bounds` already has: a cache-hit
    `re_render` from the sidecar alone must not silently drop the COCP
    marker (`TrajectoryOverlay.cocp_point`/`cocp_value` persisted into the
    landscape sidecar's `overlay` payload)."""
    n, m = 4, 2
    problem, A, B = _tiny_lqr(n, m)
    riccati = RiccatiController(problem, horizon=4)
    inputs = _unfolding_inputs(n, m, riccati)
    sink = FigureSink(tmp_path, show=False)

    render_unfolding_landscape_animations(
        riccati,
        inputs,
        sink=sink,
        spec=UnfoldingLandscapeSpec(
            t_star=1,
            process_noise_std=0.1,
            grid_resolution=6,
            name_prefix="lscape_cocp_sidecar",
            cocp_result=_cocp_result(riccati),
        ),
    )

    sidecar = load_sidecar(tmp_path / "lscape_cocp_sidecar_contour.json")
    assert sidecar.data["overlay"]["cocp_point"] is not None
    fig = re_render(tmp_path / "lscape_cocp_sidecar_contour.json")
    labels = [text.get_text() for text in fig.axes[0].get_legend().get_texts()]
    assert any(label.startswith("COCP control") for label in labels)


# --- render_unfolding_learning_curves (Phase 3, requirement 2) --------------


def _fake_contender_result(
    label: str, family: str, cost: float, *, trained: bool
) -> ContenderResult:
    arrays = {}
    if trained:
        arrays["metrics_history"] = pd.DataFrame(
            {"epoch": range(5), "loss": np.linspace(2.0, cost, 5)}
        )
    return ContenderResult(
        label=label,
        family=family,
        cache_key="k",
        content_digest="d",
        disposition="hit",
        run_dir=None,
        metrics={"eval_expected_cost": cost},
        arrays=arrays,
    )


def _fake_report(depth: int) -> ExperimentReport:
    return ExperimentReport(
        experiment_name="NB03",
        experiment_signature="sig",
        provenance_stamp="stamp",
        results={
            "sim_riccati": _fake_contender_result(
                "sim_riccati", "riccati", 1.0, trained=False
            ),
            "unfolded_alpha": _fake_contender_result(
                "unfolded_alpha", "unfolded", 1.2 + 0.1 / depth, trained=True
            ),
            "unfolded_alpha_p": _fake_contender_result(
                "unfolded_alpha_p",
                "unfolded_warmstart",
                1.1 + 0.05 / depth,
                trained=True,
            ),
        },
    )


def _fake_sweep(depths: tuple[int, ...]) -> DepthSweepResult:
    return DepthSweepResult(
        k_values=depths,
        curves={},
        reference_lines={"sim_riccati": 1.0},
        reports={depth: _fake_report(depth) for depth in depths},
        dispositions={},
    )


class TestRenderUnfoldingLearningCurves:
    def test_persists_one_figure_per_depth(self, tmp_path) -> None:
        sink = FigureSink(tmp_path, show=False)

        render_unfolding_learning_curves(
            _fake_sweep((2, 4, 6)),
            [2, 4, 6],
            sink=sink,
            extra_reference_lines={"Theo-Riccati": 0.95},
        )

        for depth in (2, 4, 6):
            pdf_path = tmp_path / f"learning_curves_depth_{depth}.pdf"
            assert pdf_path.is_file()
            assert pdf_path.read_bytes().startswith(b"%PDF-")

    def test_each_figure_carries_that_depths_own_baselines(self, tmp_path) -> None:
        sink = FigureSink(tmp_path, show=False)

        render_unfolding_learning_curves(
            _fake_sweep((2, 4)),
            [2, 4],
            sink=sink,
            extra_reference_lines={"Theo-Riccati": 0.95},
        )

        sidecar = load_sidecar(tmp_path / "learning_curves_depth_2.json")
        assert sidecar.data["reference_lines"] == {
            "Theo-Riccati": 0.95,
            "Sim-Riccati": 1.0,
        }
        assert set(sidecar.data["curves"]) == {"unfolded_alpha", "unfolded_alpha_p"}

    def test_reuses_the_registered_training_curves_renderer(self, tmp_path) -> None:
        """No second `plot_training_curves` call site -- reuses
        `render_learning_curves` verbatim, so the sidecar renderer tag and
        re_render behavior are identical to Section 5's single-depth
        figure."""
        sink = FigureSink(tmp_path, show=False)

        render_unfolding_learning_curves(_fake_sweep((3,)), [3], sink=sink)

        sidecar = load_sidecar(tmp_path / "learning_curves_depth_3.json")
        assert sidecar.renderer == "training_curves"
        assert re_render(tmp_path / "learning_curves_depth_3.json") is not None


class TestRenderDepthLearningCurvesByContender:
    """Phase C directive 2: exactly ONE figure per contender, each a family of
    Cost-vs-Epoch curves across every swept depth plus the flat Riccati
    baselines."""

    def test_persists_exactly_one_figure_per_contender(self, tmp_path) -> None:
        sink = FigureSink(tmp_path, show=False)

        render_depth_learning_curves_by_contender(
            _fake_sweep((1, 2, 4)),
            [1, 2, 4],
            sink=sink,
            baselines={"Theo-Riccati": 0.95, "Sim-Riccati": 1.0},
        )

        for label in ("unfolded_alpha", "unfolded_alpha_p"):
            pdf = tmp_path / f"learning_curves_by_contender_{label}.pdf"
            assert pdf.is_file()
            assert pdf.read_bytes().startswith(b"%PDF-")
        # exactly two figures, one per contender
        assert len(list(tmp_path.glob("learning_curves_by_contender_*.pdf"))) == 2

    def test_each_figure_carries_the_family_of_depth_curves_and_baselines(
        self, tmp_path
    ) -> None:
        sink = FigureSink(tmp_path, show=False)

        render_depth_learning_curves_by_contender(
            _fake_sweep((1, 2, 4)),
            [1, 2, 4],
            sink=sink,
            baselines={"Theo-Riccati": 0.95, "Sim-Riccati": 1.0},
        )

        sidecar = load_sidecar(
            tmp_path / "learning_curves_by_contender_unfolded_alpha.json"
        )
        assert sidecar.renderer == "training_curves"
        # one curve per swept depth, labeled by J
        assert set(sidecar.data["curves"]) == {"$J=1$", "$J=2$", "$J=4$"}
        assert sidecar.data["reference_lines"] == {
            "Theo-Riccati": 0.95,
            "Sim-Riccati": 1.0,
        }


# --- render_control_trajectories_vs_riccati (Phase 3, requirement 4) --------


def _depth_trajectories(horizon: int = 10, m: int = 2) -> dict[int, np.ndarray]:
    baseline = np.sin(np.linspace(0, 3, horizon * m)).reshape(horizon, m)
    return {
        depth: baseline
        + (0.5 / depth) * np.random.default_rng(depth).normal(size=(horizon, m))
        for depth in (2, 4, 6)
    }


class TestRenderControlTrajectoriesVsRiccati:
    def test_persists_static_and_animated_pairs(self, tmp_path) -> None:
        sink = FigureSink(tmp_path, show=False)
        horizon, m = 10, 2
        baseline = np.sin(np.linspace(0, 3, horizon * m)).reshape(horizon, m)

        render_control_trajectories_vs_riccati(
            baseline,
            _depth_trajectories(horizon, m),
            sink=sink,
            spec=DepthTrajectoryReportSpec(),
        )

        static_pdf = tmp_path / "trajectory_vs_depth_static.pdf"
        static_json = tmp_path / "trajectory_vs_depth_static.json"
        animated_gif = tmp_path / "trajectory_vs_depth_animated.gif"
        assert static_pdf.is_file() and static_pdf.read_bytes().startswith(b"%PDF-")
        assert static_json.is_file()
        assert animated_gif.is_file() and animated_gif.stat().st_size > 0

    def test_baseline_label_defaults_to_riccati_and_is_overridable(
        self, tmp_path
    ) -> None:
        """NB04 plan Sec 3.5: a box-constrained caller comparing against a
        SATURATED baseline (e.g. Truncated-Riccati) must be able to relabel
        the legend so it never claims an optimality the plotted curve
        doesn't have; the default stays "Riccati" for every existing
        (NB03) call site."""
        sink = FigureSink(tmp_path, show=False)
        horizon, m = 10, 2
        baseline = np.sin(np.linspace(0, 3, horizon * m)).reshape(horizon, m)

        render_control_trajectories_vs_riccati(
            baseline,
            _depth_trajectories(horizon, m),
            sink=sink,
            spec=DepthTrajectoryReportSpec(
                name_prefix="relabeled", baseline_label="Truncated-Riccati"
            ),
            baseline_cost=1.5,
        )

        fig = re_render(tmp_path / "relabeled_static.json")
        labels = [t.get_text() for t in fig.legends[0].get_texts()]
        assert labels[0] == "Truncated-Riccati (J=1.5000)"

    def test_box_bounds_survives_the_sidecar_round_trip(self, tmp_path) -> None:
        """NB04 plan Sec 3.5: the constraint lines must still be present
        after a cache-hit `re_render` from the sidecar alone, not only on
        the freshly-rendered figure -- `box_bounds` must be persisted into
        `config`, not silently dropped."""
        sink = FigureSink(tmp_path, show=False)
        horizon, m = 10, 2
        baseline = np.sin(np.linspace(0, 3, horizon * m)).reshape(horizon, m)

        render_control_trajectories_vs_riccati(
            baseline,
            _depth_trajectories(horizon, m),
            sink=sink,
            spec=DepthTrajectoryReportSpec(name_prefix="boxed", box_bounds=(-0.5, 0.5)),
        )

        fig = re_render(tmp_path / "boxed_static.json")
        # baseline + 3 depths + 2 bound lines, per signal row.
        for ax in fig.axes[:m]:
            assert len(ax.get_lines()) == 3 + 1 + 2

    def test_re_render_reproduces_a_figure_from_the_sidecar_alone(
        self, tmp_path
    ) -> None:
        sink = FigureSink(tmp_path, show=False)
        horizon, m = 10, 2
        baseline = np.sin(np.linspace(0, 3, horizon * m)).reshape(horizon, m)

        render_control_trajectories_vs_riccati(
            baseline,
            _depth_trajectories(horizon, m),
            sink=sink,
            spec=DepthTrajectoryReportSpec(name_prefix="depth_traj"),
        )

        sidecar = load_sidecar(tmp_path / "depth_traj_static.json")
        assert sidecar.renderer == "trajectory_vs_depth"
        fig = re_render(tmp_path / "depth_traj_static.json")
        assert isinstance(fig, Figure)
        assert len(fig.axes) == m + 1

    def test_is_shape_derived_over_control_dim(self, tmp_path) -> None:
        """The additive figure set never hardcodes a control dimension --
        m=1 must produce a 2-axes figure with zero manual adjustment."""
        sink = FigureSink(tmp_path, show=False)
        horizon, m = 8, 1
        baseline = np.sin(np.linspace(0, 3, horizon * m)).reshape(horizon, m)
        depth_trajectories = {
            depth: baseline
            + (0.3 / depth) * np.random.default_rng(depth).normal(size=(horizon, m))
            for depth in (2, 4)
        }

        render_control_trajectories_vs_riccati(
            baseline,
            depth_trajectories,
            sink=sink,
            spec=DepthTrajectoryReportSpec(name_prefix="m1"),
        )

        fig = re_render(tmp_path / "m1_static.json")
        assert len(fig.axes) == m + 1

    def test_costs_round_trip_through_the_sidecar(self, tmp_path) -> None:
        sink = FigureSink(tmp_path, show=False)
        horizon, m = 10, 2
        baseline = np.sin(np.linspace(0, 3, horizon * m)).reshape(horizon, m)
        depth_trajectories = _depth_trajectories(horizon, m)
        costs = {depth: 1.0 / depth for depth in depth_trajectories}

        render_control_trajectories_vs_riccati(
            baseline,
            depth_trajectories,
            sink=sink,
            spec=DepthTrajectoryReportSpec(
                name_prefix="cost_traj", contender_name="Unfolded-alpha"
            ),
            costs=costs,
            baseline_cost=0.5,
        )

        fig = re_render(tmp_path / "cost_traj_static.json")
        labels = [t.get_text() for t in fig.legends[0].get_texts()]
        assert labels[0] == "Riccati (J=0.5000)"
        for depth, label in zip(sorted(depth_trajectories), labels[1:]):
            assert label == f"K={depth} (J={costs[depth]:.4f})"
        assert "Unfolded-alpha" in fig._suptitle.get_text()
