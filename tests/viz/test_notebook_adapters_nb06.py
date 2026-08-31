"""NB06 smoke tests for the Tier-4 OOD adapters (`render_ood_cost_and_constraint`,
`render_ood_depth_ablation`, `render_ood_phase_portrait`): each persists a
PDF + PNG + JSON sidecar triple and displays without raising, against a REAL
(small, fast) `OODSweepResult` -- not a hand-built stand-in. Sidecar
round-trip replay (`re_render`) is deliberately out of scope here, mirroring
NB07's own documented Phase 5/6 split: the pure rendering + persistence path
is complete and tested; registering a `re_render` reconstruction function
for these three renderers is a deferred follow-up.
"""

import numpy as np
import torch

from mbl.core.runtime import Backend, ComputeContext, Precision
from mbl.experiments import ContenderSpec
from mbl.experiments.zero_shot import evaluate_under_shift
from mbl.models.analytic.truncated_riccati import TruncatedRiccatiSynthesizer
from mbl.viz.adapters.notebook import (
    render_ood_cost_and_constraint,
    render_ood_cost_vs_compute,
    render_ood_depth_ablation,
    render_ood_interaction_heatmap,
    render_ood_mandate_scatter,
    render_ood_phase_portrait,
)
from mbl.viz.adapters.sink import FigureSink
from mbl.applications.factories import GaussianBatchSpec, LQRProblemFactory
from mbl.workbench.ood_sweep import (
    MIN_OOD_SEEDS,
    OODAxis,
    OODSweepContext,
    OODSweepSpec,
    run_interaction_scale_bound_sweep,
    run_scale_ood_sweep,
)

CTX = ComputeContext(backend=Backend.TORCH, device="cpu", precision=Precision.FLOAT64)
HORIZON = 6
U_MAX = 0.3


def _context() -> OODSweepContext:
    factory = LQRProblemFactory(
        state_dim=3, control_dim=2, horizon=HORIZON, seed=0, u_max=U_MAX
    )
    problem = factory.build()
    constraint = problem.constraints[0]  # type: ignore[index]
    artifact = TruncatedRiccatiSynthesizer(HORIZON, constraint).synthesize(problem, CTX)
    spec = ContenderSpec(family="truncated_riccati", config={"horizon": HORIZON})
    batch = GaussianBatchSpec(
        state_dim=3, horizon=HORIZON, batch_size=16, seed=1, process_noise_std=0.4
    )
    return OODSweepContext(
        artifacts={"truncated_riccati": artifact},
        contender_specs={"truncated_riccati": spec},
        base_factory=factory,
        nominal_batch=batch,
        ctx=CTX,
        u_max=U_MAX,
    )


class TestRenderOodCostAndConstraint:
    def test_persists_pdf_png_json_and_displays_without_raising(self, tmp_path) -> None:
        context = _context()
        spec = OODSweepSpec(
            axis=OODAxis.SCALE,
            levels=(0.5, 1.0, 2.0),
            n_seeds=MIN_OOD_SEEDS,
            eval_batches_per_seed=1,
        )
        result = run_scale_ood_sweep(context, spec)
        sink = FigureSink(tmp_path, show=False)

        render_ood_cost_and_constraint(result, sink=sink, name="ood_scale_test")

        assert (tmp_path / "ood_scale_test.pdf").exists()
        assert (tmp_path / "ood_scale_test.png").exists()
        assert (tmp_path / "ood_scale_test.json").exists()

    def test_raw_cost_mode_also_renders(self, tmp_path) -> None:
        context = _context()
        spec = OODSweepSpec(
            axis=OODAxis.SCALE,
            levels=(1.0,),
            n_seeds=MIN_OOD_SEEDS,
            eval_batches_per_seed=1,
        )
        result = run_scale_ood_sweep(context, spec)
        sink = FigureSink(tmp_path, show=False)

        render_ood_cost_and_constraint(
            result, sink=sink, use_gap=False, name="ood_raw_cost_test"
        )
        assert (tmp_path / "ood_raw_cost_test.pdf").exists()


class TestRenderOodDepthAblation:
    def test_persists_pdf_png_json_and_displays_without_raising(self, tmp_path) -> None:
        context = _context()
        spec = OODSweepSpec(
            axis=OODAxis.SCALE,
            levels=(0.5, 1.0),
            n_seeds=MIN_OOD_SEEDS,
            eval_batches_per_seed=1,
        )
        results_by_depth = {
            3: run_scale_ood_sweep(context, spec),
            5: run_scale_ood_sweep(context, spec),
        }
        sink = FigureSink(tmp_path, show=False)

        render_ood_depth_ablation(
            results_by_depth, ["truncated_riccati"], sink=sink, name="ood_depth_test"
        )

        assert (tmp_path / "ood_depth_test.pdf").exists()
        assert (tmp_path / "ood_depth_test.png").exists()
        assert (tmp_path / "ood_depth_test.json").exists()


class TestRenderOodPhasePortrait:
    def test_persists_pdf_png_json_and_displays_without_raising(self, tmp_path) -> None:
        problem = LQRProblemFactory(
            state_dim=3, control_dim=2, horizon=HORIZON, seed=0, u_max=U_MAX
        ).build()
        constraint = problem.constraints[0]  # type: ignore[index]
        artifact = TruncatedRiccatiSynthesizer(HORIZON, constraint).synthesize(
            problem, CTX
        )
        batch = GaussianBatchSpec(
            state_dim=3, horizon=HORIZON, batch_size=4, seed=2, process_noise_std=0.4
        )
        sampler, _ = batch.build(Backend.TORCH, torch_dtype=torch.float64)
        batches = (sampler(),)
        _, arrays = evaluate_under_shift(
            artifact, problem, batches, u_max=U_MAX, capture_trajectories=True
        )
        trajectories = {
            "truncated_riccati": np.asarray(arrays["trajectory_states"])[0, 0],
        }
        sink = FigureSink(tmp_path, show=False)

        render_ood_phase_portrait(trajectories, sink=sink, name="ood_portrait_test")

        assert (tmp_path / "ood_portrait_test.pdf").exists()
        assert (tmp_path / "ood_portrait_test.png").exists()
        assert (tmp_path / "ood_portrait_test.json").exists()


class TestRenderOodCostVsCompute:
    def test_persists_pdf_png_json_and_displays_without_raising(self, tmp_path) -> None:
        sink = FigureSink(tmp_path, show=False)
        render_ood_cost_vs_compute(
            {"truncated_riccati": 50.0, "neural": 200.0},
            {"truncated_riccati": 0.1, "neural": 0.4},
            sink=sink,
            name="ood_cost_vs_compute_test",
        )
        assert (tmp_path / "ood_cost_vs_compute_test.pdf").exists()
        assert (tmp_path / "ood_cost_vs_compute_test.png").exists()
        assert (tmp_path / "ood_cost_vs_compute_test.json").exists()


class TestRenderOodMandateScatter:
    def test_persists_pdf_png_json_and_displays_without_raising(self, tmp_path) -> None:
        sink = FigureSink(tmp_path, show=False)
        render_ood_mandate_scatter(
            {"truncated_riccati": 0.85, "neural": 0.6},
            {"truncated_riccati": 0.9, "neural": 0.3},
            sink=sink,
            name="ood_mandate_scatter_test",
        )
        assert (tmp_path / "ood_mandate_scatter_test.pdf").exists()
        assert (tmp_path / "ood_mandate_scatter_test.png").exists()
        assert (tmp_path / "ood_mandate_scatter_test.json").exists()


class TestRenderOodInteractionHeatmap:
    def test_persists_pdf_png_json_and_displays_without_raising(self, tmp_path) -> None:
        context = _context()
        spec = OODSweepSpec(
            axis=OODAxis.INTERACTION_SCALE_BOUND,
            levels=((1.0, 1.0), (1.0, 0.5), (2.0, 1.0), (2.0, 0.5)),
            n_seeds=MIN_OOD_SEEDS,
            eval_batches_per_seed=1,
        )
        result = run_interaction_scale_bound_sweep(context, spec)
        scale_levels, bound_levels, grid = result.interaction_grid("truncated_riccati")
        sink = FigureSink(tmp_path, show=False)

        render_ood_interaction_heatmap(
            scale_levels, bound_levels, grid, sink=sink, name="ood_interaction_test"
        )

        assert (tmp_path / "ood_interaction_test.pdf").exists()
        assert (tmp_path / "ood_interaction_test.png").exists()
        assert (tmp_path / "ood_interaction_test.json").exists()
