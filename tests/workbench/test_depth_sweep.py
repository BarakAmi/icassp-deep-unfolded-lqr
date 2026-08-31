"""NB03 acceptance test (A-9) for `workbench.depth_sweep`: a depth sweep
correctly partitions K-dependent contenders into `curves` and the
K-invariant Riccati baseline into `reference_lines`, and a repeat sweep over
the same depths is served entirely from cache -- executed end to end against
the real NB03 declaration and `run_experiment`, never mocked.
"""

import math

import pytest

from mbl.applications.recipes import UnfoldedKind
from mbl.applications.studies import NB03Config, UnfoldedModelConfig
from mbl.engine.training_plan import LayerwiseTrainingPlan, OptimizerSpec, TrainingPlan
from mbl.workbench import BASELINE_LABELS, run_unfolding_depth_sweep_study


def _fast_plan() -> TrainingPlan:
    return TrainingPlan(optimizer=OptimizerSpec("adam", 0.05), epochs=3)


def _fast_schedule() -> LayerwiseTrainingPlan:
    return LayerwiseTrainingPlan(
        optimizer=OptimizerSpec("adam", 0.05),
        warmup_epochs_per_layer=1,
        refinement_epochs=1,
        train_matrix_from="refinement",
    )


def _fast_config(**overrides: object) -> NB03Config:
    defaults: dict[str, object] = dict(
        state_dim=3,
        control_dim=2,
        horizon=6,
        step_size_init=0.02,
        step_size_max=0.5,
        batch_size=16,
        n_eval_batches=2,
        seed=0,
        process_noise_std=0.3,
        unfolded_alpha=UnfoldedModelConfig(
            kind=UnfoldedKind.LEARNED_STEP_SIZE,
            training_mode="end_to_end",
            plan=_fast_plan(),
            schedule=_fast_schedule(),
        ),
        unfolded_alpha_p=UnfoldedModelConfig(
            kind=UnfoldedKind.LEARNED_STEP_SIZE_AND_MATRIX,
            training_mode="layerwise",
            plan=_fast_plan(),
            schedule=_fast_schedule(),
        ),
    )
    defaults.update(overrides)
    return NB03Config(**defaults)


class TestDepthSweepAggregation:
    def test_partitions_curves_from_reference_lines(self, tmp_path) -> None:
        sweep = run_unfolding_depth_sweep_study(_fast_config(), [2, 3], root=tmp_path)

        assert sweep.k_values == (2, 3)
        assert set(sweep.curves) == {
            "standard_gd",
            "unfolded_alpha",
            "unfolded_alpha_p",
        }
        assert set(sweep.reference_lines) == BASELINE_LABELS
        assert "sim_riccati" in sweep.reference_lines

    def test_curves_have_one_cost_per_depth_in_order(self, tmp_path) -> None:
        sweep = run_unfolding_depth_sweep_study(
            _fast_config(), [2, 3, 4], root=tmp_path
        )
        for label, costs in sweep.curves.items():
            assert len(costs) == 3, label
            assert all(math.isfinite(c) and c > 0 for c in costs), label

    def test_reference_line_is_a_single_finite_scalar(self, tmp_path) -> None:
        sweep = run_unfolding_depth_sweep_study(_fast_config(), [2, 3], root=tmp_path)
        cost = sweep.reference_lines["sim_riccati"]
        assert isinstance(cost, float)
        assert math.isfinite(cost) and cost > 0

    def test_reports_and_dispositions_are_keyed_by_depth(self, tmp_path) -> None:
        sweep = run_unfolding_depth_sweep_study(_fast_config(), [2, 3], root=tmp_path)
        assert set(sweep.reports) == {2, 3}
        assert set(sweep.dispositions) == {2, 3}
        assert set(sweep.dispositions[2]) == {
            "sim_riccati",
            "standard_gd",
            "unfolded_alpha",
            "unfolded_alpha_p",
        }

    def test_empty_depths_is_rejected(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="depths"):
            run_unfolding_depth_sweep_study(_fast_config(), [], root=tmp_path)


class TestDepthSweepCacheReuse:
    def test_iterative_contenders_are_fresh_at_every_new_depth(self, tmp_path) -> None:
        sweep = run_unfolding_depth_sweep_study(_fast_config(), [2, 3], root=tmp_path)
        iterative_labels = set(sweep.curves)  # K-dependent: never depth-shared
        for depth_dispositions in sweep.dispositions.values():
            for label in iterative_labels:
                assert depth_dispositions[label] == "fresh"

    def test_baseline_is_fresh_once_then_reused_across_depths(self, tmp_path) -> None:
        """sim_riccati's cost does not depend on K, so ONE cache entry
        legitimately serves every depth in the sweep after the first --
        the cache correctly recognizing identical content, not a defect."""
        sweep = run_unfolding_depth_sweep_study(
            _fast_config(), [2, 3, 4], root=tmp_path
        )
        baseline_dispositions = [
            sweep.dispositions[k]["sim_riccati"] for k in (2, 3, 4)
        ]
        assert baseline_dispositions == ["fresh", "hit", "hit"]

    def test_repeat_sweep_over_the_same_depths_is_all_hit(self, tmp_path) -> None:
        config = _fast_config()
        run_unfolding_depth_sweep_study(config, [2, 3], root=tmp_path)

        sweep = run_unfolding_depth_sweep_study(config, [2, 3], root=tmp_path)

        for depth_dispositions in sweep.dispositions.values():
            assert set(depth_dispositions.values()) == {"hit"}

    def test_extending_the_sweep_only_computes_the_new_depth(self, tmp_path) -> None:
        config = _fast_config()
        run_unfolding_depth_sweep_study(config, [2, 3], root=tmp_path)

        sweep = run_unfolding_depth_sweep_study(config, [2, 3, 4], root=tmp_path)

        assert set(sweep.dispositions[2].values()) == {"hit"}
        assert set(sweep.dispositions[3].values()) == {"hit"}
        # depth 4 is genuinely new for the K-dependent contenders...
        iterative_labels = set(sweep.curves)
        for label in iterative_labels:
            assert sweep.dispositions[4][label] == "fresh"
        # ...but sim_riccati's content was already cached at an earlier depth.
        assert sweep.dispositions[4]["sim_riccati"] == "hit"

    def test_cached_costs_match_the_original_fresh_ones(self, tmp_path) -> None:
        config = _fast_config()
        fresh = run_unfolding_depth_sweep_study(config, [2, 3], root=tmp_path)
        cached = run_unfolding_depth_sweep_study(config, [2, 3], root=tmp_path)

        assert fresh.curves == cached.curves
        assert fresh.reference_lines == cached.reference_lines
