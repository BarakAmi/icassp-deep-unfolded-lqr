"""NB04 acceptance test for `workbench.depth_sweep`'s generalized core: a
depth sweep over the box-constrained experiment correctly partitions
K-dependent contenders into `curves` and the K-invariant Truncated-Riccati
baseline into `reference_lines`, and a repeat sweep over the same depths is
served entirely from cache -- executed end to end against the real NB04
declaration and `run_experiment`, never mocked. Mirrors
`test_depth_sweep.py`'s NB03 coverage, but exercises it through NB04's own
wrapper (`run_box_constrained_depth_sweep_study`) over the SAME generalized
`run_depth_sweep_study` core -- proving the generalization is genuinely
shared, not two parallel implementations.
"""

import math

import pytest

from mbl.applications.recipes import UnfoldedKind
from mbl.applications.studies import NB04Config, UnfoldedModelConfig
from mbl.engine.training_plan import LayerwiseTrainingPlan, OptimizerSpec, TrainingPlan
from mbl.workbench import NB04_BASELINE_LABELS, run_box_constrained_depth_sweep_study


def _fast_plan() -> TrainingPlan:
    return TrainingPlan(optimizer=OptimizerSpec("adam", 0.05), epochs=3)


def _fast_schedule() -> LayerwiseTrainingPlan:
    return LayerwiseTrainingPlan(
        optimizer=OptimizerSpec("adam", 0.05),
        warmup_epochs_per_layer=1,
        refinement_epochs=1,
        train_matrix_from="refinement",
    )


def _fast_config(**overrides: object) -> NB04Config:
    defaults: dict[str, object] = dict(
        state_dim=3,
        control_dim=2,
        horizon=6,
        u_max=0.3,
        step_size_init=0.02,
        step_size_max=0.5,
        batch_size=16,
        n_eval_batches=2,
        seed=0,
        process_noise_std=0.5,
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
        cocp_plan=_fast_plan(),
    )
    defaults.update(overrides)
    return NB04Config(**defaults)


class TestBoxConstrainedDepthSweepAggregation:
    def test_partitions_curves_from_reference_lines(self, tmp_path) -> None:
        sweep = run_box_constrained_depth_sweep_study(
            _fast_config(), [2, 3], root=tmp_path
        )

        assert sweep.k_values == (2, 3)
        assert set(sweep.curves) == {
            "standard_pgd",
            "unfolded_alpha",
            "unfolded_alpha_p",
        }
        assert set(sweep.reference_lines) == NB04_BASELINE_LABELS
        assert "truncated_riccati" in sweep.reference_lines

    def test_curves_have_one_cost_per_depth_in_order(self, tmp_path) -> None:
        sweep = run_box_constrained_depth_sweep_study(
            _fast_config(), [2, 3, 4], root=tmp_path
        )
        for label, costs in sweep.curves.items():
            assert len(costs) == 3, label
            assert all(math.isfinite(c) and c > 0 for c in costs), label

    def test_reference_line_is_a_single_finite_scalar(self, tmp_path) -> None:
        sweep = run_box_constrained_depth_sweep_study(
            _fast_config(), [2, 3], root=tmp_path
        )
        cost = sweep.reference_lines["truncated_riccati"]
        assert isinstance(cost, float)
        assert math.isfinite(cost) and cost > 0

    def test_reports_and_dispositions_are_keyed_by_depth(self, tmp_path) -> None:
        sweep = run_box_constrained_depth_sweep_study(
            _fast_config(), [2, 3], root=tmp_path
        )
        assert set(sweep.reports) == {2, 3}
        assert set(sweep.dispositions) == {2, 3}
        assert set(sweep.dispositions[2]) == {
            "truncated_riccati",
            "standard_pgd",
            "unfolded_alpha",
            "unfolded_alpha_p",
            "cocp",
            "cocp_lower_bound",
        }

    def test_empty_depths_is_rejected(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="depths"):
            run_box_constrained_depth_sweep_study(_fast_config(), [], root=tmp_path)


class TestBoxConstrainedDepthSweepCacheReuse:
    def test_iterative_contenders_are_fresh_at_every_new_depth(self, tmp_path) -> None:
        sweep = run_box_constrained_depth_sweep_study(
            _fast_config(), [2, 3], root=tmp_path
        )
        iterative_labels = set(sweep.curves)  # K-dependent: never depth-shared
        for depth_dispositions in sweep.dispositions.values():
            for label in iterative_labels:
                assert depth_dispositions[label] == "fresh"

    def test_baseline_is_fresh_once_then_reused_across_depths(self, tmp_path) -> None:
        """truncated_riccati's cost does not depend on K, so ONE cache entry
        legitimately serves every depth in the sweep after the first."""
        sweep = run_box_constrained_depth_sweep_study(
            _fast_config(), [2, 3, 4], root=tmp_path
        )
        baseline_dispositions = [
            sweep.dispositions[k]["truncated_riccati"] for k in (2, 3, 4)
        ]
        assert baseline_dispositions == ["fresh", "hit", "hit"]

    def test_cocp_is_trained_once_then_reused_across_depths(self, tmp_path) -> None:
        """COCP has no unfolding depth of its own (COCP refinement plan Sec
        3.1 lever 1/Sec 3.3): it must be trained exactly ONCE across the
        whole sweep, not once per depth like the genuinely K-dependent
        unfolded contenders."""
        sweep = run_box_constrained_depth_sweep_study(
            _fast_config(), [2, 3, 4], root=tmp_path
        )
        cocp_dispositions = [sweep.dispositions[k]["cocp"] for k in (2, 3, 4)]
        assert cocp_dispositions == ["fresh", "hit", "hit"]
        assert "cocp" in sweep.reference_lines
        assert "cocp" not in sweep.curves

    def test_cocp_lower_bound_is_solved_once_then_reused_across_depths(
        self, tmp_path
    ) -> None:
        """COCP-LB is an `AnalyticRecipe` with no unfolding depth either
        (reference-bounds plan Sec 1.4): its SDP-seeded QP must be solved
        exactly ONCE across the whole sweep, the same K-invariant caching
        contract COCP and Truncated-Riccati already get. If this ever
        regresses to re-solving at every depth, the fallback in the plan
        (evaluate once outside the sweep, inject the scalar directly) is the
        documented escape hatch."""
        sweep = run_box_constrained_depth_sweep_study(
            _fast_config(), [2, 3, 4], root=tmp_path
        )
        cocp_lb_dispositions = [
            sweep.dispositions[k]["cocp_lower_bound"] for k in (2, 3, 4)
        ]
        assert cocp_lb_dispositions == ["fresh", "hit", "hit"]
        assert "cocp_lower_bound" in sweep.reference_lines
        assert "cocp_lower_bound" not in sweep.curves

    def test_repeat_sweep_over_the_same_depths_is_all_hit(self, tmp_path) -> None:
        config = _fast_config()
        run_box_constrained_depth_sweep_study(config, [2, 3], root=tmp_path)

        sweep = run_box_constrained_depth_sweep_study(config, [2, 3], root=tmp_path)

        for depth_dispositions in sweep.dispositions.values():
            assert set(depth_dispositions.values()) == {"hit"}

    def test_cached_costs_match_the_original_fresh_ones(self, tmp_path) -> None:
        config = _fast_config()
        fresh = run_box_constrained_depth_sweep_study(config, [2, 3], root=tmp_path)
        cached = run_box_constrained_depth_sweep_study(config, [2, 3], root=tmp_path)

        assert fresh.curves == cached.curves
        assert fresh.reference_lines == cached.reference_lines
