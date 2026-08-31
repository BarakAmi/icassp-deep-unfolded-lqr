"""Phase E/F acceptance test for `workbench.alignment_sweep`
(docs/planning/03_studies/nb05_ltv_box_constrained/ltv_box_constrained_lqr_benchmark.md Sec 0.2-0.3/7): a sweep over
several named LTV points correctly reports EVERY contender as axis-dependent
(unlike the depth sweep, no baseline/curve split), the REALIZED distinct-
slice count `D` at each point (not the nominal schedule parameter -- the
degeneracy-law regression this suite pins down concretely), and per-point
floors -- executed end to end against the real NB05 declaration and
`run_experiment`, never mocked.
"""

import math

import pytest

from mbl.applications.ltv_factories import LTVRegime
from mbl.applications.recipes import UnfoldedKind
from mbl.applications.studies import NB05Config, UnfoldedModelConfig
from mbl.engine.training_plan import LayerwiseTrainingPlan, OptimizerSpec, TrainingPlan
from mbl.workbench import LTVAlignmentPoint, run_ltv_alignment_sweep_study
from mbl.workbench.alignment_sweep import LTVAlignmentSweepResult


def _fast_plan() -> TrainingPlan:
    return TrainingPlan(optimizer=OptimizerSpec("adam", 0.05), epochs=3)


def _fast_schedule() -> LayerwiseTrainingPlan:
    return LayerwiseTrainingPlan(
        optimizer=OptimizerSpec("adam", 0.05),
        warmup_epochs_per_layer=1,
        refinement_epochs=1,
        train_matrix_from="refinement",
    )


def _fast_config(**overrides: object) -> NB05Config:
    defaults: dict[str, object] = dict(
        state_dim=3,
        control_dim=2,
        horizon=6,
        u_max=0.3,
        regime=LTVRegime.PERIODIC,
        period_or_block=3,
        variation_strength=0.4,
        num_unfolding_iterations=3,
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
        unfolded_alpha_p_scalarmod=UnfoldedModelConfig(
            kind=UnfoldedKind.LEARNED_STEP_SIZE_AND_SCALAR_MODULATED_MATRIX,
            training_mode="layerwise",
            plan=_fast_plan(),
            schedule=_fast_schedule(),
        ),
        unfolded_alpha_p_periter=UnfoldedModelConfig(
            kind=UnfoldedKind.LEARNED_STEP_SIZE_AND_PER_ITERATION_MATRIX,
            training_mode="layerwise",
            plan=_fast_plan(),
            schedule=_fast_schedule(),
        ),
    )
    defaults.update(overrides)
    return NB05Config(**defaults)


_POINTS = [
    LTVAlignmentPoint("NB04 anchor (eps=0)", LTVRegime.PERIODIC, 3, 0.0),
    LTVAlignmentPoint("periodic (p=3)", LTVRegime.PERIODIC, 3, 0.4),
    LTVAlignmentPoint("block (c=3)", LTVRegime.BLOCK_CONSTANT, 3, 0.4),
    LTVAlignmentPoint("fully varying", LTVRegime.FULLY_VARYING, None, 0.4),
]


class TestAlignmentSweepAggregation:
    def test_returns_the_realized_not_structural_distinct_slice_counts(
        self, tmp_path
    ) -> None:
        """The degeneracy-law regression: the eps=0 anchor point must report
        D=1 (the TRUE realized count), never `period_or_block`'s nominal
        value (3), which is what a naive `distinct_slice_count()` call
        would have reported instead."""
        result = run_ltv_alignment_sweep_study(_fast_config(), _POINTS, root=tmp_path)

        assert isinstance(result, LTVAlignmentSweepResult)
        assert result.distinct_slice_counts == (1, 3, 2, 6)

    def test_every_contender_is_axis_dependent(self, tmp_path) -> None:
        """Unlike workbench.depth_sweep.DepthSweepResult, there is no
        baseline/curve split here -- ALL SIX contenders, including
        truncated_riccati, appear in `costs`."""
        result = run_ltv_alignment_sweep_study(_fast_config(), _POINTS, root=tmp_path)

        assert set(result.costs) == {
            "truncated_riccati",
            "standard_pgd",
            "unfolded_alpha",
            "unfolded_alpha_p",
            "unfolded_alpha_p_scalarmod",
            "unfolded_alpha_p_periter",
        }
        for label, costs in result.costs.items():
            assert len(costs) == len(_POINTS), label
            assert all(math.isfinite(c) and c > 0 for c in costs), label

    def test_cost_stds_reported_when_multiple_eval_batches(self, tmp_path) -> None:
        result = run_ltv_alignment_sweep_study(_fast_config(), _POINTS, root=tmp_path)
        for label, stds in result.cost_stds.items():
            assert len(stds) == len(_POINTS), label
            assert all(s is not None and s >= 0 for s in stds), label

    def test_cost_stds_are_none_with_a_single_eval_batch(self, tmp_path) -> None:
        result = run_ltv_alignment_sweep_study(
            _fast_config(n_eval_batches=1), _POINTS, root=tmp_path
        )
        for stds in result.cost_stds.values():
            assert all(s is None for s in stds)

    def test_floors_computed_per_point_and_ordered_like_points(self, tmp_path) -> None:
        result = run_ltv_alignment_sweep_study(_fast_config(), _POINTS, root=tmp_path)

        assert len(result.floors) == len(_POINTS)
        for floors in result.floors:
            assert math.isfinite(floors.j_lqr_fin)
            assert floors.j_box_fin is not None
            assert floors.j_box_fin >= floors.j_lqr_fin - 1e-6

    def test_compute_box_floor_false_skips_floor_b_at_every_point(
        self, tmp_path
    ) -> None:
        result = run_ltv_alignment_sweep_study(
            _fast_config(), _POINTS, root=tmp_path, compute_box_floor=False
        )
        assert all(f.j_box_fin is None for f in result.floors)

    def test_reports_keyed_by_point_label(self, tmp_path) -> None:
        result = run_ltv_alignment_sweep_study(_fast_config(), _POINTS, root=tmp_path)
        assert set(result.reports) == {p.label for p in _POINTS}


class TestAlignmentSweepValidation:
    def test_empty_points_rejected(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="points"):
            run_ltv_alignment_sweep_study(_fast_config(), [], root=tmp_path)

    def test_duplicate_labels_rejected(self, tmp_path) -> None:
        dup = [
            LTVAlignmentPoint("same", LTVRegime.PERIODIC, 3, 0.1),
            LTVAlignmentPoint("same", LTVRegime.PERIODIC, 3, 0.2),
        ]
        with pytest.raises(ValueError, match="unique"):
            run_ltv_alignment_sweep_study(_fast_config(), dup, root=tmp_path)


class TestAlignmentSweepCacheReuse:
    def test_repeat_sweep_reuses_the_cache(self, tmp_path) -> None:
        config = _fast_config()
        first = run_ltv_alignment_sweep_study(config, _POINTS, root=tmp_path)
        second = run_ltv_alignment_sweep_study(config, _POINTS, root=tmp_path)

        assert first.costs == second.costs
        for report in second.reports.values():
            assert {r.disposition for r in report.results.values()} == {"hit"}
