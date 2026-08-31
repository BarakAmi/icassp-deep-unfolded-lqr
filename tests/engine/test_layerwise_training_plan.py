"""NB03 acceptance tests for `LayerwiseTrainingPlan`/`PhaseSpec` (blueprint
v2 §3.3(A), §7.1): a declarative, signable multi-phase schedule that
compiles into `PhaseSpec`s, and REJECTS a weight-decay optimizer outright
(the freeze-contract guard motivated by the empirically-demonstrated Adam/
AdamW decay leak on a masked-frozen parameter)."""

import pytest

from mbl.engine.training_plan import LayerwiseTrainingPlan, OptimizerSpec, PhaseSpec
from mbl.models.unfolded.layerwise import ParameterActivation


def _plan(**overrides: object) -> LayerwiseTrainingPlan:
    defaults: dict[str, object] = dict(
        optimizer=OptimizerSpec("adam", 0.05),
        warmup_epochs_per_layer=10,
        refinement_epochs=20,
    )
    defaults.update(overrides)
    return LayerwiseTrainingPlan(**defaults)


class TestGuards:
    def test_nonzero_weight_decay_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="weight_decay"):
            _plan(optimizer=OptimizerSpec("adamw", 0.05, {"weight_decay": 1e-4}))

    def test_zero_weight_decay_is_accepted(self) -> None:
        _plan(optimizer=OptimizerSpec("adamw", 0.05, {"weight_decay": 0.0}))  # no raise

    def test_absent_weight_decay_is_accepted(self) -> None:
        _plan(optimizer=OptimizerSpec("adam", 0.05))  # no raise

    def test_non_positive_warmup_epochs_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="warmup_epochs_per_layer"):
            _plan(warmup_epochs_per_layer=0)

    def test_negative_refinement_epochs_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="refinement_epochs"):
            _plan(refinement_epochs=-1)

    def test_zero_refinement_epochs_is_accepted_as_pure_greedy(self) -> None:
        _plan(refinement_epochs=0)  # no raise: disables the refinement tail

    def test_non_positive_gradient_clip_norm_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="gradient_clip_norm"):
            _plan(gradient_clip_norm=0.0)

    def test_unknown_loss_reduction_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="loss_reduction"):
            _plan(loss_reduction="median")


class TestCompileSingleActivation:
    def test_produces_one_warmup_phase_per_layer_plus_refinement(self) -> None:
        plan = _plan(warmup_epochs_per_layer=5, refinement_epochs=15)
        phases = plan.compile(3, has_matrix=False)
        assert [p.name for p in phases] == [
            "warmup_layer_0",
            "warmup_layer_1",
            "warmup_layer_2",
            "refinement",
        ]
        assert all(p.epochs == 5 for p in phases[:3])
        assert phases[3].epochs == 15

    def test_single_activation_activates_exactly_one_row_per_layer(self) -> None:
        plan = _plan(activation="single")
        phases = plan.compile(3, has_matrix=False)
        assert phases[0].activation.step_size_rows == frozenset({0})
        assert phases[1].activation.step_size_rows == frozenset({1})
        assert phases[2].activation.step_size_rows == frozenset({2})

    def test_refinement_phase_activates_every_row(self) -> None:
        plan = _plan(refinement_epochs=10)
        phases = plan.compile(3, has_matrix=True)
        assert phases[-1].activation.step_size_rows == "all"

    def test_zero_refinement_epochs_omits_the_trailing_phase(self) -> None:
        plan = _plan(refinement_epochs=0)
        phases = plan.compile(2, has_matrix=False)
        assert [p.name for p in phases] == ["warmup_layer_0", "warmup_layer_1"]


class TestCompileCumulativeActivation:
    def test_cumulative_activation_activates_the_growing_prefix(self) -> None:
        plan = _plan(activation="cumulative")
        phases = plan.compile(3, has_matrix=False)
        assert phases[0].activation.step_size_rows == frozenset({0})
        assert phases[1].activation.step_size_rows == frozenset({0, 1})
        assert phases[2].activation.step_size_rows == frozenset({0, 1, 2})


class TestCompileMatrixActivation:
    def test_train_matrix_from_refinement_only_activates_in_the_tail(self) -> None:
        plan = _plan(train_matrix_from="refinement", refinement_epochs=10)
        phases = plan.compile(3, has_matrix=True)
        assert [p.activation.train_matrix for p in phases] == [
            False,
            False,
            False,
            True,
        ]

    def test_train_matrix_from_each_activates_every_warmup_phase(self) -> None:
        plan = _plan(train_matrix_from="each", refinement_epochs=10)
        phases = plan.compile(3, has_matrix=True)
        assert [p.activation.train_matrix for p in phases] == [True, True, True, True]

    def test_train_matrix_from_last_layer_activates_from_the_final_warmup_onward(
        self,
    ) -> None:
        plan = _plan(train_matrix_from="last_layer", refinement_epochs=10)
        phases = plan.compile(3, has_matrix=True)
        assert [p.activation.train_matrix for p in phases] == [False, False, True, True]

    def test_has_matrix_false_forces_train_matrix_false_everywhere(self) -> None:
        plan = _plan(train_matrix_from="each", refinement_epochs=10)
        phases = plan.compile(3, has_matrix=False)
        assert all(not p.activation.train_matrix for p in phases)


class TestCompileValidation:
    def test_non_positive_num_layers_is_rejected(self) -> None:
        plan = _plan()
        with pytest.raises(ValueError, match="num_layers"):
            plan.compile(0, has_matrix=False)


class TestCacheIdentity:
    """A-4: distinct schedules (and a warm-start vs. an ordinary TrainingPlan)
    must be distinct cache identities."""

    def test_signature_distinguishes_every_schedule_field(self) -> None:
        base = _plan().get_signature()
        assert _plan(warmup_epochs_per_layer=11).get_signature() != base
        assert _plan(refinement_epochs=21).get_signature() != base
        assert _plan(train_matrix_from="each").get_signature() != base
        assert _plan(activation="cumulative").get_signature() != base

    def test_signature_type_and_optimizer_are_reported(self) -> None:
        signature = _plan().get_signature()
        assert signature["type"] == "LayerwiseTrainingPlan"
        assert signature["optimizer"]["name"] == "adam"
        assert signature["optimizer"]["learning_rate"] == 0.05


class TestProvenance:
    """A-5: the C4 law holds for the schedule too."""

    def test_training_config_derives_from_the_same_optimizer_spec(self) -> None:
        plan = _plan(optimizer=OptimizerSpec("sgd", 0.02, {"momentum": 0.9}))
        config = plan.training_config(batch_size=64, log_every=3, seed=1)
        assert config.learning_rate == 0.02
        assert config.optimizer_name == "sgd"
        assert config.optimizer_kwargs == {"momentum": 0.9}
        assert config.batch_size == 64
        assert config.log_every == 3
        assert config.seed == 1

    def test_resolve_loss_reduction_matches_the_declared_name(self) -> None:
        import torch

        assert _plan(loss_reduction="sum").resolve_loss_reduction() is torch.sum
        assert _plan(loss_reduction="mean").resolve_loss_reduction() is torch.mean


class TestPhaseSpec:
    def test_rejects_non_positive_epochs(self) -> None:
        with pytest.raises(ValueError, match="epochs"):
            PhaseSpec(
                activation=ParameterActivation(
                    step_size_rows="all", train_matrix=False
                ),
                epochs=0,
                name="bad",
            )
