"""Stage-S3 acceptance tests for the declarative training specifications
(REFACTOR_PLAN v3, T3.b): `OptimizerSpec`/`TrainingPlan` are signable data,
the engine BUILDS its optimizer from the spec, and the logged
`TrainingConfig` derives from the same spec (the systemic C4 fix)."""

import pytest
import torch
from torch import nn

from mbl.engine.strategy import GradientDescentStrategy
from mbl.engine.training_plan import OptimizerSpec, TrainingPlan, TrainingState


def _module() -> nn.Module:
    torch.manual_seed(0)
    return nn.Linear(3, 2, dtype=torch.float64)


class TestOptimizerSpec:
    def test_build_constructs_the_named_optimizer_at_the_declared_lr(self):
        spec = OptimizerSpec("adam", 0.05, {"betas": (0.8, 0.9)})
        optimizer = spec.build(_module().parameters())
        assert isinstance(optimizer, torch.optim.Adam)
        group = optimizer.param_groups[0]
        assert group["lr"] == 0.05
        assert group["betas"] == (0.8, 0.9)

    def test_sgd_is_available_by_name(self):
        optimizer = OptimizerSpec("sgd", 0.1).build(_module().parameters())
        assert isinstance(optimizer, torch.optim.SGD)

    def test_unknown_optimizer_is_a_typed_refusal_naming_the_choices(self):
        with pytest.raises(ValueError, match="adam"):
            OptimizerSpec("lion", 0.1)

    def test_non_positive_learning_rate_is_rejected(self):
        with pytest.raises(ValueError, match="learning_rate"):
            OptimizerSpec("adam", 0.0)

    def test_signature_is_the_full_specification(self):
        spec = OptimizerSpec("adam", 0.05, {"weight_decay": 1e-4})
        assert spec.get_signature() == {
            "type": "OptimizerSpec",
            "name": "adam",
            "learning_rate": 0.05,
            "hyperparameters": {"weight_decay": 1e-4},
        }


class TestTrainingPlan:
    def test_training_config_is_derived_from_the_executed_spec(self):
        """The C4 provenance law: config fields come from the SAME
        OptimizerSpec that builds the live optimizer."""
        plan = TrainingPlan(
            optimizer=OptimizerSpec("adam", 0.007, {"weight_decay": 1e-5}),
            epochs=12,
        )
        config = plan.training_config(batch_size=32, log_every=2, seed=7)
        assert config.learning_rate == 0.007
        assert config.optimizer_name == "adam"
        assert config.optimizer_kwargs == {"weight_decay": 1e-5}
        assert config.batch_size == 32
        assert config.log_every == 2
        assert config.seed == 7

    def test_invalid_epochs_clip_and_reduction_are_rejected(self):
        spec = OptimizerSpec("adam", 0.1)
        with pytest.raises(ValueError, match="epochs"):
            TrainingPlan(optimizer=spec, epochs=0)
        with pytest.raises(ValueError, match="gradient_clip_norm"):
            TrainingPlan(optimizer=spec, epochs=1, gradient_clip_norm=0.0)
        with pytest.raises(ValueError, match="loss_reduction"):
            TrainingPlan(optimizer=spec, epochs=1, loss_reduction="median")

    def test_signature_covers_the_full_plan(self):
        plan = TrainingPlan(
            optimizer=OptimizerSpec("sgd", 0.2),
            epochs=3,
            gradient_clip_norm=1.0,
            loss_reduction="sum",
        )
        signature = plan.get_signature()
        assert signature["type"] == "TrainingPlan"
        assert signature["epochs"] == 3
        assert signature["gradient_clip_norm"] == 1.0
        assert signature["loss_reduction"] == "sum"
        assert signature["optimizer"]["name"] == "sgd"

    def test_from_plan_builds_the_strategy_the_plan_describes(self):
        """`GradientDescentStrategy.from_plan` wires optimizer, reduction,
        and clipping from the plan through the as_module parameter seam."""
        module = _module()
        plan = TrainingPlan(
            optimizer=OptimizerSpec("adam", 0.03),
            epochs=2,
            gradient_clip_norm=0.5,
            loss_reduction="sum",
        )
        strategy = GradientDescentStrategy.from_plan(module, module, plan)
        assert isinstance(strategy.optimizer, torch.optim.Adam)
        assert strategy.optimizer.param_groups[0]["lr"] == 0.03
        assert strategy.loss_reduction is torch.sum
        assert strategy.gradient_clip_norm == 0.5
        # The optimizer sees exactly the module's parameters, by identity.
        assert strategy.optimizer.param_groups[0]["params"] == list(module.parameters())

    def test_gradient_clipping_actually_bounds_the_update(self):
        """A pathologically steep quadratic loss on SGD: without clipping the
        first step moves the weight by lr * |grad| >> clip * lr; with the
        clip, the step is bounded by lr * clip."""
        weight = nn.Parameter(torch.tensor([0.0], dtype=torch.float64))
        module = nn.Module()
        module.register_parameter("w", weight)

        class _SteepModel(nn.Module):
            def forward(self, x0, w_noise, v_noise):
                cost = (weight - 1000.0) ** 2
                return None, None, None, cost

        plan = TrainingPlan(
            optimizer=OptimizerSpec("sgd", 0.01),
            epochs=1,
            gradient_clip_norm=1.0,
        )
        strategy = GradientDescentStrategy.from_plan(_SteepModel(), module, plan)
        strategy.step(context=None, batch=(None, None, None))
        # |grad| = 2000, clipped to 1.0 -> step = lr * 1.0 = 0.01.
        assert abs(float(weight.detach())) <= 0.01 + 1e-12


class TestTrainingState:
    def test_round_trips_module_and_optimizer_state(self):
        module = _module()
        optimizer = OptimizerSpec("adam", 0.1).build(module.parameters())
        state = TrainingState(
            epoch=5,
            module_state_dict=module.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
        )
        restored = nn.Linear(3, 2, dtype=torch.float64)
        restored.load_state_dict(state.module_state_dict)
        assert torch.equal(restored.weight, module.weight)
        restored_optimizer = OptimizerSpec("adam", 0.1).build(restored.parameters())
        restored_optimizer.load_state_dict(state.optimizer_state_dict)
        assert state.epoch == 5
