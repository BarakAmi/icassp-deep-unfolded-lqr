"""Phase 1 acceptance test for `NeuralRecipe.init_seed`
(docs/planning/03_studies/nb07_robust_training/robust_training_under_model_mismatch.md Sec 2.4/11): the fix for the
determinism hole a multi-seed GRU sweep would otherwise walk straight into --
same seed reproduces bit-identical weights; different seed diverges; the
seed participates in the recipe's signature; construction never leaks into
the caller's own global RNG stream.
"""

import torch

from mbl.applications.factories import LQRProblemFactory
from mbl.applications.recipes.neural import NeuralRecipe
from mbl.core.runtime import Backend, ComputeContext, Precision
from mbl.engine.training_plan import OptimizerSpec, TrainingPlan


def _ctx() -> ComputeContext:
    return ComputeContext(
        backend=Backend.TORCH,
        device="cpu",
        precision=Precision.from_torch_dtype(torch.float64),
    )


def _problem():
    return LQRProblemFactory(
        state_dim=4, control_dim=2, horizon=10, seed=0, u_max=0.5
    ).build()


def _plan() -> TrainingPlan:
    return TrainingPlan(optimizer=OptimizerSpec("adam", 1e-2), epochs=1)


class TestReproducibility:
    def test_same_seed_gives_bit_identical_weights(self) -> None:
        problem = _problem()
        ctx = _ctx()
        recipe = NeuralRecipe(hidden_dim=8, plan=_plan(), init_seed=42)
        policy_a = recipe.build_controller(problem, ctx)
        policy_b = recipe.build_controller(problem, ctx)
        for (name_a, param_a), (name_b, param_b) in zip(
            policy_a.state_dict().items(), policy_b.state_dict().items()
        ):
            assert name_a == name_b
            torch.testing.assert_close(param_a, param_b, atol=0, rtol=0)

    def test_different_seed_gives_different_weights(self) -> None:
        problem = _problem()
        ctx = _ctx()
        policy_a = NeuralRecipe(
            hidden_dim=8, plan=_plan(), init_seed=1
        ).build_controller(problem, ctx)
        policy_b = NeuralRecipe(
            hidden_dim=8, plan=_plan(), init_seed=2
        ).build_controller(problem, ctx)
        state_a = dict(policy_a.state_dict())
        state_b = dict(policy_b.state_dict())
        assert any(not torch.equal(state_a[key], state_b[key]) for key in state_a)


class TestSignatureParticipation:
    def test_init_seed_participates_in_the_recipe_signature(self) -> None:
        sig_a = NeuralRecipe(hidden_dim=8, plan=_plan(), init_seed=1).get_signature()
        sig_b = NeuralRecipe(hidden_dim=8, plan=_plan(), init_seed=2).get_signature()
        assert sig_a != sig_b
        assert sig_a["init_seed"] == 1
        assert sig_b["init_seed"] == 2


class TestNoGlobalRngLeak:
    def test_construction_does_not_perturb_the_ambient_rng_stream(self) -> None:
        problem = _problem()
        ctx = _ctx()
        torch.manual_seed(123)
        before = torch.randn(5)

        torch.manual_seed(123)
        NeuralRecipe(hidden_dim=8, plan=_plan(), init_seed=999).build_controller(
            problem, ctx
        )
        after = torch.randn(5)

        torch.testing.assert_close(before, after, atol=0, rtol=0)
