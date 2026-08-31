"""Stage-S4 acceptance tests for the `Experiment` entity (T3.d): frozen,
signable end-to-end, registry-resolved contenders-as-data, and `CaseStudy`
binding."""

import dataclasses

import pytest
import torch

from mbl.applications.factories import GaussianBatchSpec, LQRProblemFactory
from mbl.applications.recipes import RiccatiRecipe
from mbl.applications.standard_lqr import StandardLQRConfig, standard_lqr_case_study
from mbl.core.runtime import Backend, ComputeContext, Precision
from mbl.experiments import ContenderSpec, EvaluationProtocol, Experiment

CTX = ComputeContext(backend=Backend.TORCH, device="cpu", precision=Precision.FLOAT64)
PROBLEM = LQRProblemFactory(state_dim=4, control_dim=2, horizon=12, seed=0)
BATCH = GaussianBatchSpec(
    state_dim=4, horizon=12, batch_size=64, seed=0, process_noise_std=0.5
)


def _experiment(**overrides) -> Experiment:
    fields = dict(
        name="entity_test",
        problem=PROBLEM,
        contenders=(ContenderSpec(family="riccati", config={"horizon": 12}),),
        evaluation=EvaluationProtocol(batch_spec=BATCH, n_batches=2),
        ctx=CTX,
    )
    fields.update(overrides)
    return Experiment(**fields)


class TestContenderSpec:
    def test_resolves_family_and_config_through_the_registry_as_data(self):
        spec = ContenderSpec(family="riccati", config={"horizon": 12})
        recipe = spec.resolve()
        assert isinstance(recipe, RiccatiRecipe)
        assert recipe.horizon == 12
        assert spec.resolved_label == "analytic"

    def test_label_override_renames_the_instance(self):
        spec = ContenderSpec(family="riccati", config={"horizon": 12}, label="ric_b")
        assert spec.resolve().label == "ric_b"
        assert spec.resolved_label == "ric_b"

    def test_from_recipe_wraps_a_composed_recipe(self):
        recipe = RiccatiRecipe(horizon=12)
        spec = ContenderSpec.from_recipe(recipe)
        assert spec.resolve() is recipe
        assert spec.get_signature() == recipe.get_signature()

    def test_needs_family_or_recipe(self):
        with pytest.raises(ValueError, match="family name or a composed recipe"):
            ContenderSpec()

    def test_signature_covers_registry_defaults(self):
        """Two spellings of the same effective configuration share one
        identity: explicit default label == omitted label."""
        implicit = ContenderSpec(family="riccati", config={"horizon": 12})
        explicit = ContenderSpec(
            family="riccati", config={"horizon": 12, "label": "analytic"}
        )
        assert implicit.get_signature() == explicit.get_signature()


class TestEvaluationProtocol:
    def test_batches_are_deterministic_and_shared_by_construction(self):
        first = EvaluationProtocol(batch_spec=BATCH, n_batches=2).build_batches(CTX)
        second = EvaluationProtocol(batch_spec=BATCH, n_batches=2).build_batches(CTX)
        assert len(first) == 2
        for (a0, a1, a2), (b0, b1, b2) in zip(first, second):
            assert torch.equal(a0, b0)
            assert torch.equal(a1, b1)
            assert torch.equal(a2, b2)

    def test_sequential_batches_advance_one_stream(self):
        b0, b1 = EvaluationProtocol(batch_spec=BATCH, n_batches=2).build_batches(CTX)
        assert not torch.equal(b0[0], b1[0])

    def test_rejects_nonpositive_batch_counts(self):
        with pytest.raises(ValueError, match="n_batches"):
            EvaluationProtocol(batch_spec=BATCH, n_batches=0)


class TestExperimentEntity:
    def test_is_frozen(self):
        experiment = _experiment()
        with pytest.raises(dataclasses.FrozenInstanceError):
            experiment.name = "mutated"  # type: ignore[misc]  # the mutation attempt is the test

    def test_signature_composes_every_leg(self):
        signature = _experiment().get_signature()
        assert signature["problem"]["type"] == "LQRProblemFactory"
        assert signature["evaluation"]["type"] == "EvaluationProtocol"
        assert signature["compute_context"]["backend"] == "torch"
        assert "analytic" in signature["contenders"]

    def test_contender_key_ignores_the_experiment_name(self):
        """Renaming an experiment must not orphan its cached results."""
        spec = ContenderSpec(family="riccati", config={"horizon": 12})
        a = _experiment(name="one").contender_content_digest(spec)
        b = _experiment(name="two").contender_content_digest(spec)
        assert a == b

    def test_contender_key_tracks_problem_evaluation_and_context(self):
        spec = ContenderSpec(family="riccati", config={"horizon": 12})
        base = _experiment().contender_content_digest(spec)
        other_problem = _experiment(
            problem=dataclasses.replace(PROBLEM, seed=1)
        ).contender_content_digest(spec)
        other_eval = _experiment(
            evaluation=EvaluationProtocol(batch_spec=BATCH, n_batches=3)
        ).contender_content_digest(spec)
        other_ctx = _experiment(
            ctx=ComputeContext(
                backend=Backend.TORCH, device="cpu", precision=Precision.FLOAT32
            )
        ).contender_content_digest(spec)
        assert len({base, other_problem, other_eval, other_ctx}) == 4

    def test_duplicate_contender_labels_are_rejected(self):
        spec = ContenderSpec(family="riccati", config={"horizon": 12})
        with pytest.raises(ValueError, match="unique"):
            _experiment(contenders=(spec, spec))

    def test_from_case_study_binds_the_declaration(self):
        case_study = standard_lqr_case_study(
            StandardLQRConfig(
                state_dim=3,
                control_dim=2,
                horizon=6,
                batch_size=8,
                unfolded_epochs=2,
                neural_epochs=2,
                neural_hidden_dim=8,
                num_unfolding_iterations=3,
            )
        )
        experiment = Experiment.from_case_study(case_study)
        assert experiment.name == case_study.name
        assert experiment.problem is case_study.problem
        assert experiment.evaluation.batch_spec is case_study.batch_spec
        assert {spec.resolved_label for spec in experiment.contenders} == {
            recipe.label for recipe in case_study.recipes
        }
