"""NB03 acceptance tests for `WarmStartUnfoldedRecipe`/`WarmStartSynthesizer`
(blueprint v2 SS3): the registry seam resolves the family as data, the
recipe's `build_engine` correctly compiles a `LayerwiseTrainingPlan` into
per-phase `LayerwiseGradientDescentStrategy`s, and the trained artifact
satisfies the same `Synthesizer` -> `TrainableController` lifecycle every
other trained family does -- executed end to end against the real engine,
never mocked (matching `tests/applications/test_recipes.py`'s own
methodology for the sibling families).
"""

import math

import pytest
import torch

from mbl.applications.factories import GaussianBatchSpec, LQRProblemFactory
from mbl.applications.recipes import (
    RECIPE_REGISTRY,
    UnfoldedKind,
    WarmStartSynthesizer,
    WarmStartUnfoldedRecipe,
    build_default_recipe_registry,
)
from mbl.applications.recipes.base import EngineHarness
from mbl.core.runtime import Backend, ComputeContext, Precision
from mbl.engine.training_plan import LayerwiseTrainingPlan, OptimizerSpec
from mbl.models.lifecycle import SynthesizedController, Synthesizer, TrainableController
from mbl.models.unfolded.iterative_refinement import RiccatiRefinement
from mbl.persistence.null_tracker import NullExperimentTracker

STATE_DIM, CONTROL_DIM, HORIZON = 3, 2, 6
NUM_ITERATIONS = 3
CTX = ComputeContext(backend=Backend.TORCH, device="cpu", precision=Precision.FLOAT64)


def _problem():
    return LQRProblemFactory(
        state_dim=STATE_DIM, control_dim=CONTROL_DIM, horizon=HORIZON, seed=0
    ).build()


def _batch_spec() -> GaussianBatchSpec:
    return GaussianBatchSpec(
        state_dim=STATE_DIM,
        horizon=HORIZON,
        batch_size=8,
        seed=0,
        process_noise_std=0.3,
    )


def _schedule(**overrides: object) -> LayerwiseTrainingPlan:
    defaults: dict[str, object] = dict(
        optimizer=OptimizerSpec("adam", 0.05),
        warmup_epochs_per_layer=2,
        refinement_epochs=2,
        train_matrix_from="refinement",
    )
    defaults.update(overrides)
    return LayerwiseTrainingPlan(**defaults)


def _recipe(**overrides: object) -> WarmStartUnfoldedRecipe:
    defaults: dict[str, object] = dict(
        kind=UnfoldedKind.LEARNED_STEP_SIZE_AND_MATRIX,
        schedule=_schedule(),
        num_iterations=NUM_ITERATIONS,
        step_size_init=0.02,
        step_size_max=0.5,
        horizon=HORIZON,
    )
    defaults.update(overrides)
    return WarmStartUnfoldedRecipe(**defaults)


def _harness() -> EngineHarness:
    return EngineHarness(
        batch_spec=_batch_spec(), tracker=NullExperimentTracker(), ctx=CTX
    )


class TestRegistryActivation:
    def test_family_is_registered(self) -> None:
        registry = build_default_recipe_registry()
        assert "unfolded_warmstart" in registry.available()

    def test_registry_resolves_the_family_as_data(self) -> None:
        recipe = RECIPE_REGISTRY.create(
            "unfolded_warmstart",
            kind=UnfoldedKind.LEARNED_STEP_SIZE_AND_MATRIX,
            schedule=_schedule(),
            num_iterations=NUM_ITERATIONS,
            step_size_init=0.02,
            step_size_max=0.5,
            horizon=HORIZON,
        )
        assert isinstance(recipe, WarmStartUnfoldedRecipe)
        assert recipe.label == "unfolded_warmstart_learned_step_size_and_matrix"


class TestConstructionGuards:
    def test_fixed_kind_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="WarmStartUnfoldedRecipe"):
            _recipe(kind=UnfoldedKind.FIXED)

    def test_label_defaults_from_kind(self) -> None:
        recipe = _recipe(kind=UnfoldedKind.LEARNED_STEP_SIZE)
        assert recipe.label == "unfolded_warmstart_learned_step_size"

    def test_explicit_label_is_honored(self) -> None:
        recipe = _recipe(label="custom")
        assert recipe.label == "custom"


class TestSignature:
    def test_signature_carries_the_schedule_and_never_embeds_problem(self) -> None:
        recipe = _recipe()
        signature = recipe.get_signature()
        assert signature["family"] == "unfolded_warmstart"
        assert signature["schedule"]["type"] == "LayerwiseTrainingPlan"

        def flat_keys(mapping):
            for key, value in mapping.items():
                yield key
                if isinstance(value, dict):
                    yield from flat_keys(value)

        assert "problem" not in set(flat_keys(signature))

    def test_distinct_schedules_are_distinct_signatures(self) -> None:
        base = _recipe().get_signature()
        other = _recipe(schedule=_schedule(warmup_epochs_per_layer=5)).get_signature()
        assert base != other


class TestAnalyticalInnerGradient:
    """A-1: the inner control-update gradient is the analytical closed form,
    never generic autograd -- the SAME RiccatiRefinement every other
    LEARNED_STEP_SIZE_AND_MATRIX contender uses."""

    def test_controller_uses_riccati_refinement_with_the_analytic_update(self) -> None:
        recipe = _recipe()
        controller = recipe.build_controller(_problem(), CTX)
        refinement = controller.config.iterative_refinement
        assert isinstance(refinement, RiccatiRefinement)

        M = torch.eye(CONTROL_DIM, dtype=torch.float64) * 2.0
        y_Ct = torch.ones(4, CONTROL_DIM, dtype=torch.float64)
        u = torch.zeros(4, CONTROL_DIM, dtype=torch.float64)
        assert torch.equal(refinement.get_gradient(u, M, y_Ct), u @ M + y_Ct)


class TestSynthesis:
    def test_build_synthesizer_returns_the_schedule_aware_sibling(self) -> None:
        recipe = _recipe()
        synthesizer = recipe.build_synthesizer(_problem(), _harness())
        assert isinstance(synthesizer, WarmStartSynthesizer)
        assert isinstance(synthesizer, Synthesizer)

    def test_synthesis_trains_through_every_compiled_phase(self) -> None:
        """End-to-end: build_engine's compiled multi-phase Runner actually
        trains -- both step_size AND the unified matrix move from init, and
        the artifact satisfies the standard trained-family lifecycle."""
        problem = _problem()
        recipe = _recipe()
        synthesizer = recipe.build_synthesizer(problem, _harness())

        reference = recipe.build_controller(problem, CTX)
        step_size_before = (
            reference.config.parameters["step_size"].get_raw().detach().clone()
        )
        matrix_before = (
            reference.config.parameters["riccati_matrix"].get_raw().detach().clone()
        )

        artifact = synthesizer.synthesize(problem, CTX)

        assert isinstance(artifact, SynthesizedController)
        assert isinstance(artifact, TrainableController)
        assert artifact.context == CTX

        trained_step_size = artifact.controller.config.parameters["step_size"].get_raw()
        trained_matrix = artifact.controller.config.parameters[
            "riccati_matrix"
        ].get_raw()
        assert not torch.equal(step_size_before, trained_step_size.detach())
        assert not torch.equal(matrix_before, trained_matrix.detach())

        provenance = artifact.provenance
        assert provenance["training"] == recipe.schedule.get_signature()
        assert math.isfinite(provenance["final_metrics"]["final_loss"])

    def test_the_reference_controller_is_untouched_by_synthesis(self) -> None:
        """Statelessness law: `synthesize` never mutates a controller built
        outside its own call."""
        problem = _problem()
        recipe = _recipe()
        reference = recipe.build_controller(problem, CTX)
        before = reference.config.parameters["step_size"].get_raw().detach().clone()

        recipe.build_synthesizer(problem, _harness()).synthesize(problem, CTX)

        after = reference.config.parameters["step_size"].get_raw().detach()
        assert torch.equal(before, after)


class TestInnerOuterGradientBoundary:
    """A-1b: the outer meta-gradient for P is gated by the schedule's
    train_matrix_from -- P is untouched through the warm-up phases
    (train_matrix_from="refinement") and only moves once the refinement
    phase actually runs, proving build_engine wires the per-phase
    activation correctly end to end (not just at the strategy unit level)."""

    def test_matrix_is_untouched_until_the_refinement_phase(self) -> None:
        problem = _problem()
        # Zero refinement epochs: the schedule never reaches the phase that
        # activates the matrix, so it must stay exactly at its init value.
        recipe = _recipe(
            schedule=_schedule(refinement_epochs=0, train_matrix_from="refinement")
        )
        reference = recipe.build_controller(problem, CTX)
        matrix_before = (
            reference.config.parameters["riccati_matrix"].get_raw().detach().clone()
        )

        artifact = recipe.build_synthesizer(problem, _harness()).synthesize(
            problem, CTX
        )

        trained_matrix = artifact.controller.config.parameters[
            "riccati_matrix"
        ].get_raw()
        assert torch.equal(matrix_before, trained_matrix.detach())
        # ... while step_size DID train during the warm-up phases.
        step_size_before = (
            reference.config.parameters["step_size"].get_raw().detach().clone()
        )
        trained_step_size = artifact.controller.config.parameters["step_size"].get_raw()
        assert not torch.equal(step_size_before, trained_step_size.detach())
