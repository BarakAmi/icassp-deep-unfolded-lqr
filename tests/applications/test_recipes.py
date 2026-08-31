"""Stage-S3 acceptance tests for the `ModelRecipe` layer (REFACTOR_PLAN v3,
T3.a/T2.f) and the trained families' lifecycle migration (T2.a/T2.b): the
registry is the routing seam, recipes own their engine wiring, and every
learned family satisfies `Synthesizer` -> `TrainableController`."""

import math

import pytest
import torch

from mbl.applications.recipes import (
    RECIPE_REGISTRY,
    EngineTrainedSynthesizer,
    FixedUnfoldedRecipe,
    NeuralRecipe,
    RiccatiRecipe,
    UnfoldedKind,
    UnfoldedRecipe,
    build_default_recipe_registry,
    null_harness,
)
from mbl.applications.standard_lqr import StandardLQRConfig, standard_lqr_case_study
from mbl.engine.training_plan import OptimizerSpec, TrainingPlan
from mbl.models.iterative.initializers import (
    ConstantInitializer,
    ControlInitMethod,
    SamplerInitializer,
    WarmStartInitializer,
)
from mbl.models.lifecycle import (
    SynthesizedController,
    Synthesizer,
    TrainableController,
)

_FAST_CONFIG = StandardLQRConfig(
    state_dim=3,
    control_dim=2,
    horizon=6,
    process_noise_std=0.3,
    batch_size=8,
    seed=0,
    num_unfolding_iterations=3,
    unfolded_learning_rate=0.05,
    unfolded_epochs=2,
    neural_hidden_dim=8,
    neural_learning_rate=1e-3,
    neural_epochs=2,
)


@pytest.fixture(scope="module")
def case_study():
    return standard_lqr_case_study(_FAST_CONFIG)


@pytest.fixture(scope="module")
def problem(case_study):
    return case_study.problem.build()


def _harness(case_study):
    return null_harness(case_study.batch_spec, case_study.ctx)


class TestRegistryActivation:
    def test_every_builtin_family_is_registered(self):
        registry = build_default_recipe_registry()
        assert {
            "riccati",
            "truncated_riccati",
            "neural",
            "unfolded",
            "unfolded_fixed",
            "cocp",
            "cocp_lower_bound",
        } <= set(registry.available())

    def test_registry_resolves_families_by_name_as_data(self):
        """The S4 `ContenderSpec` shape: (family name, config) resolves with
        no concrete-class import at the call site."""
        recipe = RECIPE_REGISTRY.create("riccati", horizon=6)
        assert isinstance(recipe, RiccatiRecipe)
        assert recipe.label == "analytic"

    def test_unknown_family_names_the_available_ones(self):
        with pytest.raises(KeyError, match="riccati"):
            RECIPE_REGISTRY.get("does_not_exist")


class TestCaseStudyAsData:
    def test_case_study_declares_no_string_ladders(self, case_study):
        """Route by label lookup, not conditionals: every recipe resolves,
        and an unknown label fails loudly naming the available ones."""
        for recipe in case_study.recipes:
            assert case_study.recipe(recipe.label) is recipe
        with pytest.raises(KeyError, match="analytic"):
            case_study.recipe("nope")

    def test_duplicate_labels_are_rejected(self, case_study):
        import dataclasses

        with pytest.raises(ValueError, match="unique"):
            dataclasses.replace(
                case_study, recipes=(case_study.recipes[0], case_study.recipes[0])
            )

    def test_case_study_signature_composes_the_specs(self, case_study):
        signature = case_study.get_signature()
        assert signature["problem"]["type"] == "LQRProblemFactory"
        assert signature["batch_spec"]["type"] == "GaussianBatchSpec"
        assert signature["compute_context"]["backend"] == "torch"
        assert set(signature["recipes"]) == {r.label for r in case_study.recipes}
        # Learned recipes sign their executed training plans (the documented
        # OptimizerSpec-in-signature provenance improvement).
        neural = signature["recipes"]["neural"]
        assert neural["plan"]["optimizer"]["learning_rate"] == (
            _FAST_CONFIG.neural_learning_rate
        )

    def test_recipe_signatures_never_embed_problem(self, case_study):
        def flat_keys(mapping):
            for key, value in mapping.items():
                yield key
                if isinstance(value, dict):
                    yield from flat_keys(value)

        for recipe in case_study.recipes:
            assert "problem" not in set(flat_keys(recipe.get_signature())), recipe.label


class TestTrainedFamilyLifecycle:
    """§7.3 for the trained families (the S3 half the S2 tests deferred)."""

    @pytest.mark.parametrize(
        "label", ["neural", "unfolded_learned_step_size", "unfolded_fixed"]
    )
    def test_synthesizer_protocol_and_trainable_artifact(
        self, case_study, problem, label
    ):
        recipe = case_study.recipe(label)
        synthesizer = recipe.build_synthesizer(problem, _harness(case_study))
        assert isinstance(synthesizer, Synthesizer)

        artifact = synthesizer.synthesize(problem, case_study.ctx)
        assert isinstance(artifact, SynthesizedController)
        assert isinstance(artifact, TrainableController)
        assert artifact.context == case_study.ctx

        module = artifact.as_module()
        assert isinstance(module, torch.nn.Module)
        assert list(module.parameters()), f"{label}: as_module exposes no parameters"

    def test_make_policy_is_fresh_per_call_and_rolls_out(self, case_study, problem):
        recipe = case_study.recipe("neural")
        artifact = recipe.build_synthesizer(problem, _harness(case_study)).synthesize(
            problem, case_study.ctx
        )
        assert artifact.make_policy() is not artifact.make_policy()

        sampler, _ = case_study.batch_spec.build(
            case_study.ctx.backend, torch_dtype=case_study.ctx.torch_dtype
        )
        x0, w, v = sampler()
        with torch.no_grad():
            X, _, U = problem.system.run(artifact.make_policy(), x0, w, v)
        assert X.shape == (_FAST_CONFIG.batch_size, _FAST_CONFIG.horizon + 1, 3)
        assert torch.isfinite(U).all()

    def test_engine_backed_synthesis_actually_trains(self, case_study, problem):
        """Offline phase = real training: parameters move, and the artifact
        carries the executed plan + final metrics as provenance."""
        recipe = case_study.recipe("unfolded_learned_step_size")
        synthesizer = recipe.build_synthesizer(problem, _harness(case_study))
        assert isinstance(synthesizer, EngineTrainedSynthesizer)

        controller = recipe.build_controller(problem, case_study.ctx)
        before = controller.config.parameters["step_size"].get_raw().detach().clone()

        artifact = synthesizer.synthesize(problem, case_study.ctx)
        trained = artifact.controller.config.parameters["step_size"].get_raw()
        # The synthesizer trains its OWN fresh controller (statelessness law:
        # the reference controller above is untouched)...
        assert torch.equal(
            before, controller.config.parameters["step_size"].get_raw().detach()
        )
        # ...and that fresh controller genuinely learned.
        assert not torch.equal(before, trained.detach())

        provenance = artifact.provenance
        assert provenance["training"] == recipe.plan.get_signature()
        assert math.isfinite(provenance["final_metrics"]["final_loss"])

    def test_frozen_family_synthesis_never_trains(self, case_study, problem):
        recipe = case_study.recipe("unfolded_fixed")
        artifact = recipe.build_synthesizer(problem, _harness(case_study)).synthesize(
            problem, case_study.ctx
        )
        raw = artifact.controller.config.parameters["step_size"].get_raw()
        assert not raw.requires_grad
        assert "training" not in artifact.provenance

    def test_artifact_signature_is_specification_time_only(self, case_study, problem):
        """C1 law at the trained-family artifact: synthesizer signature +
        residency, never problem.* and never live parameter values."""
        recipe = case_study.recipe("neural")
        artifact = recipe.build_synthesizer(problem, _harness(case_study)).synthesize(
            problem, case_study.ctx
        )
        signature = artifact.get_signature()
        assert signature["synthesizer"]["recipe"]["family"] == "neural"
        assert signature["compute_context"] == case_study.ctx.get_signature()

        def flat_keys(mapping):
            for key, value in mapping.items():
                yield key
                if isinstance(value, dict):
                    yield from flat_keys(value)

        assert "problem" not in set(flat_keys(signature))


class TestAsModuleSeam:
    def test_unfolded_as_module_aliases_the_live_raw_parameters(self, problem):
        """The T2.b law for structured-parameter families: the module's
        registered parameters ARE the refinement's raw tensors (aliased,
        never copied), so one optimizer trains what the rollout reads."""
        recipe = UnfoldedRecipe(
            kind=UnfoldedKind.LEARNED_STEP_SIZE,
            plan=TrainingPlan(optimizer=OptimizerSpec("adam", 0.05), epochs=1),
            num_iterations=3,
            step_size_init=0.05,
            step_size_max=1.0,
            horizon=6,
        )
        from mbl.core.runtime import Backend, ComputeContext

        controller = recipe.build_controller(
            problem, ComputeContext(backend=Backend.TORCH)
        )
        module = controller.as_module()
        assert module.step_size is controller.config.parameters["step_size"].get_raw()
        # Cached identity: optimizer, checkpoints, and resumes all see one module.
        assert controller.as_module() is module
        # And the checkpoint contract is no longer empty for this family.
        assert "step_size" in module.state_dict()

    def test_neural_and_fixed_recipes_expose_the_seam_too(self, case_study, problem):
        neural = case_study.recipe("neural").build_controller(problem, case_study.ctx)
        assert neural.as_module() is neural
        fixed = case_study.recipe("unfolded_fixed").build_controller(
            problem, case_study.ctx
        )
        assert list(fixed.as_module().parameters())


class TestUnfoldedRecipeGuards:
    def test_fixed_kind_is_rejected_by_the_trainable_recipe(self):
        with pytest.raises(ValueError, match="FixedUnfoldedRecipe"):
            UnfoldedRecipe(
                kind=UnfoldedKind.FIXED,
                plan=TrainingPlan(optimizer=OptimizerSpec("adam", 0.05), epochs=1),
                num_iterations=3,
                step_size_init=0.05,
                step_size_max=1.0,
                horizon=6,
            )

    def test_labels_default_from_kind(self):
        plan = TrainingPlan(optimizer=OptimizerSpec("adam", 0.05), epochs=1)
        recipe = UnfoldedRecipe(
            kind=UnfoldedKind.LEARNED_STEP_SIZE_AND_MATRIX,
            plan=plan,
            num_iterations=3,
            step_size_init=0.05,
            step_size_max=1.0,
            horizon=6,
        )
        assert recipe.label == "unfolded_learned_step_size_and_matrix"
        assert (
            FixedUnfoldedRecipe(
                num_iterations=3, step_size_init=0.05, step_size_max=1.0, horizon=6
            ).label
            == "unfolded_fixed"
        )
        assert NeuralRecipe(hidden_dim=8, plan=plan).label == "neural"


class TestUnfoldedRecipeInitMethod:
    """The control-init knob (Phase 2.1): selectable per iterative controller,
    with a backward-compatible signature so the default cold start leaves every
    existing digest -- and the golden masters -- untouched."""

    def _recipe(self, **overrides) -> UnfoldedRecipe:
        base = dict(
            kind=UnfoldedKind.LEARNED_STEP_SIZE,
            plan=TrainingPlan(optimizer=OptimizerSpec("adam", 0.05), epochs=1),
            num_iterations=3,
            step_size_init=0.05,
            step_size_max=1.0,
            horizon=6,
        )
        return UnfoldedRecipe(**{**base, **overrides})

    def test_cold_is_the_default_and_omitted_from_the_signature(self):
        recipe = self._recipe()
        signature = recipe.get_signature()

        assert recipe.init_method is ControlInitMethod.COLD
        # Backward-compat: the default contributes no init keys at all.
        assert "init_method" not in signature
        assert "random_init_std" not in signature
        assert "random_init_seed" not in signature

    def test_warm_appears_in_the_signature_without_random_params(self):
        signature = self._recipe(init_method=ControlInitMethod.WARM).get_signature()

        assert signature["init_method"] == "warm"
        assert "random_init_std" not in signature
        assert "random_init_seed" not in signature

    def test_randomized_carries_its_std_and_seed_in_the_signature(self):
        signature = self._recipe(
            init_method=ControlInitMethod.RANDOMIZED,
            random_init_std=0.5,
            random_init_seed=11,
        ).get_signature()

        assert signature["init_method"] == "randomized"
        assert signature["random_init_std"] == 0.5
        assert signature["random_init_seed"] == 11

    def test_build_controller_uses_the_selected_initializer(self, problem):
        from mbl.core.runtime import Backend, ComputeContext

        ctx = ComputeContext(backend=Backend.TORCH)

        cold = self._recipe().build_controller(problem, ctx)
        warm = self._recipe(init_method=ControlInitMethod.WARM).build_controller(
            problem, ctx
        )
        randomized = self._recipe(
            init_method=ControlInitMethod.RANDOMIZED
        ).build_controller(problem, ctx)

        assert isinstance(cold.config.control_initializer, ConstantInitializer)
        assert isinstance(warm.config.control_initializer, WarmStartInitializer)
        assert isinstance(randomized.config.control_initializer, SamplerInitializer)


class TestUnfoldedConvergenceParametersSnapshot:
    """Regression coverage for the convergence-replay snapshot staleness bug:
    `unfolded_convergence_parameters` used to call `.get()` eagerly (at
    `extra_callbacks` time, before training starts), freezing the UNTRAINED
    initial step size/Riccati-matrix into the `ParameterSnapshotCallback`
    artifact forever, however much training subsequently happened -- Phase 3
    animations replaying "learned parameter" evolution would always show the
    initial value. The fix returns the parameter's own `.get` bound method
    (a zero-argument callable), which `ParameterSnapshotCallback` now
    re-invokes at `on_train_end` time (see engine/callbacks.py)."""

    def test_step_size_snapshot_is_lazy_and_reflects_post_training_updates(
        self, problem
    ):
        from mbl.applications.recipes.unfolded import unfolded_convergence_parameters
        from mbl.core.runtime import Backend, ComputeContext

        recipe = UnfoldedRecipe(
            kind=UnfoldedKind.LEARNED_STEP_SIZE,
            plan=TrainingPlan(optimizer=OptimizerSpec("adam", 0.05), epochs=1),
            num_iterations=3,
            step_size_init=0.05,
            step_size_max=1.0,
            horizon=6,
        )
        controller = recipe.build_controller(
            problem, ComputeContext(backend=Backend.TORCH)
        )
        parameters = unfolded_convergence_parameters(controller, problem, horizon=6)

        assert callable(parameters["step_size"])
        before = parameters["step_size"]().detach().clone()
        with torch.no_grad():
            # Simulate an optimizer update on the raw (reparameterized-space)
            # parameter -- exactly what `LayerwiseGradientDescentStrategy`'s
            # Adam optimizer does during training.
            controller.config.parameters["step_size"].get_raw().add_(10.0)
        after = parameters["step_size"]()

        assert not torch.equal(before, after)

    def test_riccati_matrix_snapshot_is_lazy_and_reflects_post_training_updates(
        self, problem
    ):
        from mbl.applications.recipes.unfolded import unfolded_convergence_parameters
        from mbl.core.runtime import Backend, ComputeContext

        recipe = UnfoldedRecipe(
            kind=UnfoldedKind.LEARNED_STEP_SIZE_AND_MATRIX,
            plan=TrainingPlan(optimizer=OptimizerSpec("adam", 0.05), epochs=1),
            num_iterations=3,
            step_size_init=0.05,
            step_size_max=1.0,
            horizon=6,
        )
        controller = recipe.build_controller(
            problem, ComputeContext(backend=Backend.TORCH)
        )
        parameters = unfolded_convergence_parameters(controller, problem, horizon=6)

        assert callable(parameters["riccati_matrix"])
        before = parameters["riccati_matrix"]().detach().clone()
        with torch.no_grad():
            controller.config.parameters["riccati_matrix"].get_raw().add_(10.0)
        after = parameters["riccati_matrix"]()

        assert not torch.equal(before, after)

    def test_static_problem_matrices_are_not_wrapped_in_callables(self, problem):
        """ "A"/"B"/"R"/"P" never change during training -- unlike the
        learned entries, they are the plain arrays themselves."""
        from mbl.applications.recipes.unfolded import unfolded_convergence_parameters
        from mbl.core.runtime import Backend, ComputeContext

        recipe = UnfoldedRecipe(
            kind=UnfoldedKind.LEARNED_STEP_SIZE,
            plan=TrainingPlan(optimizer=OptimizerSpec("adam", 0.05), epochs=1),
            num_iterations=3,
            step_size_init=0.05,
            step_size_max=1.0,
            horizon=6,
        )
        controller = recipe.build_controller(
            problem, ComputeContext(backend=Backend.TORCH)
        )
        parameters = unfolded_convergence_parameters(controller, problem, horizon=6)

        for key in ("A", "B", "R", "P"):
            assert not callable(parameters[key]), key


class TestCocpSolverDeviceResolution:
    """Regression: `COCPRecipe.build_controller` previously passed
    `ctx.device` straight through to `resolve_cocp_solver`, so on any
    CUDA-available machine the canary silently tested MOREAU-CUDA (already
    known unreliable, NB04 COCP/viz refinement plan Sec 1.0.6) instead of
    the verified-safe, ~15-60x-faster MOREAU-CPU, and fell back to the much
    slower DIFFCP/SCS default even where MOREAU-CPU would have passed.
    `solver_device` (a new recipe field, defaulting to ``"cpu"``,
    deliberately independent of `ctx.device`) fixes this: the resolved
    solver must be the same regardless of which device the REST of the
    experiment runs on."""

    def _cocp_problem(self):
        from mbl.applications.factories import LQRProblemFactory

        return LQRProblemFactory(
            state_dim=2, control_dim=1, horizon=4, seed=0, u_max=0.5
        ).build()

    def test_solver_device_defaults_to_cpu_independent_of_ctx(self):
        from mbl.applications.recipes.cocp import COCPRecipe
        from mbl.core.runtime import Backend, ComputeContext

        problem = self._cocp_problem()
        recipe = COCPRecipe(
            plan=TrainingPlan(optimizer=OptimizerSpec("adam", 0.1), epochs=1)
        )
        assert recipe.solver_device == "cpu"

        ctx = ComputeContext(backend=Backend.TORCH, device="cpu")
        controller = recipe.build_controller(problem, ctx)

        # The resolved solver is evaluated/run against `solver_device`
        # ("cpu"), never `ctx.device` -- the two are DELIBERATELY decoupled.
        assert controller.config.solver.device == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
    def test_moreau_resolves_even_when_ctx_device_is_cuda(self):
        """The real regression scenario: on a CUDA-available machine,
        `ctx.device` auto-selects "cuda" -- confirm the resolved solver is
        still evaluated against `solver_device="cpu"` (MOREAU passes its
        canary here, matching this project's own verified-safe finding),
        not silently downgraded to DIFFCP/SCS because `ctx.device` leaked
        into the canary."""
        from mbl.applications.recipes.cocp import COCPRecipe
        from mbl.core.runtime import Backend, ComputeContext

        problem = self._cocp_problem()
        recipe = COCPRecipe(
            plan=TrainingPlan(optimizer=OptimizerSpec("adam", 0.1), epochs=1)
        )
        ctx = ComputeContext(backend=Backend.TORCH, device="cuda")

        controller = recipe.build_controller(problem, ctx)

        assert controller.config.solver.device == "cpu"


class TestNeuralRecipeDeviceTransfer:
    """`NeuralRecipe.build_controller` must move the constructed policy to
    BOTH `ctx.torch_dtype` AND `ctx.torch_device` -- a bug (only `dtype` was
    forwarded) that no prior notebook's contender list ever exercised
    (neither NB03 nor NB04 includes a `neural` contender), caught only by
    actually training a GRU end to end on a CUDA-available machine."""

    def test_policy_parameters_land_on_the_context_device_cpu(self) -> None:
        problem = standard_lqr_case_study(_FAST_CONFIG).problem.build()
        recipe = NeuralRecipe(
            hidden_dim=4,
            plan=TrainingPlan(optimizer=OptimizerSpec("adam", 1e-3), epochs=1),
        )
        from mbl.core.runtime import Backend, ComputeContext

        ctx = ComputeContext(backend=Backend.TORCH, device="cpu")

        policy = recipe.build_controller(problem, ctx)

        assert next(policy.parameters()).device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
    def test_policy_parameters_land_on_the_context_device_cuda(self) -> None:
        """The real regression scenario: on a CUDA-available machine,
        `ctx.device` auto-selects "cuda" -- confirm the policy's own
        parameters actually move there instead of silently staying on CPU
        while every rollout tensor is CUDA-resident."""
        from mbl.core.runtime import Backend, ComputeContext

        problem = standard_lqr_case_study(_FAST_CONFIG).problem.build()
        recipe = NeuralRecipe(
            hidden_dim=4,
            plan=TrainingPlan(optimizer=OptimizerSpec("adam", 1e-3), epochs=1),
        )
        ctx = ComputeContext(backend=Backend.TORCH, device="cuda")

        policy = recipe.build_controller(problem, ctx)

        assert next(policy.parameters()).device.type == "cuda"
