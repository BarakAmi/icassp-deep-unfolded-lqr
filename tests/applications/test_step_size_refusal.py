"""A declared step size outside its own bound must be REFUSED, not silently
clamped.

The defect this suite exists to prevent, measured on the tree before the fix:
`StepSizeParameter.initialize` reparameterises `alpha_init` through a sigmoid
and clamps the fraction into ``(tol, 1 - tol)``, so **any** declared step at or
above `alpha_max` executes as ``alpha_max * (1 - 1e-12)`` and any declared step
at or below zero executes as ``+1e-12``. Neither raises, and neither warns.

The consequence is not merely a surprising number. `step_size_init` reaches the
`ModelID` while the clamp collapses whole intervals of it onto one executed
tensor, so two declarations that produce a **bit-identical trained model** carry
**two different identifiers** -- measured, on the frozen campaign plant:

    step_size_init = 500.0  -> model_id aa75677015c3116a
    step_size_init = 1000.0 -> model_id 672027add42d6f74
    max |alpha(500) - alpha(1000)| = 0.0 (bitwise identical, before AND after
    20 Adam steps from seed 0 on an identical batch, loss curves included)

That is a store that cannot answer "have I trained this model?", which is the
one question a content-addressed store exists to answer.

The refusal already exists verbatim on the legacy surface -- `StandardLQRConfig`,
`BoxConstraintLQRConfig` and FROZEN `src/lqr/configs.py` all raise on the same
predicate. The tier that never inherited it is the Tier-4 recipe layer the
ICASSP campaign runs on, which is what this suite closes.
"""

from pathlib import Path
from shutil import copyfile

import pytest
import torch

from mbl.applications.recipes import (
    FixedUnfoldedRecipe,
    UnfoldedKind,
    UnfoldedRecipe,
    WarmStartUnfoldedRecipe,
)
from mbl.engine.training_plan import (
    LayerwiseTrainingPlan,
    OptimizerSpec,
    TrainingPlan,
)
from mbl.experiments import DEFAULT_SPEC_BINDINGS
from mbl.models.unfolded.parameters import (
    MatrixModulationParameterConfig,
    StepSizeParameterConfig,
)
from mbl.spec.errors import SpecificationError
from mbl.spec.loader import load_study

DTYPE = torch.float64
STUDIES = Path(__file__).resolve().parents[2] / "studies"
TRACKED = STUDIES / "box_lqr" / "depth_scaling.toml"


def _plan() -> TrainingPlan:
    return TrainingPlan(optimizer=OptimizerSpec("adam", 0.05), epochs=1)


def _step_size_config(
    *, alpha_init: float = 0.05, alpha_max: float = 1.0
) -> StepSizeParameterConfig:
    """The campaign's own declaration, with only the two fields under test free.

    Written with explicit keywords rather than a `**overrides` dict so that
    every argument keeps its declared type: unpacking an inferred
    `dict[str, object]` into a dataclass needs a `type: ignore` whose own
    necessity depends on the checker's mood, and an ignore that turns out to be
    unused is itself an error under this repository's strict settings.
    """
    return StepSizeParameterConfig(
        name="step_size",
        num_iterations=8,
        action_dim=2,
        alpha_init=alpha_init,
        alpha_max=alpha_max,
        dtype=DTYPE,
    )


class TestTheParameterConfigRefuses:
    """Tier 2. `build_unfolded_controller` is reached directly by
    `mbl.workbench.robustness` and by the NBxx study modules, none of which
    pass through a recipe `__post_init__` -- so the recipe-level refusal below
    does not cover them and this one is not redundant with it."""

    @pytest.mark.parametrize(
        ("alpha_init", "reason"),
        [
            (2.0, "declared above the bound"),
            (1.0, "declared AT the bound -- the check must be strict"),
            (1000.0, "declared far above the bound"),
            (0.0, "declared zero"),
            (-5.0, "a sign typo"),
            (-1e-09, "a sign typo too small to notice"),
        ],
    )
    def test_a_step_outside_the_open_bound_is_refused(
        self, alpha_init: float, reason: str
    ) -> None:
        """Each of these executed silently before the fix. Measured then:
        2.0/1.0/1000.0 all executed 0.9999999999989999 (bitwise identical to
        each other), and 0.0/-5.0/-1e-9 all executed +1.000000e-12."""
        with pytest.raises(ValueError, match="alpha_init") as error:
            _step_size_config(alpha_init=alpha_init)
        assert repr(alpha_init) in str(error.value), reason

    def test_a_non_positive_bound_is_refused(self) -> None:
        """`alpha_max = 0.0` executed a step of EXACTLY 0.0 before the fix --
        a controller that never refines, built without a word.

        The assertion is on the wording ONLY the dedicated bound check
        produces, not on `"alpha_max"` and not on the exception type. Both of
        those hold with that check deleted: the range check below it raises for
        every non-positive bound anyway (`0 < init < max` is unsatisfiable when
        `max <= 0`) and its message quotes `alpha_max=0.0` in passing. A mutant
        removing the bound check survived this test until it asserted the
        message that distinguishes them -- the same defect this project has now
        recorded three times.

        The bound check is therefore kept for its MESSAGE, which names the
        field the author has to edit, rather than for raising at all.
        """
        with pytest.raises(ValueError, match="must be positive") as error:
            _step_size_config(alpha_init=0.05, alpha_max=0.0)
        assert "never moves" in str(error.value), str(error.value)

    def test_the_legal_interior_is_untouched(self) -> None:
        """The negative control. 31,123 instrumented constructions across the
        full suite had `init < max` with zero violations, so a refusal that
        also rejected legal values would be caught here rather than by 5,451
        unrelated tests going red."""
        for alpha_init in (1e-9, 0.05, 0.5, 0.999999):
            assert _step_size_config(alpha_init=alpha_init).alpha_init == alpha_init

    def test_the_modulation_parameter_carries_the_same_refusal(self) -> None:
        """The identical clamp lives in `MatrixModulationParameter` -- measured,
        both saturate at the same fraction 0.9999999999989999 to the bit. Its
        `modulation_init` is hard-coded by the recipe layer, so THIS config
        guard is the only one that can ever fire for it."""
        with pytest.raises(ValueError, match="modulation_init"):
            MatrixModulationParameterConfig(
                name="matrix_modulation",
                num_iterations=4,
                modulation_init=5000.0,
                modulation_max=5.0,
                dtype=DTYPE,
            )
        # And its legal interior survives (the value the recipe layer passes).
        assert (
            MatrixModulationParameterConfig(
                name="matrix_modulation",
                num_iterations=4,
                modulation_init=1.0,
                modulation_max=5.0,
                dtype=DTYPE,
            ).modulation_init
            == 1.0
        )


class TestEveryUnfoldedRecipeRefuses:
    """Tier 4, and this is the tier that fires while the DOCUMENT IS OPEN.

    Measured: loading and materialising a real study document fires the three
    recipe `__post_init__`s 17 times each and constructs **zero**
    `StepSizeParameter`s -- the parameter is built inside
    `build_unfolded_controller`, i.e. after the study has loaded and a problem
    has been built. A Tier-2-only refusal would therefore fire minutes into a
    run rather than at load.

    All three families are checked separately and deliberately: the campaign's
    analytic PGD is `unfolded_fixed` and its flagship is `unfolded_warmstart`,
    so a refusal added only to `UnfoldedRecipe` would leave both of the
    contenders that matter silent.
    """

    def test_the_trainable_recipe_refuses(self) -> None:
        with pytest.raises(ValueError, match="step_size_init"):
            UnfoldedRecipe(
                kind=UnfoldedKind.LEARNED_STEP_SIZE,
                plan=_plan(),
                num_iterations=3,
                step_size_init=500.0,
                step_size_max=1.0,
                horizon=6,
            )

    def test_the_fixed_recipe_refuses(self) -> None:
        """`unfolded_fixed` is the analytic PGD -- the contender whose step size
        is the whole of its behaviour."""
        with pytest.raises(ValueError, match="step_size_init"):
            FixedUnfoldedRecipe(
                num_iterations=3,
                step_size_init=500.0,
                step_size_max=1.0,
                horizon=6,
            )

    def test_the_warm_start_recipe_refuses(self) -> None:
        """`unfolded_warmstart` is the campaign's flagship."""
        with pytest.raises(ValueError, match="step_size_init"):
            WarmStartUnfoldedRecipe(
                kind=UnfoldedKind.LEARNED_STEP_SIZE,
                schedule=LayerwiseTrainingPlan(
                    optimizer=OptimizerSpec("adam", 0.05),
                    warmup_epochs_per_layer=1,
                    refinement_epochs=1,
                ),
                num_iterations=3,
                step_size_init=500.0,
                step_size_max=1.0,
                horizon=6,
            )

    def test_the_message_names_the_value_that_would_have_executed(self) -> None:
        """Asserting only the exception TYPE cannot tell this check from the
        several others in these `__post_init__`s that raise `ValueError` too --
        a recurring defect in this project's own suites. The message must carry
        both the declared value and the one the clamp would have substituted,
        because the whole failure is that those two differ in silence."""
        with pytest.raises(ValueError) as error:
            FixedUnfoldedRecipe(
                num_iterations=3,
                step_size_init=500.0,
                step_size_max=1.0,
                horizon=6,
            )
        message = str(error.value)
        assert "FixedUnfoldedRecipe" in message
        assert "500.0" in message
        assert "step_size_max=1.0" in message
        # The value the clamp WOULD have run, which is the half this test is
        # named for and did not check until a mutant that deleted it survived.
        # Both numbers have to be present: the defect is precisely that the
        # declared one and the executed one differ without anyone being told.
        assert repr(1.0 * (1 - 1e-12)) in message, message

    def test_the_declared_step_the_campaign_uses_is_untouched(self) -> None:
        """The negative control at this tier: every one of the 162 models in
        the warm store declares (0.05, 1.0), and every tracked study declares
        it too, so a refusal that caught them would orphan the store."""
        recipe = FixedUnfoldedRecipe(
            num_iterations=3, step_size_init=0.05, step_size_max=1.0, horizon=6
        )
        assert recipe.step_size_init == 0.05

    def test_the_1_over_L_step_of_every_frozen_plant_is_accepted(self) -> None:
        """The refusal must not reject the value Phase C is about to declare.
        1/L measured on the seven frozen ICASSP plants spans 1.047e-3 (n = 50)
        to 5.703e-2 (n = 4) -- all strictly inside (0, 1.0), so the analytic
        PGD's principled step is legal at the declared bound. It would NOT be
        legal against NB03's `step_size_max = 0.05`, where the n = 4 plant's
        1/L = 0.0570 exceeds the bound; that case must now raise instead of
        silently executing 0.05, a 12.3 % error.
        """
        for one_over_l in (1.047362e-3, 4.171932e-3, 3.3498821e-2, 5.702755e-2):
            assert (
                FixedUnfoldedRecipe(
                    num_iterations=3,
                    step_size_init=one_over_l,
                    step_size_max=1.0,
                    horizon=6,
                ).step_size_init
                == one_over_l
            )
        with pytest.raises(ValueError, match="step_size_init"):
            FixedUnfoldedRecipe(
                num_iterations=3,
                step_size_init=5.702755e-2,
                step_size_max=0.05,
                horizon=6,
            )


class TestTheDocumentIsRefusedAtLoad:
    """The surface an author actually edits.

    This is the check that makes the refusal worth having: without it a typo in
    a TOML costs a run rather than a load. Measured before the fix -- a document
    declaring `step_size_init = 500.0` loaded clean and materialised 27 points.
    """

    @staticmethod
    def _perturbed(tmp_path: Path, value: str) -> Path:
        """`depth_scaling.toml` with the FIXED contender's step size replaced.

        Contender index 1 on purpose: it is `standard_pgd`, the analytic PGD,
        i.e. the `unfolded_fixed` family -- the one a refusal written only on
        `UnfoldedRecipe` would miss.

        The derived document is asserted to actually differ from its source.
        A fixture built by `str.replace` goes silently inert when its pattern
        stops matching, which in a "this must be refused" test turns the case
        into a byte-identical copy of a VALID document that goes on passing --
        a defect this project has already shipped once.
        """
        source = TRACKED.read_text(encoding="utf-8")
        derived = source.replace(
            "step_size_init = 0.05", f"step_size_init = {value}", 1
        )
        assert derived != source, (
            "the fixture's pattern no longer matches depth_scaling.toml, so this "
            "test is asserting that a VALID document is refused"
        )
        destination = tmp_path / "depth_scaling.toml"
        destination.write_text(derived, encoding="utf-8")
        copyfile(
            TRACKED.parent / "box_lqr_n4m2_N50_u0.5_s0.npz",
            tmp_path / "box_lqr_n4m2_N50_u0.5_s0.npz",
        )
        return destination

    def test_a_step_above_its_bound_is_refused_while_the_document_is_open(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(SpecificationError) as error:
            load_study(
                self._perturbed(tmp_path, "500.0"), bindings=DEFAULT_SPEC_BINDINGS
            )
        message = str(error.value)
        assert "contenders[1]" in message, message
        assert "step_size_init" in message, message

    def test_the_unperturbed_document_still_loads(self, tmp_path: Path) -> None:
        """The negative control the refusal above is worthless without: the
        same copy, same loader, same bindings, only the value differs."""
        destination = tmp_path / "depth_scaling.toml"
        destination.write_text(TRACKED.read_text(encoding="utf-8"), encoding="utf-8")
        copyfile(
            TRACKED.parent / "box_lqr_n4m2_N50_u0.5_s0.npz",
            tmp_path / "box_lqr_n4m2_N50_u0.5_s0.npz",
        )
        document = load_study(destination, bindings=DEFAULT_SPEC_BINDINGS)
        # The unfolded contenders specifically -- asserting merely that SOMETHING
        # loaded would hold for a document with no step size in it at all, and
        # this control exists to prove the refusal above is about the VALUE.
        families = {contender.family for contender in document.study.contenders}
        assert {"unfolded", "unfolded_fixed", "unfolded_warmstart"} <= families, (
            families
        )
