"""`CaseStudy` -- a case study as *data* (REFACTOR_PLAN v3, T3.a): one
problem factory, one shared batch spec, one compute context, and the tuple
of `ModelRecipe`s under comparison. `CaseStudyApplication` is the single
execution chassis that runs any such declaration through the engine --
`StandardLQRApp` and `BoxConstraintLQRApp` shrink from ~400/500-line
monoliths (M1) to declarations consumed by this one class, and adding a
model family means appending one recipe, never editing an ``if name ==``
ladder (M2)."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .base import BaseApplication
from .factories import GaussianBatchSpec, ProblemFactory
from .recipes.base import EngineHarness, ModelRecipe
from ..core.optimal_control_problem import OptimalControlProblem
from ..core.runtime import ComputeContext
from ..engine.engine import Engine
from ..models.base import Controller
from ..persistence.tracker import ExperimentTracker

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CaseStudy:
    """One benchmark comparison, fully declared: which problem, which
    sampling protocol, which compute substrate, which model families.

    Attributes:
        name: The case study's identity (tagged as ``application_name`` on
            every run it produces).
        problem: The signable problem specification.
        batch_spec: The shared batch-sampling specification.
        recipes: The model families under comparison, each with a unique
            `label`.
        ctx: The single dtype/device/backend authority for every recipe.
    """

    name: str
    problem: ProblemFactory
    batch_spec: GaussianBatchSpec
    recipes: tuple[ModelRecipe, ...]
    ctx: ComputeContext

    def __post_init__(self) -> None:
        """Fail-fast guard: recipe labels must be unique (they name runs).

        Raises:
            ValueError: If two recipes share a label.
        """
        labels = [recipe.label for recipe in self.recipes]
        if len(set(labels)) != len(labels):
            raise ValueError(f"Recipe labels must be unique, got {labels}.")

    def recipe(self, label: str) -> ModelRecipe:
        """Look up a recipe by its instance label.

        Args:
            label: The recipe's `label`.

        Returns:
            The matching recipe.

        Raises:
            KeyError: If no recipe carries `label`; the message names the
                available labels.
        """
        for recipe in self.recipes:
            if recipe.label == label:
                return recipe
        raise KeyError(
            f"Unknown model {label!r}. "
            f"Available: {[recipe.label for recipe in self.recipes]}"
        )

    def get_signature(self) -> dict[str, Any]:
        """`Signable` member: the full case-study specification.

        Returns:
            ``{"type":, "name":, "problem":, "batch_spec":,
            "compute_context":, "recipes": {label: ...}}``.
        """
        return {
            "type": type(self).__name__,
            "name": self.name,
            "problem": self.problem.get_signature(),
            "batch_spec": self.batch_spec.get_signature(),
            "compute_context": self.ctx.get_signature(),
            "recipes": {
                recipe.label: recipe.get_signature() for recipe in self.recipes
            },
        }


class CaseStudyApplication(BaseApplication):
    """The one `BaseApplication` implementation: executes any `CaseStudy`
    declaration. The three hooks route through the declaration's factory
    and recipes -- there is no model knowledge in this class."""

    def __init__(
        self,
        case_study: CaseStudy,
        tracker_factory: Callable[[str], ExperimentTracker],
    ) -> None:
        """
        Args:
            case_study: The declaration to execute.
            tracker_factory: Builds a fresh `ExperimentTracker` per model
                label (see `BaseApplication`).
        """
        super().__init__(tracker_factory)
        self.case_study = case_study

    @property
    def application_name(self) -> str:
        """The case study's name (see `BaseApplication.application_name`)."""
        return self.case_study.name

    def build_problem(self) -> OptimalControlProblem:
        """Construct the shared problem from the declared factory."""
        return self.case_study.problem.build()

    def build_models(self, problem: OptimalControlProblem) -> dict[str, Controller]:
        """Construct every recipe's controller, keyed by recipe label."""
        ctx = self.case_study.ctx
        return {
            recipe.label: recipe.build_controller(problem, ctx)
            for recipe in self.case_study.recipes
        }

    def build_engine(
        self,
        name: str,
        model: Controller,
        problem: OptimalControlProblem,
        tracker: ExperimentTracker,
    ) -> Engine:
        """Wire `model` through its recipe's family-owned engine wiring.

        Args:
            name: The recipe label.
            model: The controller built by that recipe.
            problem: The shared problem.
            tracker: This run's tracker.

        Returns:
            The fully-configured `Engine`.
        """
        recipe = self.case_study.recipe(name)
        harness = EngineHarness(
            batch_spec=self.case_study.batch_spec,
            tracker=tracker,
            ctx=self.case_study.ctx,
        )
        logger.info(
            "Wiring engine for %r (family %r) in case study %r",
            name,
            type(recipe).__name__,
            self.case_study.name,
        )
        return recipe.build_engine(model, problem, harness)
