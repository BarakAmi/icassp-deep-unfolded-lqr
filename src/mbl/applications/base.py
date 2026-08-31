"""Template-method orchestration for a case study: a family of Controllers
compared on one shared OptimalControlProblem, each run through its own
Engine (Runner) and ExperimentTracker.

Concrete applications (e.g. StandardLQRApp) implement the three "what" hooks
below; `run()` itself is the fixed "how" -- build the problem once, build
every model, run each through its own engine -- so new case studies are added
by writing a new subclass, never by editing this one (Open/Closed Principle).
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping

from ..core.optimal_control_problem import OptimalControlProblem
from ..engine.engine import Engine
from ..models.base import Controller
from ..persistence.tracker import ExperimentTracker

logger = logging.getLogger(__name__)


class BaseApplication(ABC):
    """Orchestrates System + Cost + Models + Runner + ExperimentTracker for
    one case study.

    Args:
        tracker_factory: builds a fresh ExperimentTracker for a given model
            name (e.g. `functools.partial(LocalExperimentTracker, root=...)`),
            so each compared model gets its own run directory.
    """

    def __init__(self, tracker_factory: Callable[[str], ExperimentTracker]) -> None:
        self._tracker_factory = tracker_factory

    @property
    @abstractmethod
    def application_name(self) -> str:
        """Human-readable identity of this case study (e.g. "StandardLQR"),
        logged as a param on every run this app produces (see `run` below) so
        downstream tooling (the Streamlit dashboard) can group/filter runs by
        application without having to infer it from run_id naming
        conventions -- metadata.json has no other place this lives."""
        raise NotImplementedError

    @abstractmethod
    def build_problem(self) -> OptimalControlProblem:
        """Construct the System + Cost pair shared by every model being compared."""
        raise NotImplementedError

    @abstractmethod
    def build_models(self, problem: OptimalControlProblem) -> Mapping[str, Controller]:
        """Construct the named set of Controllers to compare on `problem`."""
        raise NotImplementedError

    @abstractmethod
    def build_engine(
        self,
        name: str,
        model: Controller,
        problem: OptimalControlProblem,
        tracker: ExperimentTracker,
    ) -> Engine:
        """Wire one named model into a fully-configured Engine: the matching
        TrainingStrategy/optimizer/callbacks/epoch count for whether `model`
        is closed-form, frozen, or gradient-trained."""
        raise NotImplementedError

    def run(self) -> dict[str, Mapping[str, float]]:
        """Build the problem once, build every model, run each through its
        own Engine (train then evaluate), and return {model_name: metrics}.

        Each model's tracker is tagged with `application_name` as soon as
        it's created (before the engine runs), so the field is present in
        metadata.json even if training later raises.
        """
        logger.info("Case study %r started.", self.application_name)
        problem = self.build_problem()
        models = self.build_models(problem)
        results = {}
        for name, model in models.items():
            logger.info("Model %r started (%s).", name, type(model).__name__)
            tracker = self._tracker_factory(name)
            tracker.log_params({"application_name": self.application_name})
            results[name] = self.build_engine(name, model, problem, tracker).run()
            logger.info("Model %r finished: %s", name, dict(results[name]))
        logger.info("Case study %r finished.", self.application_name)
        return results
