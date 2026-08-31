import pytest

from mbl.applications.base import BaseApplication


class _FakeEngine:
    def __init__(self, metrics: dict) -> None:
        self.metrics = metrics
        self.run_calls = 0

    def train(self):
        return {}

    def evaluate(self):
        return {}

    def run(self):
        self.run_calls += 1
        return self.metrics


class _FakeTracker:
    """Records `log_params` calls instead of persisting anything, so
    `run()`'s application_name tagging can be verified without a real
    ExperimentTracker/filesystem."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.logged_params: dict = {}

    def log_params(self, params) -> None:
        self.logged_params.update(params)


class _FakeApplication(BaseApplication):
    """Records the arguments each hook was called with, so the template
    method's orchestration order/behavior can be verified independently of
    any real System/Cost/Controller machinery."""

    application_name = "FakeApp"

    def __init__(self, tracker_factory) -> None:
        super().__init__(tracker_factory)
        self.build_models_calls: list[str] = []
        self.build_engine_calls: list[tuple] = []
        self.engines: dict[str, _FakeEngine] = {}

    def build_problem(self):
        return "the-problem"

    def build_models(self, problem):
        self.build_models_calls.append(problem)
        return {"a": "model-a", "b": "model-b"}

    def build_engine(self, name, model, problem, tracker):
        self.build_engine_calls.append((name, model, problem, tracker))
        engine = _FakeEngine({"cost": len(name)})
        self.engines[name] = engine
        return engine


def test_run_builds_problem_once_and_models_once() -> None:
    app = _FakeApplication(tracker_factory=_FakeTracker)
    app.run()
    assert app.build_models_calls == ["the-problem"]


def test_run_builds_an_engine_per_model_with_a_fresh_tracker() -> None:
    trackers_created = []

    def tracker_factory(name):
        trackers_created.append(name)
        return _FakeTracker(name)

    app = _FakeApplication(tracker_factory)
    app.run()

    assert sorted(trackers_created) == ["a", "b"]
    called_names = {call[0] for call in app.build_engine_calls}
    assert called_names == {"a", "b"}
    for name, model, problem, tracker in app.build_engine_calls:
        assert model == f"model-{name}"
        assert problem == "the-problem"
        assert tracker.name == name


def test_run_calls_engine_run_and_returns_metrics_per_model() -> None:
    app = _FakeApplication(tracker_factory=_FakeTracker)
    results = app.run()

    assert results == {"a": {"cost": 1}, "b": {"cost": 1}}
    assert app.engines["a"].run_calls == 1
    assert app.engines["b"].run_calls == 1


def test_run_logs_application_name_on_every_tracker_before_running_the_engine() -> None:
    """The data-integrity patch: every run's metadata must record which
    Application produced it, tagged before the engine executes so it's
    present even if training later raises."""
    app = _FakeApplication(tracker_factory=_FakeTracker)
    results = app.run()

    assert results  # sanity: the run actually executed
    for _, _, _, tracker in app.build_engine_calls:
        assert tracker.logged_params == {"application_name": "FakeApp"}


def test_base_application_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        BaseApplication(  # type: ignore[abstract]  # instantiating the ABC is the test
            tracker_factory=lambda name: None  # type: ignore[return-value]
        )
