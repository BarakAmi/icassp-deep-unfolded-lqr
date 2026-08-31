import functools
import math

import pytest
import torch

from mbl.applications.standard_lqr import StandardLQRApp, StandardLQRConfig
from mbl.models.analytic.riccati import RiccatiController
from mbl.models.neural.nerual import NeuralPolicy
from mbl.models.unfolded.base import UnfoldedController
from mbl.persistence.local_tracker import LocalExperimentTracker

# Small but non-trivial: multi-dimensional state/control, several time steps
# and unfolding iterations, a couple of training epochs per learnable model --
# enough to prove the wiring actually executes without being a slow, full
# convergence run.
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


def _make_app(tmp_path) -> StandardLQRApp:
    tracker_factory = functools.partial(LocalExperimentTracker, tmp_path)
    return StandardLQRApp(tracker_factory, config=_FAST_CONFIG)


def test_build_problem_constructs_expected_dimensions(tmp_path) -> None:
    app = _make_app(tmp_path)
    problem = app.build_problem()

    assert problem.system.dimensions.state_dim == _FAST_CONFIG.state_dim
    assert problem.system.dimensions.control_dim == _FAST_CONFIG.control_dim
    assert problem.system.dimensions.observation_dim == _FAST_CONFIG.state_dim
    assert problem.cost.Q.shape == (
        _FAST_CONFIG.horizon + 1,
        _FAST_CONFIG.state_dim,
        _FAST_CONFIG.state_dim,
    )
    assert problem.cost.R.shape == (
        _FAST_CONFIG.horizon,
        _FAST_CONFIG.control_dim,
        _FAST_CONFIG.control_dim,
    )


def test_build_models_returns_all_five_expected_controllers(tmp_path) -> None:
    app = _make_app(tmp_path)
    problem = app.build_problem()
    models = app.build_models(problem)

    assert set(models) == {
        "analytic",
        "neural",
        "unfolded_fixed",
        "unfolded_learned_step_size",
        "unfolded_learned_step_size_and_matrix",
    }
    assert isinstance(models["analytic"], RiccatiController)
    assert isinstance(models["neural"], NeuralPolicy)
    for name in (
        "unfolded_fixed",
        "unfolded_learned_step_size",
        "unfolded_learned_step_size_and_matrix",
    ):
        assert isinstance(models[name], UnfoldedController)

    # (b) and (c) each learn their own step-size parameter -- must not be the
    # same object (shared state would silently couple two "independent"
    # configurations' training).
    assert (
        models["unfolded_learned_step_size"].config.parameters["step_size"]
        is not models["unfolded_learned_step_size_and_matrix"].config.parameters[
            "step_size"
        ]
    )
    # (c) has two learnable parameters (step size + the Riccati-replacement
    # matrix); (b) has exactly one.
    assert set(models["unfolded_learned_step_size"].config.parameters) == {"step_size"}
    assert set(models["unfolded_learned_step_size_and_matrix"].config.parameters) == {
        "step_size",
        "riccati_matrix",
    }


def test_run_executes_all_five_engines_without_crashing_and_returns_finite_metrics(
    tmp_path,
) -> None:
    app = _make_app(tmp_path)
    results = app.run()

    assert set(results) == {
        "analytic",
        "neural",
        "unfolded_fixed",
        "unfolded_learned_step_size",
        "unfolded_learned_step_size_and_matrix",
    }
    for name, metrics in results.items():
        assert metrics, f"{name} returned no metrics"
        for key, value in metrics.items():
            assert math.isfinite(value), f"{name}.{key} = {value} is not finite"

    # Every model's run directory (with metadata.json) was actually created.
    run_dirs = list(tmp_path.glob("run_*"))
    assert len(run_dirs) == 5


def test_unfolded_fixed_step_size_is_never_updated_by_training(tmp_path) -> None:
    """The defining property of the "no learned parameters" configuration:
    its step-size parameter must be bit-for-bit identical before and after
    the engine "runs" it."""
    app = _make_app(tmp_path)
    problem = app.build_problem()
    models = app.build_models(problem)
    step_size_param = models["unfolded_fixed"].config.parameters["step_size"]
    before = step_size_param.get_raw().detach().clone()

    tracker = LocalExperimentTracker(tmp_path, "unfolded_fixed")
    engine = app.build_engine(
        "unfolded_fixed", models["unfolded_fixed"], problem, tracker
    )
    engine.run()

    after = step_size_param.get_raw().detach()
    assert torch.equal(before, after)


def test_unfolded_learned_configurations_actually_update_their_parameters(
    tmp_path,
) -> None:
    """Contrast case for the previous test: (b) and (c) must actually learn
    -- their raw parameters must change after a real training run."""
    app = _make_app(tmp_path)
    problem = app.build_problem()
    models = app.build_models(problem)

    step_size_b = models["unfolded_learned_step_size"].config.parameters["step_size"]
    before_b = step_size_b.get_raw().detach().clone()

    step_size_c = models["unfolded_learned_step_size_and_matrix"].config.parameters[
        "step_size"
    ]
    riccati_matrix_c = models[
        "unfolded_learned_step_size_and_matrix"
    ].config.parameters["riccati_matrix"]
    before_c_step = step_size_c.get_raw().detach().clone()
    before_c_matrix = riccati_matrix_c.get_raw().detach().clone()

    for name in ("unfolded_learned_step_size", "unfolded_learned_step_size_and_matrix"):
        tracker = LocalExperimentTracker(tmp_path, name)
        engine = app.build_engine(name, models[name], problem, tracker)
        engine.run()

    assert not torch.equal(before_b, step_size_b.get_raw().detach())
    assert not torch.equal(before_c_step, step_size_c.get_raw().detach())
    assert not torch.equal(before_c_matrix, riccati_matrix_c.get_raw().detach())


@pytest.mark.parametrize("seed", [0, 1])
def test_build_problem_is_deterministic_given_a_seed(tmp_path, seed) -> None:
    config = StandardLQRConfig(seed=seed, horizon=4, state_dim=2, control_dim=1)
    app_1 = StandardLQRApp(
        functools.partial(LocalExperimentTracker, tmp_path), config=config
    )
    app_2 = StandardLQRApp(
        functools.partial(LocalExperimentTracker, tmp_path), config=config
    )

    problem_1 = app_1.build_problem()
    problem_2 = app_2.build_problem()

    assert torch.allclose(
        torch.tensor(problem_1.system.A_t.array),
        torch.tensor(problem_2.system.A_t.array),
    )
    assert torch.allclose(
        torch.tensor(problem_1.system.B_t.array),
        torch.tensor(problem_2.system.B_t.array),
    )
