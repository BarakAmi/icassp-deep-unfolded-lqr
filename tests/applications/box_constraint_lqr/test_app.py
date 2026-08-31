import functools
import json
import math

import numpy as np
import pytest
import torch

from mbl.applications.box_constraint_lqr import (
    BoxConstraintLQRApp,
    BoxConstraintLQRConfig,
)
from mbl.models.analytic.truncated_riccati import TruncatedRiccatiController
from mbl.models.constrained.cocp import COCPController
from mbl.models.neural.nerual import NeuralPolicy
from mbl.models.unfolded.base import UnfoldedController
from mbl.persistence.local_tracker import LocalExperimentTracker

# Small but non-trivial, and deliberately tight (small u_max relative to the
# noise/dynamics scale) so the box constraint actually binds -- otherwise a
# "constrained" test could pass vacuously. cvxpylayers/SCS solves the COCP
# QP once per (batch element, time step, epoch), so batch/horizon/epochs are
# kept small to keep the suite fast without shrinking state/control dims to
# degenerate 1D.
_FAST_CONFIG = BoxConstraintLQRConfig(
    state_dim=2,
    control_dim=1,
    horizon=3,
    process_noise_std=0.3,
    u_max=0.3,
    batch_size=2,
    seed=0,
    num_unfolding_iterations=2,
    unfolded_learning_rate=0.05,
    unfolded_epochs=1,
    neural_hidden_dim=4,
    neural_learning_rate=1e-3,
    neural_epochs=1,
    cocp_learning_rate=0.05,
    cocp_epochs=1,
)

_ALL_MODEL_NAMES = {
    "truncated_riccati",
    "cocp",
    "cocp_lower_bound",
    "neural",
    "unfolded_fixed",
    "unfolded_learned_step_size",
    "unfolded_learned_step_size_and_matrix",
}


def _make_app(tmp_path) -> BoxConstraintLQRApp:
    tracker_factory = functools.partial(LocalExperimentTracker, tmp_path)
    return BoxConstraintLQRApp(tracker_factory, config=_FAST_CONFIG)


def test_build_problem_attaches_a_single_box_constraint(tmp_path) -> None:
    app = _make_app(tmp_path)
    problem = app.build_problem()

    assert problem.system.dimensions.state_dim == _FAST_CONFIG.state_dim
    assert problem.system.dimensions.control_dim == _FAST_CONFIG.control_dim
    constraints = problem.constraints or []
    assert len(constraints) == 1
    assert constraints[0].u_max == _FAST_CONFIG.u_max


def test_build_models_returns_all_seven_expected_controllers(tmp_path) -> None:
    app = _make_app(tmp_path)
    problem = app.build_problem()
    models = app.build_models(problem)

    assert set(models) == _ALL_MODEL_NAMES
    assert isinstance(models["truncated_riccati"], TruncatedRiccatiController)
    assert isinstance(models["cocp"], COCPController)
    assert isinstance(models["cocp_lower_bound"], COCPController)
    assert isinstance(models["neural"], NeuralPolicy)
    for name in (
        "unfolded_fixed",
        "unfolded_learned_step_size",
        "unfolded_learned_step_size_and_matrix",
    ):
        assert isinstance(models[name], UnfoldedController)

    # cocp and cocp_lower_bound must be independent instances/parameters.
    assert models["cocp"] is not models["cocp_lower_bound"]
    assert models["cocp"].P_sqrt is not models["cocp_lower_bound"].P_sqrt


def test_cocp_lower_bound_parameters_are_frozen(tmp_path) -> None:
    app = _make_app(tmp_path)
    problem = app.build_problem()
    models = app.build_models(problem)

    assert not models["cocp_lower_bound"].P_sqrt.requires_grad
    assert not models["cocp_lower_bound"].q.requires_grad
    assert models["cocp"].P_sqrt.requires_grad
    assert models["cocp"].q.requires_grad
    # The M10 smuggling is dead: the SDP bound is declared recipe provenance
    # (see `COCPLowerBoundRecipe.provenance` and the metadata test below),
    # never an ad-hoc attribute on the live module.
    assert not hasattr(models["cocp_lower_bound"], "lower_bound_value")
    bound = app.case_study.recipe("cocp_lower_bound").provenance(problem)
    assert math.isfinite(bound["sdp_lower_bound_value"])


def test_truncated_riccati_and_unfolded_fixed_policies_respect_the_box_bound(
    tmp_path,
) -> None:
    """A large state should push the unconstrained policies well past
    u_max, so this only passes if clipping/projection is actually applied."""
    app = _make_app(tmp_path)
    problem = app.build_problem()
    models = app.build_models(problem)
    u_max = _FAST_CONFIG.u_max

    riccati_policy = models["truncated_riccati"].get_control_policy()
    x_np = np.full((3, _FAST_CONFIG.state_dim), 50.0)
    for t in range(_FAST_CONFIG.horizon):
        u = riccati_policy(t, x_np)
        assert np.all(np.abs(u) <= u_max + 1e-9)

    unfolded_policy = models["unfolded_fixed"].get_control_policy()
    x_torch = torch.full((3, _FAST_CONFIG.state_dim), 50.0, dtype=_FAST_CONFIG.dtype)
    for t in range(_FAST_CONFIG.horizon):
        u = unfolded_policy(t, x_torch)
        assert torch.all(u.abs() <= u_max + 1e-6)


def test_run_executes_all_seven_engines_without_crashing_and_returns_finite_metrics(
    tmp_path,
) -> None:
    app = _make_app(tmp_path)
    results = app.run()

    assert set(results) == _ALL_MODEL_NAMES
    for name, metrics in results.items():
        assert metrics, f"{name} returned no metrics"
        for key, value in metrics.items():
            assert math.isfinite(value), f"{name}.{key} = {value} is not finite"

    run_dirs = list(tmp_path.glob("run_*"))
    assert len(run_dirs) == len(_ALL_MODEL_NAMES)


def test_cocp_lower_bound_metadata_records_the_theoretical_sdp_bound(tmp_path) -> None:
    app = _make_app(tmp_path)
    app.run()

    run_dir = next(tmp_path.glob("*cocp_lower_bound"))
    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert "sdp_lower_bound_value" in metadata["params"]
    assert math.isfinite(metadata["params"]["sdp_lower_bound_value"])


@pytest.mark.parametrize("seed", [0, 1])
def test_build_problem_is_deterministic_given_a_seed(tmp_path, seed) -> None:
    config = BoxConstraintLQRConfig(
        seed=seed, horizon=4, state_dim=2, control_dim=1, u_max=0.5
    )
    tracker_factory = functools.partial(LocalExperimentTracker, tmp_path)
    app_1 = BoxConstraintLQRApp(tracker_factory, config=config)
    app_2 = BoxConstraintLQRApp(tracker_factory, config=config)

    problem_1 = app_1.build_problem()
    problem_2 = app_2.build_problem()

    assert np.allclose(problem_1.system.A_t.array, problem_2.system.A_t.array)
    assert np.allclose(problem_1.system.B_t.array, problem_2.system.B_t.array)
