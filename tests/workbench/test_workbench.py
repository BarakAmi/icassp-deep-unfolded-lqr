from datetime import datetime

import numpy as np
import pandas as pd
import pytest

import torch

from mbl.viz.adapters.notebook import report_benchmarks
from mbl.viz.adapters.sink import FigureSink
from mbl.viz.plots.benchmarking import MemoryBreakdown
from mbl.workbench import (
    build_benchmark_table,
    DisturbanceRealization,
    GaussianBatchSpec,
    GDSolveSettings,
    NoiseCovariances,
    ProblemDims,
    RICCATI_BENCHMARK_EXPERIMENT_NAME,
    RiccatiBenchmarkSpec,
    RunLocation,
    evaluate_cost_conventions,
    find_existing_run,
    generate_marginally_stable_system,
    load_or_run,
    load_run_artifacts,
    make_gaussian_batch_sampler,
    InferenceBenchmarkSpec,
    measure_analytic_gd_benchmark,
    measure_riccati_synthesis_benchmark,
    measure_synthesized_controller_benchmark,
    run_analytic_controller_rollout,
    run_benchmark_suite,
    setup_stochastic_lqr_experiment,
    summarize_cost_comparison,
)
from mbl.applications.factories import GaussianBatchSpec as EngineGaussianBatchSpec
from mbl.applications.recipes import UnfoldedKind, UnfoldedRecipe, null_harness
from mbl.applications.rollout import RolloutModel
from mbl.core.cost.quadratic_cost import QuadraticCost
from mbl.core.optimal_control_problem import OptimalControlProblem
from mbl.core.runtime import Backend, ComputeContext
from mbl.core.system.linear_system import LinearSystem
from mbl.engine.training_plan import OptimizerSpec, TrainingPlan
from mbl.models.analytic.riccati import (
    RiccatiController,
    RiccatiSynthesizer,
    compute_theoretical_expected_cost,
    finite_horizon_riccati,
)
from mbl.models.iterative import ConstantInitializer, JacobiSweep
from mbl.persistence.local_tracker import LocalExperimentTracker


def _fixed_clock(timestamp: datetime):
    return lambda: timestamp


def test_find_existing_run_matches_by_bare_name(tmp_path) -> None:
    LocalExperimentTracker(
        tmp_path, "standard_lqr_sanity_check", clock=_fixed_clock(datetime(2026, 1, 1))
    )

    run = find_existing_run(tmp_path, "standard_lqr_sanity_check")

    assert run is not None
    assert run.run_id == "run_20260101_000000_standard_lqr_sanity_check"


def test_find_existing_run_returns_none_when_no_run_matches(tmp_path) -> None:
    LocalExperimentTracker(
        tmp_path, "some_other_experiment", clock=_fixed_clock(datetime(2026, 1, 1))
    )

    assert find_existing_run(tmp_path, "standard_lqr_sanity_check") is None


def test_find_existing_run_returns_the_most_recent_match(tmp_path) -> None:
    LocalExperimentTracker(tmp_path, "demo", clock=_fixed_clock(datetime(2026, 1, 1)))
    LocalExperimentTracker(tmp_path, "demo", clock=_fixed_clock(datetime(2026, 6, 1)))

    run = find_existing_run(tmp_path, "demo")

    assert run.run_id == "run_20260601_000000_demo"


def test_find_existing_run_reuses_run_whose_signature_matches(tmp_path) -> None:
    tracker = LocalExperimentTracker(
        tmp_path, "demo", clock=_fixed_clock(datetime(2026, 1, 1))
    )
    tracker.log_params({"signature": "abc123"})

    run = find_existing_run(tmp_path, "demo", expected_signature="abc123")
    assert run is not None
    assert run.run_id == "run_20260101_000000_demo"


def test_find_existing_run_ignores_run_whose_signature_does_not_match(
    tmp_path,
) -> None:
    """Phase 1F: the cache-invalidation fix. A same-named run left over from a
    different problem/controller/sampling configuration (e.g. STATE_DIM
    changed) must not be silently reused just because its name still
    matches."""
    tracker = LocalExperimentTracker(
        tmp_path, "demo", clock=_fixed_clock(datetime(2026, 1, 1))
    )
    tracker.log_params({"signature": "stale-signature"})

    with pytest.warns(UserWarning, match="signature 'stale-signature'"):
        run = find_existing_run(tmp_path, "demo", expected_signature="fresh-signature")
    assert run is None


def test_find_existing_run_skips_signature_mismatch_and_finds_older_match(
    tmp_path,
) -> None:
    """A signature mismatch on the most recent run must not stop the scan --
    an older run under the same name might still match (e.g. after reverting
    a config change back to what it was)."""
    older = LocalExperimentTracker(
        tmp_path, "demo", clock=_fixed_clock(datetime(2026, 1, 1))
    )
    older.log_params({"signature": "matches"})
    newer = LocalExperimentTracker(
        tmp_path, "demo", clock=_fixed_clock(datetime(2026, 6, 1))
    )
    newer.log_params({"signature": "does-not-match"})

    with pytest.warns(UserWarning, match="signature 'does-not-match'"):
        run = find_existing_run(tmp_path, "demo", expected_signature="matches")
    assert run is not None
    assert run.run_id == "run_20260101_000000_demo"


def test_load_or_run_reruns_when_the_cached_signature_does_not_match(
    tmp_path,
) -> None:
    """The crash this whole mechanism exists to prevent: a stale cached run
    (produced under a different configuration) must not be silently reused
    just because artifacts/ is non-empty -- load_or_run must fall through to
    force_run exactly as if no run existed at all."""
    stale = LocalExperimentTracker(
        tmp_path, "demo", clock=_fixed_clock(datetime(2026, 1, 1))
    )
    stale.log_params({"signature": "stale-signature"})
    stale.save_artifact("P_arr", np.zeros((4, 4)))  # e.g. old STATE_DIM=4

    calls = []

    def _run_fn() -> None:
        calls.append(1)
        tracker = LocalExperimentTracker(
            tmp_path, "demo", clock=_fixed_clock(datetime(2026, 6, 1))
        )
        tracker.log_params({"signature": "fresh-signature"})
        tracker.save_artifact("P_arr", np.eye(7))  # e.g. new STATE_DIM=7

    with pytest.warns(UserWarning, match="signature 'stale-signature'"):
        artifacts = load_or_run(
            tmp_path,
            "demo",
            _run_fn,
            force_run=True,
            expected_signature="fresh-signature",
        )

    assert len(calls) == 1
    assert artifacts["P_arr"].shape == (7, 7)


def test_load_or_run_raises_on_signature_mismatch_when_force_run_is_false(
    tmp_path,
) -> None:
    stale = LocalExperimentTracker(
        tmp_path, "demo", clock=_fixed_clock(datetime(2026, 1, 1))
    )
    stale.log_params({"signature": "stale-signature"})
    stale.save_artifact("P_arr", np.eye(4))

    with (
        pytest.warns(UserWarning, match="signature 'stale-signature'"),
        pytest.raises(RuntimeError, match="force_run=True"),
    ):
        load_or_run(
            tmp_path, "demo", lambda: None, expected_signature="fresh-signature"
        )


def test_load_run_artifacts_loads_every_artifact_keyed_by_bare_name(tmp_path) -> None:
    tracker = LocalExperimentTracker(
        tmp_path, "demo", clock=_fixed_clock(datetime(2026, 1, 1))
    )
    tracker.save_artifact("P_arr", np.eye(2))
    tracker.save_artifact("trajectory_states", np.zeros((3, 4, 2)))

    run = find_existing_run(tmp_path, "demo")
    artifacts = load_run_artifacts(run)

    assert set(artifacts) == {"P_arr", "trajectory_states"}
    assert np.array_equal(artifacts["P_arr"], np.eye(2))


def test_load_or_run_returns_existing_artifacts_without_running(tmp_path) -> None:
    tracker = LocalExperimentTracker(
        tmp_path, "demo", clock=_fixed_clock(datetime(2026, 1, 1))
    )
    tracker.save_artifact("P_arr", np.eye(2))

    def _forbidden_run_fn() -> None:
        raise AssertionError("run_fn() must not be called when a run already exists.")

    # force_run defaults to False here -- since a run already exists, this
    # must succeed without ever needing to opt into running anything.
    artifacts = load_or_run(tmp_path, "demo", _forbidden_run_fn)

    assert np.array_equal(artifacts["P_arr"], np.eye(2))


def test_load_or_run_raises_when_no_run_exists_and_force_run_is_false(
    tmp_path,
) -> None:
    def _forbidden_run_fn() -> None:
        raise AssertionError("run_fn() must not be called when force_run=False.")

    with pytest.raises(RuntimeError, match="force_run=True"):
        load_or_run(tmp_path, "demo", _forbidden_run_fn)


def test_load_or_run_calls_run_fn_when_force_run_is_true(tmp_path) -> None:
    calls = []

    def _run_fn() -> None:
        calls.append(1)
        tracker = LocalExperimentTracker(
            tmp_path, "demo", clock=_fixed_clock(datetime(2026, 1, 1))
        )
        tracker.save_artifact("P_arr", np.eye(2))

    artifacts = load_or_run(tmp_path, "demo", _run_fn, force_run=True)

    assert len(calls) == 1
    assert np.array_equal(artifacts["P_arr"], np.eye(2))


def test_load_or_run_raises_when_run_fn_does_not_persist_a_matching_run(
    tmp_path,
) -> None:
    with pytest.raises(RuntimeError, match="no run named"):
        load_or_run(tmp_path, "demo", lambda: None, force_run=True)


def test_load_or_run_reruns_when_the_existing_run_has_no_artifacts(tmp_path) -> None:
    """A run whose metadata.json exists but whose artifacts/ directory is
    empty (e.g. committed to git while the gitignored artifacts/ directory
    was not, as happens on a fresh clone) must be treated as unusable --
    load_or_run should fall through to run_fn (given force_run=True) rather
    than returning {}."""
    LocalExperimentTracker(tmp_path, "demo", clock=_fixed_clock(datetime(2026, 1, 1)))
    calls = []

    def _run_fn() -> None:
        calls.append(1)
        tracker = LocalExperimentTracker(
            tmp_path, "demo", clock=_fixed_clock(datetime(2026, 6, 1))
        )
        tracker.save_artifact("P_arr", np.eye(2))

    artifacts = load_or_run(tmp_path, "demo", _run_fn, force_run=True)

    assert len(calls) == 1
    assert np.array_equal(artifacts["P_arr"], np.eye(2))


def test_load_or_run_never_calls_the_builtin_input(tmp_path, monkeypatch) -> None:
    """load_or_run must never block on stdin -- interactive `input()` would
    deadlock a non-interactive kernel (e.g. `jupyter nbconvert --execute`)."""

    def _forbidden_input(_prompt: str = "") -> str:
        raise AssertionError("load_or_run must never call input().")

    monkeypatch.setattr("builtins.input", _forbidden_input)

    with pytest.raises(RuntimeError, match="force_run=True"):
        load_or_run(tmp_path, "demo", lambda: None)


def test_make_gaussian_batch_sampler_returns_correctly_shaped_zero_mean_batch() -> None:
    state_dim, horizon, batch_size, std = 3, 5, 10_000, 0.5
    bundle = make_gaussian_batch_sampler(
        state_dim,
        horizon,
        GaussianBatchSpec(
            initial_state_std=std, noise_std=std, batch_size=batch_size, seed=0
        ),
    )
    sampler, distributions = bundle.sample, bundle.distributions

    x0, w, v = sampler()

    assert x0.shape == (batch_size, state_dim)
    assert w.shape == (batch_size, horizon, state_dim)
    assert v.shape == (batch_size, horizon, state_dim)
    assert np.array_equal(v, np.zeros_like(v))
    assert abs(x0.std() - std) < 0.02
    assert abs(w.std() - std) < 0.02
    assert set(distributions) == {"initial_state", "process_noise", "measurement_noise"}


def test_make_gaussian_batch_sampler_is_reproducible_given_the_same_seed() -> None:
    def make():
        return make_gaussian_batch_sampler(
            2,
            3,
            GaussianBatchSpec(
                initial_state_std=1.0, noise_std=1.0, batch_size=4, seed=42
            ),
        ).sample

    assert all(np.array_equal(a, b) for a, b in zip(make()(), make()(), strict=True))


def test_make_gaussian_batch_sampler_distributions_match_configured_std() -> None:
    """initial_state_std and noise_std are configured independently -- the
    persisted per-distribution signatures must reflect each one separately."""
    distributions = make_gaussian_batch_sampler(
        2,
        3,
        GaussianBatchSpec(initial_state_std=0.3, noise_std=0.7, batch_size=4, seed=1),
    ).distributions
    assert distributions["initial_state"].get_signature()["std"] == 0.3
    assert distributions["process_noise"].get_signature()["std"] == 0.7
    assert distributions["measurement_noise"].get_signature() == {"type": "Zero"}


def test_make_gaussian_batch_sampler_honors_independent_initial_state_std() -> None:
    """The initial state's empirical spread tracks initial_state_std, not
    noise_std -- proving the two are genuinely independent knobs."""
    sampler = make_gaussian_batch_sampler(
        4,
        6,
        GaussianBatchSpec(
            initial_state_std=2.0, noise_std=0.1, batch_size=20_000, seed=0
        ),
    ).sample
    x0, w, _ = sampler()
    assert abs(x0.std() - 2.0) < 0.05
    assert abs(w.std() - 0.1) < 0.01


def test_run_analytic_controller_rollout_persists_riccati_matrices_and_trajectories(
    tmp_path,
) -> None:
    """End-to-end smoke test at a tiny scale for the exact helper the
    "01_standard_lqr_theoretical_vs_empirical" notebook calls -- run via
    load_or_run, since we can't execute the notebook itself to verify this."""
    n, m, horizon, batch_size = 2, 1, 3, 4
    A, B = generate_marginally_stable_system(n, m, seed=0)
    system = LinearSystem.fully_observable(A, B)
    Q = np.repeat(np.eye(n)[None], horizon + 1, axis=0)
    R = np.repeat(np.eye(m)[None], horizon, axis=0)
    cost = QuadraticCost(Q=Q, R=R, include_terminal_cost=True)
    problem = OptimalControlProblem(system=system, cost=cost)
    controller = RiccatiController(problem, horizon)
    sampling = make_gaussian_batch_sampler(
        n,
        horizon,
        GaussianBatchSpec(
            initial_state_std=0.5, noise_std=0.5, batch_size=batch_size, seed=1
        ),
    )

    artifacts = load_or_run(
        tmp_path,
        "demo",
        lambda: run_analytic_controller_rollout(
            RunLocation(tmp_path, "demo"), controller, sampling=sampling
        ),
        force_run=True,
    )

    assert np.allclose(artifacts["parameter_P_arr"], controller.P_arr)
    assert np.allclose(artifacts["parameter_K_arr"], controller.K_arr)
    assert artifacts["trajectory_states"].shape == (batch_size, horizon + 1, n)
    assert artifacts["trajectory_controls"].shape == (batch_size, horizon, m)

    # Second call must hit the "already exists" path -- proving load_or_run
    # itself never re-runs a simulation it doesn't need to, even without
    # force_run.
    reloaded = load_or_run(
        tmp_path,
        "demo",
        lambda: (_ for _ in ()).throw(AssertionError("must not re-run")),
    )
    assert np.allclose(reloaded["parameter_P_arr"], controller.P_arr)


def test_generate_marginally_stable_system_returns_correctly_shaped_arrays() -> None:
    A, B = generate_marginally_stable_system(4, 2, seed=0)

    assert A.shape == (4, 4)
    assert B.shape == (4, 2)


def test_generate_marginally_stable_system_rescales_a_to_unit_spectral_radius() -> None:
    A, _ = generate_marginally_stable_system(5, 3, seed=7)

    spectral_radius = np.max(np.abs(np.linalg.eigvals(A)))
    assert abs(spectral_radius - 1.0) < 1e-10


def test_generate_marginally_stable_system_is_reproducible_given_the_same_seed() -> (
    None
):
    A1, B1 = generate_marginally_stable_system(3, 2, seed=42)
    A2, B2 = generate_marginally_stable_system(3, 2, seed=42)

    assert np.array_equal(A1, A2)
    assert np.array_equal(B1, B2)


def _build_tiny_lqr_fixture():
    n, m, horizon, batch = 2, 1, 4, 64
    A, B = generate_marginally_stable_system(n, m, seed=3)
    system = LinearSystem.fully_observable(A, B)
    Q = np.repeat(np.eye(n)[None], horizon + 1, axis=0)
    R = np.repeat(np.eye(m)[None], horizon, axis=0)
    cost = QuadraticCost(Q=Q, R=R, include_terminal_cost=True)
    problem = OptimalControlProblem(system=system, cost=cost)
    controller = RiccatiController(problem, horizon)

    rng = np.random.default_rng(11)
    noise_std = 0.5
    x0 = rng.normal(scale=noise_std, size=(batch, n))
    w = rng.normal(scale=noise_std, size=(batch, horizon, n))
    v = np.zeros((batch, horizon, n))
    states, _, controls = system.run(controller.get_control_policy(), x0, w, v)

    return system, Q, R, controller, states, controls, noise_std


def test_evaluate_cost_conventions_returns_all_four_conventions_with_expected_shapes() -> (
    None
):
    system, Q, R, controller, states, controls, noise_std = _build_tiny_lqr_fixture()
    process_noise_cov = noise_std**2 * np.eye(Q.shape[-1])
    initial_state_cov = process_noise_cov

    curves = evaluate_cost_conventions(
        system,
        QuadraticCost(Q=Q, R=R),
        controller.K_arr,
        states,
        controls,
        covariances=NoiseCovariances(
            process_noise=process_noise_cov, initial_state=initial_state_cov
        ),
    )

    assert set(curves) == {
        "Terminal, Time-Averaged",
        "No Terminal, Time-Averaged",
        "Terminal, Raw",
        "No Terminal, Raw",
    }
    horizon = controller.K_arr.shape[0]
    for curve in curves.values():
        assert set(curve) == {"empirical_cost", "theoretical_cost"}
        assert curve["empirical_cost"].shape == (horizon,)
        assert curve["theoretical_cost"].shape == (horizon,)


def test_evaluate_cost_conventions_matches_direct_computation() -> None:
    system, Q, R, controller, states, controls, noise_std = _build_tiny_lqr_fixture()
    process_noise_cov = noise_std**2 * np.eye(Q.shape[-1])
    initial_state_cov = process_noise_cov

    curves = evaluate_cost_conventions(
        system,
        QuadraticCost(Q=Q, R=R),
        controller.K_arr,
        states,
        controls,
        covariances=NoiseCovariances(
            process_noise=process_noise_cov, initial_state=initial_state_cov
        ),
    )

    expected_cost = QuadraticCost(Q=Q, R=R, include_terminal_cost=True)
    expected_empirical = expected_cost(states, controls)
    expected_theoretical = compute_theoretical_expected_cost(
        system, expected_cost, controller.K_arr, process_noise_cov, initial_state_cov
    )
    assert np.allclose(
        curves["Terminal, Time-Averaged"]["empirical_cost"], expected_empirical
    )
    assert np.allclose(
        curves["Terminal, Time-Averaged"]["theoretical_cost"], expected_theoretical
    )


def test_evaluate_cost_conventions_raw_equals_time_averaged_times_elapsed_steps() -> (
    None
):
    """The algebraic identity distinguishing the two new axes (Phase 1F):
    each "Raw" curve must equal its "Time-Averaged" counterpart multiplied by
    the elapsed step count (k+1), for both empirical and theoretical curves,
    under both terminal-cost conventions -- an independent, strong
    correctness cross-check on the whole 4-way grid."""
    system, Q, R, controller, states, controls, noise_std = _build_tiny_lqr_fixture()
    process_noise_cov = noise_std**2 * np.eye(Q.shape[-1])
    initial_state_cov = process_noise_cov

    curves = evaluate_cost_conventions(
        system,
        QuadraticCost(Q=Q, R=R),
        controller.K_arr,
        states,
        controls,
        covariances=NoiseCovariances(
            process_noise=process_noise_cov, initial_state=initial_state_cov
        ),
    )
    horizon = controller.K_arr.shape[0]
    k_steps = np.arange(1, horizon + 1)

    for terminal_label in ("Terminal", "No Terminal"):
        averaged = curves[f"{terminal_label}, Time-Averaged"]
        raw = curves[f"{terminal_label}, Raw"]
        for key in ("empirical_cost", "theoretical_cost"):
            assert np.allclose(raw[key], averaged[key] * k_steps)


def test_setup_stochastic_lqr_experiment_returns_correctly_shaped_artifacts(
    tmp_path,
) -> None:
    n, m, horizon, batch_size = 3, 2, 5, 16
    experiment = setup_stochastic_lqr_experiment(
        ProblemDims(state_dim=n, control_dim=m, horizon=horizon),
        batch_spec=GaussianBatchSpec(
            initial_state_std=0.5, noise_std=0.3, batch_size=batch_size, seed=0
        ),
        location=RunLocation(tmp_path, "demo"),
        dtype=torch.float64,
    )

    assert experiment.x0_batch.shape == (batch_size, n)
    assert experiment.w_batch.shape == (batch_size, horizon, n)
    assert experiment.v_batch.shape == (batch_size, horizon, n)
    assert experiment.x0_t.shape == experiment.x0_batch.shape
    assert experiment.x0_t.dtype == torch.float64
    assert experiment.U_opt.shape == (batch_size, horizon, m)
    assert experiment.X_opt.shape == (batch_size, horizon + 1, n)
    assert isinstance(experiment.J_opt, float)
    assert experiment.figures_dir.is_dir()
    assert experiment.figures_dir.name == "figures"


def test_setup_stochastic_lqr_experiment_baseline_matches_direct_riccati_rollout(
    tmp_path,
) -> None:
    """The returned U_opt/J_opt must be exactly what a direct rollout of the
    Riccati baseline on the same batch would produce -- the factory must not
    silently diverge from the boilerplate it replaces."""
    n, m, horizon, batch_size = 2, 1, 4, 32
    experiment = setup_stochastic_lqr_experiment(
        ProblemDims(state_dim=n, control_dim=m, horizon=horizon),
        batch_spec=GaussianBatchSpec(
            initial_state_std=0.4, noise_std=0.4, batch_size=batch_size, seed=1
        ),
        location=RunLocation(tmp_path, "demo"),
    )

    _, _, expected_U_opt_t = experiment.problem.system.run(
        experiment.riccati.get_control_policy(),
        experiment.x0_t,
        experiment.w_t,
        experiment.v_t,
    )
    expected_J_opt = float(
        RolloutModel(experiment.riccati)(
            experiment.x0_t, experiment.w_t, experiment.v_t
        )[3].mean()
    )

    assert np.allclose(experiment.U_opt, expected_U_opt_t.detach().cpu().numpy())
    assert abs(experiment.J_opt - expected_J_opt) < 1e-12


def test_setup_stochastic_lqr_experiment_is_reproducible_given_the_same_seed(
    tmp_path,
) -> None:
    def make(run_name):
        return setup_stochastic_lqr_experiment(
            ProblemDims(state_dim=2, control_dim=1, horizon=3),
            batch_spec=GaussianBatchSpec(
                initial_state_std=0.5, noise_std=0.5, batch_size=8, seed=7
            ),
            location=RunLocation(tmp_path, run_name),
        )

    a = make("run_a")
    b = make("run_b")

    assert np.array_equal(a.x0_batch, b.x0_batch)
    assert np.array_equal(a.w_batch, b.w_batch)
    assert np.allclose(a.U_opt, b.U_opt)
    assert abs(a.J_opt - b.J_opt) < 1e-12


def test_summarize_cost_comparison_builds_expected_columns_and_values() -> None:
    curves = {
        "With terminal cost": {
            "empirical_cost": np.array([3.0, 2.5, 2.1]),
            "theoretical_cost": np.array([2.9, 2.4, 2.0]),
        },
    }

    table = summarize_cost_comparison(curves)

    assert isinstance(table, pd.DataFrame)
    assert list(table.index) == ["With terminal cost"]
    assert list(table.columns) == [
        "Empirical",
        "Theoretical",
        "Absolute Error",
        "Relative Error (%)",
    ]
    row = table.loc["With terminal cost"]
    assert row["Empirical"] == 2.1
    assert row["Theoretical"] == 2.0
    assert abs(row["Absolute Error"] - 0.1) < 1e-12
    assert abs(row["Relative Error (%)"] - 5.0) < 1e-9


def _small_lqr_instance(n: int, m: int, horizon: int, *, seed: int = 0):
    A, B = generate_marginally_stable_system(n, m, seed=seed)
    system = LinearSystem.fully_observable(A, B)
    Q = np.repeat(np.eye(n)[None], horizon + 1, axis=0)
    R = np.repeat(np.eye(m)[None], horizon, axis=0)
    cost = QuadraticCost(Q=Q, R=R, include_terminal_cost=True)
    return system, cost


def test_measure_riccati_synthesis_benchmark_reports_a_nonzero_numpy_layer(
    tmp_path,
) -> None:
    n, m, horizon, batch_size = 4, 2, 20, 16
    system, cost = _small_lqr_instance(n, m, horizon)
    sampling = make_gaussian_batch_sampler(
        n,
        horizon,
        GaussianBatchSpec(
            initial_state_std=0.5, noise_std=0.5, batch_size=batch_size, seed=1
        ),
    )
    # The recursion is deterministic given (system, cost, horizon), so this
    # independently-solved P_arr/K_arr is byte-identical to whatever the
    # benchmark itself synthesizes -- an exact lower bound to check against.
    expected_arr_mb = (
        sum(a.nbytes for a in finite_horizon_riccati(system, cost, horizon)) / 1e6
    )

    t_offline_s, t_online_s, offline_memory, online_memory = (
        measure_riccati_synthesis_benchmark(
            RunLocation(tmp_path, "riccati_benchmark"),
            OptimalControlProblem(system=system, cost=cost),
            horizon,
            sampling=sampling,
        )
    )

    assert t_offline_s >= 0.0
    assert t_online_s >= 0.0
    assert isinstance(offline_memory, MemoryBreakdown)
    assert isinstance(online_memory, MemoryBreakdown)
    # Offline: the synthesized P_arr/K_arr are registered explicitly, so the
    # NumPy layer must be at least their exact structural footprint.
    assert offline_memory.numpy_mb is not None
    assert offline_memory.numpy_mb >= expected_arr_mb - 1e-9
    # Neither phase does any PyTorch work.
    assert offline_memory.torch_mb == 0.0
    assert online_memory.torch_mb == 0.0


def test_measure_analytic_gd_benchmark_reports_nonzero_numpy_and_torch_layers() -> None:
    n, m, horizon, batch_size, max_iters = 4, 2, 20, 16, 5
    system, cost = _small_lqr_instance(n, m, horizon)
    x0_batch, w_batch, _ = make_gaussian_batch_sampler(
        n,
        horizon,
        GaussianBatchSpec(
            initial_state_std=0.5, noise_std=0.5, batch_size=batch_size, seed=1
        ),
    ).sample()
    control_initializer = ConstantInitializer(
        0.0, m, torch.float32, torch.device("cpu")
    )

    t_offline_s, t_online_s, offline_memory, online_memory = (
        measure_analytic_gd_benchmark(
            OptimalControlProblem(system=system, cost=cost),
            horizon,
            DisturbanceRealization(x0=x0_batch, w=w_batch),
            control_initializer,
            settings=GDSolveSettings(
                alpha=0.02,
                max_iters=max_iters,
                dtype=torch.float32,
                sweep_strategy=JacobiSweep(),
            ),
        )
    )

    assert t_offline_s >= 0.0
    assert t_online_s >= 0.0
    # Offline: the refinement's converted M_stack/C_stack tensors.
    assert offline_memory.torch_mb is not None
    assert offline_memory.torch_mb > 0.0
    # Online: the macro-iteration loop's PyTorch buffers dominate, but the
    # returned OptimizationResult's U_history/J_history (registered
    # explicitly) still give a real, non-zero NumPy layer too.
    assert online_memory.torch_mb is not None and online_memory.numpy_mb is not None
    assert online_memory.torch_mb > 0.0
    assert online_memory.numpy_mb > 0.0


def test_measure_analytic_gd_benchmark_offline_numpy_floor_is_honored() -> None:
    n, m, horizon, batch_size, max_iters = 4, 2, 20, 16, 5
    system, cost = _small_lqr_instance(n, m, horizon)
    x0_batch, w_batch, _ = make_gaussian_batch_sampler(
        n,
        horizon,
        GaussianBatchSpec(
            initial_state_std=0.5, noise_std=0.5, batch_size=batch_size, seed=1
        ),
    ).sample()
    control_initializer = ConstantInitializer(
        0.0, m, torch.float32, torch.device("cpu")
    )
    huge_floor_mb = 10_000.0

    _, _, offline_memory, _ = measure_analytic_gd_benchmark(
        OptimalControlProblem(system=system, cost=cost),
        horizon,
        DisturbanceRealization(x0=x0_batch, w=w_batch),
        control_initializer,
        settings=GDSolveSettings(
            alpha=0.02,
            max_iters=max_iters,
            dtype=torch.float32,
            sweep_strategy=JacobiSweep(),
        ),
        shared_riccati_offline_numpy_mb=huge_floor_mb,
    )

    assert offline_memory.numpy_mb == huge_floor_mb


def test_benchmark_suite_report_returns_stratified_table_and_backed_figure(
    tmp_path,
) -> None:
    n, m, horizon, batch_size, max_iters = 4, 2, 20, 16, 5
    system, cost = _small_lqr_instance(n, m, horizon)
    x0_batch, w_batch, _ = make_gaussian_batch_sampler(
        n,
        horizon,
        GaussianBatchSpec(
            initial_state_std=0.5, noise_std=0.5, batch_size=batch_size, seed=1
        ),
    ).sample()
    control_initializer = ConstantInitializer(
        0.0, m, torch.float32, torch.device("cpu")
    )
    figures_dir = tmp_path / "figures"
    figures_dir.mkdir()

    records = run_benchmark_suite(
        OptimalControlProblem(system=system, cost=cost),
        horizon,
        DisturbanceRealization(x0=x0_batch, w=w_batch),
        control_initializer,
        settings=GDSolveSettings(alpha=0.02, max_iters=max_iters, dtype=torch.float32),
        riccati_benchmark=RiccatiBenchmarkSpec(
            location=RunLocation(
                tmp_path / "experiments", RICCATI_BENCHMARK_EXPERIMENT_NAME
            ),
            batch_spec=GaussianBatchSpec(
                initial_state_std=0.5, noise_std=0.5, batch_size=batch_size, seed=2
            ),
        ),
    )
    table = build_benchmark_table(records)
    reported = report_benchmarks(records, sink=FigureSink(figures_dir, show=False))

    pd.testing.assert_frame_equal(reported, table)

    assert isinstance(table, pd.DataFrame)
    assert list(table.index) == [
        "Riccati",
        f"GD -- Jacobi (M={max_iters}, B={batch_size})",
        f"GD -- Gauss-Seidel (M={max_iters}, B={batch_size})",
    ]
    assert list(table.columns) == [
        "Offline (s)",
        "Online (s)",
        "Offline NumPy/CPU Memory (MB)",
        "Offline PyTorch Memory (MB)",
        "Online NumPy/CPU Memory (MB)",
        "Online PyTorch Memory (MB)",
    ]
    # Physical post-condition (Micro-Prompt 5d): both Riccati and GD carry a
    # comparable, non-zero NumPy/CPU offline footprint (the shared Riccati
    # recursion), while GD alone carries a non-zero online PyTorch footprint.
    assert (table["Offline NumPy/CPU Memory (MB)"] > 0.0).all()
    assert table.loc["Riccati", "Online PyTorch Memory (MB)"] == 0.0
    assert (
        table.loc[
            f"GD -- Jacobi (M={max_iters}, B={batch_size})",
            "Online PyTorch Memory (MB)",
        ]
        > 0.0
    )
    # The "No Figure Unbacked" law (T4.e): the PDF persists WITH its
    # raw-data sidecar.
    assert (figures_dir / "computational_benchmarks.pdf").exists()
    assert (figures_dir / "computational_benchmarks.json").exists()


def test_measure_synthesized_controller_benchmark_riccati_reports_positive_latency_and_memory() -> (
    None
):
    """Generalizes measure_riccati_synthesis_benchmark to the `Synthesizer`
    protocol directly: a closed-form family reports positive offline/online
    time, and its own K_arr/P_arr resident weight gives the online phase a
    non-zero footprint even though u = -Kx allocates almost nothing new."""
    n, m, horizon = 4, 2, 20
    system, cost = _small_lqr_instance(n, m, horizon)
    problem = OptimalControlProblem(system=system, cost=cost)
    ctx = ComputeContext(backend=Backend.TORCH)
    state_batch = torch.zeros(1, n, dtype=ctx.torch_dtype)

    record = measure_synthesized_controller_benchmark(
        RiccatiSynthesizer(horizon),
        problem,
        ctx,
        state_batch,
        spec=InferenceBenchmarkSpec(
            label="Riccati", num_inference_calls=20, inference_warmup_calls=2
        ),
    )

    assert record.label == "Riccati"
    assert record.offline_s >= 0.0
    assert record.online_s >= 0.0
    assert record.online_memory.torch_mb is not None
    assert record.online_memory.torch_mb > 0.0


def test_measure_synthesized_controller_benchmark_unfolded_reports_torch_layer_and_module_memory() -> (
    None
):
    """The same benchmark against a gradient-trained family (via its
    `EngineTrainedSynthesizer`): offline times the full training run, and
    the trained module's own registered parameters give the online phase a
    non-zero PyTorch-layer footprint."""
    n, m, horizon = 3, 2, 6
    system, cost = _small_lqr_instance(n, m, horizon)
    problem = OptimalControlProblem(system=system, cost=cost)
    ctx = ComputeContext(backend=Backend.TORCH)

    recipe = UnfoldedRecipe(
        kind=UnfoldedKind.LEARNED_STEP_SIZE,
        plan=TrainingPlan(optimizer=OptimizerSpec("adam", 0.05), epochs=1),
        num_iterations=2,
        step_size_init=0.05,
        step_size_max=1.0,
        horizon=horizon,
    )
    harness = null_harness(
        EngineGaussianBatchSpec(
            state_dim=n,
            horizon=horizon,
            batch_size=4,
            seed=0,
            process_noise_std=0.1,
            initial_state_std=0.5,
        ),
        ctx,
    )
    state_batch = torch.zeros(1, n, dtype=ctx.torch_dtype)

    record = measure_synthesized_controller_benchmark(
        recipe.build_synthesizer(problem, harness),
        problem,
        ctx,
        state_batch,
        spec=InferenceBenchmarkSpec(
            label="unfolded_alpha", num_inference_calls=20, inference_warmup_calls=2
        ),
    )

    assert record.offline_s >= 0.0
    assert record.online_s >= 0.0
    assert record.online_memory.torch_mb is not None
    assert record.online_memory.torch_mb > 0.0
