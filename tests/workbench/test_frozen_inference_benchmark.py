"""NB03 Phase B (directive 3): `measure_frozen_inference_benchmark` times a
contender's ONLINE latency + resident memory from its FROZEN policy,
reconstructed from cache -- never retraining, never producing NaN. Executed
end to end against the real experiments/cache/recipe layers, never mocked."""

import numpy as np
import torch

from mbl.applications.factories import GaussianBatchSpec, LQRProblemFactory
from mbl.applications.recipes import UnfoldedKind
from mbl.core.runtime import Backend, ComputeContext, Precision
from mbl.engine.training_plan import OptimizerSpec, TrainingPlan
from mbl.experiments import (
    ContenderSpec,
    EvaluationProtocol,
    Experiment,
    run_experiment,
)
from mbl.workbench import (
    InferenceBenchmarkSpec,
    OnlineInferenceRecord,
    measure_frozen_inference_benchmark,
)

HORIZON = 6
CTX = ComputeContext(backend=Backend.TORCH, device="cpu", precision=Precision.FLOAT64)
PROBLEM = LQRProblemFactory(state_dim=3, control_dim=2, horizon=HORIZON, seed=0)
BATCH = GaussianBatchSpec(
    state_dim=3, horizon=HORIZON, batch_size=8, seed=0, process_noise_std=0.3
)

_LEARNED = ContenderSpec(
    family="unfolded",
    config={
        "kind": UnfoldedKind.LEARNED_STEP_SIZE_AND_MATRIX,
        "plan": TrainingPlan(optimizer=OptimizerSpec("adam", 0.05), epochs=4),
        "num_iterations": 3,
        "step_size_init": 0.02,
        "step_size_max": 0.5,
        "horizon": HORIZON,
    },
    label="learned",
)
_RICCATI = ContenderSpec(family="riccati", config={"horizon": HORIZON}, label="riccati")
_FIXED = ContenderSpec(
    family="unfolded_fixed",
    config={
        "num_iterations": 3,
        "step_size_init": 0.05,
        "step_size_max": 0.5,
        "horizon": HORIZON,
    },
    label="fixed",
)


def _run(tmp_path):
    experiment = Experiment(
        name="frozen_benchmark_acceptance",
        problem=PROBLEM,
        contenders=(_RICCATI, _FIXED, _LEARNED),
        evaluation=EvaluationProtocol(batch_spec=BATCH, n_batches=1),
        ctx=CTX,
    )
    return experiment, run_experiment(experiment, root=tmp_path)


def _bench(spec, result):
    state = torch.zeros(1, 3, dtype=torch.float64)
    return measure_frozen_inference_benchmark(
        spec,
        result,
        PROBLEM.build(),
        CTX,
        state,
        spec=InferenceBenchmarkSpec(
            label=spec.label or "", num_inference_calls=32, inference_warmup_calls=4
        ),
    )


def test_produces_finite_online_metrics_for_every_family(tmp_path) -> None:
    experiment, report = _run(tmp_path)
    for spec in experiment.contenders:
        record = _bench(spec, report.results[spec.resolved_label])
        assert isinstance(record, OnlineInferenceRecord)
        assert np.isfinite(record.online_latency_us) and record.online_latency_us > 0
        assert np.isfinite(record.online_numpy_mb)
        assert np.isfinite(record.online_torch_mb)


def test_reconstructs_the_trained_policy_without_retraining(tmp_path) -> None:
    """The learned contender's frozen policy is timed from cache: its resident
    torch footprint is non-zero (the loaded parameters are held), proving a
    real trained module was reconstructed rather than retrained."""
    experiment, report = _run(tmp_path)
    learned_spec = next(
        s for s in experiment.contenders if s.resolved_label == "learned"
    )
    record = _bench(learned_spec, report.results["learned"])
    assert record.online_torch_mb > 0.0


def test_loads_the_trained_parameters_not_the_init_values(tmp_path) -> None:
    """`_load_cached_parameters` must restore the TRAINED step size, distinct
    from the untrained init the freshly-built controller starts at."""
    from mbl.applications.recipes.unfolded import (
        UnfoldedBuildSpec,
        build_unfolded_controller,
    )
    from mbl.workbench.benchmarks import _load_cached_parameters

    _, report = _run(tmp_path)
    result = report.results["learned"]
    controller = build_unfolded_controller(
        PROBLEM.build(),
        CTX,
        UnfoldedBuildSpec(
            kind=UnfoldedKind.LEARNED_STEP_SIZE_AND_MATRIX,
            num_iterations=3,
            step_size_init=0.02,
            step_size_max=0.5,
            horizon=HORIZON,
        ),
    )
    init_alpha = controller.config.parameters["step_size"].get_numpy().copy()
    _load_cached_parameters(controller, result)
    loaded_alpha = controller.config.parameters["step_size"].get_numpy()
    assert not np.allclose(init_alpha, loaded_alpha)
