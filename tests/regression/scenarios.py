"""Stage S0 golden-master scenarios: the frozen, notebook-exact execution
loops whose numerics the regression harness locks down (REFACTOR_PLAN v3,
§7.1 / §8 S0).

Each ``run_*_scenario`` function re-executes, end to end, one of the two
production simulation channels under the EXACT parameters the main notebooks
use today:

* ``run_standard_lqr_scenario`` -- the standard-LQR channel of
  ``notebooks/experiments/01_standard_lqr_theoretical_vs_empirical.ipynb``:
  marginally-stable system draw, backward Riccati recursion (``P_arr``/
  ``K_arr``), one seeded Monte-Carlo rollout on the NumPy path, and the
  empirical + closed-form theoretical cost curves under all four
  ``QuadraticCost`` (terminal, time-averaged) conventions.

* ``run_signal_space_gd_scenario`` / ``run_signal_space_gd_3d_scenario`` --
  the analytical signal-space gradient-descent channel of
  ``notebooks/experiments/02_gradient_descent_lqr_poc.ipynb``: the
  ``setup_stochastic_lqr_experiment`` baseline (Riccati optimum on a fixed
  torch batch, ``J_opt`` via ``RolloutModel``), the Riccati-derived gradient
  coefficients (``M_stack``/``C_stack``), and one
  ``AnalyticalIterativeGDController.solve`` per initializer variant and per
  sweep topology (Jacobi/Gauss-Seidel), per §7.1's "one
  ``OptimizationResult.J_history`` per sweep topology".

The functions are shared verbatim by the capture tool
(``tests.regression.capture_golden``) and the regression tests
(``tests/regression/test_golden_master.py``): capture and verification can
never drift because they execute the same code object.

Large ensembles are reduced deterministically before persistence (leading
``HEAD_SIZE`` members, batch mean, batch mean-square) and additionally
content-hashed in full via ``core.utils.signing.hash_array``, so bit-level
identity of the complete arrays is still asserted without freezing hundreds
of megabytes of raw trajectories into the repository.

Determinism notes:

* All NumPy draws flow through the seeded ``make_gaussian_batch_sampler`` /
  ``generate_marginally_stable_system`` streams, exactly as the notebooks do.
* The ``random start`` initializer variant draws from torch's GLOBAL rng
  (legacy ``RandomSampler`` + ``gaussian_sampler``); the notebook does not
  seed it, so this harness pins ``torch.manual_seed(SEED)`` immediately
  before that solve to make the captured run reproducible.

STRICT ISOLATION: this module only imports from ``src/``; it never modifies
it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from mbl.workbench import (
    generate_marginally_stable_system,
    DisturbanceRealization,
    GaussianBatchSpec,
    GDSolveSettings,
    make_gaussian_batch_sampler,
    solve_signal_space_gd,
)
from mbl.applications.rollout import RolloutModel
from mbl.core.cost.quadratic_cost import QuadraticCost
from mbl.core.optimal_control_problem import OptimalControlProblem
from mbl.core.system.linear_system import LinearSystem
from mbl.core.utils.signing import (
    compute_run_signature,
    compute_signature_digest,
    hash_array,
)
from mbl.models.analytic.riccati import (
    RiccatiController,
    compute_theoretical_expected_cost,
    get_lqr_gradient_matrices,
)
from mbl.models.iterative import (
    ConstantInitializer,
    GaussSeidelSweep,
    SamplerInitializer,
    WarmStartInitializer,
)
from mbl.models.iterative.initializers import ControlInitializer
from mbl.models.iterative.sweeps import SweepStrategy
from mbl.models.samplers import RandomSampler, gaussian_sampler

#: Number of leading ensemble members persisted verbatim from each large
#: batched array (the "head" slice). Small enough to keep fixtures light,
#: large enough that any numeric drift in the rollout dynamics is caught
#: elementwise, not only through the batch statistics and content hashes.
HEAD_SIZE = 4

#: Notebook 01's control panel, verbatim
#: (`01_standard_lqr_theoretical_vs_empirical.ipynb`, parameters cell).
STANDARD_LQR_PARAMS: dict[str, Any] = {
    "state_dim": 4,
    "control_dim": 2,
    "horizon": 100,
    "noise_variance": 0.25,
    "initial_state_variance": 0.25,
    "batch_size": 2**15,
    "seed": 0,
    "include_terminal_cost": True,  # notebook 01: QuadraticCost(..., True)
}

#: Notebook 02's control panel, verbatim -- primary (m=2) instance
#: (`02_gradient_descent_lqr_poc.ipynb`, CONTROL PANEL cell).
SIGNAL_SPACE_GD_PARAMS: dict[str, Any] = {
    "state_dim": 4,
    "control_dim": 2,
    "horizon": 100,
    "seed": 0,
    "dtype": "float32",
    "alpha": 0.02,
    "max_iters": 100,
    "initial_state_std": 0.25,
    "process_noise_std": 0.25,
    "batch_size": 2**10,
}

#: Notebook 02's secondary (m=3) instance, verbatim -- solved with a
#: Gauss-Seidel sweep from a cold start (`run_and_plot_m3_scatter_experiment`).
SIGNAL_SPACE_GD_3D_PARAMS: dict[str, Any] = {
    "state_dim": 3,
    "control_dim": 3,
    "horizon": 18,
    "seed": 0,
    "dtype": "float32",
    "alpha": 0.02,
    "max_iters": 100,
    "initial_state_std": 0.25,
    "process_noise_std": 0.25,
    "batch_size": 64,
}

#: The four (include_terminal_cost, is_time_averaged) QuadraticCost
#: conventions, keyed by fixture-friendly slugs (mirrors
#: `workbench.analysis.COST_CONVENTIONS`).
COST_CONVENTIONS: dict[str, tuple[bool, bool]] = {
    "terminal_avg": (True, True),
    "no_terminal_avg": (False, True),
    "terminal_raw": (True, False),
    "no_terminal_raw": (False, False),
}

_TORCH_DTYPES = {"float32": torch.float32, "float64": torch.float64}


def _batch_reductions(
    name: str, array: "np.ndarray | torch.Tensor"
) -> "dict[str, Any]":
    """Deterministic reductions of one batched ``(B, ...)`` array/tensor:
    head slice, batch mean, and batch mean-square (informative even when the
    ensemble mean is near zero by symmetry)."""
    head = array[:HEAD_SIZE]
    head = head.clone() if isinstance(head, torch.Tensor) else head.copy()
    if isinstance(array, torch.Tensor):
        return {
            f"{name}_head": head,
            f"{name}_mean": array.mean(dim=0),
            f"{name}_meansq": (array**2).mean(dim=0),
        }
    return {
        f"{name}_head": head,
        f"{name}_mean": array.mean(axis=0),
        f"{name}_meansq": (array**2).mean(axis=0),
    }


def _build_identity_cost_problem(
    A: np.ndarray,
    B: np.ndarray,
    horizon: int,
    *,
    include_terminal_cost: bool,
) -> tuple[LinearSystem, QuadraticCost, OptimalControlProblem]:
    """The identity-weighted, time-stacked problem construction both
    notebooks share (Q = I stacked over horizon+1, R = I stacked over
    horizon; `is_time_averaged` left at QuadraticCost's default ``True``)."""
    system = LinearSystem.fully_observable(A, B)
    n, m = A.shape[0], B.shape[1]
    Q = np.repeat(np.eye(n)[None], horizon + 1, axis=0)
    R = np.repeat(np.eye(m)[None], horizon, axis=0)
    cost = QuadraticCost(Q=Q, R=R, include_terminal_cost=include_terminal_cost)
    problem = OptimalControlProblem(system=system, cost=cost)
    return system, cost, problem


def run_standard_lqr_scenario() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Execute the notebook-01 standard-LQR loop and return
    ``(arrays, scalars)``: every array golden (Riccati stacks, rollout
    reductions, cost curves) plus the scalar/digest goldens (full-array
    content hashes, run-signature digest, final costs)."""
    p = STANDARD_LQR_PARAMS
    n, m, horizon = p["state_dim"], p["control_dim"], p["horizon"]
    process_noise_std = float(np.sqrt(p["noise_variance"]))
    initial_state_std = float(np.sqrt(p["initial_state_variance"]))

    A, B = generate_marginally_stable_system(n, m, seed=p["seed"])
    system, cost, problem = _build_identity_cost_problem(
        A, B, horizon, include_terminal_cost=p["include_terminal_cost"]
    )
    controller = RiccatiController(problem, horizon)

    sampling = make_gaussian_batch_sampler(
        n,
        horizon,
        GaussianBatchSpec(
            initial_state_std=initial_state_std,
            noise_std=process_noise_std,
            batch_size=p["batch_size"],
            seed=p["seed"],
        ),
    )
    distributions = sampling.distributions
    x0, w, v = sampling.sample()
    X, _, U = system.run(controller.get_control_policy(), x0, w, v)

    arrays: dict[str, np.ndarray] = {
        "A": A,
        "B": B,
        "P_arr": controller.P_arr,
        "K_arr": controller.K_arr,
        **_batch_reductions("x0", x0),
        **_batch_reductions("w", w),
        **_batch_reductions("X", X),
        **_batch_reductions("U", U),
    }

    process_noise_cov = p["noise_variance"] * np.eye(n)
    initial_state_cov = p["initial_state_variance"] * np.eye(n)
    for slug, (include_terminal, time_averaged) in COST_CONVENTIONS.items():
        convention_cost = QuadraticCost(
            Q=cost.Q,
            R=cost.R,
            include_terminal_cost=include_terminal,
            is_time_averaged=time_averaged,
        )
        arrays[f"empirical_cost__{slug}"] = convention_cost(X, U)
        arrays[f"theoretical_cost__{slug}"] = compute_theoretical_expected_cost(
            system,
            convention_cost,
            controller.K_arr,
            process_noise_cov,
            initial_state_cov,
        )

    scalars: dict[str, Any] = {
        "run_signature_digest": compute_run_signature(
            problem, controller, distributions
        ),
        "controller_signature_digest": compute_signature_digest(
            controller.get_signature()
        ),
        "X_full_hash": hash_array(X),
        "U_full_hash": hash_array(U),
        "x0_full_hash": hash_array(x0),
        "w_full_hash": hash_array(w),
        "J_final__terminal_avg": float(arrays["empirical_cost__terminal_avg"][-1]),
        "J_final__no_terminal_avg": float(
            arrays["empirical_cost__no_terminal_avg"][-1]
        ),
    }
    return arrays, scalars


#: The initializer/topology variants notebook 02 sweeps, in its own order:
#: Section 5's three initializers under the default Jacobi topology, then
#: Section 6's Gauss-Seidel topology from the cold start. Values are
#: ``(initializer_factory, sweep_strategy_factory | None, needs_torch_seed)``.
GD_VARIANTS: dict[str, tuple[Any, Any, bool]] = {
    "cold_jacobi": (
        lambda m, dtype: ConstantInitializer(0.0, m, dtype, torch.device("cpu")),
        None,
        False,
    ),
    "random_jacobi": (
        lambda m, dtype: _make_random_start(m, dtype),
        None,
        True,  # legacy RandomSampler draws from torch's global rng
    ),
    "warm_jacobi": (
        lambda m, dtype: WarmStartInitializer(
            ConstantInitializer(0.0, m, dtype, torch.device("cpu"))
        ),
        None,
        False,
    ),
    "cold_gauss_seidel": (
        lambda m, dtype: ConstantInitializer(0.0, m, dtype, torch.device("cpu")),
        GaussSeidelSweep,
        False,
    ),
}


def _make_random_start(control_dim: int, dtype: torch.dtype) -> ControlInitializer:
    """Notebook 02's ``random_start`` construction, verbatim: the legacy
    ``RandomSampler`` + ``gaussian_sampler`` pair with covariance
    ``INITIAL_STATE_STD * I`` (the notebook passes the std, not its square,
    as ``W`` -- frozen here exactly as written, quirk included)."""
    device = torch.device("cpu")
    sampler = RandomSampler("u0", gaussian_sampler)
    sampler.unify_sampler(
        device=device,
        dtype=dtype,
        W=SIGNAL_SPACE_GD_PARAMS["initial_state_std"]
        * torch.eye(control_dim, dtype=dtype, device=device),
    )
    return SamplerInitializer(sampler, control_dim, dtype, device)


def _gd_setup(
    params: dict[str, Any],
) -> dict[str, Any]:
    """The `setup_stochastic_lqr_experiment` boilerplate, replicated without
    its `LocalExperimentTracker` side effect (a golden capture must not mint
    run directories): same system/cost construction, same seeded batch draw,
    same Riccati baseline rollout and `RolloutModel`-scored ``J_opt``."""
    n, m, horizon = params["state_dim"], params["control_dim"], params["horizon"]
    dtype = _TORCH_DTYPES[params["dtype"]]

    A, B = generate_marginally_stable_system(n, m, seed=params["seed"])
    # setup_stochastic_lqr_experiment uses QuadraticCost's defaults:
    # no terminal term, time-averaged.
    system, cost, problem = _build_identity_cost_problem(
        A, B, horizon, include_terminal_cost=False
    )

    x0_batch, w_batch, v_batch = make_gaussian_batch_sampler(
        n,
        horizon,
        GaussianBatchSpec(
            initial_state_std=params["initial_state_std"],
            noise_std=params["process_noise_std"],
            batch_size=params["batch_size"],
            seed=params["seed"],
        ),
    ).sample()
    x0_t = torch.as_tensor(x0_batch, dtype=dtype)
    w_t = torch.as_tensor(w_batch, dtype=dtype)
    v_t = torch.as_tensor(v_batch, dtype=dtype)

    riccati = RiccatiController(problem, horizon)
    X_opt, _, U_opt_t = system.run(riccati.get_control_policy(), x0_t, w_t, v_t)
    # NOTE (C1-C4 exception ledger, C2 -- RESOLVED in S2): J_opt rides
    # RolloutModel's _differentiable_cost, which now honors the cost's
    # declared flags through the dual-backend kernel's BATCH_MEAN reduction.
    # For THIS problem's default flags (False, True) that reduction executes
    # the exact pre-S2 operation sequence (mean-over-batch-and-time stage
    # costs), so this golden value survived the S2 kernel fix unchanged --
    # by construction now, not by coincidence. See
    # test_known_defect_sentinels.py::TestC2RolloutModelHonorsCostFlags.
    J_opt = float(RolloutModel(riccati)(x0_t, w_t, v_t)[3].mean())

    # The Riccati-derived gradient coefficients build_riccati_gd_refinement
    # feeds StepSizeRefinement: M = 2(R + B^T P B), C = 2(B^T P A), cast to
    # the solve dtype -- the analytical landscape components of this channel.
    M_stack, C_stack = get_lqr_gradient_matrices(
        riccati.P_arr, system.A_t.array, system.B_t.array, cost.R
    )
    return {
        "system": system,
        "cost": cost,
        "problem": problem,
        "riccati": riccati,
        "x0_batch": x0_batch,
        "w_batch": w_batch,
        "x0_t": x0_t,
        "w_t": w_t,
        "v_t": v_t,
        "X_opt": X_opt,
        "U_opt_t": U_opt_t,
        "J_opt": J_opt,
        "M_stack_t": torch.as_tensor(2 * M_stack, dtype=dtype),
        "C_stack_t": torch.as_tensor(2 * C_stack, dtype=dtype),
        "A": A,
        "B": B,
    }


def _solve_variant(
    setup: dict[str, Any],
    params: dict[str, Any],
    initializer_factory: Any,
    sweep_factory: Any,
    needs_torch_seed: bool,
) -> Any:
    """One `solve_signal_space_gd` call under a named variant, with the
    torch global rng pinned first when the variant's initializer draws from
    it (see module docstring's determinism notes)."""
    m, horizon = params["control_dim"], params["horizon"]
    dtype = _TORCH_DTYPES[params["dtype"]]
    if needs_torch_seed:
        torch.manual_seed(params["seed"])
    sweep: SweepStrategy | None = sweep_factory() if sweep_factory else None
    return solve_signal_space_gd(
        setup["problem"],
        horizon,
        DisturbanceRealization(x0=setup["x0_batch"], w=setup["w_batch"]),
        initializer_factory(m, dtype),
        settings=GDSolveSettings(
            alpha=params["alpha"],
            max_iters=params["max_iters"],
            dtype=dtype,
            sweep_strategy=sweep,
        ),
    )


def run_signal_space_gd_scenario() -> tuple[
    dict[str, np.ndarray], dict[str, torch.Tensor], dict[str, Any]
]:
    """Execute the notebook-02 primary-instance loop and return
    ``(numpy_arrays, torch_tensors, scalars)``. Torch-native goldens (the
    Riccati-baseline rollout tensors and the gradient-coefficient stacks) are
    returned separately so the capture tool can persist them via safetensors
    (REFACTOR_PLAN v3 §5 T3.i: never pickle, never ``torch.save``)."""
    params = SIGNAL_SPACE_GD_PARAMS
    setup = _gd_setup(params)

    U_opt = setup["U_opt_t"].detach().cpu().numpy()
    arrays: dict[str, np.ndarray] = {
        "A": setup["A"],
        "B": setup["B"],
        "P_arr": setup["riccati"].P_arr,
        "K_arr": setup["riccati"].K_arr,
        "x0_batch_head": setup["x0_batch"][:HEAD_SIZE].copy(),
        **_batch_reductions("U_opt", U_opt),
    }
    tensors: dict[str, torch.Tensor] = {
        "M_stack": setup["M_stack_t"].contiguous(),
        "C_stack": setup["C_stack_t"].contiguous(),
        **{
            k: v.contiguous()
            for k, v in _batch_reductions("X_opt", setup["X_opt"]).items()
        },
        **{
            k: v.contiguous()
            for k, v in _batch_reductions("U_opt", setup["U_opt_t"]).items()
        },
    }
    scalars: dict[str, Any] = {
        "J_opt": setup["J_opt"],
        "x0_full_hash": hash_array(setup["x0_batch"]),
        "w_full_hash": hash_array(setup["w_batch"]),
        "X_opt_full_hash": hash_array(setup["X_opt"]),
        "U_opt_full_hash": hash_array(setup["U_opt_t"]),
    }

    for name, (init_factory, sweep_factory, needs_seed) in GD_VARIANTS.items():
        result = _solve_variant(setup, params, init_factory, sweep_factory, needs_seed)
        arrays[f"{name}__J_history"] = result.J_history
        arrays[f"{name}__U_history_member0"] = result.U_history[:, 0].copy()
        arrays.update(_batch_reductions(f"{name}__U_final", result.U_final))
        arrays.update(_batch_reductions(f"{name}__X_final", result.X_final))
        scalars[f"{name}__J_final"] = result.J_final
        scalars[f"{name}__iterations_run"] = result.iterations_run
        scalars[f"{name}__converged"] = result.converged
        scalars[f"{name}__U_final_full_hash"] = hash_array(result.U_final)
        scalars[f"{name}__X_final_full_hash"] = hash_array(result.X_final)
        scalars[f"{name}__U_history_full_hash"] = hash_array(result.U_history)
        scalars[f"{name}__signature_digest"] = compute_signature_digest(
            result.signature
        )
    return arrays, tensors, scalars


def run_signal_space_gd_3d_scenario() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Execute notebook 02's secondary (m=3) instance: the exact
    `run_and_plot_m3_scatter_experiment` solve -- Gauss-Seidel sweep from a
    cold start on the (n=3, m=3, T=18) system. Small enough that the full
    ``U_final``/``X_final`` ensembles are frozen verbatim."""
    params = SIGNAL_SPACE_GD_3D_PARAMS
    setup = _gd_setup(params)

    result = _solve_variant(
        setup,
        params,
        lambda m, dtype: ConstantInitializer(0.0, m, dtype, torch.device("cpu")),
        GaussSeidelSweep,
        False,
    )

    arrays: dict[str, np.ndarray] = {
        "A": setup["A"],
        "B": setup["B"],
        "P_arr": setup["riccati"].P_arr,
        "K_arr": setup["riccati"].K_arr,
        "x0_batch": setup["x0_batch"],
        "w_batch": setup["w_batch"],
        "J_history": result.J_history,
        "U_final": result.U_final,
        "X_final": result.X_final,
        "U_history_member0": result.U_history[:, 0].copy(),
    }
    scalars: dict[str, Any] = {
        "J_opt": setup["J_opt"],
        "J_final": result.J_final,
        "iterations_run": result.iterations_run,
        "converged": result.converged,
        "U_history_full_hash": hash_array(result.U_history),
        "signature_digest": compute_signature_digest(result.signature),
    }
    return arrays, scalars
