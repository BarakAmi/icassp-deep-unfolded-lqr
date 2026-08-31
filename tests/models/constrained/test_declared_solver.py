"""Acceptance tests for the declared solver backend — Stage 2 Phase G-4, D17.

Written before the implementation. The defect: `resolve_cocp_solver` runs inside
`build_controller`, picks a backend by probing *this* machine, and its verdict is
discarded — so which solver produced a model's gradients is an unrecorded
property of whoever ran the job. The failure is on record: `cocp` resolved to
DIFFCP while `cocp_lower_bound` resolved to MOREAU in one study, because the two
canary at different query points.

D17's own wording contained both a design and its opposite — "resolved at
specification time" against "a probe may *validate* a declared choice and refuse
it, but may never silently substitute another". The ratified reading is the
second: **nothing resolves.** A resolution is a probe of the executing machine
wherever it is placed, so putting it at document load would only move the
machine-dependence into the identifier.

The checkpoint is therefore **negative control, three ways**, because there are
three distinct ways a declaration can be betrayed and closing one of them looks
like closing all three:

* the declaration must reach identity, or two models built by different solvers
  collide;
* a backend that does not exist must be refused at **parse** time, where the
  author is looking at the document;
* a backend this machine cannot honour must be refused at **build** time and
  never substituted — and that one is verified against the real failure mode
  rather than a mock, since MOREAU-on-CUDA is importable and available here and
  its backward pass is measurably wrong (‖Δ∂q‖ ≈ 2.2e-2 against DIFFCP, versus
  1.7e-8 on CPU).

Plus a structural test, for the same reason `derive_model_id`'s absent
evaluation parameter is stronger than any behavioural check of it: a recipe that
never mentions a resolver cannot silently start resolving again.
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pytest
import torch
from scipy.linalg import solve_discrete_are, sqrtm

from mbl.applications.factories import LQRProblemFactory
from mbl.applications.recipes import cocp as cocp_recipes
from mbl.applications.recipes.cocp import COCPLowerBoundRecipe, COCPRecipe
from mbl.core.runtime import Backend, ComputeContext, Precision
from mbl.engine.training_plan import OptimizerSpec, TrainingPlan
from mbl.models.constrained import solver_resolution
from mbl.models.constrained.solver_resolution import (
    DEFAULT_COCP_BACKEND,
    REFERENCE_COCP_BACKEND,
    COCPCanaryInstance,
    SolverValidationError,
    _probe_moreau,
    exact_solution_and_gradient,
    require_validated_cocp_solver,
    run_solver_canary,
)
from mbl.models.constrained.solver_spec import SUPPORTED_SOLVERS, COCPSolverSpec

N, M, HORIZON, U_MAX = 4, 2, 10, 0.5


def _problem() -> Any:
    return LQRProblemFactory(
        state_dim=N, control_dim=M, horizon=HORIZON, seed=0, u_max=U_MAX
    ).build()


def _ctx() -> ComputeContext:
    return ComputeContext(
        backend=Backend.TORCH, device="cpu", precision=Precision.FLOAT64
    )


def _plan() -> TrainingPlan:
    return TrainingPlan(optimizer=OptimizerSpec("adam", 1e-2), epochs=1)


def _instance() -> COCPCanaryInstance:
    """The query point `COCPRecipe.build_controller` itself validates at."""
    problem = _problem()
    A = problem.system.A_t.array
    B = problem.system.B_t.array
    R = np.asarray(problem.cost.R)[0]
    P = np.real(sqrtm(solve_discrete_are(A, B, np.asarray(problem.cost.Q)[0], R)))
    return COCPCanaryInstance(
        A=A, B=B, R=R, u_max=U_MAX, x=np.ones(N), P_sqrt=P, q=np.zeros(N)
    )


# -- the declaration reaches identity ---------------------------------------


@pytest.mark.parametrize(
    ("recipe_cls", "config"),
    [
        (COCPRecipe, {"plan": _plan()}),
        (COCPLowerBoundRecipe, {"process_noise_std": 0.5}),
    ],
    ids=["cocp", "cocp_lower_bound"],
)
def test_two_backends_are_two_models(recipe_cls: type, config: dict[str, Any]) -> None:
    """**The gate D17 states**, and the one thing today's code cannot do at all:
    the backend was never in the recipe's signature, so two models whose
    gradients came from different solvers were one model."""
    moreau = recipe_cls(**config, solver_backend="MOREAU").get_signature()
    diffcp = recipe_cls(**config, solver_backend="DIFFCP").get_signature()
    assert moreau["solver_backend"] == "MOREAU"
    assert diffcp["solver_backend"] == "DIFFCP"
    assert moreau != diffcp


@pytest.mark.parametrize(
    ("recipe_cls", "config"),
    [
        (COCPRecipe, {"plan": _plan()}),
        (COCPLowerBoundRecipe, {"process_noise_std": 0.5}),
    ],
    ids=["cocp", "cocp_lower_bound"],
)
def test_the_default_is_the_measured_one(
    recipe_cls: type, config: dict[str, Any]
) -> None:
    """§5 rule 5 demands a *measured* default, not a convenient one. The
    measurement is NB04 §1.0.6: MOREAU on CPU reproduces DIFFCP's forward
    solution and gradients to ~1e-7 and is ~64x faster; CUDA is never the
    default because its backward pass is measurably wrong."""
    recipe = recipe_cls(**config)
    assert recipe.solver_backend == DEFAULT_COCP_BACKEND == "MOREAU"
    assert recipe.solver_device == "cpu"


# -- refusal one: a backend that does not exist, at parse time --------------


@pytest.mark.parametrize(
    ("recipe_cls", "config"),
    [
        (COCPRecipe, {"plan": _plan()}),
        (COCPLowerBoundRecipe, {"process_noise_std": 0.5}),
    ],
    ids=["cocp", "cocp_lower_bound"],
)
def test_an_unknown_backend_is_refused_at_construction(
    recipe_cls: type, config: dict[str, Any]
) -> None:
    """Refused where the recipe is *built*, not where the QP is solved.

    The loader resolves every contender while the document is open, so this is
    what makes a mistyped backend a parse-time error naming `contenders[i]`
    rather than a `ValueError` an hour into a run. Both registered families
    check it, because a rule that applies twice is verified twice.
    """
    with pytest.raises(ValueError, match="GUROBI"):
        recipe_cls(**config, solver_backend="GUROBI")


def test_the_accepted_backends_are_the_supported_ones() -> None:
    """A by-name contract with `COCPSolverSpec`: the recipe must accept exactly
    what the spec supports, or a backend could be declarable and unbuildable
    (or buildable and undeclarable)."""
    for name in sorted(SUPPORTED_SOLVERS):
        assert COCPRecipe(plan=_plan(), solver_backend=name).solver_backend == name


# -- refusal two: a backend this machine cannot honour, at build time -------


def _cuda_is_declarable() -> bool:
    """Whether `solver_device="cuda"` can even be attempted here.

    **Both halves are load-bearing, and finding that out cost a real failure.**
    `_probe_moreau` asks MOREAU whether it was built with CUDA support, which it
    answers `True` for on a machine with no GPU at all -- so the tests below
    sailed past that guard and died inside `torch.tensor(..., device="cuda")`.
    `torch.cuda.is_available()` is what actually decides whether a tensor can be
    placed there. Verified by running this suite with `CUDA_VISIBLE_DEVICES=""`,
    which is the nearest thing to a CI runner this machine can offer: the sixth
    instance of environment-dependent success in this project, and the local
    suite could not see it because this machine has the device.
    """
    return torch.cuda.is_available() and _probe_moreau("cuda") is None


def test_the_reference_backend_is_graded_like_everything_else() -> None:
    """**The privilege is gone** (Annex 01 §5 rule 5, corrected 2026-08-18).

    This used to assert that declaring the reference returned `None` -- nothing
    to compare it against, so nothing to check. Being exempt is precisely what
    let it refuse correct models on the strength of its own error: at n = 50 it
    returns a control 1.4e-01 outside the box while the backend it was refusing
    sat 2.96e-13 from the truth. Validation is now against the problem, so the
    reference is graded on the same footing, and on a small plant it passes on
    its merits rather than by exemption.
    """
    verdict = require_validated_cocp_solver(
        _instance(), COCPSolverSpec(name=REFERENCE_COCP_BACKEND, device="cpu")
    )
    assert verdict.accepted
    # Nothing to compare the reference against but the problem itself.
    assert verdict.reference_deviation is None


def test_the_measured_default_validates_on_this_machine() -> None:
    """Anti-vacuity for the refusal below: if MOREAU-CPU did not validate here,
    every refusal test would pass for the wrong reason and the shipped default
    would be unusable."""
    verdict = require_validated_cocp_solver(
        _instance(), COCPSolverSpec(name="MOREAU", device="cpu")
    )
    assert verdict.accepted


@pytest.mark.parametrize(
    "unavailable",
    ["import failed: No module named 'moreau'", "device 'cuda' unavailable"],
    ids=["not installed", "device missing"],
)
def test_an_unavailable_backend_is_refused_before_it_is_solved_with(
    monkeypatch: pytest.MonkeyPatch, unavailable: str
) -> None:
    """The availability half of the refusal, and it has to be faked here.

    `moreau` is a hard dependency of this project and both its CPU and CUDA
    devices are available on this machine, so the branch cannot be reached by
    behaviour — the same shape as the golden-master containment that could not
    be tested on its own capture host. The probe is therefore stubbed, which is
    honest about what is under test: not that `moreau` can be missing, but that
    a declaration is **refused with a message naming it** when the probe says
    so, rather than dying as an `ImportError` six frames inside a QP solve.

    The message is asserted, not just the exception type: without the guard the
    run also fails, and a test that accepted any failure would call that a pass.
    """
    monkeypatch.setattr(solver_resolution, "_probe_moreau", lambda device: unavailable)
    with pytest.raises(SolverValidationError) as error:
        require_validated_cocp_solver(
            _instance(), COCPSolverSpec(name="MOREAU", device="cpu")
        )
    message = str(error.value)
    assert "solver_backend='MOREAU'" in message
    assert unavailable in message
    assert REFERENCE_COCP_BACKEND in message, (
        "the message must name the backend that would run here; an author who "
        "cannot act on a refusal routes around it"
    )


def test_a_declaration_this_machine_cannot_honour_raises() -> None:
    """**The real refusal, against the real failure mode.**

    MOREAU-on-CUDA is importable and available on this machine, its forward
    solution is fine, and its *backward* pass disagrees with DIFFCP by ~2e-2
    where CPU disagrees by ~2e-8 -- the seed-dependent ~50 %-relative error the
    canary was built for. Today that silently becomes a DIFFCP run under an
    identifier claiming nothing in particular. It must now raise, and the
    message must name the declaration so the author knows what to change.

    Both skips are guards on the *environment*, not on the property. Where the
    device is absent the refusal is still verified, by the availability branch
    above -- so no machine leaves this rule unchecked, it is only checked
    through the arm that applies. Running the canary before that guard would
    not skip on a CUDA-less runner, it would die inside `torch.tensor(...,
    device="cuda")`: the sixth instance of environment-dependent success in
    this project, and the one the local suite could not see because this
    machine has the device.
    """
    if not _cuda_is_declarable():
        pytest.skip("no usable CUDA device; the availability branch covers it")
    if run_solver_canary(
        _instance(),
        COCPSolverSpec(name="MOREAU", device="cuda"),
        COCPSolverSpec(name=REFERENCE_COCP_BACKEND, device="cpu"),
    ).passed:
        pytest.skip(
            "MOREAU-CUDA passes the canary on this machine, so the refusal has "
            "no real failure mode to be verified against here"
        )
    with pytest.raises(SolverValidationError, match="MOREAU"):
        require_validated_cocp_solver(
            _instance(), COCPSolverSpec(name="MOREAU", device="cuda")
        )


def test_a_refused_declaration_is_never_substituted() -> None:
    """The half of "refuse, do not substitute" that a raising test cannot see:
    building the controller must fail rather than come back holding a solver
    the specification does not name."""
    if not _cuda_is_declarable():
        pytest.skip("no usable CUDA device to declare")
    recipe = COCPRecipe(plan=_plan(), solver_backend="MOREAU", solver_device="cuda")
    with pytest.raises(SolverValidationError):
        recipe.build_controller(_problem(), _ctx())


# -- the declaration is what the controller gets ----------------------------


@pytest.mark.parametrize("backend", sorted(SUPPORTED_SOLVERS))
def test_the_controller_runs_the_backend_that_was_declared(backend: str) -> None:
    """End-to-end closure over **both** backends: what the document says is what
    the QP layer is built with. A single-backend test would hold under an
    implementation that ignored the field entirely."""
    controller = COCPRecipe(plan=_plan(), solver_backend=backend).build_controller(
        _problem(), _ctx()
    )
    assert controller.config.solver.name == backend


@pytest.mark.parametrize("backend", sorted(SUPPORTED_SOLVERS))
def test_both_families_declare_the_same_way(backend: str) -> None:
    """`cocp` and `cocp_lower_bound` canary at *different* query points -- one
    seeded from the DARE solution, one from the SDP -- which is precisely how
    they came to resolve to different solvers in one study. Under a declaration
    the query point cannot decide anything."""
    bound = COCPLowerBoundRecipe(
        process_noise_std=0.5, solver_backend=backend
    ).build_controller(_problem(), _ctx())
    assert bound.config.solver.name == backend


# -- structural: resolution is a diagnostic, not a dispatch -----------------


def test_no_recipe_can_reach_the_resolver() -> None:
    """The structural half, and the only one that covers the defect *returning*.

    A behavioural test cannot observe a call that was never made: an
    implementation that declared the backend and then quietly re-resolved it
    would satisfy every assertion above on a machine where the declaration and
    the resolution happen to agree -- which is this machine.
    """
    source = inspect.getsource(cocp_recipes)
    for banned in ("resolve_cocp_solver", "ResolvedCOCPSolver"):
        assert banned not in source, (
            f"{banned} is reachable from a recipe again; selection by probing "
            "belongs to the workbench diagnostic, which is what an author runs "
            "BEFORE declaring a backend"
        )


def test_the_diagnostic_still_exists_for_an_author_to_run() -> None:
    """Anti-vacuity for the ban: resolution was not deleted, it was demoted.
    Deleting it would leave an author with no way to measure which backend this
    machine can honour before declaring one."""
    from mbl.workbench.analysis import describe_cocp_solver_choice

    assert "device" in inspect.signature(describe_cocp_solver_choice).parameters


def test_the_validator_cannot_select_anything() -> None:
    """Structural, and the counterpart to the ban above: `require_validated_
    cocp_solver` takes the spec to check and returns a verdict, never a spec.
    A function that could return a *different* spec is a resolver whatever it
    is called."""
    signature = inspect.signature(require_validated_cocp_solver)
    assert list(signature.parameters) == ["instance", "spec"]
    assert COCPSolverSpec.__name__ not in str(signature.return_annotation)


# -- the gate validates against the problem, not against an implementation ----


def test_a_wrong_answer_is_refused_however_it_was_produced() -> None:
    """Anti-vacuity for the whole certificate arm.

    `certify_declared_solver` grades what a backend returned, so the gate is
    only worth having if a deliberately wrong point fails it. Nudged by 1e-2 --
    two decades above the tolerance and two below the failure the gate exists
    for, so it tests the mechanism rather than the calibration.
    """
    from mbl.models.constrained.box_qp import certify_box_qp
    from mbl.models.constrained.solver_resolution import (
        PRIMAL_TOLERANCE,
        reduce_to_box_qp,
    )

    instance = _instance()
    H, c = reduce_to_box_qp(instance)
    Ht = torch.tensor(H)
    ct = torch.tensor(c).unsqueeze(0)
    u_exact, _, _ = exact_solution_and_gradient(instance)

    honest = certify_box_qp(
        Ht,
        ct,
        torch.tensor(u_exact).unsqueeze(0),
        instance.u_max,
        bound_tol=PRIMAL_TOLERANCE,
    )
    assert honest.satisfies(PRIMAL_TOLERANCE)

    wrong = u_exact.copy()
    wrong[0] -= 1e-2
    assert not certify_box_qp(
        Ht,
        ct,
        torch.tensor(wrong).unsqueeze(0),
        instance.u_max,
        bound_tol=PRIMAL_TOLERANCE,
    ).satisfies(PRIMAL_TOLERANCE)


def test_the_reference_deviation_is_recorded_and_cannot_refuse() -> None:
    """Provenance is documented; it is not a criterion.

    The verdict carries how far the established solver sits from the certified
    point -- that is what records which answer is ours and where the two stop
    agreeing. `accepted` must not depend on it, which is asserted structurally:
    the acceptance expression is read from the source and must not mention the
    field at all. A behavioural test cannot see a term that simply is not there.
    """
    import ast
    import inspect

    from mbl.models.constrained import solver_resolution

    verdict = require_validated_cocp_solver(
        _instance(), COCPSolverSpec(name="MOREAU", device="cpu")
    )
    assert verdict.reference_deviation is not None
    assert "diagnostic" in verdict.detail

    source = inspect.getsource(solver_resolution.certify_declared_solver)
    tree = ast.parse(inspect.cleandoc(source))
    accepted = next(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "accepted"
            for target in node.targets
        )
    )
    mentioned = {n.id for n in ast.walk(accepted) if isinstance(n, ast.Name)}
    assert "reference_deviation" not in mentioned


def test_the_two_thresholds_separate_what_they_were_calibrated_on() -> None:
    """The constants are claims about measurements, so they must hold as claims.

    Their docstrings state the gap each sits in; if either were re-tuned to a
    value that no longer separates those measured decades, the calibration
    stopped being one.
    """
    from mbl.models.constrained.solver_resolution import (
        GRADIENT_TOLERANCE,
        PRIMAL_TOLERANCE,
    )

    # primal: worst honoured 1.3e-05, best refused 4.1e-03
    assert 1.3e-05 < PRIMAL_TOLERANCE < 4.1e-03
    # gradient: worst honoured 1.2e-04, best refused 1.9e-02
    assert 1.2e-04 < GRADIENT_TOLERANCE < 1.9e-02
