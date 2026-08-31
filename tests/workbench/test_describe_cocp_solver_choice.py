"""Unit tests for `workbench.analysis.describe_cocp_solver_choice` -- the
Tier-3 narration helper (NB04 COCP refinement plan Sec 3.1) that reports
which COCP solver a problem instance would resolve to, without training
anything.
"""

import numpy as np
import pytest

from mbl.core.cost.quadratic_cost import QuadraticCost
from mbl.core.optimal_control_problem import OptimalControlProblem
from mbl.core.system.linear_system import LinearSystem
from mbl.workbench.analysis import describe_cocp_solver_choice


def _problem(state_dim: int = 3, control_dim: int = 2):
    rng = np.random.default_rng(0)
    A = np.eye(state_dim) * 0.9 + rng.normal(scale=0.03, size=(state_dim, state_dim))
    B = rng.normal(scale=0.5, size=(state_dim, control_dim))
    system = LinearSystem.fully_observable(A, B)
    cost = QuadraticCost(Q=np.eye(state_dim), R=np.eye(control_dim))
    return OptimalControlProblem(system=system, cost=cost)


def test_returns_a_finite_reason_and_a_valid_solver_name() -> None:
    resolved = describe_cocp_solver_choice(_problem(), u_max=0.2, device="cpu")

    assert resolved.spec.name in {"DIFFCP", "MOREAU"}
    assert resolved.spec.device == "cpu"
    assert isinstance(resolved.reason, str) and resolved.reason


def test_forwards_eps_and_max_iters_into_the_resolved_spec() -> None:
    resolved = describe_cocp_solver_choice(
        _problem(), u_max=0.2, device="cpu", eps=1e-6, max_iters=500
    )

    assert resolved.spec.eps == 1e-6
    assert resolved.spec.max_iters == 500


moreau = pytest.importorskip("moreau")


def test_selects_moreau_on_this_machine_for_a_realistic_problem() -> None:
    resolved = describe_cocp_solver_choice(_problem(), u_max=0.2, device="cpu")

    assert resolved.spec.name == "MOREAU"
    assert resolved.canary is not None
    assert resolved.canary.passed
