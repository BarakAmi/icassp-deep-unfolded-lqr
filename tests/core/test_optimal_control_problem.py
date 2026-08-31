import numpy as np
import pytest

from mbl.core.cost.cost import Cost
from mbl.core.optimal_control_problem import OptimalControlProblem
from mbl.core.system.system import System


class _ConcreteSystem(System):
    def run(self, *args, **kwargs):
        return np.array([0.0])


class _ConcreteCost(Cost):
    def __call__(self, *args, **kwargs):
        return np.array([0.0])


def test_optimal_control_problem_constructs_with_valid_types() -> None:
    problem = OptimalControlProblem(system=_ConcreteSystem(), cost=_ConcreteCost())
    assert isinstance(problem.system, System)
    assert isinstance(problem.cost, Cost)


def test_optimal_control_problem_rejects_invalid_system_type() -> None:
    with pytest.raises(TypeError, match="System"):
        OptimalControlProblem(system="not-system", cost=_ConcreteCost())


def test_optimal_control_problem_rejects_invalid_cost_type() -> None:
    with pytest.raises(TypeError, match="Cost"):
        OptimalControlProblem(system=_ConcreteSystem(), cost="not-cost")


def test_optimal_control_problem_get_signature_composes_system_and_cost() -> None:
    problem = OptimalControlProblem(system=_ConcreteSystem(), cost=_ConcreteCost())
    signature = problem.get_signature()

    assert signature["system"] == {"type": "_ConcreteSystem"}
    assert signature["cost"] == {"type": "_ConcreteCost"}
    assert signature["constraints"] == []


def test_optimal_control_problem_get_signature_aggregates_constraints() -> None:
    from mbl.core.constraint.box_constraint import BoxConstraint

    problem = OptimalControlProblem(
        system=_ConcreteSystem(),
        cost=_ConcreteCost(),
        constraints=[BoxConstraint(u_max=1.0), BoxConstraint(u_max=2.0)],
    )
    signature = problem.get_signature()

    assert signature["constraints"] == [
        {"type": "BoxConstraint", "u_max": 1.0, "u_min": -1.0},
        {"type": "BoxConstraint", "u_max": 2.0, "u_min": -2.0},
    ]
