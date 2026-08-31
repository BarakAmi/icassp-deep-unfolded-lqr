import pytest
from cvxpylayers.torch import CvxpyLayer

from mbl.models.constrained.solver_spec import COCPSolverSpec


def test_diffcp_translates_to_scs_shaped_args() -> None:
    spec = COCPSolverSpec(name="DIFFCP", eps=1e-6, max_iters=500)
    assert spec.to_solver_args() == {
        "acceleration_lookback": 0,
        "eps": 1e-6,
        "max_iters": 500,
    }


def test_moreau_translates_to_ipm_shaped_args() -> None:
    spec = COCPSolverSpec(name="MOREAU", eps=1e-6, max_iters=500)
    assert spec.to_solver_args() == {
        "max_iter": 500,
        "ipm_settings": {
            "tol_gap_abs": 1e-6,
            "tol_gap_rel": 1e-6,
            "tol_feas": 1e-6,
        },
    }


def test_diffcp_default_matches_todays_preexisting_defaults() -> None:
    """Locks in the values `_DEFAULT_SOLVER_ARGS` used before this spec
    existed, so the spec's default is a behavior-preserving replacement."""
    assert COCPSolverSpec().to_solver_args() == {
        "acceleration_lookback": 0,
        "eps": 1e-8,
        "max_iters": 10000,
    }


def test_extra_args_merge_into_diffcp_args() -> None:
    spec = COCPSolverSpec(name="DIFFCP", extra_args={"verbose": True})
    args = spec.to_solver_args()
    assert args["verbose"] is True
    assert args["eps"] == 1e-8


def test_extra_args_merge_into_moreau_args() -> None:
    spec = COCPSolverSpec(name="MOREAU", extra_args={"verbose": True})
    args = spec.to_solver_args()
    assert args["verbose"] is True
    assert args["max_iter"] == 10000


def test_unknown_solver_name_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown COCP solver"):
        COCPSolverSpec(name="SCS")  # "SCS" is not cvxpylayers' own name


@pytest.mark.parametrize("bad_eps", [0.0, -1e-8])
def test_non_positive_eps_rejected(bad_eps: float) -> None:
    with pytest.raises(ValueError, match="eps must be positive"):
        COCPSolverSpec(eps=bad_eps)


@pytest.mark.parametrize("bad_max_iters", [0, -100])
def test_non_positive_max_iters_rejected(bad_max_iters: int) -> None:
    with pytest.raises(ValueError, match="max_iters must be positive"):
        COCPSolverSpec(max_iters=bad_max_iters)


def test_get_signature_reports_every_field() -> None:
    spec = COCPSolverSpec(
        name="MOREAU",
        device="cuda:0",
        eps=1e-7,
        max_iters=200,
        extra_args={"verbose": True},
    )
    assert spec.get_signature() == {
        "type": "COCPSolverSpec",
        "name": "MOREAU",
        "device": "cuda:0",
        "eps": 1e-7,
        "max_iters": 200,
        "extra_args": {"verbose": True},
    }


def test_diffcp_name_is_accepted_by_cvxpylayer_solver_kwarg() -> None:
    """`CvxpyLayer(solver="DIFFCP")` must behave identically to the bare
    `CvxpyLayer(...)` (solver=None) this codebase relied on before this
    spec existed -- pins the "DIFFCP", not "SCS", naming decision."""
    import cvxpy as cp

    x = cp.Variable(2)
    a = cp.Parameter((3, 2))
    b = cp.Parameter(3)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(a @ x - b)), [x >= 0])

    default_layer = CvxpyLayer(problem, parameters=[a, b], variables=[x])
    explicit_layer = CvxpyLayer(
        problem, parameters=[a, b], variables=[x], solver="DIFFCP"
    )

    import torch

    a_t = torch.randn(3, 2, dtype=torch.float64)
    b_t = torch.randn(3, dtype=torch.float64)
    (default_out,) = default_layer(a_t, b_t)
    (explicit_out,) = explicit_layer(a_t, b_t)
    assert torch.allclose(default_out, explicit_out)
