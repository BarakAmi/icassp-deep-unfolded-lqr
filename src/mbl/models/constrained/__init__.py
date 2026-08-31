"""Convex-optimization-based constrained control: COCP (a differentiable
per-step QP policy), its solver seam (which conic backend, validated by a
capability probe + canary), and its SDP-relaxation theoretical lower bound."""

from .box_lagrangian import (
    BoxLQRData,
    DualBound,
    finite_horizon_box_bound,
    infinite_horizon_box_bound,
)
from .box_qp import (
    BoxQPCertificate,
    BoxQPSolution,
    box_qp,
    certify_box_qp,
    residual_tolerance,
    solve_box_qp,
)
from .cocp import COCPConfig, COCPController
from .cocp_exact import ExactCOCPConfig, ExactCOCPController
from .lower_bound import solve_box_constrained_lower_bound
from .solver_resolution import (
    CanaryResult,
    COCPCanaryInstance,
    ResolvedCOCPSolver,
    resolve_cocp_solver,
    run_solver_canary,
)
from .solver_spec import COCPSolverSpec

__all__ = [
    "BoxLQRData",
    "BoxQPCertificate",
    "DualBound",
    "finite_horizon_box_bound",
    "infinite_horizon_box_bound",
    "BoxQPSolution",
    "COCPConfig",
    "COCPController",
    "ExactCOCPConfig",
    "ExactCOCPController",
    "box_qp",
    "certify_box_qp",
    "residual_tolerance",
    "solve_box_qp",
    "COCPSolverSpec",
    "COCPCanaryInstance",
    "CanaryResult",
    "ResolvedCOCPSolver",
    "resolve_cocp_solver",
    "run_solver_canary",
    "solve_box_constrained_lower_bound",
]
