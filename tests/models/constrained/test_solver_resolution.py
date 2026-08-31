import numpy as np
import pytest
from scipy.linalg import solve_discrete_are, sqrtm

from mbl.models.constrained import solver_resolution as sr
from mbl.models.constrained.solver_resolution import (
    CanaryResult,
    COCPCanaryInstance,
    resolve_cocp_solver,
    run_solver_canary,
)
from mbl.models.constrained.solver_spec import COCPSolverSpec

DIFFCP = COCPSolverSpec(name="DIFFCP", device="cpu")
MOREAU_CPU = COCPSolverSpec(name="MOREAU", device="cpu")


def _toy_instance(u_max: float = 0.5) -> COCPCanaryInstance:
    """A tiny, arbitrary-but-valid (n=2, m=1) instance for tests that
    monkeypatch the actual solve -- shapes matter, values do not."""
    return COCPCanaryInstance(
        A=np.array([[0.9, 0.0], [0.0, 0.85]]),
        B=np.array([[0.5], [0.5]]),
        R=np.array([[1.0]]),
        u_max=u_max,
        x=np.array([1.0, -1.0]),
        P_sqrt=np.eye(2),
        q=np.zeros(2),
    )


def _nb04_shaped_instance(seed: int = 0, u_max: float = 0.2) -> COCPCanaryInstance:
    """A realistic (n=4, m=2) instance matching NB04's configured scale, with
    a DARE-derived cost-to-go seed -- exactly what `COCPRecipe.build_controller`
    would hand a freshly-initialized COCPController."""
    rng = np.random.default_rng(seed)
    n, m = 4, 2
    A = np.eye(n) * 0.9 + rng.normal(scale=0.05, size=(n, n))
    B = rng.normal(scale=0.5, size=(n, m))
    R = np.eye(m)
    p_are = solve_discrete_are(A, B, np.eye(n), R)
    return COCPCanaryInstance(
        A=A,
        B=B,
        R=R,
        u_max=u_max,
        x=np.ones(n),
        P_sqrt=np.real(sqrtm(p_are)),
        q=np.zeros(n),
    )


def _patch_solve_and_grad(
    monkeypatch: pytest.MonkeyPatch,
    results: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    """Replace the real QP solve with a fixed, keyed-by-solver-name lookup,
    isolating `run_solver_canary`'s comparison logic from any actual solver."""

    def _fake(
        instance: COCPCanaryInstance, spec: COCPSolverSpec
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return results[spec.name]

    monkeypatch.setattr(sr, "_solve_and_grad", _fake)


# --- run_solver_canary: comparison logic (isolated from any real solver) ---


def test_canary_passes_when_candidate_matches_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    u = np.array([0.3])
    grad_p = np.eye(2) * 0.1
    grad_q = np.array([0.05, -0.02])
    _patch_solve_and_grad(
        monkeypatch,
        {
            "DIFFCP": (u, grad_p, grad_q),
            "MOREAU": (u.copy(), grad_p.copy(), grad_q.copy()),
        },
    )

    result = run_solver_canary(_toy_instance(), MOREAU_CPU, DIFFCP)

    assert result == CanaryResult(
        passed=True,
        finite=True,
        feasible=True,
        primal_close=True,
        grad_close=True,
        detail=result.detail,
    )


def test_canary_fails_on_primal_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    grad_p = np.eye(2) * 0.1
    grad_q = np.array([0.05, -0.02])
    _patch_solve_and_grad(
        monkeypatch,
        {
            "DIFFCP": (np.array([0.3]), grad_p, grad_q),
            "MOREAU": (np.array([1.3]), grad_p.copy(), grad_q.copy()),
        },
    )

    result = run_solver_canary(_toy_instance(u_max=5.0), MOREAU_CPU, DIFFCP)

    assert result.primal_close is False
    assert result.passed is False


def test_canary_fails_on_gradient_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The exact failure mode found empirically for MOREAU-CUDA: the primal
    solution agrees closely but a gradient does not."""
    u = np.array([0.3])
    grad_p = np.eye(2) * 0.1
    _patch_solve_and_grad(
        monkeypatch,
        {
            "DIFFCP": (u, grad_p, np.array([0.05, -0.02])),
            "MOREAU": (u.copy(), grad_p.copy(), np.array([0.05, -0.02]) + 10.0),
        },
    )

    result = run_solver_canary(_toy_instance(u_max=5.0), MOREAU_CPU, DIFFCP)

    assert result.primal_close is True
    assert result.grad_close is False
    assert result.passed is False


def test_canary_fails_on_infeasible_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    grad_p = np.eye(2) * 0.1
    grad_q = np.array([0.05, -0.02])
    _patch_solve_and_grad(
        monkeypatch,
        {
            "DIFFCP": (np.array([0.3]), grad_p, grad_q),
            "MOREAU": (np.array([10.0]), grad_p.copy(), grad_q.copy()),
        },
    )

    result = run_solver_canary(_toy_instance(u_max=0.5), MOREAU_CPU, DIFFCP)

    assert result.feasible is False
    assert result.passed is False


def test_canary_fails_on_non_finite_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    grad_p = np.eye(2) * 0.1
    grad_q = np.array([0.05, -0.02])
    _patch_solve_and_grad(
        monkeypatch,
        {
            "DIFFCP": (np.array([0.3]), grad_p, grad_q),
            "MOREAU": (np.array([np.nan]), grad_p.copy(), grad_q.copy()),
        },
    )

    result = run_solver_canary(_toy_instance(u_max=5.0), MOREAU_CPU, DIFFCP)

    assert result.finite is False
    assert result.passed is False


# --- resolve_cocp_solver: the fallback chain ---


def test_resolve_falls_back_when_moreau_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sr, "_probe_moreau", lambda device: "import failed: simulated")

    resolved = resolve_cocp_solver(_toy_instance(), device="cpu")

    assert resolved.spec.name == "DIFFCP"
    assert resolved.attempted == ("DIFFCP",)
    assert resolved.canary is None
    assert "unavailable" in resolved.reason


def test_resolve_falls_back_when_canary_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sr, "_probe_moreau", lambda device: None)
    failing = CanaryResult(
        passed=False,
        finite=True,
        feasible=True,
        primal_close=True,
        grad_close=False,
        detail="simulated grad mismatch",
    )
    monkeypatch.setattr(
        sr, "run_solver_canary", lambda instance, candidate, reference: failing
    )

    resolved = resolve_cocp_solver(_toy_instance(), device="cpu")

    assert resolved.spec.name == "DIFFCP"
    assert resolved.attempted == ("MOREAU", "DIFFCP")
    assert resolved.canary is failing
    assert "canary failed" in resolved.reason


def test_resolve_selects_moreau_when_available_and_canary_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sr, "_probe_moreau", lambda device: None)
    passing = CanaryResult(
        passed=True,
        finite=True,
        feasible=True,
        primal_close=True,
        grad_close=True,
        detail="simulated match",
    )
    monkeypatch.setattr(
        sr, "run_solver_canary", lambda instance, candidate, reference: passing
    )

    resolved = resolve_cocp_solver(
        _toy_instance(), device="cpu", eps=1e-7, max_iters=250
    )

    assert resolved.spec.name == "MOREAU"
    assert resolved.spec.device == "cpu"
    assert resolved.spec.eps == 1e-7
    assert resolved.spec.max_iters == 250
    assert resolved.attempted == ("MOREAU",)
    assert resolved.canary is passing


def test_resolve_forwards_requested_device_to_the_moreau_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, str] = {}

    def _record(device: str) -> str:
        seen["device"] = device
        return "simulated unavailable"

    monkeypatch.setattr(sr, "_probe_moreau", _record)

    resolve_cocp_solver(_toy_instance(), device="cuda:0")

    assert seen["device"] == "cuda:0"


def test_resolve_never_attempts_cuda_unless_explicitly_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§1.0.6: CUDA is only ever attempted when the caller explicitly passes
    a CUDA device -- the default is plain 'cpu', matching the resolved chain
    MOREAU(cpu) -> DIFFCP(cpu)."""
    probed: list[str] = []

    def _record_and_reject(device: str) -> str:
        probed.append(device)
        return "simulated unavailable"

    monkeypatch.setattr(sr, "_probe_moreau", _record_and_reject)

    resolve_cocp_solver(_toy_instance())

    assert probed == ["cpu"]


# --- Integration: the real, installed MOREAU on a realistic instance ---

moreau = pytest.importorskip("moreau")


def test_run_solver_canary_moreau_cpu_matches_diffcp_on_nb04_shaped_instance() -> None:
    result = run_solver_canary(_nb04_shaped_instance(), MOREAU_CPU, DIFFCP)

    assert result.finite
    assert result.feasible
    assert result.primal_close
    assert result.grad_close
    assert result.passed


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_run_solver_canary_moreau_cpu_matches_diffcp_across_seeds(seed: int) -> None:
    """Regression pin for the empirical finding driving this module: MOREAU-
    CPU's gradients matched DIFFCP's on every one of these 5 seeds, unlike
    MOREAU-CUDA (docs/planning/.../Sec 1.0.6)."""
    result = run_solver_canary(
        _nb04_shaped_instance(seed=seed, u_max=5.0), MOREAU_CPU, DIFFCP
    )

    assert result.passed, result.detail


def test_resolve_cocp_solver_selects_moreau_on_this_machine() -> None:
    resolved = resolve_cocp_solver(_nb04_shaped_instance(), device="cpu")

    assert resolved.spec.name == "MOREAU"
    assert resolved.spec.device == "cpu"
    assert resolved.canary is not None
    assert resolved.canary.passed
