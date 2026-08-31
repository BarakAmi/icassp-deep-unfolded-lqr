import numpy as np
import pytest
import torch

from mbl.core.cost.quadratic_cost import QuadraticCost
from mbl.core.optimal_control_problem import OptimalControlProblem
from mbl.core.system.linear_system import LinearSystem
from mbl.models.neural.nerual import (
    NeuralConfig,
    NeuralPolicy,
    SequenceModelType,
    build_sequence_backbone,
)


def _build_problem() -> OptimalControlProblem:
    system = LinearSystem.fully_observable(np.eye(2), np.eye(2))
    cost = QuadraticCost(Q=np.eye(2), R=np.eye(2))
    return OptimalControlProblem(system=system, cost=cost)


def test_neural_policy_forward_and_control_policy_shapes_with_gru_backbone() -> None:
    problem = _build_problem()
    config = NeuralConfig(
        state_dim=2, control_dim=2, hidden_dim=4, model_type=SequenceModelType.GRU
    )
    policy_net = NeuralPolicy(problem, config)

    assert policy_net.problem is problem
    assert policy_net.config is config

    batch, seq_len = 3, 5
    x = torch.zeros((batch, seq_len, 2))
    u, _ = policy_net(x)
    assert u.shape == (batch, seq_len, 2)

    control_policy = policy_net.get_control_policy()
    y = torch.zeros((batch, 2))
    u_t = control_policy(0, y)
    assert u_t.shape == (batch, 2)


def test_neural_policy_does_not_require_mamba_ssm_for_gru() -> None:
    """Regression test: importing/using the GRU backbone must not require mamba_ssm to be
    installed (it previously failed at module import time for every backbone type)."""
    with pytest.raises(ModuleNotFoundError):
        import mamba_ssm  # noqa: F401 -- confirms the test environment lacks it

    config = NeuralConfig(state_dim=2, control_dim=1, model_type=SequenceModelType.GRU)
    backbone = build_sequence_backbone(config)
    assert backbone is not None


def test_mamba_backbone_only_fails_lazily_when_actually_requested() -> None:
    config = NeuralConfig(
        state_dim=2, control_dim=1, model_type=SequenceModelType.MAMBA
    )
    with pytest.raises(ModuleNotFoundError):
        build_sequence_backbone(config)


def test_neural_policy_threads_hidden_state_across_timesteps() -> None:
    """V-1 regression: the recurrent hidden state must be carried across time
    steps of a rollout. Two consecutive steps fed the *same* observation must
    yield *different* controls (the GRU state has advanced); the previous bug
    passed h=None every step, cold-restarting from a zero hidden state and
    producing identical controls."""
    torch.manual_seed(0)
    problem = _build_problem()
    config = NeuralConfig(
        state_dim=2, control_dim=2, hidden_dim=8, model_type=SequenceModelType.GRU
    )
    policy_net = NeuralPolicy(problem, config)

    policy = policy_net.get_control_policy()
    y = torch.ones(3, 2)
    u0 = policy(0, y)
    u1 = policy(1, y)
    assert not torch.allclose(u0, u1)  # hidden state advanced between steps

    # A fresh policy resets the hidden state (scope = one rollout), so its first
    # step reproduces the very first control exactly.
    fresh_policy = policy_net.get_control_policy()
    u0_again = fresh_policy(0, y)
    assert torch.allclose(u0, u0_again)


def test_neural_policy_get_signature_reports_type_and_config() -> None:
    """`problem` is intentionally absent from this controller's own signature
    (Phase 1E): `ProblemSignatureCallback` already logs it once at the root."""
    problem = _build_problem()
    config = NeuralConfig(state_dim=2, control_dim=2, hidden_dim=8)
    policy_net = NeuralPolicy(problem, config)

    signature = policy_net.get_signature()

    assert signature["type"] == "NeuralPolicy"
    assert signature["config"]["state_dim"] == 2
    assert signature["config"]["hidden_dim"] == 8
    assert "problem" not in signature
