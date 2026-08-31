from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Optional, Protocol, Tuple, cast

import torch
import torch.nn as nn

from ..base import Config
from ..registry import register_model
from ...core.optimal_control_problem import OptimalControlProblem
from ...core.system.state_space_system import ControlPolicy


class SequenceModelType(StrEnum):
    """Which recurrent/sequence backbone `build_sequence_backbone` constructs."""

    RNN = "RNN"
    GRU = "GRU"
    LSTM = "LSTM"
    MAMBA = "MAMBA"
    LRU = "LRU"


@dataclass(frozen=True)
class NeuralConfig(Config):
    """Hyperparameters for `NeuralPolicy` and its sequence backbone.

    Attributes:
        state_dim: Dimension ``n`` of the observation fed into the backbone.
        control_dim: Dimension ``m`` of the control head's output.
        num_layers: Number of stacked recurrent layers (native RNN/GRU/LSTM only).
        dropout: Inter-layer dropout probability (native RNN/GRU/LSTM only,
            applied only when `num_layers > 1`).
        model_type: Which backbone family to build (see `SequenceModelType`).
        hidden_dim: The backbone's hidden state dimension.
        ssm_state_dim: The Mamba/SSM expansion state dimension (MAMBA/LRU only).
    """

    state_dim: int
    control_dim: int
    num_layers: int = 1
    dropout: float = 0.2
    model_type: SequenceModelType = SequenceModelType.GRU
    hidden_dim: int = 64
    ssm_state_dim: int = 16  # Added specifically for SSM expansion states


# --- 1. Define the Unified Interface (Protocol) ---


class SequenceBackbone(Protocol):
    """
    Protocol enforcing a unified signature for all recurrent sequence models.
    Any class implementing this must accept a sequence tensor and an optional
    hidden state (of Any type), returning the processed sequence and updated state.
    """

    def __call__(self, x: torch.Tensor, h: Any = None) -> Tuple[torch.Tensor, Any]:
        """
        Args:
            x: Input sequence, shape ``(batch, seq_len, state_dim)``.
            h: Optional prior hidden state/cache; backbone-specific type
                (``None`` starts from a zero/default initial state).

        Returns:
            A ``(output, hidden)`` pair: `output` of shape
            ``(batch, seq_len, hidden_dim)``, and the updated hidden state/cache.
        """
        ...


# --- 2. Create Adapters for Model Families ---


class NativeRNNAdapter(nn.Module):
    """Adapter for standard PyTorch RNNs (RNN, GRU, LSTM)."""

    def __init__(self, config: NeuralConfig):
        """
        Args:
            config: Selects `config.model_type` (RNN/GRU/LSTM) and its
                `state_dim`/`hidden_dim`/`num_layers`/`dropout`.

        Raises:
            ValueError: If `config.model_type` is not RNN, GRU, or LSTM.
        """
        super().__init__()
        rnn_kwargs = {
            "input_size": config.state_dim,
            "hidden_size": config.hidden_dim,
            "num_layers": config.num_layers,
            "batch_first": True,
            "dropout": config.dropout if config.num_layers > 1 else 0.0,
        }

        self.rnn: nn.RNNBase
        if config.model_type == SequenceModelType.RNN:
            self.rnn = nn.RNN(**rnn_kwargs)
        elif config.model_type == SequenceModelType.GRU:
            self.rnn = nn.GRU(**rnn_kwargs)
        elif config.model_type == SequenceModelType.LSTM:
            self.rnn = nn.LSTM(**rnn_kwargs)
        else:
            raise ValueError(f"Invalid native RNN type: {config.model_type}")

    def forward(self, x: torch.Tensor, h: Any = None) -> Tuple[torch.Tensor, Any]:
        """See `SequenceBackbone.__call__`; delegates directly to the
        underlying ``nn.RNN``/``nn.GRU``/``nn.LSTM``, whose signature already
        matches the protocol."""
        # Native PyTorch RNNs already perfectly match our Protocol signature
        return cast("Tuple[torch.Tensor, Any]", self.rnn(x, h))


class MambaAdapter(nn.Module):
    """
    Adapter for Mamba/SSM architectures.
    Abstracts away input projection and specialized hidden state caching.
    """

    def __init__(self, config: NeuralConfig):
        """
        Args:
            config: Supplies `state_dim`/`hidden_dim` (for the optional input
                projection) and `ssm_state_dim` (the Mamba expansion state size).

        Raises:
            ModuleNotFoundError: If the optional ``mamba_ssm`` dependency is not
                installed (only raised when a MAMBA/LRU backbone is actually
                requested, not merely on module import).
        """
        super().__init__()
        # Imported lazily so that `mamba_ssm` (a heavy, optional dependency) is only
        # required when a MAMBA/LRU backbone is actually requested, not merely when
        # this module is imported.
        from mamba_ssm import Mamba

        # SSMs often expect d_model to match input_dim. If not, project it.
        self.proj_in = (
            nn.Linear(config.state_dim, config.hidden_dim)
            if config.state_dim != config.hidden_dim
            else nn.Identity()
        )

        self.mamba = Mamba(d_model=config.hidden_dim, d_state=config.ssm_state_dim)

    def forward(
        self, x: torch.Tensor, h: Optional[Any] = None
    ) -> Tuple[torch.Tensor, Any]:
        """See `SequenceBackbone.__call__`.

        Args:
            x: Input sequence, shape ``(batch, seq_len, state_dim)``.
            h: Optional Mamba ``InferenceParams`` cache from a prior call.

        Returns:
            A ``(output, h)`` pair: `output` of shape
            ``(batch, seq_len, hidden_dim)``; `h` returned unchanged (Mamba
            mutates its ``InferenceParams`` cache in place).
        """
        x = self.proj_in(x)

        # Mamba uses an InferenceParams object for its hidden state during autoregression.
        # Here we map our generic 'h' to the specific requirement of the Mamba library.
        out = self.mamba(x, inference_params=h)

        # out = self.mamba(x) # Mock forward pass

        # Return the output and the updated inference parameters cache
        return out, h


# --- 3. The Factory Method ---


def build_sequence_backbone(config: NeuralConfig) -> nn.Module:
    """Factory function to build the correct adapter based on the config.

    Args:
        config: Selects the backbone family via `config.model_type`.

    Returns:
        A `NativeRNNAdapter` (RNN/GRU/LSTM) or `MambaAdapter` (MAMBA/LRU),
        both satisfying `SequenceBackbone`.

    Raises:
        NotImplementedError: If `config.model_type` is not one of the known
            `SequenceModelType` values.
        ModuleNotFoundError: (from `MambaAdapter.__init__`) if a MAMBA/LRU
            backbone is requested and ``mamba_ssm`` is not installed.
    """
    if config.model_type in {
        SequenceModelType.RNN,
        SequenceModelType.GRU,
        SequenceModelType.LSTM,
    }:
        return NativeRNNAdapter(config)
    elif config.model_type in {SequenceModelType.MAMBA, SequenceModelType.LRU}:
        return MambaAdapter(config)

    raise NotImplementedError(f"Model type {config.model_type} is not supported.")


# --- 4. The Clean Facade ---


@register_model("neural")
class NeuralPolicy(nn.Module):
    """
    Unified Control Policy utilizing a plug-and-play sequence backbone.
    The policy is completely agnostic to whether it is running an LSTM or Mamba.

    Attributes:
        problem: The `OptimalControlProblem` this policy controls; its
            `problem.constraints` (if any) are applied to every control output.
        config: This policy's `NeuralConfig`.
        backbone: The `SequenceBackbone` (RNN/GRU/LSTM/Mamba) mapping
            observations to hidden features.
        fc: Linear head mapping backbone hidden features to control space.
    """

    def __init__(self, problem: OptimalControlProblem, config: NeuralConfig) -> None:
        """
        Args:
            problem: The `OptimalControlProblem` this policy controls.
            config: Selects and sizes the sequence backbone and control head.

        Raises:
            NotImplementedError: (from `build_sequence_backbone`) if
                `config.model_type` is unknown.
            ModuleNotFoundError: (from `build_sequence_backbone`) if a
                MAMBA/LRU backbone is requested and ``mamba_ssm`` is missing.
        """
        super().__init__()
        self.problem = problem
        self.config = config

        # Instantiate via the factory
        self.backbone: SequenceBackbone = build_sequence_backbone(config)

        # The classification/control head
        self.fc = nn.Linear(config.hidden_dim, config.control_dim)

    def apply_constraints(self, u: torch.Tensor) -> torch.Tensor:
        """Project `u` through every constraint in `self.problem.constraints`,
        in order (a no-op if there are none).

        Args:
            u: Unconstrained control, shape ``(batch, ..., control_dim)``.

        Returns:
            `u` after sequential projection through each constraint.
        """
        if self.problem.constraints:
            for constraint in self.problem.constraints:
                u = cast(torch.Tensor, constraint(u))
        return u

    def forward(
        self, x: torch.Tensor, h: Optional[Any] = None
    ) -> Tuple[torch.Tensor, Any]:
        """
        Forward pass for the policy.
        Args:
            x: Input state tensor of shape (batch, seq_len, state_dim)
            h: Optional hidden state/cache. Type depends on the underlying backbone.

        Returns:
            A ``(u, h)`` pair: constrained control `u` of shape
            ``(batch, seq_len, control_dim)``, and the backbone's updated
            hidden state/cache.
        """
        # 1. Process sequence and get updated state
        out, h = self.backbone(x, h)

        # 2. Map to control space
        u_unconstrained = self.fc(out)

        # 3. Project to feasible bounds
        u = self.apply_constraints(u_unconstrained)

        return u, h

    def get_control_policy(self) -> ControlPolicy:
        """Return a control policy that **threads the recurrent hidden state**
        across the rollout.

        Called once per rollout, it creates a fresh closure whose ``hidden``
        starts as None and is carried forward between time steps (mirroring
        UnfoldedController's ``prev_u`` scoping). A recurrent policy must retain
        its state across the trajectory: the previous implementation passed
        ``h=None`` every step, silently cold-restarting the sequence model from
        a zero hidden state at each time step -- discarding all temporal memory.
        The hidden state is *not* detached, so gradients propagate through time
        (BPTT), which is the correct training signal for a recurrent controller.
        """
        hidden: Any = None

        def control_policy(t: int, y: torch.Tensor) -> torch.Tensor:
            nonlocal hidden
            u, hidden = self.forward(y.unsqueeze(1), hidden)  # seq_len=1 step
            return u.squeeze(1)

        return cast("ControlPolicy", control_policy)

    def as_module(self) -> nn.Module:
        """`TrainableController` seam (T2.b): the trainable ``nn.Module``
        through which the engine sees this family's parameters.

        A neural policy *is* its own module -- backbone and control head are
        already registered submodules, so ``as_module().parameters()`` is
        exactly the optimizer's parameter source and
        ``as_module().state_dict()`` the checkpoint contract.

        Returns:
            ``self``.
        """
        return self

    def get_signature(self) -> dict[str, Any]:
        """`Signable` member: type + config (all plain scalars/enum, safe to
        `asdict` directly -- no live/trained weights involved).

        The controlled `problem`'s signature is intentionally omitted --
        `ProblemSignatureCallback` already logs it once at the root
        (``problem.*``); embedding it again here would just duplicate every
        key under ``controller.problem.*``.

        Returns:
            ``{"type": "NeuralPolicy", "config": {...}}``.
        """
        return {
            "type": type(self).__name__,
            "config": asdict(self.config),
        }
