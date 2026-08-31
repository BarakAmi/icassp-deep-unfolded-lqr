"""Base abstractions shared by every controller/model family: the `Config`
base for per-model hyperparameter dataclasses, and the DEPRECATED
single-phase `Controller` protocol.

The architectural contract layer now lives in ``models.lifecycle``
(`Synthesizer` → `SynthesizedController` → policy, plus
`TrainableController`): the explicit two-phase offline/online lifecycle
that replaces the single-phase `Controller` shape (REFACTOR_PLAN v3,
T2.a/T2.b, Stage S2).
"""

from dataclasses import asdict, dataclass
from typing import Protocol, runtime_checkable

from ..core.optimal_control_problem import OptimalControlProblem
from ..core.system.state_space_system import ControlPolicy


@dataclass(frozen=True)
class Config:
    """Base class for per-model hyperparameter/configuration dataclasses
    (e.g. `NeuralConfig`, `UnfoldingConfig`, `TrainingConfig`).

    A plain frozen dataclass — the previous vacuous ``ABC`` base (no
    abstract members) is gone (N7).
    """

    def as_dict(self) -> dict[str, object]:
        """This config's fields as a plain dict, e.g. for logging via
        `ExperimentTracker.log_params`.

        Uses `dataclasses.asdict` (N7 fix): only genuine dataclass fields
        are reported (never stray instance attributes), and nested
        dataclass values convert recursively.

        Returns:
            The dataclass' field values, deep-converted.
        """
        return asdict(self)

    def get_config(self) -> dict[str, object]:
        """Deprecated pre-S2 spelling of `as_dict` (retained one stage for
        the engine layer, which is frozen until Stage S3).

        Returns:
            `as_dict()`.
        """
        return self.as_dict()


@runtime_checkable
class Controller(Protocol):
    """DEPRECATED: the single-phase controller contract, scheduled for
    deletion once the engine and application layers migrate onto the
    two-phase lifecycle (`models.lifecycle`, Stage S3/S4).

    This shape fuses offline synthesis and online inference in one object
    — exactly the implicit boundary T2.a makes contractual. It survives
    this stage solely because ``engine/`` and ``applications/`` still
    import it for typing; no new implementor may target it.

    C1 fix (S2): this is now a *pure* Protocol — no method bodies. The
    previous concrete `get_signature` default embedded
    ``self.problem.get_signature()`` in direct contradiction of its own
    docstring (duplicating every ``problem.*`` key the engine's
    `ProblemSignatureCallback` already logs at the root), and, being a
    Protocol body, was never structurally inherited anyway — simultaneously
    dead and misleading. The correct, inheritable default lives on
    ``models.lifecycle.SynthesizerBase``.

    Attributes:
        problem: The `OptimalControlProblem` this controller solves/serves.
        config: This controller's `Config`, or ``None`` for non-learnable
            controllers with no hyperparameters to log. Both are declared
            read-only so implementors may hold narrower concrete types
            (a mutable protocol attribute would be invariant).
    """

    @property
    def problem(self) -> OptimalControlProblem:
        """The `OptimalControlProblem` this controller solves/serves."""
        ...

    @property
    def config(self) -> Config | None:
        """This controller's `Config` (``None`` when non-learnable)."""
        ...

    def get_control_policy(self) -> ControlPolicy:
        """Return a callable control policy ``(t, y_t) -> u_t`` (batched)."""
        ...

    def get_signature(self) -> dict:
        """Describe this controller's own type/config — never
        ``problem.*`` (see `models.lifecycle` for the signature laws)."""
        ...
