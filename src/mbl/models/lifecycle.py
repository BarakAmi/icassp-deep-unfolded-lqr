"""The two-phase controller lifecycle contracts (REFACTOR_PLAN v3, T2.a/T2.b):

    Synthesizer  --synthesize-->  SynthesizedController  --make_policy-->  policy
      (OFFLINE, expensive, once)     (frozen artifact)       (ONLINE, per rollout)

These protocols make the offline-synthesis/online-inference boundary
*contractual*, replacing the deprecated single-phase `models.base.Controller`
shape in which Riccati solves, GD refinement, and neural training were fused
into the same object that later served rollouts.

Contractual laws (docstring-normative, enforced by
``tests/models/test_lifecycle_contracts.py``):

* **Phase disjointness.** Everything that depends only on
  ``(problem, config)`` happens inside `Synthesizer.synthesize`;
  `SynthesizedController.make_policy` and the rollout perform *no*
  synthesis work. Heterogeneous offline/online benchmarking reads the
  phase boundary off the contract instead of reverse-engineering it per
  family.
* **Statelessness of the synthesizer (zero state leakage).**
  ``synthesize`` *returns* its artifact and never mutates ``self`` — no
  ``solve()``-before-``get_control_policy()`` temporal coupling, no
  attribute smuggling.
* **Freshness.** Every `make_policy` call returns a *fresh* policy
  closure; stateful policies (hidden state, warm starts) can never leak
  state across rollouts.
* **Backend-agnostic synthesis.** ``synthesize`` executes on the substrate
  the injected `ComputeContext` dictates, through the Tier-1.e kernel
  layer. No synthesizer contains a backend or device literal; a family
  with a narrower envelope *declares* it via ``SUPPORTED_BACKENDS`` and
  fails at ingress (`UnsupportedBackendError`), never mid-run.
* **Signature discipline (the C1 law).** `Synthesizer.get_signature`
  covers specification-time configuration only — never ``problem.*``
  (the engine's `ProblemSignatureCallback` logs the problem exactly once
  at the root), never live/trained parameter values.
  `SynthesizedController.get_signature` = the synthesizer's signature
  plus synthesis provenance (e.g. the effective compute context).
* **Introspectable residency.** A `SynthesizedController` answers where
  its artifact bytes live via its ``context`` — profiling attribution is
  read off the artifact, never hand-maintained in a benchmark harness
  (T2.i groundwork).
"""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Protocol, runtime_checkable

from torch import nn

from ..core.optimal_control_problem import OptimalControlProblem
from ..core.runtime import (
    Backend,
    ComputeContext,
    UnsupportedBackendError,
    default_compute_context,
)
from ..core.system.state_space_system import ControlPolicy
from .base import Config

ALL_BACKENDS: frozenset[Backend] = frozenset(Backend)
"""The full backend envelope — the default `SUPPORTED_BACKENDS` declaration."""


@runtime_checkable
class SynthesizedController(Protocol):
    """The frozen OFFLINE artifact: solved gains / trained weights plus
    provenance. Cheap to hold, persistable, signable.

    Attributes:
        context: The *effective* `ComputeContext` the artifact was
            synthesized under — where its bytes live (residency law).
            Declared read-only so frozen-dataclass artifacts satisfy the
            protocol structurally.
    """

    @property
    def context(self) -> ComputeContext:
        """The artifact's effective compute context (residency law)."""
        ...

    def make_policy(self) -> ControlPolicy:
        """ONLINE factory: a *fresh* batched ``(t, y_t) -> u_t`` closure per
        call (freshness law) performing no synthesis work."""
        ...

    def get_signature(self) -> dict[str, Any]:
        """Synthesizer signature + synthesis provenance; never ``problem.*``."""
        ...


@runtime_checkable
class Synthesizer(Protocol):
    """The OFFLINE phase. Everything expensive and one-time lives behind
    `synthesize`: Riccati/DARE solves, iterative-GD refinement, cvxpylayers
    compilation, neural / deep-unfolded training (which, from Stage S3,
    internally delegates to the engine)."""

    def synthesize(
        self, problem: OptimalControlProblem, ctx: ComputeContext
    ) -> SynthesizedController:
        """Solve/compile/train against `problem` on `ctx`'s substrate and
        return the frozen artifact (statelessness law: never mutate self)."""
        ...

    def get_signature(self) -> dict[str, Any]:
        """Specification-time configuration ONLY (C1 law)."""
        ...


@runtime_checkable
class TrainableController(SynthesizedController, Protocol):
    """Torch-native sub-contract (T2.b, deep-unfolding readiness): a
    synthesized controller whose artifact is (or wraps) a trainable
    ``nn.Module``.

    `as_module` is the single seam through which the engine sees
    parameters — optimizer construction, checkpointing
    (``state_dict()`` in memory, safetensors on disk per T3.i), and
    train/eval mode discipline all flow through the returned module,
    deleting the per-family ``rollout.parameters()`` / ``p.get_raw()``
    extraction ladders (consumed by the Stage-S3 `TrainingPlan` work).
    """

    def as_module(self) -> nn.Module:
        """The trainable ``nn.Module``: every learnable parameter of this
        controller is a registered ``nn.Parameter`` of the returned module
        (families whose parameters are structured objects expose them as
        registered parameters; a neural policy returns itself)."""
        ...


def supported_backends(synthesizer: object) -> frozenset[Backend]:
    """A synthesizer family's declared backend envelope (T1.e).

    Args:
        synthesizer: Any `Synthesizer` (class or instance).

    Returns:
        The family's ``SUPPORTED_BACKENDS`` declaration, defaulting to the
        full envelope for families that omit it.
    """
    cls = synthesizer if isinstance(synthesizer, type) else type(synthesizer)
    return getattr(cls, "SUPPORTED_BACKENDS", ALL_BACKENDS)


def ensure_backend_supported(
    synthesizer: object, ctx: ComputeContext
) -> ComputeContext:
    """Fail-fast ingress gate: `ctx` must lie inside the family's envelope.

    Args:
        synthesizer: The `Synthesizer` about to run.
        ctx: The injected compute context.

    Returns:
        `ctx`, unchanged, when supported.

    Raises:
        UnsupportedBackendError: If ``ctx.backend`` is outside the family's
            declared ``SUPPORTED_BACKENDS`` (declared capability, not
            discovered failure — the error names both sides).
    """
    envelope = supported_backends(synthesizer)
    if ctx.backend not in envelope:
        name = (
            type(synthesizer).__name__
            if not isinstance(synthesizer, type)
            else synthesizer.__name__
        )
        raise UnsupportedBackendError(
            f"{name} declares the backend envelope "
            f"{sorted(backend.value for backend in envelope)}; the injected "
            f"ComputeContext requests backend {ctx.backend_enum.value!r}. "
            "Choose a context inside the envelope, or a family supporting "
            "the requested backend."
        )
    return ctx


class SynthesizerBase(ABC):
    """Opt-in base for concrete synthesizers: the correct, *inheritable*
    signature default the deprecated `Controller` Protocol only pretended
    to provide (its body was never structurally inherited — C1), plus the
    envelope/context ingress resolution every family needs.

    Attributes:
        SUPPORTED_BACKENDS: The family's declared backend envelope;
            override to narrow it (e.g. a cvxpylayers family is
            torch-only, a constraint-projecting family may be NumPy-only).
        config: Optional specification-time `Config`, folded into the
            default signature when present.
    """

    SUPPORTED_BACKENDS: ClassVar[frozenset[Backend]] = ALL_BACKENDS

    config: Config | None = None

    @abstractmethod
    def synthesize(
        self, problem: OptimalControlProblem, ctx: ComputeContext | None = None
    ) -> SynthesizedController:
        """See `Synthesizer.synthesize`; ``ctx=None`` inherits the module
        default (`default_compute_context`)."""

    def get_signature(self) -> dict[str, Any]:
        """The C1-correct default: this synthesizer's own type and
        specification-time config — never ``problem.*``, never live
        parameters.

        Returns:
            ``{"type": <class name>}``, plus ``"config"`` when `config`
            is set.
        """
        signature: dict[str, Any] = {"type": type(self).__name__}
        if self.config is not None:
            signature["config"] = self.config.as_dict()
        return signature

    def _resolve_context(self, ctx: ComputeContext | None) -> ComputeContext:
        """Ingress resolution shared by every concrete `synthesize`: default
        the context, then gate it against the declared envelope.

        Args:
            ctx: The injected context, or ``None`` for the module default.

        Returns:
            The effective, envelope-validated `ComputeContext`.

        Raises:
            UnsupportedBackendError: Via `ensure_backend_supported`.
        """
        effective = ctx if ctx is not None else default_compute_context()
        return ensure_backend_supported(self, effective)
