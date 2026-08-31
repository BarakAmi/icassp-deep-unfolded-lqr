"""`ComputeContext` — the single numeric-substrate authority (T1.b).

One frozen, signable value object answers the three questions no other
component may answer for itself: which array **backend** executes the
mathematics (NumPy or PyTorch), on which **device** tensors live, and at
which floating-point **precision**. It is injected, never ambient: every
constructor that previously took loose ``dtype=``/``device=`` keywords
takes one context (or none, inheriting its owner's).

Contractual laws implemented here:

* **Declared and normalised at specification; validated at execution**
  (D17 as extended 2026-08-23, Annex 06 §3.5). Constructing a context parses
  the device, refuses an unsupported type or an incoherent backend/device
  pair, and normalises ``cuda`` → ``cuda:0`` so signatures are deterministic —
  **all without touching hardware**, because this value signs and a probe of
  the *reading* machine would make `ModelID` machine-dependent, which is what
  D19 removed. Requesting an unavailable device (CUDA on a GPU-less host, an
  out-of-range index) raises `DeviceUnavailableError` from
  `require_available_device`, called at ingress — `ComputeContext.resolve`
  and the runner — so a run on the wrong host fails **before it computes
  anything**, and never by downgrading to CPU silently mid-run. The only
  fallback is explicit and at ingress:
  ``ComputeContext.resolve(..., fallback_to_cpu=True)``, whose *effective*
  resolved context is what propagates and signs.

  *This previously happened at construction, so a CPU-only host could not even
  READ a study declaring CUDA — not resolve it, not analyse it, not replay it —
  and thirteen tracked documents failed CI on a check about document
  well-formedness.*
* **No global mutable singleton.** `default_compute_context` is a
  function returning a fresh default; overriding happens only by explicit
  injection.
* **Array authorship flows through the context.** Interior code obtains
  new arrays via `asarray`/`zeros` instead of importing a backend and
  deciding placement locally (the seed of the T1.f conversion-boundary law).
* **Signature-relevant.** Backend, device, and precision all change
  numerics, so the context participates in experiment signatures — in
  contrast to `ValidationMode` (see ``modes.py``), which is
  signature-exempt by design.

The float64 default encodes the convention both application configs
previously documented in comments ("cvxpylayers/SCS needs double
precision") as the typed default.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, cast

import numpy as np
import torch

from .exceptions import DeviceUnavailableError, InvalidComputeContextError


class Backend(StrEnum):
    """The array framework executing the mathematics."""

    NUMPY = "numpy"
    TORCH = "torch"


class Precision(StrEnum):
    """The working floating-point width, backend-agnostic by name."""

    FLOAT32 = "float32"
    FLOAT64 = "float64"

    @property
    def numpy_dtype(self) -> np.dtype:
        """This precision as a ``numpy.dtype``."""
        return np.dtype(self.value)

    @property
    def torch_dtype(self) -> torch.dtype:
        """This precision as a ``torch.dtype``."""
        return _TORCH_DTYPES[self]

    @classmethod
    def from_torch_dtype(cls, dtype: torch.dtype) -> "Precision":
        """The `Precision` naming a ``torch.dtype`` (ingress helper for
        configs that still carry a torch dtype field).

        Args:
            dtype: ``torch.float32`` or ``torch.float64``.

        Returns:
            The matching `Precision`.

        Raises:
            ValueError: If `dtype` has no named precision.
        """
        for precision, torch_dtype in _TORCH_DTYPES.items():
            if torch_dtype == dtype:
                return precision
        raise ValueError(
            f"No Precision corresponds to {dtype}; "
            f"supported: {sorted(p.value for p in cls)}."
        )


_TORCH_DTYPES = {
    Precision.FLOAT32: torch.float32,
    Precision.FLOAT64: torch.float64,
}

# The declared device envelope. Families/backends with narrower support
# declare it (T1.e); anything outside this set fails at ingress with a
# typed error, never deep inside a run.
_SUPPORTED_DEVICE_TYPES = frozenset({"cpu", "cuda"})


def _normalise_device(backend: Backend, device: str) -> str:
    """Normalize a device request. **Specification only — no hardware.**

    D17 as extended 2026-08-23 (Annex 06 §3.5): a resolution performed while a
    document is read is a probe of the *reading* machine, and this function's
    result **signs**. Everything here is therefore decided from the string and
    the backend alone — measured, not assumed: ``torch.device("cuda").index``
    parses on a CPU-only host, so the ``cuda`` → ``cuda:0`` normalisation that
    makes signatures deterministic needs no CUDA runtime.

    The availability of the named device is a separate question, asked by
    `require_available_device` where the context is used to compute.

    Args:
        backend: The already-normalized array backend.
        device: The requested device string (``"cpu"``, ``"cuda"``,
            ``"cuda:1"``, ...).

    Returns:
        The canonical device string (``"cpu"``, or ``"cuda:<index>"`` with
        an explicit index so signatures are deterministic).

    Raises:
        InvalidComputeContextError: If `device` is unparseable, outside the
            supported envelope, or structurally incompatible with `backend`
            (NumPy executes on CPU only).
    """
    try:
        parsed = torch.device(device)
    except (ValueError, RuntimeError, TypeError) as error:
        raise InvalidComputeContextError(
            f"Unparseable device string {device!r}: {error}"
        ) from error

    if parsed.type not in _SUPPORTED_DEVICE_TYPES:
        raise InvalidComputeContextError(
            f"Device type {parsed.type!r} is outside the supported envelope "
            f"{sorted(_SUPPORTED_DEVICE_TYPES)}."
        )
    if backend is Backend.NUMPY and parsed.type != "cpu":
        raise InvalidComputeContextError(
            f"Backend {Backend.NUMPY.value!r} executes on CPU only; "
            f"got device {device!r}. Use backend={Backend.TORCH.value!r} "
            "for accelerator placement."
        )

    if parsed.type == "cuda":
        # An explicit index so the signature is deterministic. `index` is None
        # for a bare "cuda", and torch parses that without a CUDA runtime.
        return f"cuda:{parsed.index if parsed.index is not None else 0}"
    return "cpu"


def require_available_device(device: str) -> None:
    """Refuse a declared device this host cannot honour. **The capability probe.**

    The other half of `_normalise_device`, and the half that needs hardware.
    Called at **execution ingress** — once, before a run computes anything —
    rather than while a document is read, so that reading, resolving,
    analysing and replaying a study do not require the machine it was measured
    on. Annex 06 §3.5.

    **This is a refusal relocated, not weakened.** A run on the wrong host
    still fails, and still fails before it does any work; what it must never do
    is downgrade to CPU silently mid-run, which is the failure the law was
    written against. The one sanctioned downgrade stays explicit and at
    ingress: `ComputeContext.resolve(..., fallback_to_cpu=True)`.

    Args:
        device: A device string already normalised by `_normalise_device`.

    Raises:
        DeviceUnavailableError: If the request is valid but this host cannot
            satisfy it (no CUDA runtime, or CUDA index out of range).
    """
    parsed = torch.device(device)
    if parsed.type != "cuda":
        return
    if not torch.cuda.is_available():
        raise DeviceUnavailableError(
            f"Device {device!r} was explicitly requested but CUDA is not "
            "available on this host. Refusing to fall back silently; use "
            "ComputeContext.resolve(..., fallback_to_cpu=True) for an "
            "explicit ingress-time fallback."
        )
    index = parsed.index if parsed.index is not None else 0
    count = torch.cuda.device_count()
    if index >= count:
        raise DeviceUnavailableError(
            f"Device {device!r} requested but this host exposes only "
            f"{count} CUDA device(s) (valid indices: 0..{count - 1})."
        )


@dataclass(frozen=True, slots=True)
class ComputeContext:
    """The injected execution environment: backend x device x precision.

    Attributes:
        backend: The array framework (`Backend`; accepts its string value).
        device: Canonical device string; validated and normalized at
            construction (see `_resolve_device`).
        precision: Working float width (`Precision`; accepts its string
            value).
    """

    # Declared as unions because construction accepts the enums' string
    # values (the documented ingress contract); __post_init__ normalizes both
    # to enums, and `backend_enum`/`precision_enum` expose that invariant to
    # the type checker.
    backend: Backend | str = Backend.NUMPY
    device: str = "cpu"
    precision: Precision | str = Precision.FLOAT64

    @property
    def backend_enum(self) -> Backend:
        """`backend` as the `Backend` enum `__post_init__` normalized it to."""
        return cast(Backend, self.backend)

    @property
    def precision_enum(self) -> Precision:
        """`precision` as the enum `__post_init__` normalized it to."""
        return cast(Precision, self.precision)

    def __post_init__(self) -> None:
        # Normalize string inputs into enums, then resolve hardware loudly.
        # object.__setattr__ is the sanctioned frozen-dataclass idiom.
        backend = Backend(self.backend)
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "precision", Precision(self.precision))
        object.__setattr__(self, "device", _normalise_device(backend, self.device))

    @classmethod
    def resolve(
        cls,
        backend: Backend | str = Backend.NUMPY,
        device: str = "cpu",
        precision: Precision | str = Precision.FLOAT64,
        *,
        fallback_to_cpu: bool = False,
    ) -> ComputeContext:
        """Build a context with an explicit, opt-in ingress fallback policy.

        Args:
            backend: The requested array backend.
            device: The requested device string.
            precision: The requested floating-point width.
            fallback_to_cpu: If ``True``, a `DeviceUnavailableError` (hardware
                absent) resolves to the same context on ``"cpu"`` instead of
                raising; the *effective* context is what propagates and signs.
                Structurally invalid requests (`InvalidComputeContextError`)
                always raise.

        Returns:
            The resolved, effective `ComputeContext`.

        Raises:
            DeviceUnavailableError: If the device is absent and
                `fallback_to_cpu` is ``False``.
            InvalidComputeContextError: If the combination is structurally
                unsupported regardless of hardware.
        """
        # `cls(...)` is the SPECIFICATION constructor and performs no hardware
        # probe (D17 as extended; Annex 06 §3.5). `resolve` is the INGRESS
        # one, so the probe belongs here -- which is also what makes an
        # explicit fallback decidable at all.
        context = cls(backend=backend, device=device, precision=precision)
        try:
            require_available_device(str(context.device))
        except DeviceUnavailableError:
            if not fallback_to_cpu:
                raise
            return cls(backend=backend, device="cpu", precision=precision)
        return context

    # --- signing -------------------------------------------------------------

    def get_signature(self) -> dict[str, Any]:
        """This context's provenance contribution (see ``core.utils.signing``).

        Backend, device, and precision all change numerics, so all three
        participate in experiment signatures and hence cache keys.

        Returns:
            A flat, JSON-serializable signature dict.
        """
        return {
            "type": "ComputeContext",
            "backend": self.backend_enum.value,
            "device": self.device,
            "precision": self.precision_enum.value,
        }

    # --- typed accessors -----------------------------------------------------

    @property
    def numpy_dtype(self) -> np.dtype:
        """The working precision as a ``numpy.dtype``."""
        return self.precision_enum.numpy_dtype

    @property
    def torch_dtype(self) -> torch.dtype:
        """The working precision as a ``torch.dtype``."""
        return self.precision_enum.torch_dtype

    @property
    def torch_device(self) -> torch.device:
        """The resolved placement as a ``torch.device``."""
        return torch.device(self.device)

    # --- array authorship (the designated conversion boundary, T1.f seed) ----

    def asarray(self, data: Any) -> np.ndarray | torch.Tensor:
        """Materialize `data` as an array of this context's backend, dtype,
        and placement.

        Args:
            data: Anything ``np.asarray``/``torch.as_tensor`` accepts
                (sequences, scalars, arrays, tensors).

        Returns:
            A ``numpy.ndarray`` (NumPy backend) or ``torch.Tensor`` (torch
            backend) at this context's precision and device.
        """
        if self.backend is Backend.NUMPY:
            return np.asarray(data, dtype=self.numpy_dtype)
        return torch.as_tensor(data, dtype=self.torch_dtype, device=self.torch_device)

    def zeros(self, shape: tuple[int, ...] | int) -> np.ndarray | torch.Tensor:
        """Author a zero-filled array on this context's substrate.

        Args:
            shape: The desired array shape.

        Returns:
            A zero array of `shape` at this context's backend, precision,
            and device.
        """
        if self.backend is Backend.NUMPY:
            return np.zeros(shape, dtype=self.numpy_dtype)
        return torch.zeros(shape, dtype=self.torch_dtype, device=self.torch_device)


def default_compute_context() -> ComputeContext:
    """The module-level default-context *function* (never a mutable global):
    NumPy on CPU at float64 — byte-identical to the modern tree's dominant
    convention. Scripts/tests without an owning application use this;
    everything else receives its context by injection.

    Returns:
        A fresh default `ComputeContext`.
    """
    return ComputeContext()
