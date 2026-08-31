"""Typed exceptions for the Tier 1 runtime substrate.

Both errors are raised at *construction/ingress* time (fail-fast law,
T1.b): a compute context that cannot honor its request must never be
half-built, and hardware unavailability must never silently downgrade
mid-run.
"""

from __future__ import annotations


class InvalidComputeContextError(ValueError):
    """The requested backend/device/precision combination is structurally
    unsupported (e.g. a NumPy backend placed on CUDA, or an unrecognized
    device string) — no hardware probe could ever satisfy it."""


class DeviceUnavailableError(RuntimeError):
    """The requested device is valid but absent on this host (e.g. CUDA
    requested with no visible GPU, or a CUDA index beyond the device count).

    Raised at context construction — never deferred, never silently
    downgraded to CPU. Opt-in fallback exists only at ingress via
    ``ComputeContext.resolve(..., fallback_to_cpu=True)``.
    """


class UnsupportedBackendError(TypeError):
    """The requested `ComputeContext` backend lies outside a controller
    family's declared support envelope (T1.e: declared capability, not
    discovered failure).

    Raised at synthesis ingress — e.g. handing a torch context to a family
    whose mathematics is NumPy-only — never deep inside a run. Families
    declare their envelope as queryable metadata
    (``Synthesizer.SUPPORTED_BACKENDS``); this error names both the request
    and the envelope so the mismatch is immediately actionable.
    """
