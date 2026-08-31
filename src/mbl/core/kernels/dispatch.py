"""The kernel layer's single backend-dispatch site (T1.e).

Every dual-backend kernel resolves its executing array library through
`array_namespace` and nowhere else, so "solvers never branch on backend"
is a checkable property: no ``isinstance(x, torch.Tensor)`` ladder may
appear outside this module.

NumPy and PyTorch expose the operations the kernels need under identical
names and (positional) signatures — ``einsum``, ``cumsum``, ``stack``,
``linalg.solve``, the ``@`` operator, ``.mT`` — so the returned module
object is usable directly as the executing namespace.
"""

from types import ModuleType

import numpy as np
import torch


def array_namespace(*arrays: object) -> ModuleType:
    """Resolve the executing array library for a kernel invocation.

    This is the kernel layer's ONE backend branch (T1.e): callers pass the
    arrays participating in an operation and receive the module (``numpy``
    or ``torch``) that executes it natively.

    Args:
        *arrays: The arrays participating in one kernel operation. Entries
            that are neither ``torch.Tensor`` nor ``numpy.ndarray`` (e.g. a
            duck-typed per-step matrix container) are ignored for dispatch.

    Returns:
        ``torch`` if any argument is a ``torch.Tensor``, else ``numpy``.

    Raises:
        TypeError: If the arguments mix ``torch.Tensor`` and
            ``numpy.ndarray`` operands — an implicit NumPy<->Torch
            conversion inside a kernel is a conversion-boundary violation
            (T1.f); materialize all operands onto one substrate via
            `ComputeContext.asarray` first.
    """
    has_torch = any(isinstance(array, torch.Tensor) for array in arrays)
    has_numpy = any(isinstance(array, np.ndarray) for array in arrays)
    if has_torch and has_numpy:
        kinds = ", ".join(type(array).__name__ for array in arrays)
        raise TypeError(
            "Kernel operands mix NumPy and torch arrays "
            f"({kinds}). Conversions are boundary-only (T1.f): materialize "
            "every operand onto one substrate via ComputeContext.asarray "
            "before invoking a kernel."
        )
    return torch if has_torch else np
