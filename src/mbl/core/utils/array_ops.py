"""Generic array/tensor interop helpers shared across numpy- and torch-based
code paths (rollouts, cost evaluation, system matrices): batch-dimension
reshaping, cumulative mean, framework-agnostic stacking, and numpy<->torch
type conversion. Domain-specific numerics (e.g. the quadratic-cost kernel)
live in their own domain package, not here."""

from typing import Sequence, cast

import numpy as np
import torch

from .validation import ensure_numpy_ndarray


def add_batch_dim(array: np.ndarray) -> np.ndarray:
    """Prepend a batch axis of size 1 if `array` is unbatched.

    Args:
        array: Either an unbatched array, shape ``(N, d)``, or an already
            batched one, shape ``(batch, N, d)``.

    Returns:
        `array` with a leading batch axis: shape ``(1, N, d)`` if the input
        was 2D, otherwise `array` unchanged.

    Raises:
        TypeError: If `array` is not a ``numpy.ndarray``.
    """
    ensure_numpy_ndarray("array", array)
    return array[None] if array.ndim == 2 else array


def add_batch_dim_to_multiple(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """Apply `add_batch_dim` to each of `arrays`.

    Args:
        *arrays: Any number of 2D or 3D arrays.

    Returns:
        A tuple of the same length as `arrays`, each element batched.
    """
    return tuple(add_batch_dim(array) for array in arrays)


def cummean(array: np.ndarray, axis: int = 0) -> np.ndarray:
    """Cumulative mean of `array` along `axis`: ``result[k] = mean(array[:k+1])``.

    Args:
        array: The array to accumulate, any shape.
        axis: The axis along which to accumulate.

    Returns:
        An array of the same shape as `array`, holding the running mean.

    Raises:
        TypeError: If `array` is not a ``numpy.ndarray``.
    """
    ensure_numpy_ndarray("array", array)
    cumsum = np.cumsum(array, axis=axis)
    counts = np.arange(1, array.shape[axis] + 1)
    shape = [1] * array.ndim
    shape[axis] = -1
    return cumsum / counts.reshape(shape)


def stack_arrays(
    arrays: Sequence[np.ndarray | torch.Tensor], axis: int = 0
) -> np.ndarray | torch.Tensor:
    """Stack a sequence of same-type arrays along `axis`.

    Uses torch.stack when the sequence holds torch.Tensor (preserving the
    autograd graph -- np.stack would silently detach it via the tensor's
    __array__ protocol), or np.stack for plain numpy arrays. This lets
    StateSpaceSystem.run() work identically for numpy-based rollouts (e.g.
    RiccatiController) and torch-based rollouts (e.g. UnfoldedController).

    Args:
        arrays: A non-empty, homogeneously-typed list of ``numpy.ndarray`` or
            ``torch.Tensor``, each of the same shape.
        axis: The new axis along which to stack (torch's ``dim`` argument).

    Returns:
        The stacked array/tensor, with a new axis of length ``len(arrays)``
        inserted at `axis`; same type as `arrays[0]`.
    """
    if isinstance(arrays[0], torch.Tensor):
        return torch.stack(cast("list[torch.Tensor]", list(arrays)), dim=axis)
    return np.stack(cast("Sequence[np.ndarray]", arrays), axis=axis)


def match_array_type(
    reference: np.ndarray | torch.Tensor, array: np.ndarray
) -> np.ndarray | torch.Tensor:
    """Convert a plain numpy `array` (e.g. a system matrix slice from
    TimeSeriesMatrix) to match `reference`'s type, so it can be combined with
    `reference` via matrix ops without crashing or silently losing gradients.

    Mixing a grad-requiring torch.Tensor with a raw numpy array in the same
    expression raises RuntimeError (numpy tries to coerce the tensor via
    .numpy(), which torch refuses whenever requires_grad=True) -- this isn't
    just a silent-conversion nuisance, it's an outright crash for any
    genuinely learnable rollout.

    Args:
        reference: The tensor/array whose type (and, if a tensor, dtype and
            device) `array` should be converted to match.
        array: A plain ``numpy.ndarray`` to convert.

    Returns:
        A ``torch.Tensor`` matching `reference`'s dtype/device if `reference`
        is a ``torch.Tensor``; otherwise `array` unchanged.
    """
    if isinstance(reference, torch.Tensor):
        return torch.as_tensor(array, dtype=reference.dtype, device=reference.device)
    return array


def to_numpy(array: np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert to a plain numpy array, detaching from any autograd graph if
    `array` is a torch.Tensor. No-op if already numpy.

    Args:
        array: A ``numpy.ndarray`` or ``torch.Tensor``, any shape.

    Returns:
        A ``numpy.ndarray`` with the same values (and, for a CUDA tensor, moved
        to CPU first).
    """
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return array
