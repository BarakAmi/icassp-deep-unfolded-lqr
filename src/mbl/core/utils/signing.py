"""Polymorphic experiment signing (Phase 1D): a structural `Signable` protocol
any problem-defining object (System, Cost, Constraint, Controller, Distribution)
implements via `get_signature() -> dict`, plus the primitives that turn such a
signature tree into (a) content hashes for tensors/arrays, (b) a flat,
collision-proof set of persisted keys, and (c) one composite identity digest.

Design invariant (load-bearing -- see docs/planning/01_foundation/phase_1d_persistence_plan_v2.md
Part 4/5): signatures cover *specification-time* tensors (system matrices, cost
matrices, distribution parameters) never *live/trained* parameter values. The
latter already have a home as callback-persisted artifacts (e.g.
`ParameterSnapshotCallback`); signing them here would also reintroduce real
cross-version/GPU nondeterminism that specification-time NumPy-constructed
tensors do not have.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch

from .array_ops import to_numpy
from .guards import array_content_key


@runtime_checkable
class Signable(Protocol):
    """Structural protocol: any object that can describe its own mathematical
    configuration as a small, JSON-serializable, nested dict -- never raw
    tensors, only shapes/dtypes/content-hashes and scalar hyperparameters."""

    def get_signature(self) -> dict[str, Any]: ...


def hash_array(array: np.ndarray | torch.Tensor, *, algorithm: str = "sha256") -> str:
    """Content hash of an array/tensor's exact numeric value.

    ``np.ascontiguousarray`` (applied inside `array_content_key`) is not
    optional: a transposed view (e.g. ``A.T``) or a sliced view (e.g.
    ``A[::2]``) shares its parent's underlying buffer with different strides,
    so hashing raw bytes without first forcing a C-contiguous copy would hash
    a byte layout that doesn't correspond to the logical element order.

    Args:
        array: A ``numpy.ndarray`` or ``torch.Tensor`` (converted via
            `to_numpy`, detaching any autograd graph, before hashing).
        algorithm: A `hashlib`-supported digest name.

    Returns:
        ``f"{algorithm}:{hexdigest}"``, e.g. ``"sha256:ab12ef..."``.
    """
    arr = to_numpy(array)
    shape, dtype_str, raw_bytes = array_content_key(arr)
    hasher = hashlib.new(algorithm)
    hasher.update(str(shape).encode())
    hasher.update(dtype_str.encode())
    hasher.update(raw_bytes)
    return f"{algorithm}:{hasher.hexdigest()}"


def flatten_signature(tree: Mapping[str, Any], *, prefix: str = "") -> dict[str, Any]:
    """Recursively flatten a nested `get_signature()` tree into dot-joined keys.

    ``{"system": {"A_hash": "..."}}`` -> ``{"system.A_hash": "..."}``. Lists
    (e.g. ``constraints``) flatten with an integer index segment:
    ``{"constraints": [{"type": "Box"}]}`` -> ``{"constraints.0.type": "Box"}``.

    This is the single mechanism guaranteeing collision-proof persisted keys:
    no `get_signature()` implementation ever hand-prefixes its own field
    names, so it is structurally impossible for two components to disagree on
    a naming convention -- a naive ``{**a, **b}`` merge of two nested trees
    (which would silently drop any key both share, e.g. ``"type"``) is exactly
    the failure mode this function exists to make unnecessary.

    Args:
        tree: A nested signature dict (or sub-dict, for the recursive case).
        prefix: Dot-terminated path prefix already accumulated (internal use).

    Returns:
        A flat dict with dot-joined keys; every leaf value from `tree` is
        preserved under a unique path.
    """
    flat: dict[str, Any] = {}
    for key, value in tree.items():
        path = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(flatten_signature(value, prefix=f"{path}."))
        elif isinstance(value, (list, tuple)):
            if not value:
                flat[path] = []  # preserve "zero constraints" as an explicit leaf
                continue
            for i, item in enumerate(value):
                if isinstance(item, Mapping):
                    flat.update(flatten_signature(item, prefix=f"{path}.{i}."))
                else:
                    flat[f"{path}.{i}"] = item
        else:
            flat[path] = value
    return flat


def compute_signature_digest(signature: Mapping[str, Any]) -> str:
    """Compact composite identity for a whole `get_signature()` tree.

    Args:
        signature: The nested (pre-flatten) signature tree.

    Returns:
        A 16-hex-character SHA256 prefix of the tree's canonical
        (``sort_keys=True``) JSON encoding -- stable regardless of key
        insertion order, suitable as a human-shareable, dedup-friendly
        identifier (not a security boundary; nothing here is adversarial).
    """
    canonical = json.dumps(signature, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def compute_run_signature(
    problem: Signable,
    controller: Signable,
    distributions: Mapping[str, Signable] | None = None,
) -> str:
    """The exact composite digest `ProblemSignatureCallback` persists under
    the ``"signature"`` param key -- extracted so a caller can compute the
    *expected* signature of a not-yet-run configuration (e.g. to decide
    whether a cached run is stale) without duplicating the signature-tree
    assembly logic in two places (Phase 1F).

    Args:
        problem: The `OptimalControlProblem` a run solves.
        controller: The `Controller` a run evaluates/trains.
        distributions: Named sampling distributions, as passed to
            `ProblemSignatureCallback`.

    Returns:
        The same composite digest `ProblemSignatureCallback.on_train_start`
        persists for an identical `(problem, controller, distributions)`.
    """
    signature_tree = {
        "problem": problem.get_signature(),
        "controller": controller.get_signature(),
        "sampling": {
            name: dist.get_signature() for name, dist in (distributions or {}).items()
        },
    }
    return compute_signature_digest(signature_tree)
