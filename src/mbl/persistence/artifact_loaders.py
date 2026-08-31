"""Strategy pattern for loading heavy artifact payloads back from disk --
the read-side mirror of `artifact_serializers.py`.

Each ArtifactLoader handles exactly one file suffix; ArtifactLoaderRegistry
picks the highest-priority registered loader whose `can_handle` matches.

Read-path policy (REFACTOR_PLAN v3, T3.i — binding):

* The **modern read path executes no code on load**: safetensors, ``.npz``/
  ``.npy`` (``allow_pickle=False``), CSV, and opaque image paths only. The
  former catch-all `PickleLoader` and the ``torch.load(weights_only=False)``
  `TorchLoader` are **quarantined**: they are no longer registered in the
  default registry and are reachable only through
  `legacy_artifact_loader_registry()`, which exists solely so the dashboard
  can still open grandfathered pre-v3 runs. The `ExperimentCache` (T3.e)
  never touches the legacy registry — its code-provenance stamp already
  guarantees pre-v3 artifacts are never served as current results.
"""

import json
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open

#: Priority the quarantined legacy loaders register at inside the LEGACY
#: registry, so they're always tried last regardless of registration order.
FALLBACK_PRIORITY = -1000


@runtime_checkable
class ArtifactLoader(Protocol):
    def can_handle(self, path: Path) -> bool:
        """Whether this loader knows how to read `path`."""
        ...

    def load(self, path: Path) -> Any:
        """Read and return the payload stored at `path`."""
        ...


class ArtifactLoaderRegistry:
    """Dispatches to the highest-priority registered loader that can handle
    the path (ties broken by registration order)."""

    def __init__(self) -> None:
        # Each entry: (priority, insertion_index, name, loader).
        self._entries: list[tuple[int, int, str, ArtifactLoader]] = []
        self._next_index = 0

    def register(self, name: str, loader: ArtifactLoader, *, priority: int = 0) -> None:
        self._entries.append((priority, self._next_index, name, loader))
        self._next_index += 1
        self._entries.sort(key=lambda entry: (-entry[0], entry[1]))

    def available(self) -> tuple[str, ...]:
        return tuple(name for _, _, name, _ in self._entries)

    def load(self, path: Path) -> Any:
        for _, _, _, loader in self._entries:
            if loader.can_handle(path):
                return loader.load(path)
        raise TypeError(
            f"No loader registered for artifact '{path}'; "
            f"available codecs: {self.available()}."
        )


#: Global registry, populated by the @register_loader decorators below as a
#: side effect of this module being imported.
ARTIFACT_LOADER_REGISTRY = ArtifactLoaderRegistry()


def register_loader(
    name: str, *, priority: int = 0
) -> Callable[[type[ArtifactLoader]], type[ArtifactLoader]]:
    """Class decorator: instantiate the decorated ArtifactLoader and register
    it under `name` in the global ARTIFACT_LOADER_REGISTRY. This is how new
    artifact formats get added without touching this file (Open/Closed
    Principle) -- define a class satisfying ArtifactLoader anywhere and
    decorate it; importing that module registers it."""

    def decorator(cls: type[ArtifactLoader]) -> type[ArtifactLoader]:
        ARTIFACT_LOADER_REGISTRY.register(name, cls(), priority=priority)
        return cls

    return decorator


@register_loader("numpy")
class NumpyArrayLoader:
    """Reads legacy ``.npy`` artifacts. ``allow_pickle`` stays at NumPy's
    ``False`` default, so this loader cannot execute code — it remains on
    the modern read path even though new writes ride ``.npz``."""

    def can_handle(self, path: Path) -> bool:
        return path.suffix == ".npy"

    def load(self, path: Path) -> np.ndarray:
        return cast(np.ndarray, np.load(path))


@register_loader("npz")
class NpzLoader:
    """Reads the compressed ``.npz`` artifacts `NumpyArraySerializer` writes:
    a single-member file stored under the canonical ``"array"`` key loads back
    as the bare array; a multi-member file loads as a name->array dict."""

    def can_handle(self, path: Path) -> bool:
        return path.suffix == ".npz"

    def load(self, path: Path) -> np.ndarray | dict[str, np.ndarray]:
        with np.load(path) as payload:
            members = {name: payload[name] for name in payload.files}
        if set(members) == {"array"}:
            return cast(np.ndarray, members["array"])
        return cast("dict[str, np.ndarray]", members)


@register_loader("safetensors")
class SafetensorsLoader:
    """Reads ``.safetensors`` artifacts, reconstructing the in-memory shape
    the file's ``payload`` metadata marker declares:

    * ``"tensor"`` — the bare tensor stored under the canonical key;
    * ``"state_dict"`` (or no marker) — a flat name->tensor dict;
    * ``"training_state"`` — a full `engine.training_plan.TrainingState`
      (module + optimizer state dicts, epoch), inverted from
      `TrainingStateSerializer`'s flat key scheme.
    """

    def can_handle(self, path: Path) -> bool:
        return path.suffix == ".safetensors"

    def load(self, path: Path) -> Any:
        with safe_open(path, framework="pt") as payload:
            metadata = payload.metadata() or {}
            tensors = {key: payload.get_tensor(key) for key in payload.keys()}
        kind = metadata.get("payload", "state_dict")
        if kind == "tensor":
            return tensors["tensor"]
        if kind == "training_state":
            return self._rebuild_training_state(tensors, metadata)
        return tensors

    @staticmethod
    def _rebuild_training_state(
        tensors: dict[str, torch.Tensor], metadata: dict[str, str]
    ) -> Any:
        from ..engine.training_plan import TrainingState

        structure = json.loads(metadata["structure"])
        module_state: dict[str, torch.Tensor] = {}
        optimizer_state: dict[int, dict[str, Any]] = {}
        for key, tensor in tensors.items():
            scope, _, rest = key.partition("/")
            if scope == "module":
                module_state[rest] = tensor
            else:  # "optimizer/<param_id>/<slot>"
                param_id, _, slot = rest.partition("/")
                optimizer_state.setdefault(int(param_id), {})[slot] = tensor
        for param_id, slots in structure["nontensor_state"].items():
            optimizer_state.setdefault(int(param_id), {}).update(slots)
        return TrainingState(
            epoch=int(structure["epoch"]),
            module_state_dict=module_state,
            optimizer_state_dict={
                "state": optimizer_state,
                "param_groups": structure["param_groups"],
            },
        )


@register_loader("dataframe")
class DataFrameLoader:
    def can_handle(self, path: Path) -> bool:
        return path.suffix == ".csv"

    def load(self, path: Path) -> pd.DataFrame:
        return pd.read_csv(path)


@register_loader("image")
class ImagePathLoader:
    """Rendered figures (.png) are opaque pixels, not reloadable data -- hand
    back the path itself so callers (e.g. the dashboard) can display it
    directly (`st.image(path)`) instead of failing to unpickle an image."""

    def can_handle(self, path: Path) -> bool:
        return path.suffix in {".png", ".jpg", ".jpeg"}

    def load(self, path: Path) -> Path:
        return path


class TorchLoader:
    """QUARANTINED legacy loader (T3.i): ``torch.load(weights_only=False)``
    is arbitrary-code-execution on load. Not registered in the default
    registry; reachable only through `legacy_artifact_loader_registry()`,
    solely so the dashboard can open grandfathered pre-v3 ``.pt`` runs."""

    def can_handle(self, path: Path) -> bool:
        return path.suffix == ".pt"

    def load(self, path: Path) -> Any:
        return torch.load(path, weights_only=False)


class PickleLoader:
    """QUARANTINED legacy loader (T3.i): unpickling executes arbitrary code.
    Not registered in the default registry; reachable only through
    `legacy_artifact_loader_registry()` for grandfathered pre-v3 runs."""

    def can_handle(self, path: Path) -> bool:
        return path.suffix == ".pkl"

    def load(self, path: Path) -> Any:
        with path.open("rb") as f:
            return pickle.load(f)


def default_artifact_loader_registry() -> ArtifactLoaderRegistry:
    """The modern read path: executes no code on load, ever. This is the
    only registry the `ExperimentCache` (T3.e) may read through."""
    return ARTIFACT_LOADER_REGISTRY


def legacy_artifact_loader_registry() -> ArtifactLoaderRegistry:
    """The modern registry plus the quarantined pickle-era loaders — for the
    dashboard's grandfathered pre-v3 runs ONLY. Never used by the
    `ExperimentCache`; new code must not adopt it (the T3.e provenance stamp
    already guarantees pre-v3 artifacts are never served as current results).

    Returns:
        A fresh registry: every modern loader, then `TorchLoader` and
        `PickleLoader` at fallback priority.
    """
    registry = ArtifactLoaderRegistry()
    for priority, index, name, loader in ARTIFACT_LOADER_REGISTRY._entries:
        registry.register(name, loader, priority=priority)
    registry.register("torch_legacy", TorchLoader(), priority=FALLBACK_PRIORITY)
    registry.register("pickle_legacy", PickleLoader(), priority=FALLBACK_PRIORITY)
    return registry
