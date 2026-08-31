"""Strategy pattern for saving heavy, heterogeneous artifact payloads to disk.

Each ArtifactSerializer handles exactly one payload type;
ArtifactSerializerRegistry picks the highest-priority registered serializer
whose `can_handle` matches.

Serialization policy (REFACTOR_PLAN v3, T3.i — binding):

* **Pickle is prohibited on the write path**, absolutely: no
  ``pickle.dump``, no ``torch.save`` (a zip-pickle envelope). The former
  catch-all `PickleSerializer` fallback is gone; an object no registered
  codec can handle is a loud `UnserializableArtifactError` naming the object
  and the available codecs — never a silent pickle.
* **Torch weights ride safetensors.** The in-memory contract stays
  ``nn.Module.state_dict()`` (T2.b); the on-disk representation is
  ``.safetensors`` — memory-mappable, zero-copy-loadable, incapable of
  executing code on load.
* **NumPy arrays ride compressed ``.npz``** (``np.savez_compressed``);
  structured records (DataFrames) stay CSV; scalars/metadata stay JSON in
  ``metadata.json``, as today.
"""

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from safetensors.torch import save_file


class UnserializableArtifactError(TypeError):
    """No registered codec can persist this object — the loud, typed error
    the T3.i policy mandates in place of the removed pickle fallback."""


@runtime_checkable
class ArtifactSerializer(Protocol):
    def can_handle(self, obj: Any) -> bool:
        """Whether this serializer knows how to persist `obj`."""
        ...

    def save(self, path: Path, obj: Any) -> Path:
        """Persist `obj` under `path` (without a required suffix) and return the
        actual file path written (with whatever suffix this format needs)."""
        ...


class ArtifactSerializerRegistry:
    """Dispatches to the highest-priority registered serializer that can handle
    the object (ties broken by registration order)."""

    def __init__(self) -> None:
        # Each entry: (priority, insertion_index, name, serializer).
        self._entries: list[tuple[int, int, str, ArtifactSerializer]] = []
        self._next_index = 0

    def register(
        self, name: str, serializer: ArtifactSerializer, *, priority: int = 0
    ) -> None:
        self._entries.append((priority, self._next_index, name, serializer))
        self._next_index += 1
        self._entries.sort(key=lambda entry: (-entry[0], entry[1]))

    def available(self) -> tuple[str, ...]:
        return tuple(name for _, _, name, _ in self._entries)

    def save(self, directory: Path, name: str, obj: Any) -> Path:
        for _, _, _, serializer in self._entries:
            if serializer.can_handle(obj):
                return serializer.save(directory / name, obj)
        raise UnserializableArtifactError(
            f"No serializer registered for artifact {name!r} of type "
            f"{type(obj).__name__}; available codecs: {self.available()}. "
            "Pickle fallback is prohibited (REFACTOR_PLAN v3, T3.i) — register "
            "a dedicated codec for this payload type instead."
        )


#: Global registry, populated by the @register_serializer decorators below as a
#: side effect of this module being imported.
ARTIFACT_SERIALIZER_REGISTRY = ArtifactSerializerRegistry()


def register_serializer(
    name: str, *, priority: int = 0
) -> Callable[[type[ArtifactSerializer]], type[ArtifactSerializer]]:
    """Class decorator: instantiate the decorated ArtifactSerializer and register
    it under `name` in the global ARTIFACT_SERIALIZER_REGISTRY. This is how new
    artifact formats get added without touching this file (Open/Closed
    Principle) -- define a class satisfying ArtifactSerializer anywhere and
    decorate it; importing that module registers it."""

    def decorator(cls: type[ArtifactSerializer]) -> type[ArtifactSerializer]:
        ARTIFACT_SERIALIZER_REGISTRY.register(name, cls(), priority=priority)
        return cls

    return decorator


@register_serializer("numpy")
class NumpyArraySerializer:
    """Dense NumPy arrays ride compressed ``.npz`` (T3.i) — never raw pickle,
    and smaller at rest than the former ``.npy`` for the trajectory-scale
    payloads the tracker persists. The single array is stored under the
    canonical key ``"array"`` so the paired loader can hand back the bare
    array instead of an `NpzFile` wrapper."""

    #: Canonical member name of a single-array `.npz` artifact.
    ARRAY_KEY = "array"

    def can_handle(self, obj: Any) -> bool:
        return isinstance(obj, np.ndarray)

    def save(self, path: Path, obj: np.ndarray) -> Path:
        out_path = path.with_suffix(".npz")
        np.savez_compressed(out_path, **{self.ARRAY_KEY: obj})  # type: ignore[arg-type]  # numpy stubs mistype **kwds
        return out_path


def dense_state_dict(state_dict: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Contiguous, detached CPU copies of a state dict's tensors — the shape
    ``safetensors.torch.save_file`` requires (it refuses live autograd views
    and shared storage).

    Non-strided (sparse) tensors are skipped with a WARNING naming the keys:
    safetensors stores dense strided layouts only, and the sparse entries in
    this tree's state dicts are specification-derived problem-data buffers
    (e.g. cvxpylayers' compiled constraint matrices), reconstructible from
    the signable spec — never trained weights.
    """
    dense = {
        key: value.detach().cpu().contiguous()
        for key, value in state_dict.items()
        if isinstance(value, torch.Tensor) and value.layout == torch.strided
    }
    skipped = [
        key
        for key, value in state_dict.items()
        if isinstance(value, torch.Tensor) and value.layout != torch.strided
    ]
    if skipped:
        logging.getLogger(__name__).warning(
            "Skipping %d non-strided (sparse) state-dict entrie(s) %s: "
            "safetensors persists dense tensors only; these are "
            "specification-derived buffers, not trained weights.",
            len(skipped),
            skipped,
        )
    return dense


@register_serializer("safetensors")
class SafetensorsSerializer:
    """Torch payloads ride ``.safetensors`` (T3.i): a bare tensor is stored
    under the canonical ``"tensor"`` key, an ``nn.Module`` as its
    ``state_dict()``. A ``payload`` metadata marker tells the paired loader
    which in-memory shape to reconstruct."""

    #: Canonical member name of a single-tensor `.safetensors` artifact.
    TENSOR_KEY = "tensor"

    def can_handle(self, obj: Any) -> bool:
        return isinstance(obj, torch.Tensor | torch.nn.Module)

    def save(self, path: Path, obj: torch.Tensor | torch.nn.Module) -> Path:
        out_path = path.with_suffix(".safetensors")
        if isinstance(obj, torch.nn.Module):
            tensors = dense_state_dict(dict(obj.state_dict()))
            metadata = {"payload": "state_dict"}
        else:
            tensors = {self.TENSOR_KEY: obj.detach().cpu().contiguous()}
            metadata = {"payload": "tensor"}
        save_file(tensors, out_path, metadata=metadata)
        return out_path


@register_serializer("training_state")
class TrainingStateSerializer:
    """`engine.training_plan.TrainingState` snapshots ride one
    ``.safetensors`` file (T3.i, T3.b resumability seam): every tensor of the
    module and optimizer state dicts is stored flat under a path-prefixed key
    (``module/...`` / ``optimizer/<param_id>/<slot>``), and the non-tensor
    remainder (epoch, param groups, scalar optimizer state such as Adam's
    ``step`` counts) rides the file's JSON string metadata — one file, zero
    pickle bytes, reconstructed exactly by the paired loader."""

    def can_handle(self, obj: Any) -> bool:
        # Imported lazily: persistence must not import the engine at module
        # scope (the engine's callbacks already import this package).
        from ..engine.training_plan import TrainingState

        return isinstance(obj, TrainingState)

    def save(self, path: Path, obj: Any) -> Path:
        out_path = path.with_suffix(".safetensors")
        tensors: dict[str, torch.Tensor] = {
            f"module/{key}": value
            for key, value in dense_state_dict(dict(obj.module_state_dict)).items()
        }
        optimizer_state = dict(obj.optimizer_state_dict)
        nontensor_state: dict[str, dict[str, Any]] = {}
        for param_id, slots in dict(optimizer_state.get("state", {})).items():
            for slot, value in dict(slots).items():
                if isinstance(value, torch.Tensor):
                    tensors[f"optimizer/{param_id}/{slot}"] = (
                        value.detach().cpu().contiguous()
                    )
                else:
                    nontensor_state.setdefault(str(param_id), {})[slot] = value
        metadata = {
            "payload": "training_state",
            "structure": json.dumps(
                {
                    "epoch": obj.epoch,
                    "param_groups": optimizer_state.get("param_groups", []),
                    "nontensor_state": nontensor_state,
                },
                default=str,
            ),
        }
        save_file(tensors, out_path, metadata=metadata)
        return out_path


@register_serializer("dataframe")
class DataFrameSerializer:
    def can_handle(self, obj: Any) -> bool:
        return isinstance(obj, pd.DataFrame)

    def save(self, path: Path, obj: pd.DataFrame) -> Path:
        out_path = path.with_suffix(".csv")
        obj.to_csv(out_path, index=False)
        return out_path


@register_serializer("matplotlib_figure")
class MatplotlibFigureSerializer:
    def can_handle(self, obj: Any) -> bool:
        return isinstance(obj, Figure)

    def save(self, path: Path, obj: Figure) -> Path:
        out_path = path.with_suffix(".png")
        obj.savefig(out_path)
        return out_path


def default_artifact_serializer_registry() -> ArtifactSerializerRegistry:
    """Return the shared global registry (already populated by the
    @register_serializer decorators above, since this module defines them)."""
    return ARTIFACT_SERIALIZER_REGISTRY
