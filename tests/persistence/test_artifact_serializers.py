"""Write-path tests for the T3.i serialization policy (Stage S4): safetensors
for torch payloads, compressed .npz for NumPy arrays, CSV for DataFrames --
and NO pickle fallback: an unhandled object is a loud, typed error."""

import numpy as np
import pandas as pd
import pytest
import torch
from matplotlib.figure import Figure
from safetensors import safe_open
from safetensors.torch import load_file

from mbl.engine.training_plan import TrainingState
from mbl.persistence.artifact_serializers import (
    ARTIFACT_SERIALIZER_REGISTRY,
    ArtifactSerializer,
    ArtifactSerializerRegistry,
    DataFrameSerializer,
    MatplotlibFigureSerializer,
    NumpyArraySerializer,
    SafetensorsSerializer,
    TrainingStateSerializer,
    UnserializableArtifactError,
    default_artifact_serializer_registry,
    register_serializer,
)


def test_serializers_satisfy_the_protocol() -> None:
    for serializer in (
        NumpyArraySerializer(),
        SafetensorsSerializer(),
        TrainingStateSerializer(),
        DataFrameSerializer(),
        MatplotlibFigureSerializer(),
    ):
        assert isinstance(serializer, ArtifactSerializer)


def test_numpy_array_serializer_writes_compressed_npz(tmp_path) -> None:
    serializer = NumpyArraySerializer()
    array = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert serializer.can_handle(array)

    out_path = serializer.save(tmp_path / "matrix", array)
    assert out_path == tmp_path / "matrix.npz"
    with np.load(out_path) as payload:
        assert np.array_equal(payload["array"], array)


def test_safetensors_serializer_round_trips_a_tensor(tmp_path) -> None:
    serializer = SafetensorsSerializer()
    tensor = torch.tensor([1.0, 2.0, 3.0])
    assert serializer.can_handle(tensor)

    out_path = serializer.save(tmp_path / "weights", tensor)
    assert out_path == tmp_path / "weights.safetensors"
    assert torch.equal(load_file(out_path)["tensor"], tensor)
    with safe_open(out_path, framework="pt") as f:
        assert f.metadata()["payload"] == "tensor"


def test_safetensors_serializer_saves_module_state_dict(tmp_path) -> None:
    serializer = SafetensorsSerializer()
    module = torch.nn.Linear(2, 1)
    assert serializer.can_handle(module)

    out_path = serializer.save(tmp_path / "model", module)
    loaded = load_file(out_path)
    assert loaded.keys() == module.state_dict().keys()
    for key, value in module.state_dict().items():
        assert torch.equal(loaded[key], value)


def test_training_state_serializer_round_trips_without_pickle(tmp_path) -> None:
    """A full trained-module + Adam optimizer snapshot survives one
    safetensors file: tensors bit-exact, non-tensor optimizer state (step
    counts, param groups) intact -- the T3.b resumability seam on T3.i disk."""
    module = torch.nn.Linear(3, 2)
    optimizer = torch.optim.Adam(module.parameters(), lr=0.01)
    loss = module(torch.randn(8, 3)).square().mean()
    loss.backward()
    optimizer.step()  # populate exp_avg/exp_avg_sq/step slots

    state = TrainingState(
        epoch=7,
        module_state_dict=module.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
    )
    serializer = TrainingStateSerializer()
    assert serializer.can_handle(state)
    out_path = serializer.save(tmp_path / "training_state", state)
    assert out_path.suffix == ".safetensors"

    from mbl.persistence.artifact_loaders import SafetensorsLoader

    restored = SafetensorsLoader().load(out_path)
    assert isinstance(restored, TrainingState)
    assert restored.epoch == 7
    for key, value in module.state_dict().items():
        assert torch.equal(restored.module_state_dict[key], value)
    original_opt = optimizer.state_dict()
    # JSON canonicalizes tuples to lists (e.g. Adam's `betas`); torch's
    # load_state_dict accepts either, so compare canonicalized structures.
    import json

    assert json.loads(
        json.dumps(restored.optimizer_state_dict["param_groups"])
    ) == json.loads(json.dumps(original_opt["param_groups"]))
    for param_id, slots in original_opt["state"].items():
        restored_slots = restored.optimizer_state_dict["state"][param_id]
        for slot, value in slots.items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(restored_slots[slot], value)
            else:
                assert restored_slots[slot] == value

    # The restored snapshot actually resumes: a fresh module/optimizer pair
    # accepts both state dicts.
    fresh_module = torch.nn.Linear(3, 2)
    fresh_module.load_state_dict(restored.module_state_dict)
    fresh_optimizer = torch.optim.Adam(fresh_module.parameters(), lr=0.01)
    fresh_optimizer.load_state_dict(restored.optimizer_state_dict)


def test_dataframe_serializer_round_trips(tmp_path) -> None:
    serializer = DataFrameSerializer()
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    assert serializer.can_handle(df)

    out_path = serializer.save(tmp_path / "table", df)
    assert out_path == tmp_path / "table.csv"
    loaded = pd.read_csv(out_path)
    pd.testing.assert_frame_equal(loaded, df)


def test_matplotlib_figure_serializer_writes_png(tmp_path) -> None:
    serializer = MatplotlibFigureSerializer()
    fig = Figure()
    ax = fig.add_subplot()
    ax.plot([0, 1], [0, 1])
    assert serializer.can_handle(fig)

    out_path = serializer.save(tmp_path / "plot", fig)
    assert out_path == tmp_path / "plot.png"
    assert out_path.exists()
    assert out_path.stat().st_size > 0


class _Arbitrary:
    """An object no registered codec handles."""

    def __init__(self, value):
        self.value = value


def test_no_pickle_law_unhandled_objects_raise_a_loud_typed_error(tmp_path) -> None:
    """The T3.i write-path prohibition: the registry no longer swallows
    arbitrary objects into .pkl -- it raises a typed error naming the object
    and the available codecs."""
    registry = default_artifact_serializer_registry()
    with pytest.raises(UnserializableArtifactError, match="_Arbitrary"):
        registry.save(tmp_path, "blob", _Arbitrary(5))
    with pytest.raises(UnserializableArtifactError, match="available codecs"):
        registry.save(tmp_path, "obj", {"nested": "dict"})
    assert not list(tmp_path.iterdir())  # nothing silently written


def test_no_pickle_serializer_is_registered() -> None:
    assert all(
        "pickle" not in name
        for name in default_artifact_serializer_registry().available()
    )


def test_registry_dispatches_to_the_matching_serializer(tmp_path) -> None:
    registry = default_artifact_serializer_registry()

    array_path = registry.save(tmp_path, "arr", np.zeros(3))
    assert array_path.suffix == ".npz"

    df_path = registry.save(tmp_path, "df", pd.DataFrame({"x": [1]}))
    assert df_path.suffix == ".csv"

    tensor_path = registry.save(tmp_path, "w", torch.ones(2))
    assert tensor_path.suffix == ".safetensors"


def test_priority_ordering_is_respected_regardless_of_registration_order(
    tmp_path,
) -> None:
    """A low-priority catch-all registered FIRST must still be tried last --
    dispatch order depends on `priority`, not import/registration order."""

    class _CatchAll:
        def can_handle(self, obj) -> bool:
            return True

        def save(self, path, obj):
            out_path = path.with_suffix(".catchall")
            out_path.write_text("catchall")
            return out_path

    registry = ArtifactSerializerRegistry()
    registry.register("catchall", _CatchAll(), priority=-1000)
    registry.register("numpy", NumpyArraySerializer())  # default priority 0

    out_path = registry.save(tmp_path, "arr", np.ones(2))
    assert out_path.suffix == ".npz"  # not .catchall, despite registering first


def test_available_lists_registered_names_in_priority_order() -> None:
    registry = ArtifactSerializerRegistry()
    registry.register("low", NumpyArraySerializer(), priority=-1000)
    registry.register("numpy", NumpyArraySerializer())
    registry.register("safetensors", SafetensorsSerializer())

    assert registry.available() == ("numpy", "safetensors", "low")


def test_empty_registry_raises_type_error(tmp_path) -> None:
    registry = ArtifactSerializerRegistry()
    with pytest.raises(TypeError, match="No serializer registered"):
        registry.save(tmp_path, "anything", object())


class _MarkerPayload:
    """A sentinel type with no built-in serializer, used to prove a new
    serializer can be added purely via @register_serializer (OCP)."""


@register_serializer("_test_ocp_demo_marker")
class _MarkerSerializer:
    def can_handle(self, obj) -> bool:
        return isinstance(obj, _MarkerPayload)

    def save(self, path, obj):
        out_path = path.with_suffix(".marker")
        out_path.write_text("marker")
        return out_path


def test_register_serializer_decorator_extends_global_registry_without_editing_it(
    tmp_path,
) -> None:
    assert "_test_ocp_demo_marker" in ARTIFACT_SERIALIZER_REGISTRY.available()

    out_path = ARTIFACT_SERIALIZER_REGISTRY.save(tmp_path, "thing", _MarkerPayload())
    assert out_path == tmp_path / "thing.marker"
    assert out_path.read_text() == "marker"


def test_register_serializer_decorator_returns_class_unchanged() -> None:
    serializer = _MarkerSerializer()
    assert serializer.can_handle(_MarkerPayload())
    assert not serializer.can_handle(object())
