"""Read-path tests for the T3.i serialization policy (Stage S4): the modern
registry executes no code on load (safetensors/.npz/.npy/CSV/images only);
the pickle-era loaders are quarantined into the legacy registry, reachable
solely by the dashboard for grandfathered pre-v3 runs."""

import pickle

import numpy as np
import pandas as pd
import pytest
import torch

from mbl.persistence.artifact_loaders import (
    ARTIFACT_LOADER_REGISTRY,
    FALLBACK_PRIORITY,
    ArtifactLoader,
    ArtifactLoaderRegistry,
    DataFrameLoader,
    ImagePathLoader,
    NpzLoader,
    NumpyArrayLoader,
    PickleLoader,
    SafetensorsLoader,
    TorchLoader,
    default_artifact_loader_registry,
    legacy_artifact_loader_registry,
    register_loader,
)
from mbl.persistence.artifact_serializers import (
    DataFrameSerializer,
    MatplotlibFigureSerializer,
    NumpyArraySerializer,
    SafetensorsSerializer,
)


def test_loaders_satisfy_the_protocol() -> None:
    for loader in (
        NumpyArrayLoader(),
        NpzLoader(),
        SafetensorsLoader(),
        DataFrameLoader(),
        ImagePathLoader(),
        TorchLoader(),
        PickleLoader(),
    ):
        assert isinstance(loader, ArtifactLoader)


def test_npz_loader_round_trips_with_serializer(tmp_path) -> None:
    array = np.arange(24, dtype=np.float64).reshape(4, 6)
    out_path = NumpyArraySerializer().save(tmp_path / "trajectory", array)

    loader = NpzLoader()
    assert loader.can_handle(out_path)
    assert np.array_equal(loader.load(out_path), array)


def test_npy_loader_still_reads_legacy_arrays(tmp_path) -> None:
    """Pre-S4 runs persisted bare .npy; those stay readable on the MODERN
    path (np.load's allow_pickle stays False -- no code execution)."""
    array = np.linspace(0.0, 1.0, 7)
    out_path = tmp_path / "legacy.npy"
    np.save(out_path, array)

    loader = NumpyArrayLoader()
    assert loader.can_handle(out_path)
    assert np.array_equal(loader.load(out_path), array)


def test_safetensors_loader_round_trips_tensor_and_state_dict(tmp_path) -> None:
    tensor = torch.linspace(0.0, 1.0, steps=32)
    tensor_path = SafetensorsSerializer().save(tmp_path / "weights", tensor)
    loader = SafetensorsLoader()
    assert loader.can_handle(tensor_path)
    assert torch.equal(loader.load(tensor_path), tensor)

    module = torch.nn.Linear(2, 2)
    module_path = SafetensorsSerializer().save(tmp_path / "model", module)
    restored = loader.load(module_path)
    assert restored.keys() == module.state_dict().keys()
    module.load_state_dict(restored)  # a real, loadable state dict


def test_dataframe_loader_round_trips_with_serializer(tmp_path) -> None:
    df = pd.DataFrame({"epoch": range(10), "loss": np.linspace(1.0, 0.1, 10)})
    out_path = DataFrameSerializer().save(tmp_path / "history", df)

    loader = DataFrameLoader()
    assert loader.can_handle(out_path)
    pd.testing.assert_frame_equal(loader.load(out_path), df)


def test_image_path_loader_returns_the_path_unchanged(tmp_path) -> None:
    from matplotlib.figure import Figure

    fig = Figure()
    ax = fig.add_subplot()
    ax.plot([0, 1, 2], [0, 1, 4])
    out_path = MatplotlibFigureSerializer().save(tmp_path / "plot", fig)

    loader = ImagePathLoader()
    assert loader.can_handle(out_path)
    assert loader.load(out_path) == out_path


class _Arbitrary:
    """Module-level (not nested) so it's actually picklable."""

    def __init__(self, value):
        self.value = value


def test_modern_registry_refuses_pickle_and_pt_files(tmp_path) -> None:
    """The read-path quarantine (T3.i): the default registry has NO loader
    for .pkl or .pt -- pre-v3 pickle-era artifacts are unreachable from the
    modern path (and hence from the ExperimentCache)."""
    registry = default_artifact_loader_registry()

    pkl_path = tmp_path / "blob.pkl"
    with pkl_path.open("wb") as f:
        pickle.dump(_Arbitrary(7), f)
    with pytest.raises(TypeError, match="No loader registered"):
        registry.load(pkl_path)

    pt_path = tmp_path / "weights.pt"
    torch.save(torch.ones(2), pt_path)
    with pytest.raises(TypeError, match="No loader registered"):
        registry.load(pt_path)


def test_legacy_registry_still_opens_grandfathered_artifacts(tmp_path) -> None:
    """The quarantined legacy registry -- the dashboard's escape hatch for
    pre-v3 runs -- reads .pkl/.pt AND everything modern."""
    registry = legacy_artifact_loader_registry()

    pkl_path = tmp_path / "blob.pkl"
    with pkl_path.open("wb") as f:
        pickle.dump(_Arbitrary(7), f)
    assert registry.load(pkl_path).value == 7

    pt_path = tmp_path / "weights.pt"
    torch.save(torch.ones(2), pt_path)
    assert torch.equal(registry.load(pt_path), torch.ones(2))

    npz_path = NumpyArraySerializer().save(tmp_path / "arr", np.zeros(3))
    assert np.array_equal(registry.load(npz_path), np.zeros(3))


def test_registry_dispatches_to_the_matching_loader(tmp_path) -> None:
    registry = default_artifact_loader_registry()

    array_path = NumpyArraySerializer().save(tmp_path / "arr", np.zeros(3))
    assert np.array_equal(registry.load(array_path), np.zeros(3))

    df_path = DataFrameSerializer().save(tmp_path / "df", pd.DataFrame({"x": [1]}))
    pd.testing.assert_frame_equal(registry.load(df_path), pd.DataFrame({"x": [1]}))

    tensor_path = SafetensorsSerializer().save(tmp_path / "w", torch.ones(2))
    assert torch.equal(registry.load(tensor_path), torch.ones(2))


def test_priority_ordering_is_respected_regardless_of_registration_order(
    tmp_path,
) -> None:
    """Even if a catch-all is registered FIRST, a low-priority fallback must
    still be tried last -- proving dispatch order depends on `priority`, not on
    import/registration order."""
    registry = ArtifactLoaderRegistry()
    registry.register("pickle", PickleLoader(), priority=FALLBACK_PRIORITY)
    registry.register("npz", NpzLoader())  # default priority 0

    out_path = NumpyArraySerializer().save(tmp_path / "arr", np.ones(2))
    assert np.array_equal(registry.load(out_path), np.ones(2))


def test_available_lists_registered_names_in_priority_order() -> None:
    registry = ArtifactLoaderRegistry()
    registry.register("pickle", PickleLoader(), priority=FALLBACK_PRIORITY)
    registry.register("numpy", NumpyArrayLoader())
    registry.register("torch", TorchLoader())

    assert registry.available() == ("numpy", "torch", "pickle")


def test_empty_registry_raises_type_error(tmp_path) -> None:
    registry = ArtifactLoaderRegistry()
    with pytest.raises(TypeError, match="No loader registered"):
        registry.load(tmp_path / "anything.npy")


class _MarkerLoaderPayload:
    """A sentinel type persisted with a bespoke suffix, used to prove a new
    loader can be added purely via @register_loader (OCP)."""


@register_loader("_test_ocp_demo_marker_loader")
class _MarkerLoader:
    def can_handle(self, path) -> bool:
        return path.suffix == ".marker"

    def load(self, path):
        return path.read_text()


def test_register_loader_decorator_extends_global_registry_without_editing_it(
    tmp_path,
) -> None:
    assert "_test_ocp_demo_marker_loader" in ARTIFACT_LOADER_REGISTRY.available()

    marker_path = tmp_path / "thing.marker"
    marker_path.write_text("marker")

    assert ARTIFACT_LOADER_REGISTRY.load(marker_path) == "marker"


def test_register_loader_decorator_returns_class_unchanged(tmp_path) -> None:
    loader = _MarkerLoader()
    marker_path = tmp_path / "thing.marker"
    marker_path.write_text("hi")
    assert loader.can_handle(marker_path)
    assert not loader.can_handle(tmp_path / "thing.npy")
