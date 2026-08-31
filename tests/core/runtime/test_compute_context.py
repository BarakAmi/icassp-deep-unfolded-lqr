"""Stage S1 acceptance tests for `ComputeContext` (T1.b).

Hardware-dependent paths are exercised by monkeypatching the CUDA probes,
so this suite is deterministic and passes identically on GPU-less hosts
(the plan's first-class-GPU-less mandate, §7.9).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from mbl.core.runtime import (
    Backend,
    ComputeContext,
    DeviceUnavailableError,
    InvalidComputeContextError,
    Precision,
    default_compute_context,
)
from mbl.core.runtime.compute_context import require_available_device


class TestConstructionAndNormalization:
    def test_default_is_numpy_cpu_float64(self):
        ctx = ComputeContext()
        assert ctx.backend is Backend.NUMPY
        assert ctx.device == "cpu"
        assert ctx.precision is Precision.FLOAT64

    def test_string_inputs_normalize_to_enums(self):
        ctx = ComputeContext(backend="torch", device="cpu", precision="float32")
        assert ctx.backend is Backend.TORCH
        assert ctx.precision is Precision.FLOAT32

    def test_unknown_backend_rejected(self):
        with pytest.raises(ValueError):
            ComputeContext(backend="jax")

    def test_unknown_precision_rejected(self):
        with pytest.raises(ValueError):
            ComputeContext(precision="float16")

    def test_context_is_frozen(self):
        ctx = ComputeContext()
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.device = "cuda"  # type: ignore[misc]

    def test_default_compute_context_is_a_fresh_default_each_call(self):
        first, second = default_compute_context(), default_compute_context()
        assert first == ComputeContext()
        assert first == second
        assert first is not second  # a function, never a mutable singleton


class TestFailFastHardwareResolution:
    def test_numpy_backend_rejects_accelerator_placement(self):
        with pytest.raises(InvalidComputeContextError, match="CPU only"):
            ComputeContext(backend="numpy", device="cuda")

    def test_device_outside_declared_envelope_rejected(self):
        with pytest.raises(InvalidComputeContextError, match="envelope"):
            ComputeContext(backend="torch", device="mps")

    def test_unparseable_device_rejected(self):
        with pytest.raises(InvalidComputeContextError, match="Unparseable"):
            ComputeContext(backend="torch", device="not a device!")

    def test_bare_cuda_canonicalizes_without_asking_for_hardware(
        self, monkeypatch
    ) -> None:
        """D17 as extended (Annex 06 §3.5), and the property the whole split
        rests on: this value SIGNS, so deriving it from the reading machine
        would make `ModelID` machine-dependent.

        Asserted with CUDA masked ABSENT — the old contract raised here.
        """
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert ComputeContext(backend="torch", device="cuda").device == "cuda:0"

    def test_the_normalisation_is_the_same_with_and_without_hardware(
        self, monkeypatch
    ) -> None:
        """The identity claim, stated as a comparison rather than a constant:
        two hosts must sign one document identically."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
        present = ComputeContext(backend="torch", device="cuda").get_signature()
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        absent = ComputeContext(backend="torch", device="cuda").get_signature()
        assert present == absent

    def test_an_out_of_range_index_still_normalises_at_specification(
        self, monkeypatch
    ) -> None:
        """A document may name a device this host does not have. Reading it is
        not running it, and the refusal is `require_available_device`'s."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        assert ComputeContext(backend="torch", device="cuda:3").device == "cuda:3"


class TestTheCapabilityProbe:
    """`require_available_device` — the half that needs hardware.

    The refusal was RELOCATED, not weakened, so these are the assertions the
    two deleted construction-time tests used to make, at the seam that now
    owns them. Losing them would be exactly the "declaration that takes no
    effect" this project keeps finding.
    """

    def test_cuda_without_hardware_is_refused(self, monkeypatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with pytest.raises(DeviceUnavailableError, match="CUDA is not available"):
            require_available_device("cuda:0")

    def test_an_index_beyond_the_device_count_is_refused(self, monkeypatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        with pytest.raises(DeviceUnavailableError, match="valid indices"):
            require_available_device("cuda:3")

    def test_an_available_device_passes(self, monkeypatch) -> None:
        """Anti-vacuity: a probe that refused everything would pass both
        assertions above and be useless."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
        require_available_device("cuda:1")

    def test_cpu_is_never_probed(self, monkeypatch) -> None:
        """A CPU context must not consult CUDA at all — detonate if it does."""

        def _detonate() -> bool:
            raise AssertionError("a CPU device asked whether CUDA was available")

        monkeypatch.setattr(torch.cuda, "is_available", _detonate)
        require_available_device("cpu")


class TestExplicitIngressFallback:
    def test_resolve_without_fallback_propagates_unavailability(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with pytest.raises(DeviceUnavailableError):
            ComputeContext.resolve("torch", "cuda")

    def test_resolve_with_fallback_yields_effective_cpu_context(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        ctx = ComputeContext.resolve("torch", "cuda", "float32", fallback_to_cpu=True)
        assert ctx == ComputeContext("torch", "cpu", "float32")
        # The effective (resolved) context is what signs.
        assert ctx.get_signature()["device"] == "cpu"

    def test_fallback_never_masks_structural_invalidity(self):
        with pytest.raises(InvalidComputeContextError):
            ComputeContext.resolve("numpy", "cuda", fallback_to_cpu=True)


class TestSigning:
    def test_signature_is_flat_and_complete(self):
        signature = ComputeContext("torch", "cpu", "float32").get_signature()
        assert signature == {
            "type": "ComputeContext",
            "backend": "torch",
            "device": "cpu",
            "precision": "float32",
        }

    def test_equal_contexts_sign_identically(self):
        assert (
            ComputeContext().get_signature()
            == ComputeContext("numpy", "cpu", "float64").get_signature()
        )


class TestTypedAccessorsAndArrayAuthorship:
    def test_dtype_and_device_accessors(self):
        ctx = ComputeContext("torch", "cpu", "float32")
        assert ctx.numpy_dtype == np.dtype(np.float32)
        assert ctx.torch_dtype is torch.float32
        assert ctx.torch_device == torch.device("cpu")

    def test_numpy_authorship(self):
        ctx = ComputeContext()
        array = ctx.asarray([[1, 2], [3, 4]])
        assert isinstance(array, np.ndarray)
        assert array.dtype == np.float64
        zeros = ctx.zeros((3, 2))
        assert isinstance(zeros, np.ndarray)
        assert zeros.shape == (3, 2) and zeros.dtype == np.float64

    def test_torch_authorship(self):
        ctx = ComputeContext("torch", "cpu", "float32")
        tensor = ctx.asarray(np.arange(6.0).reshape(2, 3))
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype is torch.float32
        assert tensor.device == torch.device("cpu")
        zeros = ctx.zeros((4,))
        assert isinstance(zeros, torch.Tensor)
        assert zeros.dtype is torch.float32
