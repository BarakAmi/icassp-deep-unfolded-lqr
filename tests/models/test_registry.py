import pytest

from mbl.models.registry import (
    MODEL_REGISTRY,
    ControllerRegistry,
    build_default_controller_registry,
    register_model,
)
from mbl.models.analytic.riccati import RiccatiController
from mbl.models.neural.nerual import NeuralPolicy
from mbl.models.unfolded.base import UnfoldedController


class _DummyController:
    def __init__(self, value: int) -> None:
        self.value = value


# Module-level (executes exactly once per test session) so the real
# @register_model decorator is exercised without polluting other tests via
# re-registration attempts.
@register_model("_test_ocp_demo_controller")
class _OcpDemoController:
    """Proves a brand-new controller family can be added via the decorator
    alone -- no edits to registry.py needed (Open/Closed Principle)."""

    def __init__(self, value: int) -> None:
        self.value = value


def test_registry_register_create_and_list() -> None:
    registry: ControllerRegistry = ControllerRegistry()
    registry.register("dummy", _DummyController)

    created = registry.create("dummy", value=7)
    assert created.value == 7
    assert registry.available() == ("dummy",)


def test_registry_rejects_duplicate_registration() -> None:
    registry: ControllerRegistry = ControllerRegistry()
    registry.register("dummy", _DummyController)
    with pytest.raises(ValueError, match="already registered"):
        registry.register("dummy", _DummyController)


def test_registry_rejects_unknown_model_type() -> None:
    registry: ControllerRegistry = ControllerRegistry()
    with pytest.raises(KeyError, match="Unknown model type"):
        registry.create("missing")


def test_default_controller_registry_registers_all_three_families() -> None:
    registry = build_default_controller_registry()
    # Not an exact-tuple check: MODEL_REGISTRY is a shared global singleton, and
    # other decorated test classes (see _OcpDemoController above) also live in
    # it by design.
    assert registry is MODEL_REGISTRY
    assert set(registry.available()) >= {"analytic", "neural", "unfolded"}
    assert registry.get("analytic") is RiccatiController
    assert registry.get("neural") is NeuralPolicy
    assert registry.get("unfolded") is UnfoldedController


def test_register_model_decorator_registers_class_in_global_registry() -> None:
    """OCP demonstration: @register_model alone makes a new class discoverable,
    with zero changes to registry.py."""
    assert "_test_ocp_demo_controller" in MODEL_REGISTRY.available()
    assert MODEL_REGISTRY.get("_test_ocp_demo_controller") is _OcpDemoController

    created = MODEL_REGISTRY.create("_test_ocp_demo_controller", value=42)
    assert created.value == 42


def test_register_model_decorator_returns_class_unchanged() -> None:
    """The decorator must not alter the class it decorates -- just register it."""
    instance = _OcpDemoController(value=1)
    assert isinstance(instance, _OcpDemoController)
    assert instance.value == 1
