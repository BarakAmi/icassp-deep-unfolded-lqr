"""State-protecting data descriptors and a validated-dataclass companion.

Where a plain ``@guard`` validator checks a value *once at a call site*, a
:class:`ValidatedField` descriptor binds the invariant to the attribute itself:
the value is validated on assignment and the attribute is **write-once**
(re-assignment raises), so an object can never hold -- or later drift into --
an invalid state. This is the "fundamentally protect the state of objects"
layer: the invariant travels with the attribute, not with each caller.

Descriptors work on ordinary classes. Frozen dataclasses build their instances
with ``object.__setattr__``, bypassing descriptor ``__set__`` entirely, so for
those the same predicate registry is applied field-by-field in ``__post_init__``
via :func:`validated_dataclass` + ``typing.Annotated`` -- one source of truth,
two delivery mechanisms.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from functools import wraps
from typing import Annotated, Any, Protocol, get_args, get_origin, get_type_hints, cast

from .validation import (
    ensure_positive_integer,
    validate_positive_definite,
    validate_positive_definite_stack,
    validate_positive_semi_definite,
    validate_positive_semi_definite_stack,
    validate_square_matrix,
)


class FieldValidator(Protocol):
    """``(name, value) -> validated_value``; raises on an invalid value."""

    def __call__(self, name: str, value: Any) -> Any: ...


class ValidatedField:
    """Base write-once data descriptor: validates on first assignment, rejects
    reassignment, and stores the (possibly normalized) value per instance."""

    def __set_name__(self, owner: type, name: str) -> None:
        """Called automatically by Python when the descriptor is assigned as a
        class attribute; records the attribute's name and its private,
        per-instance storage key.

        Args:
            owner: The class the descriptor is defined on.
            name: The attribute name the descriptor was assigned to.
        """
        self._name = name
        self._store = f"__validated_{name}"

    def __get__(self, obj: object, objtype: type | None = None) -> Any:
        """Return the validated value previously stored on `obj`.

        Args:
            obj: The instance being accessed, or ``None`` for class-level access.
            objtype: The owning class.

        Returns:
            The descriptor itself if accessed on the class (``obj is None``);
            otherwise the stored, validated value.

        Raises:
            AttributeError: If accessed on an instance before any value has
                been assigned.
        """
        if obj is None:
            return self
        return getattr(obj, self._store)

    def __set__(self, obj: object, value: Any) -> None:
        """Validate `value` via `validate` and store it, once.

        Args:
            obj: The instance being assigned to.
            value: The candidate value.

        Raises:
            AttributeError: If this attribute has already been assigned on `obj`.
            Exception: Whatever `validate` raises for an invalid `value`
                (typically ``TypeError``/``ValueError``).
        """
        if hasattr(obj, self._store):
            raise AttributeError(
                f"{self._name!r} is validated-immutable and cannot be reassigned."
            )
        object.__setattr__(obj, self._store, self.validate(self._name, value))

    def validate(self, name: str, value: Any) -> Any:  # pragma: no cover - abstract
        """Validate and normalize `value`; subclasses must override.

        Args:
            name: The attribute name, for error messages.
            value: The candidate value.

        Returns:
            The (possibly normalized) value to store.

        Raises:
            NotImplementedError: Always, on the base class.
        """
        raise NotImplementedError


def _field(validator: Callable[[str, Any], Any]) -> type[ValidatedField]:
    """Build a concrete write-once descriptor from a ``(name, value)`` validator.

    Args:
        validator: A ``(name, value) -> validated_value`` function (e.g.
            `validation.validate_positive_definite`), raising on an invalid
            `value`.

    Returns:
        A `ValidatedField` subclass whose `validate` delegates to `validator`.
    """

    class _Concrete(ValidatedField):
        def validate(self, name: str, value: Any) -> Any:
            return validator(name, value)

    return _Concrete


SquareMatrix = _field(validate_square_matrix)
PositiveSemiDefiniteMatrix = _field(validate_positive_semi_definite)
PositiveDefiniteMatrix = _field(validate_positive_definite)
TimeStackedPSD = _field(validate_positive_semi_definite_stack)
TimeStackedPD = _field(validate_positive_definite_stack)


class PositiveInt(ValidatedField):
    """Write-once ``int`` descriptor: validates a strictly positive integer."""

    def validate(self, name: str, value: Any) -> int:
        """Validate `value` as a positive integer.

        Args:
            name: The attribute name, for error messages.
            value: The candidate value.

        Returns:
            `value`, unchanged.

        Raises:
            ValueError: If `value` is not a positive ``int``.
        """
        ensure_positive_integer(value, name)
        return cast(int, value)


class BoundedFloat(ValidatedField):
    """Write-once float in ``[low, high]`` (bounds optional; ``inclusive``
    controls strictness on whichever bounds are given)."""

    def __init__(
        self,
        *,
        low: float | None = None,
        high: float | None = None,
        inclusive: bool = True,
    ) -> None:
        """
        Args:
            low: Optional lower bound; ``None`` means unbounded below.
            high: Optional upper bound; ``None`` means unbounded above.
            inclusive: If ``True``, `low`/`high` are themselves valid values
                (``low <= value <= high``); if ``False``, they are excluded
                (``low < value < high``).
        """
        self._low = low
        self._high = high
        self._inclusive = inclusive

    def validate(self, name: str, value: Any) -> float:
        """Validate `value` as a real number within the configured bounds.

        Args:
            name: The attribute name, for error messages.
            value: The candidate value.

        Returns:
            `value` coerced to ``float``.

        Raises:
            TypeError: If `value` is not a real number (``bool`` is rejected
                even though it is an ``int`` subclass).
            ValueError: If `value` falls outside ``[low, high]`` (or the open
                interval, when `inclusive` is ``False``).
        """
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(
                f"{name} must be a real number, got {type(value).__name__}."
            )
        low, high, inc = self._low, self._high, self._inclusive
        too_low = low is not None and (value < low if inc else value <= low)
        too_high = high is not None and (value > high if inc else value >= high)
        if too_low or too_high:
            edge = "[]" if inc else "()"
            raise ValueError(
                f"{name} must lie in {edge[0]}{low}, {high}{edge[1]}, got {value}."
            )
        return float(value)


# --- validated frozen-dataclass companion ------------------------------------


def _extract_validators(cls: type) -> dict[str, list[FieldValidator]]:
    """Map field name -> validators declared via ``Annotated[T, validator, ...]``.

    Args:
        cls: A ``@dataclass``-decorated class to inspect.

    Returns:
        A dict from field name to the list of callable metadata objects found
        in that field's ``Annotated[...]`` type hint (fields without any
        callable metadata are omitted).
    """
    hints = get_type_hints(cls, include_extras=True)
    validators: dict[str, list[FieldValidator]] = {}
    for field in dataclasses.fields(cls):
        hint = hints.get(field.name)
        if get_origin(hint) is not Annotated:
            continue
        found = [meta for meta in get_args(hint)[1:] if callable(meta)]
        if found:
            validators[field.name] = found
    return validators


def validated_dataclass(cls: type) -> type:
    """Class decorator (apply *above* ``@dataclass``): after the dataclass'
    ``__init__`` (and any ``__post_init__``) has run, validate every field
    carrying ``Annotated`` validators. Wrapping ``__init__`` rather than
    injecting ``__post_init__`` means it fires whether or not the dataclass
    already defines one. Uses the same validators as the descriptors, so frozen
    configs and mutable classes enforce identical invariants.

    Args:
        cls: The ``@dataclass``-decorated class to instrument (decorate this
            *above* ``@dataclass``, i.e. apply after it, so `cls` is already a
            dataclass when this runs).

    Returns:
        `cls`, with its ``__init__`` wrapped to validate every
        ``Annotated``-declared field after construction.

    Raises:
        Exception: Whatever a field's validator raises for an invalid value
            (typically ``TypeError``/``ValueError``), propagated from the
            wrapped ``__init__``.
    """
    validators = _extract_validators(cls)
    original_init = cls.__init__  # type: ignore[misc]  # deliberate wrap of the class-level __init__

    @wraps(original_init)
    def __init__(self: Any, *args: object, **kwargs: object) -> None:
        original_init(self, *args, **kwargs)
        for name, field_validators in validators.items():
            value = getattr(self, name)
            for validate in field_validators:
                validate(name, value)

    cls.__init__ = __init__  # type: ignore[misc]  # deliberate method replacement
    return cls
