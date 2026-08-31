"""Framework-agnostic, autograd-safe tensor-shape enforcement.

:func:`enforce_tensor_shapes` wraps a rollout/``forward`` boundary and validates
the ranks and named axes of selected arguments (and the return) *before* the
call, turning a silent NumPy/PyTorch broadcast mismatch into an immediate,
located error instead of an opaque failure several frames deep in autograd.

Two invariants make it safe on the hot path:

* **Never detach / never convert.** Only ``.ndim`` / ``.shape`` are read -- both
  are graph-neutral on a ``torch.Tensor``. Coercing a grad-requiring tensor to
  numpy would break (and, per ``core.utils.array_ops.match_array_type``, crash)
  autograd, so the wrapper never does it.
* **Resolve on/off once, at decoration time.** When the ``SHAPE`` tier is
  inactive (``python -O`` or an env override) the decorator returns the
  *original, unwrapped* function -- zero per-call overhead and nothing for
  ``torch.compile`` / ``torch.jit.script`` to trip over. ``functools.wraps``
  preserves ``__wrapped__`` so tooling can always recover the raw callable.

Spec grammar (per argument, and per return): a tuple of axis labels.
    * ``int``            -> exact size.
    * ``"B"``, ``"n"``   -> symbolic; every occurrence of a label across all
                            arguments and the return must bind to one size.
    * ``"T+1"`` / ``"T-1"`` -> symbolic base label with an integer offset
                            (lets a rollout assert ``states`` has ``T+1`` steps
                            given ``T``-step inputs).
    * ``None``           -> wildcard: any size, no binding.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable, Sequence
from functools import wraps
from typing import Any, TypeVar

from .guards import GuardTier, is_tier_active

AnyFunc = TypeVar("AnyFunc", bound=Callable[..., Any])

type AxisLabel = int | str | None
type ShapeSpec = Sequence[AxisLabel]

_LABEL_RE = re.compile(r"^([A-Za-z_]\w*)([+-]\d+)?$")


def _is_array_like(value: object) -> bool:
    """Duck-typed: has both ``.shape`` and ``.ndim`` (numpy array or torch
    tensor). Deliberately avoids importing torch so this stays framework-free.

    Args:
        value: The object to check.

    Returns:
        ``True`` iff `value` exposes both a ``.shape`` and ``.ndim`` attribute.
    """
    return hasattr(value, "shape") and hasattr(value, "ndim")


def _parse_label(label: str) -> tuple[str, int]:
    """Split an axis label into its symbolic base and integer offset.

    Args:
        label: An axis label, e.g. ``"T"``, ``"T+1"``, or ``"T-1"``.

    Returns:
        A ``(base, offset)`` pair, e.g. ``("T", 0)`` or ``("T", 1)``.

    Raises:
        ValueError: If `label` does not match the ``name[+-]int`` grammar.
    """
    match = _LABEL_RE.match(label)
    if match is None:
        raise ValueError(f"Invalid axis label {label!r} in shape spec.")
    base, offset = match.group(1), match.group(2)
    return base, int(offset) if offset else 0


def _check_one(
    name: str,
    value: object,
    spec: ShapeSpec,
    bindings: dict[str, int],
    where: str,
) -> None:
    """Validate one array/tensor `value` against `spec`, unifying any symbolic
    axis labels into `bindings` (shared across every argument and the return
    within one `enforce_tensor_shapes`-wrapped call).

    Args:
        name: Human-readable identifier for `value` (an argument name, or
            ``"return[i]"``) used in the error message.
        value: The candidate array/tensor. Non-array-like values (see
            `_is_array_like`) are silently skipped -- the wrapped function may
            legitimately receive non-tensor arguments.
        spec: The expected per-axis labels for `value` (see module docstring).
        bindings: Mutable symbolic-label -> size map, updated in place.
        where: A qualified-name prefix (e.g. ``"MyClass.method()"``) used in
            error messages.

    Raises:
        ValueError: If `value`'s rank does not match ``len(spec)``, an
            ``int`` label's exact size does not match, or a symbolic label's
            implied size conflicts with an existing binding.
    """
    if not _is_array_like(value):
        return
    if value.ndim != len(spec):  # type: ignore[attr-defined]
        raise ValueError(
            f"{where}: '{name}' must be {len(spec)}D "
            f"(spec {tuple(spec)}), got shape {tuple(value.shape)}."  # type: ignore[attr-defined]
        )
    for axis, (label, size) in enumerate(zip(spec, value.shape, strict=True)):  # type: ignore[attr-defined]
        if label is None:
            continue
        if isinstance(label, int):
            if size != label:
                raise ValueError(
                    f"{where}: '{name}' axis {axis} must be {label}, got {size}."
                )
            continue
        base, offset = _parse_label(label)
        expected_base = size - offset
        bound = bindings.get(base)
        if bound is None:
            bindings[base] = expected_base
        elif bound != expected_base:
            raise ValueError(
                f"{where}: '{name}' axis {axis} (label '{label}') implies "
                f"{base}={expected_base}, but {base}={bound} was already bound "
                f"(size mismatch across arguments)."
            )


def _normalize_returns(
    returns: ShapeSpec | Sequence[ShapeSpec] | None,
) -> list[ShapeSpec] | None:
    """Normalize the `enforce_tensor_shapes(returns=...)` argument into a
    per-output-position list of specs.

    Args:
        returns: ``None`` (no return validation), a single `ShapeSpec` (for a
            single-value return), or a list of `ShapeSpec`/``None`` entries
            (for a tuple-valued return).

    Returns:
        ``None``, or a list of `ShapeSpec`/``None`` aligned with the wrapped
        function's return value(s).
    """
    if returns is None:
        return None
    # A list whose entries are themselves specs (tuples/lists/None) => multiple
    # returns; otherwise a single spec.
    if isinstance(returns, list) and all(
        item is None or isinstance(item, (tuple, list)) for item in returns
    ):
        return returns
    return [returns]  # type: ignore[list-item]


def enforce_tensor_shapes(
    returns: ShapeSpec | Sequence[ShapeSpec] | None = None,
    *,
    tier: GuardTier = GuardTier.SHAPE,
    **arg_specs: ShapeSpec,
) -> Callable[[AnyFunc], AnyFunc]:
    """Validate named axes of selected tensor/array arguments (and the return),
    with cross-argument unification. See module docstring for the spec grammar.

    Args:
        returns: The expected shape spec(s) for the wrapped function's return
            value; see `_normalize_returns`. ``None`` skips return validation.
        tier: The `GuardTier` this check belongs to; resolved once at
            decoration time (see module docstring's "Resolve on/off once"
            invariant).
        **arg_specs: Maps parameter name -> `ShapeSpec` for each argument to
            validate; unlisted parameters are not checked.

    Returns:
        The original `func` unchanged if `tier` is inactive at decoration time;
        otherwise a wrapper that validates shapes before delegating to `func`.

    Raises:
        ValueError: (from the wrapper, at call time) if any checked argument or
            return value has the wrong rank, an ``int``-labeled axis of the
            wrong size, or a symbolic-labeled axis whose implied size conflicts
            with another argument's binding of the same label.
    """
    return_specs = _normalize_returns(returns)

    def decorate(func: AnyFunc) -> AnyFunc:
        # Resolve once, at import/decoration time (see module docstring).
        if not is_tier_active(tier):
            return func

        signature = inspect.signature(func)
        qualified = getattr(func, "__qualname__", func.__name__)

        @wraps(func)
        def wrapper(*args: object, **kwargs: object) -> Any:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            bindings: dict[str, int] = {}
            for name, spec in arg_specs.items():
                if name in bound.arguments:
                    _check_one(
                        name, bound.arguments[name], spec, bindings, f"{qualified}()"
                    )

            result = func(*args, **kwargs)

            if return_specs is not None:
                outputs = result if isinstance(result, tuple) else (result,)
                for index, spec in enumerate(return_specs):
                    if spec is None or index >= len(outputs):
                        continue
                    _check_one(
                        f"return[{index}]",
                        outputs[index],
                        spec,
                        bindings,
                        f"{qualified}()",
                    )
            return result

        return wrapper  # type: ignore[return-value]

    return decorate
