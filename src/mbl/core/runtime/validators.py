"""Decorator-based validation surfaces under the Import-Time Identity Law
(T1.5.b), plus the always-on Pydantic boundary surface (T1.5.a).

**The Import-Time Identity Law:** in ``FAST`` mode, every decorator-based
type/shape/boundary validation surface evaluates — at import/decoration
time — directly to the **original, undecorated function object**. Not a
pass-through wrapper, not a disabled branch inside a wrapper:
``decorated is original`` holds, so hot paths carry provably zero wrapper
stack frames, nothing for ``torch.compile``/``torch.jit`` to trace
through, and nothing to pay per call. This generalizes the
decoration-time-resolution discipline ``core.utils.shapes
.enforce_tensor_shapes`` already practices.

Accepted consequence (stated in ``modes.py``): the mode is effectively
frozen at import for these surfaces — escalating ``FAST -> STRICT``
requires setting the mode before the decorated modules are imported.

Three surfaces:

* `mode_gated` — the generic identity-law gate: lifts any validation
  decorator into one that vanishes identically under ``FAST``.
* `shape_checked` — jaxtyping tensor contracts (shape/dtype/backend of
  annotated arrays), typechecked by beartype, active only under
  ``STRICT``. The declarative replacement for hand-written
  ``enforce_tensor_shapes`` spec strings on new Tier 2 contracts.
* `validate_boundary` — Pydantic ingress for external data (config files,
  CLI/notebook parameters, persisted metadata). **Always on, in both
  modes**: its one-time cost is negligible and silent config corruption is
  never acceptable. It is a function, not a decorator, so the identity law
  does not apply to it — by design.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from typing import Any, TypeVar, cast

from beartype import beartype
from jaxtyping import jaxtyped
from pydantic import TypeAdapter

from .modes import ValidationMode, get_validation_mode

F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")

type ValidationDecorator = Callable[[Any], Any]


def mode_gated(validation_decorator: ValidationDecorator) -> ValidationDecorator:
    """Lift a validation decorator under the Import-Time Identity Law.

    Args:
        validation_decorator: Any function-wrapping validation decorator
            (jaxtyping, beartype, a custom contract checker, ...).

    Returns:
        A decorator that, resolved once at decoration time, returns the
        original function object unchanged (``decorated is original``) when
        the mode is ``FAST``, and `validation_decorator`'s wrapping when
        ``STRICT``.
    """

    def decorate(func: F) -> F:
        if get_validation_mode() is ValidationMode.FAST:
            return func
        # The wrapped callable keeps func's signature; the decorator factory
        # itself is untyped from mypy's perspective.
        return cast("F", validation_decorator(func))

    return decorate


shape_checked = mode_gated(jaxtyped(typechecker=beartype))
"""Enforce jaxtyping tensor annotations (``Float[Tensor, "batch time n"]``,
``Float[np.ndarray, "horizon n n"]``) on a function's arguments and return.

STRICT: every annotated array is checked for shape, dtype, and array
backend, with symbolic axis labels unified across the signature.
FAST: identity — the original function object, zero overhead (T1.5.b).
"""


@cache
def _adapter_for(schema: type) -> TypeAdapter[Any]:
    """Memoize `TypeAdapter` construction per schema type (adapter building
    walks the annotation tree; ingress sites reuse the same schemas)."""
    return TypeAdapter(schema)


def validate_boundary(schema: type[T] | TypeAdapter[T], payload: object) -> T:
    """Validate external data crossing a system boundary (Pydantic plane).

    Boundary ingress is **always on** — mode-independent — because it runs
    once per object (file load, CLI/notebook entry), never in hot loops,
    and silently accepting corrupt configuration is never acceptable.

    Args:
        schema: The expected type — a Pydantic model, dataclass, or any
            annotation `TypeAdapter` accepts — or a prebuilt `TypeAdapter`.
        payload: The untrusted external data (parsed JSON/dict/...).

    Returns:
        The validated (and coerced) instance of `schema`.

    Raises:
        pydantic.ValidationError: If `payload` does not conform.
    """
    adapter = (
        schema if isinstance(schema, TypeAdapter) else _adapter_for(schema)  # type: ignore[arg-type]  # lru_cache keyed on the class object (hashable in practice)
    )
    return adapter.validate_python(payload)
