"""Human-legible semantic names for stored records (Annex 02 §3).

Content hashes are correct and unusable: nobody recognises
`74ad4e8f36bfbb15`. Every record therefore carries a deterministic name derived
from what it is, with the short digest appended:

    boxlqr-n7m3-N100-u0.1-s0/unfolded-aP-J10/adam-lr1e-2-ep200-b8192-seed0#3f9a1c
    └──────── problem ──────┘└── contender ─┘└─────────── training ───────┘└digest┘

Three properties, in the annex's own words:

* **Deterministic** -- the same descriptors always yield the same name, so it can
  be typed and predicted.
* **Lossy, but ordered by discriminating power** -- problem first, then
  contender, then training, so a truncated display still separates two otherwise
  similar models.
* **The digest is what guarantees uniqueness.** The name is a label, never an
  identity: two models differing only in a field naming does not render share a
  name and are separated by the digest alone.

Naming takes explicit descriptors, not a specification object. `u_max`, `depth`
and `learning_rate` are grammar-level concepts, the grammar is Tier 3, and it
does not exist yet -- so reaching into a spec shape from here would couple this
tier to something unbuilt. The grammar supplies these descriptors when it
arrives; until then a caller fills in what it knows.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass

#: Separates the three segments. A path-safe character, so a name can also be a
#: directory prefix, and shell-safe so it can be typed without quoting.
SEGMENT_SEPARATOR = "/"

#: Introduces the digest.
DIGEST_SEPARATOR = "#"

#: Stands in for a segment with nothing to render (an analytic contender has no
#: training). An empty segment would collapse the separators and make the name
#: ambiguous to split.
EMPTY_SEGMENT = "-"

_UNSAFE = re.compile(r"[^a-z0-9._]+")


def _slug(text: str) -> str:
    """Lowercase, ASCII, hyphen-joined -- safe in a path and in a shell.

    Non-ASCII is transliterated where possible and dropped otherwise, so a
    contender written `unfolded α+P` cannot produce a name that needs quoting or
    that two filesystems normalise differently.
    """
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return _UNSAFE.sub("-", folded.lower()).strip("-")


def _number(value: float | int) -> str:
    """Compact, stable rendering of a number.

    Exact powers of ten at or below 1e-2 use exponent form, because `lr1e-2`
    reads better than `lr0.01` and is what the annex's example shows; everything
    else uses the shortest plain form. Integral floats lose their trailing
    `.0`, so `u_max=1.0` and `u_max=1` cannot produce two different names for
    one value.
    """
    if isinstance(value, bool):  # bool is an int; guard before the int branch
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    if value == int(value) and abs(value) < 1e16:
        return str(int(value))
    exponent = _power_of_ten(value)
    if exponent is not None and exponent <= -2:
        return f"1e{exponent}"
    # `repr`, not `%g`: %g at default precision is LOSSY (123456789.0 renders as
    # 1.23457e+08, so two distinct values would share a name segment), and it
    # emits `+`, which is unsafe in a shell. A name is lossy in which fields it
    # includes, never in a value it does include.
    return repr(value).replace("e+", "e")


def _power_of_ten(value: float) -> int | None:
    """The exponent if `value` is exactly ten to an integer power, else None."""
    if value <= 0:
        return None
    exponent = round(math.log10(value))
    return exponent if math.isclose(10.0**exponent, value, rel_tol=1e-12) else None


@dataclass(frozen=True)
class ProblemName:
    """What distinguishes one problem instance from another.

    Attributes:
        family: The problem family, e.g. `boxlqr`.
        state_dim: State dimension.
        control_dim: Control dimension.
        horizon: Finite horizon.
        u_max: Infinity-norm control bound; `None` for an unconstrained problem,
            in which case the bound is omitted rather than rendered as zero.
        seed: The problem instance's own seed.
    """

    family: str
    state_dim: int | None = None
    control_dim: int | None = None
    horizon: int | None = None
    u_max: float | None = None
    seed: int | None = None

    def render(self) -> str:
        parts = [_slug(self.family)]
        if self.state_dim is not None and self.control_dim is not None:
            parts.append(f"n{_number(self.state_dim)}m{_number(self.control_dim)}")
        if self.horizon is not None:
            parts.append(f"N{_number(self.horizon)}")
        if self.u_max is not None:
            parts.append(f"u{_number(self.u_max)}")
        if self.seed is not None:
            parts.append(f"s{_number(self.seed)}")
        return "-".join(parts)


@dataclass(frozen=True)
class ContenderName:
    """What distinguishes one contender from another.

    Attributes:
        family: Registry family, e.g. `unfolded` or `truncated_riccati`.
        kind: Short variant marker, e.g. `aP` for a learned step size and matrix.
        depth: Unfolding depth, rendered as `J<depth>`.
    """

    family: str
    kind: str | None = None
    depth: int | None = None

    def render(self) -> str:
        parts = [_slug(self.family)]
        if self.kind is not None:
            parts.append(self.kind)
        if self.depth is not None:
            parts.append(f"J{_number(self.depth)}")
        return "-".join(parts)


@dataclass(frozen=True)
class TrainingName:
    """What distinguishes one fit from another.

    Every field is optional: an analytic contender has no training phase at all,
    and its segment renders as `EMPTY_SEGMENT` rather than collapsing.
    """

    optimizer: str | None = None
    learning_rate: float | None = None
    epochs: int | None = None
    batch_size: int | None = None
    seed: int | None = None

    def render(self) -> str:
        parts = []
        if self.optimizer is not None:
            parts.append(_slug(self.optimizer))
        if self.learning_rate is not None:
            parts.append(f"lr{_number(self.learning_rate)}")
        if self.epochs is not None:
            parts.append(f"ep{_number(self.epochs)}")
        if self.batch_size is not None:
            parts.append(f"b{_number(self.batch_size)}")
        if self.seed is not None:
            parts.append(f"seed{_number(self.seed)}")
        return "-".join(parts) if parts else EMPTY_SEGMENT


def semantic_name(
    *,
    problem: ProblemName,
    contender: ContenderName,
    training: TrainingName,
    digest: str,
) -> str:
    """Compose the full semantic name.

    Args:
        problem: The problem descriptor.
        contender: The contender descriptor.
        training: The training descriptor; may be entirely empty.
        digest: The short identifier suffix that guarantees uniqueness -- the
            name alone never does.

    Returns:
        `problem/contender/training#digest`.
    """
    segments = (problem.render(), contender.render(), training.render())
    return SEGMENT_SEPARATOR.join(segments) + DIGEST_SEPARATOR + digest


def split_semantic_name(name: str) -> tuple[str, str, str, str]:
    """Decompose a semantic name into its four parts.

    Args:
        name: A name produced by `semantic_name`.

    Returns:
        `(problem, contender, training, digest)`.

    Raises:
        ValueError: If `name` does not have the expected shape.
    """
    head, _, digest = name.rpartition(DIGEST_SEPARATOR)
    segments = head.split(SEGMENT_SEPARATOR)
    if not digest or len(segments) != 3:
        raise ValueError(
            f"{name!r} is not a semantic name: expected "
            f"problem{SEGMENT_SEPARATOR}contender{SEGMENT_SEPARATOR}"
            f"training{DIGEST_SEPARATOR}digest"
        )
    return segments[0], segments[1], segments[2], digest
