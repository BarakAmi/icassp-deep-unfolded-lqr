"""Writing into a specification tree by dotted path.

Two things write into a study by path and they must agree exactly: the sweep
algebra, which sets an axis value at every point (`study.py`), and the tier
catalogue, which sets an effort knob before the sweep expands (`tiers.py`). Two
copies of this traversal would drift, and the two callers would then disagree
about what `training.plan.epochs` addresses -- so it lives here, once.

The traversal is deliberately **duck-typed** over frozen dataclasses and
mappings, which is what a spec tree is made of, rather than importing the
concrete plan and protocol types. Tier 3 may not reach into `engine` or
`experiments`, which is the same reason those fields are typed `Signable` in
the first place.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any, cast

from .errors import SpecificationError

#: The segment that fans a path out over a sequence field. The same character
#: the sweep algebra already writes in `contenders.*.config.…`, and deliberately
#: the same constant: two spellings of one wildcard would be a second grammar.
WILDCARD = "*"


def replace_at(target: Any, path: Sequence[str], value: Any) -> Any:
    """Rebuild `target` with `path` set to `value`.

    Args:
        target: A frozen dataclass or a mapping, at the level `path` starts at.
        path: The remaining segments, outermost first. Empty is not accepted --
            a caller replacing the whole object does not need this function.
        value: What to write at the leaf.

    Returns:
        A new object of the same type; nothing is mutated.

    Raises:
        SpecificationError: If a segment names no field or key, or if the path
            descends into something that is neither a specification object nor
            a configuration mapping. Named rather than defaulted: a misspelled
            path that silently created a field would be a knob nobody is
            turning, reported in provenance as one that was.
    """
    head, rest = path[0], path[1:]
    if isinstance(target, dict):
        if not rest:
            return {**target, head: value}
        if head not in target:
            raise SpecificationError(f"{head!r} is not a key of this configuration")
        return {**target, head: replace_at(target[head], rest, value)}
    if not dataclasses.is_dataclass(target):
        raise SpecificationError(
            f"cannot set {head!r} on {type(target).__name__}, which is neither a "
            "specification object nor a configuration mapping"
        )
    if head not in {f.name for f in dataclasses.fields(target)}:
        raise SpecificationError(
            f"{head!r} is not a field of {type(target).__name__}; available: "
            f"{sorted(f.name for f in dataclasses.fields(target))}"
        )
    inner = value if not rest else replace_at(getattr(target, head), rest, value)
    # `is_dataclass` narrows to "instance or class"; only an instance reaches
    # here, and `replace` cannot express that in its own type variable.
    return dataclasses.replace(cast("Any", target), **{head: inner})


def addresses(target: Any, path: Sequence[str]) -> bool:
    """Whether `path` names something that already exists under `target`.

    Checked *before* writing rather than by catching what `replace_at` raises.
    Catching would conflate "this contender has no plan", which is ordinary and
    must be skipped, with "this tree is malformed", which must not be.

    Note the asymmetry with `replace_at`, which will happily create a new key at
    a mapping leaf: here a leaf that is absent reads as absent. That is what a
    fan-out needs -- a tier writing an epoch count onto a contender that never
    declared a plan would be inventing a fitting procedure for a closed-form
    solve, not scaling one.

    Args:
        target: Where to start.
        path: The segments to follow.

    Returns:
        Whether every segment resolves.
    """
    for segment in path:
        if isinstance(target, dict):
            if segment not in target:
                return False
            target = target[segment]
        elif dataclasses.is_dataclass(target) and not isinstance(target, type):
            if segment not in {f.name for f in dataclasses.fields(target)}:
                return False
            target = getattr(target, segment)
        else:
            return False
    return True


def _names_an_element(target: Any, segment: str) -> bool:
    """Whether `segment` is the label of one element of a sequence `target`.

    Deliberately narrow: only a sequence whose elements carry a `label` can be
    addressed this way, and only by a label one of them actually has. A segment
    that matches nothing falls through to the ordinary attribute walk, so a
    misspelling still refuses with the message it always did rather than
    silently turning into a fan-out that matches nobody.
    """
    if isinstance(target, (str, bytes)) or not isinstance(target, Sequence):
        return False
    return any(getattr(element, "label", None) == segment for element in target)


def replace_at_all(target: Any, path: Sequence[str], value: Any) -> tuple[Any, int]:
    """`replace_at`, with a `WILDCARD` segment fanning out over a sequence.

    The one path a tier needs this for is
    `contenders.*.config.plan.epochs`: the fitting effort that actually executes
    lives per contender, and until F2a nothing could address "every contender
    that has one". An element *without* the remaining path is left untouched
    rather than refused -- three of NB04's six contenders never train.

    **The count is returned rather than acted on**, because the two callers
    disagree about what zero means for a good reason (plan §F2a). A shared
    catalogue cannot know which families a given study declares, so a path
    matching nothing is ordinary there; a per-study overlay or a `--set` is
    written by someone looking at that study, so it is a typo, and a knob
    nobody turns that provenance reports as one that was is the defect this
    whole phase exists to close.

    Args:
        target: A frozen dataclass or mapping at the level `path` starts at.
        path: The remaining segments, outermost first.
        value: What to write at each addressed leaf.

    Returns:
        `(rebuilt, matched)` -- a new object of the same type, and how many
        elements accepted the path. `matched` is 1 for a path with no wildcard
        in it, which cannot partially apply.

    Raises:
        SpecificationError: If a non-wildcard segment names no field or key, or
            if a wildcard is applied to something that is not a sequence.
    """
    head, rest = path[0], path[1:]
    if head == WILDCARD or _names_an_element(target, head):
        if isinstance(target, (str, bytes)) or not isinstance(target, Sequence):
            raise SpecificationError(
                f"{WILDCARD!r} fans out over a sequence, but this path reaches "
                f"{type(target).__name__}, which is not one"
            )
        rebuilt = []
        matched = 0
        for element in target:
            # A segment that is not the wildcard selects ONE element by its
            # label; every other element is carried through untouched. This is
            # `applies_to`'s addressing, borrowed rather than reinvented: a
            # sweep already names contenders by label, and an override that
            # could only ever say "all of them" cannot express a budget the
            # campaign measured per family (Annex 01 §2.3.1, 2026-08-12).
            selected = head == WILDCARD or getattr(element, "label", None) == head
            if not selected:
                rebuilt.append(element)
            elif not rest:
                rebuilt.append(value)
                matched += 1
            elif addresses(element, rest):
                rebuilt.append(replace_at(element, rest, value))
                matched += 1
            else:
                rebuilt.append(element)
        # A study holds its contenders in a tuple, and a frozen dataclass
        # rebuilt with a list where it declares one compares unequal to itself.
        return type(target)(rebuilt), matched  # type: ignore[call-arg]  # tuple/list both take an iterable
    if not rest:
        return replace_at(target, path, value), 1
    # Descend one level and recurse, ALWAYS. This used to short-cut to
    # `replace_at` whenever no `*` appeared in the remaining segments, which
    # was right while the wildcard was the only way to reach into a sequence:
    # a label segment further down never got the chance to be recognised, and
    # `contenders.neural.config.plan.epochs` refused with "cannot set 'neural'
    # on tuple". Recursing uniformly costs one frame per segment and lets the
    # element-addressing branch above see every level it might apply at.
    inner, matched = replace_at_all(_descend(target, head), rest, value)
    return replace_at(target, [head], inner), matched


def _descend(target: Any, segment: str) -> Any:
    """One step down, refusing in the same words `replace_at` would."""
    if isinstance(target, dict):
        if segment not in target:
            raise SpecificationError(f"{segment!r} is not a key of this configuration")
        return target[segment]
    if not dataclasses.is_dataclass(target):
        raise SpecificationError(
            f"cannot set {segment!r} on {type(target).__name__}, which is neither "
            "a specification object nor a configuration mapping"
        )
    if segment not in {f.name for f in dataclasses.fields(target)}:
        raise SpecificationError(
            f"{segment!r} is not a field of {type(target).__name__}; available: "
            f"{sorted(f.name for f in dataclasses.fields(target))}"
        )
    return getattr(target, segment)
