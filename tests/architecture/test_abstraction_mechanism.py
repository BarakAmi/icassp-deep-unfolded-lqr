"""The `ABC`/`Protocol` rule — Stage 7 Phase B.

§2.5 of the parent architecture calls the split *"the same architectural role
expressed through two different mechanisms without a stated rule"*. Half of that
is wrong, and the survey measured which half: **the mechanism differs because the
role differs, consistently**, and the code already says so — `models/lifecycle.py`
ships both for one concept and explains it, `SynthesizerBase(ABC)` being *"the
correct, **inheritable** signature default the deprecated `Controller` Protocol
only pretended to provide"*. What was missing is the *statement*, not the
consistency, so this module states it and guards it. **No production file
changes; nothing is refactored.**

Three clauses, each written so a real, nameable defect violates it — a rule loose
enough to explain everything forbids nothing:

1. an `ABC` supplies inheritable implementation **or** is gated by a nominal
   `isinstance`, both being things a `Protocol` cannot do;
2. a `Protocol` carries no concrete method body, because a body on a `Protocol`
   is implementation nobody inherits;
3. `isinstance` against a `Protocol` implies `@runtime_checkable`, or it is a
   `TypeError` at runtime rather than a type error at review.

Measured before this was written: **10 `ABC`s, 26 `Protocol`s, 0 violations of
each clause.** The guard therefore locks in a property the tree already has.

**Recorded and deliberately not guarded:** seven `Protocol`s are
`@runtime_checkable` while nothing `isinstance`-checks them — `BatchSpec`,
`ExperimentCache`, `Distribution`, and four in `viz/`. The decorator is unused,
not wrong; the clause with teeth is the converse, and it holds. Removing seven
decorators to satisfy a symmetry nobody needs is churn.

**Why the `isinstance` census spans `tests/` too.** `Constraint`'s only nominal
checks are in `tests/core/constraint/test_box_constraint.py`, one of them a
*negative* regression lock (`assert not isinstance(_LegacyConstraint(),
Constraint)`) that fails if the decorator is removed. A census over `src/` alone
would have reported `Constraint` as unchecked — and the survey's first count did
exactly that, for a different reason: a regex whose `[^)]*` could not cross the
inner parenthesis of `isinstance(BoxConstraint(u_max=1.0), Constraint)`. Hence
an AST walk here, not a pattern.

Static source checks only: nothing here imports or executes `src`.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src" / "mbl"

#: Where an abstraction may be *defined*. Only the package is governed; a test
#: helper is free to declare whatever shape it needs.
GOVERNED = SRC

#: Where a nominal check may be *written*. Wider than `GOVERNED` on purpose --
#: see the module docstring on `Constraint`.
CHECK_ROOTS = (SRC, REPO / "tests", REPO / "tools")

#: The frozen legacy package, excluded from every architecture check.
LEGACY_PREFIX = "lqr"


@dataclass(frozen=True)
class Abstraction:
    """One `ABC` or `Protocol`, as the source declares it.

    Attributes:
        key: `path::QualName`, the identifier every failure names.
        name: The bare class name, which is what an `isinstance` mentions.
        kind: ``"abc"`` or ``"protocol"``.
        decorators: Decorator names, unqualified.
        concrete: Methods whose body is real implementation, not a stub.
    """

    key: str
    name: str
    kind: str
    decorators: tuple[str, ...]
    concrete: tuple[str, ...]


def _names(nodes: list[ast.expr]) -> list[str]:
    """Unqualified names of base classes or decorators.

    `Protocol`, `typing.Protocol`, `Protocol[T]` and `@runtime_checkable` all
    have to read the same, or the census silently under-counts and every clause
    below becomes vacuous for whatever it missed.
    """
    out: list[str] = []
    for node in nodes:
        if isinstance(node, ast.Call):
            node = node.func
        if isinstance(node, ast.Subscript):
            node = node.value
        if isinstance(node, ast.Name):
            out.append(node.id)
        elif isinstance(node, ast.Attribute):
            out.append(node.attr)
    return out


def is_stub(function: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Whether a method declares a signature and no implementation.

    The four forms this tree actually uses: a bare docstring, `...`, `pass`, and
    `raise NotImplementedError`. A detector that called everything a stub would
    make clause 2 pass by construction, so
    `test_the_stub_detector_tells_a_body_from_a_signature` exercises it directly.
    """
    body = [
        statement
        for statement in function.body
        if not (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        )
    ]
    if not body:
        return True
    if len(body) > 1:
        return False
    (only,) = body
    if isinstance(only, ast.Pass):
        return True
    if (
        isinstance(only, ast.Expr)
        and isinstance(only.value, ast.Constant)
        and only.value.value is Ellipsis
    ):
        return True
    if isinstance(only, ast.Raise) and only.exc is not None:
        raised = only.exc.func if isinstance(only.exc, ast.Call) else only.exc
        return _names([raised]) == ["NotImplementedError"]
    return False


def _source_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.py")
        if LEGACY_PREFIX not in path.relative_to(root.parent).parts[:1]
    )


def _census() -> tuple[list[Abstraction], dict[str, list[str]]]:
    """Every governed abstraction, and every nominal check in the tree.

    Both halves come from one AST walk each, never from a regex: the survey's
    first attempt at clause 3 used a pattern and under-reported, which would
    have hidden a genuine violation rather than merely miscounting.
    """
    abstractions: list[Abstraction] = []
    for path in _source_files(GOVERNED):
        relative = path.relative_to(REPO.joinpath("src", "mbl")).as_posix()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.ClassDef):
                continue
            bases = _names(node.bases)
            kind = (
                "protocol" if "Protocol" in bases else "abc" if "ABC" in bases else ""
            )
            if not kind:
                continue
            abstractions.append(
                Abstraction(
                    key=f"{relative}::{node.name}",
                    name=node.name,
                    kind=kind,
                    decorators=tuple(_names(node.decorator_list)),
                    concrete=tuple(
                        member.name
                        for member in node.body
                        if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and not is_stub(member)
                    ),
                )
            )

    checked: dict[str, list[str]] = defaultdict(list)
    for root in CHECK_ROOTS:
        for path in _source_files(root):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if (
                    not isinstance(node, ast.Call)
                    or not isinstance(node.func, ast.Name)
                    or node.func.id not in {"isinstance", "issubclass"}
                ):
                    continue
                # The second argument may be a tuple, a subscript or a name; walk
                # it, because `isinstance(BoxConstraint(u_max=1.0), Constraint)`
                # is where the survey's regex stopped reading.
                for argument in node.args[1:]:
                    for inner in ast.walk(argument):
                        if isinstance(inner, ast.Name):
                            checked[inner.id].append(path.relative_to(REPO).as_posix())
    return abstractions, dict(checked)


ABSTRACTIONS, NOMINAL_CHECKS = _census()
ABCS = [a for a in ABSTRACTIONS if a.kind == "abc"]
PROTOCOLS = [a for a in ABSTRACTIONS if a.kind == "protocol"]


def _ids(items: list[Abstraction]) -> list[str]:
    return [item.key for item in items]


# --------------------------------------------------------------------------
# The census itself must not be empty, or every clause below is vacuous
# --------------------------------------------------------------------------


def test_the_census_finds_both_mechanisms() -> None:
    """A discovery that silently found nothing would pass all three clauses.

    Named specimens rather than exact counts: the tree is allowed to grow an
    abstraction without this file changing, but it is not allowed to lose the
    two that carry the rule's own explanation.
    """
    assert ABCS and PROTOCOLS
    assert "models/lifecycle.py::SynthesizerBase" in _ids(ABCS)
    assert "models/lifecycle.py::Synthesizer" in _ids(PROTOCOLS)
    assert NOMINAL_CHECKS, "no isinstance/issubclass found anywhere -- census broken"


def test_the_stub_detector_tells_a_body_from_a_signature() -> None:
    """Clause 2 is only as strong as this predicate."""
    stubs = """
class S:
    def a(self) -> None: ...
    def b(self) -> None:
        pass
    def c(self) -> None:
        '''Only a docstring.'''
    def d(self) -> None:
        raise NotImplementedError
    def e(self) -> None:
        '''Docstring and an ellipsis.'''
        ...
"""
    bodies = """
class B:
    def a(self) -> int:
        return 1
    def b(self) -> None:
        raise ValueError("a real refusal is a body")
    def c(self) -> None:
        '''Docstring, then work.'''
        self.x = 1
"""
    for source, expected in ((stubs, True), (bodies, False)):
        klass = ast.parse(source).body[0]
        assert isinstance(klass, ast.ClassDef)
        methods = [m for m in klass.body if isinstance(m, ast.FunctionDef)]
        assert methods
        assert all(is_stub(m) is expected for m in methods), source


# --------------------------------------------------------------------------
# The three clauses
# --------------------------------------------------------------------------


@pytest.mark.parametrize("abstraction", ABCS, ids=_ids(ABCS))
def test_an_abc_implements_something_or_is_nominally_checked(
    abstraction: Abstraction,
) -> None:
    """Clause 1. Both halves are things a `Protocol` cannot do, so an `ABC` that
    does neither is a `Protocol` written with the wrong keyword."""
    checked_in = NOMINAL_CHECKS.get(abstraction.name, [])
    assert abstraction.concrete or checked_in, (
        f"{abstraction.key} is an ABC that supplies no inheritable "
        "implementation and is never isinstance-checked, so it uses nothing an "
        "ABC has and a Protocol lacks. Either give it a concrete default, "
        "isinstance-check it where the nominal type matters, or make it a "
        "Protocol."
    )


@pytest.mark.parametrize("abstraction", PROTOCOLS, ids=_ids(PROTOCOLS))
def test_a_protocol_carries_no_concrete_method_body(
    abstraction: Abstraction,
) -> None:
    """Clause 2. A body on a `Protocol` is implementation nobody inherits: a
    structural implementer never sees it, so it runs only for whoever
    accidentally subclasses the `Protocol` itself."""
    assert not abstraction.concrete, (
        f"{abstraction.key} is a Protocol carrying implementation: "
        f"{sorted(abstraction.concrete)}. Nobody inherits it -- structural "
        "implementers never see those bodies. Move the default to an ABC "
        "(the SynthesizerBase pattern) or reduce the methods to signatures."
    )


@pytest.mark.parametrize("abstraction", PROTOCOLS, ids=_ids(PROTOCOLS))
def test_a_nominally_checked_protocol_is_runtime_checkable(
    abstraction: Abstraction,
) -> None:
    """Clause 3. The failure this prevents is a `TypeError` raised by
    `isinstance` itself, at runtime, on a line that reads as a type check."""
    checked_in = sorted(set(NOMINAL_CHECKS.get(abstraction.name, [])))
    if not checked_in:
        return
    assert "runtime_checkable" in abstraction.decorators, (
        f"{abstraction.key} is isinstance-checked in {checked_in} but is not "
        "@runtime_checkable, so that check raises TypeError when it runs. Add "
        "the decorator, or stop checking it nominally."
    )
