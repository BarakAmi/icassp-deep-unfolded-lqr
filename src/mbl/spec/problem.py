"""The frozen problem specification (Stage 2 Phase A, decision **D19**).

**Problem data is frozen, not generated.** A `ProblemSpec` holds the system and
cost matrices as data; the generator that produced them is recorded as
provenance and never participates in identity.

The reason is measured. `generate_marginally_stable_system` ends in
``A /= max|eigvals(A)|`` -- a LAPACK call whose last bits differ between
machines -- so a problem *regenerated* on a new host is a different problem by
content, and every model trained on it would retrain. Two alternatives were
tested and rejected: hashing the *recipe* is blind to generator changes (a bug
fix would silently reuse models trained on different matrices) and cannot
express hand-specified matrices at all; quantising the mantissa fails on
rounding boundaries and worsens with dimension, disagreeing on 0.6% of
1-ULP-perturbed 7x7 matrices and 26.5% at 50x50.

Freezing resolves it: the nondeterminism happens **once**, at authoring time,
and its output is committed. Hashing the stored data is then both portable and
complete, and hand-specified matrices become the native case rather than a
special one.

Two consequences the type enforces rather than documents:

* **Provenance is not in the signature tree at all.** Not merely equal-hashing
  -- absent. Re-recording how a problem was authored cannot orphan its models.
* **Matrices are float64, whatever precision training runs at.** Storing a
  problem at the training dtype would make one problem two under D20, and
  holding a float32 quantity to a float64 contract is precisely the defect that
  made the golden-master suite depend on which CPU ran it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..core.constraint.box_constraint import BoxConstraint
from ..core.cost.quadratic_cost import QuadraticCost
from ..core.optimal_control_problem import OptimalControlProblem
from ..core.system.linear_system import LinearSystem
from ..core.utils.signing import hash_array
from ..store.ids import ProblemID, problem_id
from .errors import SpecificationError

#: The state and input matrices every problem must carry.
SYSTEM_KEYS = ("A", "B")

#: The state and control cost matrices every problem must carry.
COST_KEYS = ("Q", "R")

#: The one precision problem data is ever stored at.
STORAGE_DTYPE = np.float64

#: Key under which `save` stores the JSON side-car inside the `.npz`, chosen so
#: it cannot collide with a matrix name.
METADATA_KEY = "__spec__"

#: Separates a group from a matrix name in the flat `.npz` namespace.
_GROUP_SEPARATOR = "/"


#: Re-exported so `from mbl.spec.problem import SpecificationError` keeps
#: working now that the error is shared with `contender.py` and everything
#: after it.
__all__ = [
    "COST_KEYS",
    "METADATA_KEY",
    "STORAGE_DTYPE",
    "SYSTEM_KEYS",
    "GeneratorProvenance",
    "ProblemData",
    "ProblemSpec",
    "SpecificationError",
]


@dataclass(frozen=True)
class GeneratorProvenance:
    """How a problem's matrices were produced. Recorded, never signed.

    Attributes:
        generator: The function that authored the matrices.
        params: The arguments it was called with.
        package_version: The version of this package at authoring time.
    """

    generator: str
    params: Mapping[str, Any] = field(default_factory=dict)
    package_version: str = ""

    def as_json(self) -> dict[str, Any]:
        """The record, for storage alongside the data it explains."""
        return {
            "generator": self.generator,
            "params": dict(self.params),
            "package_version": self.package_version,
        }


def _validated(
    group: str, matrices: Mapping[str, NDArray[Any]], keys: tuple[str, ...]
) -> dict[str, NDArray[np.float64]]:
    """Every required matrix present, float64, and finite."""
    missing = [key for key in keys if key not in matrices]
    if missing:
        raise SpecificationError(
            f"{group} is missing {', '.join(missing)}; a problem needs "
            f"{' and '.join(keys)}"
        )
    validated: dict[str, NDArray[np.float64]] = {}
    for name, matrix in matrices.items():
        array = np.asarray(matrix)
        if array.dtype != STORAGE_DTYPE:
            raise SpecificationError(
                f"{group}.{name} is {array.dtype}; problem data is stored as "
                "float64 whatever precision training runs at, because storing "
                "it at the training dtype would make one problem two"
            )
        if not np.all(np.isfinite(array)):
            raise SpecificationError(f"{group}.{name} contains a non-finite entry")
        validated[name] = array
    return validated


@dataclass(frozen=True)
class ProblemData:
    """The matrices themselves. Never regenerated.

    Attributes:
        system: `A` and `B`. Shaped `(n, n)` and `(n, m)` for a time-invariant
            system, or `(N, n, n)` and `(N, n, m)` for a time-varying one.
        cost: `Q` shaped `(N + 1, n, n)` and `R` shaped `(N, m, m)`, always
            time-stacked -- the Riccati recursion indexes `Q[N]` and `R[k]`
            even where the cost is itself constant.
        horizon: The finite horizon `N`.
        control_bound: Infinity-norm bound on the control, or `None` for an
            unconstrained problem.
    """

    system: Mapping[str, NDArray[np.float64]]
    cost: Mapping[str, NDArray[np.float64]]
    horizon: int
    control_bound: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "system", _validated("system", self.system, SYSTEM_KEYS)
        )
        object.__setattr__(self, "cost", _validated("cost", self.cost, COST_KEYS))
        if self.horizon < 1:
            raise SpecificationError(f"horizon must be positive, got {self.horizon}")
        if self.control_bound is not None and self.control_bound <= 0:
            raise SpecificationError(
                f"control_bound must be positive or None, got {self.control_bound}; "
                "an unconstrained problem omits it rather than setting it to zero"
            )
        self._require_consistent_shapes()

    def _require_consistent_shapes(self) -> None:
        a, b = self.system["A"], self.system["B"]
        if a.shape[-1] != a.shape[-2]:
            raise SpecificationError(f"A must be square, got {a.shape}")
        if b.shape[-2] != a.shape[-1]:
            raise SpecificationError(
                f"A and B disagree on the state dimension: {a.shape} against {b.shape}"
            )
        n, m = self.state_dim, self.control_dim
        for name, matrix, expected in (
            ("Q", self.cost["Q"], (self.horizon + 1, n, n)),
            ("R", self.cost["R"], (self.horizon, m, m)),
        ):
            if matrix.shape != expected:
                raise SpecificationError(
                    f"cost.{name} has shape {matrix.shape}; a horizon of "
                    f"{self.horizon} requires {expected}"
                )

    @property
    def state_dim(self) -> int:
        """`n`, read from `A` whether the system is time-varying or not."""
        return int(self.system["A"].shape[-1])

    @property
    def control_dim(self) -> int:
        """`m`, read from `B`."""
        return int(self.system["B"].shape[-1])

    @property
    def is_time_varying(self) -> bool:
        """Whether `A` carries a leading time axis."""
        return self.system["A"].ndim == 3

    def get_signature(self) -> dict[str, Any]:
        """`Signable` member: content hashes of the frozen matrices.

        Keys are sorted, so two authors who spell the same problem in a
        different order collide rather than diverge.
        """
        return {
            "type": type(self).__name__,
            "system": {
                name: hash_array(self.system[name]) for name in sorted(self.system)
            },
            "cost": {name: hash_array(self.cost[name]) for name in sorted(self.cost)},
            "horizon": self.horizon,
            "control_bound": self.control_bound,
        }


@dataclass(frozen=True)
class ProblemSpec:
    """A fully specified optimal-control problem, with its authoring record.

    Attributes:
        data: The frozen matrices, which are what identity is derived from.
        provenance: How they were produced. Recorded for the reader, excluded
            from `get_signature` by construction.
    """

    data: ProblemData
    provenance: GeneratorProvenance | None = None

    @property
    def state_dim(self) -> int:
        return self.data.state_dim

    @property
    def control_dim(self) -> int:
        return self.data.control_dim

    @property
    def horizon(self) -> int:
        return self.data.horizon

    def get_signature(self) -> dict[str, Any]:
        """The signature tree. `provenance` is absent, not merely unequal."""
        return self.data.get_signature()

    @property
    def problem_id(self) -> ProblemID:
        """This problem's identifier, derived from its data alone."""
        return problem_id(self.get_signature())

    def build(self) -> OptimalControlProblem:
        """Construct the problem these matrices describe."""
        system = LinearSystem.fully_observable(
            self.data.system["A"], self.data.system["B"]
        )
        cost = QuadraticCost(Q=self.data.cost["Q"], R=self.data.cost["R"])
        if self.data.control_bound is None:
            return OptimalControlProblem(system=system, cost=cost)
        return OptimalControlProblem(
            system=system,
            cost=cost,
            constraints=[BoxConstraint(u_max=self.data.control_bound)],
        )

    # -- storage ----------------------------------------------------------

    def save(self, path: Path | str) -> Path:
        """Freeze this problem to a single `.npz`.

        Matrices are stored verbatim; the scalars and the provenance record
        travel as a JSON side-car inside the same archive, so a problem is one
        file and nothing executes on load (the no-pickle law).
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "horizon": self.data.horizon,
            "control_bound": self.data.control_bound,
            "provenance": self.provenance.as_json() if self.provenance else None,
        }
        # One `dict[str, Any]`, unpacked once: `np.savez`'s signature is
        # `(file, *args, allow_pickle=..., **kwds)`, so unpacking two mappings
        # lets a checker match the second onto `allow_pickle`.
        payload: dict[str, Any] = {
            f"{group}{_GROUP_SEPARATOR}{name}": matrix
            for group, matrices in (
                ("system", self.data.system),
                ("cost", self.data.cost),
            )
            for name, matrix in matrices.items()
        }
        payload[METADATA_KEY] = json.dumps(metadata, sort_keys=True)
        # Compressed, because a problem's cost matrices are time-stacked
        # identities: measured at n=50, N=100, the same archive is 2.045 MB
        # stored and 0.032 MB deflated, a factor of 64. That difference is what
        # decides whether a frozen problem can be committed beside the study
        # that names it, which is what D19 assumes. Identity is unaffected --
        # a `ProblemID` is derived from the matrices, never from the bytes on
        # disk -- and `np.load` reads either form, so archives written before
        # this still open.
        np.savez_compressed(target, **payload)
        return target

    @classmethod
    def load(cls, path: Path | str) -> ProblemSpec:
        """Read a problem frozen by `save`.

        Raises:
            SpecificationError: If the archive is not a stored problem.
        """
        with np.load(Path(path), allow_pickle=False) as archive:
            if METADATA_KEY not in archive.files:
                raise SpecificationError(
                    f"{path} carries no {METADATA_KEY} record, so it is not a "
                    "stored problem"
                )
            metadata = json.loads(str(archive[METADATA_KEY]))
            groups: dict[str, dict[str, NDArray[np.float64]]] = {
                "system": {},
                "cost": {},
            }
            for key in archive.files:
                if key == METADATA_KEY:
                    continue
                group, _, name = key.partition(_GROUP_SEPARATOR)
                if group not in groups:
                    raise SpecificationError(f"{path} holds an unknown group {group!r}")
                groups[group][name] = np.asarray(archive[key])

        record = metadata.get("provenance")
        return cls(
            data=ProblemData(
                system=groups["system"],
                cost=groups["cost"],
                horizon=int(metadata["horizon"]),
                control_bound=metadata["control_bound"],
            ),
            provenance=GeneratorProvenance(**record) if record else None,
        )
