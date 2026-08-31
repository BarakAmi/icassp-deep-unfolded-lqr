"""Learnable parameters for deep-unfolded (UnfoldedController) models: each
`UnfoldedParameter` owns a raw, unconstrained `nn.Parameter` and a
reparameterization (`get`) enforcing the mathematical constraint the raw form
cannot violate (e.g. a bounded step size, or a PSD matrix)."""

import math
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn

from ..base import Config
from ...core.utils.predicates import is_positive_semi_definite
from ...core.utils.signing import hash_array


@dataclass(frozen=True)
class UnfoldedParameterConfig(Config):
    """Base configuration shared by every `UnfoldedParameter`.

    Attributes:
        name: Human-readable identifier (used in `__str__`/logging).
        num_iterations: Number of unfolding iterations this parameter has one
            value per (its leading dimension).
        dtype: The parameter tensor's dtype.
        device: The parameter tensor's device; defaults to CPU for
            deterministic, machine-independent behavior -- callers that want
            GPU execution must opt in explicitly.
    """

    name: str
    num_iterations: int
    # kw_only so that subclasses can add their own required, positional fields
    # after these without violating dataclass field-ordering rules.
    dtype: torch.dtype = field(
        default=torch.float32, kw_only=True
    )  # For higher performance, use float32 or below
    # Defaults to CPU for deterministic, machine-independent behavior. Callers that want
    # GPU execution must opt in explicitly, e.g. UnfoldedParameterConfig(..., device=torch.device("cuda")).
    device: torch.device = field(default=torch.device("cpu"), kw_only=True)


class UnfoldedParameter[C: UnfoldedParameterConfig](ABC):
    """Base class for a learnable, reparameterized unfolding parameter,
    generic over its own concrete config type (so subclass bodies read their
    extra config fields type-safely)."""

    def __init__(self, unfolding_config: C) -> None:
        """
        Args:
            unfolding_config: This parameter's configuration; `initialize` is
                called immediately to construct the underlying `nn.Parameter`.
        """
        self.config: C = unfolding_config
        self.initialize()

    def get_numpy(self) -> np.ndarray:
        """Return `self.get()` as a detached numpy array.

        Returns:
            The reparameterized value (see `get`), converted via
            ``.detach().cpu().numpy()``.
        """
        return self.get().detach().cpu().numpy()

    @abstractmethod
    def initialize(self) -> None:
        """Construct (or reconstruct) the underlying raw `nn.Parameter`.

        Raises:
            NotImplementedError: If a subclass does not implement this method.
        """
        pass

    @abstractmethod
    def get(self) -> torch.Tensor:
        """Return the reparameterized (constraint-satisfying) value.

        Raises:
            NotImplementedError: If a subclass does not implement this method.
        """
        pass

    @abstractmethod
    def get_raw(self) -> torch.Tensor:
        """Return the underlying raw, unconstrained `nn.Parameter`.

        Raises:
            NotImplementedError: If a subclass does not implement this method.
        """
        pass

    @abstractmethod
    def load(self, value: np.ndarray) -> None:
        """Load a previously saved value, inverting the reparameterization to
        set the raw parameter.

        Args:
            value: The saved (reparameterized-space) value to load.

        Raises:
            NotImplementedError: If a subclass does not implement this method.
        """
        pass

    def __str__(self) -> str:
        return f"{self.config.name} = {self.get_numpy()}"

    def get_signature(self) -> dict[str, Any]:
        """`Signable` member: signs the *init* `Config` (name, num_iterations,
        dtype, device, and any subclass-specific scalars/arrays), never the
        live/trained `.get()` value.

        The trained value already has a home as a `ParameterSnapshotCallback`
        artifact; signing it here would also reintroduce the exact
        cross-version/GPU nondeterminism risk specification-time tensors do
        not have (see docs/planning/01_foundation/phase_1d_persistence_plan_v2.md Part 5).
        Array-valued config fields (e.g. `RiccatiMatrixParameterConfig.L_init`)
        are content-hashed, never embedded raw.

        Returns:
            ``{"type": <class name>, **config fields}``, any array-valued
            field replaced by its `hash_array` digest.
        """
        fields = asdict(self.config)
        for key, value in fields.items():
            if isinstance(value, np.ndarray):
                fields[key] = hash_array(value)
        return {"type": type(self).__name__, **fields}


def require_saturating_bound(
    owner: str, stem: str, init: float, maximum: float
) -> None:
    """Refuse `init` unless it lies strictly inside ``(0, maximum)``.

    The one answer to "what is a legal value for a sigmoid-reparameterized
    scalar", shared by the two families that have one and re-used by the Tier-4
    recipes so the tiers cannot disagree. Their clamps are identical to the bit
    -- both saturate at the fraction 0.9999999999989999.

    The bounds are STRICT on both sides. ``init == maximum`` is *already*
    saturated: a declared 1.0 against a bound of 1.0 executes
    0.9999999999989999, not 1.0. And ``init <= 0`` is silently rewritten to a
    positive ``+1e-12``, so a sign typo becomes a tiny forward step rather than
    an error.

    The predicate is the one FROZEN `lqr.configs.UnfoldingConfig` (D11) and the
    legacy `StandardLQRConfig` / `BoxConstraintLQRConfig` have always used;
    it is copied here rather than reinvented, because the tier that never
    inherited it is the only one that silently clamps.

    Args:
        owner: The declaring type's name, for the message.
        stem: The field-name stem (``"alpha"``, ``"modulation"``,
            ``"step_size"``), so the message names the field the author wrote
            rather than the one the parameter tier calls it.
        init: The declared initial value.
        maximum: The declared saturating bound.

    Raises:
        ValueError: If `maximum` is not positive, or `init` does not lie
            strictly inside ``(0, maximum)``.
    """
    if not maximum > 0.0:
        raise ValueError(
            f"{owner}.{stem}_max={maximum!r} must be positive; the sigmoid "
            f"reparameterization saturates to it, so a non-positive bound "
            f"executes a {stem} of exactly 0.0 -- a parameter that never moves."
        )
    if not 0.0 < init < maximum:
        executed = maximum * (1 - 1e-12) if init >= maximum else 1e-12
        raise ValueError(
            f"{owner}.{stem}_init={init!r} must lie strictly inside "
            f"(0, {stem}_max={maximum!r}); the sigmoid reparameterization "
            f"would silently execute {executed!r} instead, while the declared "
            f"value would still move the model identity -- two identifiers for "
            f"one model."
        )


@dataclass(frozen=True)
class StepSizeParameterConfig(UnfoldedParameterConfig):
    """Configuration for `StepSizeParameter`.

    Attributes:
        action_dim: Control dimension ``m`` (one step size per control
            component, per iteration).
        alpha_init: Initial step size, reconstructed (approximately) at
            `initialize` time.
        alpha_max: Upper bound the sigmoid reparameterization saturates to.
    """

    action_dim: int
    alpha_init: float
    alpha_max: float

    def __post_init__(self) -> None:
        """Refuse a step size the reparameterization could only reach by
        saturating.

        `initialize` clamps ``alpha_init / alpha_max`` into ``(tol, 1 - tol)``,
        so every declaration at or above the bound executes the SAME tensor and
        every declaration at or below zero executes ``+tol``. Because
        `alpha_init` is signed into the model identity while the clamp collapses
        whole intervals of it onto one executed value, two declarations that
        train to a bit-identical model can carry two different identifiers --
        which is a store that cannot answer whether a model has been trained.

        The predicate is the one FROZEN `lqr.configs.UnfoldingConfig` and the
        legacy `StandardLQRConfig` / `BoxConstraintLQRConfig` already use,
        copied rather than reinvented so the tiers cannot disagree about what
        a legal step is.

        Raises:
            ValueError: If `alpha_max` is not positive, or `alpha_init` does
                not lie strictly inside ``(0, alpha_max)``.
        """
        require_saturating_bound(
            type(self).__name__, "alpha", self.alpha_init, self.alpha_max
        )


class StepSizeParameter(UnfoldedParameter[StepSizeParameterConfig]):
    """Manages learnable step-size parameters (alphas) with sigmoid reparameterization."""

    tol = 1e-12

    def initialize(self) -> None:
        """Initialize rho such that sigmoid(rho) * alpha_max ≈ alpha_init.

        alpha_init/alpha_max are plain Python floats (not tensors), so this
        clamps/logs with math/builtins rather than torch.clip/torch.log --
        both require a Tensor as their first argument and raise TypeError on
        a bare float.
        """
        raw_frac = self.config.alpha_init / max(self.tol, self.config.alpha_max)
        frac = min(max(raw_frac, self.tol), 1 - self.tol)
        rho_init = math.log(frac / (1.0 - frac))
        rho_array = rho_init * torch.ones(
            (self.config.num_iterations, self.config.action_dim),
            dtype=self.config.dtype,
            device=self.config.device,
        )
        self.rho = nn.Parameter(rho_array)

    def get(self) -> torch.Tensor:
        """Return parameters of shape (num_iterations, action_dim) in [0, alpha_max].

        Returns:
            ``sigmoid(rho) * alpha_max``, shape ``(num_iterations, action_dim)``.
        """
        return torch.sigmoid(self.get_raw()) * self.config.alpha_max

    def get_raw(self) -> torch.Tensor:
        """Return raw parameters rho of shape (num_iterations, action_dim).

        Returns:
            The underlying `nn.Parameter` `self.rho`.
        """
        return self.rho

    def for_iteration(self, i: int) -> torch.Tensor:
        """`models.iterative.step_size.StepSizeProvider` member: the live,
        autograd-connected step size for iteration `i`, so this learnable
        parameter structurally satisfies the same protocol as the fixed
        `StepSizeSchedule` -- both are usable wherever a `GradientDescentRefinement`
        subclass (`RiccatiRefinement`, `StepSizeRefinement`, or an analytical
        solver's refinement) expects a `StepSizeProvider`.

        Args:
            i: The gradient-descent iteration index.

        Returns:
            ``self.get()[i]``, shape ``(action_dim,)``.
        """
        return self.get()[i]

    def load(self, alphas: np.ndarray) -> None:
        """Load alphas from file and update rho via inverse logit.

        Args:
            alphas: Step-size values to load, shape ``(num_iterations, action_dim)``;
                clipped into ``(tol, alpha_max - tol)`` before inversion.
        """
        alphas = np.clip(alphas, self.tol, self.config.alpha_max - self.tol)
        frac = alphas / self.config.alpha_max
        rho_values = np.log(frac / (1.0 - frac))
        with torch.no_grad():
            self.rho.copy_(
                torch.tensor(
                    rho_values, dtype=self.config.dtype, device=self.config.device
                )
            )


@dataclass(frozen=True)
class RiccatiMatrixParameterConfig(UnfoldedParameterConfig):
    """Configuration for `RiccatiMatrixParameter`.

    Attributes:
        state_dim: State dimension ``n`` (the Riccati matrix is ``n x n``).
        L_init: Optional initial Cholesky factor, shape ``(n, n)``; defaults
            to the identity (so the initial Riccati matrix is also identity).
    """

    state_dim: int
    L_init: np.ndarray | None = None


class RiccatiMatrixParameter(UnfoldedParameter[RiccatiMatrixParameterConfig]):
    """Manages learnable Riccati matrix parameters.

    Riccati matrix is parameterized via its Cholesky factor L such that L @ L.T is PSD.
    """

    def initialize(self) -> None:
        """Initialize raw Cholesky factor such that L @ L.T = I."""
        # L = I is the Cholesky factor of identity matrix I @ I.T = I
        L_init = (
            self.config.L_init
            if self.config.L_init is not None
            else np.eye(self.config.state_dim)
        )
        self.L = nn.Parameter(
            torch.tensor(L_init, dtype=self.config.dtype, device=self.config.device)
        )

    def get(self) -> torch.Tensor:
        """Return Riccati matrix as PSD via Cholesky decomposition L @ L.T.

        Returns:
            ``L @ L.T``, shape ``(state_dim, state_dim)``, guaranteed PSD by
            construction for any real `L`.
        """
        L = torch.tril(self.get_raw())
        return L @ L.T

    def get_raw(self) -> torch.Tensor:
        """Return raw Cholesky factor L.

        Returns:
            The underlying `nn.Parameter` `self.L`.
        """
        return self.L

    def load(self, riccati_matrix: np.ndarray) -> None:
        """Load Riccati matrix from file and convert to Cholesky factor L.

        Args:
            riccati_matrix: The PSD matrix to load, shape ``(state_dim, state_dim)``.

        Raises:
            ValueError: If `riccati_matrix` is not positive semi-definite (a
                precondition for the Cholesky decomposition to exist).
        """
        # Ensure input is PSD
        if not is_positive_semi_definite(riccati_matrix):
            raise ValueError("Loaded Riccati matrix must be positive semi-definite.")

        # Compute Cholesky decomposition to recover L such that L @ L.T = riccati_matrix
        L_values = np.linalg.cholesky(riccati_matrix)
        with torch.no_grad():
            self.L.copy_(
                torch.tensor(
                    L_values, dtype=self.config.dtype, device=self.config.device
                )
            )

    def __str__(self) -> str:
        return f"Learnable Riccati matrix = {self.get_numpy()}"
        # return f"Riccati matrix = {self.get_numpy()}\nIs positive semi-definite: {is_positive_semi_definite(self.get_numpy())}"


@dataclass(frozen=True)
class PerIterationRiccatiMatrixParameterConfig(UnfoldedParameterConfig):
    """Configuration for `PerIterationRiccatiMatrixParameter`.

    Attributes:
        state_dim: State dimension ``n`` (each per-iteration Riccati matrix
            is ``n x n``).
        L_init: Optional initial Cholesky factor stack, shape
            ``(num_iterations, n, n)``; defaults to ``num_iterations``
            copies of the identity (so every initial P^(j) is also
            identity, matching `RiccatiMatrixParameter`'s own default and
            making a fresh controller's forward pass bit-identical to the
            single-matrix R1 contender before any training step).
    """

    state_dim: int
    L_init: np.ndarray | None = None


class PerIterationRiccatiMatrixParameter(
    UnfoldedParameter[PerIterationRiccatiMatrixParameterConfig]
):
    """Manages ``num_iterations`` fully independent, learnable Riccati
    matrices P^(0), ..., P^(J-1) -- one per unfolding iteration (the NB05
    plan's rung R2, docs/planning/03_studies/nb05_ltv_box_constrained/ltv_box_constrained_lqr_benchmark.md Sec 14) --
    each parameterized via its own Cholesky factor exactly like
    `RiccatiMatrixParameter`, batched over the leading iteration axis
    (`torch.tril`/`@`/`.mT` all batch over leading dims identically to the
    single-matrix case, so this is one batched op, not a Python loop over
    the raw tensor)."""

    def initialize(self) -> None:
        """Initialize each per-iteration raw Cholesky factor to the
        identity's (``L^(j) @ L^(j).T = I``) unless `L_init` overrides it."""
        num_iterations, n = self.config.num_iterations, self.config.state_dim
        L_init = (
            self.config.L_init
            if self.config.L_init is not None
            else np.tile(np.eye(n), (num_iterations, 1, 1))
        )
        self.L = nn.Parameter(
            torch.tensor(L_init, dtype=self.config.dtype, device=self.config.device)
        )

    def get(self) -> torch.Tensor:
        """Return the per-iteration Riccati matrix stack, each slice PSD.

        Returns:
            ``L @ L.mT`` (batched over the leading iteration axis), shape
            ``(num_iterations, state_dim, state_dim)``.
        """
        L = torch.tril(self.get_raw())
        return L @ L.mT

    def get_raw(self) -> torch.Tensor:
        """Return the raw per-iteration Cholesky factor stack.

        Returns:
            The underlying `nn.Parameter` `self.L`, shape
            ``(num_iterations, state_dim, state_dim)``.
        """
        return self.L

    def load(self, riccati_matrix: np.ndarray) -> None:
        """Load a per-iteration Riccati matrix stack and convert each slice
        to its Cholesky factor.

        Args:
            riccati_matrix: The PSD matrices to load, shape
                ``(num_iterations, state_dim, state_dim)``.

        Raises:
            ValueError: If any slice is not positive semi-definite.
        """
        L_values = np.empty_like(riccati_matrix)
        for j in range(riccati_matrix.shape[0]):
            if not is_positive_semi_definite(riccati_matrix[j]):
                raise ValueError(
                    "Loaded per-iteration Riccati matrix at iteration "
                    f"{j} must be positive semi-definite."
                )
            L_values[j] = np.linalg.cholesky(riccati_matrix[j])
        with torch.no_grad():
            self.L.copy_(
                torch.tensor(
                    L_values, dtype=self.config.dtype, device=self.config.device
                )
            )

    def __str__(self) -> str:
        return f"Learnable per-iteration Riccati matrices = {self.get_numpy()}"


@dataclass(frozen=True)
class MatrixModulationParameterConfig(UnfoldedParameterConfig):
    """Configuration for `MatrixModulationParameter`.

    Attributes:
        modulation_init: Initial modulation scale, reconstructed
            (approximately) at `initialize` time.
        modulation_max: Upper bound the sigmoid reparameterization
            saturates to.
    """

    modulation_init: float
    modulation_max: float

    def __post_init__(self) -> None:
        """Refuse a modulation scale the reparameterization could only reach by
        saturating -- the same guard, and the same measured clamp, as
        `StepSizeParameterConfig`.

        This tier is the ONLY one that can ever fire for this family: its
        `modulation_init` is hard-coded by the recipe layer against a
        module-level `MATRIX_MODULATION_MAX`, so no study document can declare
        it and a Tier-4 refusal would be unreachable.

        Raises:
            ValueError: If `modulation_max` is not positive, or
                `modulation_init` does not lie strictly inside
                ``(0, modulation_max)``.
        """
        require_saturating_bound(
            type(self).__name__, "modulation", self.modulation_init, self.modulation_max
        )


class MatrixModulationParameter(UnfoldedParameter[MatrixModulationParameterConfig]):
    """Manages R1.5's per-iteration positive scale c_j that modulates a
    SHARED learned Riccati matrix, P^(j) = c_j * P (NB05 plan Sec 14.9) --
    one learned scalar per unfolding iteration, sigmoid-reparameterized into
    ``(0, modulation_max)`` exactly like `StepSizeParameter`'s own bounded,
    positive construction. Kept as its own distinct type (rather than a
    literal reuse of `StepSizeParameter`) so `LayerFreezeBuilder`'s
    `isinstance` dispatch -- and any future provenance/logging code -- can
    never conflate "the actual learned step size" with "the P-modulation
    scalar", even though the two share a reparameterization shape and a
    layer-wise row-freeze mechanism."""

    tol = 1e-12

    def initialize(self) -> None:
        """Initialize rho such that sigmoid(rho) * modulation_max ≈
        modulation_init (mirrors `StepSizeParameter.initialize`)."""
        raw_frac = self.config.modulation_init / max(
            self.tol, self.config.modulation_max
        )
        frac = min(max(raw_frac, self.tol), 1 - self.tol)
        rho_init = math.log(frac / (1.0 - frac))
        rho_array = rho_init * torch.ones(
            (self.config.num_iterations,),
            dtype=self.config.dtype,
            device=self.config.device,
        )
        self.rho = nn.Parameter(rho_array)

    def get(self) -> torch.Tensor:
        """Return the per-iteration modulation scale, shape
        ``(num_iterations,)``, in ``(0, modulation_max)``.

        Returns:
            ``sigmoid(rho) * modulation_max``.
        """
        return torch.sigmoid(self.get_raw()) * self.config.modulation_max

    def get_raw(self) -> torch.Tensor:
        """Return the raw per-iteration modulation parameter rho.

        Returns:
            The underlying `nn.Parameter` `self.rho`, shape
            ``(num_iterations,)``.
        """
        return self.rho

    def load(self, modulation: np.ndarray) -> None:
        """Load per-iteration modulation scales and update rho via inverse
        logit.

        Args:
            modulation: Values to load, shape ``(num_iterations,)``;
                clipped into ``(tol, modulation_max - tol)`` before
                inversion.
        """
        modulation = np.clip(
            modulation, self.tol, self.config.modulation_max - self.tol
        )
        frac = modulation / self.config.modulation_max
        rho_values = np.log(frac / (1.0 - frac))
        with torch.no_grad():
            self.rho.copy_(
                torch.tensor(
                    rho_values, dtype=self.config.dtype, device=self.config.device
                )
            )

    def __str__(self) -> str:
        return f"Learnable matrix modulation c_j = {self.get_numpy()}"
