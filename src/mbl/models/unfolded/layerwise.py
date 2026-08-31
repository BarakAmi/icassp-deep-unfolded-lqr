"""Layer-wise freeze plans for warm-start deep-unfolded training (NB03
blueprint v2 §3.4, the Freeze Contract): shape-aware knowledge of
`StepSizeParameter`'s per-iteration row structure and
`RiccatiMatrixParameter`'s whole-tensor structure lives HERE, in the models
layer -- never in `engine.strategy`, which stays parameter-shape-agnostic
(SRP; a `TrainingStrategy` must not reflect on a specific family's
structured parameters).

Two freeze mechanisms, chosen per parameter's own semantics:

* `step_size` has partial-row semantics (one row per unfolding iteration),
  so it freezes via a **gradient mask** applied after `.backward()`. The
  NB05 P-resolution extension's two iteration-varying matrix parameters
  (`PerIterationRiccatiMatrixParameter`, `MatrixModulationParameter`,
  docs/planning/03_studies/nb05_ltv_box_constrained/ltv_box_constrained_lqr_benchmark.md Sec 14.7 item 4) have the
  IDENTICAL row structure -- one P^(j) / c_j per unfolding iteration -- so
  they reuse the exact same mask mechanism via `ParameterActivation`'s
  existing `step_size_rows` field (no new field: layer-wise phase j trains
  step-size row j AND the matching P^(j)/c_j row together).
* `riccati_matrix` (the SINGLE, iteration-invariant matrix of R1) has no
  partial structure (it is shared across every iteration), so it freezes
  via a **whole-tensor `requires_grad` toggle** -- verified to be a true
  freeze even under a nonzero `weight_decay`, since a `requires_grad=False`
  parameter's `.grad` stays `None` and the optimizer skips it entirely (see
  the blueprint's §7.1 empirical proof). A row-masked parameter shares the
  SAME weight-decay caveat as `step_size`, NOT the toggle-freeze's
  immunity to it (the toggle's immunity comes specifically from
  `requires_grad=False`, which a masked-but-still-trainable tensor never
  sets) -- `LayerwiseTrainingPlan.__post_init__`'s unconditional
  ``weight_decay == 0`` guard is what actually keeps every row-masked
  parameter here (`step_size` and the two new matrix parameters alike)
  correctly frozen, not the mask by itself.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import torch

from .parameters import (
    MatrixModulationParameter,
    PerIterationRiccatiMatrixParameter,
    RiccatiMatrixParameter,
    StepSizeParameter,
    UnfoldedParameter,
)


@dataclass(frozen=True)
class ParameterActivation:
    """Declares which entries of a deep-unfolded controller's learnable
    parameters train during one layer-wise training phase.

    Attributes:
        step_size_rows: Which per-iteration rows of the step-size parameter
            train this phase -- a `frozenset` of row indices, or ``"all"``
            for every row (e.g. an end-to-end refinement phase).
        train_matrix: Whether the unified Riccati-replacement matrix (when
            present) trains this phase.
    """

    step_size_rows: frozenset[int] | Literal["all"]
    train_matrix: bool

    def get_signature(self) -> dict[str, Any]:
        """`Signable` member: which rows/matrix are active this phase.

        Returns:
            ``{"type": "ParameterActivation", "step_size_rows": [...] |
            "all", "train_matrix": bool}``.
        """
        rows: Any = (
            "all" if self.step_size_rows == "all" else sorted(self.step_size_rows)
        )
        return {
            "type": type(self).__name__,
            "step_size_rows": rows,
            "train_matrix": self.train_matrix,
        }


@dataclass(frozen=True)
class LayerFreeze:
    """One phase's compiled freeze plan, keyed by the SAME names
    `UnfoldedController.as_module()` registers its raw parameters under.

    Attributes:
        grad_masks: name -> boolean mask, same shape as the raw parameter;
            after `.backward()`, ``grad *= mask`` zeroes the frozen
            entries' gradients -- the row-freeze mechanism for
            ``"step_size"``.
        trainable: name -> whether the WHOLE raw parameter trains this
            phase (a `requires_grad_` plan) -- the freeze mechanism for
            ``"riccati_matrix"``, which has no partial-row semantics.
    """

    grad_masks: Mapping[str, torch.Tensor]
    trainable: Mapping[str, bool]


class LayerFreezeBuilder:
    """Compiles a `ParameterActivation` into a `LayerFreeze` against the
    controller's LIVE parameter objects -- the only place both the
    activation's intent and the parameters' actual tensor shapes are
    available together, so an out-of-bounds row index is caught here
    (rather than silently ignored or guessed at from a bare integer, the
    blueprint's schedule<->depth validation, reframed against real data).
    """

    def build(
        self,
        activation: ParameterActivation,
        parameters: Mapping[str, UnfoldedParameter[Any]],
    ) -> LayerFreeze:
        """Compile `activation` against `parameters`' live raw tensors.

        Args:
            activation: This phase's declared activation.
            parameters: The controller's ``config.parameters`` (the SAME
                names `as_module()` registers).

        Returns:
            The compiled `LayerFreeze`.

        Raises:
            ValueError: If `activation.step_size_rows` names a row index
                outside the step-size parameter's actual row count.
            TypeError: If `parameters` holds an `UnfoldedParameter` subtype
                this builder does not know how to freeze.
        """
        grad_masks: dict[str, torch.Tensor] = {}
        trainable: dict[str, bool] = {}
        for name, parameter in parameters.items():
            if isinstance(
                parameter,
                (
                    StepSizeParameter,
                    PerIterationRiccatiMatrixParameter,
                    MatrixModulationParameter,
                ),
            ):
                grad_masks[name] = self._row_mask(name, parameter, activation)
                # The whole tensor stays requires_grad=True; row selection
                # is enforced purely by the gradient mask above.
                trainable[name] = True
            elif isinstance(parameter, RiccatiMatrixParameter):
                trainable[name] = activation.train_matrix
            else:
                raise TypeError(
                    f"LayerFreezeBuilder has no freeze rule for parameter "
                    f"{name!r} of type {type(parameter).__name__}."
                )
        return LayerFreeze(grad_masks=grad_masks, trainable=trainable)

    def _row_mask(
        self,
        name: str,
        parameter: UnfoldedParameter[Any],
        activation: ParameterActivation,
    ) -> torch.Tensor:
        """The boolean row mask for one iteration-row-structured parameter
        (`StepSizeParameter`, shape ``(num_iterations, action_dim)``;
        `PerIterationRiccatiMatrixParameter`, shape ``(num_iterations, n,
        n)``; `MatrixModulationParameter`, shape ``(num_iterations,)``) --
        the masking logic only ever looks at the raw tensor's LEADING
        dimension, so it is shared verbatim across all three shapes: one
        row per unfolding iteration is exactly what "layer-wise phase j"
        means for every one of them.

        Args:
            name: The parameter's registered name (for the error message).
            parameter: The live parameter, one of the three types above.
            activation: This phase's declared activation.

        Returns:
            A boolean mask, shape-matching `parameter`'s raw tensor, `True`
            on every entry of an active row.

        Raises:
            ValueError: If `activation.step_size_rows` names a row index
                outside ``parameter``'s actual row count.
        """
        raw = parameter.get_raw()
        num_iterations = raw.shape[0]
        if activation.step_size_rows == "all":
            return torch.ones_like(raw, dtype=torch.bool)
        out_of_bounds = sorted(
            row for row in activation.step_size_rows if row < 0 or row >= num_iterations
        )
        if out_of_bounds:
            raise ValueError(
                f"ParameterActivation.step_size_rows for {name!r} names "
                f"row(s) {out_of_bounds} outside its {num_iterations} "
                "actual row(s)."
            )
        mask = torch.zeros_like(raw, dtype=torch.bool)
        for row in activation.step_size_rows:
            mask[row] = True
        return mask
