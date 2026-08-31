"""The `Experiment` entity — the unit of reproducible research (REFACTOR_PLAN
v3, T3.d): one frozen, signable object binding the optimization problem, the
competing controllers, and the evaluation protocol. `CaseStudy` (T3.a) remains
the reusable *family declaration*; an `Experiment` is such a declaration bound
to concrete settings, seeds, and compute context — everything a notebook
previously wired by hand becomes constructor arguments of one object.

Execution is `runner.run_experiment(experiment, ...) -> ExperimentReport`,
the single unified entry point for all research execution.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import torch

from ..applications.case_study import CaseStudy
from ..applications.factories import BatchSpec, ProblemFactory
from ..applications.recipes.base import ModelRecipe, build_default_recipe_registry
from ..core.runtime import Backend, ComputeContext
from ..core.utils.signing import compute_signature_digest
from ..spec.contender import ContenderSpec as SpecContenderSpec
from ..spec.contender import RecipeRegistry

#: One evaluation batch: ``(initial_state, process_noise, measurement_noise)``.
EvaluationBatch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class ContenderSpec(SpecContenderSpec):
    """The Tier-3 `spec.ContenderSpec`, bound to this project's own recipe
    registry (Stage 2 Phase B).

    The grammar's contender is deliberately registry-agnostic: Tier 3 may not
    import `applications`, because `build_default_recipe_registry()` pulls in
    every recipe module and with it the engine, the models and the persistence
    layer — the dependency that made `Experiment` un-reusable in the first
    place. So the specification declares what it needs structurally and the
    registry is injected.

    This subclass is where the injection happens for this project. It adds no
    fields and changes no behaviour; it supplies the default that Tier 3 has no
    way to name, which is what keeps every existing call site, study module and
    notebook working unchanged.
    """

    def resolve(self, registry: RecipeRegistry | None = None) -> ModelRecipe:
        """The concrete `ModelRecipe` this spec names.

        Args:
            registry: The recipe registry to resolve through; defaults to
                the built-in one (`build_default_recipe_registry`).

        Returns:
            The composed recipe (label override applied when set). Narrowed
            from the protocol Tier 3 returns: everything downstream of a
            contender in this package builds controllers and engines from it.
        """
        # `is not None`, not truthiness: a registry is a container, and an
        # empty one passed explicitly must fail on the unknown family rather
        # than silently resolve through the built-in registry instead.
        supplied = registry if registry is not None else build_default_recipe_registry()
        return cast("ModelRecipe", super().resolve(supplied))


@dataclass(frozen=True)
class EvaluationProtocol:
    """The shared ONLINE evaluation contract every contender is scored under:
    one seeded batch specification, drawn `n_batches` times from a single
    advancing stream — the same realizations for every contender **by
    construction** (batches are materialized once per `run_experiment`, never
    per contender).

    Attributes:
        batch_spec: The seeded Gaussian batch specification (dimensions,
            horizon, batch size, seed, noise scales).
        n_batches: Number of evaluation batches drawn and averaged over.
    """

    batch_spec: BatchSpec
    n_batches: int = 1

    def __post_init__(self) -> None:
        if self.n_batches < 1:
            raise ValueError(f"n_batches must be >= 1, got {self.n_batches}.")

    def build_batches(self, ctx: ComputeContext) -> tuple[EvaluationBatch, ...]:
        """Materialize the evaluation batches, once.

        Batches are always authored on the torch substrate (every policy in
        the tree consumes torch tensors; NumPy-native analytic policies
        broadcast against them transparently), at the context's precision —
        the common-noise-realization law.

        Args:
            ctx: The experiment's compute context (precision authority).

        Returns:
            `n_batches` sequential draws from one seeded stream.
        """
        sampler, _distributions = self.batch_spec.build(
            Backend.TORCH, torch_dtype=ctx.torch_dtype, torch_device=ctx.torch_device
        )
        return tuple(sampler() for _ in range(self.n_batches))

    def get_signature(self) -> dict[str, Any]:
        """`Signable` member: the full evaluation specification."""
        return {
            "type": type(self).__name__,
            "batch_spec": self.batch_spec.get_signature(),
            "n_batches": self.n_batches,
        }


@dataclass(frozen=True)
class Experiment:
    """One research run, fully bound and signable end-to-end (T3.d): the apex
    Tier-3 abstraction, one level above `CaseStudy`/`ModelRecipe`.

    Attributes:
        name: The experiment's identity (run/registry naming).
        problem: The signable problem specification (factory + params).
        contenders: The competing controller specs, registry-resolvable.
        evaluation: The shared online evaluation protocol.
        ctx: The compute context — signature-relevant (backend, device, and
            precision all change numerics), in contrast to `ValidationMode`,
            which never enters signatures.
    """

    name: str
    problem: ProblemFactory
    contenders: tuple[ContenderSpec, ...]
    evaluation: EvaluationProtocol
    ctx: ComputeContext

    def __post_init__(self) -> None:
        labels = [spec.resolved_label for spec in self.contenders]
        if len(set(labels)) != len(labels):
            raise ValueError(f"Contender labels must be unique, got {labels}.")

    @classmethod
    def from_case_study(
        cls,
        case_study: CaseStudy,
        *,
        evaluation: EvaluationProtocol | None = None,
        name: str | None = None,
    ) -> "Experiment":
        """Bind a reusable `CaseStudy` declaration into an `Experiment`.

        Args:
            case_study: The family declaration (problem factory, shared
                batch spec, recipes, context).
            evaluation: Optional evaluation override; defaults to one batch
                of the case study's own `batch_spec`.
            name: Optional name override; defaults to the case study's.

        Returns:
            The bound `Experiment`.
        """
        return cls(
            name=name or case_study.name,
            problem=case_study.problem,
            contenders=tuple(
                ContenderSpec.from_recipe(recipe) for recipe in case_study.recipes
            ),
            evaluation=evaluation
            or EvaluationProtocol(batch_spec=case_study.batch_spec),
            ctx=case_study.ctx,
        )

    def contender_signature_tree(self, spec: ContenderSpec) -> dict[str, Any]:
        """The per-contender content-signature tree — the ``(problem,
        contender, evaluation, context)`` sub-key the two-level cache
        granularity (T3.e) derives everything from. The experiment `name`
        deliberately does NOT participate: renaming an experiment must not
        orphan its results.

        Args:
            spec: The contender to key.

        Returns:
            The nested signature tree.
        """
        return {
            "problem": self.problem.get_signature(),
            "contender": spec.get_signature(),
            "evaluation": self.evaluation.get_signature(),
            "compute_context": self.ctx.get_signature(),
        }

    def contender_content_digest(self, spec: ContenderSpec) -> str:
        """Compact content digest of `contender_signature_tree` (stamp-free;
        the cache composes it with the code-provenance stamp)."""
        return compute_signature_digest(self.contender_signature_tree(spec))

    def get_signature(self) -> dict[str, Any]:
        """`Signable` member: the complete experiment specification.

        Returns:
            ``{"type":, "name":, "problem":, "contenders": {label: ...},
            "evaluation":, "compute_context":}``.
        """
        return {
            "type": type(self).__name__,
            "name": self.name,
            "problem": self.problem.get_signature(),
            "contenders": {
                spec.resolved_label: spec.get_signature() for spec in self.contenders
            },
            "evaluation": self.evaluation.get_signature(),
            "compute_context": self.ctx.get_signature(),
        }

    def signature_digest(self) -> str:
        """Compact composite identity of the whole experiment."""
        return compute_signature_digest(self.get_signature())


@dataclass(frozen=True)
class ContenderResult:
    """One contender's persisted product: scalar metrics plus the dense
    evaluation payload, with full cache provenance.

    Attributes:
        label: The contender's instance label.
        family: The recipe family name.
        cache_key: The stamped cache key this result is stored under.
        content_digest: The stamp-free content digest (stale-detection aid).
        disposition: ``"fresh"`` (computed this execution), ``"hit"``
            (served from cache), or ``"recompute"`` (forced re-execution).
        run_dir: The producing run directory, as a string.
        metrics: Flat scalar metrics (synthesis finals + evaluation).
        arrays: Dense evaluation payloads (e.g. per-batch expected costs).
    """

    label: str
    family: str
    cache_key: str
    content_digest: str
    disposition: str
    run_dir: str | None
    metrics: Mapping[str, float]
    arrays: Mapping[str, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentReport:
    """The persisted product of one `run_experiment` execution (T3.d).

    Attributes:
        experiment_name: The experiment's declared name.
        experiment_signature: The experiment's composite signature digest.
        provenance_stamp: The code-provenance stamp the execution ran under.
        results: Per-contender results, keyed by contender label.
    """

    experiment_name: str
    experiment_signature: str
    provenance_stamp: str
    results: Mapping[str, ContenderResult]

    def metric(self, label: str, key: str) -> float:
        """Convenience accessor: one contender's scalar metric.

        Args:
            label: The contender label.
            key: The metric key.

        Returns:
            The metric value.
        """
        return float(self.results[label].metrics[key])


def problem_dimensions(problem: ProblemFactory) -> dict[str, int]:
    """The ``(n, m, T)`` triple the history registry records (T3.j).

    Args:
        problem: The experiment's problem factory.

    Returns:
        ``{"state_dim":, "control_dim":, "horizon":}``.
    """
    return {
        "state_dim": problem.state_dim,
        "control_dim": problem.control_dim,
        "horizon": problem.horizon,
    }
