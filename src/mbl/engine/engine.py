"""Common interface for training/inference orchestration, decoupling callers from
the concrete strategy (end-to-end, warm-start, incremental, ...)."""

from collections.abc import Mapping
from typing import Protocol, runtime_checkable


@runtime_checkable
class Engine(Protocol):
    """Coordinates a model, an optimizer, and an ExperimentTracker to run one
    training/evaluation experiment."""

    def train(self) -> Mapping[str, float]:
        """Run the training loop.

        Returns:
            The run's final scalar metrics (keys are implementation-defined,
            e.g. `Runner.train` returns ``final_``-prefixed keys).
        """
        ...

    def evaluate(self) -> Mapping[str, float]:
        """Evaluate the current model without updating its weights.

        Returns:
            Scalar evaluation metrics (keys are implementation-defined, e.g.
            `Runner.evaluate` returns ``eval_``-prefixed keys).
        """
        ...

    def run(self) -> Mapping[str, float]:
        """Convenience entrypoint: train then evaluate.

        Returns:
            The union of `train`'s and `evaluate`'s metrics.
        """
        ...
