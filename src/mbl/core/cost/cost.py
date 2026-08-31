from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Cost(ABC):
    """Abstract base class for cost functions defined over state and control
    trajectories. Concrete subclasses (e.g. `core.cost.quadratic_cost.QuadraticCost`)
    define the actual cost formula."""

    @abstractmethod
    def __call__(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        """Evaluate the cost for given state and control trajectories.

        Args:
            X: State trajectory, shape ``(batch, N+1, n)`` or ``(N+1, n)``.
            U: Control trajectory, shape ``(batch, N, m)`` or ``(N, m)``.

        Returns:
            The evaluated cost, shape and semantics defined by the subclass.

        Raises:
            NotImplementedError: Always, on this abstract base class.
        """
        raise NotImplementedError

    def get_signature(self) -> dict[str, Any]:
        """`Signable` default: just the concrete class name.

        Subclasses that carry actual cost matrices (e.g.
        `core.cost.quadratic_cost.QuadraticCost`) override this with richer
        content; this default lets any future `Cost` satisfy `Signable`
        immediately, even before it's updated (Open/Closed).

        Returns:
            ``{"type": <class name>}``.
        """
        return {"type": type(self).__name__}
