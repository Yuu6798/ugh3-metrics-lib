from abc import ABC, abstractmethod
from typing import Any


class BaseMetric(ABC):
    """Common interface for metric classes."""

    @abstractmethod
    def score(self, a: str, b: str) -> float:
        """Compute a score between two sentences."""
        ...

    @abstractmethod
    def set_params(self, **kwargs: Any) -> None:
        """Set parameters all at once."""
        ...
