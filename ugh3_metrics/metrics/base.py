from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable


@runtime_checkable
class Dataclass(Protocol):
    """Minimal protocol for dataclass instances."""

    __dataclass_fields__: Mapping[str, Any]


class BaseMetric(ABC):
    """Metric interface absorbing version gaps."""

    @abstractmethod
    def score(self, *, x: str, y: str) -> float:
        """Return metric value for given texts."""
        raise NotImplementedError

    @abstractmethod
    def set_params(self, params: Dataclass | dict[str, Any]) -> None:
        """Update metric parameters."""
        raise NotImplementedError
