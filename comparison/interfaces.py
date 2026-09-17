from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, Sequence, runtime_checkable

import numpy as np


@dataclass
class PlanningResult:
    """Planner output with common and optional planner-specific diagnostics."""

    success: bool
    path: Sequence[np.ndarray] = field(default_factory=list)
    planning_time_s: float = 0.0
    failure_reason: Optional[str] = None
    collision_checks: Optional[int] = None
    expanded_nodes: Optional[int] = None
    samples: Optional[int] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkObservation:
    """Information made available to a benchmark agent for one simulation step."""

    environment: Any
    time_step: int
    bounds: Any = None


@runtime_checkable
class Planner(Protocol):
    """Common contract for APF, graph-search, and sampling-based planners."""

    name: str

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        known_obstacles: Sequence[Any],
        bounds: Any = None,
    ) -> PlanningResult:
        ...


@runtime_checkable
class BenchmarkAgent(Protocol):
    """Interface consumed by the common benchmark runner."""

    @property
    def position(self) -> np.ndarray:
        ...

    @property
    def goal(self) -> np.ndarray:
        ...

    @property
    def reached_goal(self) -> bool:
        ...

    def reset(self, environment: Any, seed: int) -> None:
        ...

    def step(
        self,
        observation: BenchmarkObservation,
        dt: float,
        speed_limit: float,
    ) -> None:
        ...

    def diagnostics(self) -> Dict[str, Any]:
        ...
