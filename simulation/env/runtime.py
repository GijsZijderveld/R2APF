from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np


class BaseObstacle:
    """Base class for obstacles with distance queries."""

    type: str = "generic"

    def distance_to_surface(self, point: np.ndarray) -> float:
        raise NotImplementedError

    def representation(self, point: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Return (center, effective_radius, distance_to_center)."""

        raise NotImplementedError


class CircularObstacle(BaseObstacle):
    """Disk obstacle defined by a center and radius."""

    def __init__(self, center: Sequence[float], radius: float, *, kind: str = "rock") -> None:
        self.center = np.array(center, dtype=float)
        self.radius = float(radius)
        self.type = kind

    def distance_to_surface(self, point: np.ndarray) -> float:
        return float(np.linalg.norm(point - self.center) - self.radius)

    def representation(self, point: np.ndarray) -> Tuple[np.ndarray, float, float]:
        dist_center = float(np.linalg.norm(point - self.center))
        return self.center, self.radius, dist_center
    
    def normal_at(self, p: np.ndarray) -> np.ndarray:
        """Outward-pointing unit normal at the surface nearest to p."""
        v = p - np.array([self.x, self.y])
        n = np.linalg.norm(v)
        if n < 1e-9:
            # arbitrary fallback if exactly at center
            return np.array([1.0, 0.0])
        return v / n
    
    @property
    def x(self) -> float:
        return float(self.center[0])

    @property
    def y(self) -> float:
        return float(self.center[1])
    
    @property
    def r(self) -> float:
        return float(self.radius)
    


class WallObstacle(BaseObstacle):
    """Wall modelled as a capsule (line segment with thickness)."""

    def __init__(self, p0: Sequence[float], p1: Sequence[float], thickness: float) -> None:
        self.p0 = np.array(p0, dtype=float)
        self.p1 = np.array(p1, dtype=float)
        self.thickness = float(thickness)
        self.type = "wall"
        self._segment = self.p1 - self.p0
        self._seg_len_sq = float(np.dot(self._segment, self._segment))
        self.center = (self.p0 + self.p1) / 2.0
        self.radius = self.thickness * 0.5

    def _nearest_point_on_segment(self, point: np.ndarray) -> np.ndarray:
        if self._seg_len_sq <= 1e-12:
            return self.p0.copy()
        t = float(np.dot(point - self.p0, self._segment) / self._seg_len_sq)
        t = max(0.0, min(1.0, t))
        return self.p0 + t * self._segment

    def distance_to_surface(self, point: np.ndarray) -> float:
        nearest = self._nearest_point_on_segment(point)
        dist_midline = float(np.linalg.norm(point - nearest))
        return dist_midline - self.radius

    def representation(self, point: np.ndarray) -> Tuple[np.ndarray, float, float]:
        nearest = self._nearest_point_on_segment(point)
        dist_midline = float(np.linalg.norm(point - nearest))
        return nearest, self.radius, dist_midline


def _ensure_array_list(items: Iterable[Sequence[float]]) -> List[np.ndarray]:
    return [np.array(item, dtype=float) for item in items]


class World:
    """Runtime environment used by the simulator and agents."""

    def __init__(
        self,
        size: Tuple[float, float],
        obstacles: Iterable[BaseObstacle],
        starts: Iterable[Sequence[float]],
        goals: Iterable[Sequence[float]],
        *,
        dt: float,
        goal_radius: float,
    ) -> None:
        self.size = tuple(float(v) for v in size)
        self.obstacles: List[BaseObstacle] = list(obstacles)
        self.starts = _ensure_array_list(starts)
        self.goals = _ensure_array_list(goals)
        self.num_agents = len(self.starts)
        self.dt = float(dt)
        self.goal_radius = float(goal_radius)

    def nearest_obstacles(self, pos: np.ndarray, radius: float) -> List[Tuple[np.ndarray, float, float]]:
        hits: List[Tuple[np.ndarray, float, float]] = []
        for obs in self.obstacles:
            d_surface = obs.distance_to_surface(pos)
            if d_surface <= radius:
                center, eff_radius, dist_center = obs.representation(pos)
                hits.append((np.array(center, dtype=float), float(eff_radius), float(dist_center)))
        return hits

    def bounds(self) -> Tuple[float, float, float, float]:
        return (0.0, float(self.size[0]), 0.0, float(self.size[1]))
