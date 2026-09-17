from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Tuple

import numpy as np

from comparison.sensing import obstacle_center

Cell = Tuple[int, int]


@dataclass(frozen=True)
class GridMap:
    """Occupancy grid shared by A* and D* Lite."""

    resolution: float
    bounds: tuple[float, float, float, float]
    occupied: frozenset[Cell]
    width: int
    height: int
    connectivity: int = 8
    forbid_corner_cutting: bool = True

    @classmethod
    def from_obstacles(
        cls,
        obstacles: Iterable[Any],
        bounds: Any,
        *,
        resolution: float = 0.30,
        agent_radius: float = 0.30,
        safety_margin: float = 0.0,
        connectivity: int = 8,
        forbid_corner_cutting: bool = True,
    ) -> "GridMap":
        if resolution <= 0.0:
            raise ValueError("resolution must be positive")
        xmin, xmax, ymin, ymax = normalize_bounds(bounds)
        width = int(math.ceil((xmax - xmin) / resolution))
        height = int(math.ceil((ymax - ymin) / resolution))
        occupied: set[Cell] = set()

        for obstacle in obstacles:
            center = obstacle_center(obstacle)
            inflated = float(obstacle.radius) + agent_radius + safety_margin
            ix0 = max(0, int(math.floor((center[0] - inflated - xmin) / resolution)))
            ix1 = min(width - 1, int(math.floor((center[0] + inflated - xmin) / resolution)))
            iy0 = max(0, int(math.floor((center[1] - inflated - ymin) / resolution)))
            iy1 = min(height - 1, int(math.floor((center[1] + inflated - ymin) / resolution)))
            for ix in range(ix0, ix1 + 1):
                for iy in range(iy0, iy1 + 1):
                    point = np.array(
                        [xmin + (ix + 0.5) * resolution, ymin + (iy + 0.5) * resolution]
                    )
                    if np.linalg.norm(point - center) < inflated:
                        occupied.add((ix, iy))

        return cls(
            resolution=float(resolution),
            bounds=(xmin, xmax, ymin, ymax),
            occupied=frozenset(occupied),
            width=width,
            height=height,
            connectivity=int(connectivity),
            forbid_corner_cutting=bool(forbid_corner_cutting),
        )

    def in_bounds(self, cell: Cell) -> bool:
        return 0 <= cell[0] < self.width and 0 <= cell[1] < self.height

    def is_free(self, cell: Cell) -> bool:
        return self.in_bounds(cell) and cell not in self.occupied

    def world_to_cell(self, point: np.ndarray) -> Cell:
        xmin, _, ymin, _ = self.bounds
        point = np.asarray(point, dtype=float)
        return (
            int(math.floor((point[0] - xmin) / self.resolution)),
            int(math.floor((point[1] - ymin) / self.resolution)),
        )

    def cell_to_world(self, cell: Cell) -> np.ndarray:
        xmin, _, ymin, _ = self.bounds
        return np.array(
            [
                xmin + (cell[0] + 0.5) * self.resolution,
                ymin + (cell[1] + 0.5) * self.resolution,
            ],
            dtype=float,
        )

    def neighbors(self, cell: Cell) -> Iterator[tuple[Cell, float]]:
        if not self.is_free(cell):
            return
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        if self.connectivity == 8:
            directions += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        elif self.connectivity != 4:
            raise ValueError("connectivity must be 4 or 8")

        for dx, dy in directions:
            neighbor = (cell[0] + dx, cell[1] + dy)
            if not self.is_free(neighbor):
                continue
            if dx and dy and self.forbid_corner_cutting:
                if not self.is_free((cell[0] + dx, cell[1])):
                    continue
                if not self.is_free((cell[0], cell[1] + dy)):
                    continue
            cost = self.resolution * (math.sqrt(2.0) if dx and dy else 1.0)
            yield neighbor, cost


def normalize_bounds(bounds: Any) -> tuple[float, float, float, float]:
    if bounds is None:
        raise ValueError("Grid planners require finite environment bounds")
    values = bounds() if callable(bounds) else bounds
    if len(values) != 4:
        raise ValueError("bounds must be (xmin, xmax, ymin, ymax)")
    xmin, xmax, ymin, ymax = map(float, values)
    if not xmin < xmax or not ymin < ymax:
        raise ValueError("bounds must have positive area")
    return xmin, xmax, ymin, ymax


def octile_distance(a: Cell, b: Cell, resolution: float) -> float:
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    return resolution * (max(dx, dy) + (math.sqrt(2.0) - 1.0) * min(dx, dy))


def cells_to_path(
    grid: GridMap,
    cells: list[Cell],
    start: np.ndarray,
    goal: np.ndarray,
) -> list[np.ndarray]:
    if not cells:
        return []
    path = [np.asarray(start, dtype=float).copy()]
    path.extend(grid.cell_to_world(cell) for cell in cells[1:-1])
    path.append(np.asarray(goal, dtype=float).copy())
    return path
