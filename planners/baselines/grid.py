from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Tuple

import numpy as np

from comparison.sensing import obstacle_center

Cell = Tuple[int, int]
Edge = Tuple[Cell, Cell]
Circle = Tuple[float, float, float]


def canonical_edge(a: Cell, b: Cell) -> Edge:
    return (a, b) if a <= b else (b, a)


def segment_intersects_circle(start: np.ndarray, end: np.ndarray, circle: Circle) -> bool:
    center = np.array(circle[:2], dtype=float)
    segment = end - start
    denominator = float(np.dot(segment, segment))
    if denominator <= 1e-12:
        distance = float(np.linalg.norm(center - start))
    else:
        fraction = float(np.dot(center - start, segment) / denominator)
        fraction = max(0.0, min(1.0, fraction))
        distance = float(np.linalg.norm(center - (start + fraction * segment)))
    return distance < circle[2]


@dataclass(frozen=True)
class GridMap:
    """Occupancy grid with cached continuous-validity graph edges."""

    resolution: float
    bounds: tuple[float, float, float, float]
    occupied: frozenset[Cell]
    blocked_edges: frozenset[Edge]
    width: int
    height: int
    inflated_obstacles: tuple[Circle, ...] = ()
    connectivity: int = 8
    forbid_corner_cutting: bool = True

    @classmethod
    def from_obstacles(
        cls, obstacles: Iterable[Any], bounds: Any, *, resolution: float = 0.30,
        agent_radius: float = 0.30, safety_margin: float = 0.0,
        connectivity: int = 8, forbid_corner_cutting: bool = True,
    ) -> "GridMap":
        if resolution <= 0.0:
            raise ValueError("resolution must be positive")
        xmin, xmax, ymin, ymax = normalize_bounds(bounds)
        grid = cls(
            resolution=float(resolution), bounds=(xmin, xmax, ymin, ymax),
            occupied=frozenset(), blocked_edges=frozenset(),
            width=int(math.ceil((xmax - xmin) / resolution)),
            height=int(math.ceil((ymax - ymin) / resolution)),
            connectivity=int(connectivity),
            forbid_corner_cutting=bool(forbid_corner_cutting),
        )
        return grid.with_added_obstacles(
            obstacles, agent_radius=agent_radius, safety_margin=safety_margin
        )

    def with_added_obstacles(
        self, obstacles: Iterable[Any], *, agent_radius: float,
        safety_margin: float = 0.0,
    ) -> "GridMap":
        """Return a grid after locally rasterizing only previously unseen circles."""
        existing = set(self.inflated_obstacles)
        additions: list[Circle] = []
        for obstacle in obstacles:
            center = obstacle_center(obstacle)
            circle = (
                float(center[0]), float(center[1]),
                float(obstacle.radius) + float(agent_radius) + float(safety_margin),
            )
            if circle not in existing:
                existing.add(circle)
                additions.append(circle)
        if not additions:
            return self

        occupied = set(self.occupied)
        blocked_edges = set(self.blocked_edges)
        for circle in additions:
            occupied.update(self.cells_covered_by_circle(circle))
            blocked_edges.update(self.edges_blocked_by_circle(circle))
        return GridMap(
            resolution=self.resolution, bounds=self.bounds,
            occupied=frozenset(occupied), blocked_edges=frozenset(blocked_edges),
            width=self.width, height=self.height,
            inflated_obstacles=tuple(sorted(existing)),
            connectivity=self.connectivity,
            forbid_corner_cutting=self.forbid_corner_cutting,
        )

    def in_bounds(self, cell: Cell) -> bool:
        return 0 <= cell[0] < self.width and 0 <= cell[1] < self.height

    def is_free(self, cell: Cell) -> bool:
        return self.in_bounds(cell) and cell not in self.occupied

    def world_to_cell(self, point: np.ndarray) -> Cell:
        xmin, _, ymin, _ = self.bounds
        point = np.asarray(point, dtype=float)
        return (int(math.floor((point[0] - xmin) / self.resolution)),
                int(math.floor((point[1] - ymin) / self.resolution)))

    def cell_to_world(self, cell: Cell) -> np.ndarray:
        xmin, _, ymin, _ = self.bounds
        return np.array([xmin + (cell[0] + 0.5) * self.resolution,
                         ymin + (cell[1] + 0.5) * self.resolution], dtype=float)

    def cells_covered_by_circle(self, circle: Circle) -> set[Cell]:
        x, y, radius = circle
        xmin, _, ymin, _ = self.bounds
        ix0 = max(0, int(math.floor((x - radius - xmin) / self.resolution)))
        ix1 = min(self.width - 1, int(math.floor((x + radius - xmin) / self.resolution)))
        iy0 = max(0, int(math.floor((y - radius - ymin) / self.resolution)))
        iy1 = min(self.height - 1, int(math.floor((y + radius - ymin) / self.resolution)))
        center = np.array([x, y], dtype=float)
        return {(ix, iy) for ix in range(ix0, ix1 + 1) for iy in range(iy0, iy1 + 1)
                if np.linalg.norm(self.cell_to_world((ix, iy)) - center) < radius}

    def cells_affected_by_circle(self, circle: Circle, padding_cells: int = 2) -> set[Cell]:
        x, y, radius = circle
        xmin, _, ymin, _ = self.bounds
        padding = padding_cells * self.resolution
        ix0 = max(0, int(math.floor((x - radius - padding - xmin) / self.resolution)))
        ix1 = min(self.width - 1, int(math.floor((x + radius + padding - xmin) / self.resolution)))
        iy0 = max(0, int(math.floor((y - radius - padding - ymin) / self.resolution)))
        iy1 = min(self.height - 1, int(math.floor((y + radius + padding - ymin) / self.resolution)))
        return {(ix, iy) for ix in range(ix0, ix1 + 1) for iy in range(iy0, iy1 + 1)}

    def edges_blocked_by_circle(self, circle: Circle) -> set[Edge]:
        blocked: set[Edge] = set()
        directions = [(1, 0), (0, 1)]
        if self.connectivity == 8:
            directions += [(1, 1), (1, -1)]
        for cell in self.cells_affected_by_circle(circle):
            for dx, dy in directions:
                neighbor = (cell[0] + dx, cell[1] + dy)
                if self.in_bounds(neighbor) and segment_intersects_circle(
                    self.cell_to_world(cell), self.cell_to_world(neighbor), circle
                ):
                    blocked.add(canonical_edge(cell, neighbor))
        return blocked

    def edge_is_collision_free(self, start: Cell, end: Cell) -> bool:
        return canonical_edge(start, end) not in self.blocked_edges

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
                if not self.is_free((cell[0] + dx, cell[1])) or not self.is_free((cell[0], cell[1] + dy)):
                    continue
            if not self.edge_is_collision_free(cell, neighbor):
                continue
            yield neighbor, self.resolution * (math.sqrt(2.0) if dx and dy else 1.0)


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


def cells_to_path(grid: GridMap, cells: list[Cell], start: np.ndarray,
                  goal: np.ndarray) -> list[np.ndarray]:
    if not cells:
        return []
    path = [np.asarray(start, dtype=float).copy()]
    path.extend(grid.cell_to_world(cell) for cell in cells)
    path.append(np.asarray(goal, dtype=float).copy())
    return [point for index, point in enumerate(path)
            if index == 0 or np.linalg.norm(point - path[index - 1]) > 1e-12]
