from __future__ import annotations

import heapq
import itertools
import math
import time
from typing import Any, Sequence

import numpy as np

from comparison.interfaces import PlanningResult
from .grid import Cell, GridMap, cells_to_path, normalize_bounds, octile_distance

INF = math.inf
INF_KEY = (INF, INF)


class DStarLitePlanner:
    """Incremental D* Lite for a fixed grid and goal.

    Search state is retained between plan calls. Newly occupied cells update the
    affected vertices instead of restarting the search.
    """

    name = "dstar_lite"

    def __init__(
        self,
        resolution: float = 0.30,
        agent_radius: float = 0.30,
        safety_margin: float = 0.0,
        connectivity: int = 8,
        forbid_corner_cutting: bool = True,
    ) -> None:
        self.resolution = float(resolution)
        self.agent_radius = float(agent_radius)
        self.safety_margin = float(safety_margin)
        self.connectivity = int(connectivity)
        self.forbid_corner_cutting = bool(forbid_corner_cutting)
        self._reset_state()

    def _reset_state(self) -> None:
        self.grid: GridMap | None = None
        self.goal: Cell | None = None
        self.last_start: Cell | None = None
        self.km = 0.0
        self.g: dict[Cell, float] = {}
        self.rhs: dict[Cell, float] = {}
        self.queue: list[tuple[float, float, int, Cell]] = []
        self.queued_keys: dict[Cell, tuple[float, float]] = {}
        self.counter = itertools.count()
        self.expanded = 0
        self.updated = 0

    def _value(self, table: dict[Cell, float], cell: Cell) -> float:
        return table.get(cell, INF)

    def _heuristic(self, a: Cell, b: Cell) -> float:
        assert self.grid is not None
        return octile_distance(a, b, self.grid.resolution)

    def _key(self, cell: Cell, start: Cell) -> tuple[float, float]:
        best = min(self._value(self.g, cell), self._value(self.rhs, cell))
        return best + self._heuristic(start, cell) + self.km, best

    def _push(self, cell: Cell, start: Cell) -> None:
        key = self._key(cell, start)
        self.queued_keys[cell] = key
        heapq.heappush(self.queue, (key[0], key[1], next(self.counter), cell))

    def _remove(self, cell: Cell) -> None:
        self.queued_keys.pop(cell, None)

    def _top(self) -> tuple[tuple[float, float], Cell | None]:
        while self.queue:
            k1, k2, _, cell = self.queue[0]
            if self.queued_keys.get(cell) == (k1, k2):
                return (k1, k2), cell
            heapq.heappop(self.queue)
        return INF_KEY, None

    def _pop(self) -> tuple[tuple[float, float], Cell]:
        while self.queue:
            k1, k2, _, cell = heapq.heappop(self.queue)
            if self.queued_keys.get(cell) == (k1, k2):
                self.queued_keys.pop(cell, None)
                return (k1, k2), cell
        raise IndexError("empty priority queue")

    def _successors(self, cell: Cell):
        assert self.grid is not None
        return self.grid.neighbors(cell)

    def _update_vertex(self, cell: Cell, start: Cell) -> None:
        assert self.goal is not None
        self.updated += 1
        if cell != self.goal:
            self.rhs[cell] = min(
                (
                    cost + self._value(self.g, successor)
                    for successor, cost in self._successors(cell)
                ),
                default=INF,
            )
        self._remove(cell)
        if self._value(self.g, cell) != self._value(self.rhs, cell):
            self._push(cell, start)

    def _compute_shortest_path(self, start: Cell) -> None:
        while True:
            top_key, _ = self._top()
            start_key = self._key(start, start)
            queue_can_affect_start = (
                top_key != INF_KEY
                and top_key[0] <= start_key[0] + 1e-12
            )
            if not (
                queue_can_affect_start
                or self._value(self.rhs, start) != self._value(self.g, start)
            ):
                return
            if top_key == INF_KEY:
                return
            old_key, cell = self._pop()
            new_key = self._key(cell, start)
            if old_key < new_key:
                self._push(cell, start)
            elif self._value(self.g, cell) > self._value(self.rhs, cell):
                self.g[cell] = self._value(self.rhs, cell)
                self.expanded += 1
                for predecessor, _ in self._successors(cell):
                    self._update_vertex(predecessor, start)
            else:
                self.g[cell] = INF
                self.expanded += 1
                self._update_vertex(cell, start)
                for predecessor, _ in self._successors(cell):
                    self._update_vertex(predecessor, start)

    def _initialize(self, grid: GridMap, start: Cell, goal: Cell) -> None:
        self._reset_state()
        self.grid = grid
        self.goal = goal
        self.last_start = start
        self.rhs[goal] = 0.0
        self._push(goal, start)

    def _update_grid(self, new_grid: GridMap, start: Cell) -> tuple[int, int]:
        assert self.grid is not None
        changed = self.grid.occupied.symmetric_difference(new_grid.occupied)
        changed_edges = self.grid.blocked_edges.symmetric_difference(
            new_grid.blocked_edges
        )
        self.grid = new_grid
        affected: set[Cell] = set(changed)
        for edge in changed_edges:
            affected.update(edge)
        for cell in changed:
            x, y = cell
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    candidate = (x + dx, y + dy)
                    if self.grid.in_bounds(candidate):
                        affected.add(candidate)
        for cell in affected:
            self._update_vertex(cell, start)
        return len(changed), len(changed_edges)

    def plan(self, start, goal, known_obstacles: Sequence[Any], bounds=None) -> PlanningResult:
        started = time.perf_counter()
        normalized_bounds = normalize_bounds(bounds)
        map_started = time.perf_counter()
        metadata_compatible = (
            self.grid is not None
            and self.grid.bounds == normalized_bounds
            and self.grid.resolution == self.resolution
            and self.grid.connectivity == self.connectivity
            and self.grid.forbid_corner_cutting == self.forbid_corner_cutting
        )
        if metadata_compatible:
            assert self.grid is not None
            grid = self.grid.with_added_obstacles(
                known_obstacles,
                agent_radius=self.agent_radius,
                safety_margin=self.safety_margin,
            )
        else:
            grid = GridMap.from_obstacles(
                known_obstacles,
                normalized_bounds,
                resolution=self.resolution,
                agent_radius=self.agent_radius,
                safety_margin=self.safety_margin,
                connectivity=self.connectivity,
                forbid_corner_cutting=self.forbid_corner_cutting,
            )
        map_update_time = time.perf_counter() - map_started
        start_cell, goal_cell = grid.world_to_cell(start), grid.world_to_cell(goal)
        if not grid.is_free(start_cell) or not grid.is_free(goal_cell):
            return PlanningResult(
                False, planning_time_s=time.perf_counter() - started,
                failure_reason="start_or_goal_occupied",
            )

        compatible = (
            metadata_compatible
            and self.goal == goal_cell
        )
        changed_cells = 0
        changed_edges = 0
        search_started = time.perf_counter()
        if not compatible:
            self._initialize(grid, start_cell, goal_cell)
        else:
            assert self.last_start is not None
            self.km += self._heuristic(self.last_start, start_cell)
            changed_cells, changed_edges = self._update_grid(grid, start_cell)
            self.last_start = start_cell

        expanded_before, updated_before = self.expanded, self.updated
        self._compute_shortest_path(start_cell)
        call_expanded = self.expanded - expanded_before
        call_updated = self.updated - updated_before
        search_time = time.perf_counter() - search_started

        if self._value(self.g, start_cell) == INF:
            return PlanningResult(
                False,
                planning_time_s=time.perf_counter() - started,
                failure_reason="no_path",
                expanded_nodes=call_expanded,
                diagnostics={
                    "updated_vertices": call_updated,
                    "changed_cells": changed_cells,
                    "changed_edges": changed_edges,
                    "map_update_time_s": map_update_time,
                    "search_time_s": search_time,
                },
            )

        cells = [start_cell]
        current = start_cell
        visited = {current}
        while current != goal_cell:
            choices = [
                (cost + self._value(self.g, neighbor), neighbor)
                for neighbor, cost in self._successors(current)
                if neighbor not in visited
                and self._value(self.g, neighbor) < INF
                and self._value(self.g, neighbor) < self._value(self.g, current)
            ]
            if not choices:
                return PlanningResult(
                    False,
                    planning_time_s=time.perf_counter() - started,
                    failure_reason="path_extraction_failed",
                    expanded_nodes=call_expanded,
                )
            _, current = min(
                choices,
                key=lambda item: (item[0], self._value(self.g, item[1]), item[1]),
            )
            visited.add(current)
            cells.append(current)

        return PlanningResult(
            True,
            path=cells_to_path(grid, cells, start, goal),
            planning_time_s=time.perf_counter() - started,
            expanded_nodes=call_expanded,
            diagnostics={
                "updated_vertices": call_updated,
                "changed_cells": changed_cells,
                "changed_edges": changed_edges,
                "map_update_time_s": map_update_time,
                "search_time_s": search_time,
                "state_reused": compatible,
            },
        )
