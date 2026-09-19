from __future__ import annotations

import heapq
import itertools
import time
from typing import Any, Sequence

import numpy as np

from comparison.interfaces import PlanningResult
from .grid import GridMap, cells_to_path, octile_distance


class AStarPlanner:
    """Deterministic 4- or 8-connected occupancy-grid A*."""

    name = "astar"

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

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        known_obstacles: Sequence[Any],
        bounds: Any = None,
    ) -> PlanningResult:
        started = time.perf_counter()
        map_started = time.perf_counter()
        grid = GridMap.from_obstacles(
            known_obstacles,
            bounds,
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

        counter = itertools.count()
        queue = [(octile_distance(start_cell, goal_cell, grid.resolution), next(counter), start_cell)]
        g_score = {start_cell: 0.0}
        came_from = {}
        closed = set()
        collision_checks = 0
        search_started = time.perf_counter()

        while queue:
            _, _, current = heapq.heappop(queue)
            if current in closed:
                continue
            closed.add(current)
            if current == goal_cell:
                cells = [current]
                while current in came_from:
                    current = came_from[current]
                    cells.append(current)
                cells.reverse()
                return PlanningResult(
                    True,
                    path=cells_to_path(grid, cells, start, goal),
                    planning_time_s=time.perf_counter() - started,
                    collision_checks=collision_checks,
                    expanded_nodes=len(closed),
                    diagnostics={
                        "map_update_time_s": map_update_time,
                        "search_time_s": time.perf_counter() - search_started,
                        "blocked_edges": len(grid.blocked_edges),
                    },
                )

            for neighbor, edge_cost in grid.neighbors(current):
                collision_checks += 1
                tentative = g_score[current] + edge_cost
                if tentative >= g_score.get(neighbor, float("inf")):
                    continue
                came_from[neighbor] = current
                g_score[neighbor] = tentative
                priority = tentative + octile_distance(neighbor, goal_cell, grid.resolution)
                heapq.heappush(queue, (priority, next(counter), neighbor))

        return PlanningResult(
            False,
            planning_time_s=time.perf_counter() - started,
            failure_reason="no_path",
            collision_checks=collision_checks,
            expanded_nodes=len(closed),
            diagnostics={
                "map_update_time_s": map_update_time,
                "search_time_s": time.perf_counter() - search_started,
                "blocked_edges": len(grid.blocked_edges),
            },
        )
