from __future__ import annotations

import time
from typing import Any, Sequence

import numpy as np

from comparison.collision import segment_is_collision_free
from comparison.interfaces import PlanningResult
from .grid import normalize_bounds


class RRTStarPlanner:
    """Seeded two-dimensional RRT* with circular-obstacle collision checks."""

    name = "rrt_star"

    def __init__(
        self,
        max_samples: int = 2000,
        step_size: float = 0.50,
        neighbor_radius: float = 1.50,
        goal_sample_rate: float = 0.05,
        goal_tolerance: float = 0.50,
        agent_radius: float = 0.30,
        safety_margin: float = 0.0,
        seed_offset: int = 100000,
    ) -> None:
        self.max_samples = int(max_samples)
        self.step_size = float(step_size)
        self.neighbor_radius = float(neighbor_radius)
        self.goal_sample_rate = float(goal_sample_rate)
        self.goal_tolerance = float(goal_tolerance)
        self.agent_radius = float(agent_radius)
        self.safety_margin = float(safety_margin)
        self.seed_offset = int(seed_offset)
        self._active_seed = self.seed_offset

    def reset(self, environment_seed: int) -> None:
        """Select a deterministic stream paired with the environment seed."""
        self._active_seed = int(environment_seed) + self.seed_offset

    def _free(self, a, b, obstacles) -> bool:
        return segment_is_collision_free(
            a, b, obstacles, self.agent_radius, self.safety_margin
        )

    def plan(self, start, goal, known_obstacles: Sequence[Any], bounds=None) -> PlanningResult:
        started = time.perf_counter()
        xmin, xmax, ymin, ymax = normalize_bounds(bounds)
        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)
        rng = np.random.default_rng(self._active_seed)
        nodes = [start.copy()]
        parents = [-1]
        costs = [0.0]
        collision_checks = 0
        goal_index = None

        for sample_count in range(1, self.max_samples + 1):
            sample = goal if rng.random() < self.goal_sample_rate else rng.uniform(
                [xmin, ymin], [xmax, ymax]
            )
            distances = np.linalg.norm(np.asarray(nodes) - sample, axis=1)
            nearest = int(np.argmin(distances))
            direction = sample - nodes[nearest]
            distance = float(np.linalg.norm(direction))
            if distance <= 1e-12:
                continue
            new = nodes[nearest] + min(self.step_size, distance) * direction / distance

            collision_checks += 1
            if not self._free(nodes[nearest], new, known_obstacles):
                continue

            node_array = np.asarray(nodes)
            near = np.flatnonzero(
                np.linalg.norm(node_array - new, axis=1) <= self.neighbor_radius
            ).tolist()
            best_parent = nearest
            best_cost = costs[nearest] + float(np.linalg.norm(new - nodes[nearest]))
            for index in near:
                collision_checks += 1
                candidate_cost = costs[index] + float(np.linalg.norm(new - nodes[index]))
                if candidate_cost < best_cost and self._free(
                    nodes[index], new, known_obstacles
                ):
                    best_parent, best_cost = index, candidate_cost

            new_index = len(nodes)
            nodes.append(new.copy())
            parents.append(best_parent)
            costs.append(best_cost)

            rewires = 0
            for index in near:
                proposed = best_cost + float(np.linalg.norm(nodes[index] - new))
                if proposed >= costs[index]:
                    continue
                collision_checks += 1
                if self._free(new, nodes[index], known_obstacles):
                    parents[index] = new_index
                    costs[index] = proposed
                    rewires += 1

            if np.linalg.norm(new - goal) <= self.goal_tolerance:
                collision_checks += 1
                if self._free(new, goal, known_obstacles):
                    goal_index = len(nodes)
                    nodes.append(goal.copy())
                    parents.append(new_index)
                    costs.append(best_cost + float(np.linalg.norm(goal - new)))
                    break
        else:
            sample_count = self.max_samples

        if goal_index is None:
            return PlanningResult(
                False,
                planning_time_s=time.perf_counter() - started,
                failure_reason="sample_budget_exhausted",
                collision_checks=collision_checks,
                samples=sample_count,
                diagnostics={"nodes": len(nodes)},
            )

        path = []
        index = goal_index
        while index >= 0:
            path.append(nodes[index].copy())
            index = parents[index]
        path.reverse()
        return PlanningResult(
            True,
            path=path,
            planning_time_s=time.perf_counter() - started,
            collision_checks=collision_checks,
            samples=sample_count,
            diagnostics={"nodes": len(nodes), "path_cost": costs[goal_index]},
        )
