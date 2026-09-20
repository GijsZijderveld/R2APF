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

        # At most one node can be accepted per sample, plus start and goal.
        capacity = self.max_samples + 2
        nodes = np.empty((capacity, 2), dtype=float)
        parents = np.empty(capacity, dtype=np.int32)
        costs = np.empty(capacity, dtype=float)
        nodes[0] = start
        parents[0] = -1
        costs[0] = 0.0
        node_count = 1

        collision_checks = 0
        goal_index = None
        neighbor_radius_sq = self.neighbor_radius * self.neighbor_radius
        goal_tolerance_sq = self.goal_tolerance * self.goal_tolerance
        step_size_sq = self.step_size * self.step_size

        for sample_count in range(1, self.max_samples + 1):
            sample = goal if rng.random() < self.goal_sample_rate else rng.uniform(
                [xmin, ymin], [xmax, ymax]
            )

            deltas = nodes[:node_count] - sample
            distance_sq = np.einsum("ij,ij->i", deltas, deltas)
            nearest = int(np.argmin(distance_sq))
            direction = sample - nodes[nearest]
            nearest_distance_sq = float(distance_sq[nearest])
            if nearest_distance_sq <= 1e-24:
                continue
            distance = float(np.sqrt(nearest_distance_sq))
            if nearest_distance_sq <= step_size_sq:
                new = sample.copy()
            else:
                new = nodes[nearest] + (self.step_size / distance) * direction

            collision_checks += 1
            if not self._free(nodes[nearest], new, known_obstacles):
                continue

            near_deltas = nodes[:node_count] - new
            near_distance_sq = np.einsum("ij,ij->i", near_deltas, near_deltas)
            near = np.flatnonzero(near_distance_sq <= neighbor_radius_sq)

            best_parent = nearest
            best_cost = costs[nearest] + float(np.sqrt(near_distance_sq[nearest]))
            for index_value in near:
                index = int(index_value)
                collision_checks += 1
                edge_length = float(np.sqrt(near_distance_sq[index]))
                candidate_cost = costs[index] + edge_length
                if candidate_cost < best_cost and self._free(
                    nodes[index], new, known_obstacles
                ):
                    best_parent, best_cost = index, candidate_cost

            new_index = node_count
            nodes[new_index] = new
            parents[new_index] = best_parent
            costs[new_index] = best_cost
            node_count += 1

            for index_value in near:
                index = int(index_value)
                proposed = best_cost + float(np.sqrt(near_distance_sq[index]))
                if proposed >= costs[index]:
                    continue
                collision_checks += 1
                if self._free(new, nodes[index], known_obstacles):
                    parents[index] = new_index
                    costs[index] = proposed

            goal_delta = new - goal
            if float(np.dot(goal_delta, goal_delta)) <= goal_tolerance_sq:
                collision_checks += 1
                if self._free(new, goal, known_obstacles):
                    goal_index = node_count
                    nodes[goal_index] = goal
                    parents[goal_index] = new_index
                    costs[goal_index] = best_cost + float(np.linalg.norm(goal_delta))
                    node_count += 1
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
                diagnostics={"nodes": node_count},
            )

        path = []
        index = goal_index
        while index >= 0:
            path.append(nodes[index].copy())
            index = int(parents[index])
        path.reverse()
        return PlanningResult(
            True,
            path=path,
            planning_time_s=time.perf_counter() - started,
            collision_checks=collision_checks,
            samples=sample_count,
            diagnostics={"nodes": node_count, "path_cost": float(costs[goal_index])},
        )
