from __future__ import annotations

import time
from typing import Any, Sequence

import numpy as np

from comparison.collision import segment_is_collision_free
from comparison.interfaces import PlanningResult
from comparison.sensing import obstacle_center


class APFPlanner:
    """Classical attractive/repulsive artificial potential field baseline."""

    name = "apf"

    def __init__(
        self,
        attractive_gain: float = 1.0,
        repulsive_gain: float = 1.0,
        influence_distance: float = 1.5,
        step_size: float = 0.30,
        goal_tolerance: float = 0.50,
        agent_radius: float = 0.30,
        safety_margin: float = 0.0,
        max_iterations: int = 5000,
        minimum_force: float = 1e-9,
    ) -> None:
        self.attractive_gain = float(attractive_gain)
        self.repulsive_gain = float(repulsive_gain)
        self.influence_distance = float(influence_distance)
        self.step_size = float(step_size)
        self.goal_tolerance = float(goal_tolerance)
        self.agent_radius = float(agent_radius)
        self.safety_margin = float(safety_margin)
        self.max_iterations = int(max_iterations)
        self.minimum_force = float(minimum_force)

    def plan(self, start, goal, known_obstacles: Sequence[Any], bounds=None) -> PlanningResult:
        started = time.perf_counter()
        position = np.asarray(start, dtype=float).copy()
        goal = np.asarray(goal, dtype=float)
        path = [position.copy()]
        collision_checks = 0

        for iteration in range(self.max_iterations):
            to_goal = goal - position
            goal_distance = float(np.linalg.norm(to_goal))
            if goal_distance < self.goal_tolerance:
                if goal_distance > 1e-12:
                    path.append(goal.copy())
                return PlanningResult(
                    True,
                    path=path,
                    planning_time_s=time.perf_counter() - started,
                    collision_checks=collision_checks,
                    diagnostics={"iterations": iteration},
                )

            force = self.attractive_gain * to_goal
            for obstacle in known_obstacles:
                center = obstacle_center(obstacle)
                offset = position - center
                center_distance = float(np.linalg.norm(offset))
                surface_clearance = (
                    center_distance
                    - float(obstacle.radius)
                    - self.agent_radius
                    - self.safety_margin
                )
                if surface_clearance < self.influence_distance:
                    if center_distance <= 1e-12 or surface_clearance <= 1e-9:
                        return PlanningResult(
                            False,
                            path=path,
                            planning_time_s=time.perf_counter() - started,
                            failure_reason="inside_inflated_obstacle",
                            collision_checks=collision_checks,
                            diagnostics={"iterations": iteration},
                        )
                    magnitude = self.repulsive_gain * (
                        1.0 / surface_clearance - 1.0 / self.influence_distance
                    ) / (surface_clearance * surface_clearance)
                    force += magnitude * offset / center_distance

            force_norm = float(np.linalg.norm(force))
            if force_norm <= self.minimum_force:
                return PlanningResult(
                    False,
                    path=path,
                    planning_time_s=time.perf_counter() - started,
                    failure_reason="local_minimum",
                    collision_checks=collision_checks,
                    diagnostics={"iterations": iteration},
                )

            step = min(self.step_size, goal_distance) * force / force_norm
            candidate = position + step
            if bounds is not None:
                values = bounds() if callable(bounds) else bounds
                xmin, xmax, ymin, ymax = map(float, values)
                candidate = np.clip(candidate, [xmin, ymin], [xmax, ymax])

            collision_checks += len(known_obstacles)
            if not segment_is_collision_free(
                position,
                candidate,
                known_obstacles,
                self.agent_radius,
                self.safety_margin,
            ):
                return PlanningResult(
                    False,
                    path=path,
                    planning_time_s=time.perf_counter() - started,
                    failure_reason="blocked_by_obstacle",
                    collision_checks=collision_checks,
                    diagnostics={"iterations": iteration},
                )

            if np.linalg.norm(candidate - position) <= 1e-12:
                return PlanningResult(
                    False,
                    path=path,
                    planning_time_s=time.perf_counter() - started,
                    failure_reason="stagnation",
                    collision_checks=collision_checks,
                    diagnostics={"iterations": iteration},
                )
            position = candidate
            path.append(position.copy())

        return PlanningResult(
            False,
            path=path,
            planning_time_s=time.perf_counter() - started,
            failure_reason="max_iterations",
            collision_checks=collision_checks,
            diagnostics={"iterations": self.max_iterations},
        )
