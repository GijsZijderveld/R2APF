from __future__ import annotations

import time
from typing import Any, Dict, List

import numpy as np

from comparison.collision import path_is_collision_free
from comparison.execution import move_towards
from comparison.interfaces import BenchmarkObservation, Planner, PlanningResult
from comparison.sensing import merge_known_obstacles, reveal_obstacles


class NavigationAgent:
    """Shared sensing, replanning, and path execution for baseline planners."""

    def __init__(
        self,
        planner: Planner,
        agent_id: int = 0,
        sensing_range: float = 3.0,
        sensing_pad: float = 0.05,
        agent_radius: float = 0.30,
        goal_tolerance: float = 0.50,
        waypoint_tolerance: float = 0.10,
        lookahead_segments: int = 8,
    ) -> None:
        self.name = planner.name
        self.planner = planner
        self.agent_id = int(agent_id)
        self.sensing_range = float(sensing_range)
        self.sensing_pad = float(sensing_pad)
        self.agent_radius = float(agent_radius)
        self.goal_tolerance = float(goal_tolerance)
        self.waypoint_tolerance = float(waypoint_tolerance)
        self.lookahead_segments = int(lookahead_segments)

        self._position = np.zeros(2, dtype=float)
        self._goal = np.zeros(2, dtype=float)
        self._done = False
        self._known_obstacles: List[Any] = []
        self._path: List[np.ndarray] = []
        self._path_index = 0

        self._planning_calls = 0
        self._replans = 0
        self._plan_failures = 0
        self._blocked_steps = 0
        self._planning_times: List[float] = []
        self._planner_diagnostics: Dict[str, Any] = {}
        self._total_obstacles = 0

    def reset(self, environment: Any, seed: int) -> None:
        planner_reset = getattr(self.planner, "reset", None)
        if callable(planner_reset):
            planner_reset(seed)
        self._position = np.asarray(
            environment.starts[self.agent_id], dtype=float
        ).copy()
        if hasattr(environment, "goals"):
            self._goal = np.asarray(
                environment.goals[self.agent_id], dtype=float
            ).copy()
        else:
            self._goal = np.asarray(environment.goal, dtype=float).copy()

        self._done = False
        self._known_obstacles = []
        self._path = []
        self._path_index = 0
        self._planning_calls = 0
        self._replans = 0
        self._plan_failures = 0
        self._blocked_steps = 0
        self._planning_times = []
        self._planner_diagnostics = {}
        self._total_obstacles = len(getattr(environment, "obstacles", []))

    @property
    def position(self) -> np.ndarray:
        return self._position

    @property
    def goal(self) -> np.ndarray:
        return self._goal

    @property
    def reached_goal(self) -> bool:
        return self._done

    def _remaining_path_is_valid(self) -> bool:
        if not self._path or self._path_index >= len(self._path):
            return False

        path = [self._position.copy()]
        path.extend(self._path[self._path_index :])
        return path_is_collision_free(
            path,
            self._known_obstacles,
            self.agent_radius,
            start_index=0,
            lookahead_segments=self.lookahead_segments,
        )

    def _replan(self, bounds: Any) -> bool:
        started = time.perf_counter()
        result = self.planner.plan(
            start=self._position.copy(),
            goal=self._goal.copy(),
            known_obstacles=list(self._known_obstacles),
            bounds=bounds,
        )
        measured_time = time.perf_counter() - started

        if not isinstance(result, PlanningResult):
            raise TypeError(
                f"{self.planner.name}.plan() must return PlanningResult."
            )

        planning_time = (
            float(result.planning_time_s)
            if result.planning_time_s > 0.0
            else measured_time
        )
        self._planning_times.append(planning_time)
        self._planning_calls += 1
        if self._planning_calls > 1:
            self._replans += 1

        self._planner_diagnostics = dict(result.diagnostics)
        for key in ("collision_checks", "expanded_nodes", "samples"):
            value = getattr(result, key)
            if value is not None:
                self._planner_diagnostics[key] = value

        if not result.success or not result.path:
            self._path = []
            self._path_index = 0
            self._plan_failures += 1
            return False

        path = [np.asarray(point, dtype=float).copy() for point in result.path]
        if len(path) > 1 and np.linalg.norm(path[0] - self._position) < 1e-6:
            path = path[1:]

        self._path = path
        self._path_index = 0
        return bool(self._path)

    def step(
        self,
        observation: BenchmarkObservation,
        dt: float,
        speed_limit: float,
    ) -> None:
        if self._done:
            return

        environment = observation.environment
        newly_visible = reveal_obstacles(
            self._position,
            environment.obstacles,
            self.sensing_range,
            self.sensing_pad,
        )
        self._known_obstacles = merge_known_obstacles(
            self._known_obstacles,
            newly_visible,
        )

        if not self._remaining_path_is_valid():
            self._replan(observation.bounds)

        if self._path_index < len(self._path):
            target = self._path[self._path_index]
            new_position, moved, blocked = move_towards(
                self._position,
                target,
                speed_limit,
                dt,
                self._known_obstacles,
                self.agent_radius,
            )
            self._position = new_position

            if blocked:
                self._blocked_steps += 1
                self._path = []
                self._path_index = 0
            elif moved and np.linalg.norm(self._position - target) < self.waypoint_tolerance:
                self._path_index += 1

        if np.linalg.norm(self._position - self._goal) < self.goal_tolerance:
            self._done = True

    def diagnostics(self) -> Dict[str, Any]:
        total_time = float(sum(self._planning_times))
        maximum_time = (
            float(max(self._planning_times)) if self._planning_times else 0.0
        )
        return {
            "planning_calls": self._planning_calls,
            "replans": self._replans,
            "planning_time_total_s": total_time,
            "planning_time_max_s": maximum_time,
            "plan_failures": self._plan_failures,
            "blocked_steps": self._blocked_steps,
            "known_obstacles": len(self._known_obstacles),
            "total_obstacles": self._total_obstacles,
            **self._planner_diagnostics,
        }
