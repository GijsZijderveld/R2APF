from __future__ import annotations

import time
from typing import Any, Dict

import numpy as np

from agents.navigation_core import NavigationCore
from comparison.interfaces import BenchmarkObservation, PlanningResult


class _CommonPlannerBridge:
    """Translate the common planner contract to the verified navigation loop."""

    def __init__(self, planner: Any) -> None:
        self.planner = planner
        self.bounds = None
        self.total_plan_calls = 0
        self.total_plan_failures = 0
        self.total_planning_time_s = 0.0
        self.maximum_planning_time_s = 0.0
        self.last_diagnostics: Dict[str, Any] = {}

    def reset(self, seed: int) -> None:
        reset = getattr(self.planner, "reset", None)
        if callable(reset):
            reset(seed)

    def plan(self, start_pos, goal_pos, real_obstacles, *, virtual_obstacles=None) -> dict:
        started = time.perf_counter()
        legacy_plan = getattr(self.planner, "plan_legacy", None)
        if callable(legacy_plan):
            result = legacy_plan(
                start_pos, goal_pos, real_obstacles,
                virtual_obstacles=virtual_obstacles,
            )
        else:
            common = self.planner.plan(
                start=np.asarray(start_pos, dtype=float),
                goal=np.asarray(goal_pos, dtype=float),
                known_obstacles=list(real_obstacles),
                bounds=self.bounds,
            )
            if not isinstance(common, PlanningResult):
                raise TypeError(f"{self.planner.name}.plan() must return PlanningResult.")
            result = {
                "success": common.success,
                "path": common.path,
                "break_reason": common.failure_reason,
                "virtual_obstacles": virtual_obstacles or [],
                "planning_effort": common.diagnostics.get("iterations", 0),
            }
            self.last_diagnostics = dict(common.diagnostics)
            for key in ("collision_checks", "expanded_nodes", "samples"):
                value = getattr(common, key)
                if value is not None:
                    self.last_diagnostics[key] = value

        elapsed = time.perf_counter() - started
        # Published timing uses this one shared wall-clock boundary for every planner.
        measured = elapsed
        self.total_planning_time_s += measured
        self.maximum_planning_time_s = max(self.maximum_planning_time_s, measured)
        self.total_plan_calls += 1
        if not result.get("success", False):
            self.total_plan_failures += 1
        self.last_diagnostics.update(result.get("diagnostics", {}))
        return result


class NavigationAgent(NavigationCore):
    """One historical-compatible sensing and execution loop for every planner.

    The inherited code is retained temporarily as the regression-verified
    behavioral specification. It is not a distinct agent in the comparison:
    every planner is executed through this public, unversioned class.
    """

    def __init__(
        self,
        planner: Any,
        agent_id: int = 0,
        sensing_range: float = 3.0,
        sensing_pad: float = 0.05,
        agent_radius: float = 0.30,
        goal_tolerance: float = 0.50,
        waypoint_tolerance: float = 0.10,
        lookahead_segments: int = 8,
        **navigation_params: Any,
    ) -> None:
        self._comparison_planner = planner
        self._comparison_agent_id = int(agent_id)
        self._navigation_params = {
            "SENSE_RANGE": float(sensing_range),
            "SENSE_PAD": float(sensing_pad),
            "RHO_L": float(agent_radius),
            "GOAL_TOLERANCE": float(goal_tolerance),
            "WAYPOINT_TOLERANCE": float(waypoint_tolerance),
            "LOOKAHEAD_SEGMENTS": int(lookahead_segments),
            **navigation_params,
        }
        self._bridge: _CommonPlannerBridge | None = None
        self._total_obstacles = 0

    def reset(self, environment: Any, seed: int) -> None:
        start = np.asarray(environment.starts[self._comparison_agent_id], dtype=float)
        # The single-agent benchmark historically used the canonical global
        # goal (env.goal), not the nearby randomized visualization goal.
        goal_source = (
            environment.goal
            if hasattr(environment, "goal")
            else environment.goals[self._comparison_agent_id]
        )
        goal = np.asarray(goal_source, dtype=float)
        NavigationCore.__init__(
            self, self._comparison_agent_id, start, goal, **self._navigation_params
        )
        self._bridge = _CommonPlannerBridge(self._comparison_planner)
        self._bridge.reset(seed)
        self.planner = self._bridge
        self._total_obstacles = len(getattr(environment, "obstacles", []))

    @property
    def reached_goal(self) -> bool:
        return bool(self.done)

    def step(self, observation: BenchmarkObservation, dt: float, speed_limit: float) -> None:
        if self._bridge is None:
            raise RuntimeError("Call reset(environment, seed) before step().")
        self._bridge.bounds = observation.bounds
        NavigationCore.step(
            self, observation.environment, None, float(dt), float(speed_limit),
            observation.time_step, observation.bounds,
        )

    def diagnostics(self) -> Dict[str, Any]:
        if self._bridge is None:
            raise RuntimeError("Call reset(environment, seed) before diagnostics().")
        return {
            "planning_calls": self._bridge.total_plan_calls,
            "replans": int(getattr(self, "replan_count", 0)),
            "planning_time_total_s": self._bridge.total_planning_time_s,
            "planning_time_max_s": self._bridge.maximum_planning_time_s,
            "planning_effort": float(getattr(self, "planning_effort", 0.0)),
            "plan_failures": self._bridge.total_plan_failures,
            "blocked_steps": int(getattr(self, "blocked_steps", 0)),
            "known_obstacles": len(getattr(self, "perceived_obstacles", [])),
            "total_obstacles": self._total_obstacles,
            "virtual_obstacles_created": int(getattr(self, "virtual_total_created", 0)),
            "max_active_virtual_obstacles": int(getattr(self, "virtual_max_active", 0)),
            "backtrack_tries": int(getattr(self, "_agent_backtrack_tries", 0)),
            "backtrack_failed": bool(getattr(self, "backtrack_failed", False)),
            **self._bridge.last_diagnostics,
        }
