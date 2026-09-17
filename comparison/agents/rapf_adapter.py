from __future__ import annotations

from typing import Any, Dict

import numpy as np

from agents.RAPF_Agent_v4 import RAPF_Agent_v4

from comparison.interfaces import BenchmarkObservation


class RAPFAgentV4Adapter:
    """Expose the unchanged RAPF v4 agent through the benchmark interface."""

    name = "r2apf"

    def __init__(
        self,
        agent_id: int = 0,
        speed_limit: float = 1.0,
        **rapf_params: Any,
    ) -> None:
        self.agent_id = int(agent_id)
        self.default_speed_limit = float(speed_limit)
        self.rapf_params = dict(rapf_params)
        self._agent: RAPF_Agent_v4 | None = None
        self._total_obstacles = 0

    @property
    def wrapped_agent(self) -> RAPF_Agent_v4:
        if self._agent is None:
            raise RuntimeError("Call reset(environment, seed) before using the adapter.")
        return self._agent

    def reset(self, environment: Any, seed: int) -> None:
        del seed  # The environment is already generated from the campaign seed.
        start = np.asarray(environment.starts[self.agent_id], dtype=float)
        if hasattr(environment, "goals"):
            goal = np.asarray(environment.goals[self.agent_id], dtype=float)
        else:
            goal = np.asarray(environment.goal, dtype=float)

        self._total_obstacles = len(getattr(environment, "obstacles", []))
        self._agent = RAPF_Agent_v4(
            agent_id=self.agent_id,
            position=start,
            goal=goal,
            **self.rapf_params,
        )

    @property
    def position(self) -> np.ndarray:
        return np.asarray(self.wrapped_agent.position, dtype=float)

    @property
    def goal(self) -> np.ndarray:
        return np.asarray(self.wrapped_agent.goal, dtype=float)

    @property
    def reached_goal(self) -> bool:
        return bool(self.wrapped_agent.done)

    def step(
        self,
        observation: BenchmarkObservation,
        dt: float,
        speed_limit: float,
    ) -> None:
        environment = observation.environment
        bounds = (
            environment.bounds()
            if hasattr(environment, "bounds")
            else observation.bounds
        )
        self.wrapped_agent.step(
            environment,
            None,
            dt=float(dt),
            speed_limit=float(speed_limit),
            time_step=observation.time_step,
            bounds=bounds,
        )

    def diagnostics(self) -> Dict[str, Any]:
        agent = self.wrapped_agent
        sensed = len(getattr(agent, "perceived_obstacles", []))
        total = self._total_obstacles
        return {
            "planning_calls": int(getattr(agent, "replan_count", 0)),
            "replans": int(getattr(agent, "replan_count", 0)),
            "planning_time_total_s": 0.0,
            "planning_time_max_s": 0.0,
            "planning_effort": float(getattr(agent, "planning_effort", 0.0)),
            "plan_failures": int(getattr(agent, "plan_fail_count", 0)),
            "blocked_steps": int(getattr(agent, "blocked_steps", 0)),
            "known_obstacles": sensed,
            "total_obstacles": total,
            "virtual_obstacles_created": int(
                getattr(agent, "virtual_total_created", 0)
            ),
            "backtrack_tries": int(
                getattr(agent, "_agent_backtrack_tries", 0)
            ),
            "backtrack_failed": bool(
                getattr(agent, "backtrack_failed", False)
            ),
        }
