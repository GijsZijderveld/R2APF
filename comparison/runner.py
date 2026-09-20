from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional

import numpy as np

from comparison.collision import position_is_in_collision
from comparison.interfaces import BenchmarkAgent, BenchmarkObservation
from comparison.metrics import EpisodeMetrics


@dataclass(frozen=True)
class BenchmarkConfig:
    dt: float = 0.1
    speed_limit: float = 1.0
    max_steps: int = 1000
    agent_radius: float = 0.30
    stop_on_collision: bool = False
    max_consecutive_stationary_plan_failures: int = 3
    max_total_planning_time_s: Optional[float] = None


def run_episode(
    agent: BenchmarkAgent,
    environment: Any,
    *,
    planner_name: str,
    scenario: str,
    seed: int,
    config: BenchmarkConfig,
) -> EpisodeMetrics:
    random.seed(seed)
    np.random.seed(seed)
    agent.reset(environment, seed)

    path_length = 0.0
    collision_count = 0
    failure_reason = None
    execution_steps = 0
    stationary_plan_failures = 0
    previous_diagnostics = dict(agent.diagnostics())

    for step_number in range(config.max_steps):
        before = np.asarray(agent.position, dtype=float).copy()
        bounds = (
            environment.bounds()
            if hasattr(environment, "bounds")
            else None
        )
        agent.step(
            BenchmarkObservation(
                environment=environment,
                time_step=step_number,
                bounds=bounds,
            ),
            dt=config.dt,
            speed_limit=config.speed_limit,
        )
        after = np.asarray(agent.position, dtype=float)
        path_length += float(np.linalg.norm(after - before))
        execution_steps = step_number + 1

        current_diagnostics = dict(agent.diagnostics())
        planning_failed = int(current_diagnostics.get("plan_failures", 0)) > int(
            previous_diagnostics.get("plan_failures", 0)
        )
        stationary = float(np.linalg.norm(after - before)) <= 1e-12
        recovery_state_unchanged = all(
            current_diagnostics.get(key, 0) == previous_diagnostics.get(key, 0)
            for key in (
                "known_obstacles",
                "virtual_obstacles_created",
                "max_active_virtual_obstacles",
                "backtrack_tries",
            )
        )
        if planning_failed and stationary and recovery_state_unchanged:
            stationary_plan_failures += 1
        else:
            stationary_plan_failures = 0

        if (
            config.max_consecutive_stationary_plan_failures > 0
            and stationary_plan_failures
            >= config.max_consecutive_stationary_plan_failures
        ):
            failure_reason = "repeated_stationary_planning_failure"
            break

        if (
            config.max_total_planning_time_s is not None
            and float(current_diagnostics.get("planning_time_total_s", 0.0))
            >= config.max_total_planning_time_s
        ):
            failure_reason = "planning_time_budget"
            break

        previous_diagnostics = current_diagnostics

        collided = position_is_in_collision(
            after,
            environment.obstacles,
            config.agent_radius,
        )
        if collided:
            collision_count += 1
            if config.stop_on_collision:
                failure_reason = "collision"
                break

        if agent.reached_goal:
            break
    else:
        failure_reason = "max_steps"

    success = bool(agent.reached_goal)
    if not success and failure_reason is None:
        failure_reason = "not_reached"

    diagnostics = dict(agent.diagnostics())
    known = int(diagnostics.pop("known_obstacles", 0))
    total = int(
        diagnostics.pop(
            "total_obstacles",
            len(getattr(environment, "obstacles", [])),
        )
    )
    sensing_ratio = known / total if total > 0 else 1.0

    planning_calls = int(diagnostics.pop("planning_calls", 0))
    replans = int(diagnostics.pop("replans", max(0, planning_calls - 1)))
    total_planning_time = float(
        diagnostics.pop("planning_time_total_s", 0.0)
    )
    maximum_planning_time = float(
        diagnostics.pop("planning_time_max_s", 0.0)
    )

    return EpisodeMetrics(
        planner=planner_name,
        scenario=scenario,
        seed=seed,
        success=success,
        failure_reason=failure_reason,
        collision_count=collision_count,
        executed_path_length=path_length,
        execution_steps=execution_steps,
        final_goal_distance=float(
            np.linalg.norm(
                np.asarray(agent.goal, dtype=float)
                - np.asarray(agent.position, dtype=float)
            )
        ),
        planning_calls=planning_calls,
        total_planning_time_s=total_planning_time,
        maximum_planning_time_s=maximum_planning_time,
        replans=replans,
        sensed_obstacle_ratio=sensing_ratio,
        planner_diagnostics=diagnostics,
    )


def run_campaign(
    agent_factory: Callable[[], BenchmarkAgent],
    environment_factory: Callable[[str, int], Any],
    *,
    planner_name: str,
    scenarios: Iterable[str],
    seeds: Iterable[int],
    config: BenchmarkConfig,
) -> List[EpisodeMetrics]:
    results: List[EpisodeMetrics] = []
    for scenario in scenarios:
        for seed in seeds:
            results.append(
                run_episode(
                    agent_factory(),
                    environment_factory(scenario, seed),
                    planner_name=planner_name,
                    scenario=scenario,
                    seed=seed,
                    config=config,
                )
            )
    return results


def write_results(
    results: Iterable[EpisodeMetrics],
    output_path: str | Path,
) -> None:
    rows = [result.to_dict() for result in results]
    if not rows:
        raise ValueError("No results to write.")

    fieldnames = sorted({key for row in rows for key in row})
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
