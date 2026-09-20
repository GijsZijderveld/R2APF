import unittest

import numpy as np

from comparison.agents import NavigationAgent
from comparison.interfaces import BenchmarkObservation, PlanningResult
from comparison.registry import available_planners, build_agent
from comparison.rapf_adapter import RAPFPlannerAdapter
from comparison.runner import BenchmarkConfig, run_episode
from planners.rapf_global_planner import RAPFGlobalPlanner as R2APFPlanner
from planners.rapf_paper_planner import RAPFGlobalPlanner as RAPFPaperPlanner
from simulation.env.runtime import CircularObstacle


class EmptyEnvironment:
    def __init__(self):
        self.starts = [np.array([0.0, 0.0])]
        self.goals = [np.array([1.0, 0.0])]
        self.goal = self.goals[0]
        self.obstacles = []

    def bounds(self):
        return (0.0, 2.0, 0.0, 2.0)


class StraightLinePlanner:
    name = "straight"

    def plan(self, start, goal, known_obstacles, bounds=None):
        del known_obstacles, bounds
        return PlanningResult(
            success=True,
            path=[start.copy(), goal.copy()],
            planning_time_s=123.0,
        )


class RepeatedFailurePlanner:
    name = "repeated_failure"

    def plan(self, start, goal, known_obstacles, bounds=None):
        del start, goal, known_obstacles, bounds
        return PlanningResult(success=False, failure_reason="local_minimum")


class FailureRecoveryAgent:
    """Alternates failed plans with movement-only recovery steps."""

    def reset(self, environment, seed):
        del seed
        self._position = environment.starts[0].copy()
        self._goal = environment.goal.copy()
        self._planning_calls = 0
        self._plan_failures = 0
        self._backtrack_tries = 0

    @property
    def position(self):
        return self._position

    @property
    def goal(self):
        return self._goal

    @property
    def reached_goal(self):
        return False

    def step(self, observation, dt, speed_limit):
        del dt, speed_limit
        if observation.time_step % 2 == 0:
            self._planning_calls += 1
            self._plan_failures += 1
            self._backtrack_tries += 1
        else:
            self._position = self._position + np.array([0.1, 0.0])

    def diagnostics(self):
        return {
            "planning_calls": self._planning_calls,
            "plan_failures": self._plan_failures,
            "backtrack_tries": self._backtrack_tries,
            "planning_time_total_s": 0.0,
            "planning_time_max_s": 0.0,
            "known_obstacles": 0,
            "total_obstacles": 0,
        }


class ComparisonContractTests(unittest.TestCase):
    def test_r2apf_records_midpoint_only_rejections(self):
        planner = R2APFPlanner()
        obstacle = CircularObstacle(
            center=np.array([0.29, 0.0]),
            radius=0.05,
        )
        prepared = planner._prepare_obstacles([obstacle])

        planner._compute_internal_step(
            np.array([0.0, 0.0]),
            np.array([2.0, 0.0]),
            prepared,
        )

        diagnostics = planner._diagnostics()
        self.assertGreaterEqual(diagnostics["unique_midpoint_rejections"], 1)
        self.assertGreaterEqual(
            diagnostics["selection_changes_due_to_midpoint_checking"],
            1,
        )
        self.assertIsNone(diagnostics["path_reuse_ratio"])

    def test_repeated_stationary_planning_failures_stop_episode(self):
        environment = EmptyEnvironment()
        metrics = run_episode(
            NavigationAgent(planner=RepeatedFailurePlanner()),
            environment,
            planner_name="repeated_failure",
            scenario="test",
            seed=42,
            config=BenchmarkConfig(
                max_steps=100,
                max_consecutive_plan_failures=0,
                max_consecutive_stationary_plan_failures=3,
            ),
        )
        self.assertFalse(metrics.success)
        self.assertEqual(
            metrics.failure_reason,
            "repeated_stationary_planning_failure",
        )
        self.assertEqual(metrics.execution_steps, 3)

    def test_recovery_movement_does_not_reset_consecutive_plan_failures(self):
        metrics = run_episode(
            FailureRecoveryAgent(),
            EmptyEnvironment(),
            planner_name="failure_recovery",
            scenario="test",
            seed=42,
            config=BenchmarkConfig(
                max_steps=100,
                max_consecutive_plan_failures=3,
                max_consecutive_stationary_plan_failures=0,
            ),
        )
        self.assertFalse(metrics.success)
        self.assertEqual(
            metrics.failure_reason,
            "consecutive_planning_failures",
        )
        self.assertEqual(metrics.planning_calls, 3)
        self.assertEqual(metrics.failed_planner_calls, 3)
        self.assertEqual(metrics.execution_steps, 5)

    def test_unsupported_planner_diagnostics_are_absent(self):
        metrics = run_episode(
            NavigationAgent(planner=StraightLinePlanner(), goal_tolerance=0.05),
            EmptyEnvironment(),
            planner_name="straight",
            scenario="test",
            seed=42,
            config=BenchmarkConfig(max_steps=20),
        )
        row = metrics.to_dict()
        self.assertNotIn("planner_unique_midpoint_rejections", row)
        self.assertNotIn("planner_repair_events", row)

    def test_generic_agent_accepts_interchangeable_planner(self):
        environment = EmptyEnvironment()
        agent = NavigationAgent(
            planner=StraightLinePlanner(),
            goal_tolerance=0.05,
        )
        agent.reset(environment, seed=42)

        for step in range(20):
            agent.step(
                BenchmarkObservation(
                    environment=environment,
                    time_step=step,
                    bounds=environment.bounds(),
                ),
                dt=0.1,
                speed_limit=1.0,
            )
            if agent.reached_goal:
                break

        self.assertTrue(agent.reached_goal)
        diagnostics = agent.diagnostics()
        self.assertGreaterEqual(diagnostics["planning_calls"], 1)
        self.assertLess(diagnostics["planning_time_total_s"], 1.0)

    def test_all_planners_use_the_same_navigation_agent(self):
        environment = EmptyEnvironment()
        for name in available_planners():
            agent = build_agent(name)
            agent.reset(environment, seed=42)
            self.assertIsInstance(agent, NavigationAgent)
            self.assertEqual(agent.diagnostics()["total_obstacles"], 0)

    def test_rapf_names_select_the_historical_planners(self):
        environment = EmptyEnvironment()
        self.assertIn("rapf", available_planners())
        self.assertIn("r2apf", available_planners())

        rapf = build_agent("rapf")
        rapf.reset(environment, seed=42)
        self.assertIsInstance(rapf._comparison_planner, RAPFPlannerAdapter)
        self.assertIsInstance(rapf._comparison_planner.wrapped_planner, RAPFPaperPlanner)

        r2apf = build_agent("r2apf")
        r2apf.reset(environment, seed=42)
        self.assertIsInstance(r2apf._comparison_planner, RAPFPlannerAdapter)
        self.assertIsInstance(r2apf._comparison_planner.wrapped_planner, R2APFPlanner)


if __name__ == "__main__":
    unittest.main()
