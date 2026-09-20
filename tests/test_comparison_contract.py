import unittest

import numpy as np

from comparison.agents import NavigationAgent
from comparison.interfaces import BenchmarkObservation, PlanningResult
from comparison.registry import available_planners, build_agent
from comparison.rapf_adapter import RAPFPlannerAdapter
from planners.rapf_global_planner import RAPFGlobalPlanner as R2APFPlanner
from planners.rapf_paper_planner import RAPFGlobalPlanner as RAPFPaperPlanner


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
        )


class ComparisonContractTests(unittest.TestCase):
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
        self.assertGreaterEqual(agent.diagnostics()["planning_calls"], 1)

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
