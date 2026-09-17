import unittest

import numpy as np

from agents.RAPF_Agent_v4 import RAPF_Agent_v4
from comparison.agents import NavigationAgent, RAPFAgentV4Adapter
from comparison.interfaces import BenchmarkObservation, PlanningResult


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

    def test_rapf_is_wrapped_without_inheritance(self):
        environment = EmptyEnvironment()
        adapter = RAPFAgentV4Adapter()
        adapter.reset(environment, seed=42)

        self.assertIsInstance(adapter.wrapped_agent, RAPF_Agent_v4)
        self.assertNotIsInstance(adapter, RAPF_Agent_v4)
        self.assertEqual(adapter.diagnostics()["total_obstacles"], 0)


if __name__ == "__main__":
    unittest.main()
