import unittest
from dataclasses import dataclass

import numpy as np

from comparison.collision import path_is_collision_free
from comparison.registry import available_planners
from planners.baselines import (
    APFPlanner,
    AStarPlanner,
    DStarLitePlanner,
    RRTStarPlanner,
)


@dataclass(frozen=True)
class Circle:
    x: float
    y: float
    radius: float


BOUNDS = (0.0, 6.0, 0.0, 6.0)
START = np.array([0.5, 0.5])
GOAL = np.array([5.5, 5.5])


class BaselinePlannerTests(unittest.TestCase):
    def test_builtin_planners_are_registered(self):
        self.assertTrue(
            {"rapf", "r2apf", "apf", "astar", "dstar_lite", "rrt_star"}
            <= set(available_planners())
        )

    def test_apf_reaches_goal_in_empty_space(self):
        result = APFPlanner().plan(START, GOAL, [], BOUNDS)
        self.assertTrue(result.success, result.failure_reason)
        self.assertLess(np.linalg.norm(result.path[-1] - GOAL), 1e-12)

    def test_astar_routes_around_inflated_circle(self):
        obstacle = Circle(3.0, 3.0, 0.8)
        result = AStarPlanner().plan(START, GOAL, [obstacle], BOUNDS)
        self.assertTrue(result.success, result.failure_reason)
        self.assertGreater(result.expanded_nodes, 0)
        self.assertTrue(path_is_collision_free(result.path, [obstacle], 0.3))

    def test_dstar_lite_reuses_state_after_map_update(self):
        planner = DStarLitePlanner()
        first = planner.plan(START, GOAL, [], BOUNDS)
        second = planner.plan(
            np.array([1.0, 1.0]), GOAL, [Circle(3.0, 3.0, 0.8)], BOUNDS
        )
        self.assertTrue(first.success, first.failure_reason)
        self.assertTrue(second.success, second.failure_reason)
        self.assertTrue(second.diagnostics["state_reused"])
        self.assertGreater(second.diagnostics["changed_cells"], 0)
        self.assertGreater(second.diagnostics["changed_edges"], 0)
        self.assertTrue(
            path_is_collision_free(second.path, [Circle(3.0, 3.0, 0.8)], 0.3)
        )

        third = planner.plan(
            np.array([1.2, 1.2]), GOAL, [Circle(3.0, 3.0, 0.8)], BOUNDS
        )
        self.assertTrue(third.success, third.failure_reason)
        self.assertTrue(third.diagnostics["state_reused"])
        self.assertEqual(third.diagnostics["changed_cells"], 0)
        self.assertEqual(third.diagnostics["changed_edges"], 0)

    def test_grid_edges_reject_circle_intersections_between_free_centers(self):
        obstacle = Circle(3.0, 3.0, 0.31)
        start = np.array([1.5, 2.55])
        goal = np.array([4.5, 3.45])
        for planner in (AStarPlanner(), DStarLitePlanner()):
            result = planner.plan(start, goal, [obstacle], BOUNDS)
            self.assertTrue(result.success, result.failure_reason)
            self.assertTrue(path_is_collision_free(result.path, [obstacle], 0.3))

    def test_rrt_star_is_deterministic_for_fixed_seed(self):
        planner = RRTStarPlanner(max_samples=1500, seed_offset=123)
        planner.reset(42)
        first = planner.plan(START, GOAL, [Circle(3.0, 3.0, 0.8)], BOUNDS)
        second = planner.plan(START, GOAL, [Circle(3.0, 3.0, 0.8)], BOUNDS)
        self.assertTrue(first.success, first.failure_reason)
        self.assertTrue(second.success, second.failure_reason)
        self.assertEqual(first.samples, second.samples)
        np.testing.assert_allclose(np.asarray(first.path), np.asarray(second.path))


if __name__ == "__main__":
    unittest.main()
