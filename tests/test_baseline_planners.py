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

    def test_astar_reuses_rasterized_obstacles_between_replans(self):
        obstacle = Circle(3.0, 3.0, 0.8)
        planner = AStarPlanner()
        first = planner.plan(START, GOAL, [obstacle], BOUNDS)
        first_grid = planner.grid
        second = planner.plan(np.array([0.8, 0.8]), GOAL, [obstacle], BOUNDS)
        self.assertTrue(first.success, first.failure_reason)
        self.assertTrue(second.success, second.failure_reason)
        self.assertIs(planner.grid, first_grid)

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

    def test_dstar_lite_repairs_tied_priority_keys_after_new_obstacles(self):
        # This update used to terminate with an unrepaired vertex on the route:
        # numerically equal primary keys were ordered differently by the heap.
        bounds = (0.0, 30.0, 0.0, 30.0)
        goal = np.array([28.0, 28.0])
        planner = DStarLitePlanner(resolution=0.3)
        first = planner.plan(
            np.array([1.193676061598316, 2.8142897459337686]), goal,
            [], bounds,
        )
        obstacles = [
            Circle(6.993457028442698, 5.948503213132516, 0.07062496655906744),
            Circle(6.862680737041771, 6.721820138182368, 0.0935604026750793),
            Circle(7.0922453939734975, 6.906594295826049, 0.037209736874709944),
            Circle(7.430784507768058, 7.794658487652297, 0.7351600706985476),
        ]
        start = np.array([4.177299576402272, 5.977299576402133])
        repaired = planner.plan(start, goal, obstacles, bounds)
        oracle = AStarPlanner(resolution=0.3).plan(start, goal, obstacles, bounds)
        self.assertTrue(first.success, first.failure_reason)
        self.assertTrue(repaired.success, repaired.failure_reason)
        self.assertTrue(oracle.success, oracle.failure_reason)
        self.assertTrue(path_is_collision_free(repaired.path, obstacles, 0.3))

        def grid_cost(path):
            cells = [planner.grid.world_to_cell(point) for point in path]
            return sum(
                np.linalg.norm(np.subtract(b, a)) * planner.grid.resolution
                for a, b in zip(cells, cells[1:])
            )

        self.assertAlmostEqual(grid_cost(repaired.path), grid_cost(oracle.path), places=6)

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
