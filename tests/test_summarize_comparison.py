import unittest

import numpy as np
import pandas as pd

from scripts.summarize_comparison import summarize, summarize_failures


class SummarizeComparisonTests(unittest.TestCase):
    def setUp(self):
        self.results = pd.DataFrame(
            [
                {
                    "planner": "r2apf",
                    "scenario": "A",
                    "seed": 1,
                    "success": True,
                    "failure_reason": np.nan,
                    "collision_count": 0,
                    "collision_episode": False,
                    "executed_path_length": 10.0,
                    "execution_steps": 100,
                    "final_goal_distance": 0.1,
                    "planning_calls": 2,
                    "failed_planner_calls": 0,
                    "replans": 1,
                    "total_planning_time_s": 0.4,
                    "maximum_planning_time_s": 0.3,
                    "planner_unique_midpoint_rejections": 3,
                    "planner_path_reuse_ratio": np.nan,
                },
                {
                    "planner": "r2apf",
                    "scenario": "A",
                    "seed": 2,
                    "success": False,
                    "failure_reason": "max_steps",
                    "collision_count": 1,
                    "collision_episode": True,
                    "executed_path_length": 4.0,
                    "execution_steps": 1000,
                    "final_goal_distance": 6.0,
                    "planning_calls": 3,
                    "failed_planner_calls": 1,
                    "replans": 2,
                    "total_planning_time_s": 0.6,
                    "maximum_planning_time_s": 0.4,
                    "planner_unique_midpoint_rejections": 5,
                    "planner_path_reuse_ratio": 0.75,
                },
                {
                    "planner": "astar",
                    "scenario": "A",
                    "seed": 1,
                    "success": True,
                    "failure_reason": np.nan,
                    "collision_count": 0,
                    "collision_episode": False,
                    "executed_path_length": 9.0,
                    "execution_steps": 90,
                    "final_goal_distance": 0.2,
                    "planning_calls": 1,
                    "failed_planner_calls": 0,
                    "replans": 0,
                    "total_planning_time_s": 0.1,
                    "maximum_planning_time_s": 0.1,
                    "planner_unique_midpoint_rejections": np.nan,
                    "planner_path_reuse_ratio": np.nan,
                },
            ]
        )

    def test_overall_summary_uses_success_only_path_length_separately(self):
        summary = summarize(self.results, ["planner"]).set_index("planner")
        r2apf = summary.loc["r2apf"]
        self.assertEqual(r2apf["episodes"], 2)
        self.assertEqual(r2apf["success_rate"], 0.5)
        self.assertEqual(r2apf["executed_path_length_mean"], 7.0)
        self.assertEqual(r2apf["successful_executed_path_length_mean"], 10.0)
        self.assertEqual(r2apf["planning_time_per_call_s"], 0.2)

    def test_unsupported_optional_diagnostic_remains_blank(self):
        summary = summarize(self.results, ["planner"]).set_index("planner")
        self.assertTrue(
            pd.isna(summary.loc["astar", "planner_unique_midpoint_rejections_mean"])
        )
        self.assertTrue(
            pd.isna(summary.loc["astar", "planner_path_reuse_ratio_mean"])
        )

    def test_failure_reasons_are_counted(self):
        failures = summarize_failures(
            self.results, ["planner", "scenario"]
        )
        row = failures.iloc[0]
        self.assertEqual(row["planner"], "r2apf")
        self.assertEqual(row["failure_reason"], "max_steps")
        self.assertEqual(row["count"], 1)
        self.assertEqual(row["episode_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
