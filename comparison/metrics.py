from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class EpisodeMetrics:
    planner: str
    scenario: str
    seed: int
    success: bool
    failure_reason: Optional[str]
    collision_count: int
    executed_path_length: float
    execution_steps: int
    final_goal_distance: float
    planning_calls: int
    failed_planner_calls: int
    total_planning_time_s: float
    maximum_planning_time_s: float
    replans: int
    sensed_obstacle_ratio: float
    planner_diagnostics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        row = asdict(self)
        diagnostics = row.pop("planner_diagnostics")
        for key, value in diagnostics.items():
            row[f"planner_{key}"] = value
        return row
