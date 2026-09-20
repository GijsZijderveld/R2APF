from __future__ import annotations

from typing import Any

from planners.rapf_global_planner import RAPFGlobalPlanner as R2APFPlanner
from planners.rapf_paper_planner import RAPFGlobalPlanner as RAPFPlanner


class RAPFPlannerAdapter:
    """Select RAPF or R2APF behind a neutral planner-facing adapter."""

    def __init__(self, variant: str, **planner_params: Any) -> None:
        self.name = variant.strip().lower()
        if self.name not in {"rapf", "r2apf"}:
            raise ValueError("variant must be 'rapf' or 'r2apf'.")
        planner_class = R2APFPlanner if self.name == "r2apf" else RAPFPlanner
        self._planner = planner_class(**planner_params)

    @property
    def wrapped_planner(self):
        return self._planner

    def plan_legacy(self, start, goal, known_obstacles, *, virtual_obstacles=None) -> dict:
        return self._planner.plan(
            start, goal, known_obstacles, virtual_obstacles=virtual_obstacles
        )
