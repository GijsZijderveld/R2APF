from __future__ import annotations

from typing import Any, Callable, Dict

from comparison.agents import NavigationAgent
from comparison.interfaces import Planner
from comparison.rapf_adapter import RAPFPlannerAdapter

PlannerFactory = Callable[[Dict[str, Any]], Planner]

_PLANNER_FACTORIES: Dict[str, PlannerFactory] = {}


def register_planner(name: str, factory: PlannerFactory) -> None:
    key = name.strip().lower()
    if not key:
        raise ValueError("Planner name cannot be empty.")
    if key in {"rapf", "r2apf"}:
        raise ValueError(f"'{key}' is reserved for a RAPF v4 adapter.")
    _PLANNER_FACTORIES[key] = factory


def available_planners() -> tuple[str, ...]:
    return ("rapf", "r2apf", *sorted(_PLANNER_FACTORIES))


def build_agent(
    name: str,
    planner_config: Dict[str, Any] | None = None,
    agent_config: Dict[str, Any] | None = None,
):
    planner_config = dict(planner_config or {})
    agent_config = dict(agent_config or {})
    key = name.strip().lower()

    if key in {"rapf", "r2apf"}:
        return NavigationAgent(
            planner=RAPFPlannerAdapter(key, **planner_config),
            **agent_config,
        )

    try:
        planner_factory = _PLANNER_FACTORIES[key]
    except KeyError as exc:
        known = ", ".join(available_planners())
        raise KeyError(f"Unknown planner '{name}'. Available: {known}") from exc

    planner = planner_factory(planner_config)
    return NavigationAgent(planner=planner, **agent_config)


def _register_builtin_planners() -> None:
    from planners.baselines.apf import APFPlanner
    from planners.baselines.astar import AStarPlanner
    from planners.baselines.dstar_lite import DStarLitePlanner
    from planners.baselines.rrt_star import RRTStarPlanner

    register_planner("apf", lambda config: APFPlanner(**config))
    register_planner("astar", lambda config: AStarPlanner(**config))
    register_planner("dstar_lite", lambda config: DStarLitePlanner(**config))
    register_planner("rrt_star", lambda config: RRTStarPlanner(**config))


_register_builtin_planners()
