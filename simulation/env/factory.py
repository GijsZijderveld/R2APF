from __future__ import annotations

from typing import Iterable, List

import numpy as np

from simulation.constants import DT, GOAL_RADIUS

from .runtime import World, CircularObstacle, WallObstacle, BaseObstacle
from .scenarios_split import EnvConfig


def _make_obstacle(defn: dict) -> BaseObstacle:
    kind = defn.get("type", "rock")
    if kind in {"rock", "crater"}:
        center = defn.get("center")
        radius = float(defn.get("radius", 0.0))
        if center is None:
            raise ValueError("Disk obstacle requires 'center'.")
        return CircularObstacle(center, radius, kind=kind)
    if kind == "wall":
        p0 = defn.get("p0")
        p1 = defn.get("p1")
        thickness = float(defn.get("thickness", 0.5))
        if p0 is None or p1 is None:
            raise ValueError("Wall obstacle requires 'p0' and 'p1'.")
        return WallObstacle(p0, p1, thickness)
    raise ValueError(f"Unsupported obstacle type '{kind}'.")


def build_env_from_config(cfg: EnvConfig) -> World:
    """Instantiate a deterministic environment from an :class:`EnvConfig`."""

    obstacles: List[BaseObstacle] = [_make_obstacle(o) for o in cfg.obstacles]
    start_positions = [np.array(p, dtype=float) for p in cfg.start_positions]
    goal_point = np.array(cfg.goal, dtype=float)
    goals = [goal_point.copy() for _ in start_positions]

    return World(
        size=cfg.map_size,
        obstacles=obstacles,
        starts=start_positions,
        goals=goals,
        dt=DT,
        goal_radius=cfg.goal_radius if cfg.goal_radius is not None else GOAL_RADIUS,
    )


def build_world_from_geometry(
    size: Iterable[float],
    obstacles: Iterable[BaseObstacle],
    starts: Iterable[Iterable[float]],
    goals: Iterable[Iterable[float]],
    *,
    dt: float = DT,
    goal_radius: float = GOAL_RADIUS,
) -> World:
    """Helper used by stochastic generators to construct :class:`World`."""

    return World(size=size, obstacles=obstacles, starts=starts, goals=goals, dt=dt, goal_radius=goal_radius)
