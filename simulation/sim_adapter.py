from __future__ import annotations

import csv
import os
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from simulation.constants import (
    DT,
    GOAL_RADIUS,
    K_STABLE,
    MAX_STEPS,
    N_AGENTS,
    ROBOT_CLEARANCE,
    ROBOT_RADIUS,
    VEL_CLAMP,
)
from simulation.env.factory import build_world_from_geometry
from simulation.env.runtime import BaseObstacle, CircularObstacle, World
from simulation.env_gen_lunar import EnvConfig, Obstacle, generate_lunar_env


@dataclass
class EpisodeMetrics:
    reached: bool
    time_steps: int
    path_length: float
    collisions: int
    split_success: bool
    restick_incidence: int
    follow_rate: float
    stuck_ratio_mean: float
    free_centroid_alignment_mean: float
    oscillation_index: float


# ---------------------------
# Helpers (duck-typed agents)
# ---------------------------

def agent_get_pos(agent: Any) -> np.ndarray:
    """Return agent position; supports agent.position or agent.get_position()."""
    if hasattr(agent, "position"):
        return np.asarray(agent.position, dtype=float)
    if hasattr(agent, "get_position") and callable(agent.get_position):
        return np.asarray(agent.get_position(), dtype=float)
    raise AttributeError("Agent must expose `position` or `get_position()`")

def agent_get_goal(agent: Any) -> np.ndarray:
    if hasattr(agent, "goal"):
        return np.asarray(agent.goal, dtype=float)
    if hasattr(agent, "get_goal") and callable(agent.get_goal):
        return np.asarray(agent.get_goal(), dtype=float)
    raise AttributeError("Agent must expose `goal` or `get_goal()`")

def agent_step(agent: Any, env: World, agents: List[Any], t: int) -> None:
    """
    Call agent's step with a forgiving signature:
      preferred: step(env, agents, dt=DT, speed_limit=VEL_CLAMP, time_step=t*DT, bounds=bounds)
      fallback : step(dt) or step()
    """
    bounds = env.bounds() if hasattr(env, "bounds") else None
    # Most complete signature
    try:
        agent.step(env, agents, dt=DT, speed_limit=VEL_CLAMP, time_step=t * DT, bounds=bounds)
        return
    except TypeError:
        pass
    # Fallbacks
    try:
        agent.step(DT)
        return
    except TypeError:
        pass
    try:
        agent.step()
        return
    except TypeError:
        pass
    raise TypeError("Agent.step(...) signature not supported by sim_adapter; implement one of the accepted forms.")

def compute_free_centroid_alignment(
    agent: Any,
    neighbors_free_centroid: np.ndarray,
    field_vec: np.ndarray,
) -> float:
    a = np.array(field_vec, dtype=float)
    b = np.array(neighbors_free_centroid, dtype=float) - agent_get_pos(agent)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------
# Env builders (unchanged)
# ---------------------------

def _clustered_points(
    L: float,
    n: int,
    center_xy: Tuple[float, float],
    box_half: float,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    pts = []
    cx, cy = center_xy
    for _ in range(n):
        x = cx + rng.uniform(-box_half, box_half)
        y = cy + rng.uniform(-box_half, box_half)
        pts.append(np.array([x, y], dtype=float))
    return pts

def relax_starts(starts, env, *, min_sep, jitter=1e-3, max_iter=150, rng=None):
    rng = rng or np.random.default_rng()
    pts = [np.asarray(s, float).copy() for s in starts]
    L = getattr(env, "L", None)
    robot_r = getattr(env, "robot_radius", 0.25)
    obs_margin = robot_r + 0.05

    for i in range(len(pts)):
        pts[i] += jitter * rng.uniform(-1, 1, size=2)

    for _ in range(max_iter):
        moved = False
        # pairwise pushes if closer than min_sep
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                v = pts[i] - pts[j]
                d = float(np.linalg.norm(v)) + 1e-12
                if d < min_sep:
                    overlap = min_sep - d
                    shift = 0.5 * overlap * (v / d)
                    pts[i] += shift
                    pts[j] -= shift
                    moved = True

        # keep inside bounds
        if L is not None:
            for k in range(len(pts)):
                pts[k][0] = float(np.clip(pts[k][0], obs_margin, L - obs_margin))
                pts[k][1] = float(np.clip(pts[k][1], obs_margin, L - obs_margin))

        # nudge off obstacles
        for k in range(len(pts)):
            if any(o.distance_to_surface(pts[k]) < obs_margin for o in getattr(env, "obstacles", [])):
                pts[k] += 0.01 * rng.uniform(-1, 1, size=2)
                moved = True

        if not moved:
            break
    return [p.copy() for p in pts]

def _convert_geometry(obstacles: Iterable[Obstacle]) -> List[BaseObstacle]:
    converted: List[BaseObstacle] = []
    for obs in obstacles:
        center = (obs.x, obs.y)
        converted.append(CircularObstacle(center, obs.radius, kind=getattr(obs, "kind", "rock")))
    return converted

def make_runtime_env_from_geometry(
    obstacles: Iterable[Obstacle],
    L: float,
    n_agents: int = N_AGENTS,
    seed: int = 0,
) -> World:
    rng = np.random.default_rng(seed)
    starts = _clustered_points(L, n_agents, (2.0, 2.0), 1.5, rng)
    starts = relax_starts(starts, None, min_sep=0.8, rng=rng)
    goals  = _clustered_points(L, n_agents, (L - 2.0, L - 2.0), 0.5, rng)
    converted = _convert_geometry(obstacles)
    world = build_world_from_geometry(
        size=(L, L),
        obstacles=converted,
        starts=starts,
        goals=goals,
        dt=DT,
        goal_radius=GOAL_RADIUS,
    )
    world.start_positions = [p.copy() for p in starts]
    world.goal = np.array([L - 2.0, L - 2.0], dtype=float)
    return world

def build_env_single(cfg):
    """Build a simplified environment for single-agent RAPF planning."""
    geometry = generate_lunar_env(cfg)
    env = make_runtime_env_from_geometry(
        geometry, cfg.L, n_agents=1, seed=cfg.rng_seed
    )
    env.cfg = cfg
    return env

def build_env(cfg: EnvConfig) -> World:
    geometry = generate_lunar_env(cfg)
    env = make_runtime_env_from_geometry(geometry, cfg.L, n_agents=N_AGENTS, seed=cfg.rng_seed)
    env.cfg = cfg
    return env


# ----------------------------------------
# Agent construction (factory or class)
# ----------------------------------------

def build_agents_for_env(
    params: Dict[str, Any],
    env: World,
    *,
    agent_cls: Optional[type] = None,
    agent_factory: Optional[Callable[[World, int, np.ndarray, np.ndarray, Dict[str, Any]], Any]] = None,
) -> List[Any]:
    """
    Create agents without relying on any base class.
    Provide either:
      - agent_factory(env, j, start, goal, params) -> agent
      - agent_cls with constructor signature (agent_id, position, goal, **params)
    """
    agents: List[Any] = []
    for j in range(env.num_agents):
        start = env.starts[j].copy()
        goal = env.goals[j].copy()
        if agent_factory is not None:
            a = agent_factory(env, j, start, goal, params)
        elif agent_cls is not None:
            a = agent_cls(agent_id=j, position=start, goal=goal, **params)
        else:
            raise ValueError("Provide agent_cls or agent_factory to build agents.")
        # Preserve legacy attribute used elsewhere, if present on env
        if hasattr(env, "goal_radius"):
            try:
                setattr(a, "goal_radius", env.goal_radius)
            except Exception:
                pass
        agents.append(a)
    return agents


# ---------------------------
# Simulation (agent-agnostic)
# ---------------------------

def _distance_to_nearest_obstacle(point: np.ndarray, obstacles: Iterable[BaseObstacle]) -> float:
    best = np.inf
    for obs in obstacles:
        d = obs.distance_to_surface(point)
        if d < best:
            best = d
    return best

def _prepare_csv_writer(csv_logger: Optional[object]) -> tuple[Optional[csv.writer], Optional[object]]:
    if csv_logger is None:
        return None, None
    if isinstance(csv_logger, csv.writer):
        return csv_logger, None
    if hasattr(csv_logger, "writerow"):
        return csv_logger, None
    if isinstance(csv_logger, str):
        file_exists = os.path.exists(csv_logger)
        fh = open(csv_logger, "a", newline="", encoding="utf-8")
        writer = csv.writer(fh)
        if not file_exists or os.path.getsize(csv_logger) == 0:
            writer.writerow(
                [
                    "scenario",
                    "seed",
                    "agent_params_id",
                    "reached",
                    "time_steps",
                    "path_length",
                    "collisions",
                    "split_success",
                    "restick_incidence",
                    "follow_rate",
                    "stuck_ratio_mean",
                    "free_centroid_alignment_mean",
                    "oscillation_index",
                ]
            )
        return writer, fh
    return None, None

AgentStepDebug = Dict[str, Any]

def simulate_episode(
    agents: List[Any],
    env: World,
    *,
    seed: Optional[int] = None,
    max_steps: int = MAX_STEPS,
    csv_logger: Optional[object] = None,
    scenario_name: Optional[str] = None,
    agent_params_id: Optional[str] = None,
    on_step: Optional[Callable[[int, List[Any], World, List[AgentStepDebug]], None]] = None,
    **kwargs,
) -> EpisodeMetrics:
    """Run the APF simulation for a single episode (no base class required)."""

    np.random.default_rng(seed)
    np.random.seed(seed)
    random.seed(seed)

    num_agents = len(agents)
    path_len = 0.0
    collisions = 0
    stable_counter = 0
    split_success = False
    stuck_ratio_sum = 0.0
    alignment_values: List[float] = []
    oscillation_flips = [0 for _ in range(num_agents)]
    oscillation_prev_sign = [0 for _ in range(num_agents)]
    stuck_buffers = [0 for _ in range(num_agents)]
    restick_buffer = [0 for _ in range(num_agents)]
    freed_since_stuck = [False for _ in range(num_agents)]
    ever_stuck = [False for _ in range(num_agents)]
    restick_incidence = 0
    restick_threshold = 6

    bounds = env.bounds()
    reached = False

    last_goal_dists: List[float] = []

    for t in range(max_steps):
        step_path = 0.0

        # Integrate all agents
        for idx, agent in enumerate(agents):
            before = agent_get_pos(agent).copy()
            agent_step(agent, env, agents, t)
            after = agent_get_pos(agent)
            step_path += float(np.linalg.norm(after - before))

        path_len += step_path

        # Debug payload for visualizers (tolerant to absent attributes)
        debug_list: List[AgentStepDebug] = []
        for agent in agents:
            dbg: AgentStepDebug = {
                "is_stuck": bool(getattr(agent, "_is_stuck", False)),
                "velocity": getattr(agent, "velocity", None),
                "free_centroid": getattr(agent, "_last_free_centroid", None),
                "bias_vec": getattr(agent, "_last_bias_vec", None),
            }
            debug_list.append(dbg)

        if on_step is not None:
            on_step(t, agents, env, debug_list)

        # Collisions: agent–obstacle
        for i, agent in enumerate(agents):
            if _distance_to_nearest_obstacle(agent_get_pos(agent), env.obstacles) < ROBOT_CLEARANCE:
                collisions += 1

        # Collisions: agent–agent
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                if np.linalg.norm(agent_get_pos(agents[i]) - agent_get_pos(agents[j])) < 2 * ROBOT_RADIUS:
                    collisions += 1

        # Goal distance / stuck stats
        goal_dists = [float(np.linalg.norm(agent_get_goal(a) - agent_get_pos(a))) for a in agents]
        last_goal_dists = goal_dists
        stuck_flags = [bool(getattr(a, "_is_stuck", False)) for a in agents]
        stuck_ratio = sum(stuck_flags) / max(num_agents, 1)
        stuck_ratio_sum += stuck_ratio

        # split-success heuristic
        if env.obstacles and (min(goal_dists) < env.goal_radius * 2.0) and (max(goal_dists) - min(goal_dists) > 0.4):
            split_success = True

        # re-stick tracking
        for i, a in enumerate(agents):
            if stuck_flags[i]:
                stuck_buffers[i] += 1
                ever_stuck[i] = True
                if freed_since_stuck[i]:
                    restick_buffer[i] += 1
                    if restick_buffer[i] >= restick_threshold:
                        restick_incidence += 1
                        freed_since_stuck[i] = False
                        restick_buffer[i] = 0
                else:
                    restick_buffer[i] = 0
            else:
                if stuck_buffers[i] > 0:
                    freed_since_stuck[i] = True
                stuck_buffers[i] = 0
                restick_buffer[i] = 0

        # oscillation metric near obstacles + free-centroid alignment (if agent exposes hints)
        for i, a in enumerate(agents):
            nearby = env.nearest_obstacles(agent_get_pos(a), getattr(a, "obs_sense_radius", 3.0))
            vel = getattr(a, "velocity", np.zeros(2))
            if nearby:
                center = nearby[0][0]
                rel = agent_get_pos(a) - center
                tangential = float(rel[0] * vel[1] - rel[1] * vel[0])
                sign = 0
                if abs(tangential) > 1e-5:
                    sign = 1 if tangential > 0 else -1
                if sign != 0 and oscillation_prev_sign[i] != 0 and sign != oscillation_prev_sign[i]:
                    oscillation_flips[i] += 1
                if sign != 0:
                    oscillation_prev_sign[i] = sign
            if getattr(a, "_debug_nbr_free", 0) > 0:
                centroid = np.array(getattr(a, "_debug_bias_target", agent_get_pos(a)), dtype=float)
                field_vec = np.array(getattr(a, "_last_field", np.zeros(2)), dtype=float)
                alignment_values.append(compute_free_centroid_alignment(a, centroid, field_vec))

        # Termination on all-inside goal radius for K_STABLE consecutive steps
        if all(d <= env.goal_radius for d in goal_dists):
            stable_counter += 1
            if stable_counter >= K_STABLE:
                time_steps = t + 1
                reached = True
                break
        else:
            stable_counter = 0
    else:
        time_steps = max_steps
        reached = False

    # Edge-case: inside at last step counts as reached
    if not reached:
        final_dists = last_goal_dists or [float(np.linalg.norm(agent_get_goal(a) - agent_get_pos(a))) for a in agents]
        if final_dists and all(d <= env.goal_radius for d in final_dists):
            reached = True
            time_steps = max_steps

    total_steps = time_steps if reached else max_steps
    path_length_mean = path_len / max(num_agents, 1)
    stuck_ratio_mean = stuck_ratio_sum / max(total_steps, 1)
    free_centroid_alignment_mean = float(np.mean(alignment_values)) if alignment_values else 0.0
    oscillation_index = float(sum(oscillation_flips) / max(num_agents, 1))

    ever_stuck_count = sum(ever_stuck)
    cleared_count = sum(1 for i in range(num_agents) if ever_stuck[i] and not getattr(agents[i], "_is_stuck", False))
    follow_rate = (cleared_count / ever_stuck_count) if ever_stuck_count > 0 else 1.0

    metrics = EpisodeMetrics(
        reached=reached,
        time_steps=total_steps,
        path_length=path_length_mean,
        collisions=collisions,
        split_success=split_success,
        restick_incidence=restick_incidence,
        follow_rate=float(follow_rate),
        stuck_ratio_mean=float(stuck_ratio_mean),
        free_centroid_alignment_mean=float(free_centroid_alignment_mean),
        oscillation_index=oscillation_index,
    )

    writer, fh = _prepare_csv_writer(csv_logger)
    if writer is not None:
        writer.writerow(
            [
                scenario_name or "",
                seed,
                agent_params_id or "",
                int(metrics.reached),
                metrics.time_steps,
                round(metrics.path_length, 6),
                metrics.collisions,
                int(metrics.split_success),
                metrics.restick_incidence,
                round(metrics.follow_rate, 6),
                round(metrics.stuck_ratio_mean, 6),
                round(metrics.free_centroid_alignment_mean, 6),
                round(metrics.oscillation_index, 6),
            ]
        )
    if fh is not None:
        fh.close()

    return metrics
