#!/usr/bin/env python3
import argparse
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from simulation.constants import DT
from simulation.env_gen_lunar import generate_lunar_env, make_config
from simulation.sim_adapter import make_runtime_env_from_geometry
from agents.RAPF_Agent_v4 import RAPF_Agent_v4, RAPF_V4_PARAMS


def obs_center(obs) -> np.ndarray:
    if hasattr(obs, "x") and hasattr(obs, "y"):
        return np.array([obs.x, obs.y], dtype=float)
    return np.array(obs.center, dtype=float)


def draw_circle(ax, center, radius, **kwargs):
    patch = plt.Circle(tuple(center), float(radius), **kwargs)
    ax.add_patch(patch)
    return patch


def point_to_segment_nearest(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, float]:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        return a.copy(), 0.0
    t = float(np.dot(p - a, ab) / denom)
    t = max(0.0, min(1.0, t))
    return a + t * ab, t


def point_clear(p: np.ndarray, obstacles, safety_radius: float) -> bool:
    for obs in obstacles:
        c = obs_center(obs)
        r = float(obs.radius)
        if np.linalg.norm(p - c) < (r + safety_radius):
            return False
    return True


def segment_clear(a: np.ndarray, b: np.ndarray, obstacles, agent_radius: float) -> bool:
    for obs in obstacles:
        c = obs_center(obs)
        r = float(obs.radius)
        nearest, _ = point_to_segment_nearest(c, a, b)
        if np.linalg.norm(c - nearest) < (r + agent_radius):
            return False
    return True


def run_until_blocked(scenario: str, seed: int, max_ticks: int, speed: float):
    cfg = make_config(scenario, seed=seed)
    geom = generate_lunar_env(cfg)
    env = make_runtime_env_from_geometry(geom, cfg.L, n_agents=1, seed=seed)

    agent = RAPF_Agent_v4(
        agent_id=0,
        position=np.array(env.starts[0], dtype=float),
        goal=np.array(env.goal, dtype=float),
        **RAPF_V4_PARAMS,
    )

    executed_positions = [np.array(agent.q, dtype=float).copy()]
    blocked_tick = None
    blocked = None

    for t in range(max_ticks):
        agent.step(env, None, DT, speed, t)
        executed_positions.append(np.array(agent.q, dtype=float).copy())

        if getattr(agent, "last_blocked_move", None) is not None:
            blocked_tick = t
            blocked = agent.last_blocked_move
            break

        if getattr(agent, "done", False):
            break

    return env, agent, executed_positions, blocked_tick, blocked


def get_tail_path(planned_path_remaining, active_waypoint):
    if not planned_path_remaining:
        return np.empty((0, 2), dtype=float)

    tail = [np.array(p, dtype=float) for p in planned_path_remaining]

    if np.linalg.norm(tail[0] - active_waypoint) > 1e-9:
        tail = [np.array(active_waypoint, dtype=float)] + tail

    return np.array(tail, dtype=float)


def make_full_plot(env, executed_positions, blocked, out_prefix, show_all_obs=False):
    robot_pos = np.array(blocked["robot_pos"], dtype=float)
    attempted_step_end = np.array(blocked["attempted_step_end"], dtype=float)
    active_waypoint = np.array(blocked["active_waypoint"], dtype=float)
    planned_path_remaining = blocked.get("planned_path_remaining", [])
    obs_pos = np.array(blocked["obs_pos"], dtype=float)
    obs_r = float(blocked["obs_radius"])
    agent_r = float(blocked["agent_radius"])

    fig, ax = plt.subplots(figsize=(9, 9))

    for obs in env.obstacles:
        c = obs_center(obs)
        r = float(obs.radius)
        alpha = 0.23 if show_all_obs else 0.14
        draw_circle(ax, c, r, facecolor="gray", edgecolor="gray", alpha=alpha)

    ep = np.array(executed_positions, dtype=float)
    ax.plot(ep[:, 0], ep[:, 1], "-", linewidth=2.2, label="executed path")

    tail = get_tail_path(planned_path_remaining, active_waypoint)
    if len(tail) > 0:
        ax.plot(tail[:, 0], tail[:, 1], "o--", linewidth=1.8, markersize=4.5, alpha=0.9, label="planned path to goal")

    ax.plot(
        [robot_pos[0], active_waypoint[0]],
        [robot_pos[1], active_waypoint[1]],
        "-",
        linewidth=3.0,
        label="blocked local edge",
    )

    ax.plot(
        [robot_pos[0], attempted_step_end[0]],
        [robot_pos[1], attempted_step_end[1]],
        "-",
        linewidth=3.0,
        label="attempted executed step",
    )

    draw_circle(ax, obs_pos, obs_r, facecolor="red", edgecolor="red", alpha=0.28)
    draw_circle(ax, obs_pos, obs_r + agent_r, fill=False, edgecolor="red", linestyle="--", linewidth=2.0)

    ax.plot(robot_pos[0], robot_pos[1], "o", markersize=8, label="robot position")
    ax.plot(active_waypoint[0], active_waypoint[1], "x", markersize=10, label="active waypoint")
    ax.plot(attempted_step_end[0], attempted_step_end[1], "o", markersize=6, label="attempted step end")
    ax.plot(env.goal[0], env.goal[1], "*", markersize=13, label="goal")

    ax.set_aspect("equal")
    ax.set_title("Full environment: blocked local edge with remaining path to goal")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_full.png", dpi=300)
    plt.close(fig)


def make_zoom_plot(env, blocked, out_prefix, zoom_margin=1.0):
    active_waypoint = np.array(blocked["active_waypoint"], dtype=float)
    planned_path_remaining = blocked.get("planned_path_remaining", [])
    obs_pos = np.array(blocked["obs_pos"], dtype=float)
    obs_r = float(blocked["obs_radius"])
    agent_r = float(blocked["agent_radius"])

    # pick two consecutive waypoints from the stored planned path
    tail = [np.array(p, dtype=float) for p in planned_path_remaining]
    if len(tail) < 2:
        print("Need at least two planned waypoints to show a planner edge.")
        return

    wp0 = tail[0]
    wp1 = tail[1]

    midpoint = 0.5 * (wp0 + wp1)
    nearest, t_near = point_to_segment_nearest(obs_pos, wp0, wp1)

    fig, ax = plt.subplots(figsize=(8, 8))

    # nearby obstacles
    for obs in env.obstacles:
        c = obs_center(obs)
        r = float(obs.radius)
        if np.linalg.norm(c - obs_pos) <= 3.5:
            draw_circle(ax, c, r, facecolor="gray", edgecolor="gray", alpha=0.18)

    # obstacle and inflated boundary
    draw_circle(ax, obs_pos, obs_r, facecolor="red", edgecolor="red", alpha=0.28)
    draw_circle(ax, obs_pos, obs_r + agent_r, fill=False, edgecolor="red", linestyle="--", linewidth=2.0)

    # remaining path to goal
    tail_arr = np.array(tail, dtype=float)
    ax.plot(tail_arr[:, 0], tail_arr[:, 1], "o--", linewidth=1.7, markersize=5, alpha=0.7, label="planned path")

    # highlight offending edge
    ax.plot(
        [wp0[0], wp1[0]],
        [wp0[1], wp1[1]],
        "-",
        linewidth=3.0,
        label="offending edge",
    )
    # agent body at waypoints (radius = 0.2)
    draw_circle(ax, wp0, agent_r, fill=False, edgecolor="black", linewidth=1.5)
    draw_circle(ax, wp1, agent_r, fill=False, edgecolor="black", linewidth=1.5)

    # waypoint markers
    ax.plot(wp0[0], wp0[1], "o", markersize=8, label="waypoint A")
    ax.plot(wp1[0], wp1[1], "o", markersize=8, label="waypoint B")

    # midpoint / nearest point
    ax.plot(midpoint[0], midpoint[1], "s", markersize=7, label="midpoint")
    ax.plot(nearest[0], nearest[1], "*", markersize=12, label="nearest point on edge")

    xs = [wp0[0], wp1[0], obs_pos[0]]
    ys = [wp0[1], wp1[1], obs_pos[1]]
    inflate = obs_r + agent_r + zoom_margin
    ax.set_xlim(min(xs) - inflate, max(xs) + inflate)
    ax.set_ylim(min(ys) - inflate, max(ys) + inflate)

    ax.set_aspect("equal")
    ax.set_title("Planner edge invalid despite valid endpoints")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_zoom.png", dpi=300)
    plt.close(fig)

    # diagnostics
    wp0_ok = point_clear(wp0, env.obstacles, safety_radius=agent_r)
    wp1_ok = point_clear(wp1, env.obstacles, safety_radius=agent_r)
    midpoint_ok = point_clear(midpoint, env.obstacles, safety_radius=agent_r)
    edge_ok = segment_clear(wp0, wp1, env.obstacles, agent_radius=agent_r)

    print("\\n--- PLANNER EDGE DIAGNOSTICS ---")
    print(f"waypoint A valid : {wp0_ok}")
    print(f"waypoint B valid : {wp1_ok}")
    print(f"midpoint valid   : {midpoint_ok}")
    print(f"edge valid       : {edge_ok}")
    print(f"nearest t        : {t_near:.4f}")
    print(f"nearest point    : {np.round(nearest, 4)}")
    print("-------------------------------\\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, type=str)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--max_ticks", default=1500, type=int)
    parser.add_argument("--speed", default=1.0, type=float)
    parser.add_argument("--out_prefix", default="figures/blocked_case", type=str)
    parser.add_argument("--show_all_obs", action="store_true")
    args = parser.parse_args()

    out_dir = os.path.dirname(args.out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    env, agent, executed_positions, blocked_tick, blocked = run_until_blocked(
        scenario=args.scenario,
        seed=args.seed,
        max_ticks=args.max_ticks,
        speed=args.speed,
    )

    if blocked is None:
        print("No blocked step found for this seed.")
        print("Did you add the last_blocked_move snapshot patch to RAPF_Agent_v4.py?")
        return

    print(f"Blocked step found at tick : {blocked_tick}")
    print(f"Blocked steps counter      : {getattr(agent, 'blocked_steps', 0)}")
    print(f"Final debug_status         : {getattr(agent, 'debug_status', '')}")

    make_full_plot(env, executed_positions, blocked, args.out_prefix, show_all_obs=args.show_all_obs)
    make_zoom_plot(env, blocked, args.out_prefix)

    print(f"Saved: {args.out_prefix}_full.png")
    print(f"Saved: {args.out_prefix}_zoom.png")


if __name__ == "__main__":
    main()