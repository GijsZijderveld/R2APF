#!/usr/bin/env python
"""Replay either RAPF planner with the shared navigation behavior."""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

from simulation.env_gen_lunar import make_config, generate_lunar_env
from simulation.sim_adapter import make_runtime_env_from_geometry
from simulation.constants import DT
from agents.navigation_core import NavigationCore


def _normalize_virtual_obstacles(vobs):
    out = []
    for v in vobs or []:
        if isinstance(v, dict):
            pos = np.array(v.get("position", v.get("pos")), dtype=float)
            rad = float(v["radius"])
        else:
            pos = np.array([v.x, v.y], dtype=float)
            rad = float(getattr(v, "radius"))
        out.append({"position": pos, "radius": rad})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--speed", type=float, default=1)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--scenario", type=str, default="B")
    args = parser.parse_args()

    # Environment
    cfg = make_config(args.scenario, seed=args.seed)
    geom = generate_lunar_env(cfg)
    env = make_runtime_env_from_geometry(geom, cfg.L, n_agents=1, seed=args.seed)

    # Agent
    agent = NavigationCore(agent_id=0, position=env.starts[0], goal=env.goal)

    history = {
        "pos": [],
        "path": [],
        "remaining_path": [],
        "path_index": [],
        "replan_count": [],
        "plan_fail_count": [],
        "invalid_from_idx": [],
        "virtual_obstacles": [],
        "executed_opt": [],
        "all_len": [],
    }

    all_points = [np.array(agent.q, dtype=float).copy()]
    last_all_len = len(getattr(agent, "executed_path_all", []))

    print("--- Simulating RAPF v4 (Anchor-Based) ---")

    for t in range(args.steps):
        agent.step(env, None, DT, args.speed, t)

        # Position
        history["pos"].append(agent.q.copy())

        # Planned path
        full_path = [np.array(p, dtype=float).copy() for p in getattr(agent, "planned_path", [])]
        path_index = int(getattr(agent, "path_index", 0))

        history["path"].append(full_path)
        history["path_index"].append(path_index)

        if full_path and path_index < len(full_path):
            history["remaining_path"].append(full_path[path_index:])
        else:
            history["remaining_path"].append([])

        # Counters
        history["replan_count"].append(int(getattr(agent, "replan_count", 0)))
        history["plan_fail_count"].append(int(getattr(agent, "plan_fail_count", 0)))
        history["invalid_from_idx"].append(getattr(agent, "last_failure_invalid_from_index", None))

        # Virtual obstacles
        history["virtual_obstacles"].append(
            _normalize_virtual_obstacles(getattr(agent, "last_virtual_obstacles", []) or [])
        )

        # Executed paths
        curr_opt = getattr(agent, "executed_path_opt", [])
        history["executed_opt"].append([np.array(p, dtype=float).copy() for p in curr_opt])

        curr_all = getattr(agent, "executed_path_all", [])
        if len(curr_all) > last_all_len:
            for k in range(last_all_len, len(curr_all)):
                all_points.append(np.array(curr_all[k], dtype=float).copy())
            last_all_len = len(curr_all)
        history["all_len"].append(len(all_points))

        if agent.done:
            print(f"Goal Reached at t={t}")
            break

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.6)

    ax.plot(env.goal[0], env.goal[1], "r*", markersize=15, label="Goal")

    # Obstacles
    for obs in env.obstacles:
        o_pos = (obs.x, obs.y)
        ax.add_patch(Circle(o_pos, obs.radius, color="lightgray", alpha=0.5))

    agent_body = Circle(history["pos"][0], 0.2, color="blue")
    plan_line, = ax.plot([], [], color="orange", linewidth=1.5)
    exec_line, = ax.plot([], [], color="blue", linewidth=1.2)
    opt_line, = ax.plot([], [], color="green", linewidth=2.0)

    ax.add_patch(agent_body)

    info_text = ax.text(0.5, 19.5, "", fontsize=10,
                        bbox=dict(facecolor="white", alpha=0.7))

    def update(frame):
        pos = history["pos"][frame]
        path = history["remaining_path"][frame]

        agent_body.center = pos

        # Planned path (remaining)
        if len(path) >= 2:
            pts = np.array(path)
            plan_line.set_data(pts[:, 0], pts[:, 1])
        else:
            plan_line.set_data([], [])

        # Executed all
        n_all = history["all_len"][frame]
        if n_all >= 2:
            pts = np.array(all_points[:n_all])
            exec_line.set_data(pts[:, 0], pts[:, 1])
        else:
            exec_line.set_data([], [])

        # Executed optimal
        opt_pts = history["executed_opt"][frame]
        if len(opt_pts) >= 2:
            pts = np.array(opt_pts)
            opt_line.set_data(pts[:, 0], pts[:, 1])
        else:
            opt_line.set_data([], [])

        # Replan detection
        replan = False
        if frame > 0:
            replan = history["replan_count"][frame] > history["replan_count"][frame - 1]
            plan_line.set_color("purple" if replan else "orange")

        # Info
        info_text.set_text(
            f"Step: {frame}\n"
            f"Replans: {history['replan_count'][frame]}\n"
            f"Fails: {history['plan_fail_count'][frame]}\n"
            f"Blocked: {getattr(agent, 'blocked_steps', 0)}\n"
            f"Invalid idx: {history['invalid_from_idx'][frame]}"
        )

        return agent_body, plan_line, exec_line, opt_line, info_text

    ani = animation.FuncAnimation(fig, update, frames=len(history["pos"]), interval=30)

    ax.set_xlim(0, 30)
    ax.set_ylim(0, 30)

    plt.title("RAPF v4 — Anchor-Based Replanning")
    plt.show()


if __name__ == "__main__":
    main()
