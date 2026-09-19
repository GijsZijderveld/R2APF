#!/usr/bin/env python
"""Animate one planner in the common partially observed lunar environment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from comparison.interfaces import BenchmarkObservation
from comparison.registry import available_planners, build_agent
from comparison.sensing import obstacle_center, obstacle_key
from simulation.env_gen_lunar import generate_lunar_env, make_config
from simulation.sim_adapter import make_runtime_env_from_geometry


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def make_environment(scenario: str, seed: int):
    config = make_config(scenario, seed=seed)
    geometry = generate_lunar_env(config)
    return make_runtime_env_from_geometry(
        geometry,
        config.L,
        n_agents=1,
        seed=seed,
    )


def current_path(agent: Any) -> list[np.ndarray]:
    """Return only the unexecuted part of either supported agent type."""
    if hasattr(agent, "wrapped_agent"):
        wrapped = agent.wrapped_agent
        path = getattr(wrapped, "planned_path", []) or []
        index = int(getattr(wrapped, "path_index", 0))
    else:
        path = getattr(agent, "_path", []) or []
        index = int(getattr(agent, "_path_index", 0))
    return [np.asarray(point, dtype=float).copy() for point in path[index:]]


def known_obstacles(agent: Any) -> list[Any]:
    if hasattr(agent, "wrapped_agent"):
        return list(getattr(agent.wrapped_agent, "perceived_obstacles", []))
    return list(getattr(agent, "_known_obstacles", []))


def virtual_obstacles(agent: Any) -> list[tuple[np.ndarray, float]]:
    if not hasattr(agent, "wrapped_agent"):
        return []
    raw = getattr(agent.wrapped_agent, "last_virtual_obstacles", []) or []
    normalized = []
    for obstacle in raw:
        if isinstance(obstacle, dict):
            center = obstacle.get("position", obstacle.get("pos"))
            radius = obstacle.get("radius")
        else:
            center = obstacle_center(obstacle)
            radius = getattr(obstacle, "radius")
        if center is not None and radius is not None:
            normalized.append((np.asarray(center, dtype=float), float(radius)))
    return normalized


def planner_agent_config(campaign: dict[str, Any], planner: str) -> dict[str, Any]:
    shared = {
        "sensing_range": float(campaign["sensing_range"]),
        "sensing_pad": float(campaign["sensing_pad"]),
        "agent_radius": float(campaign["agent_radius"]),
        "goal_tolerance": float(campaign["goal_tolerance"]),
        "lookahead_segments": int(campaign["lookahead_segments"]),
    }
    if planner != "r2apf":
        return shared
    return {
        "SENSE_RANGE": shared["sensing_range"],
        "SENSE_PAD": shared["sensing_pad"],
        "RHO_L": shared["agent_radius"],
        "GOAL_TOLERANCE": shared["goal_tolerance"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Animate planning, sensing, movement, and replanning."
    )
    parser.add_argument("--planner", default="astar", choices=available_planners())
    parser.add_argument("--scenario", default="A", choices=("A", "B", "C", "D", "E"))
    parser.add_argument("--seed", type=int, default=5000)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--interval-ms", type=int, default=35)
    parser.add_argument("--benchmark-config", default="configs/benchmark.json")
    parser.add_argument(
        "--planner-config",
        default=None,
        help="Defaults to configs/planners/<planner>.json.",
    )
    parser.add_argument("--save", default=None, help="Optional .gif or .mp4 output path.")
    parser.add_argument("--no-show", action="store_true", help="Do not open a plot window.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    campaign = load_json(args.benchmark_config)
    config_path = args.planner_config or f"configs/planners/{args.planner}.json"
    planner_config = load_json(config_path)
    environment = make_environment(args.scenario, args.seed)
    agent_config = planner_agent_config(campaign, args.planner)
    if args.planner == "r2apf":
        agent_config.update(planner_config)
        planner_config = {}

    agent = build_agent(
        args.planner,
        planner_config=planner_config,
        agent_config=agent_config,
    )
    agent.reset(environment, args.seed)

    dt = float(campaign["dt"])
    speed_limit = float(campaign["speed_limit"])
    agent_radius = float(campaign["agent_radius"])
    sensing_range = float(campaign["sensing_range"])
    bounds = environment.bounds()
    xmin, xmax, ymin, ymax = map(float, bounds)

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.set_title(
        f"{args.planner} — scenario {args.scenario}, seed {args.seed}"
    )

    obstacle_patches: dict[tuple[float, float, float], Circle] = {}
    for obstacle in environment.obstacles:
        center = obstacle_center(obstacle)
        patch = Circle(
            center,
            float(obstacle.radius),
            facecolor="0.88",
            edgecolor="0.72",
            linewidth=0.7,
            alpha=0.35,
            zorder=1,
        )
        obstacle_patches[obstacle_key(obstacle)] = patch
        ax.add_patch(patch)

    start = np.asarray(agent.position, dtype=float).copy()
    goal = np.asarray(agent.goal, dtype=float).copy()
    ax.plot(start[0], start[1], marker="o", color="tab:green", markersize=7, label="Start")
    ax.plot(goal[0], goal[1], marker="*", color="tab:red", markersize=14, label="Goal")

    rover = Circle(start, agent_radius, color="tab:blue", alpha=0.85, zorder=6)
    sensor = Circle(
        start,
        sensing_range,
        fill=False,
        edgecolor="tab:blue",
        linestyle="--",
        linewidth=1.0,
        alpha=0.35,
        zorder=2,
    )
    ax.add_patch(rover)
    ax.add_patch(sensor)

    executed_line, = ax.plot([], [], color="tab:blue", linewidth=2.0, label="Executed", zorder=4)
    plan_line, = ax.plot([], [], color="tab:orange", linewidth=1.8, label="Current plan", zorder=3)
    replan_scatter = ax.scatter([], [], marker="x", s=65, color="purple", label="Replan", zorder=7)
    status = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        family="monospace",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.75"},
        zorder=10,
    )
    ax.legend(loc="lower right")

    executed = [start]
    replan_positions: list[np.ndarray] = []
    virtual_patches: list[Circle] = []
    previous_planning_calls = 0
    finished = False

    def draw_frame(frame: int):
        nonlocal previous_planning_calls, finished, virtual_patches

        if not finished:
            before = np.asarray(agent.position, dtype=float).copy()
            agent.step(
                BenchmarkObservation(environment=environment, time_step=frame, bounds=bounds),
                dt=dt,
                speed_limit=speed_limit,
            )
            after = np.asarray(agent.position, dtype=float).copy()
            if np.linalg.norm(after - before) > 0.0:
                executed.append(after)

        diagnostics = dict(agent.diagnostics())
        planning_calls = int(diagnostics.get("planning_calls", 0))
        replanned = planning_calls > previous_planning_calls
        if replanned:
            replan_positions.append(np.asarray(agent.position, dtype=float).copy())
        previous_planning_calls = planning_calls

        position = np.asarray(agent.position, dtype=float)
        rover.center = position
        sensor.center = position

        executed_array = np.asarray(executed)
        executed_line.set_data(executed_array[:, 0], executed_array[:, 1])

        path = current_path(agent)
        if path:
            path_array = np.asarray([position, *path])
            plan_line.set_data(path_array[:, 0], path_array[:, 1])
        else:
            plan_line.set_data([], [])
        plan_line.set_color("purple" if replanned else "tab:orange")

        sensed_keys = {obstacle_key(obstacle) for obstacle in known_obstacles(agent)}
        for key, patch in obstacle_patches.items():
            if key in sensed_keys:
                patch.set_facecolor("tab:gray")
                patch.set_edgecolor("black")
                patch.set_alpha(0.75)

        for patch in virtual_patches:
            patch.remove()
        virtual_patches = []
        for center, radius in virtual_obstacles(agent):
            patch = Circle(
                center,
                radius,
                fill=False,
                edgecolor="magenta",
                linestyle=":",
                linewidth=1.3,
                zorder=2,
            )
            ax.add_patch(patch)
            virtual_patches.append(patch)

        if replan_positions:
            replan_scatter.set_offsets(np.asarray(replan_positions))

        status.set_text(
            f"step:       {frame + 1}\n"
            f"known obs:  {len(sensed_keys)}/{len(environment.obstacles)}\n"
            f"plans:      {planning_calls}\n"
            f"replans:    {diagnostics.get('replans', max(0, planning_calls - 1))}\n"
            f"failures:   {diagnostics.get('plan_failures', 0)}\n"
            f"blocked:    {diagnostics.get('blocked_steps', 0)}\n"
            f"goal dist:  {np.linalg.norm(position - goal):.2f} m"
        )

        if agent.reached_goal:
            finished = True
            status.set_text(status.get_text() + "\nGOAL REACHED")
            if args.save is None and ani.event_source is not None:
                ani.event_source.stop()

        return (
            rover,
            sensor,
            executed_line,
            plan_line,
            replan_scatter,
            status,
            *obstacle_patches.values(),
            *virtual_patches,
        )

    ani = animation.FuncAnimation(
        fig,
        draw_frame,
        frames=args.steps,
        interval=args.interval_ms,
        repeat=False,
        blit=False,
        cache_frame_data=False,
    )

    if args.save:
        output = Path(args.save)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.suffix.lower() == ".gif":
            ani.save(output, writer="pillow", fps=max(1, round(1000 / args.interval_ms)))
        else:
            ani.save(output, fps=max(1, round(1000 / args.interval_ms)))
        print(f"Saved animation to {output}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
