#!/usr/bin/env python3
"""Plot a collision-free local example of RAPF restart and R2APF repair.

Run from the repository root::

    python scripts/plot_rapf_repair_concept.py

The shared initial route and the two replanned routes are generated with the
repository's A* implementation and validated against the map obstacles. They
illustrate the recovery geometry; they are not recorded RAPF/R2APF runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch

from planners.baselines.astar import AStarPlanner
from planners.baselines.grid import segment_intersects_circle
from planners.rapf_global_planner import RAPFGlobalPlanner
from simulation.env.runtime import CircularObstacle
from simulation.env_gen_lunar import generate_lunar_env, make_config

# Map colors match scripts/plot_scenario_difficulty.py.
ROCK_COLOR = "#454545"
CRATER_COLOR = "#e5d4ad"
INITIAL_COLOR = "#565656"
ROUTE_COLOR = "#0072b2"
BACKTRACK_COLOR = "#cc79a7"
ARTIFICIAL_COLOR = "#d55e00"
START_COLOR = "#009e73"


def load_influence_radius() -> float:
    config_path = ROOT / "configs" / "planners" / "r2apf.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if "RHO_U" in config:
            return float(config["RHO_U"])
    return float(RAPFGlobalPlanner.DEFAULT_PARAMS["RHO_U"])


def inflated_obstacles(obstacles, clearance: float):
    return [CircularObstacle(center=o.center, radius=o.radius + clearance,
                             kind=o.kind) for o in obstacles]


def segment_clear(a, b, obstacles) -> bool:
    return not any(segment_intersects_circle(
        a, b, (o.x, o.y, o.radius)) for o in obstacles)


def validate_path(path, obstacles, description: str):
    points = np.asarray(path, dtype=float)
    if len(points) < 2 or not np.isfinite(points).all():
        raise RuntimeError(f"{description} is empty or non-finite")
    for a, b in zip(points[:-1], points[1:]):
        if not segment_clear(a, b, obstacles):
            raise RuntimeError(f"{description} intersects an obstacle")
    return points


def shortcut_path(path, obstacles):
    """Greedily remove grid corners while checking every new line segment."""
    points = np.asarray(path, dtype=float)
    if len(points) <= 2:
        return points
    result = [points[0]]
    start = 0
    while start < len(points) - 1:
        end = len(points) - 1
        while end > start + 1 and not segment_clear(points[start], points[end], obstacles):
            end -= 1
        result.append(points[end])
        start = end
    return np.asarray(result)


def sample_polyline(path, spacing: float | None = None):
    """Resample a route so the last point outside H is a visible waypoint."""
    if spacing is None:
        spacing = float(RAPFGlobalPlanner.DEFAULT_PARAMS["RHO_B"])
    points = np.asarray(path, dtype=float)
    sampled = [points[0]]
    for a, b in zip(points[:-1], points[1:]):
        length = float(np.linalg.norm(b - a))
        count = max(1, int(np.ceil(length / spacing)))
        sampled.extend(a + (b - a) * (i / count) for i in range(1, count + 1))
    return np.asarray(sampled)


def find_repair_anchor(initial_path, stuck, horizon):
    distances = np.linalg.norm(initial_path - stuck, axis=1)
    outside = np.flatnonzero(distances > horizon)
    if len(outside) == 0:
        raise RuntimeError("Initial route has no waypoint outside the influence horizon")
    index = int(outside[-1])
    return index, initial_path[index], float(distances[index])


def make_routes(obstacles, start, stuck, goal, horizon, radius, resolution, bounds):
    """Generate shared, restart and repair routes with segment clearance checks."""
    static_clear = inflated_obstacles(obstacles, radius)
    full_planner = AStarPlanner(resolution=resolution, agent_radius=radius)
    initial_result = full_planner.plan(start, stuck, obstacles, bounds)
    if not initial_result.success:
        raise RuntimeError("Could not find a collision-free initial route: "
                           f"{initial_result.failure_reason}")
    initial = shortcut_path(initial_result.path, static_clear)
    validate_path(initial, static_clear, "Shared initial route")
    initial = sample_polyline(initial)
    anchor_index, anchor, anchor_distance = find_repair_anchor(initial, stuck, horizon)

    # The repulsive influence boundary acts as a keep-out region for the new
    # suffix. Static obstacles are already expanded by the agent radius.
    virtual = CircularObstacle(center=tuple(stuck), radius=horizon, kind="virtual")
    suffix_obstacles = [*static_clear, virtual]
    suffix_planner = AStarPlanner(resolution=resolution, agent_radius=0.0)

    # Baseline RAPF restarts at p_stuck. Since p_stuck is at the artificial
    # obstacle centre, show its collision-free escape to just beyond H, then
    # route the rest around H and the real obstacles.
    escape_path = None
    escape_radius = horizon + max(0.2, resolution * 2.0)
    for angle in np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False):
        escape = stuck + escape_radius * np.array([np.cos(angle), np.sin(angle)])
        if not segment_clear(stuck, escape, static_clear):
            continue
        result = suffix_planner.plan(escape, goal, suffix_obstacles, bounds)
        if result.success:
            suffix = shortcut_path(result.path, suffix_obstacles)
            validate_path(suffix, suffix_obstacles, "RAPF replanned route outside H")
            escape_path = np.vstack((stuck, suffix))
            break
    if escape_path is None:
        raise RuntimeError("Could not find a collision-free RAPF escape and replanned route")

    # R2APF backtracks along the common initial path to its last waypoint
    # outside H, then generates a new suffix from that anchor.
    repair_result = suffix_planner.plan(anchor, goal, suffix_obstacles, bounds)
    if not repair_result.success:
        raise RuntimeError("Could not find a collision-free R2APF repaired route: "
                           f"{repair_result.failure_reason}")
    repaired_suffix = shortcut_path(repair_result.path, suffix_obstacles)
    validate_path(repaired_suffix, suffix_obstacles, "R2APF repaired route outside H")
    return initial, anchor_index, anchor, anchor_distance, escape_path, repaired_suffix


def draw_arrow(ax, points, color, index=None):
    if len(points) < 2:
        return
    if index is None:
        index = max(0, (len(points) - 2) // 2)
    index = min(index, len(points) - 2)
    a, b = points[index], points[index + 1]
    ax.annotate("", xy=a + 0.82 * (b - a), xytext=a + 0.48 * (b - a),
                arrowprops={"arrowstyle": "-|>", "color": color,
                            "linewidth": 2.2, "mutation_scale": 18},
                zorder=12)


def draw_map(ax, obstacles, stuck, half_width):
    for kind in ("crater", "rock"):
        for obstacle in obstacles:
            if obstacle.kind == kind:
                ax.add_patch(Circle(
                    obstacle.center, obstacle.radius,
                    facecolor=CRATER_COLOR if kind == "crater" else ROCK_COLOR,
                    edgecolor="#76633e" if kind == "crater" else "black",
                    linewidth=0.65 if kind == "crater" else 0.4,
                    zorder=2 if kind == "crater" else 3,
                ))
    ax.set(xlim=(stuck[0] - half_width, stuck[0] + half_width),
           ylim=(stuck[1] - half_width, stuck[1] + half_width),
           xlabel="x [m]", ylabel="y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.set_axisbelow(True)
    ax.grid(color="0.9", linewidth=0.5)
    ax.tick_params(length=3, pad=3, labelsize=10)
    ax.set_xticks(np.arange(np.ceil(stuck[0] - half_width),
                            np.floor(stuck[0] + half_width) + 1, 2))
    ax.set_yticks(np.arange(np.ceil(stuck[1] - half_width),
                            np.floor(stuck[1] + half_width) + 1, 2))


def draw_event(ax, stuck, artificial_radius, horizon):
    ax.add_patch(Circle(stuck, horizon, facecolor="none",
                        edgecolor=ARTIFICIAL_COLOR, linewidth=1.4,
                        linestyle=(0, (4, 2)), zorder=6))
    ax.add_patch(Circle(stuck, artificial_radius, facecolor=ARTIFICIAL_COLOR,
                        edgecolor=ARTIFICIAL_COLOR, alpha=0.38,
                        linewidth=1.0, zorder=7))
    ax.plot(*stuck, marker="x", color="black", markersize=7,
            markeredgewidth=1.5, zorder=13)
    ax.annotate("$p_{\\rm stuck}$", xy=stuck, xytext=(5, -17),
                textcoords="offset points", fontsize=10,
                bbox={"boxstyle": "round,pad=0.12", "fc": "white",
                      "ec": "none", "alpha": 0.85}, zorder=14)


def create_figure(scenario="C", seed=5073, width_mm=210.0, font_size=10.0,
                  artificial_radius=None, influence_radius=None,
                  start=(9.0, 13.0), stuck=(14.0, 9.0), goal=(21.0, 9.0),
                  agent_radius=0.3, resolution=0.1, half_width=5.8):
    if artificial_radius is None:
        artificial_radius = float(RAPFGlobalPlanner.DEFAULT_PARAMS["ARTIFICIAL_R"])
    if influence_radius is None:
        influence_radius = load_influence_radius()
    horizon = artificial_radius + influence_radius
    start, stuck, goal = (np.asarray(p, dtype=float) for p in (start, stuck, goal))
    config = make_config(scenario, seed=seed)
    obstacles = generate_lunar_env(config)
    routes = make_routes(obstacles, start, stuck, goal, horizon, agent_radius,
                         resolution, (0.0, config.L, 0.0, config.L))
    initial, anchor_index, anchor, anchor_distance, restart_path, repaired = routes
    backtrack = initial[anchor_index:][::-1]

    style = {"font.family": "DejaVu Sans", "font.size": font_size,
             "axes.labelsize": font_size, "xtick.labelsize": font_size,
             "ytick.labelsize": font_size, "legend.fontsize": font_size,
             "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"}
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=(width_mm / 25.4, 145 / 25.4),
                                 sharex=True, sharey=True)
        fig.subplots_adjust(left=0.075, right=0.985, bottom=0.235,
                            top=0.89, wspace=0.17)
        axes[0].set_title("RAPF — full restart", fontsize=font_size + 1,
                          fontweight="bold", pad=9)
        axes[1].set_title("R2APF — partial-trajectory repair",
                          fontsize=font_size + 1, fontweight="bold", pad=9)

        for ax in axes:
            draw_map(ax, obstacles, stuck, half_width)
            draw_event(ax, stuck, artificial_radius, horizon)
            # Identical collision-free route in both panels.
            ax.plot(initial[:, 0], initial[:, 1], color=INITIAL_COLOR,
                    linewidth=1.8, zorder=8)
            ax.plot(*start, marker="o", color=START_COLOR,
                    markeredgecolor="black", markeredgewidth=0.5,
                    markersize=5.5, zorder=11)

        # RAPF clears the existing path and starts the new path at p_stuck.
        axes[0].plot(restart_path[:, 0], restart_path[:, 1],
                     color=ROUTE_COLOR, linewidth=2.3, zorder=10)
        draw_arrow(axes[0], restart_path, ROUTE_COLOR, index=2)

        # R2APF backtracks from p_stuck to a, then replans from a.
        axes[1].plot(backtrack[:, 0], backtrack[:, 1],
                     color=BACKTRACK_COLOR, linewidth=2.5, zorder=9)
        draw_arrow(axes[1], backtrack, BACKTRACK_COLOR)
        axes[1].plot(repaired[:, 0], repaired[:, 1],
                     color=ROUTE_COLOR, linewidth=2.3, zorder=10)
        axes[1].plot(*anchor, marker="o", color=BACKTRACK_COLOR,
                     markeredgecolor="black", markeredgewidth=0.5,
                     markersize=6, zorder=12)
        axes[1].annotate("$a$", xy=anchor, xytext=(-5, 7),
                         textcoords="offset points", fontsize=font_size,
                         ha="right", va="bottom",
                         bbox={"boxstyle": "round,pad=0.12", "fc": "white",
                               "ec": "none", "alpha": 0.88}, zorder=14)

        handles = [
            Line2D([], [], color=INITIAL_COLOR, linewidth=1.8,
                   label="Initial path (shared)"),
            Line2D([], [], color=ROUTE_COLOR, linewidth=2.3,
                   label="Illustrative replanned path"),
            Line2D([], [], color=BACKTRACK_COLOR, linewidth=2.5,
                   label="Backtrack to anchor"),
            Line2D([], [], marker="o", color=START_COLOR,
                   markeredgecolor="black", linestyle="none", markersize=5.5,
                   label="Start"),
            Line2D([], [], marker="x", color="black", linestyle="none",
                   markersize=7, label="$p_{\\rm stuck}$"),
            Line2D([], [], marker="o", color=BACKTRACK_COLOR,
                   markeredgecolor="black", linestyle="none", markersize=6,
                   label="Repair anchor $a$"),
            Patch(facecolor=ARTIFICIAL_COLOR, edgecolor=ARTIFICIAL_COLOR,
                  alpha=0.38, label="Artificial obstacle"),
            Line2D([], [], color=ARTIFICIAL_COLOR, linestyle=(0, (4, 2)),
                   linewidth=1.4, label="Influence boundary $R_{AO}+R_u$"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=4,
                   bbox_to_anchor=(0.5, 0.015), frameon=False,
                   columnspacing=1.15, handlelength=1.8, handletextpad=0.5)

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for artist in [*fig.legends, *[item for ax in axes for item in
                       [ax.title, ax.xaxis.label, ax.yaxis.label,
                        *ax.get_xticklabels(), *ax.get_yticklabels()]]]:
            extent = artist.get_window_extent(renderer)
            if extent.width and (not fig.bbox.contains(*extent.min)
                                 or not fig.bbox.contains(*extent.max)):
                raise ValueError("Text extends outside the figure; increase width or height")

    metadata = {
        "scenario_code": scenario, "seed": seed, "width_mm": width_mm,
        "font_size_pt": font_size, "start_xy_m": start.tolist(),
        "stuck_xy_m": stuck.tolist(), "goal_xy_m": goal.tolist(),
        "initial_route_waypoints": len(initial), "initial_route_collision_free": True,
        "artificial_obstacle_radius_m": artificial_radius,
        "repulsive_influence_radius_m": influence_radius,
        "influence_horizon_m": horizon, "repair_anchor_xy_m": anchor.tolist(),
        "repair_anchor_distance_m": anchor_distance,
        "repair_anchor_outside_horizon": anchor_distance > horizon,
        "agent_clearance_radius_m": agent_radius,
        "route_visualization": "A* routes, line-segment validated; recovery schematic",
    }
    return fig, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", choices=("A", "B", "C", "D", "E"), default="C")
    parser.add_argument("--seed", type=int, default=5073)
    parser.add_argument("--width-mm", type=float, default=210.0)
    parser.add_argument("--font-size", type=float, default=10.0)
    parser.add_argument("--artificial-radius", type=float, default=None)
    parser.add_argument("--influence-radius", type=float, default=None)
    parser.add_argument("--start", type=float, nargs=2, default=(9.0, 13.0), metavar=("X", "Y"))
    parser.add_argument("--stuck", type=float, nargs=2, default=(14.0, 9.0), metavar=("X", "Y"))
    parser.add_argument("--goal", type=float, nargs=2, default=(21.0, 9.0), metavar=("X", "Y"))
    parser.add_argument("--agent-radius", type=float, default=0.3)
    parser.add_argument("--resolution", type=float, default=0.1)
    parser.add_argument("--half-width", type=float, default=5.8)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "figures" / "rapf_repair_concept",
                        help="Output stem; writes .pdf, .svg, .png and .json")
    args = parser.parse_args()
    if (not np.isfinite(args.width_mm) or args.width_mm <= 0
            or not np.isfinite(args.font_size) or args.font_size <= 0
            or args.dpi <= 0 or args.agent_radius < 0 or args.resolution <= 0
            or args.half_width <= 0):
        parser.error("width, font size, resolution and crop must be positive; "
                     "DPI positive and agent radius nonnegative")
    fig, metadata = create_figure(
        scenario=args.scenario, seed=args.seed, width_mm=args.width_mm,
        font_size=args.font_size, artificial_radius=args.artificial_radius,
        influence_radius=args.influence_radius, start=args.start, stuck=args.stuck,
        goal=args.goal, agent_radius=args.agent_radius, resolution=args.resolution,
        half_width=args.half_width)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "svg", "png"):
        target = args.output.with_suffix("." + extension)
        fig.savefig(target, dpi=args.dpi, facecolor="white")
        print(f"Saved {target}")
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    plt.close(fig)
    print(f"H={metadata['influence_horizon_m']:.3f} m; "
          f"anchor distance={metadata['repair_anchor_distance_m']:.3f} m")


if __name__ == "__main__":
    main()
