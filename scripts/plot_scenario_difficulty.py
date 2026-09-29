#!/usr/bin/env python3
"""Paper maps: code A/E/C become paper A/C/E (easy/intermediate/hard).

Run from the repository root::

    python scripts/plot_scenario_difficulty.py
    python scripts/plot_scenario_difficulty.py --no-route
    python scripts/plot_scenario_difficulty.py --width-mm 170 --font-size 10

Defaults: exactly 210 mm wide (A4), 10 pt text, seed 5000, PDF/SVG/PNG.
Use the PDF or SVG at its intended width to preserve small obstacle outlines.
For a paper with margins, set --width-mm to its actual text width rather than
shrinking the 210 mm figure; fonts then retain their physical point sizes.

Routes are illustrative full-information A* paths, NOT sensing-based benchmark
trajectories or evidence of measured difficulty. All panels share a seed,
start/goal generation, scale and planner settings. Difficulty labels follow the
paper's prescribed ordering; summed obstacle area budgets remain constant.
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
from simulation.env_gen_lunar import generate_lunar_env, make_config
from simulation.sim_adapter import make_runtime_env_from_geometry

PANELS = (("A", "A", "Easy"), ("E", "C", "Intermediate"), ("C", "E", "Hard"))
ROCK_COLOR = "#454545"
CRATER_COLOR = "#e5d4ad"
ROUTE_COLOR = "#0072b2"


def positive(value: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def validate_route(path, obstacles, radius, bounds):
    """Check every segment, including the grid-to-start/goal connectors."""
    xmin, xmax, ymin, ymax = bounds
    points = np.asarray(path, dtype=float)
    if len(points) < 2 or not np.isfinite(points).all():
        raise RuntimeError("Illustrative route is empty or non-finite")
    if not ((points[:, 0] >= xmin + radius).all()
            and (points[:, 0] <= xmax - radius).all()
            and (points[:, 1] >= ymin + radius).all()
            and (points[:, 1] <= ymax - radius).all()):
        raise RuntimeError("Illustrative route violates map boundary clearance")
    for obstacle in obstacles:
        circle = (obstacle.x, obstacle.y, obstacle.radius + radius)
        if any(segment_intersects_circle(a, b, circle)
               for a, b in zip(points[:-1], points[1:])):
            raise RuntimeError("Illustrative route violates obstacle clearance")
    return points


def create_figure(seed=5000, width_mm=210.0, font_size=10.0,
                  resolution=0.1, agent_radius=0.25, show_route=True):
    style = {"font.family": "DejaVu Sans", "font.size": font_size,
             "axes.labelsize": font_size, "xtick.labelsize": font_size,
             "ytick.labelsize": font_size, "legend.fontsize": font_size,
             "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"}
    with plt.rc_context(style):
        # Extra physical height leaves room for titles, counts and one legend.
        fig, axes = plt.subplots(1, 3, figsize=(width_mm / 25.4,
                                               (width_mm / 3 + 34) / 25.4),
                                 sharex=True, sharey=True)
        fig.subplots_adjust(left=0.075, right=0.985, bottom=0.25,
                            top=0.82, wspace=0.14)
        metadata = []
        for ax, (code, paper, difficulty) in zip(axes, PANELS):
            config = make_config(code, seed=seed)
            geometry = generate_lunar_env(config)
            world = make_runtime_env_from_geometry(geometry, config.L,
                                                   n_agents=1, seed=seed)
            start, goal = world.starts[0], world.goals[0]
            route = None
            if show_route:
                planner = AStarPlanner(resolution=resolution, agent_radius=agent_radius)
                result = planner.plan(start, goal, world.obstacles, world.bounds())
                if not result.success:
                    plt.close(fig)
                    raise RuntimeError(f"No illustrative route for code {code}, seed {seed}: "
                                       f"{result.failure_reason}; choose another seed")
                route = validate_route(result.path, geometry, agent_radius, world.bounds())
            # Craters first, so small rocks remain visible when disks overlap.
            for kind in ("crater", "rock"):
                for obstacle in geometry:
                    if obstacle.kind == kind:
                        ax.add_patch(Circle(obstacle.center, obstacle.radius,
                            facecolor=ROCK_COLOR if kind == "rock" else CRATER_COLOR,
                            edgecolor="black" if kind == "rock" else "#76633e",
                            linewidth=0.4 if kind == "rock" else 0.65,
                            zorder=3 if kind == "rock" else 2))
            if show_route:
                ax.plot(route[:, 0], route[:, 1], color=ROUTE_COLOR, linewidth=1.15, zorder=4)
            ax.plot(*start, "o", color="#009e73", markeredgecolor="black",
                    markeredgewidth=0.6, markersize=6, zorder=5)
            ax.plot(*goal, "*", color="#d55e00", markeredgecolor="black",
                    markeredgewidth=0.6, markersize=10, zorder=5)
            ax.set(xlim=(0, config.L), ylim=(0, config.L),
                   xticks=(0, 10, 20, 30), yticks=(0, 10, 20, 30), xlabel="x [m]")
            ax.set_aspect("equal", adjustable="box")
            ax.set_axisbelow(True)
            ax.grid(color="0.9", linewidth=0.5)
            ax.tick_params(length=3, pad=3)
            ax.set_title(f"{paper} — {difficulty}", fontsize=font_size + 1,
                         fontweight="bold", pad=25)
            ax.text(0.5, 1.035,
                    f"{config.rock_count} rocks · {config.crater_count} craters",
                    transform=ax.transAxes, ha="center", fontsize=font_size)
            metadata.append({"code_scenario": code, "paper_scenario": paper,
                "difficulty": difficulty, "seed": seed,
                "rock_count": config.rock_count, "crater_count": config.crater_count,
                "start": start.tolist(), "goal": goal.tolist(),
                "route_length_m": (float(np.linalg.norm(np.diff(route, axis=0), axis=1).sum())
                                   if show_route else None)})
        axes[0].set_ylabel("y [m]")
        handles = [Patch(facecolor=ROCK_COLOR, edgecolor="black", label="Rock"),
                   Patch(facecolor=CRATER_COLOR, edgecolor="#76633e", label="Crater"),
                   Line2D([], [], marker="o", color="none", markerfacecolor="#009e73",
                          markeredgecolor="black", markersize=6, label="Start"),
                   Line2D([], [], marker="*", color="none", markerfacecolor="#d55e00",
                          markeredgecolor="black", markersize=10, label="Goal")]
        if show_route:
            handles.append(Line2D([], [], color=ROUTE_COLOR, linewidth=1.15,
                                  label="A* route (full map)"))
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.025),
                   ncol=3 if show_route else 4, frameon=False, columnspacing=1.5, handlelength=1.6)
        fig.canvas.draw()
        # Fail explicitly if an unusually small width or large font clips text.
        renderer = fig.canvas.get_renderer()
        for artist in [*fig.legends, *[item for ax in axes for item in
                       [ax.title, *ax.texts, ax.xaxis.label, ax.yaxis.label,
                        *ax.get_xticklabels(), *ax.get_yticklabels()]]]:
            if artist.get_visible() and artist.get_window_extent(renderer).width:
                if not fig.bbox.contains(*artist.get_window_extent(renderer).min) or \
                   not fig.bbox.contains(*artist.get_window_extent(renderer).max):
                    raise ValueError("Text extends outside the figure; increase --width-mm "
                                     "or reduce --font-size")
    return fig, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-route", action="store_true",
                        help="Show maps, obstacles, start and goal without running a planner")
    parser.add_argument("--seed", type=int, default=5000)
    parser.add_argument("--width-mm", type=positive, default=210.0)
    parser.add_argument("--font-size", type=positive, default=10.0)
    parser.add_argument("--resolution", type=positive, default=0.1,
                        help="Illustrative A* grid spacing in metres (not benchmark config)")
    parser.add_argument("--agent-radius", type=positive, default=0.25)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "figures" / "scenario_difficulty",
                        help="Output stem; writes .pdf, .svg, .png and .json")
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    fig, panels = create_figure(args.seed, args.width_mm, args.font_size,
                               args.resolution, args.agent_radius,
                               show_route=not args.no_route)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Do not use bbox_inches='tight': it changes the requested physical width.
    for extension in ("pdf", "svg", "png"):
        target = args.output.with_suffix("." + extension)
        fig.savefig(target, dpi=args.dpi, facecolor="white")
        print(f"Saved {target}")
    args.output.with_suffix(".json").write_text(json.dumps({
        "width_mm": args.width_mm, "font_size_pt": args.font_size,
        "route_method": (None if args.no_route else
                         "full-information A*; illustrative, not benchmark execution"),
        "resolution_m": args.resolution, "agent_radius_m": args.agent_radius,
        "panels": panels}, indent=2) + "\n", encoding="utf-8")
    plt.close(fig)
    print("Paper labels A/C/E correspond to code A/E/C. Coverage budgets are fixed.")


if __name__ == "__main__":
    main()
