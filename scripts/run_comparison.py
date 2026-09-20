#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from comparison.registry import available_planners, build_agent
from comparison.runner import BenchmarkConfig, run_campaign, write_results
from simulation.env_gen_lunar import generate_lunar_env, make_config
from simulation.sim_adapter import make_runtime_env_from_geometry


def load_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def environment_factory(scenario: str, seed: int):
    config = make_config(scenario, seed=seed)
    geometry = generate_lunar_env(config)
    return make_runtime_env_from_geometry(
        geometry,
        config.L,
        n_agents=1,
        seed=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a planner through the common comparison framework."
    )
    parser.add_argument("--planner", default="r2apf")
    parser.add_argument(
        "--benchmark-config",
        default="configs/benchmark.json",
    )
    parser.add_argument("--planner-config", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument(
        "--output",
        default="experiments/comparisons/results.csv",
    )
    args = parser.parse_args()

    if args.planner not in available_planners():
        choices = ", ".join(available_planners())
        raise SystemExit(
            f"Planner '{args.planner}' is not registered. Available: {choices}"
        )

    campaign = load_json(args.benchmark_config)
    planner_config = (
        load_json(args.planner_config)
        if args.planner_config
        else {}
    )
    episodes = (
        int(args.episodes)
        if args.episodes is not None
        else int(campaign["episodes_per_scenario"])
    )
    seeds = range(
        int(campaign["seed_start"]),
        int(campaign["seed_start"]) + episodes,
    )

    runner_config = BenchmarkConfig(
        dt=float(campaign["dt"]),
        speed_limit=float(campaign["speed_limit"]),
        max_steps=int(campaign["max_steps"]),
        agent_radius=float(campaign["agent_radius"]),
        stop_on_collision=bool(campaign["stop_on_collision"]),
    )
    shared_agent_config = {
        "sensing_range": float(campaign["sensing_range"]),
        "sensing_pad": float(campaign["sensing_pad"]),
        "agent_radius": float(campaign["agent_radius"]),
        "goal_tolerance": float(campaign["goal_tolerance"]),
        "lookahead_segments": int(campaign["lookahead_segments"]),
    }

    if args.planner in {"rapf", "r2apf"}:
        agent_config = {
            "SENSE_RANGE": shared_agent_config["sensing_range"],
            "SENSE_PAD": shared_agent_config["sensing_pad"],
            "RHO_L": shared_agent_config["agent_radius"],
            "GOAL_TOLERANCE": shared_agent_config["goal_tolerance"],
            **planner_config,
        }
    else:
        agent_config = shared_agent_config

    results = run_campaign(
        lambda: build_agent(
            args.planner,
            planner_config=planner_config,
            agent_config=agent_config,
        ),
        environment_factory,
        planner_name=args.planner,
        scenarios=campaign["scenarios"],
        seeds=seeds,
        config=runner_config,
    )
    write_results(results, args.output)
    print(f"Wrote {len(results)} episodes to {args.output}")


if __name__ == "__main__":
    main()
