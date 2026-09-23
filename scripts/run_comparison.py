#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from comparison.registry import available_planners, build_agent
from comparison.runner import BenchmarkConfig, run_episode, write_results
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


def interleaved_jobs(planner_names, scenarios, seeds):
    """Yield every planner for one scenario/seed before advancing."""
    for scenario in scenarios:
        for seed in seeds:
            for planner_name in planner_names:
                yield planner_name, scenario, seed


def load_existing_results(path: str | Path):
    """Keep existing CSV rows and index completed planner/scenario/seed jobs."""
    path = Path(path)
    if not path.exists():
        return [], set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"planner", "scenario", "seed"}.issubset(reader.fieldnames):
            raise ValueError(f"{path} must have planner, scenario, and seed columns")
        rows = list(reader)
    completed = set()
    for row in rows:
        try:
            key = (row["planner"], row["scenario"], int(row["seed"]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid seed in {path}: {row['seed']!r}") from exc
        if not key[0] or not key[1] or key in completed:
            raise ValueError(f"Invalid or duplicate benchmark case in {path}: {key}")
        completed.add(key)
    return rows, completed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a planner through the common comparison framework."
    )
    parser.add_argument(
        "--planner",
        default="r2apf",
        help="One registered planner, or 'all'.",
    )
    parser.add_argument(
        "--benchmark-config",
        default="configs/benchmark.json",
    )
    parser.add_argument("--planner-config", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument(
        "--output",
        default="experiments/comparisons/results_22092026.csv",
    )
    args = parser.parse_args()

    if args.planner != "all" and args.planner not in available_planners():
        choices = ", ".join(available_planners())
        raise SystemExit(
            f"Planner '{args.planner}' is not registered. Available: {choices}"
        )

    campaign = load_json(args.benchmark_config)
    if args.planner == "all" and args.planner_config:
        raise SystemExit("--planner-config cannot be combined with --planner all")
    episodes = (
        int(args.episodes)
        if args.episodes is not None
        else int(campaign["episodes_per_scenario"])
    )
    seeds = tuple(range(
        int(campaign["seed_start"]),
        int(campaign["seed_start"]) + episodes,
    ))

    runner_config = BenchmarkConfig(
        dt=float(campaign["dt"]),
        speed_limit=float(campaign["speed_limit"]),
        max_steps=int(campaign["max_steps"]),
        agent_radius=float(campaign["agent_radius"]),
        collision_radius=float(campaign.get("collision_radius", 0.1)),
        stop_on_collision=bool(campaign["stop_on_collision"]),
        max_consecutive_plan_failures=int(
            campaign.get("max_consecutive_plan_failures", 0)
        ),
        max_consecutive_stationary_plan_failures=int(
            campaign.get("max_consecutive_stationary_plan_failures", 0)
        ),
        apf_stationary_steps=int(campaign.get("apf_stationary_steps", 100)),
        max_total_planning_time_s=(
            None
            if campaign.get("max_total_planning_time_s") is None
            else float(campaign["max_total_planning_time_s"])
        ),
    )
    shared_agent_config = {
        "sensing_range": float(campaign["sensing_range"]),
        "sensing_pad": float(campaign["sensing_pad"]),
        "agent_radius": float(campaign["agent_radius"]),
        "goal_tolerance": float(campaign["goal_tolerance"]),
        "lookahead_segments": int(campaign["lookahead_segments"]),
    }

    planner_names = available_planners() if args.planner == "all" else (args.planner,)
    planner_setups = {}
    for planner_name in planner_names:
        default_config = Path(f"configs/planners/{planner_name}.json")
        config_path = Path(args.planner_config) if args.planner_config else default_config
        planner_config = load_json(config_path) if config_path.exists() else {}

        if planner_name in {"rapf", "r2apf"}:
            agent_config = {
                "SENSE_RANGE": shared_agent_config["sensing_range"],
                "SENSE_PAD": shared_agent_config["sensing_pad"],
                "RHO_L": shared_agent_config["agent_radius"],
                "GOAL_TOLERANCE": shared_agent_config["goal_tolerance"],
                **planner_config,
            }
        else:
            agent_config = shared_agent_config
        planner_setups[planner_name] = (planner_config, agent_config)

    try:
        all_results, completed = load_existing_results(args.output)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Loaded {len(all_results)} existing rows from {args.output}; completed jobs will be skipped")
    completed_cases = 0
    total_cases = len(campaign["scenarios"]) * len(seeds)
    new_rows_in_case = 0
    for planner_name, scenario, seed in interleaved_jobs(
        planner_names,
        campaign["scenarios"],
        seeds,
    ):
        key = (planner_name, scenario, seed)
        if key not in completed:
            planner_config, agent_config = planner_setups[planner_name]
            all_results.append(run_episode(
                build_agent(
                    planner_name,
                    planner_config=planner_config,
                    agent_config=agent_config,
                ),
                environment_factory(scenario, seed),
                planner_name=planner_name,
                scenario=scenario,
                seed=seed,
                config=runner_config,
            ))
            completed.add(key)
            new_rows_in_case += 1

        # Checkpoint after each scenario/seed case, including partially completed cases.
        if planner_name == planner_names[-1]:
            completed_cases += 1
            if new_rows_in_case:
                write_results(all_results, args.output)
                print(
                    f"Completed case {completed_cases}/{total_cases}: "
                    f"scenario={scenario}, seed={seed}; "
                    f"added {new_rows_in_case} rows, checkpointed {len(all_results)} "
                    f"rows to {args.output}"
                )
            new_rows_in_case = 0
    print(f"Done: {len(all_results)} rows in {args.output}; skipped existing jobs")


if __name__ == "__main__":
    main()
