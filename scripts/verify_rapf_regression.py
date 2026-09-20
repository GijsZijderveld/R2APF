#!/usr/bin/env python
"""Replay historical RAPF/R2APF seeds and compare deterministic metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from agents.navigation_core import NavigationCore, NAVIGATION_PARAMS
from benchmark.benchmark_sensing_RAPF import _run_one_episode
from planners.rapf_global_planner import RAPFGlobalPlanner as R2APFPlanner
from planners.rapf_paper_planner import RAPFGlobalPlanner as RAPFPlanner
from simulation.env_gen_lunar import generate_lunar_env, make_config
from simulation.sim_adapter import make_runtime_env_from_geometry
import simulation.sim_adapter as simulation_adapter


PROFILES = {
    "r2apf": {
        "planner": R2APFPlanner,
        "runner": "legacy_simulator",
        "reference": "experiments/benchmarks/sensing_v4_withvo.csv",
        "columns": {
            "Success": "Success",
            "PathLength": "PathLength",
            "ExecSteps": "Steps",
            "MaxActiveVirt": "MaxActiveVirt",
        },
    },
    "rapf": {
        "planner": RAPFPlanner,
        "runner": "manual_loop",
        "reference": "experiments/benchmarks/sensing_v4_oldRAPF.csv",
        "columns": {
            "Success": "Success",
            "PathLength": "PathLength",
            "ExecSteps": "ExecSteps",
            "MaxActiveVirt": "MaxActiveVirt",
        },
    },
}


def make_episode(planner_class, scenario: str, seed: int):
    config = make_config(scenario, seed=seed)
    geometry = generate_lunar_env(config)
    environment = make_runtime_env_from_geometry(
        geometry, config.L, n_agents=1, seed=seed
    )
    agent = NavigationCore(
        agent_id=0,
        position=environment.starts[0],
        goal=environment.goal,
        **NAVIGATION_PARAMS,
    )
    agent.planner = planner_class(**agent.p)
    return environment, agent


def replay(profile: dict, scenario: str, seed: int) -> dict:
    environment, agent = make_episode(profile["planner"], scenario, seed)
    if profile["runner"] == "legacy_simulator":
        # sensing_v4_withvo.csv was generated at swarm_apf_sim commit
        # 051040a. Its simulate_episode imported VEL_CLAMP=1.5 m/s.
        previous_speed = simulation_adapter.VEL_CLAMP
        simulation_adapter.VEL_CLAMP = 1.5
        try:
            metrics = simulation_adapter.simulate_episode(
                [agent], environment, seed=seed, max_steps=1000, dt=0.1
            )
        finally:
            simulation_adapter.VEL_CLAMP = previous_speed
        return {
            "Success": bool(metrics.reached),
            "PathLength": round(metrics.path_length, 2),
            "ExecSteps": int(metrics.time_steps),
            "MaxActiveVirt": int(agent.virtual_max_active),
        }

    return _run_one_episode(
        environment,
        agent,
        max_ticks=1000,
        speed=1.0,
        vo_round_decimals=3,
    )


def equal_value(actual, expected, metric: str) -> bool:
    if metric == "PathLength":
        return abs(float(actual) - float(expected)) <= 0.001
    return actual == expected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--planner", choices=("rapf", "r2apf", "both"), default="both"
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=5000)
    parser.add_argument(
        "--scenarios", nargs="+", default=["A", "B", "C", "D", "E"]
    )
    args = parser.parse_args()

    names = ("rapf", "r2apf") if args.planner == "both" else (args.planner,)
    failed = False
    for name in names:
        profile = PROFILES[name]
        reference = pd.read_csv(REPOSITORY_ROOT / profile["reference"])
        matches = 0
        total = 0
        print(f"\n{name.upper()} vs {profile['reference']}")
        for scenario in args.scenarios:
            for seed in range(args.seed_start, args.seed_start + args.episodes):
                expected_rows = reference[
                    (reference["Scenario"] == scenario)
                    & (reference["Seed"] == seed)
                ]
                if len(expected_rows) != 1:
                    raise ValueError(
                        f"Expected one reference row for {scenario}/{seed}, "
                        f"found {len(expected_rows)}."
                    )
                expected = expected_rows.iloc[0]
                actual = replay(profile, scenario, seed)
                differences = []
                for actual_key, reference_key in profile["columns"].items():
                    if not equal_value(
                        actual[actual_key], expected[reference_key], actual_key
                    ):
                        differences.append(
                            f"{actual_key}={actual[actual_key]} "
                            f"(reference {expected[reference_key]})"
                        )
                total += 1
                if differences:
                    failed = True
                    print(f"  MISMATCH {scenario}/{seed}: " + "; ".join(differences))
                else:
                    matches += 1
                    print(f"  MATCH    {scenario}/{seed}")
        print(f"  Result: {matches}/{total} episodes match all deterministic metrics")

    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
