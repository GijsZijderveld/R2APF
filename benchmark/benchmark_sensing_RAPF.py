#!/usr/bin/env python3
import argparse
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd

from simulation.env_gen_lunar import make_config, generate_lunar_env
from simulation.sim_adapter import make_runtime_env_from_geometry
from simulation.constants import DT
from agents.navigation_core import NavigationCore, NAVIGATION_PARAMS


def _normalize_virtual_obstacles(vobs):
    out = []
    for v in vobs or []:
        if isinstance(v, dict):
            pos = np.array(v.get("position", v.get("pos")), dtype=float)
            rad = float(v["radius"])
        else:
            if hasattr(v, "x") and hasattr(v, "y"):
                pos = np.array([v.x, v.y], dtype=float)
            elif hasattr(v, "center"):
                pos = np.array(v.center, dtype=float)
            elif hasattr(v, "position"):
                pos = np.array(v.position, dtype=float)
            else:
                raise TypeError(f"Unknown virtual obstacle type: {type(v)}")
            rad = float(getattr(v, "radius"))
        out.append({"position": np.array(pos, dtype=float), "radius": rad})
    return out


def _path_signature(path):
    if path and len(path) >= 2:
        return (
            len(path),
            float(path[0][0]),
            float(path[0][1]),
            float(path[-1][0]),
            float(path[-1][1]),
        )
    return None


def _distance_to_obstacle_surface(q, obs):
    if hasattr(obs, "distance_to_surface"):
        return float(obs.distance_to_surface(q))
    o_pos = np.array([obs.x, obs.y]) if hasattr(obs, "x") else np.array(obs.center, dtype=float)
    return float(np.linalg.norm(q - o_pos) - float(obs.radius))


def _compute_smoothness(path_points):
    if len(path_points) <= 2:
        return 0.0
    headings = [
        np.arctan2(path_points[j + 1][1] - path_points[j][1], path_points[j + 1][0] - path_points[j][0])
        for j in range(len(path_points) - 1)
    ]
    return float(np.sum(np.abs(np.diff(np.unwrap(headings)))))


def _compute_vo_metrics(vo_history, round_decimals=3):
    final_active = len(vo_history[-1]) if vo_history else 0
    max_active = max((len(frame) for frame in vo_history), default=0)
    seen = set()
    for frame_v in vo_history:
        for v in frame_v:
            tup = (
                round(float(v["position"][0]), round_decimals),
                round(float(v["position"][1]), round_decimals),
                round(float(v["radius"]), round_decimals),
            )
            seen.add(tup)
    unique_seen = len(seen)
    return final_active, max_active, unique_seen


def _run_one_episode(env, agent, *, max_ticks, speed, vo_round_decimals):
    history = {
        "pos": [],
        "path": [],
        "replan_tick": [],
        "virtual_obstacles": [],
        "perceived_keys": [],
        "path_index": [],
        "debug_status": [],
    }

    start_time = time.time()
    last_sig = None
    path_length = 0.0
    collisions = 0
    exec_steps = 0

    prev_q = np.array(agent.q, dtype=float).copy()

    for t in range(max_ticks):
        agent.step(env, None, DT, speed, t)

        curr_q = np.array(agent.q, dtype=float).copy()
        path_length += float(np.linalg.norm(curr_q - prev_q))
        prev_q = curr_q
        exec_steps = t + 1

        # Simple single-agent obstacle collision check
        for obs in env.obstacles:
            if _distance_to_obstacle_surface(curr_q, obs) < 0.1:
                collisions += 1

        path_copy = [np.array(p, dtype=float).copy() for p in getattr(agent, "planned_path", [])]
        vo_frame = _normalize_virtual_obstacles(getattr(agent, "last_virtual_obstacles", []) or [])

        history["pos"].append(curr_q.copy())
        history["path"].append(path_copy)
        history["virtual_obstacles"].append(vo_frame)
        history["perceived_keys"].append(set(getattr(agent, "_known_obstacle_keys", set())))
        history["path_index"].append(int(getattr(agent, "path_index", 0)))
        history["debug_status"].append(str(getattr(agent, "debug_status", "")))

        sig = _path_signature(path_copy)
        is_replan = sig is not None and last_sig is not None and sig != last_sig
        history["replan_tick"].append(bool(is_replan))
        last_sig = sig

        if agent.done:
            break

    compute_time = time.time() - start_time

    total_rocks = len(env.obstacles)
    sensed_rocks = len(getattr(agent, "_known_obstacle_keys", set()))
    sensing_ratio = (sensed_rocks / total_rocks) if total_rocks > 0 else 1.0

    final_active_virt, max_active_virt, unique_virt_seen = _compute_vo_metrics(
        history["virtual_obstacles"], round_decimals=vo_round_decimals
    )

    plan_calls = int(getattr(getattr(agent, "planner", None), "total_plan_calls", 0))
    plan_failures = int(getattr(getattr(agent, "planner", None), "total_plan_failures", 0))
    planning_effort_total = float(getattr(agent, "planning_effort", 0.0))
    avg_planning_effort_per_call = (planning_effort_total / plan_calls) if plan_calls > 0 else 0.0
    successful_plan_calls = max(plan_calls - plan_failures, 0)
    avg_planning_effort_per_success = (
        planning_effort_total / successful_plan_calls if successful_plan_calls > 0 else 0.0
    )

    path_points = getattr(agent, "executed_path_all", []) or [np.array(agent.q, dtype=float).copy()]
    smoothness = _compute_smoothness(path_points)

    success = bool(getattr(agent, "done", False))
    fail_reason = "None"
    if not success:
        if collisions > 0:
            fail_reason = "Collision"
        elif bool(getattr(agent, "backtrack_failed", False)):
            fail_reason = "Backtrack_Limit"
        elif exec_steps >= max_ticks:
            fail_reason = "Timeout"
        elif plan_failures > 0 or getattr(agent, "plan_fail_count", 0) > 0:
            fail_reason = "Planner_Fail"
        else:
            fail_reason = "Unknown"

    row = {
        "Success": success,
        "FailReason": fail_reason,
        "PathLength": round(path_length, 3),
        "ExecSteps": exec_steps,
        "SensingRatio": round(sensing_ratio, 4),
        "ReplansCounter": int(getattr(agent, "replan_count", 0)),
        "ReplansInferredFromPath": int(sum(history["replan_tick"])),
        "PlanCalls": plan_calls,
        "PlanFailures": plan_failures,
        "PlanningEffortTotal": round(planning_effort_total, 3),
        "AvgPlanningEffortPerCall": round(avg_planning_effort_per_call, 3),
        "AvgPlanningEffortPerSuccess": round(avg_planning_effort_per_success, 3),
        "FinalActiveVirt": int(final_active_virt),
        "MaxActiveVirt": int(max_active_virt),
        "UniqueVirtSeen": int(unique_virt_seen),
        "SmoothnessRad": round(smoothness, 3),
        "BlockedSteps": int(getattr(agent, "blocked_steps", 0)),
        "PlanFailCount": int(getattr(agent, "plan_fail_count", 0)),
        "BacktrackTries": int(getattr(agent, "_agent_backtrack_tries", 0)),
        "BacktrackFail": bool(getattr(agent, "backtrack_failed", False)),
        "DebugStatus": str(getattr(agent, "debug_status", "")),
        "AgentVirtualTotalSeenRaw": int(getattr(agent, "virtual_total_seen", 0)),
        "AgentVirtualMaxActiveRaw": int(getattr(agent, "virtual_max_active", 0)),
        "AgentVirtualTotalCreatedRaw": int(getattr(agent, "virtual_total_created", 0)),
        "ComputeTime": round(compute_time, 4),
        "CollisionCount": collisions,
        "FinalGoalDist": round(float(np.linalg.norm(np.array(agent.q, dtype=float) - np.array(env.goal, dtype=float))), 4),
        "RocksSensed": sensed_rocks,
        "TotalRocks": total_rocks,
    }
    return row


def run_benchmark(args):
    print("--- RAPF v4 SENSING BENCHMARK (manual replay-style loop) ---")
    print(f"speed={args.speed} | DT={DT} | vo_round_decimals={args.vo_round_decimals}")

    os.makedirs("experiments/benchmarks", exist_ok=True)
    agent_params = NAVIGATION_PARAMS.copy()
    all_episode_data = []

    for scn in args.scenarios:
        print(f"\nScenario {scn}:")
        for i in range(args.episodes):
            seed = args.seed_start + i
            cfg = make_config(scn, seed=seed)
            geom = generate_lunar_env(cfg)
            env = make_runtime_env_from_geometry(geom, cfg.L, n_agents=1, seed=seed)

            agent = NavigationCore(agent_id=0, position=env.starts[0], goal=env.goal, **agent_params)

            row = _run_one_episode(
                env,
                agent,
                max_ticks=args.max_ticks,
                speed=args.speed,
                vo_round_decimals=args.vo_round_decimals,
            )
            row["Scenario"] = scn
            row["Seed"] = seed
            all_episode_data.append(row)

            print(
                f"  seed={seed} "
                f"success={row['Success']} "
                f"fail={row['FailReason']} "
                f"exec_steps={row['ExecSteps']} "
                f"plan_calls={row['PlanCalls']} "
                f"plan_failures={row['PlanFailures']} "
                f"plan_effort={row['PlanningEffortTotal']} "
                f"virt(final/max/unique)=({row['FinalActiveVirt']}/{row['MaxActiveVirt']}/{row['UniqueVirtSeen']})"
            )
            if args.print_episode_summary:
                print("    episode data snapshot:")
                print(
                    {
                        "Scenario": row["Scenario"],
                        "Seed": row["Seed"],
                        "Success": row["Success"],
                        "FailReason": row["FailReason"],
                        "PathLength": row["PathLength"],
                        "ExecSteps": row["ExecSteps"],
                        "ReplansCounter": row["ReplansCounter"],
                        "ReplansInferredFromPath": row["ReplansInferredFromPath"],
                        "PlanCalls": row["PlanCalls"],
                        "PlanFailures": row["PlanFailures"],
                        "PlanningEffortTotal": row["PlanningEffortTotal"],
                        "AvgPlanningEffortPerCall": row["AvgPlanningEffortPerCall"],
                        "AvgPlanningEffortPerSuccess": row["AvgPlanningEffortPerSuccess"],
                        "FinalActiveVirt": row["FinalActiveVirt"],
                        "MaxActiveVirt": row["MaxActiveVirt"],
                        "UniqueVirtSeen": row["UniqueVirtSeen"],
                        "BlockedSteps": row["BlockedSteps"],
                        "PlanFailCount": row["PlanFailCount"],
                        "BacktrackTries": row["BacktrackTries"],
                        "BacktrackFail": row["BacktrackFail"],
                        "DebugStatus": row["DebugStatus"],
                        "AgentVirtualTotalSeenRaw": row["AgentVirtualTotalSeenRaw"],
                        "AgentVirtualMaxActiveRaw": row["AgentVirtualMaxActiveRaw"],
                        "AgentVirtualTotalCreatedRaw": row["AgentVirtualTotalCreatedRaw"],
                        "CollisionCount": row["CollisionCount"],
                        "FinalGoalDist": row["FinalGoalDist"],
                        "RocksSensed": row["RocksSensed"],
                        "TotalRocks": row["TotalRocks"],
                    }
                )

    df = pd.DataFrame(all_episode_data)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"experiments/benchmarks/sensing_v4_oldRAPF_{timestamp}.csv"
    df.to_csv(filename, index=False)

    print(f"\nResults saved to {filename}")

    summary = (
        df.groupby("Scenario")
        .agg(
            SuccessRate=("Success", "mean"),
            MeanPathLength=("PathLength", "mean"),
            MeanExecSteps=("ExecSteps", "mean"),
            MeanPlanCalls=("PlanCalls", "mean"),
            MeanPlanFailures=("PlanFailures", "mean"),
            MeanPlanningEffort=("PlanningEffortTotal", "mean"),
            MeanUniqueVirtSeen=("UniqueVirtSeen", "mean"),
            MeanMaxActiveVirt=("MaxActiveVirt", "mean"),
        )
        .round(3)
    )

    print("\n--- AGGREGATED SUMMARY ---")
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10, help="Number of seeds to test per scenario")
    parser.add_argument("--max_ticks", type=int, default=1000)
    parser.add_argument("--scenarios", type=str, nargs="+", default=["A", "B", "C", "D", "E"])
    parser.add_argument("--seed_start", type=int, default=5000)
    parser.add_argument("--speed", type=float, default=1.0, help="Replay-style speed limit passed into agent.step()")
    parser.add_argument("--vo_round_decimals", type=int, default=3, help="Rounding used for unique virtual obstacle keys")
    parser.add_argument(
        "--print_episode_summary",
        action="store_true",
        help="Print a dictionary of collected metrics after every episode",
    )
    args = parser.parse_args()
    run_benchmark(args)
