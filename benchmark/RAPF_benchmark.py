import argparse
import optuna
import numpy as np
import pandas as pd

# --- Sim Imports ---
from simulation.sim_adapter import (
    make_runtime_env_from_geometry,
    build_agents_for_env,
    simulate_episode,
)
from simulation.env_gen_lunar import make_config, generate_lunar_env
from agents.RAPF_Agent_v3 import RAPF_Agent as AgentClass

# --- COMPARISON DATA (Manteaux et al. 2024) ---
PAPER_RESULTS = {
    "A": {"Reachability": 96.4},
    "B": {"Reachability": 93.8},
    "C": {"Reachability": 91.8}
}

def run_benchmark(args):
    print(f"--- LOADING STUDY ---")
    print(f"DB: {args.db}")
    print(f"Study Name: {args.name}")
    
    try:
        study = optuna.load_study(
            study_name=args.name,
            storage=args.db
        )
        print(f"Loaded Best Trial #{study.best_trial.number}")
        print(f"Best Value (Cost): {study.best_value:.4f}")
    except Exception as e:
        print(f"\n[ERROR] Could not load study. Check your DB path and Study Name.\n{e}")
        return

    # 1. Retrieve Optimized Params
    best_params = study.best_params

    # 2. Inject Fixed Params (These were hardcoded in rapf_tune.py, so Optuna doesn't know them)
    best_params.update({
        'RHO_L': 0.20,
        'MAX_PLANNING_STEPS': 3000, 
        'GOAL_TOLERANCE': 0.8,
        # Ensure these are float/int as expected
        'K_STUCK': int(best_params.get('K_STUCK', 1)),
        'N_B': int(best_params.get('N_B', 10)),
        'MAX_RESTARTS': int(best_params.get('MAX_RESTARTS', 30))
    })

    print("\n--- BENCHMARK CONFIGURATION ---")
    print(f"Episodes per Scenario: {args.episodes}")
    print(f"Seed Start: {args.seed_start}")
    
    results = []
    scenarios = ["A", "B", "C"]

    print(f"\n--- STARTING SIMULATION ---")

    for scn in scenarios:
        print(f"Running Scenario {scn}...", end="", flush=True)
        
        success_count = 0
        path_lengths = []
        collisions_list = []
        efforts = []
        
        for i in range(args.episodes):
            # Unique seed for every run, offset from tuning seeds
            seed = args.seed_start + (scenarios.index(scn) * 1000) + i
            
            # A. Build Env
            cfg = make_config(scn, seed=seed)
            geometry = generate_lunar_env(cfg)
            env = make_runtime_env_from_geometry(geometry, cfg.L, n_agents=1, seed=seed)
            
            # B. Build Agent with BEST_PARAMS
            agents = build_agents_for_env(best_params, env, agent_cls=AgentClass)
            
            # C. Run Sim
            # Note: We use 1000 steps to match the tuning script capability
            metrics = simulate_episode(
                agents, 
                env, 
                seed=seed, 
                max_steps=1000, 
                csv_logger=None
            )
            
            # D. Collect Stats
            if metrics.reached:
                success_count += 1
                path_lengths.append(metrics.path_length)
                
            collisions_list.append(metrics.collisions)
            
            # Collect planning effort (metric from RAPF agent internals)
            # Agents[0] is our RAPF agent
            if hasattr(agents[0], 'planning_effort'):
                efforts.append(agents[0].planning_effort)

            # Progress dot every 10%
            if args.episodes > 10 and (i+1) % (args.episodes // 10) == 0:
                print(".", end="", flush=True)
        
        # Calculate Aggregates
        reachability = (success_count / args.episodes) * 100.0
        avg_path = np.mean(path_lengths) if path_lengths else 0.0
        avg_collisions = np.mean(collisions_list)
        avg_effort = np.mean(efforts) if efforts else 0.0
        
        results.append({
            "Scenario": scn,
            "Reachability (%)": reachability,
            "Paper Ref (%)": PAPER_RESULTS[scn]["Reachability"],
            "Diff (%)": reachability - PAPER_RESULTS[scn]["Reachability"],
            "Avg Path (m)": round(avg_path, 2),
            "Avg Collisions": round(avg_collisions, 2),
            "Avg Planning Steps": round(avg_effort, 1)
        })
        print(" Done!")

    # =============================================================================
    # RESULTS TABLE
    # =============================================================================
    print("\n\n=== FINAL BENCHMARK RESULTS ===")
    df = pd.DataFrame(results)

    # Save
    csv_name = f"benchmarkv3_rapf_500_{args.name.replace(' ', '_')}.csv"
    df.to_csv(csv_name, index=False)
    
    # Display
    cols = ["Scenario", "Reachability (%)", "Paper Ref (%)", "Diff (%)", "Avg Path (m)", "Avg Planning Steps"]
    print(df[cols].to_string(index=False))
    
    avg_diff = df["Diff (%)"].mean()
    print(f"\nOverall improvement over paper: {avg_diff:+.2f}%")
    print(f"Detailed results saved to: {csv_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark RAPF Agent using Tuned Params.")
    
    # Defaults matched to rapf_tune.py provided in prompt
    parser.add_argument("--db", type=str, default="sqlite:///experiments/optuna/data/rapf_tuning_official.db", help="Path to Optuna DB")
    parser.add_argument("--name", type=str, default="Offical RAPF parameter test", help="Optuna Study Name")
    
    parser.add_argument("--episodes", type=int, default=100, help="Episodes per scenario")
    parser.add_argument("--seed_start", type=int, default=2000, help="Starting seed for tests (avoid tuning seeds)")
    
    args = parser.parse_args()
    
    run_benchmark(args)