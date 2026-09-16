import optuna
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- IMPORTS ---
from simulation.env_suite import sample_batch
from simulation.sim_adapter import (
    make_runtime_env_from_geometry,
    build_agents_for_env,
    simulate_episode
)
from simulation.env_gen_lunar import generate_lunar_env, make_config
# Make sure this imports the file you just updated above
from agents.RAPF_Agent_v3 import RAPF_Agent as AgentClass

# --- Configuration ---
TUNING_N_AGENTS = 1

def objective(trial, n_agents):
    # --- 1. Define Search Space ---
    params = {
        # Potential Field Params
        'ALPHA_A': trial.suggest_float('ALPHA_A', 2.0, 20.0), 
        'MU_A': trial.suggest_float('MU_A', 0.0002, 0.0008),       
        'ALPHA_O': trial.suggest_float('ALPHA_O', 2.0, 20), 
        'MU_O': trial.suggest_float('MU_O', 50.0, 100.0),
        'RHO_L': 0.20,
        'RHO_U': trial.suggest_float('RHO_U', 1.0, 2.5),

        # Bacteria Params
        'N_B': trial.suggest_int('N_B', 4, 16), 
        'RHO_B': trial.suggest_float('RHO_B', 0.25, 0.6),
        
        # Planner Params
        # K_STUCK REMOVED - Logic is now immediate
        
        # ARTIFICIAL_R: 
        # Defines the hard radius of the virtual obstacle.
        # Combined with RHO_U, this determines how far we prune back.
        'ARTIFICIAL_R': trial.suggest_float('ARTIFICIAL_R', 0.2, 0.8),
        
        'MAX_PLANNING_STEPS': 5000, 
        'GOAL_TOLERANCE': 0.8,
        'MAX_RESTARTS': 50, # Generous budget for local repairs
    }

    # --- 2. Batch Sampling (Robust Tuning) ---
    batch = sample_batch("train", per_scenario=20)
    scores = []
    
    for i, cfg in enumerate(batch):
        # Setup Env
        seed = getattr(cfg, "rng_seed", trial.number)
        geometry = generate_lunar_env(cfg)
        env = make_runtime_env_from_geometry(
            obstacles=geometry, L=cfg.L, n_agents=n_agents, seed=seed
        )

        # Build Agent
        agents = build_agents_for_env(params, env, agent_cls=AgentClass)
        agent = agents[0]
        
        # Run Sim
        metrics = simulate_episode(agents, env, seed=seed, max_steps=1000, csv_logger=None)
        
        # 1. Gather Metrics
        effort = getattr(agent, 'planning_effort', 0)
        restarts = getattr(agent, 'restart_count', 0)
        
        # 2. Calculate Penalty
        if not getattr(agent, 'is_planning_complete', False) or len(agent.planned_path) == 0:
            # CRITICAL FAILURE
            scores.append(20000.0)
            
        elif not metrics.reached:
            # EXECUTION FAILURE
            dist = np.linalg.norm(agent.q - env.goal)
            scores.append(5000.0 + (dist * 100.0))
            
        else:
            # SUCCESS
            base_score = metrics.path_length 
            
            # Thinking Penalty (Internal iterations)
            thinking_penalty = (effort * params['N_B']) * 0.001 
            
            # Repair Penalty (Low cost, as repair is a valid strategy now)
            complexity_penalty = (restarts * 1.0) 
            
            # Total
            scenario_score = base_score + thinking_penalty + complexity_penalty
            
            if metrics.collisions > 0:
                scenario_score += 2000.0 * metrics.collisions
            
            scores.append(scenario_score)

    return sum(scores) / len(scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune RAPF Agent (Generalist Batch).")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--name", type=str, default="rapf_local_repair_v5") 
    parser.add_argument("--db", type=str, default="sqlite:///experiments/optuna/data/rapf_tuning_official.db")
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    print(f"--- Tuning RAPF (Local Repair) on Batch (20 maps/scenario) ---")
    
    # --- WARM START PARAMETERS ---
    # K_STUCK removed from this list
    OLD_BEST_PARAMS = {
        'ALPHA_A': 9.131907971874075,
        'MU_A': 0.00032328720173746485,
        'ALPHA_O': 18.705193949868313,
        'MU_O': 98.75500366685941,
        'RHO_U': 1.723968407669323,
        'N_B': 11,
        'RHO_B': 0.4732975890061234,
        'ARTIFICIAL_R': 0.35758377024584537,
        # K_STUCK removed
    }

    study = optuna.create_study(
        study_name=args.name,
        storage=args.db,
        load_if_exists=True,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )

    print("-> Enqueuing previous best parameters (Warm Start)...")
    study.enqueue_trial(OLD_BEST_PARAMS)

    study.optimize(lambda t: objective(t, TUNING_N_AGENTS), n_trials=args.trials)

    print("\n--- Best Parameters Found ---")
    for k, v in study.best_params.items():
        print(f"{k}: {v}")