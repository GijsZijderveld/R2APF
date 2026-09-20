# R2APF

Research code for the R2APF path-planning method developed for autonomous navigation in partially observed obstacle environments.

This repository is a focused extraction from the original thesis development repository, [`swarm_apf_sim`](https://github.com/GijsZijderveld/swarm_apf_sim). The current baseline preserves the selected implementation and its parameters without changing the algorithmic behavior. Known issues and proposed improvements are recorded in [`TODO.md`](TODO.md).

## What is included

- `comparison/agents/navigation_agent.py` — shared sensing, path tracking, recovery, and execution agent used by every planner.
- `agents/navigation_core.py` — regression-verified implementation of that shared navigation behavior.
- `planners/rapf_global_planner.py` — the R2APF planner.
- `planners/rapf_paper_planner.py` — the original RAPF planner retained from the paper implementation.
- `simulation/` — the environment, obstacle, and adapter modules required by the demonstration.
- `demos/replay_rapf.py` — a single-agent simulation and visualization.
- `benchmark/benchmark_sensing_RAPF.py` — the historical sensing and replanning benchmark.
- `experiments/benchmarks/` — preserved 2,500-episode result tables.
- `experiments/optuna/` — the retained RAPF v3 tuning script and original Optuna database, included as parameter-development provenance.

Development caches, unrelated swarm experiments, obsolete tuning databases, patches, and bulk generated media from the original repository are intentionally excluded.

## Installation

Python 3.10 or newer is required.

```bash
git clone https://github.com/GijsZijderveld/R2APF.git
cd R2APF

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Run the demonstration

Run commands from the repository root so the existing imports resolve correctly:

```bash
python -m demos.replay_rapf
```

Optional arguments:

```bash
python -m demos.replay_rapf --scenario B --seed 42 --steps 1000 --speed 1
```

The demonstration generates a lunar-style obstacle environment, simulates one agent with incrementally perceived obstacles, and displays the executed and currently planned paths.

## Visualize any comparison planner

The common visualizer animates sensing, planning, movement, path invalidation,
and replanning for R2APF and all baseline planners:

```bash
python scripts/visualize_planner.py --planner astar --scenario A --seed 5000
```

Valid planner names are `rapf`, `r2apf`, `apf`, `astar`, `dstar_lite`, and `rrt_star`.

All planners use the same unversioned `NavigationAgent`. `rapf` and `r2apf`
select different planner adapters; sensing, execution, recovery, and metric
collection remain shared. Replay a historical regression sample with:

```bash
python scripts/verify_rapf_regression.py --planner both --episodes 3
```

The verifier intentionally uses each artifact's original execution route.
`sensing_v4_withvo.csv` used the legacy `simulate_episode` runner with a
1.5 m/s velocity clamp, while `sensing_v4_oldRAPF.csv` used the later manual
loop at 1.0 m/s. These files are reproducibility references, not a controlled
head-to-head comparison. New comparisons must run both planners through
`scripts/run_comparison.py` with the same benchmark configuration.
Unknown obstacles are drawn faintly until sensed, the orange line is the
remaining plan, the blue line is the executed trajectory, and purple crosses
mark planning and replanning locations.

To slow the animation down or save it as a GIF:

```bash
python scripts/visualize_planner.py \
  --planner dstar_lite \
  --scenario B \
  --seed 5000 \
  --interval-ms 100 \
  --save experiments/comparisons/dstar_lite_seed5000.gif
```

Planner parameters default to the matching file in `configs/planners/`. Use
`--planner-config` to select a different configuration explicitly.

## Experiments

Install the additional dependencies with:

```bash
python -m pip install -r requirements-experiments.txt
```

The preserved benchmark, tuning database, result tables, and analysis commands are documented in [`experiments/README.md`](experiments/README.md). The Optuna artifacts tune the planner parameters and are retained as provenance.

A modular planner-comparison framework is documented in [`comparison/README.md`](comparison/README.md). Every method uses the same `NavigationAgent` sensing, execution, recovery, collision, and metric layer.

## Repository status

The historical R2APF implementation remains behavior-preserving. Classical
APF, A*, D* Lite, and RRT* have been added behind a separate common comparison
interface. They still require sensitivity studies and frozen computation
budgets before publication experiments. See [`TODO.md`](TODO.md) for known
inconsistencies and reproducibility work that remains.

## Academic context

The method was developed as part of Gijs Zijderveld's MSc thesis in Robotics at the Faculty of Mechanical Engineering.

A formal paper citation and archived release identifier will be added when the revised publication is available.

## License

No open-source license has been selected yet. Until a license is added, copyright remains with the author and reuse is not automatically permitted.
