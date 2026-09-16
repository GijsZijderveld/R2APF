# R2APF

Research code for the R2APF path-planning method developed for autonomous navigation in partially observed obstacle environments.

This repository is a focused extraction from the original thesis development repository, [`swarm_apf_sim`](https://github.com/GijsZijderveld/swarm_apf_sim). The current baseline preserves the selected implementation and its parameters without changing the algorithmic behavior. Known issues and proposed improvements are recorded in [`TODO.md`](TODO.md).

## What is included

- `agents/RAPF_Agent_v4.py` — sensing, path tracking, and replanning agent.
- `planners/rapf_global_planner.py` — global R2APF/RAPF planner used by the agent.
- `planners/rapf_paper_planner.py` — paper-oriented planner implementation retained from the source repository.
- `simulation/` — the environment, obstacle, and adapter modules required by the demonstration.
- `demos/replay_RAPF_v4.py` — a single-agent simulation and visualization.
- `benchmark/benchmark_sensing_RAPF.py` — the sensing and replanning benchmark for `RAPF_Agent_v4`.
- `experiments/benchmarks/` — preserved 2,500-episode result tables.
- `experiments/optuna/` — the retained RAPF v3 tuning script and original Optuna database, included as parameter-development provenance.

Development caches, unrelated swarm experiments, obsolete tuning databases, patches, and bulk generated media from the original repository are intentionally excluded.

## Installation

Python 3.8 or newer is recommended.

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
python -m demos.replay_RAPF_v4
```

Optional arguments:

```bash
python -m demos.replay_RAPF_v4 --scenario B --seed 42 --steps 1000 --speed 1
```

The demonstration generates a lunar-style obstacle environment, simulates one agent with incrementally perceived obstacles, and displays the executed and currently planned paths.

## Experiments

Install the additional dependencies with:

```bash
python -m pip install -r requirements-experiments.txt
```

The preserved benchmark, tuning database, result tables, and analysis commands are documented in [`experiments/README.md`](experiments/README.md). The Optuna artifacts tune RAPF v3 and are retained as provenance; they do not directly tune the v4 agent.

## Repository status

This is the initial, behavior-preserving extraction. It is intended to establish a clear reference point before adding benchmark planners and publication experiments. The code has deliberately not been refactored or corrected during extraction. See [`TODO.md`](TODO.md) for known inconsistencies and reproducibility work that remains.

## Academic context

The method was developed as part of Gijs Zijderveld's MSc thesis in Robotics at the Faculty of Mechanical Engineering.

A formal paper citation and archived release identifier will be added when the revised publication is available.

## License

No open-source license has been selected yet. Until a license is added, copyright remains with the author and reuse is not automatically permitted.
