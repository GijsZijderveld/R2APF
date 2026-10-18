# R2APF

Research code and final benchmark results for R2APF, a path-planning method for autonomous navigation in partially observed obstacle environments. The repository compares R2APF with RAPF, classical APF, A*, D* Lite, and RRT* through a shared sensing, navigation, recovery, and measurement layer.

## Final results

The authoritative dataset is [`experiments/comparisons/totalResults.csv`](experiments/comparisons/totalResults.csv), finalized in commit [`ca128cf`](https://github.com/GijsZijderveld/R2APF/commit/ca128cffa98e9f7b2323c542e5a60e2369ce3bfa) on 2 October 2026. It contains **15,000 episodes**: six planners × five scenarios (`A`–`E`) × 500 seeds (`5000`–`5499`). The RRT* rows use the final setting of 600 additional sampling attempts after the first solution, subject to a 2,000-attempt cap per planning call.

| Planner | Successful episodes | Success rate |
| --- | ---: | ---: |
| APF | 191 / 2,500 | 7.64% |
| A* | 2,500 / 2,500 | 100% |
| D* Lite | 2,500 / 2,500 | 100% |
| R2APF | 2,496 / 2,500 | 99.84% |
| RAPF | 2,449 / 2,500 | 97.96% |
| RRT* | 2,500 / 2,500 | 100% |

The final summaries are in [`experiments/comparisons/summary/`](experiments/comparisons/summary/):

- [`summary_total_results.csv`](experiments/comparisons/summary/summary_total_results.csv): overall metrics for each planner.
- [`summary_total_results_by_scenario.csv`](experiments/comparisons/summary/summary_total_results_by_scenario.csv): metrics for each planner and scenario.
- [`failure_reasons_total_results.csv`](experiments/comparisons/summary/failure_reasons_total_results.csv): failure counts and episode rates by planner and scenario.

All final episodes report zero executed collisions. Path-length summaries distinguish all episodes from successful episodes; planning time is measured in seconds. These are results for the recorded simulation campaign. The CSV does not embed per-row configurations, software versions, or the generating commit, so the repository settings alone do not establish a complete execution record. See the [result documentation](experiments/comparisons/README.md) for scope and reproduction instructions.

## Repository structure

```text
R2APF/
├── agents/                 Shared navigation implementation; legacy agent wrapper
├── benchmark/              Historical RAPF benchmark entry points
├── comparison/             Common planner interface, agent, runner, sensing and metrics
├── configs/
│   ├── benchmark.json      Campaign scenarios, seeds and execution settings
│   ├── comparison_policy.yaml  Experimental rationale and policy notes
│   └── planners/           One JSON configuration per planner
├── demos/                  Interactive RAPF-family demonstration
├── experiments/
│   ├── comparisons/        Final 15,000-row dataset and its three summaries
│   ├── benchmarks/         Historical sensing tables used by legacy checks/analysis
│   └── optuna/             Historical tuning script and retained SQLite database
├── figures/                Scenario and recovery illustrations, with JSON metadata
├── planners/
│   ├── rapf_global_planner.py   R2APF implementation
│   ├── rapf_paper_planner.py    RAPF implementation
│   └── baselines/          APF, A*, D* Lite, RRT* and grid utilities
├── scripts/                Comparison, summaries, visualization and figure generation
├── simulation/             Environment generation, geometry and runtime adapters
├── tests/                  Planner, comparison-contract and summary tests
├── sensing_benchmark.py    Historical sensing-table analysis
├── sensing_benchmark_compare.py  Historical RAPF comparison analysis
├── blocked_steps_analysis.py     Historical blocked-step analysis
├── requirements.txt        Core simulation and visualization dependencies
├── requirements-experiments.txt  Additional analysis and tuning dependencies
└── TODO.md                 Development backlog; not a result-selection guide
```

`comparison/agents/navigation_agent.py` exposes the shared `NavigationAgent`, built on `agents/navigation_core.py`. Planner selection changes planning logic while retaining the common navigation layer. Implementation details are in [`comparison/README.md`](comparison/README.md), and planner settings are described in [`configs/planners/README.md`](configs/planners/README.md).

The retained historical sensing tables and Optuna study provide regression and development provenance. They are separate from the final comparison. Superseded comparison campaigns, smoke-test CSVs, redundant rerun exports, and unused historical result exports have been removed; earlier versions remain in Git history.

## Installation

Use Python 3.10 or newer and run commands from the repository root:

```bash
git clone https://github.com/GijsZijderveld/R2APF.git
cd R2APF
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-experiments.txt
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. For visualization only, `requirements.txt` is sufficient. The requirements describe the available tooling; they are not a frozen environment capture from the original campaign.

## Inspect and summarize the final results

```bash
python scripts/summarize_comparison.py \
  experiments/comparisons/totalResults.csv \
  --output-dir outputs/final-summary
```

This regenerates all three summary tables in a separate directory for comparison with the committed summaries. No simulation run is needed.

## Run a new comparison

```bash
python scripts/run_comparison.py --planner all \
  --output outputs/comparison/results.csv
python scripts/summarize_comparison.py outputs/comparison/results.csv \
  --output-dir outputs/comparison/summary
```

The full configuration runs 15,000 episodes. Add `--episodes 1` for a 30-episode smoke run (one seed per scenario and planner). Existing rows at the output path are skipped by planner/scenario/seed; resume only with the same settings. Use a fresh output path when settings change. Timing measurements depend on the machine and runtime environment.

## Visualize a planner

```bash
python scripts/visualize_planner.py --planner r2apf --scenario A --seed 5000
```

Planner names are `rapf`, `r2apf`, `apf`, `astar`, `dstar_lite`, and `rrt_star`. Add `--interval-ms 100` to slow the animation, or `--save outputs/r2apf.gif` to save it. Unknown obstacles appear faintly until sensed; orange shows the remaining plan, blue the executed route, and purple crosses the planning locations.

The original demonstration is also available:

```bash
python -m demos.replay_rapf --scenario B --seed 42 --steps 1000 --speed 1
```

## Figures and historical checks

`figures/scenario_difficulty.*` illustrates environment difficulty; `figures/rapf_repair_concept.*` illustrates recovery. PDF, SVG, PNG, and JSON files are retained for each figure. The recovery routes are illustrative, collision-checked A* routes, not measured RAPF/R2APF trajectories. Regenerate figures with `scripts/plot_scenario_difficulty.py` and `scripts/plot_rapf_repair_concept.py`; each script provides `--help`.

```bash
python scripts/verify_rapf_regression.py --planner both --episodes 3
```

This checks historical samples against their original execution routes. Those tables use different execution settings and are not a controlled planner comparison. See [`experiments/README.md`](experiments/README.md) for the retained files and historical analysis commands.

To run the automated checks, install `pytest` and run `python -m pytest`.

## Academic context and reuse

R2APF was developed during Gijs Zijderveld's MSc thesis in Robotics at the Faculty of Mechanical Engineering. This repository was extracted from [`swarm_apf_sim`](https://github.com/GijsZijderveld/swarm_apf_sim).

The repository currently has no open-source license, formal citation file, or archived release DOI. Copyright remains with the author; public visibility does not grant an open-source reuse license.
