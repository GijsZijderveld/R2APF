# Final planner comparison

## Authoritative files

| File | Contents |
| --- | --- |
| `totalResults.csv` | Final raw results: 15,000 unique planner/scenario/seed cases |
| `summary/summary_total_results.csv` | Six overall planner summaries |
| `summary/summary_total_results_by_scenario.csv` | Thirty planner/scenario summaries |
| `summary/failure_reasons_total_results.csv` | Failure counts and rates by planner/scenario |

The final dataset was updated on 2 October 2026 in commit `ca128cffa98e9f7b2323c542e5a60e2369ce3bfa`. Each of APF, A*, D* Lite, R2APF, RAPF, and RRT* has 500 episodes per scenario A–E, with seeds 5000–5499. The final RRT* rows contain the measurements from the 600-extra-sample rerun, replacing the earlier first-solution results. The separate rerun export is redundant and has been removed alongside older campaigns and smoke outputs. Git history retains those exports.

## Configuration and interpretation

- `configs/benchmark.json` contains the current campaign and shared execution settings.
- `configs/planners/*.json` contains the current planner settings. RRT* uses `optimization_samples: 600` and `max_samples: 2000`.
- `configs/comparison_policy.yaml` records design rationale and historical/provisional policy notes. The runner reads the JSON settings; it does not load this YAML as executable configuration.
- The CSV stores episode outcomes, execution metrics, planner timing, failures, and supported mechanism diagnostics. Blank planner-specific diagnostics mean unavailable, not zero.
- Executed path length is measured in metres and timing in seconds. Overall tables include both all-episode and successful-episode path-length statistics. Failed episodes can have short partial paths.
- The final dataset combines retained runs and replacement reruns. It does not embed per-row settings, dependency versions, machine details, or generating commits. Current configuration files are available, but complete run provenance cannot be recovered from the CSV alone.

## Regenerate summaries

From the repository root, with `requirements-experiments.txt` installed:

```bash
python scripts/summarize_comparison.py experiments/comparisons/totalResults.csv \
  --output-dir outputs/final-summary
```

The script writes the three `*_total_results*` summary files listed above. Compare the generated files with the committed summaries before replacing them.

## Run a separate campaign

```bash
python scripts/run_comparison.py --planner all \
  --output outputs/comparison/results.csv
python scripts/summarize_comparison.py outputs/comparison/results.csv \
  --output-dir outputs/comparison/summary
```

Add `--episodes 1` for one seed per scenario (30 episodes across all planners). The full configuration has 15,000 episodes. The runner checkpoints completed cases and resumes by skipping existing planner/scenario/seed keys. Resume only with unchanged settings; use a new path for a different configuration. Keep the committed final dataset unchanged when experimenting.
