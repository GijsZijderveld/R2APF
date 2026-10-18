# Experiments

## Final comparison

[`comparisons/totalResults.csv`](comparisons/totalResults.csv) is the final 15,000-episode dataset. Its three committed summaries are in [`comparisons/summary/`](comparisons/summary/). See the [comparison documentation](comparisons/README.md) for file meanings, campaign coverage, configuration, and commands.

## Historical sensing references

`benchmarks/` retains three development tables because existing tools use them:

| File | Consumer and purpose |
| --- | --- |
| `sensing_v4_withvo.csv` | `scripts/verify_rapf_regression.py`: historical R2APF regression reference |
| `sensing_v4_oldRAPF.csv` | Regression verifier, `sensing_benchmark_compare.py`, and `blocked_steps_analysis.py`: historical RAPF reference |
| `sensing_v4_manual.csv` | Historical comparison and blocked-step analysis |

These tables are not the final controlled comparison. The regression verifier preserves their different original execution routes, including a 1.5 m/s velocity clamp for `withvo` versus a 1.0 m/s manual loop for `oldRAPF`. Do not interpret those historical samples as a common-setting head-to-head comparison.

Run historical analysis from the repository root:

```bash
python sensing_benchmark.py experiments/benchmarks/sensing_v4_manual.csv
python sensing_benchmark_compare.py
python blocked_steps_analysis.py
python scripts/verify_rapf_regression.py --planner both --episodes 3
```

The two comparison analysis scripts use the retained table paths internally. Historical benchmark generators remain in `benchmark/`; their outputs are development artifacts rather than replacements for `comparisons/totalResults.csv`.

## Tuning provenance

[`optuna/`](optuna/README.md) retains `rapf_tune_v3.py` and the original SQLite study database. These describe historical parameter development, not a direct tuning pipeline for the complete final sensing/navigation comparison. Copy the database before running additional tuning.

Install analysis and tuning dependencies with `python -m pip install -r requirements-experiments.txt`.
