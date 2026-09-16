# Experiments

This directory preserves the tuning and sensing-benchmark artifacts used during development of the RAPF implementations.

The files copied from `swarm_apf_sim` are intentionally unchanged. This preserves traceability to the original results, including legacy filenames and parameter choices.

## Contents

- `benchmarks/` contains the 2,500-episode sensing benchmark tables for five scenarios and 500 seeds per scenario.
- `optuna/` contains the final retained tuning script and its original SQLite study database.
- The aggregate three-scenario benchmark tables remain at the repository root because the original benchmark script writes them there.

Install the additional experiment dependencies with:

```bash
python -m pip install -r requirements-experiments.txt
```

## RAPF v4 sensing benchmark

Run from the repository root:

```bash
python -m benchmark.benchmark_sensing_RAPF --episodes 10
```

The script imports `RAPF_Agent_v4`. Its output filename includes the legacy label `oldRAPF`; this does not mean that the script imports an older agent version.

Existing result tables are retained as historical artifacts and should not be overwritten when reproducing experiments. Use a separate output location or preserve generated files under a new name.

## Analysis

General result analysis:

```bash
python sensing_benchmark.py experiments/benchmarks/sensing_v4_manual.csv
```

Comparison and blocked-step analysis currently contain the original hard-coded input paths:

```bash
python sensing_benchmark_compare.py
python blocked_steps_analysis.py
```

See `TODO.md` for unresolved naming, provenance, and reproducibility issues.
